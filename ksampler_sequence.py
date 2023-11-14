import hashlib
import math
import random
import re
import torch
import torch.nn.functional as F

import comfy.sample
import comfy.samplers
import comfy.model_management
import nodes

def slerp(strength, tensor_from, tensor_to, epsilon=1e-6):
    """
    Perform Spherical Linear Interpolation (Slerp) between two tensors.
    
    Parameters:
    - strength (float): The interpolation factor between tensor_from and tensor_to. 
    - tensor_from (Tensor): The starting tensor.
    - tensor_to (Tensor): The ending tensor.
    - epsilon (float): division by zero offset
    
    Returns:
    - Tensor: Interpolated tensor.
    """
    low_norm = F.normalize(tensor_from, p=2, dim=-1, eps=epsilon)
    high_norm = F.normalize(tensor_to, p=2, dim=-1, eps=epsilon)
    
    dot_product = torch.clamp((low_norm * high_norm).sum(dim=-1), -1.0, 1.0)
    omega = torch.acos(dot_product)
    so = torch.sin(omega)
    zero_so_mask = torch.isclose(so, torch.tensor([0.0], device=so.device), atol=epsilon)
    so = torch.where(zero_so_mask, torch.tensor([1.0], device=so.device), so)
    sin_omega_minus_strength = torch.sin((1.0 - strength) * omega) / so
    sin_strength_omega = torch.sin(strength * omega) / so
    
    res = sin_omega_minus_strength.unsqueeze(-1) * tensor_from + sin_strength_omega.unsqueeze(-1) * tensor_to
    res = torch.where(zero_so_mask.unsqueeze(-1), 
                      tensor_from if strength < 0.5 else tensor_to, 
                      res)
    
    return res

# from  https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475
def slerp_latents(val, low, high):
    dims = low.shape

    #flatten to batches
    low = low.reshape(dims[0], -1)
    high = high.reshape(dims[0], -1)

    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)

    # in case we divide by zero
    low_norm[low_norm != low_norm] = 0.0
    high_norm[high_norm != high_norm] = 0.0

    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res.reshape(dims)

def blend_latents(alpha, latent_1, latent_2):
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor([alpha], dtype=latent_1.dtype, device=latent_1.device)
    
    blended_latent = (1 - alpha) * latent_1 + alpha * latent_2
    
    return blended_latent

def cosine_interp_latents(val, low, high):
    if not isinstance(val, torch.Tensor):
        val = torch.tensor([val], dtype=low.dtype, device=low.device)        
    t = (1 - torch.cos(val * math.pi)) / 2
    return (1 - t) * low + t * high

def unsample(model, seed, cfg, sampler_name, steps, end_at_step, scheduler, normalize, positive, negative, latent_image):
    device = comfy.model_management.get_torch_device()
    end_at_step = steps - min(end_at_step, steps - 1)

    latent = latent_image
    latent_image = latent["samples"].to(device)
    
    noise_shape = latent_image.size()
    noise = torch.zeros(noise_shape, dtype=latent_image.dtype, layout=latent_image.layout, device=device)
    noise_mask = comfy.sample.prepare_mask(latent.get("noise_mask"), noise, device) if "noise_mask" in latent else None

    positive_copy = comfy.sample.broadcast_cond(positive, noise.shape[0], device)
    negative_copy = comfy.sample.broadcast_cond(negative, noise.shape[0], device)

    models, inference_memory = comfy.sample.get_additional_models(positive, negative, model.model_dtype())
    comfy.model_management.load_models_gpu([model] + models, comfy.model_management.batch_area_memory(noise.numel() // noise.shape[0]) + inference_memory)

    real_model = model.model
    sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)
    sigmas = sampler.sigmas.flip(0) + 0.0001

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps): pbar.update_absolute(step + 1, total_steps)

    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image, force_full_denoise=False, denoise_mask=noise_mask, sigmas=sigmas, start_step=0, last_step=end_at_step, callback=callback, seed=seed)
    
    if normalize == "enable":
        samples = (samples - samples.mean()) / samples.std()
    
    comfy.sample.cleanup_additional_models(models)
    
    out = latent.copy()
    out["samples"] = samples.cpu()
    return (out,)

CLIPTextEncode = nodes.CLIPTextEncode()
USE_BLK, BLK_ADV = (False, None)
if "BNK_CLIPTextEncodeAdvanced" in nodes.NODE_CLASS_MAPPINGS:
    BLK_ADV = nodes.NODE_CLASS_MAPPINGS['BNK_CLIPTextEncodeAdvanced']
    USE_BLK = True

if USE_BLK:
    print(f"Found `\33[1mComfyUI_ADV_CLIP_emb\33[0m`. Using \33[93mBLK Advanced CLIPTextEncode\33[0m for Conditioning Sequencing")
    blk_adv = BLK_ADV()

class CLIPTextEncodeSequence:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++"],),
                "text": ("STRING", {"multiline": True, "default": '''0:A portrait of a rosebud
5:A portrait of a blooming rosebud
10:A portrait of a blooming rose
15:A portrait of a rose'''}),
                }
            }
        
    RETURN_TYPES = ("CONDITIONING_SEQ",)
    RETURN_NAMES = ("conditioning_sequence",)
    IS_LIST_OUTPUT = (True,)

    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip, text, token_normalization, weight_interpretation):
        text = text.strip()
        conditionings = []
        for l in text.splitlines():
            match = re.match(r'(\d+):', l)
            
            if match:
                idx = int(match.group(1))
                _, line = l.split(":", 1)
                line = line.strip()
                
                if USE_BLK:
                    encoded = blk_adv.encode(clip=clip, text=line, token_normalization=token_normalization, weight_interpretation=weight_interpretation)
                else:
                    encoded = CLIPTextEncode.encode(clip=clip, text=line)
                
                conditioning = (idx, [encoded[0][0][0], encoded[0][0][1]])
                conditionings.append(conditioning)

        return (conditionings, )
    
class KSamplerSeq:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "seed_mode_seq": (["increment", "decrement", "random", "fixed"],),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "sequence_loop_count": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                    "positive_seq": ("CONDITIONING_SEQ", ),
                    "negative_seq": ("CONDITIONING_SEQ", ),
                    "use_conditioning_slerp": ("BOOLEAN", {"default": False}),
                    "cond_slerp_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "latent_image": ("LATENT", ),
                    "use_latent_interpolation": ("BOOLEAN", {"default": False}),
                    "latent_interpolation_mode": (["Blend", "Slerp", "Cosine Interp"],),
                    "latent_interp_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "denoise_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "denoise_seq": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "unsample_latents": ("BOOLEAN", {"default": False})
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def update_seed(self, seed, seed_mode):
        if seed_mode == "increment":
            return seed + 1
        elif seed_mode == "decrement":
            return seed - 1
        elif seed_mode == "random":
            return random.randint(0, 0xffffffffffffffff)
        elif seed_mode == "fixed":
            return seed

    def hash_tensor(self, tensor):
        tensor = tensor.cpu().contiguous()
        return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

    def update_conditioning(self, conditioning_seq, loop_count, last_conditioning):
        matching_conditioning = None
        for idx, conditioning, *_ in conditioning_seq:
            if int(idx) == loop_count:
                matching_conditioning = conditioning
                break
        return matching_conditioning if matching_conditioning else (last_conditioning if last_conditioning else None)

    def sample(self, model, seed, seed_mode_seq, steps, cfg, sampler_name, scheduler, sequence_loop_count, positive_seq, negative_seq, cond_slerp_strength, latent_image, use_latent_interpolation, latent_interpolation_mode, latent_interp_strength, denoise_start=1.0, denoise_seq=0.5, use_conditioning_slerp=False, unsample_latents=False):
        positive_seq = positive_seq
        negative_seq = negative_seq

        results = []
        positive_conditioning = None
        negative_conditioning = None
        for loop_count in range(sequence_loop_count):

            seq_seed = seed if loop_count <= 0 else self.update_seed(seq_seed, seed_mode_seq)

            print(f"Loop count: {loop_count}, Seed: {seq_seed}")

            last_positive_conditioning = positive_conditioning[0] if positive_conditioning else None
            last_negative_conditioning = negative_conditioning[0] if negative_conditioning else None

            positive_conditioning = self.update_conditioning(positive_seq, loop_count, last_positive_conditioning)
            negative_conditioning = self.update_conditioning(negative_seq, loop_count, last_negative_conditioning)

            if use_conditioning_slerp and (last_positive_conditioning and last_negative_conditioning):
                a = last_positive_conditioning[0].clone()
                b = positive_conditioning[0].clone()
                na = last_negative_conditioning[0].clone()
                nb = negative_conditioning[0].clone()

                pa = last_positive_conditioning[1]["pooled_output"].clone()
                pb = positive_conditioning[1]["pooled_output"].clone()
                npa = last_negative_conditioning[1]["pooled_output"].clone()
                npb = negative_conditioning[1]["pooled_output"].clone()

                pos_cond = slerp(cond_slerp_strength, a, b)
                pos_pooled = slerp(cond_slerp_strength, pa, pb)
                neg_cond = slerp(cond_slerp_strength, na, nb)
                neg_pooled = slerp(cond_slerp_strength, npa, npb)
                
                positive_conditioning = [pos_cond, {"pooled_output": pos_pooled}]
                negative_conditioning = [neg_cond, {"pooled_output": neg_pooled}]

            positive_conditioning = [positive_conditioning]
            negative_conditioning = [negative_conditioning]

            if positive_conditioning is not None or negative_conditioning is not None:

                end_at_step = steps
                if results is not None and len(results) > 0:
                    latent_input = {'samples': results[-1]}
                    denoise = denoise_seq
                    start_at_step = round((1 - denoise) * steps)
                    end_at_step = steps
                else:
                    latent_input = latent_image
                    denoise = denoise_start

                if unsample_latents and loop_count > 0:
                    force_full_denoise = False if loop_count > 0 or loop_count <= steps - 1 else True
                    disable_noise = False
                    unsampled_latent = unsample(model=model, seed=seq_seed, cfg=cfg, sampler_name=sampler_name, steps=steps, end_at_step=end_at_step, scheduler=scheduler, normalize=False, positive=positive_conditioning, negative=negative_conditioning, latent_image=latent_input)[0]
                    sample = nodes.common_ksampler(model, seq_seed, steps, cfg, sampler_name, scheduler, positive_conditioning, negative_conditioning, unsampled_latent, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)[0]['samples']
                else:
                    sample = nodes.common_ksampler(model, seq_seed, steps, cfg, sampler_name, scheduler, positive_conditioning, negative_conditioning, latent_input, denoise=denoise)[0]['samples']

                if use_latent_interpolation and results and loop_count > 0:
                    if latent_interpolation_mode == "Blend":
                        sample = blend_latents(latent_interp_strength, results[-1], sample)
                    elif latent_interpolation_mode == "Slerp":
                        sample = slerp_latents(latent_interp_strength, results[-1], sample)
                    elif latent_interpolation_mode == "Cosine Interp":
                        sample = cosine_interp_latents(latent_interp_strength, results[-1], sample)
                    else:
                        sample = sample

                results.append(sample)

        results = torch.cat(results, dim=0)
        results = {'samples': results}

        return (results,)




NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeList": CLIPTextEncodeSequence,
    "KSamplerSeq": KSamplerSeq,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeList": "CLIP Text Encode Sequence (Advanced)",
    "KSamplerSeq": "KSampler Sequence",

}
