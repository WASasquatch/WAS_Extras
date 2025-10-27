import hashlib
import math
import random
import re
import torch
import torch.nn.functional as F
import numpy as np

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

    positive_copy = comfy.sample.convert_cond(positive)
    negative_copy = comfy.sample.convert_cond(negative)

    models, inference_memory = comfy.sample.get_additional_models(positive, negative, model.model_dtype())
    comfy.model_management.load_models_gpu([model] + models, model.memory_required(noise.shape) + inference_memory)

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
                "clip": ("CLIP", {"tooltip": "CLIP model used to encode text prompts."}),
                "token_normalization": (["none", "mean", "length", "length+mean"], {"tooltip": "Normalization strategy for token weights."}),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++"], {"tooltip": "How to interpret weights and syntax in the text."}),
                "text": ("STRING", {"multiline": True, "default": '''0:A portrait of a rosebud
5:A portrait of a blooming rosebud
10:A portrait of a blooming rose
15:A portrait of a rose''', "tooltip": "One entry per line in the form 'frameIndex:prompt'."})
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
    
class CLIPTextEncodeSequence2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP model to encode prompts."}),
                "token_normalization": (["none", "mean", "length", "length+mean"], {"tooltip": "Normalization strategy for token weights."}),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++"], {"tooltip": "How to interpret weights and syntax in the text."}),
                "cond_keyframes_type": (["linear", "sinus", "sinus_inverted", "half_sinus", "half_sinus_inverted"], {"tooltip": "Schedule describing when to switch to the next prompt."}),
                "frame_count": ("INT", {"default": 100, "min": 1, "max": 1024, "step": 1, "tooltip": "Total frames for which to generate conditioning."}),
                "text": ("STRING", {"multiline": True, "default": '''A portrait of a rosebud
A portrait of a blooming rosebud
A portrait of a blooming rose
A portrait of a rose''', "tooltip": "List of prompts; keyframe schedule determines when to advance."})
            }
        }
        
    RETURN_TYPES = ("CONDITIONING", "INT", "INT")
    RETURN_NAMES = ("conditioning_sequence", "cond_keyframes", "frame_count")
    IS_LIST_OUTPUT = (True, True, False)

    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip, text, cond_keyframes_type, frame_count, token_normalization, weight_interpretation):
        text = text.strip()
        conditionings = []
        for line in text.splitlines():
            if USE_BLK:
                encoded = blk_adv.encode(clip=clip, text=line, token_normalization=token_normalization, weight_interpretation=weight_interpretation)
            else:
                encoded = CLIPTextEncode.encode(clip=clip, text=line)

            conditionings.append([encoded[0][0][0], encoded[0][0][1]])

        conditioning_count = len(conditionings)
        cond_keyframes = self.calculate_cond_keyframes(cond_keyframes_type, frame_count, conditioning_count)

        return (conditionings, cond_keyframes, frame_count)

    def calculate_cond_keyframes(self, type, frame_count, conditioning_count):
        if type == "linear":
            return np.linspace(frame_count // conditioning_count, frame_count, conditioning_count, dtype=int).tolist()

        elif type == "sinus":
            # Create a sinusoidal distribution
            t = np.linspace(0, np.pi, conditioning_count)
            sinus_values = np.sin(t) 
            # Normalize the sinusoidal values to 0-1 range
            normalized_values = (sinus_values - sinus_values.min()) / (sinus_values.max() - sinus_values.min())
            # Scale to frame count and shift to avoid starting at frame 0
            scaled_values = normalized_values * (frame_count - 1) + 1
            # Ensure unique keyframes by rounding and converting to integer
            unique_keyframes = np.round(scaled_values).astype(int)
            # Deduplicate while preserving order
            unique_keyframes = np.unique(unique_keyframes, return_index=True)[1]
            return sorted(unique_keyframes.tolist())
    
        elif type == "sinus_inverted":
            return (np.cos(np.linspace(0, np.pi, conditioning_count)) * (frame_count - 1) + 1).astype(int).tolist()

        elif type == "half_sinus":
            return (np.sin(np.linspace(0, np.pi / 2, conditioning_count)) * (frame_count - 1) + 1).astype(int).tolist()

        elif type == "half_sinus_inverted":
            return (np.cos(np.linspace(0, np.pi / 2, conditioning_count)) * (frame_count - 1) + 1).astype(int).tolist()

        else:
            raise ValueError("Unsupported cond_keyframes_type: " + type)
    
class KSamplerSeq:

    def __init__(self):
        self.previous_seed = None
        self.current_seed = None

    def initialize_seeds(self, initial_seed):
        self.previous_seed = initial_seed
        self.current_seed = initial_seed

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Base seed for the sequence."}),
                    "seed_mode_seq": (["increment", "decrement", "random", "fixed"], {"tooltip": "How to evolve the seed each loop."}),
                    "alternate_values": ("BOOLEAN", {"default": True, "tooltip": "Alternate certain parameters every other loop."}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Sampler steps per loop."}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01, "tooltip": "Classifier-free guidance."}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampler algorithm."} ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Noise schedule."} ),
                    "sequence_loop_count": ("INT", {"default": 20, "min": 1, "max": 1024, "step": 1, "tooltip": "How many loops to run."}),
                    "positive_seq": ("CONDITIONING_SEQ", {"tooltip": "List of positive conditionings with frame indices."} ),
                    "negative_seq": ("CONDITIONING_SEQ", {"tooltip": "List of negative conditionings with frame indices."} ),
                    "use_conditioning_slerp": ("BOOLEAN", {"default": False, "tooltip": "Interpolate between consecutive conditionings using slerp."}),
                    "cond_slerp_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Interpolation amount for slerp."}),
                    "latent_image": ("LATENT", {"tooltip": "Initial latent input."} ),
                    "use_latent_interpolation": ("BOOLEAN", {"default": False, "tooltip": "Blend/slerp/cosine between consecutive outputs."}),
                    "latent_interpolation_mode": (["Blend", "Slerp", "Cosine Interp"], {"tooltip": "Method for latent interpolation."}),
                    "latent_interp_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Interpolation weight for latents."}),
                    "denoise_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise for the first loop (1.0 = full)."}),
                    "denoise_seq": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise for subsequent loops."}),
                    "unsample_latents": ("BOOLEAN", {"default": False, "tooltip": "Reverse a few steps before resampling to add variation."})
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

    def update_alternate_seed(self, loop_count):
        if loop_count % 3 == 0:
            if self.previous_seed is None:
                self.previous_seed = self.current_seed
            else:
                self.previous_seed, self.current_seed = self.current_seed, self.previous_seed + 1 if loop_count // 2 % 2 == 0 else self.previous_seed - 1
        return self.current_seed

    def alternate_denoise(self, current_denoise):
        return 0.95 if current_denoise == 0.75 else 0.75

    def sample(self, model, seed, seed_mode_seq, alternate_values, steps, cfg, sampler_name, scheduler, sequence_loop_count, positive_seq, negative_seq, cond_slerp_strength, latent_image, use_latent_interpolation, latent_interpolation_mode, latent_interp_strength, denoise_start=1.0, denoise_seq=0.5, use_conditioning_slerp=False, unsample_latents=False, alternate_mode=False):
        positive_seq = positive_seq
        negative_seq = negative_seq
        results = []
        positive_conditioning = None
        negative_conditioning = None

        self.initialize_seeds(seed)

        for loop_count in range(sequence_loop_count):
            if alternate_values and loop_count % 2 == 0:
                seq_seed = self.update_alternate_seed(seed) if seed_mode_seq != "fixed" else seed
                #denoise_seq = self.alternate_denoise(denoise_seq)
            else:
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


class KSamplerSeq2:

    def __init__(self):
        self.previous_seed = None
        self.current_seed = None

    def initialize_seeds(self, initial_seed):
        self.previous_seed = initial_seed
        self.current_seed = initial_seed

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Base seed for the sequence."}),
                    "seed_mode_seq": (["increment", "decrement", "random", "fixed"], {"tooltip": "How to evolve the seed each loop."}),
                    "alternate_values": ("BOOLEAN", {"default": True, "tooltip": "Alternate certain parameters every other loop."}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Sampler steps per loop."}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01, "tooltip": "Classifier-free guidance."}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampler algorithm."} ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Noise schedule."} ),
                    "frame_count": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1, "tooltip": "Total frames to run. If 0, inferred from conditionings."}),
                    "cond_keyframes": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1, "tooltip": "Keyframe indices to advance conditionings (list)."}),
                    "positive_seq": ("CONDITIONING", {"tooltip": "Positive conditioning list."} ),
                    "negative_seq": ("CONDITIONING", {"tooltip": "Negative conditioning list."} ),
                    "use_conditioning_slerp": ("BOOLEAN", {"default": False, "tooltip": "Interpolate between consecutive conditionings using slerp."}),
                    "cond_slerp_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Interpolation amount for slerp."}),
                    "latent_image": ("LATENT", {"tooltip": "Initial latent input."} ),
                    "use_latent_interpolation": ("BOOLEAN", {"default": False, "tooltip": "Blend/slerp/cosine between consecutive outputs."}),
                    "latent_interpolation_mode": (["Blend", "Slerp", "Cosine Interp"], {"tooltip": "Method for latent interpolation."}),
                    "latent_interp_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Interpolation weight for latents."}),
                    "denoise_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise for the first loop (1.0 = full)."}),
                    "denoise_seq": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise for subsequent loops."}),
                    "unsample_latents": ("BOOLEAN", {"default": False, "tooltip": "Reverse a few steps before resampling to add variation."}),
                    "inject_noise": ("BOOLEAN", {"default": True, "tooltip": "Add random noise between loops for variety."}),
                    "noise_strength": ("FLOAT", {"default": 0.1, "max": 1.0, "min": 0.001, "step": 0.001, "tooltip": "Magnitude of injected noise."}),
                    "denoise_sine": ("BOOLEAN", {"default": True, "tooltip": "Vary denoise over loops using a sine wave."}),
                    "denoise_max": ("FLOAT", {"default": 0.9, "max": 1.0, "min": 0.0, "step": 0.001, "tooltip": "Max denoise when sine modulation is enabled."}),
                    "seed_keying": ("BOOLEAN", {"default": True, "tooltip": "Modulate seed by a schedule to create patterns."}),
                    "seed_keying_mode": (["sine", "modulo"], {"tooltip": "Mode for seed modulation."}),
                    "seed_divisor": ("INT", {"default": 4, "max": 1024, "min": 2, "step": 1, "tooltip": "Period or divisor for seed/keyframe modulation."}),
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
    
    def alternate_seed_modulo(self, current, seed, divisor):
        if current % divisor == 0:
            new_seed = (seed + current) % 0xffffffffffffffff
        else:
            new_seed = seed
        return new_seed
    
    def alternate_seed_sine(self, current, start_seed, divisor):
        seed = 1000 * np.sin(2 * math.pi * current / divisor) + start_seed
        return seed

    def alternate_denoise(self, curent, total, start_denoise=0.5, max_denoise=0.95):
        amplitude = (max_denoise - start_denoise) / 2
        mid_point = (max_denoise + start_denoise) / 2
        cycle_position = (math.pi * 2 * curent) / total
        current_denoise = amplitude * math.sin(cycle_position) + mid_point
        return current_denoise
    
    def inject_noise(self, latent_image, noise_strength):
        noise = torch.randn_like(latent_image) * noise_strength
        return latent_image + noise

    def sample(self, model, seed, seed_mode_seq, alternate_values, steps, cfg, sampler_name, scheduler, 
               frame_count, cond_keyframes, positive_seq, negative_seq, cond_slerp_strength, latent_image, 
               use_latent_interpolation, latent_interpolation_mode, latent_interp_strength, denoise_start=1.0, 
               denoise_seq=0.5, use_conditioning_slerp=False, unsample_latents=False, alternate_mode=False, 
               inject_noise=True, noise_strength=0.1, denoise_sine=True, denoise_max=0.9, seed_keying=True, 
               seed_keying_mode="sine", seed_divisor=4):
        
        if not isinstance(positive_seq, list):
            positive_seq = [positive_seq]
        if not isinstance(negative_seq, list):
            negative_seq = [negative_seq]
        if not isinstance(cond_keyframes, list):
            cond_keyframes = [cond_keyframes]
        cond_keyframes.sort()

        positive_cond_idx = 0
        negative_cond_idx = 0
        results = []

        self.initialize_seeds(seed)
        sequence_loop_count = max(frame_count, len(positive_seq)) if cond_keyframes else len(positive_seq)

        print(f"Starting loop sequence with {sequence_loop_count} frames.")
        print(f"Using {len(positive_seq)} positive conditionings and {len(negative_seq)} negative conditionings")
        print(f"Conditioning keyframe schedule is: {', '.join(map(str, cond_keyframes))}")

        for loop_count in range(sequence_loop_count):
            if loop_count in cond_keyframes:
                positive_cond_idx = min(positive_cond_idx + 1, len(positive_seq) - 1)
                negative_cond_idx = min(negative_cond_idx + 1, len(negative_seq) - 1)

            positive_conditioning = positive_seq[positive_cond_idx]
            negative_conditioning = negative_seq[negative_cond_idx]

            if seed_keying:
                if seed_keying_mode == "sine":
                    seq_seed = seed if loop_count <= 0 else self.alternate_seed_sine(loop_count, seed, seed_divisor)
                else:
                    seq_seed = seed if loop_count <= 0 else self.alternate_seed_modulo(loop_count, seed, seed_divisor)
            else:
                seq_seed = seed if loop_count <= 0 else self.update_seed(seq_seed, seed_mode_seq)

            seq_seed = seed if loop_count <= 0 else self.update_seed(seq_seed, seed_mode_seq)
            print(f"Loop count: {loop_count}, Seed: {seq_seed}")

            last_positive_conditioning = positive_conditioning if positive_conditioning else None
            last_negative_conditioning = negative_conditioning if negative_conditioning else None

            if use_conditioning_slerp and (last_positive_conditioning and last_negative_conditioning):
                a, b = last_positive_conditioning[0].clone(), positive_conditioning[0].clone()
                na, nb = last_negative_conditioning[0].clone(), negative_conditioning[0].clone()
                pa, pb = last_positive_conditioning[1]["pooled_output"].clone(), positive_conditioning[1]["pooled_output"].clone()
                npa, npb = last_negative_conditioning[1]["pooled_output"].clone(), negative_conditioning[1]["pooled_output"].clone()
                pos_cond = slerp(cond_slerp_strength, a, b)
                pos_pooled = slerp(cond_slerp_strength, pa, pb)
                neg_cond = slerp(cond_slerp_strength, na, nb)
                neg_pooled = slerp(cond_slerp_strength, npa, npb)
                positive_conditioning = [pos_cond, {"pooled_output": pos_pooled}]
                negative_conditioning = [neg_cond, {"pooled_output": neg_pooled}]

            positive_conditioning = [positive_conditioning]
            negative_conditioning = [negative_conditioning]

            end_at_step = steps
            if results and len(results) > 0:
                latent_input = {'samples': results[-1]}
                denoise = self.alternate_denoise(loop_count, sequence_loop_count, denoise_seq, denoise_max) if denoise_sine else denoise_seq
                start_at_step = round((1 - denoise) * steps)
                end_at_step = steps
            else:
                latent_input = latent_image
                denoise = denoise_start

            if unsample_latents and loop_count > 0:
                force_full_denoise = not (loop_count > 0 or loop_count <= steps - 1)
                disable_noise = False
                if seed_keying:
                    if seed_keying_mode == "modulo" and loop_count % seed_divisor == 0:
                        unsampled_latent = latent_input
                    else:
                        unsampled_latent = unsample(model=model, seed=seq_seed, cfg=cfg, sampler_name=sampler_name, steps=steps, end_at_step=end_at_step, scheduler=scheduler, normalize=False, positive=positive_conditioning, negative=negative_conditioning, latent_image=latent_input)[0]
                else:
                    unsampled_latent = unsample(model=model, seed=seq_seed, cfg=cfg, sampler_name=sampler_name, steps=steps, end_at_step=end_at_step, scheduler=scheduler, normalize=False, positive=positive_conditioning, negative=negative_conditioning, latent_image=latent_input)[0]
                if inject_noise and loop_count > 0:
                    print(f"Injecting noise at {noise_strength} strength.")
                    unsampled_latent['samples'] = self.inject_noise(unsampled_latent['samples'], noise_strength)
                sample = nodes.common_ksampler(model, seq_seed, steps, cfg, sampler_name, scheduler, positive_conditioning, negative_conditioning, unsampled_latent, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)[0]['samples']
            else:
                if inject_noise and loop_count > 0:
                    print(f"Injecting noise at {noise_strength} strength.")
                    latent_input['samples'] = self.inject_noise(latent_input['samples'], noise_strength)
                sample = nodes.common_ksampler(model, seq_seed, steps, cfg, sampler_name, scheduler, positive_conditioning, negative_conditioning, latent_input, denoise=denoise)[0]['samples']

            if use_latent_interpolation and results and loop_count > 0:
                if latent_interpolation_mode == "Blend":
                    sample = blend_latents(latent_interp_strength, results[-1], sample)
                elif latent_interpolation_mode == "Slerp":
                    sample = slerp_latents(latent_interp_strength, results[-1], sample)
                elif latent_interpolation_mode == "Cosine Interp":
                    sample = cosine_interp_latents(latent_interp_strength, results[-1], sample)

            results.append(sample)

        results = torch.cat(results, dim=0)
        results = {'samples': results}
        return (results,)
    

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeList": CLIPTextEncodeSequence,
    "CLIPTextEncodeSequence2": CLIPTextEncodeSequence2,
    "KSamplerSeq": KSamplerSeq,
    "KSamplerSeq2": KSamplerSeq2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeList": "CLIP Text Encode Sequence (Advanced)",
    "CLIPTextEncodeSequence2": "CLIP Text Encode Sequence (v2)",
    "KSamplerSeq": "KSampler Sequence",
    "KSamplerSeq2": "KSampler Sequence (v2)",
}
