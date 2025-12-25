import torch
import torch.nn.functional as F
import gc
import logging

from typing import Tuple, Optional, Dict, List

from comfy.samplers import KSampler
from nodes import common_ksampler

try:
    from comfy_extras.nodes_wan import WanImageToVideo
except Exception:
    WanImageToVideo = None

try:
    from comfy_extras.nodes_model_advanced import ModelSamplingSD3
except Exception:
    ModelSamplingSD3 = None

MASK64 = 0xffffffffffffffff
SAFE_TORCH_SEED_MOD = (1 << 63) - 1
UPSCALE_MODES = ["nearest", "nearest-exact", "bilinear", "bicubic", "area", "bislerp", "bilserp"]
LOW_PRE_MULT = 0.75


# ---------- utilities ----------

def latent_stats(t: torch.Tensor) -> str:
    try:
        return (
            f"shape={tuple(t.shape)} dtype={t.dtype} dev={t.device} "
            f"min={float(t.min()):.6f} max={float(t.max()):.6f} "
            f"mean={float(t.mean()):.6f} std={float(t.std()):.6f}"
        )
    except Exception:
        return (
            f"shape={tuple(getattr(t, 'shape', []))} dtype={getattr(t, 'dtype', None)} "
            f"dev={getattr(t, 'device', None)}"
        )

def log_sampler_settings(pipeline: str, tag: str, model_name: str, seed: int, steps: int, cfg_val: float, sampler: str, sched: str,
    denoise_val: float, start_step: int, last_step: int, add_noise_flag: bool, force_full: bool,
    latent_in: torch.Tensor):
    
    def _print_info(msg: str):
        try:
            logging.info(f"[WanMoE {pipeline}] {msg}")
        except Exception:
            pass
    
    _print_info(
        f"model={model_name} seed={seed} steps={steps} cfg={cfg_val} sampler={sampler} scheduler={sched} "
        f"denoise={denoise_val} start={start_step} last={last_step} add_noise={add_noise_flag} force_full_denoise={force_full} \n"
        f"    in_latent: {latent_stats(latent_in)}"
    )

def randn_like(x: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed & MASK64)
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)

def inject_noise(x: torch.Tensor, strength: float, sigma: float, seed: int) -> torch.Tensor:
    if strength <= 0.0:
        return x
    x.add_(randn_like(x, seed).mul_(sigma * strength))
    return x

def lerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    return a * (1.0 - t) + b * t

def ensure_latents_5d_channels_first(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if x.dim() == 5:
        return x, False
    if x.dim() == 4:
        B, C, H, W = x.shape
        return x.view(B, C, 1, H, W), True
    raise ValueError(f"Expected 4D/5D, got {x.shape}")

def restore_original_dims_channels_first(x: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return x[:, :, 0] if squeezed else x

def pick_interpolation_mode(mode: str):
    if mode == "nearest-exact": return "nearest-exact"
    if mode in ("nearest", "bilinear", "bicubic", "area"): return mode
    return "bilinear"

def safe_interpolate(x: torch.Tensor, size, mode: str, use_antialias: bool):
    try:
        if mode in ("bilinear", "bicubic"):
            return F.interpolate(x, size=size, mode=mode, align_corners=False, antialias=use_antialias)
        if mode == "nearest-exact":
            return F.interpolate(x, size=size, mode=mode)
        return F.interpolate(x, size=size, mode=mode)
    except TypeError:
        if mode == "nearest-exact": mode = "nearest"
        if mode in ("bilinear", "bicubic"):
            return F.interpolate(x, size=size, mode=mode, align_corners=False)
        return F.interpolate(x, size=size, mode=mode)

def bi_slerp_spatial(x4: torch.Tensor, new_hw, use_antialias: bool) -> torch.Tensor:
    y_lin = safe_interpolate(x4, new_hw, "bilinear", use_antialias)
    eps = 1e-8
    src_norm = torch.linalg.vector_norm(x4, ord=2, dim=1, keepdim=True).clamp_min(eps)
    tgt_norm = safe_interpolate(src_norm, new_hw, "nearest", use_antialias)
    dir_vec = y_lin / torch.linalg.vector_norm(y_lin, ord=2, dim=1, keepdim=True).clamp_min(eps)
    return dir_vec * tgt_norm

def upscale_latents(x5d: torch.Tensor, factor: float, mode: str, use_antialias: bool) -> torch.Tensor:
    if factor <= 1.0:
        return x5d
    B, C, F, H, W = x5d.shape
    new_h = max(1, int(round(H * factor)))
    new_w = max(1, int(round(W * factor)))
    x4 = x5d.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
    y4 = bi_slerp_spatial(x4, (new_h, new_w), use_antialias) if mode in ("bislerp", "bilserp") else safe_interpolate(x4, (new_h, new_w), pick_interpolation_mode(mode), use_antialias)
    return y4.view(B, F, C, new_h, new_w).permute(0, 2, 1, 3, 4)

def get_sigmas_for(model, steps: int, scheduler: str, device: torch.device):
    steps = max(1, int(steps))
    try:
        ks = KSampler(model, steps, device)
        try:
            sig = ks.get_sigmas(scheduler)
        except Exception:
            sig = ks.sigmas
    except TypeError:
        ks = KSampler(model)
        sig = ks.get_sigmas(steps, scheduler)
    if isinstance(sig, torch.Tensor):
        return sig.to(device)
    return torch.tensor(sig, device=device)

def get_sigma_for_step_cached(cache: Dict, key: tuple, model, steps: int, scheduler: str,
                              step_index: int, device: torch.device) -> float:
    k = (key, steps, scheduler)
    if k not in cache:
        cache[k] = get_sigmas_for(model, steps, scheduler, device)
    sigmas = cache[k]
    idx = max(0, min(step_index, sigmas.shape[0] - 1))
    return float(sigmas[idx].item())

def ksampler_range(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                   latent_dict, denoise, start_step, last_step, add_noise: bool, force_full_denoise: bool = True):
    seed = int(seed) & MASK64
    disable_noise = not add_noise
    return common_ksampler(
        model, seed, steps, cfg, sampler_name, scheduler,
        positive, negative, latent_dict,
        denoise=denoise, disable_noise=disable_noise,
        start_step=start_step, last_step=last_step,
        force_full_denoise=force_full_denoise
    )[0]

def apply_model_shift(model, shift: float):
    if not ModelSamplingSD3:
        return model
    ms = ModelSamplingSD3()
    return ms.patch(model, float(shift))[0]

def pick_autocast_dtype(device: torch.device, precision_mode: str) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    pm = precision_mode.lower()
    if pm == "fp32": return None
    if pm == "bf16": return torch.bfloat16
    if pm == "fp16": return torch.float16
    return torch.bfloat16

 


# ---------- transitions (image space; NHWC) ----------

def progress_01(t: int, T: int) -> float:
    if T <= 1:
        return 1.0
    return float(t) / float(T - 1)

def smoothstep(e0: torch.Tensor, e1: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    t = torch.clamp((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def build_transition_mask(mode: str, t_index: int, t_total: int, H: int, W: int,
                          device: torch.device, softness: float):
    if t_total <= 0:
        return None
    p = progress_01(t_index, t_total)
    soft = max(1e-6, float(softness))

    if mode == "crossfade":
        return torch.full((1, 1, H, W), p, device=device)

    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, H, device=device),
        torch.linspace(0.0, 1.0, W, device=device),
        indexing="ij"
    )

    def smooth01(t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    if mode == "swipe-left":
        d = 1.0 - xx
        a = smooth01((p - d) / soft)
        return a.unsqueeze(0).unsqueeze(0)
    if mode == "swipe-right":
        d = xx
        a = smooth01((p - d) / soft)
        return a.unsqueeze(0).unsqueeze(0)
    if mode == "swipe-up":
        d = 1.0 - yy
        a = smooth01((p - d) / soft)
        return a.unsqueeze(0).unsqueeze(0)
    if mode == "swipe-down":
        d = yy
        a = smooth01((p - d) / soft)
        return a.unsqueeze(0).unsqueeze(0)

    if mode in ("ellipse-in", "ellipse-out"):
        yy2, xx2 = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, W, device=device),
            indexing="ij"
        )
        w = torch.tensor(float(W), device=device)
        h = torch.tensor(float(H), device=device)
        xpix = xx2 * (w * 0.5)
        ypix = yy2 * (h * 0.5)
        r = torch.sqrt(xpix * xpix + ypix * ypix)
        r_corner = torch.sqrt((w * 0.5) * (w * 0.5) + (h * 0.5) * (h * 0.5))
        r = r / r_corner
        if mode == "ellipse-in":
            d = 1.0 - r
        else:
            d = r
        a = smooth01((p - d) / soft)
        return a.unsqueeze(0).unsqueeze(0)

    return torch.full((1, 1, H, W), p, device=device)

def make_even(n: int) -> int:
    return n if (n % 2 == 0) else max(0, n - 1)

def decode_latents_to_images(vae, x5d: torch.Tensor) -> torch.Tensor:
    img = vae.decode(x5d)
    img = torch.clamp(img, 0.0, 1.0)
    if img.dim() not in (4, 5):
        raise ValueError(f"Unexpected VAE decode shape: {tuple(img.shape)}")
    return img


def compose_forward_overlap_images(A: torch.Tensor,
                                   B: torch.Tensor,
                                   mode: str,
                                   T: int,
                                   device: torch.device,
                                   store_cpu: bool,
                                   softness: float) -> torch.Tensor:
    if T <= 0 or mode == "cut":
        Bsz, _, H, W, C = A.shape
        return torch.empty((Bsz, 0, H, W, C), device="cpu" if store_cpu else device, dtype=A.dtype)

    Bsz, FA, H, W, C = A.shape
    FB = B.shape[1]
    T = min(T, FA, FB)

    out_device = "cpu" if store_cpu else device
    out = torch.empty((Bsz, T, H, W, C), device=out_device, dtype=A.dtype)

    for t in range(T):
        a_idx = FA - T + t
        b_idx = t
        frameA = A[:, a_idx:a_idx+1, ...].to(device)
        frameB = B[:, b_idx:b_idx+1, ...].to(device)
        alpha2d = build_transition_mask(mode, t, T, H, W, device, softness)
        alpha = alpha2d.unsqueeze(-1)
        blended = frameA * (1.0 - alpha) + frameB * alpha
        out[:, t:t+1, ...] = blended.to(out_device)

        del frameA, frameB, alpha2d, alpha, blended
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out

def assemble_scenes_overlap_ordered(scenes_img: List[torch.Tensor],
                                    scene_transition: str,
                                    scene_transition_frames: int,
                                    transition_softness: float,
                                    store_scenes_cpu: bool,
                                    device: torch.device) -> torch.Tensor:
    """
    For pair i→i+1 with Ti:
      emit A[start_i : FA - Ti]
      emit blend(A tail Ti → B head Ti)
    where start_i = 0 for i==0 else T_{i-1}.
    Finally emit last[T_{N-2}: ].
    """
    N = len(scenes_img)
    if N == 1 or scene_transition == "cut" or scene_transition_frames <= 0:
        out = scenes_img[0]
        for i in range(1, N):
            out = torch.cat([out, scenes_img[i]], dim=1)
        return out.to(device) if out.device.type == "cpu" else out

    T_raw = scene_transition_frames
    T_list: List[int] = []
    for i in range(N - 1):
        FA = scenes_img[i].shape[1]
        FB = scenes_img[i + 1].shape[1]
        T_list.append(max(0, min(T_raw, FA, FB)))

    pieces: List[torch.Tensor] = []

    for i in range(N - 1):
        A = scenes_img[i]
        B = scenes_img[i + 1]
        FA = A.shape[1]
        Ti = T_list[i]
        prev_T = T_list[i - 1] if i > 0 else 0

        start_a = prev_T if i > 0 else 0
        end_a = max(0, FA - Ti)
        if end_a > start_a:
            pieces.append(A[:, start_a:end_a, ...])

        if Ti > 0:
            tr = compose_forward_overlap_images(
                A, B, scene_transition, Ti,
                device=device, store_cpu=store_scenes_cpu, softness=float(transition_softness)
            )
            if tr.numel() > 0:
                pieces.append(tr)

    last = scenes_img[-1]
    last_T = T_list[-1] if len(T_list) > 0 else 0
    FL = last.shape[1]
    if FL > last_T:
        pieces.append(last[:, last_T:, ...])

    if all(p.device.type == "cpu" for p in pieces):
        final_img = torch.cat(pieces, dim=1).to(device)
    else:
        final_img = torch.cat([p.to(device) for p in pieces], dim=1)

    return final_img


# ---------- Node-age ----------

class WASWan22MoESamplerCtx:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # High expert
                "high_model": ("MODEL", {"tooltip": "Primary high noise Wan 2.2 UNet model used for the first stage."}),
                "high_model_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01, "tooltip": "Sampling shift amount applied to the high model (SD3 patch)."}),
                "high_steps": ("INT", {"default": 10, "min": 0, "max": 10000, "tooltip": "Number of sampling steps for the high model."}),
                "high_cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 64.0, "step": 0.1, "round": 0.1, "tooltip": "Classifier-free guidance for the high model."}),
                "high_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001, "tooltip": "Denoise strength for the high model pass."}),

                # Low expert
                "low_model": ("MODEL", {"tooltip": "Secondary low noise Wan 2.2 UNet model for refinement."}),
                "low_model_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01, "tooltip": "Sampling shift amount applied to the low model (SD3 patch)."}),
                "low_steps": ("INT", {"default": 10, "min": 0, "max": 10000, "tooltip": "Number of sampling steps for the low model."}),
                "low_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 64.0, "step": 0.1, "round": 0.1, "tooltip": "Classifier-free guidance for the low model."}),
                "low_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001, "tooltip": "Denoise strength for the low model pass."}),
                "low_step_offset": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001, "tooltip": "Start the low model at this fraction of its total steps (0–1)."}),

                # Final pass
                "final_low_pass": (["disable", "enable"], {"default": "enable", "tooltip": "Run an extra refinement using the low model at the end."}),
                "final_low_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01, "tooltip": "Sampling shift amount for the final low pass (SD3 patch)."}),
                "final_pass_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001, "tooltip": "Denoise strength for the final low pass."}),
                # Upscaling
                "upscale_stage": (["off", "early", "late", "both"], {"default": "early", "tooltip": "When to upscale latents: before low pass, after, both, or off."}),
                "upscale_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 8.0, "step": 0.1, "round": 0.1, "tooltip": "Scale factor for early/both upscaling."}),
                "upscale_mode": (UPSCALE_MODES, {"default": "nearest-exact", "tooltip": "Interpolation method for early/both upscaling."}),
                "late_upscale_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.1, "round": 0.1, "tooltip": "Scale factor for late/both upscaling."}),
                "late_upscale_mode": (UPSCALE_MODES, {"default": "bislerp", "tooltip": "Interpolation method for late/both upscaling."}),

                # Sampling & scheduler
                "sampler_name": (KSampler.SAMPLERS, {"default": "euler", "tooltip": "Sampler algorithm to use."}),
                "scheduler": (KSampler.SCHEDULERS, {"default": "simple", "tooltip": "Noise schedule for the sampler."}),
                "pipeline_mode": (["moe","non_moe","i2v_non_moe", "i2v_continuation_non_moe"], {"default": "moe", "tooltip": "Mixture-of-Experts pipeline, simpler non-MoE variant, or I2V-optimized non-MoE."}),

                # Noise & precision
                "noise_inject_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.001, "round": 0.001, "tooltip": "Extra noise injected between phases (0 disables)."}),
                "precision_mode": (["auto", "fp16", "bf16", "fp32"], {"default": "auto", "tooltip": "GPU AMP precision to use (auto selects best)."}),

                # Scenes / transitions
                "scene_batches": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "How many scenes to generate from the same input latent."}),
                "store_scenes_cpu": (["disable", "enable"], {"default": "enable", "tooltip": "Move intermediate tensors to CPU to save VRAM."}),
                "scene_transition": (["cut","crossfade","swipe-up","swipe-down","swipe-left","swipe-right","ellipse-out","ellipse-in"], {"default": "cut", "tooltip": "Transition mode when concatenating multiple scenes."}),
                "scene_transition_frames": ("INT", {"default": 0, "min": 0, "max": 4096, "tooltip": "Number of frames to overlap for transitions (0 cuts)."}),
                "transition_softness": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.001, "round": 0.001, "tooltip": "Softness/feathering used in transition masks."}),

                # IO (VAE before latent)
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning for the sampler (can be a list per scene)."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning for the sampler (can be a list per scene)."}),
                "vae": ("VAE", {"tooltip": "VAE used to decode latents to images."}),
                "latent": ("LATENT", {"tooltip": "Input latent to process (B,C,F,H,W or B,C,H,W)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Base random seed; scenes use deterministic offsets."}),
            },
            "optional": {
                "final_low_model": ("MODEL", {"tooltip": "Final low noise Wan 2.2 UNet model for refinement. If not provided, the low model is used."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "sampling/moe"

    def run(
        self,
        high_model, high_model_shift: float, high_steps: int, high_cfg: float, high_denoise: float,
        low_model, low_model_shift: float, low_steps: int, low_cfg: float, low_denoise: float, low_step_offset: float,
        final_low_pass: str, final_low_shift: float, final_pass_denoise: float,
        upscale_stage: str, upscale_factor: float, upscale_mode: str,
        late_upscale_factor: float, late_upscale_mode: str,
        sampler_name: str, scheduler: str, pipeline_mode: str,
        noise_inject_strength: float,
        precision_mode: str,
        scene_batches: int, store_scenes_cpu: str, scene_transition: str, scene_transition_frames: int,
        transition_softness: float,
        positive, negative, vae, latent, seed: int,
        final_low_model = None
    ):
        x_in = latent["samples"]
        x5d, squeezed = ensure_latents_5d_channels_first(x_in)
        B, C, F, H, W = x5d.shape
        device = x5d.device if x5d.is_cuda else torch.device("cpu")
        target_dtype = pick_autocast_dtype(device, precision_mode)

        high_model = apply_model_shift(high_model, high_model_shift)
        low_model  = apply_model_shift(low_model,  low_model_shift)

 

        inj_hi1 = max(0.0, float(noise_inject_strength))
        inj_hi2 = inj_hi1 * 0.625
        inj_low = inj_hi1 * 0.5
        sigma_cache: Dict = {}

        def run_moe_pipeline(lat: torch.Tensor, scene_seed: int, high_unet, low_unet, positive_in, negative_in) -> torch.Tensor:

            def get_model_device(m) -> torch.device:
                try:
                    if hasattr(m, 'model') and hasattr(m.model, 'parameters'):
                        p = next(m.model.parameters(), None)
                        if p is not None:
                            return p.device
                    if hasattr(m, 'diffusion_model') and hasattr(m.diffusion_model, 'parameters'):
                        p = next(m.diffusion_model.parameters(), None)
                        if p is not None:
                            return p.device
                except Exception:
                    pass
                return device

            acm = torch.cuda.amp.autocast if (device.type == "cuda") else torch.cpu.amp.autocast
            ac_kwargs = {"dtype": target_dtype} if device.type == "cuda" else {}

            with acm(**ac_kwargs):

                def ensure_conditioning(x):
                    if isinstance(x, list) and len(x) > 0:
                        first = x[0]
                        if isinstance(first, (list, tuple)) and len(first) >= 2 and isinstance(first[1], dict):
                            return x
                        if isinstance(x, (list, tuple)) and len(x) >= 2 and isinstance(x[1], dict):
                            return [list(x) if isinstance(x, tuple) else x]
                    if isinstance(x, (list, tuple)) and len(x) >= 2 and isinstance(x[1], dict):
                        return [list(x) if isinstance(x, tuple) else x]
                    return x

                positive_in = ensure_conditioning(positive_in)
                negative_in = ensure_conditioning(negative_in)
                hs = max(0, int(high_steps))
                ls = max(0, int(low_steps))
                total_steps = hs + ls
                cur = lat
                cur0 = cur

                if hs > 0:
                    h1_end = max(1, hs // 2)
                    h2_start, h2_end = h1_end, hs

                    model_dev = get_model_device(high_unet)
                    if cur.device != model_dev:
                        cur = cur.to(model_dev)
                        log_sampler_settings(
                            pipeline="MoE",
                            tag="HighPass#1",
                            model_name=getattr(high_unet, 'name', 'high_unet'),
                            seed=scene_seed, steps=total_steps, cfg_val=high_cfg, sampler=sampler_name, sched=scheduler,
                            denoise_val=high_denoise, start_step=0, last_step=h1_end, add_noise_flag=True, force_full=False,
                            latent_in=cur,
                        )
                        r1 = ksampler_range(
                            high_unet, scene_seed, total_steps, high_cfg, sampler_name, scheduler,
                            positive_in, negative_in, {"samples": cur}, high_denoise,
                            start_step=0, last_step=h1_end, add_noise=True, force_full_denoise=False
                        )["samples"]
                        cur = lerp(cur0, r1 * 0.5, 0.75) * 1.5

                        if inj_hi1 > 0.0:
                            s1 = get_sigma_for_step_cached(sigma_cache, ("H", id(high_unet)), high_unet, hs, scheduler, h1_end - 1, device)
                            cur = inject_noise(cur, inj_hi1, s1, (scene_seed + 101) & MASK64)

                        if h2_end > h2_start:
                            half_cfg = max(1.0, high_cfg * 0.5)
                            model_dev = get_model_device(high_unet)
                            if cur.device != model_dev:
                                cur = cur.to(model_dev)
                            log_sampler_settings(
                                pipeline="MoE",
                                tag="HighPass#2",
                                model_name=getattr(high_unet, 'name', 'high_unet'),
                                seed=scene_seed, steps=total_steps, cfg_val=high_cfg, sampler=sampler_name, sched=scheduler,
                                denoise_val=high_denoise, start_step=h2_start, last_step=h2_end, add_noise_flag=True, force_full=False,
                                latent_in=cur,
                            )
                            r2 = ksampler_range(
                                high_unet, scene_seed, total_steps, half_cfg, sampler_name, scheduler,
                                positive_in, negative_in, {"samples": cur}, high_denoise,
                                start_step=h2_start, last_step=h2_end, add_noise=True, force_full_denoise=False
                            )["samples"]
                            cur = r2 * 0.75
                            if inj_hi2 > 0.0:
                                s2 = get_sigma_for_step_cached(sigma_cache, ("H", id(high_unet)), high_unet, hs, scheduler, h2_end - 1, device)
                                cur = inject_noise(cur, inj_hi2, s2, (scene_seed + 202) & MASK64)

                    if ls > 0:
                        if (upscale_stage in ("early", "both") and upscale_factor > 1.0):
                            cur = upscale_latents(cur, float(upscale_factor), upscale_mode, use_antialias=True)

                        cur = cur * LOW_PRE_MULT

                        off = max(0.0, min(1.0, low_step_offset))
                        start_low = int(round(ls * off))
                        if start_low >= ls:
                            start_low = max(0, ls - 1)

                        model_dev = get_model_device(low_unet)
                        if cur.device != model_dev:
                            cur = cur.to(model_dev)
                        log_sampler_settings(
                            pipeline="MoE",
                            tag="LowPass",
                            model_name=getattr(low_unet, 'name', 'low_unet'),
                            seed=scene_seed, steps=total_steps, cfg_val=low_cfg, sampler=sampler_name, sched=scheduler,
                            denoise_val=low_denoise, start_step=start_low, last_step=ls, add_noise_flag=True, force_full=False,
                            latent_in=cur,
                        )
                        rL = ksampler_range(
                            low_unet, scene_seed, total_steps, low_cfg, sampler_name, scheduler,
                            positive_in, negative_in, {"samples": cur}, low_denoise,
                            start_step=start_low, last_step=total_steps, add_noise=True, force_full_denoise=True
                        )["samples"]
                        cur = rL

                    if final_low_pass == "enable":
                        if upscale_stage in ("late", "both") and late_upscale_factor > 1.0:
                            cur = upscale_latents(cur, late_upscale_factor, late_upscale_mode, use_antialias=True)

                        base_final_low = final_low_model if final_low_model is not None else low_unet
                        final_low = apply_model_shift(base_final_low, final_low_shift)

                        ls2 = max(0, int(low_steps))
                        off2 = max(0.0, min(1.0, float(low_step_offset)))
                        start_low2 = int(round(ls2 * off2))
                        if start_low2 >= ls2:
                            start_low2 = max(0, ls2 - 1)

                        model_dev = get_model_device(final_low)
                        if cur.device != model_dev:
                            cur = cur.to(model_dev)
                        log_sampler_settings(
                            pipeline="MoE",
                            tag="FinalLowPass",
                            model_name=getattr(final_low, 'name', 'final_low'),
                            seed=scene_seed, steps=total_steps, cfg_val=low_cfg, sampler=sampler_name, sched=scheduler,
                            denoise_val=final_pass_denoise, start_step=start_low2, last_step=total_steps, add_noise_flag=True, force_full=False,
                            latent_in=cur,
                        )
                        cur = ksampler_range(
                            final_low, scene_seed, total_steps, low_cfg,
                            sampler_name, scheduler, positive_in, negative_in,
                            {"samples": cur}, denoise=float(final_pass_denoise),
                            start_step=start_low2, last_step=total_steps, add_noise=True, force_full_denoise=True
                        )["samples"]

            return cur

        def run_nonmoe_pipeline(lat: torch.Tensor, scene_seed: int, high_unet, low_unet, positive_in, negative_in) -> torch.Tensor:
            acm = torch.cuda.amp.autocast if (device.type == "cuda") else torch.cpu.amp.autocast
            ac_kwargs = {"dtype": target_dtype} if device.type == "cuda" else {}

            def get_model_device(m) -> torch.device:
                try:
                    if hasattr(m, 'model') and hasattr(m.model, 'parameters'):
                        p = next(m.model.parameters(), None)
                        if p is not None:
                            return p.device
                    if hasattr(m, 'diffusion_model') and hasattr(m.diffusion_model, 'parameters'):
                        p = next(m.diffusion_model.parameters(), None)
                        if p is not None:
                            return p.device
                except Exception:
                    pass
                return device

            with acm(**ac_kwargs):

                def latent_stats(t: torch.Tensor) -> str:
                    try:
                        return f"shape={tuple(t.shape)} dtype={t.dtype} dev={t.device} min={float(t.min()):.6f} max={float(t.max()):.6f} mean={float(t.mean()):.6f} std={float(t.std()):.6f}"
                    except Exception:
                        return f"shape={tuple(getattr(t, 'shape', []))} dtype={getattr(t, 'dtype', None)} dev={getattr(t, 'device', None)}"

                def ensure_conditioning(x):
                    if isinstance(x, list) and len(x) > 0:
                        first = x[0]
                        if isinstance(first, (list, tuple)) and len(first) >= 2 and isinstance(first[1], dict):
                            return x
                        if isinstance(x, (list, tuple)) and len(x) >= 2 and isinstance(x[1], dict):
                            return [list(x) if isinstance(x, tuple) else x]
                    if isinstance(x, (list, tuple)) and len(x) >= 2 and isinstance(x[1], dict):
                        return [list(x) if isinstance(x, tuple) else x]
                    return x

                positive_in = ensure_conditioning(positive_in)
                negative_in = ensure_conditioning(negative_in)

                hs = max(0, int(high_steps))
                ls = max(0, int(low_steps))
                cur = lat
                logging.info(f"initial latent: {latent_stats(cur)}")

                if hs > 0:
                    h1_end = max(1, hs // 2)
                    h2_start, h2_end = h1_end, hs
                    model_dev = get_model_device(high_unet)
                    if cur.device != model_dev:
                        cur = cur.to(model_dev)
                    log_sampler_settings(
                        pipeline="Non-MoE",
                        tag="HighPass#1",
                        model_name=getattr(high_unet, 'name', 'high_unet'),
                        seed=scene_seed, steps=hs, cfg_val=high_cfg, sampler=sampler_name, sched=scheduler,
                        denoise_val=high_denoise, start_step=0, last_step=h1_end, add_noise_flag=True, force_full=False,
                        latent_in=cur,
                    )
                    r1 = ksampler_range(
                        high_unet, scene_seed, hs, high_cfg, sampler_name, scheduler,
                        positive_in, negative_in, {"samples": cur}, high_denoise,
                        start_step=0, last_step=h1_end, add_noise=True, force_full_denoise=False
                    )["samples"]
                    cur = r1
                    logging.info(f"HighPass#1 out: {latent_stats(cur)}")

                    # Second high pass: add_noise=False, leftover noise kept
                    if h2_end > h2_start:
                        if cur.device != model_dev:
                            cur = cur.to(model_dev)
                            log_sampler_settings(
                                pipeline="Non-MoE",
                                tag="HighPass#2",
                                model_name=getattr(high_unet, 'name', 'high_unet'),
                                seed=scene_seed, steps=hs, cfg_val=high_cfg, sampler=sampler_name, sched=scheduler,
                                denoise_val=high_denoise, start_step=h2_start, last_step=h2_end, add_noise_flag=False, force_full=False,
                                latent_in=cur,
                            )
                            r2 = ksampler_range(
                                high_unet, scene_seed, hs, high_cfg, sampler_name, scheduler,
                                positive_in, negative_in, {"samples": cur}, high_denoise,
                                start_step=h2_start, last_step=h2_end, add_noise=False, force_full_denoise=False
                            )["samples"]
                            cur = r2
                            logging.info(f"HighPass#2 out: {latent_stats(cur)}")

                    if ls > 0:
                        # Early upscaling like MoE
                        if (upscale_stage in ("early", "both") and upscale_factor > 1.0):
                            cur = upscale_latents(cur, float(upscale_factor), upscale_mode, use_antialias=True)

                        start_low = 0
                        if hs > 0:
                            start_low = int(h2_end)
                        # clamp to valid range [0, ls-1]
                        if start_low >= ls:
                            start_low = max(0, ls - 1)

                        model_dev = get_model_device(low_unet)
                        if cur.device != model_dev:
                            cur = cur.to(model_dev)
                        log_sampler_settings(
                            pipeline="Non-MoE",
                            tag="LowPass",
                            model_name=getattr(low_unet, 'name', 'low_unet'),
                            seed=scene_seed, steps=ls, cfg_val=low_cfg, sampler=sampler_name, sched=scheduler,
                            denoise_val=low_denoise, start_step=start_low, last_step=ls, add_noise_flag=False, force_full=False,
                            latent_in=cur,
                        )
                        rL = ksampler_range(
                            low_unet, scene_seed, ls, low_cfg, sampler_name, scheduler,
                            positive_in, negative_in, {"samples": cur}, low_denoise,
                            start_step=start_low, last_step=ls, add_noise=False, force_full_denoise=True
                        )["samples"]
                        cur = rL
                        logging.info(f"LowPass out: {latent_stats(cur)}")

                    if final_low_pass == "enable":
                        if upscale_stage in ("late", "both") and late_upscale_factor > 1.0:
                            cur = upscale_latents(cur, late_upscale_factor, late_upscale_mode, use_antialias=True)

                        base_final_low = final_low_model if final_low_model is not None else low_unet
                        final_low = apply_model_shift(base_final_low, final_low_shift)

                        ls2 = max(0, int(low_steps))
                        off2 = max(0.0, min(1.0, float(low_step_offset)))
                        start_low2 = int(round(ls2 * off2))
                        if start_low2 >= ls2:
                            start_low2 = max(0, ls2 - 1)

                        model_dev = get_model_device(final_low)
                        if cur.device != model_dev:
                            cur = cur.to(model_dev)
                        log_sampler_settings(
                            pipeline="Non-MoE",
                            tag="FinalLowPass",
                            model_name=getattr(final_low, 'name', 'final_low'),
                            seed=(scene_seed + 1) & MASK64, steps=max(1, low_steps), cfg_val=low_cfg, sampler=sampler_name, sched=scheduler,
                            denoise_val=final_pass_denoise, start_step=start_low2, last_step=ls2, add_noise_flag=True, force_full=False,
                            latent_in=cur,
                        )
                        cur = ksampler_range(
                            final_low, (scene_seed + 1) & MASK64, max(1, int(low_steps)), low_cfg,
                            sampler_name, scheduler, positive_in, negative_in,
                            {"samples": cur}, denoise=float(final_pass_denoise),
                            start_step=start_low2, last_step=ls2, add_noise=True, force_full_denoise=True
                        )["samples"]
                        logging.info(f"FinalLowPass out: {latent_stats(cur)}")

                    return cur

        def run_i2v_nonmoe_pipeline(lat: torch.Tensor, scene_seed: int, high_unet, low_unet, positive_in, negative_in) -> torch.Tensor:
            """I2V-optimized non-MoE pipeline that mimics official I2V workflow"""
            acm = torch.cuda.amp.autocast if (device.type == "cuda") else torch.cpu.amp.autocast
            ac_kwargs = {"dtype": target_dtype} if device.type == "cuda" else {}

            def get_model_device(m) -> torch.device:
                try:
                    if hasattr(m, 'model') and hasattr(m.model, 'parameters'):
                        p = next(m.model.parameters(), None)
                        if p is not None:
                            return p.device
                    if hasattr(m, 'diffusion_model') and hasattr(m.diffusion_model, 'parameters'):
                        p = next(m.diffusion_model.parameters(), None)
                        if p is not None:
                            return p.device
                except Exception:
                    pass
                return device

            with acm(**ac_kwargs):

                def ensure_conditioning(x):
                    if isinstance(x, list) and len(x) > 0:
                        first = x[0]
                        if isinstance(first, (list, tuple)) and len(first) >= 2 and isinstance(first[1], dict):
                            return x
                        if isinstance(x, (list, tuple)) and len(x) >= 2 and isinstance(x[1], dict):
                            return [list(x) if isinstance(x, tuple) else x]
                    if isinstance(x, (list, tuple)) and len(x) >= 2 and isinstance(x[1], dict):
                        return [list(x) if isinstance(x, tuple) else x]
                    return x

                positive_in = ensure_conditioning(positive_in)
                negative_in = ensure_conditioning(negative_in)
                total_steps = max(0, int(high_steps)) + max(0, int(low_steps))
                cur = lat
                
                logging.info(f"[I2V Non-MoE] Starting with latent: {latent_stats(cur)}")
                hs = max(0, int(high_steps))
                if hs > 0:
                    model_dev = get_model_device(high_unet)
                    if cur.device != model_dev:
                        cur = cur.to(model_dev)
                    
                    log_sampler_settings(
                        pipeline="I2V-Non-MoE",
                        tag="HighPass",
                        model_name=getattr(high_unet, 'name', 'i2v_model'),
                        seed=scene_seed, steps=total_steps, cfg_val=high_cfg, sampler=sampler_name, sched=scheduler,
                        denoise_val=high_denoise, start_step=0, last_step=hs, add_noise_flag=True, force_full=False,
                        latent_in=cur,
                    )
                    
                    result = ksampler_range(
                        high_unet, scene_seed, total_steps, high_cfg, sampler_name, scheduler,
                        positive_in, negative_in, {"samples": cur}, high_denoise,
                        start_step=0, last_step=hs, add_noise=True, force_full_denoise=False
                    )["samples"]
                    
                    cur = result
                    logging.info(f"[I2V Non-MoE] I2V pass output: {latent_stats(cur)}")
                
                if upscale_stage in ("early", "both") and upscale_factor > 1.0:
                    #cur = upscale_latents(cur, float(upscale_factor), upscale_mode, use_antialias=True)
                    logging.info(f"[I2V Non-MoE] Early upscaling not comptable with I2V Non-MoE")
                
                ls = max(0, int(low_steps))
                if ls > 0:

                    model_dev = get_model_device(low_unet)
                    if cur.device != model_dev:
                        cur = cur.to(model_dev)

                    log_sampler_settings(
                        pipeline="I2V-Non-MoE",
                        tag="LowPass",
                        model_name=getattr(low_unet, 'name', 'low_unet'),
                        seed=scene_seed, steps=total_steps, cfg_val=low_cfg, sampler=sampler_name, sched=scheduler,
                        denoise_val=low_denoise, start_step=hs, last_step=total_steps, add_noise_flag=False, force_full=False,
                        latent_in=cur,
                    )

                    cur = ksampler_range(
                        low_unet, scene_seed, total_steps, low_cfg, sampler_name, scheduler,
                        positive_in, negative_in, {"samples": cur}, low_denoise,
                        start_step=hs, last_step=total_steps, add_noise=False, force_full_denoise=True
                    )["samples"]

                if int(low_steps) > 0 and final_low_pass == "enable":

                    if upscale_stage in ("late", "both") and late_upscale_factor > 1.0:
                        cur = upscale_latents(cur, late_upscale_factor, late_upscale_mode, use_antialias=True)
                        logging.info(f"[I2V Non-MoE] After late upscaling: {latent_stats(cur)}")
                    
                    base_final_low = final_low_model if final_low_model is not None else low_unet
                    final_low = apply_model_shift(base_final_low, final_low_shift)
                    
                    refine_steps = max(1, int(low_steps))
                    off2 = max(0.0, min(1.0, float(low_step_offset)))
                    start_low2 = int(round(refine_steps * off2))
                    if start_low2 >= refine_steps:
                        start_low2 = max(0, refine_steps - 1)
                    model_dev = get_model_device(final_low)
                    if cur.device != model_dev:
                        cur = cur.to(model_dev)
                    
                    log_sampler_settings(
                        pipeline="I2V-Non-MoE",
                        tag="RefinementPass",
                        model_name=getattr(final_low, 'name', 'refinement_model'),
                        seed=scene_seed, steps=refine_steps, cfg_val=low_cfg, sampler=sampler_name, sched=scheduler,
                        denoise_val=final_pass_denoise, start_step=start_low2, last_step=refine_steps, add_noise_flag=False, force_full=True,
                        latent_in=cur,
                    )
                    
                    cur = ksampler_range(
                        final_low, scene_seed, refine_steps, low_cfg,
                        sampler_name, scheduler, positive_in, negative_in,
                        {"samples": cur}, denoise=float(final_pass_denoise),
                        start_step=start_low2, last_step=refine_steps, add_noise=False, force_full_denoise=True
                    )["samples"]
                    
                    logging.info(f"[I2V Non-MoE] Refinement pass output: {latent_stats(cur)}")
                
                return cur

        def is_conditioning(obj) -> bool:
            return (
                isinstance(obj, list) and len(obj) > 0 and
                isinstance(obj[0], (list, tuple)) and len(obj[0]) >= 2 and isinstance(obj[0][1], dict)
            )

        def is_cond_list(obj) -> bool:
            return isinstance(obj, list) and len(obj) > 0 and is_conditioning(obj[0])

        def pick_cond_for_scene(cond_any, idx: int):
            if is_cond_list(cond_any):
                j = idx if idx < len(cond_any) else (len(cond_any) - 1)
                return cond_any[j]
            return cond_any

        num_scenes = max(1, int(scene_batches))
        pos_is_list = is_cond_list(positive)
        neg_is_list = is_cond_list(negative)
        if num_scenes > 1 and not pos_is_list:
            try:
                print("WASWan22MoESamplerCtx: positive is single CONDITIONING; reusing across", num_scenes, "scenes")
            except Exception:
                pass
        if num_scenes > 1 and not neg_is_list:
            try:
                print("WASWan22MoESamplerCtx: negative is single CONDITIONING; reusing across", num_scenes, "scenes")
            except Exception:
                pass
        if pos_is_list and isinstance(positive, list) and len(positive) != num_scenes:
            try:
                print("WASWan22MoESamplerCtx: positive list length", len(positive), "!= scene_batches", num_scenes, "; clamping to last for remaining scenes")
            except Exception:
                pass
        if neg_is_list and isinstance(negative, list) and len(negative) != num_scenes:
            try:
                print("WASWan22MoESamplerCtx: negative list length", len(negative), "!= scene_batches", num_scenes, "; clamping to last for remaining scenes")
            except Exception:
                pass

        scenes_lat: List[torch.Tensor] = []
        for s_idx in range(num_scenes):
            
            scene_seed = seed + s_idx
            if scene_seed > 0xffffffffffffffff:
                scene_seed = seed
            
            scene_in = x5d.clone()
            pos_scene = pick_cond_for_scene(positive, s_idx)
            neg_scene = pick_cond_for_scene(negative, s_idx)

            def ensure_conditioning(x):
                if isinstance(x, list) and len(x) > 0:
                    first = x[0]
                    if isinstance(first, (list, tuple)) and len(first) >= 2 and isinstance(first[1], dict):
                        return x
                    if isinstance(x, (list, tuple)) and len(x) >= 2 and isinstance(x[1], dict):
                        return [list(x) if isinstance(x, tuple) else x]
                if isinstance(x, (list, tuple)) and len(x) >= 2 and isinstance(x[1], dict):
                    return [list(x) if isinstance(x, tuple) else x]
                return x
            
            pos_scene = ensure_conditioning(pos_scene)
            neg_scene = ensure_conditioning(neg_scene)
            
            if pipeline_mode == "non_moe":
                scene_out = run_nonmoe_pipeline(scene_in, scene_seed, high_model, low_model, pos_scene, neg_scene)
            elif pipeline_mode == "i2v_non_moe":
                scene_out = run_i2v_nonmoe_pipeline(scene_in, scene_seed, high_model, low_model, pos_scene, neg_scene)
            elif pipeline_mode == "i2v_continuation_non_moe":
                prev_available = (len(scenes_lat) > 0) and (WanImageToVideo is not None)
                if prev_available:
                    try:
                        prev_lat = scenes_lat[-1]
                        prev_img = decode_latents_to_images(vae, prev_lat)
                        if prev_img.dim() == 5:
                            kf = max(1, min(3, prev_img.shape[1]))
                            start_img = prev_img[0, -kf:, ...]
                        else:
                            kf = 1
                            start_img = prev_img

                        Bc, Cc, Fc, Hc, Wc = scene_in.shape
                        desired_len = int((Fc - 1) * 4 + 1)
                        width = int(start_img.shape[2])
                        height = int(start_img.shape[1])
                        batch_size = int(Bc)

                        wan_node = WanImageToVideo()
                        wan_out = wan_node.execute(pos_scene, neg_scene, vae, width, height, desired_len, batch_size, start_image=start_img)

                        try:
                            pos2, neg2, out_latent = wan_out
                        except Exception:
                            # Some builds return an object with attributes
                            pos2, neg2, out_latent = wan_out[0], wan_out[1], wan_out[2]

                        scene_seed2 = scene_seed
                        scene_in2 = out_latent["samples"]
                        scene_out = run_i2v_nonmoe_pipeline(scene_in2, scene_seed2, high_model, low_model, pos2, neg2)
                        if scene_out.dim() == 5 and kf > 0 and scene_out.shape[2] > kf:
                            scene_out = scene_out[:, :, 0:, ...]
                    except Exception:
                        scene_out = run_i2v_nonmoe_pipeline(scene_in, scene_seed, high_model, low_model, pos_scene, neg_scene)
                else:
                    scene_out = run_i2v_nonmoe_pipeline(scene_in, scene_seed, high_model, low_model, pos_scene, neg_scene)
            else:
                scene_out = run_moe_pipeline(scene_in, scene_seed, high_model, low_model, pos_scene, neg_scene)
            
            if store_scenes_cpu == "enable":
                scene_out = scene_out.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            scenes_lat.append(scene_out)
            
            del scene_in, scene_out
            
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                finally:
                    torch.cuda.empty_cache()
            gc.collect()

        # Decode scenes to IMAGE (NHWC)
        scenes_img: List[torch.Tensor] = []
        
        for s in scenes_lat:
            s_dev = device if s.device.type == "cpu" else s.device
            s = s.to(s_dev)
            img = decode_latents_to_images(vae, s)  # [B,F,H,W,3]
            img = img.cpu() if store_scenes_cpu == "enable" else img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            scenes_img.append(img)

        scenes_lat.clear()
        del scenes_lat
        
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
        gc.collect()

        final_img = assemble_scenes_overlap_ordered(
            scenes_img=scenes_img,
            scene_transition=scene_transition,
            scene_transition_frames=scene_transition_frames,
            transition_softness=transition_softness,
            store_scenes_cpu=(store_scenes_cpu == "enable"),
            device=device
        )  # [B,Fnew,H,W,3]

        scenes_img.clear()
        del scenes_img
        
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
        
        gc.collect()

        if final_img.dim() == 5:
            B_out, F_out, Himg, Wimg, C = final_img.shape
            images_out = final_img.reshape(B_out * F_out, Himg, Wimg, C).to("cpu") # [B, H, W, C]
        else:
            images_out = final_img.to("cpu")

        del final_img
        del x_in
        del x5d
        del sigma_cache
        
        if 'scene_out' in locals():
            del scene_out
        if 'scene_in' in locals():
            del scene_in
        if 'lat' in locals():
            del lat
        if 'cur' in locals():
            del cur
        if 'cur0' in locals():
            del cur0
        if 'r1' in locals():
            del r1
        if 'r2' in locals():
            del r2
        if 'rL' in locals():
            del rL
        if 'img' in locals():
            del img
        if 's' in locals():
            del s
        
        del high_model
        del low_model
        del vae
        
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        
        gc.collect()

        return (images_out,)


class WASMoEConditioningListAppend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cond_a": ("CONDITIONING", {"tooltip": "First conditioning or list of conditionings."}),
                "cond_b": ("CONDITIONING", {"tooltip": "Second conditioning or list of conditionings."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "count")
    FUNCTION = "append"
    CATEGORY = "conditioning"

    def append(self, cond_a, cond_b):
        def is_conditioning(obj) -> bool:
            return (
                isinstance(obj, list) and len(obj) > 0 and
                isinstance(obj[0], (list, tuple)) and len(obj[0]) >= 2 and isinstance(obj[0][1], dict)
            )
        def is_list_of_conditionings(obj) -> bool:
            return isinstance(obj, list) and len(obj) > 0 and is_conditioning(obj[0])
        def to_list_of_conditionings(x):
            if is_list_of_conditionings(x):
                return x
            if is_conditioning(x):
                return [x]
            return []
        la = to_list_of_conditionings(cond_a)
        lb = to_list_of_conditionings(cond_b)
        out = la + lb
        return (out, len(out))

NODE_CLASS_MAPPINGS = {
    "WASWan22MoESamplerCtx": WASWan22MoESamplerCtx,
    "WASMoEConditioningListAppend": WASMoEConditioningListAppend,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WASWan22MoESamplerCtx": "Wan 2.2 MoE Sampler (WAS)",
    "WASMoEConditioningListAppend": "Wan 2.2 MoE Conditioning Append (WAS)",
}
