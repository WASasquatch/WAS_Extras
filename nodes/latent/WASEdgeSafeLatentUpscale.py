from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


def is_latent_5d(samples: torch.Tensor) -> bool:
    return samples.dim() == 5


def flatten_5d_to_4d(samples: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    b, c, t, h, w = samples.shape
    flat = samples.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).contiguous()
    return flat, b, t


def unflatten_4d_to_5d(samples_btchw: torch.Tensor, b: int, t: int) -> torch.Tensor:
    bt, c, h, w = samples_btchw.shape
    if bt != b * t:
        raise ValueError(f"Shape mismatch: bt={bt} != b*t={b*t}")
    return samples_btchw.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def resize_tensor(x: torch.Tensor, size_hw: Tuple[int, int], mode: str) -> torch.Tensor:
    if mode == "bilinear":
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
    if mode == "bicubic":
        return F.interpolate(x, size=size_hw, mode="bicubic", align_corners=False)
    if mode == "area":
        return F.interpolate(x, size=size_hw, mode="area")
    if mode == "nearest-exact":
        return F.interpolate(x, size=size_hw, mode="nearest-exact")
    return F.interpolate(x, size=size_hw, mode=mode)


def gaussian_blur_depthwise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return x

    radius = max(1, int(math.ceil(3.0 * sigma)))
    ksize = 2 * radius + 1

    device = x.device
    dtype = x.dtype

    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_x = kernel_1d.view(1, 1, 1, ksize)
    kernel_y = kernel_1d.view(1, 1, ksize, 1)

    c = x.shape[1]
    x_pad = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    x_blur = F.conv2d(x_pad, kernel_x.expand(c, 1, 1, ksize), groups=c)
    x_blur = F.conv2d(x_blur, kernel_y.expand(c, 1, ksize, 1), groups=c)
    return x_blur


def sigmoid_weight(d: torch.Tensor, threshold: float, softness: float) -> torch.Tensor:
    t = float(threshold)
    s = max(1e-8, float(softness))
    return torch.sigmoid((d - t) / s)


def apply_temporal_ema(weight_bt1hw: torch.Tensor, b: int, t: int, ema: float) -> torch.Tensor:
    if ema <= 0.0 or t <= 1:
        return weight_bt1hw

    ema = float(ema)
    bt, c, h, w = weight_bt1hw.shape
    wgt = weight_bt1hw.reshape(b, t, c, h, w).contiguous()

    for bi in range(b):
        prev = wgt[bi, 0]
        for ti in range(1, t):
            cur = wgt[bi, ti]
            prev = prev * ema + cur * (1.0 - ema)
            wgt[bi, ti] = prev

    return wgt.reshape(bt, c, h, w).contiguous()


def sobel_grad_mag(x_bt1hw: torch.Tensor) -> torch.Tensor:
    device = x_bt1hw.device
    dtype = x_bt1hw.dtype

    kx = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(x_bt1hw, kx, padding=1)
    gy = F.conv2d(x_bt1hw, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    mx = x.amax(dim=(2, 3), keepdim=True)
    return x / (mx + 1e-8)


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def compute_damp_mask(
    latent_btchw_fp32: torch.Tensor,
    weight_bt1hw_fp32: torch.Tensor,
    gate_mode: str,
    grad_blur_sigma: float,
    damp_threshold: float,
    damp_softness: float,
    damp_power: float,
    damp_mask_blur_sigma: float,
) -> torch.Tensor:
    energy = latent_btchw_fp32.abs().mean(dim=1, keepdim=True)

    if grad_blur_sigma > 0.0:
        energy = gaussian_blur_depthwise(energy, float(grad_blur_sigma))

    grad = sobel_grad_mag(energy)
    grad = normalize_map(grad)

    mask = sigmoid_weight(grad, float(damp_threshold), float(damp_softness))
    mask = clamp01(mask)

    if gate_mode == "weight":
        mask = mask * clamp01(weight_bt1hw_fp32)
    elif gate_mode == "weight_sqrt":
        mask = mask * torch.sqrt(clamp01(weight_bt1hw_fp32) + 1e-8)
    elif gate_mode == "none":
        pass
    else:
        pass

    if damp_power != 1.0:
        mask = clamp01(mask).pow(float(damp_power))

    if damp_mask_blur_sigma > 0.0:
        mask = gaussian_blur_depthwise(mask, float(damp_mask_blur_sigma))
        mask = clamp01(mask)

    return mask


def apply_highpass_damping(
    latent_btchw_fp32: torch.Tensor,
    damp_mask_bt1hw_fp32: torch.Tensor,
    strength: float,
    highpass_sigma: float,
) -> torch.Tensor:
    s = float(strength)
    if s <= 0.0:
        return latent_btchw_fp32

    low = gaussian_blur_depthwise(latent_btchw_fp32, float(highpass_sigma)) if highpass_sigma > 0.0 else latent_btchw_fp32
    high = latent_btchw_fp32 - low

    m = clamp01(damp_mask_bt1hw_fp32)
    return low + high * (1.0 - s * m)


@dataclass
class AdaptiveBlendConfig:
    scale: float = 2.0
    smooth_mode: str = "bilinear"
    diff_blur_sigma: float = 0.6
    threshold: float = 0.12
    softness: float = 0.05
    weight_power: float = 1.0
    weight_blur_sigma: float = 0.0
    temporal_ema: float = 0.0

    enable_directional_damping: bool = True
    damping_strength: float = 0.35
    damping_gate_mode: str = "weight_sqrt"  # none|weight|weight_sqrt
    damping_grad_blur_sigma: float = 0.0
    damping_threshold: float = 0.25
    damping_softness: float = 0.08
    damping_power: float = 1.0
    damping_mask_blur_sigma: float = 0.6
    damping_highpass_sigma: float = 1.0
    damping_temporal_ema: float = 0.25

    preview_mode: str = "both"  # weight|damp|both
    output_mask_pixel_scale: int = 8


class WASAdaptiveDifferenceLatentUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05}),
                "smooth_mode": (["bilinear", "bicubic", "area"], {"default": "bilinear"}),

                "diff_blur_sigma": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 8.0, "step": 0.05}),
                "threshold": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.005}),
                "softness": ("FLOAT", {"default": 0.05, "min": 0.0005, "max": 1.0, "step": 0.001}),
                "weight_power": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 6.0, "step": 0.05}),
                "weight_blur_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "temporal_ema": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01}),

                "enable_directional_damping": ("BOOLEAN", {"default": True}),
                "damping_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "damping_gate_mode": (["none", "weight", "weight_sqrt"], {"default": "weight_sqrt"}),
                "damping_grad_blur_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "damping_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.005}),
                "damping_softness": ("FLOAT", {"default": 0.08, "min": 0.0005, "max": 1.0, "step": 0.001}),
                "damping_power": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 6.0, "step": 0.05}),
                "damping_mask_blur_sigma": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 8.0, "step": 0.05}),
                "damping_highpass_sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "damping_temporal_ema": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.99, "step": 0.01}),

                "preview_mode": (["weight", "damp", "both"], {"default": "both"}),
                "output_mask_pixel_scale": ("INT", {"default": 8, "min": 1, "max": 16, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "MASK", "IMAGE")
    RETURN_NAMES = ("latent", "mask", "mask_preview")
    FUNCTION = "upscale"
    CATEGORY = "WAS/Latent"

    def upscale(
        self,
        latent,
        scale: float,
        smooth_mode: str,
        diff_blur_sigma: float,
        threshold: float,
        softness: float,
        weight_power: float,
        weight_blur_sigma: float,
        temporal_ema: float,
        enable_directional_damping: bool,
        damping_strength: float,
        damping_gate_mode: str,
        damping_grad_blur_sigma: float,
        damping_threshold: float,
        damping_softness: float,
        damping_power: float,
        damping_mask_blur_sigma: float,
        damping_highpass_sigma: float,
        damping_temporal_ema: float,
        preview_mode: str,
        output_mask_pixel_scale: int,
    ):
        if "samples" not in latent:
            raise ValueError("LATENT input must be a dict containing key 'samples'.")

        samples: torch.Tensor = latent["samples"]
        orig_dtype = samples.dtype

        cfg = AdaptiveBlendConfig(
            scale=float(scale),
            smooth_mode=str(smooth_mode),
            diff_blur_sigma=float(diff_blur_sigma),
            threshold=float(threshold),
            softness=float(softness),
            weight_power=float(weight_power),
            weight_blur_sigma=float(weight_blur_sigma),
            temporal_ema=float(temporal_ema),
            enable_directional_damping=bool(enable_directional_damping),
            damping_strength=float(damping_strength),
            damping_gate_mode=str(damping_gate_mode),
            damping_grad_blur_sigma=float(damping_grad_blur_sigma),
            damping_threshold=float(damping_threshold),
            damping_softness=float(damping_softness),
            damping_power=float(damping_power),
            damping_mask_blur_sigma=float(damping_mask_blur_sigma),
            damping_highpass_sigma=float(damping_highpass_sigma),
            damping_temporal_ema=float(damping_temporal_ema),
            preview_mode=str(preview_mode),
            output_mask_pixel_scale=int(output_mask_pixel_scale),
        )

        was_5d = is_latent_5d(samples)
        if was_5d:
            samples_4d, b, t = flatten_5d_to_4d(samples)
            h, w = samples.shape[-2], samples.shape[-1]
        else:
            b = samples.shape[0]
            t = 1
            samples_4d = samples
            h, w = samples.shape[-2], samples.shape[-1]

        target_h = int(round(h * cfg.scale))
        target_w = int(round(w * cfg.scale))
        if target_h < 1 or target_w < 1:
            raise ValueError("Invalid target size computed from scale.")

        base = resize_tensor(samples_4d, (target_h, target_w), mode="nearest-exact")
        smooth = resize_tensor(samples_4d, (target_h, target_w), mode=cfg.smooth_mode)

        base_fp = base.float()
        smooth_fp = smooth.float()

        diff = (base_fp - smooth_fp).abs().mean(dim=1, keepdim=True)
        if cfg.diff_blur_sigma > 0.0:
            diff = gaussian_blur_depthwise(diff, cfg.diff_blur_sigma)

        wgt = sigmoid_weight(diff, cfg.threshold, cfg.softness)
        wgt = clamp01(wgt)

        if cfg.weight_power != 1.0:
            wgt = clamp01(wgt).pow(cfg.weight_power)

        if cfg.weight_blur_sigma > 0.0:
            wgt = gaussian_blur_depthwise(wgt, cfg.weight_blur_sigma)
            wgt = clamp01(wgt)

        if was_5d and cfg.temporal_ema > 0.0:
            wgt = apply_temporal_ema(wgt, b=b, t=t, ema=cfg.temporal_ema)

        out_fp = base_fp * (1.0 - wgt) + smooth_fp * wgt

        damp_mask = torch.zeros_like(wgt)
        if cfg.enable_directional_damping and cfg.damping_strength > 0.0:
            damp_mask = compute_damp_mask(
                latent_btchw_fp32=out_fp,
                weight_bt1hw_fp32=wgt,
                gate_mode=cfg.damping_gate_mode,
                grad_blur_sigma=cfg.damping_grad_blur_sigma,
                damp_threshold=cfg.damping_threshold,
                damp_softness=cfg.damping_softness,
                damp_power=cfg.damping_power,
                damp_mask_blur_sigma=cfg.damping_mask_blur_sigma,
            )

            if was_5d and cfg.damping_temporal_ema > 0.0:
                damp_mask = apply_temporal_ema(damp_mask, b=b, t=t, ema=cfg.damping_temporal_ema)

            out_fp = apply_highpass_damping(
                latent_btchw_fp32=out_fp,
                damp_mask_bt1hw_fp32=damp_mask,
                strength=cfg.damping_strength,
                highpass_sigma=cfg.damping_highpass_sigma,
            )

        out = out_fp.to(dtype=orig_dtype)

        if was_5d:
            out = unflatten_4d_to_5d(out, b=b, t=t)

        out_latent = dict(latent)
        out_latent["samples"] = out

        mask_for_output = damp_mask if cfg.preview_mode in ("damp", "both") else wgt
        mask_out = mask_for_output[:, 0, :, :].contiguous()

        ps = max(1, int(cfg.output_mask_pixel_scale))

        def to_preview_image(m_bt1hw: torch.Tensor) -> torch.Tensor:
            m = m_bt1hw
            if ps != 1:
                m = resize_tensor(m, (target_h * ps, target_w * ps), mode="nearest-exact")
            img = m[:, 0:1, :, :].repeat(1, 3, 1, 1).permute(0, 2, 3, 1).contiguous()
            return clamp01(img)

        if cfg.preview_mode == "weight":
            prev_img = to_preview_image(wgt)
        elif cfg.preview_mode == "damp":
            prev_img = to_preview_image(damp_mask)
        else:
            a = to_preview_image(wgt)
            bimg = to_preview_image(damp_mask)
            prev_img = torch.cat([a, bimg], dim=2)

        return (out_latent, mask_out, prev_img)


NODE_CLASS_MAPPINGS = {
    "WAS_AdaptiveDifferenceLatentUpscale": WASAdaptiveDifferenceLatentUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WAS_AdaptiveDifferenceLatentUpscale": "WAS Adaptive Difference Latent Upscale (Damped)",
}
