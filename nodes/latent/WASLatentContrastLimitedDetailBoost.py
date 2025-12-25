from __future__ import annotations

import math
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


def sobel_grad_mag(x_bt1hw: torch.Tensor) -> torch.Tensor:
    device = x_bt1hw.device
    dtype = x_bt1hw.dtype

    kx = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        device=device, dtype=dtype
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 2.0, 1.0]],
        device=device, dtype=dtype
    ).view(1, 1, 3, 3)

    gx = F.conv2d(x_bt1hw, kx, padding=1)
    gy = F.conv2d(x_bt1hw, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def resize_mask_for_preview(mask_bt1hw: torch.Tensor, pixel_scale: int) -> torch.Tensor:
    ps = max(1, int(pixel_scale))
    if ps == 1:
        m = mask_bt1hw
    else:
        h, w = mask_bt1hw.shape[-2], mask_bt1hw.shape[-1]
        m = F.interpolate(mask_bt1hw, size=(h * ps, w * ps), mode="nearest-exact")
    img = m[:, 0:1].repeat(1, 3, 1, 1).permute(0, 2, 3, 1).contiguous()
    return clamp01(img)


class WASLatentContrastLimitedDetailBoost:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),

                # Band-pass (DoG) parameters
                "sigma_small": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 8.0, "step": 0.05}),
                "sigma_large": ("FLOAT", {"default": 1.4, "min": 0.0, "max": 16.0, "step": 0.05}),

                # Strength and limiting
                "gain": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01}),
                "limit": ("FLOAT", {"default": 1.25, "min": 0.1, "max": 8.0, "step": 0.05}),

                # Local energy normalization (prevents halos)
                "rms_sigma": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 16.0, "step": 0.05}),
                "rms_floor": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 1.0, "step": 0.005}),

                # Optional edge protection (reduces dark outlines at strong boundaries)
                "edge_protect": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_sigma": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 8.0, "step": 0.05}),
                "edge_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_softness": ("FLOAT", {"default": 0.10, "min": 0.0005, "max": 1.0, "step": 0.01}),

                # Preview
                "preview_mask_scale": ("INT", {"default": 8, "min": 1, "max": 16, "step": 1}),
                "preview_mode": (["edge_mask", "detail_mask"], {"default": "detail_mask"}),
            }
        }

    RETURN_TYPES = ("LATENT", "MASK", "IMAGE")
    RETURN_NAMES = ("latent", "mask", "mask_preview")
    FUNCTION = "boost"
    CATEGORY = "WAS/Latent"

    def boost(
        self,
        latent,
        sigma_small: float,
        sigma_large: float,
        gain: float,
        limit: float,
        rms_sigma: float,
        rms_floor: float,
        edge_protect: float,
        edge_sigma: float,
        edge_threshold: float,
        edge_softness: float,
        preview_mask_scale: int,
        preview_mode: str,
    ):
        if "samples" not in latent:
            raise ValueError("LATENT input must be a dict containing key 'samples'.")

        samples: torch.Tensor = latent["samples"]
        orig_dtype = samples.dtype

        was_5d = is_latent_5d(samples)
        if was_5d:
            x, b, t = flatten_5d_to_4d(samples)
        else:
            x = samples
            b, t = x.shape[0], 1

        x_fp = x.float()

        s_small = float(sigma_small)
        s_large = float(sigma_large)
        if s_large < s_small:
            s_small, s_large = s_large, s_small

        # DoG band-pass: emphasizes microdetail without classic unsharp overshoot tendencies
        low_small = gaussian_blur_depthwise(x_fp, s_small) if s_small > 0.0 else x_fp
        low_large = gaussian_blur_depthwise(x_fp, s_large) if s_large > 0.0 else x_fp
        dog = low_small - low_large  # band-pass

        # Local RMS normalization (contrast-limited): prevents dark halos / emboss
        if float(rms_sigma) > 0.0:
            rms = gaussian_blur_depthwise(dog * dog, float(rms_sigma))
            rms = torch.sqrt(torch.clamp(rms, min=0.0) + 1e-8)
        else:
            rms = torch.sqrt(torch.mean(dog * dog, dim=(2, 3), keepdim=True) + 1e-8)

        rms = rms + float(rms_floor)
        dog_n = dog / rms

        # Soft limiter on normalized detail (prevents ringing)
        lim = float(limit)
        dog_l = torch.tanh(dog_n * lim) / max(1e-6, lim)

        # Detail magnitude mask (for inspection / optional gating)
        detail_mag = dog_l.abs().mean(dim=1, keepdim=True)
        detail_mag = detail_mag / (detail_mag.amax(dim=(2, 3), keepdim=True) + 1e-8)
        detail_mag = clamp01(detail_mag)

        # Edge protection mask (reduce enhancement at strong boundaries)
        if float(edge_protect) > 0.0:
            energy = x_fp.abs().mean(dim=1, keepdim=True)
            if float(edge_sigma) > 0.0:
                energy = gaussian_blur_depthwise(energy, float(edge_sigma))
            gmag = sobel_grad_mag(energy)
            gmag = gmag / (gmag.amax(dim=(2, 3), keepdim=True) + 1e-8)

            t0 = float(edge_threshold)
            s0 = max(1e-6, float(edge_softness))
            edge = torch.sigmoid((gmag - t0) / s0)  # 0..1 strong edges -> 1
            edge = clamp01(edge)

            protect = float(edge_protect)
            edge_gate = 1.0 - protect * edge
            edge_gate = torch.clamp(edge_gate, 0.0, 1.0)
        else:
            edge = torch.zeros_like(detail_mag)
            edge_gate = 1.0

        # Apply enhancement
        out = x_fp + float(gain) * dog_l * edge_gate
        out = out.to(dtype=orig_dtype)

        if was_5d:
            out = unflatten_4d_to_5d(out, b=b, t=t)

        out_latent = dict(latent)
        out_latent["samples"] = out

        # ComfyUI MASK output: [BT,H,W]
        if preview_mode == "edge_mask":
            mask_bt = edge[:, 0]
            prev_img = resize_mask_for_preview(edge, preview_mask_scale)
        else:
            mask_bt = detail_mag[:, 0]
            prev_img = resize_mask_for_preview(detail_mag, preview_mask_scale)

        return (out_latent, mask_bt.contiguous(), prev_img)


NODE_CLASS_MAPPINGS = {
    "WASLatentContrastLimitedDetailBoost": WASLatentContrastLimitedDetailBoost,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WASLatentContrastLimitedDetailBoost": "WAS Latent Detail Boost",
}
