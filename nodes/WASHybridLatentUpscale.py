from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

import numpy as np

try:
    import cv2
except Exception as e:
    cv2 = None
    _CV2_IMPORT_ERROR = e


@dataclass
class EdgeBlendConfig:
    # upscale factor
    scale: float = 2.0

    # edge detection
    pre_blur_sigma_px: float = 1.0
    canny_threshold1: int = 100
    canny_threshold2: int = 200
    canny_l2gradient: bool = True

    # mask shaping
    dilate_radius_px: int = 2
    feather_sigma_px: float = 1.0

    # mask clamp
    mask_min: float = 0.0
    mask_max: float = 1.0

    # upscale
    nearest_exact: bool = True
    align_corners: Optional[bool] = False

    # mask size
    output_mask_resolution: str = "image"


def upscaleLatentNearestExact(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    try:
        return F.interpolate(x, size=size, mode="nearest-exact")
    except Exception:
        return F.interpolate(x, size=size, mode="nearest")


def upscaleLatentBilinear(x: torch.Tensor, size: Tuple[int, int], align_corners: Optional[bool]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)


def decodeLatentToImageBHWCViaVAE(vae, latent_samples: torch.Tensor) -> torch.Tensor:
    images = vae.decode(latent_samples)
    if images.dim() == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def _makeEllipticalKernel(radius_px: int):
    r = int(radius_px)
    if r <= 0:
        return None
    k = 2 * r + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _clamp01_u8_to_f32(mask_u8: np.ndarray) -> np.ndarray:
    return (mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _decodeAndComputeTargetPixelSize(
    vae,
    latent_samples: torch.Tensor,
    target_latent_size: Tuple[int, int],
):

    images = decodeLatentToImageBHWCViaVAE(vae, latent_samples)
    if images.dim() != 4 or images.shape[-1] < 3:
        raise ValueError(f"VAE decode must return [B,H,W,C>=3], got {tuple(images.shape)}")

    b, himg, wimg, c = images.shape

    _, _, h_lat, w_lat = latent_samples.shape
    if h_lat <= 0 or w_lat <= 0:
        raise ValueError("Invalid latent shape.")

    # pixel-per-latent scaling
    scale_y = float(himg) / float(h_lat)
    scale_x = float(wimg) / float(w_lat)

    ht, wt = target_latent_size
    target_himg = int(round(ht * scale_y))
    target_wimg = int(round(wt * scale_x))

    target_himg = max(1, target_himg)
    target_wimg = max(1, target_wimg)

    return images, (himg, wimg), (target_himg, target_wimg)


def buildEdgeMasksOpenCV(
    vae,
    latent_samples: torch.Tensor,
    target_latent_size: Tuple[int, int],
    cfg: EdgeBlendConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if cv2 is None:
        raise RuntimeError(f"OpenCV (cv2) is required but could not be imported: {_CV2_IMPORT_ERROR}")

    images, (himg, wimg), (target_himg, target_wimg) = _decodeAndComputeTargetPixelSize(
        vae=vae,
        latent_samples=latent_samples,
        target_latent_size=target_latent_size,
    )

    b, _, _, c = images.shape

    dilate_kernel = _makeEllipticalKernel(cfg.dilate_radius_px)

    masks_img = []
    masks_lat = []

    ht, wt = target_latent_size

    for i in range(b):
        img = images[i].detach().cpu().numpy()
        img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

        if img_u8.shape[-1] >= 4:
            img_u8 = img_u8[:, :, :3]

        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)

        pre_sigma = float(cfg.pre_blur_sigma_px)
        if pre_sigma > 0.0:
            gray = cv2.GaussianBlur(
                gray,
                (0, 0),
                sigmaX=pre_sigma,
                sigmaY=pre_sigma,
                borderType=cv2.BORDER_REPLICATE,
            )

        edges = cv2.Canny(
            gray,
            int(cfg.canny_threshold1),
            int(cfg.canny_threshold2),
            L2gradient=bool(cfg.canny_l2gradient),
        )  # uint8 0/255

        if dilate_kernel is not None and int(cfg.dilate_radius_px) > 0:
            edges = cv2.dilate(edges, dilate_kernel, iterations=1)

        feather_sigma = float(cfg.feather_sigma_px)
        if feather_sigma > 0.0:
            edges = cv2.GaussianBlur(
                edges,
                (0, 0),
                sigmaX=feather_sigma,
                sigmaY=feather_sigma,
                borderType=cv2.BORDER_REPLICATE,
            )

        # float32 0..1
        mask_f = _clamp01_u8_to_f32(edges)

        # Resize to target decoded pixel size
        if (target_himg != himg) or (target_wimg != wimg):
            mask_img_f = cv2.resize(mask_f, (target_wimg, target_himg), interpolation=cv2.INTER_LINEAR)
        else:
            mask_img_f = mask_f

        # Downsample to target latent size for blending
        if (target_himg != ht) or (target_wimg != wt):
            mask_lat_f = cv2.resize(mask_img_f, (wt, ht), interpolation=cv2.INTER_AREA)
        else:
            mask_lat_f = mask_img_f

        mask_img_f = np.clip(mask_img_f, float(cfg.mask_min), float(cfg.mask_max))
        mask_lat_f = np.clip(mask_lat_f, float(cfg.mask_min), float(cfg.mask_max))

        masks_img.append(mask_img_f)
        masks_lat.append(mask_lat_f)

    mask_img_np = np.stack(masks_img, axis=0).astype(np.float32)  # [B,target_himg,target_wimg]
    mask_lat_np = np.stack(masks_lat, axis=0).astype(np.float32)  # [B,Ht,Wt]

    mask_img = torch.from_numpy(mask_img_np).unsqueeze(1).to(device=latent_samples.device, dtype=torch.float32)
    mask_lat = torch.from_numpy(mask_lat_np).unsqueeze(1).to(device=latent_samples.device, dtype=torch.float32)

    return mask_img, mask_lat


def latentMaskToComfyMask(mask_b1hw: torch.Tensor) -> torch.Tensor:
    if mask_b1hw.dim() != 4 or mask_b1hw.shape[1] != 1:
        raise ValueError(f"Expected [B,1,H,W], got {tuple(mask_b1hw.shape)}")
    return torch.clamp(mask_b1hw[:, 0, :, :], 0.0, 1.0)


def runHybridUpscaleWithOpenCVMask(
    latent_samples: torch.Tensor,
    cfg: EdgeBlendConfig,
    vae,
    donor_latent: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if latent_samples.dim() != 4:
        raise ValueError(f"Expected latent [B,C,H,W], got {tuple(latent_samples.shape)}")

    _, _, h, w = latent_samples.shape
    ht = int(round(h * float(cfg.scale)))
    wt = int(round(w * float(cfg.scale)))
    if ht <= 0 or wt <= 0:
        raise ValueError("Invalid target size computed from scale.")
    target_latent_size = (ht, wt)

    # Base
    if cfg.nearest_exact:
        base = upscaleLatentNearestExact(latent_samples, target_latent_size)
    else:
        base = F.interpolate(latent_samples, size=target_latent_size, mode="nearest")
    
    # Optional donor
    donor = None
    if donor_latent is not None:
        if donor_latent.dim() != 4:
            raise ValueError("donor_latent must be [B,C,H,W]")
        donor = F.interpolate(donor_latent, size=target_latent_size, mode="bilinear", align_corners=cfg.align_corners)
    else:
        donor = upscaleLatentBilinear(latent_samples, target_latent_size, cfg.align_corners)

    # Masks
    mask_img, mask_lat = buildEdgeMasksOpenCV(
        vae=vae,
        latent_samples=latent_samples,
        target_latent_size=target_latent_size,
        cfg=cfg,
    )

    mask_lat = mask_lat.to(dtype=base.dtype)
    out = base * (1.0 - mask_lat) + donor * mask_lat
    return out, mask_img, mask_lat


class WASLatentUpscaleHybrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE",),

                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.01}),

                "pre_blur_sigma_px": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01}),
                "canny_threshold1": ("INT", {"default": 25, "min": 0, "max": 1000, "step": 1}),
                "canny_threshold2": ("INT", {"default": 155, "min": 0, "max": 1000, "step": 1}),
                "canny_l2gradient": ("BOOLEAN", {"default": True}),

                "dilate_radius_px": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "feather_sigma_px": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 50.0, "step": 0.01}),

                "mask_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                "use_nearest_exact": ("BOOLEAN", {"default": True}),
                "output_mask_resolution": (["image", "latent"], {"default": "image"}),
            },
            "optional": {
                "donor_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT", "MASK")
    RETURN_NAMES = ("latent", "edge_mask")
    FUNCTION = "execute"
    CATEGORY = "latent/upscale"

    def execute(
        self,
        latent,
        vae,
        scale: float,
        pre_blur_sigma_px: float,
        canny_threshold1: int,
        canny_threshold2: int,
        canny_l2gradient: bool,
        dilate_radius_px: int,
        feather_sigma_px: float,
        mask_min: float,
        mask_max: float,
        use_nearest_exact: bool,
        output_mask_resolution: str,
        donor_latent=None,
    ):
        latent_samples = latent["samples"]
        donor_samples = donor_latent["samples"] if donor_latent is not None else None

        cfg = EdgeBlendConfig(
            scale=float(scale),
            pre_blur_sigma_px=float(pre_blur_sigma_px),
            canny_threshold1=int(canny_threshold1),
            canny_threshold2=int(canny_threshold2),
            canny_l2gradient=bool(canny_l2gradient),
            dilate_radius_px=int(dilate_radius_px),
            feather_sigma_px=float(feather_sigma_px),
            mask_min=float(mask_min),
            mask_max=float(mask_max),
            nearest_exact=bool(use_nearest_exact),
            output_mask_resolution=str(output_mask_resolution).strip().lower(),
        )

        up_latent, mask_img, mask_lat = runHybridUpscaleWithOpenCVMask(
            latent_samples=latent_samples,
            cfg=cfg,
            vae=vae,
            donor_latent=donor_samples,
        )

        out = dict(latent)
        out["samples"] = up_latent

        if cfg.output_mask_resolution == "latent":
            edge_mask = latentMaskToComfyMask(mask_lat)
        else:
            edge_mask = latentMaskToComfyMask(mask_img)

        return (out, edge_mask)


NODE_CLASS_MAPPINGS = {
    "WASLatentUpscaleHybrid": WASLatentUpscaleHybrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WASLatentUpscaleHybrid": "Latent Hybrid Upscale",
}
