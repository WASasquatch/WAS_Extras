from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict, Any

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
    scale: float = 2.0

    pre_blur_sigma_px: float = 1.0
    canny_threshold1: int = 25
    canny_threshold2: int = 155
    canny_l2gradient: bool = True

    dilate_radius_px: int = 8
    feather_sigma_px: float = 6.0

    mask_min: float = 0.0
    mask_max: float = 1.0

    nearest_exact: bool = True
    align_corners: Optional[bool] = False

    output_mask_resolution: str = "image"

    video_decode_horizontal_tiles: int = 2
    video_decode_vertical_tiles: int = 2
    video_decode_overlap_latent: int = 4
    video_decode_last_frame_fix: bool = False
    video_decode_enable_cudnn: bool = True


LatentLayout = Literal["4d_bchw", "5d_bcthw", "5d_btchw"]


def is_probable_latent_channels(v: int) -> bool:
    return int(v) in (4, 8, 16)


def normalize_latent_to_bchw(x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Normalize latents to 4D BCHW.

    Accepts:
        4D: [B,C,H,W]
        5D: [B,C,T,H,W]
        5D: [B,T,C,H,W]

    Returns:
        x4: [B',C,H,W] where B' = B (image) or B*T (video)
        meta: dict used to restore original layout
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("latent_samples must be a torch.Tensor")

    if x.dim() == 4:
        b, c, h, w = x.shape
        return x, {"layout": "4d_bchw", "B": int(b), "C": int(c), "H": int(h), "W": int(w)}

    if x.dim() != 5:
        raise ValueError(f"Expected latent 4D or 5D, got {tuple(x.shape)}")

    b = int(x.shape[0])

    if is_probable_latent_channels(int(x.shape[1])):
        c = int(x.shape[1])
        t = int(x.shape[2])
        h = int(x.shape[3])
        w = int(x.shape[4])
        x4 = x.permute(0, 2, 1, 3, 4).contiguous().reshape(b * t, c, h, w)
        return x4, {"layout": "5d_bcthw", "B": b, "C": c, "T": t, "H": h, "W": w}

    if is_probable_latent_channels(int(x.shape[2])):
        t = int(x.shape[1])
        c = int(x.shape[2])
        h = int(x.shape[3])
        w = int(x.shape[4])
        x4 = x.contiguous().reshape(b * t, c, h, w)
        return x4, {"layout": "5d_btchw", "B": b, "C": c, "T": t, "H": h, "W": w}

    c = int(x.shape[1])
    t = int(x.shape[2])
    h = int(x.shape[3])
    w = int(x.shape[4])
    x4 = x.permute(0, 2, 1, 3, 4).contiguous().reshape(b * t, c, h, w)
    return x4, {"layout": "5d_bcthw", "B": b, "C": c, "T": t, "H": h, "W": w}


def restore_latent_from_bchw(x4: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
    """
    Restore latents back to original 4D/5D layout described by meta.
    """
    layout = meta["layout"]

    if layout == "4d_bchw":
        return x4

    if x4.dim() != 4:
        raise ValueError(f"Expected 4D [B',C,H,W], got {tuple(x4.shape)}")

    b = int(meta["B"])
    c = int(meta["C"])
    t = int(meta["T"])
    h = int(x4.shape[-2])
    w = int(x4.shape[-1])

    if int(x4.shape[0]) != b * t:
        raise ValueError(f"Expected batch {b*t}, got {int(x4.shape[0])}")
    if int(x4.shape[1]) != c:
        raise ValueError(f"Expected channels {c}, got {int(x4.shape[1])}")

    if layout == "5d_bcthw":
        return x4.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    if layout == "5d_btchw":
        return x4.reshape(b, t, c, h, w).contiguous()

    raise ValueError(f"Unknown latent layout: {layout}")


def upscale_latent_nearest_exact(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Upscale latents with nearest-exact when available.
    """
    try:
        return F.interpolate(x, size=size, mode="nearest-exact")
    except Exception:
        return F.interpolate(x, size=size, mode="nearest")


def upscale_latent_bilinear(x: torch.Tensor, size: Tuple[int, int], align_corners: Optional[bool]) -> torch.Tensor:
    """
    Upscale latents with bilinear interpolation.
    """
    return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)


def make_elliptical_kernel(radius_px: int):
    if cv2 is None:
        return None
    r = int(radius_px)
    if r <= 0:
        return None
    k = 2 * r + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def clamp01_u8_to_f32(mask_u8: np.ndarray) -> np.ndarray:
    return (mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def latent_mask_to_comfy_mask(mask_b1hw: torch.Tensor) -> torch.Tensor:
    """
    Convert [B,1,H,W] to Comfy MASK [B,H,W].
    """
    if mask_b1hw.dim() != 4 or int(mask_b1hw.shape[1]) != 1:
        raise ValueError(f"Expected [B,1,H,W], got {tuple(mask_b1hw.shape)}")
    return torch.clamp(mask_b1hw[:, 0, :, :], 0.0, 1.0)


def get_vae_scale_factors(vae) -> Tuple[int, int, int]:
    """
    Return (time_scale, width_scale, height_scale) for decode.
    """
    df = getattr(vae, "downscale_index_formula", None)
    if df:
        try:
            t, w, h = df
            return max(1, int(t)), max(1, int(w)), max(1, int(h))
        except Exception:
            pass

    spatial = 1
    temporal = 1

    scd = getattr(vae, "spacial_compression_decode", None)
    if callable(scd):
        try:
            v = scd()
            spatial = 1 if v is None else int(v)
        except Exception:
            spatial = 1

    tcd = getattr(vae, "temporal_compression_decode", None)
    if callable(tcd):
        try:
            v = tcd()
            temporal = 1 if v is None else int(v)
        except Exception:
            temporal = 1

    return max(1, int(temporal)), max(1, int(spatial)), max(1, int(spatial))


def decode_image_latent_regular(vae, latent_bchw: torch.Tensor) -> torch.Tensor:
    """
    Decode 4D image latents.

    Args:
        vae: ComfyUI VAE object
        latent_bchw: [B,C,H,W]

    Returns:
        images_bhwc: [B,H,W,C]
    """
    if latent_bchw.dim() != 4:
        raise ValueError(f"Expected 4D [B,C,H,W], got {tuple(latent_bchw.shape)}")
    images = vae.decode(latent_bchw)
    if not isinstance(images, torch.Tensor):
        raise ValueError("vae.decode did not return a torch.Tensor")
    if images.dim() == 5 and int(images.shape[1]) == 1:
        images = images[:, 0, :, :, :]
    if images.dim() != 4:
        raise ValueError(f"Expected decoded [B,H,W,C], got {tuple(images.shape)}")
    if int(images.shape[-1]) < 3:
        raise ValueError(f"Decoded channels must be >=3, got {int(images.shape[-1])}")
    return images


def decode_video_latent_lazy_tiled(
    vae,
    latent_bcthw: torch.Tensor,
    horizontal_tiles: int,
    vertical_tiles: int,
    overlap_latent: int,
    last_frame_fix: bool,
    enable_cudnn: bool,
) -> torch.Tensor:
    """
    Decode 5D video latents with spatial tiling.

    Args:
        latent_bcthw: [B,C,T,H,W]

    Returns:
        images_btHWC: [B,T_out,H_px,W_px,C]
    """
    if latent_bcthw.dim() != 5:
        raise ValueError(f"Expected 5D [B,C,T,H,W], got {tuple(latent_bcthw.shape)}")

    with torch.backends.cudnn.flags(enabled=bool(enable_cudnn)):
        samples = latent_bcthw
        b, _, t, h, w = samples.shape

        time_sf, w_sf, h_sf = get_vae_scale_factors(vae)

        if last_frame_fix and t > 0:
            last_frame = samples[:, :, -1:, :, :]
            samples = torch.cat([samples, last_frame], dim=2)
            t = int(samples.shape[2])

        t_out = 1 + (t - 1) * int(time_sf)
        out_h = int(h) * int(h_sf)
        out_w = int(w) * int(w_sf)

        horizontal_tiles = max(1, int(horizontal_tiles))
        vertical_tiles = max(1, int(vertical_tiles))
        overlap_latent = max(0, int(overlap_latent))

        base_tile_h = (int(h) + (vertical_tiles - 1) * overlap_latent) // vertical_tiles
        base_tile_w = (int(w) + (horizontal_tiles - 1) * overlap_latent) // horizontal_tiles

        output = None
        weights = None

        for vv in range(vertical_tiles):
            for hh in range(horizontal_tiles):
                w_start = hh * (base_tile_w - overlap_latent)
                h_start = vv * (base_tile_h - overlap_latent)
                w_end = min(w_start + base_tile_w, int(w)) if hh < horizontal_tiles - 1 else int(w)
                h_end = min(h_start + base_tile_h, int(h)) if vv < vertical_tiles - 1 else int(h)

                tile = samples[:, :, :, h_start:h_end, w_start:w_end]
                decoded_tile = vae.decode(tile)
                if not isinstance(decoded_tile, torch.Tensor):
                    raise ValueError("vae.decode did not return a torch.Tensor for video tile")

                if decoded_tile.dim() == 4:
                    decoded_tile = decoded_tile.unsqueeze(1)
                elif decoded_tile.dim() != 5:
                    raise RuntimeError(f"Unexpected decoded tile shape: {tuple(decoded_tile.shape)}")

                if int(decoded_tile.shape[0]) != int(b):
                    raise RuntimeError("Decoded tile batch mismatch")

                c_out = int(decoded_tile.shape[-1])
                if c_out < 3:
                    raise RuntimeError("Decoded tile channels must be >=3")

                if output is None:
                    output = torch.zeros(
                        (b, t_out, out_h, out_w, c_out),
                        device=decoded_tile.device,
                        dtype=decoded_tile.dtype,
                    )
                    weights = torch.zeros(
                        (b, t_out, out_h, out_w, 1),
                        device=decoded_tile.device,
                        dtype=decoded_tile.dtype,
                    )

                out_h_start = int(h_start) * int(h_sf)
                out_h_end = int(h_end) * int(h_sf)
                out_w_start = int(w_start) * int(w_sf)
                out_w_end = int(w_end) * int(w_sf)

                expected_h = out_h_end - out_h_start
                expected_w = out_w_end - out_w_start

                dec_h = int(decoded_tile.shape[2])
                dec_w = int(decoded_tile.shape[3])

                if dec_h != expected_h or dec_w != expected_w:
                    mh = min(dec_h, expected_h)
                    mw = min(dec_w, expected_w)
                    decoded_tile = decoded_tile[:, :, :mh, :mw, :]
                    expected_h = mh
                    expected_w = mw
                    out_h_end = out_h_start + mh
                    out_w_end = out_w_start + mw

                tile_weights = torch.ones(
                    (b, t_out, expected_h, expected_w, 1),
                    device=decoded_tile.device,
                    dtype=decoded_tile.dtype,
                )

                overlap_out_h = min(int(overlap_latent) * int(h_sf), expected_h)
                overlap_out_w = min(int(overlap_latent) * int(w_sf), expected_w)

                if hh > 0 and overlap_out_w > 0:
                    hb = torch.linspace(0, 1, overlap_out_w, device=decoded_tile.device, dtype=decoded_tile.dtype)
                    tile_weights[:, :, :, :overlap_out_w, :] *= hb.view(1, 1, 1, -1, 1)
                if hh < horizontal_tiles - 1 and overlap_out_w > 0:
                    hb = torch.linspace(1, 0, overlap_out_w, device=decoded_tile.device, dtype=decoded_tile.dtype)
                    tile_weights[:, :, :, -overlap_out_w:, :] *= hb.view(1, 1, 1, -1, 1)

                if vv > 0 and overlap_out_h > 0:
                    vb = torch.linspace(0, 1, overlap_out_h, device=decoded_tile.device, dtype=decoded_tile.dtype)
                    tile_weights[:, :, :overlap_out_h, :, :] *= vb.view(1, 1, -1, 1, 1)
                if vv < vertical_tiles - 1 and overlap_out_h > 0:
                    vb = torch.linspace(1, 0, overlap_out_h, device=decoded_tile.device, dtype=decoded_tile.dtype)
                    tile_weights[:, :, -overlap_out_h:, :, :] *= vb.view(1, 1, -1, 1, 1)

                t_dec = int(decoded_tile.shape[1])
                if t_dec == t_out:
                    decoded_for_add = decoded_tile
                elif t_dec == 1:
                    decoded_for_add = decoded_tile.repeat(1, t_out, 1, 1, 1)
                else:
                    if t_out % t_dec == 0:
                        factor = t_out // t_dec
                        decoded_for_add = decoded_tile.repeat(1, factor, 1, 1, 1)
                    else:
                        if t_dec > t_out:
                            decoded_for_add = decoded_tile[:, :t_out, :, :, :]
                        else:
                            reps = (t_out + t_dec - 1) // t_dec
                            decoded_for_add = decoded_tile.repeat(1, reps, 1, 1, 1)[:, :t_out, :, :, :]

                output[:, :, out_h_start:out_h_end, out_w_start:out_w_end, :] += decoded_for_add * tile_weights
                weights[:, :, out_h_start:out_h_end, out_w_start:out_w_end, :] += tile_weights

        output = output / (weights + 1e-8)

        if bool(last_frame_fix) and int(time_sf) > 0:
            output = output[:, :-int(time_sf), :, :, :]

        return output.contiguous()


def decode_for_edge_detection(
    vae,
    latent_bchw_or_flat: torch.Tensor,
    meta: Dict[str, Any],
    cfg: EdgeBlendConfig,
) -> Tuple[torch.Tensor, Tuple[int, int], int]:
    """
    Decode latents for edge detection.

    Returns:
        images_bhwc: [B',H,W,C]
        (himg, wimg): per-frame decoded size
        frames_out: decoded frames per batch (1 for images)
    """
    layout = meta.get("layout", "4d_bchw")

    if layout == "4d_bchw":
        images = decode_image_latent_regular(vae, latent_bchw_or_flat)
        _, himg, wimg, _ = images.shape
        return images, (int(himg), int(wimg)), 1

    b = int(meta["B"])
    c = int(meta["C"])
    t = int(meta["T"])
    h = int(meta["H"])
    w = int(meta["W"])

    if latent_bchw_or_flat.dim() != 4:
        raise ValueError(f"Expected flattened [B*T,C,H,W], got {tuple(latent_bchw_or_flat.shape)}")
    if int(latent_bchw_or_flat.shape[0]) != b * t or int(latent_bchw_or_flat.shape[1]) != c:
        raise ValueError(f"Video flatten mismatch: expected {(b*t, c, h, w)}, got {tuple(latent_bchw_or_flat.shape)}")

    latent_bcthw = latent_bchw_or_flat.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    images_bt = decode_video_latent_lazy_tiled(
        vae=vae,
        latent_bcthw=latent_bcthw,
        horizontal_tiles=cfg.video_decode_horizontal_tiles,
        vertical_tiles=cfg.video_decode_vertical_tiles,
        overlap_latent=cfg.video_decode_overlap_latent,
        last_frame_fix=cfg.video_decode_last_frame_fix,
        enable_cudnn=cfg.video_decode_enable_cudnn,
    )

    b2, t_out, himg, wimg, ch = images_bt.shape
    if int(b2) != b:
        raise ValueError(f"Decoded batch mismatch: got {int(b2)} expected {b}")
    if int(ch) < 3:
        raise ValueError("Decoded channels must be >=3")

    images_bhwc = images_bt.reshape(b * t_out, int(himg), int(wimg), int(ch)).contiguous()
    return images_bhwc, (int(himg), int(wimg)), int(t_out)


def build_edge_masks_opencv(
    vae,
    latent_samples_bchw_or_flat: torch.Tensor,
    meta: Dict[str, Any],
    target_latent_size: Tuple[int, int],
    cfg: EdgeBlendConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge masks from decoded pixels.

    Returns:
        mask_img: [B',1,target_himg,target_wimg]
        mask_lat: [B',1,Ht,Wt]
    """
    if cv2 is None:
        raise RuntimeError(f"OpenCV (cv2) is required but could not be imported: {_CV2_IMPORT_ERROR}")

    images, (himg, wimg), _ = decode_for_edge_detection(
        vae=vae,
        latent_bchw_or_flat=latent_samples_bchw_or_flat,
        meta=meta,
        cfg=cfg,
    )

    if images.dim() != 4 or int(images.shape[-1]) < 3:
        raise ValueError(f"Decoded images must be [B',H,W,C>=3], got {tuple(images.shape)}")

    b_prime = int(images.shape[0])

    if meta["layout"] == "4d_bchw":
        _, _, h_lat, w_lat = latent_samples_bchw_or_flat.shape
    else:
        h_lat = int(meta["H"])
        w_lat = int(meta["W"])

    scale_y = float(himg) / float(h_lat)
    scale_x = float(wimg) / float(w_lat)

    ht, wt = target_latent_size
    target_himg = max(1, int(round(ht * scale_y)))
    target_wimg = max(1, int(round(wt * scale_x)))

    dilate_kernel = make_elliptical_kernel(cfg.dilate_radius_px)

    masks_img = []
    masks_lat = []

    for i in range(b_prime):
        img = images[i].detach().cpu().numpy()
        img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

        if img_u8.shape[-1] >= 4:
            img_u8 = img_u8[:, :, :3]

        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)

        pre_sigma = float(cfg.pre_blur_sigma_px)
        if pre_sigma > 0.0:
            gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=pre_sigma, sigmaY=pre_sigma, borderType=cv2.BORDER_REPLICATE)

        edges = cv2.Canny(
            gray,
            int(cfg.canny_threshold1),
            int(cfg.canny_threshold2),
            L2gradient=bool(cfg.canny_l2gradient),
        )

        if dilate_kernel is not None and int(cfg.dilate_radius_px) > 0:
            edges = cv2.dilate(edges, dilate_kernel, iterations=1)

        feather_sigma = float(cfg.feather_sigma_px)
        if feather_sigma > 0.0:
            edges = cv2.GaussianBlur(edges, (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma, borderType=cv2.BORDER_REPLICATE)

        mask_f = clamp01_u8_to_f32(edges)

        if (target_himg != himg) or (target_wimg != wimg):
            mask_img_f = cv2.resize(mask_f, (target_wimg, target_himg), interpolation=cv2.INTER_LINEAR)
        else:
            mask_img_f = mask_f

        if (target_himg != ht) or (target_wimg != wt):
            mask_lat_f = cv2.resize(mask_img_f, (wt, ht), interpolation=cv2.INTER_AREA)
        else:
            mask_lat_f = mask_img_f

        mask_img_f = np.clip(mask_img_f, float(cfg.mask_min), float(cfg.mask_max))
        mask_lat_f = np.clip(mask_lat_f, float(cfg.mask_min), float(cfg.mask_max))

        masks_img.append(mask_img_f)
        masks_lat.append(mask_lat_f)

    mask_img_np = np.stack(masks_img, axis=0).astype(np.float32)
    mask_lat_np = np.stack(masks_lat, axis=0).astype(np.float32)

    mask_img = torch.from_numpy(mask_img_np).unsqueeze(1).to(device=latent_samples_bchw_or_flat.device, dtype=torch.float32)
    mask_lat = torch.from_numpy(mask_lat_np).unsqueeze(1).to(device=latent_samples_bchw_or_flat.device, dtype=torch.float32)

    return mask_img, mask_lat


def run_hybrid_upscale_with_opencv_mask(
    latent_samples: torch.Tensor,
    cfg: EdgeBlendConfig,
    vae,
    donor_latent: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hybrid upscale with edge-based mask blending.

    Returns:
        up_latent: upscaled latent in original layout
        mask_img: [B',1,Himg,Wimg]
        mask_lat: [B',1,Ht,Wt]
    """
    latent_bchw, meta = normalize_latent_to_bchw(latent_samples)

    donor_bchw = None
    if donor_latent is not None:
        donor_bchw, donor_meta = normalize_latent_to_bchw(donor_latent)
        for k in ("layout", "B", "C"):
            if donor_meta.get(k) != meta.get(k):
                raise ValueError(f"donor_latent mismatch on {k}: latent={meta.get(k)} donor={donor_meta.get(k)}")
        if meta["layout"] != "4d_bchw" and int(donor_meta.get("T", -1)) != int(meta.get("T", -1)):
            raise ValueError(f"donor_latent mismatch on T: latent={meta.get('T')} donor={donor_meta.get('T')}")

    _, _, h, w = latent_bchw.shape
    ht = int(round(h * float(cfg.scale)))
    wt = int(round(w * float(cfg.scale)))
    if ht <= 0 or wt <= 0:
        raise ValueError("Invalid target size computed from scale.")
    target_latent_size = (ht, wt)

    if cfg.nearest_exact:
        base = upscale_latent_nearest_exact(latent_bchw, target_latent_size)
    else:
        base = F.interpolate(latent_bchw, size=target_latent_size, mode="nearest")

    if donor_bchw is not None:
        donor = F.interpolate(donor_bchw, size=target_latent_size, mode="bilinear", align_corners=cfg.align_corners)
    else:
        donor = upscale_latent_bilinear(latent_bchw, target_latent_size, cfg.align_corners)

    mask_img, mask_lat = build_edge_masks_opencv(
        vae=vae,
        latent_samples_bchw_or_flat=latent_bchw,
        meta=meta,
        target_latent_size=target_latent_size,
        cfg=cfg,
    )

    if meta["layout"] != "4d_bchw":
        b = int(meta["B"])
        t = int(meta["T"])
        b_flat = int(latent_bchw.shape[0])
        b_prime = int(mask_lat.shape[0])

        if b_prime != b_flat:
            time_sf, _, _ = get_vae_scale_factors(vae)
            time_sf = max(1, int(time_sf))

            t_out = b_prime // b
            m = mask_lat.reshape(b, t_out, 1, ht, wt)

            idx = torch.arange(0, t * time_sf, step=time_sf, device=m.device)
            idx = torch.clamp(idx, 0, t_out - 1)
            m_sel = torch.index_select(m, dim=1, index=idx)

            mask_lat = m_sel.reshape(b_flat, 1, ht, wt)

    mask_lat = mask_lat.to(dtype=base.dtype)
    out_bchw = base * (1.0 - mask_lat) + donor * mask_lat

    out_latent = restore_latent_from_bchw(out_bchw, meta)
    return out_latent, mask_img, mask_lat


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

                "video_decode_horizontal_tiles": ("INT", {"default": 2, "min": 1, "max": 8}),
                "video_decode_vertical_tiles": ("INT", {"default": 2, "min": 1, "max": 8}),
                "video_decode_overlap_latent": ("INT", {"default": 4, "min": 0, "max": 32}),
                "video_decode_last_frame_fix": ("BOOLEAN", {"default": False}),
                "video_decode_enable_cudnn": ("BOOLEAN", {"default": True}),
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
        video_decode_horizontal_tiles: int,
        video_decode_vertical_tiles: int,
        video_decode_overlap_latent: int,
        video_decode_last_frame_fix: bool,
        video_decode_enable_cudnn: bool,
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
            video_decode_horizontal_tiles=int(video_decode_horizontal_tiles),
            video_decode_vertical_tiles=int(video_decode_vertical_tiles),
            video_decode_overlap_latent=int(video_decode_overlap_latent),
            video_decode_last_frame_fix=bool(video_decode_last_frame_fix),
            video_decode_enable_cudnn=bool(video_decode_enable_cudnn),
        )

        up_latent, mask_img, mask_lat = run_hybrid_upscale_with_opencv_mask(
            latent_samples=latent_samples,
            cfg=cfg,
            vae=vae,
            donor_latent=donor_samples,
        )

        out = dict(latent)
        out["samples"] = up_latent

        if cfg.output_mask_resolution == "latent":
            edge_mask = latent_mask_to_comfy_mask(mask_lat)
        else:
            edge_mask = latent_mask_to_comfy_mask(mask_img)

        return (out, edge_mask)


NODE_CLASS_MAPPINGS = {
    "WASLatentUpscaleHybrid": WASLatentUpscaleHybrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WASLatentUpscaleHybrid": "Latent Hybrid Upscale",
}
