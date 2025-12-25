from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ExposureStats:
    log_mean: torch.Tensor
    log_std: torch.Tensor


def compute_luma(rgb: torch.Tensor) -> torch.Tensor:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    return (0.2126 * r) + (0.7152 * g) + (0.0722 * b)


def downscale_for_stats(images_bhwc: torch.Tensor, proxy_size: int) -> torch.Tensor:
    b, h, w, c = images_bhwc.shape
    if proxy_size <= 0:
        return images_bhwc
    if h == proxy_size and w == proxy_size:
        return images_bhwc
    x = images_bhwc.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(proxy_size, proxy_size), mode="area")
    return x.permute(0, 2, 3, 1)


def compute_exposure_stats(
    images_bhwc: torch.Tensor,
    eps: float,
    clip_low: float,
    clip_high: float,
) -> ExposureStats:
    rgb = images_bhwc[..., :3].clamp(0.0, 1.0)
    luma = compute_luma(rgb).clamp(0.0, 1.0)

    if clip_low > 0.0 or clip_high < 1.0:
        luma = luma.clamp(clip_low, clip_high)

    log_luma = torch.log(luma + eps)
    log_mean = log_luma.mean(dim=(1, 2))
    log_std = log_luma.std(dim=(1, 2), unbiased=False)
    return ExposureStats(log_mean=log_mean, log_std=log_std)


def smooth_1d(x: torch.Tensor, window: int) -> torch.Tensor:
    window = int(window)
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    v = x.view(1, 1, -1)
    v = F.pad(v, (pad, pad), mode="replicate")
    kernel = torch.ones((1, 1, window), device=x.device, dtype=x.dtype) / float(window)
    y = F.conv1d(v, kernel)
    return y.view(-1)


def find_settle_index(
    log_mean: torch.Tensor,
    ref_log_mean: float,
    tolerance_log: float,
    stable_count: int,
) -> int:
    b = int(log_mean.numel())
    if b == 0:
        return 0
    stable_count = max(int(stable_count), 1)

    diff = (log_mean - float(ref_log_mean)).abs()
    within = diff <= float(tolerance_log)

    run = 0
    for i in range(b):
        if bool(within[i].item()):
            run += 1
            if run >= stable_count:
                return i - stable_count + 1
        else:
            run = 0
    return b


def apply_exposure_correction(images_bhwc: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    b, h, w, c = images_bhwc.shape
    g = gains.view(b, 1, 1, 1).to(dtype=images_bhwc.dtype, device=images_bhwc.device)
    rgb = (images_bhwc[..., :3] * g).clamp(0.0, 1.0)
    if c > 3:
        rest = images_bhwc[..., 3:]
        return torch.cat([rgb, rest], dim=-1)
    return rgb


def ev_to_log(ev: float) -> float:
    return float(ev) * float(torch.log(torch.tensor(2.0)).item())


def log_to_ev(logv: torch.Tensor) -> torch.Tensor:
    return logv / float(torch.log(torch.tensor(2.0)).item())


def build_anchor_range(
    b: int,
    anchor_mode: str,
    ref_tail_frames: int,
    anchor_center: float,
    anchor_window: int,
) -> Tuple[int, int]:
    b = int(b)
    anchor_mode = str(anchor_mode).strip().lower()

    if anchor_mode == "tail":
        n = max(int(ref_tail_frames), 1)
        n = min(n, b)
        return (b - n, b)

    w = max(int(anchor_window), 1)
    w = min(w, b)
    center = float(anchor_center)
    if center < 0.0:
        center = 0.0
    if center > 1.0:
        center = 1.0

    cidx = int(round(center * (b - 1)))
    start = cidx - (w // 2)
    end = start + w
    if start < 0:
        start = 0
        end = w
    if end > b:
        end = b
        start = b - w
        if start < 0:
            start = 0
    return (start, end)


class WASWanExposureStabilizer:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),

                "anchor_mode": (
                    ["middle", "tail"],
                    {
                        "default": "middle",
                        "tooltip": (
                            "How the exposure reference is chosen.\n"
                            "middle: use a window around anchor_center as the reference (recommended for WAN drift at BOTH start/end).\n"
                            "tail: use the last ref_tail_frames as the reference (useful when the end is known-stable)."
                        ),
                    },
                ),
                "ref_tail_frames": (
                    "INT",
                    {
                        "default": 12,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": (
                            "Only used when anchor_mode=tail.\n"
                            "Number of final frames sampled to compute the reference exposure (median log-luma).\n"
                            "Increase if the tail is stable but noisy; decrease if the tail contains fades/changes."
                        ),
                    },
                ),
                "anchor_center": (
                    "FLOAT",
                    {
                        "default": 0.55,
                        "min": 0.00,
                        "max": 1.00,
                        "step": 0.01,
                        "tooltip": (
                            "Only used when anchor_mode=middle.\n"
                            "Normalized position (0..1) for the center of the anchor window.\n"
                            "0.50 anchors the middle; 0.55 biases slightly later if the beginning is more unstable."
                        ),
                    },
                ),
                "anchor_window": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": (
                            "Only used when anchor_mode=middle.\n"
                            "Number of frames in the anchor window used to compute the reference exposure.\n"
                            "Larger is more robust, but avoid spanning major scene changes."
                        ),
                    },
                ),

                "correct_ends": (
                    ["start_and_end", "start_only"],
                    {
                        "default": "start_and_end",
                        "tooltip": (
                            "Which regions to correct.\n"
                            "start_only: correct only the initial transient until it settles.\n"
                            "start_and_end: also correct tail drift if the last stable_count frames are outside tolerance_ev."
                        ),
                    },
                ),
                "tolerance_ev": (
                    "FLOAT",
                    {
                        "default": 0.10,
                        "min": 0.00,
                        "max": 2.00,
                        "step": 0.01,
                        "tooltip": (
                            "Stability tolerance in exposure stops (EV).\n"
                            "Lower = stricter (detects drift longer, may correct more frames).\n"
                            "Higher = more forgiving (corrects fewer frames, less risk of reacting to content changes)."
                        ),
                    },
                ),
                "stable_count": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "How many consecutive frames must be within tolerance_ev to be considered stable.\n"
                            "Higher values reduce false-stability on noisy sequences but may delay settle detection."
                        ),
                    },
                ),
                "max_correct_frames": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                        "tooltip": (
                            "Hard cap on how many frames can be corrected at the start and (if enabled) at the end.\n"
                            "0 disables correction entirely (stats/report only)."
                        ),
                    },
                ),

                "proxy_size": (
                    "INT",
                    {
                        "default": 96,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                        "tooltip": (
                            "Downscale size used to compute luminance statistics.\n"
                            "0 uses full resolution (slower). 64–128 is usually sufficient and much faster."
                        ),
                    },
                ),
                "gain_min": (
                    "FLOAT",
                    {
                        "default": 0.70,
                        "min": 0.05,
                        "max": 2.00,
                        "step": 0.01,
                        "tooltip": (
                            "Minimum allowed exposure gain applied to any frame.\n"
                            "Lower values allow stronger darkening correction but can crush highlights if too low."
                        ),
                    },
                ),
                "gain_max": (
                    "FLOAT",
                    {
                        "default": 1.30,
                        "min": 0.05,
                        "max": 4.00,
                        "step": 0.01,
                        "tooltip": (
                            "Maximum allowed exposure gain applied to any frame.\n"
                            "Higher values allow stronger brightening correction but can clip highlights if too high."
                        ),
                    },
                ),
                "gain_smooth_window": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 51,
                        "step": 2,
                        "tooltip": (
                            "Temporal smoothing window for the per-frame gain curve.\n"
                            "Use odd values. Larger values reduce pumping but can lag real drift.\n"
                            "The anchor window is forced to gain=1.0 after smoothing."
                        ),
                    },
                ),

                "clip_low": (
                    "FLOAT",
                    {
                        "default": 0.00,
                        "min": 0.00,
                        "max": 0.50,
                        "step": 0.01,
                        "tooltip": (
                            "Luminance clamp floor used ONLY for computing stats (not applied to output pixels).\n"
                            "Raise slightly (e.g., 0.02–0.05) to reduce influence of deep blacks/noise on exposure estimation."
                        ),
                    },
                ),
                "clip_high": (
                    "FLOAT",
                    {
                        "default": 1.00,
                        "min": 0.50,
                        "max": 1.00,
                        "step": 0.01,
                        "tooltip": (
                            "Luminance clamp ceiling used ONLY for computing stats (not applied to output pixels).\n"
                            "Lower slightly (e.g., 0.98–0.995) to reduce influence of specular peaks on exposure estimation."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "stabilize"
    CATEGORY = "WAS/Video"

    def stabilize(
        self,
        images: torch.Tensor,
        anchor_mode: str = "middle",
        ref_tail_frames: int = 12,
        anchor_center: float = 0.55,
        anchor_window: int = 16,
        correct_ends: str = "start_and_end",
        tolerance_ev: float = 0.10,
        stable_count: int = 4,
        max_correct_frames: int = 96,
        proxy_size: int = 96,
        gain_min: float = 0.70,
        gain_max: float = 1.30,
        gain_smooth_window: int = 5,
        clip_low: float = 0.00,
        clip_high: float = 1.00,
    ) -> Tuple[torch.Tensor, str]:
        if images is None or images.ndim != 4 or images.shape[-1] < 3:
            return images, "WASWanExposureStabilizer: invalid IMAGE input"

        b = int(images.shape[0])
        if b <= 1:
            return images, "WASWanExposureStabilizer: batch too small (no temporal stabilization needed)"

        device = images.device
        images_f32 = images.to(torch.float32)

        proxy = downscale_for_stats(images_f32, int(proxy_size))
        stats = compute_exposure_stats(proxy, eps=1e-6, clip_low=float(clip_low), clip_high=float(clip_high))

        a0, a1 = build_anchor_range(
            b=b,
            anchor_mode=str(anchor_mode),
            ref_tail_frames=int(ref_tail_frames),
            anchor_center=float(anchor_center),
            anchor_window=int(anchor_window),
        )
        anchor_slice = stats.log_mean[a0:a1]
        ref_log_mean = float(anchor_slice.median().item())

        tolerance_log = ev_to_log(float(tolerance_ev))

        settle_index = find_settle_index(
            log_mean=stats.log_mean,
            ref_log_mean=ref_log_mean,
            tolerance_log=tolerance_log,
            stable_count=int(stable_count),
        )

        tail_diff = (stats.log_mean[-int(stable_count):] - ref_log_mean).abs()
        tail_within = bool((tail_diff <= tolerance_log).all().item())

        gains = torch.ones((b,), device=device, dtype=torch.float32)
        needed = torch.exp(torch.tensor(ref_log_mean, device=device) - stats.log_mean.to(device))
        needed = needed.clamp(float(gain_min), float(gain_max))

        if int(max_correct_frames) <= 0:
            max_correct_frames = 0

        correct_start_upto = min(settle_index, int(max_correct_frames)) if int(max_correct_frames) > 0 else settle_index
        do_end = (str(correct_ends).strip().lower() == "start_and_end")

        if correct_start_upto > 0:
            gains[:correct_start_upto] = needed[:correct_start_upto]

        end_start = b
        if do_end and not tail_within and b > int(stable_count):
            within = ((stats.log_mean - ref_log_mean).abs() <= tolerance_log).detach().cpu()
            run = 0
            for i in range(b - 1, -1, -1):
                if bool(within[i].item()):
                    run += 1
                    if run >= int(stable_count):
                        end_start = i + int(stable_count)
                        break
                else:
                    run = 0

            if end_start >= b:
                end_start = b - 1

            end_len = b - end_start
            if int(max_correct_frames) > 0:
                end_len = min(end_len, int(max_correct_frames))
                end_start = b - end_len

            if end_len > 0 and end_start < b:
                gains[end_start:] = needed[end_start:]

        if int(gain_smooth_window) > 1:
            if bool((gains != 1.0).any().item()):
                gains = smooth_1d(gains, int(gain_smooth_window)).clamp(float(gain_min), float(gain_max))
                gains[a0:a1] = 1.0

        corrected = apply_exposure_correction(images, gains)

        ev_delta = log_to_ev(stats.log_mean - ref_log_mean)
        ev_delta_cpu = ev_delta.detach().cpu().tolist()
        gains_cpu = gains.detach().cpu().tolist()

        tail_last = min(12, b)
        tail_ev = ev_delta[-tail_last:].detach().cpu()
        tail_min = float(tail_ev.min().item())
        tail_max = float(tail_ev.max().item())

        report = (
            f"WASWanExposureStabilizer: b={b}, anchor_mode={anchor_mode}, anchor=[{a0},{a1}), "
            f"ref_log_mean={ref_log_mean:.6f}, tolerance_ev={tolerance_ev:.3f}, stable_count={stable_count}, "
            f"settle_index={settle_index}, corrected_start={correct_start_upto}, "
            f"tail_within={tail_within}, correct_ends={correct_ends}, "
            f"gain_range=[{min(gains_cpu):.4f},{max(gains_cpu):.4f}], tail_ev_range_last{tail_last}=[{tail_min:+.3f},{tail_max:+.3f}]\n"
            f"per_frame_ev_delta_vs_ref={['{:+.3f}'.format(x) for x in ev_delta_cpu]}\n"
            f"per_frame_gain={['{:.4f}'.format(x) for x in gains_cpu]}"
        )

        return corrected, report


NODE_CLASS_MAPPINGS = {
    "WASWanExposureStabilizer": WASWanExposureStabilizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WASWanExposureStabilizer": "WAN 2.2 Exposure Stabilizer",
}
