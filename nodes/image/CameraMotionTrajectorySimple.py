# was_camera_path_video.py

import math
import json
import random
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F


DEFAULT_TRAJECTORY_SPEC = """{
  "loop": false,
  "default_ease": "linear",
  "keyframes": [
    {
      "frame": 0,
      "zoom": 1.0,
      "center": [0.5, 0.5],
      "angle": 0.0,
      "pan": [0.0, 0.0],
      "tilt": [0.0, 0.0],
      "dolly_strength": 0.0,
      "dolly_radius": [0.3, 0.3],
      "dolly_feather": 0.5,
      "dolly_mode": "radial",
      "sphereize_strength": 0.0,
      "sphereize_radius": [0.4, 0.4],
      "sphereize_feather": 0.5,
      "depth_strength": 0.0,
      "ease": "ease_in_out"
    },
    {
      "frame": 30,
      "zoom": 1.5,
      "center": [0.5, 0.5],
      "angle": 90.0,
      "pan": [0.0, 0.0],
      "tilt": [0.0, 0.15],
      "dolly_strength": 0.3,
      "dolly_radius": [0.35, 0.35],
      "dolly_feather": 0.5,
      "dolly_mode": "radial",
      "sphereize_strength": 0.0,
      "sphereize_radius": [0.4, 0.4],
      "sphereize_feather": 0.5,
      "depth_strength": 0.0,
      "ease": "ease_in_out"
    },
    {
      "frame": 59,
      "zoom": 2.0,
      "center": [0.6, 0.4],
      "angle": 360.0,
      "pan": [0.1, 0.0],
      "tilt": [0.0, 0.3],
      "dolly_strength": 0.6,
      "dolly_radius": [0.4, 0.4],
      "dolly_feather": 0.5,
      "dolly_mode": "radial",
      "sphereize_strength": 0.0,
      "sphereize_radius": [0.4, 0.4],
      "sphereize_feather": 0.5,
      "depth_strength": 0.0,
      "ease": "linear"
    }
  ]
}"""


def clamp_value(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def apply_easing(t: float, ease_type: str) -> float:
    t = clamp_value(t, 0.0, 1.0)
    et = (ease_type or "linear").lower()

    if et == "linear":
        return t
    if et == "ease_in":
        return t * t
    if et == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if et == "ease_in_out":
        if t < 0.5:
            return 2.0 * t * t
        return 1.0 - 2.0 * (1.0 - t) * (1.0 - t)
    if et == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    if et == "smootherstep":
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    return t


def load_trajectory_config(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {
            "loop": False,
            "default_ease": "linear",
            "keyframes": [
                {
                    "frame": 0,
                    "zoom": 1.0,
                    "center": [0.5, 0.5],
                    "angle": 0.0,
                    "pan": [0.0, 0.0],
                    "tilt": [0.0, 0.0],
                    "dolly_strength": 0.0,
                    "dolly_radius": [0.3, 0.3],
                    "dolly_feather": 0.5,
                    "sphereize_strength": 0.0,
                    "sphereize_radius": [0.4, 0.4],
                    "sphereize_feather": 0.5,
                    "ease": "linear"
                }
            ]
        }

    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse trajectory JSON. Error: {exc}\n"
            "Ensure the text is valid JSON (no comments, trailing commas, etc.)."
        )


def extract_vec2(
    data: Dict[str, Any],
    combined_key: str,
    x_key: str,
    y_key: str,
    default_x: float,
    default_y: float,
) -> Tuple[float, float, bool]:
    explicit = False
    x = default_x
    y = default_y

    if combined_key in data and isinstance(data[combined_key], (list, tuple)) and len(data[combined_key]) == 2:
        x = float(data[combined_key][0])
        y = float(data[combined_key][1])
        explicit = True
    else:
        x_present = x_key in data
        y_present = y_key in data
        if x_present:
            x = float(data[x_key])
        if y_present:
            y = float(data[y_key])
        explicit = x_present or y_present

    return x, y, explicit


def extract_vec2_speed(
    data: Dict[str, Any],
    combined_key: str,
    x_key: str,
    y_key: str,
) -> Tuple[Optional[float], Optional[float]]:
    speed_x: Optional[float] = None
    speed_y: Optional[float] = None

    if combined_key in data and isinstance(data[combined_key], (list, tuple)) and len(data[combined_key]) == 2:
        speed_x = float(data[combined_key][0])
        speed_y = float(data[combined_key][1])
    else:
        if x_key in data:
            speed_x = float(data[x_key])
        if y_key in data:
            speed_y = float(data[y_key])

    return speed_x, speed_y


def normalize_keyframes(
    config: Dict[str, Any],
    num_frames: int,
    default_dolly_mode: str,
    default_depth_strength: float,
) -> Tuple[List[Dict[str, Any]], bool]:
    loop = bool(config.get("loop", False))
    default_ease = str(config.get("default_ease", "linear"))
    raw_keyframes = config.get("keyframes", [])

    if not isinstance(raw_keyframes, list) or not raw_keyframes:
        return (
            [
                {
                    "frame": 0,
                    "zoom": 1.0,
                    "zoom_explicit": True,
                    "zoom_speed": None,
                    "center_x": 0.5,
                    "center_y": 0.5,
                    "center_explicit": True,
                    "center_speed_x": None,
                    "center_speed_y": None,
                    "angle": 0.0,
                    "angle_explicit": True,
                    "angle_speed": None,
                    "pan_x": 0.0,
                    "pan_y": 0.0,
                    "pan_explicit": True,
                    "pan_speed_x": None,
                    "pan_speed_y": None,
                    "tilt_x": 0.0,
                    "tilt_y": 0.0,
                    "tilt_explicit": True,
                    "tilt_speed_x": None,
                    "tilt_speed_y": None,
                    "dolly_strength": 0.0,
                    "dolly_strength_explicit": True,
                    "dolly_strength_speed": None,
                    "dolly_radius_x": 0.3,
                    "dolly_radius_y": 0.3,
                    "dolly_radius_explicit": True,
                    "dolly_radius_speed_x": None,
                    "dolly_radius_speed_y": None,
                    "dolly_feather": 0.5,
                    "dolly_feather_explicit": True,
                    "dolly_feather_speed": None,
                    "sphereize_strength": 0.0,
                    "sphereize_strength_explicit": True,
                    "sphereize_strength_speed": None,
                    "sphereize_radius_x": 0.4,
                    "sphereize_radius_y": 0.4,
                    "sphereize_radius_explicit": True,
                    "sphereize_radius_speed_x": None,
                    "sphereize_radius_speed_y": None,
                    "sphereize_feather": 0.5,
                    "sphereize_feather_explicit": True,
                    "sphereize_feather_speed": None,
                    "ease": default_ease,
                    "dolly_mode": default_dolly_mode,
                    "depth_strength": float(default_depth_strength),
                    "depth_strength_explicit": True,
                    "depth_strength_speed": None
                }
            ],
            loop,
        )

    normalized_keyframes: List[Dict[str, Any]] = []

    for kf in raw_keyframes:
        if not isinstance(kf, dict):
            continue

        frame = int(kf.get("frame", 0))
        frame = int(clamp_value(frame, 0, num_frames - 1))

        zoom_explicit = "zoom" in kf
        zoom = float(kf.get("zoom", 1.0))
        if zoom <= 0.0:
            zoom = 1.0
        zoom_speed = float(kf["zoom_speed"]) if "zoom_speed" in kf else None

        center_x, center_y, center_explicit = extract_vec2(
            kf,
            "center",
            "center_x",
            "center_y",
            0.5,
            0.5,
        )
        center_x = clamp_value(center_x, 0.0, 1.0)
        center_y = clamp_value(center_y, 0.0, 1.0)
        center_speed_x, center_speed_y = extract_vec2_speed(
            kf,
            "center_speed",
            "center_speed_x",
            "center_speed_y",
        )

        angle_explicit = "angle" in kf
        angle = float(kf.get("angle", 0.0))
        angle_speed = float(kf["angle_speed"]) if "angle_speed" in kf else None

        pan_x, pan_y, pan_explicit = extract_vec2(
            kf,
            "pan",
            "pan_x",
            "pan_y",
            0.0,
            0.0,
        )
        pan_x = clamp_value(pan_x, -1.0, 1.0)
        pan_y = clamp_value(pan_y, -1.0, 1.0)
        pan_speed_x, pan_speed_y = extract_vec2_speed(
            kf,
            "pan_speed",
            "pan_speed_x",
            "pan_speed_y",
        )

        tilt_x, tilt_y, tilt_explicit = extract_vec2(
            kf,
            "tilt",
            "tilt_x",
            "tilt_y",
            0.0,
            0.0,
        )
        tilt_x = clamp_value(tilt_x, -1.0, 1.0)
        tilt_y = clamp_value(tilt_y, -1.0, 1.0)
        tilt_speed_x, tilt_speed_y = extract_vec2_speed(
            kf,
            "tilt_speed",
            "tilt_speed_x",
            "tilt_speed_y",
        )

        dolly_strength_explicit = "dolly_strength" in kf
        dolly_strength = float(kf.get("dolly_strength", 0.0))
        dolly_strength_speed = None

        dolly_radius_x, dolly_radius_y, dolly_radius_explicit = extract_vec2(
            kf,
            "dolly_radius",
            "dolly_radius_x",
            "dolly_radius_y",
            0.3,
            0.3,
        )
        dolly_radius_x = clamp_value(dolly_radius_x, 0.0, 1.0)
        dolly_radius_y = clamp_value(dolly_radius_y, 0.0, 1.0)
        dolly_radius_speed_x = None
        dolly_radius_speed_y = None

        dolly_feather_explicit = "dolly_feather" in kf
        dolly_feather = float(kf.get("dolly_feather", 0.5))
        dolly_feather = clamp_value(dolly_feather, 0.0, 1.0)
        dolly_feather_speed = None

        sphereize_strength_explicit = "sphereize_strength" in kf
        sphereize_strength = float(kf.get("sphereize_strength", 0.0))
        sphereize_strength_speed = None

        sphereize_radius_x, sphereize_radius_y, sphereize_radius_explicit = extract_vec2(
            kf,
            "sphereize_radius",
            "sphereize_radius_x",
            "sphereize_radius_y",
            0.4,
            0.4,
        )
        sphereize_radius_x = clamp_value(sphereize_radius_x, 0.0, 1.0)
        sphereize_radius_y = clamp_value(sphereize_radius_y, 0.0, 1.0)
        sphereize_radius_speed_x = None
        sphereize_radius_speed_y = None

        sphereize_feather_explicit = "sphereize_feather" in kf
        sphereize_feather = float(kf.get("sphereize_feather", 0.5))
        sphereize_feather = clamp_value(sphereize_feather, 0.0, 1.0)
        sphereize_feather_speed = None

        dolly_mode = str(kf.get("dolly_mode", default_dolly_mode)).lower()
        if dolly_mode not in ("radial", "aspect", "box"):
            dolly_mode = default_dolly_mode

        depth_strength_explicit = "depth_strength" in kf
        depth_strength_val = float(kf.get("depth_strength", default_depth_strength))
        depth_strength_speed = None

        ease = str(kf.get("ease", default_ease))

        normalized_keyframes.append(
            {
                "frame": frame,
                "zoom": zoom,
                "zoom_explicit": zoom_explicit,
                "zoom_speed": zoom_speed,
                "center_x": center_x,
                "center_y": center_y,
                "center_explicit": center_explicit,
                "center_speed_x": center_speed_x,
                "center_speed_y": center_speed_y,
                "angle": angle,
                "angle_explicit": angle_explicit,
                "angle_speed": angle_speed,
                "pan_x": pan_x,
                "pan_y": pan_y,
                "pan_explicit": pan_explicit,
                "pan_speed_x": pan_speed_x,
                "pan_speed_y": pan_speed_y,
                "tilt_x": tilt_x,
                "tilt_y": tilt_y,
                "tilt_explicit": tilt_explicit,
                "tilt_speed_x": tilt_speed_x,
                "tilt_speed_y": tilt_speed_y,
                "dolly_strength": dolly_strength,
                "dolly_strength_explicit": dolly_strength_explicit,
                "dolly_strength_speed": dolly_strength_speed,
                "dolly_radius_x": dolly_radius_x,
                "dolly_radius_y": dolly_radius_y,
                "dolly_radius_explicit": dolly_radius_explicit,
                "dolly_radius_speed_x": dolly_radius_speed_x,
                "dolly_radius_speed_y": dolly_radius_speed_y,
                "dolly_feather": dolly_feather,
                "dolly_feather_explicit": dolly_feather_explicit,
                "dolly_feather_speed": dolly_feather_speed,
                "sphereize_strength": sphereize_strength,
                "sphereize_strength_explicit": sphereize_strength_explicit,
                "sphereize_strength_speed": sphereize_strength_speed,
                "sphereize_radius_x": sphereize_radius_x,
                "sphereize_radius_y": sphereize_radius_y,
                "sphereize_radius_explicit": sphereize_radius_explicit,
                "sphereize_radius_speed_x": sphereize_radius_speed_x,
                "sphereize_radius_speed_y": sphereize_radius_speed_y,
                "sphereize_feather": sphereize_feather,
                "sphereize_feather_explicit": sphereize_feather_explicit,
                "sphereize_feather_speed": sphereize_feather_speed,
                "ease": ease,
                "dolly_mode": dolly_mode,
                "depth_strength": depth_strength_val,
                "depth_strength_explicit": depth_strength_explicit,
                "depth_strength_speed": depth_strength_speed,
            }
        )

    if not normalized_keyframes:
        normalized_keyframes.append(
            {
                "frame": 0,
                "zoom": 1.0,
                "zoom_explicit": True,
                "zoom_speed": None,
                "center_x": 0.5,
                "center_y": 0.5,
                "center_explicit": True,
                "center_speed_x": None,
                "center_speed_y": None,
                "angle": 0.0,
                "angle_explicit": True,
                "angle_speed": None,
                "pan_x": 0.0,
                "pan_y": 0.0,
                "pan_explicit": True,
                "pan_speed_x": None,
                "pan_speed_y": None,
                "tilt_x": 0.0,
                "tilt_y": 0.0,
                "tilt_explicit": True,
                "tilt_speed_x": None,
                "tilt_speed_y": None,
                "dolly_strength": 0.0,
                "dolly_strength_explicit": True,
                "dolly_strength_speed": None,
                "dolly_radius_x": 0.3,
                "dolly_radius_y": 0.3,
                "dolly_radius_explicit": True,
                "dolly_radius_speed_x": None,
                "dolly_radius_speed_y": None,
                "dolly_feather": 0.5,
                "dolly_feather_explicit": True,
                "dolly_feather_speed": None,
                "sphereize_strength": 0.0,
                "sphereize_strength_explicit": True,
                "sphereize_strength_speed": None,
                "sphereize_radius_x": 0.4,
                "sphereize_radius_y": 0.4,
                "sphereize_radius_explicit": True,
                "sphereize_radius_speed_x": None,
                "sphereize_radius_speed_y": None,
                "sphereize_feather": 0.5,
                "sphereize_feather_explicit": True,
                "sphereize_feather_speed": None,
                "ease": default_ease,
                "dolly_mode": default_dolly_mode,
                "depth_strength": float(default_depth_strength),
                "depth_strength_explicit": True,
                "depth_strength_speed": None,
            }
        )

    normalized_keyframes.sort(key=lambda k: k["frame"])
    return normalized_keyframes, loop


def compute_segment_end_value_scalar(
    prop_name: str,
    before_kf: Dict[str, Any],
    after_kf: Dict[str, Any],
    frames_delta: int,
) -> Tuple[float, float]:
    start_key = prop_name
    explicit_key = f"{prop_name}_explicit"
    speed_key = f"{prop_name}_speed"

    v0 = float(before_kf[start_key])
    v1 = float(after_kf[start_key])

    explicit_after = bool(after_kf.get(explicit_key, True))
    speed = before_kf.get(speed_key, None)

    if not explicit_after and speed is not None and frames_delta > 0:
        v1 = v0 + float(speed) * float(frames_delta)

    return v0, v1


def compute_segment_end_value_vec2(
    base_name: str,
    before_kf: Dict[str, Any],
    after_kf: Dict[str, Any],
    frames_delta: int,
) -> Tuple[float, float, float, float]:
    x_key = f"{base_name}_x"
    y_key = f"{base_name}_y"
    explicit_key = f"{base_name}_explicit"
    speed_x_key = f"{base_name}_speed_x"
    speed_y_key = f"{base_name}_speed_y"

    v0x = float(before_kf[x_key])
    v0y = float(before_kf[y_key])
    v1x = float(after_kf[x_key])
    v1y = float(after_kf[y_key])

    explicit_after = bool(after_kf.get(explicit_key, True))
    sx = before_kf.get(speed_x_key, None)
    sy = before_kf.get(speed_y_key, None)

    if not explicit_after and frames_delta > 0 and (sx is not None or sy is not None):
        if sx is not None:
            v1x = v0x + float(sx) * float(frames_delta)
        if sy is not None:
            v1y = v0y + float(sy) * float(frames_delta)

    return v0x, v0y, v1x, v1y


def interpolate_camera_paths(
    keyframes: List[Dict[str, Any]],
    num_frames: int,
    loop: bool,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[str],
]:
    zoom_list: List[float] = [1.0] * num_frames
    center_x_list: List[float] = [0.5] * num_frames
    center_y_list: List[float] = [0.5] * num_frames
    angle_list: List[float] = [0.0] * num_frames
    pan_x_list: List[float] = [0.0] * num_frames
    pan_y_list: List[float] = [0.0] * num_frames
    tilt_x_list: List[float] = [0.0] * num_frames
    tilt_y_list: List[float] = [0.0] * num_frames
    dolly_strength_list: List[float] = [0.0] * num_frames
    dolly_radius_x_list: List[float] = [0.3] * num_frames
    dolly_radius_y_list: List[float] = [0.3] * num_frames
    dolly_feather_list: List[float] = [0.5] * num_frames
    sphereize_strength_list: List[float] = [0.0] * num_frames
    sphereize_radius_x_list: List[float] = [0.4] * num_frames
    sphereize_radius_y_list: List[float] = [0.4] * num_frames
    sphereize_feather_list: List[float] = [0.5] * num_frames
    depth_strength_list: List[float] = [0.0] * num_frames
    dolly_mode_list: List[str] = ["radial"] * num_frames

    first_kf = keyframes[0]
    last_kf = keyframes[-1]

    for frame_index in range(num_frames):
        before_kf = None
        after_kf = None

        for kf in keyframes:
            if kf["frame"] <= frame_index:
                before_kf = kf
            if kf["frame"] >= frame_index and after_kf is None:
                after_kf = kf

        if before_kf is None:
            before_kf = first_kf
        if after_kf is None:
            if loop and len(keyframes) > 1 and frame_index > last_kf["frame"]:
                before_kf = last_kf
                after_kf = first_kf
            else:
                after_kf = last_kf

        if not loop or after_kf is not first_kf or frame_index <= last_kf["frame"]:
            before_frame = before_kf["frame"]
            after_frame = after_kf["frame"]
        else:
            before_frame = last_kf["frame"]
            after_frame = first_kf["frame"] + num_frames

        if before_frame == after_frame:
            t = 0.0
            frames_delta = 1
        else:
            frames_delta = after_frame - before_frame
            if frames_delta <= 0:
                frames_delta = 1
            t = (frame_index - before_frame) / float(frames_delta)
            t = clamp_value(t, 0.0, 1.0)

        ease_type = before_kf.get("ease", "linear")
        t_eased = apply_easing(t, ease_type)

        def lerp(a: float, b: float, alpha: float) -> float:
            return a + (b - a) * alpha

        zoom_start, zoom_end = compute_segment_end_value_scalar(
            "zoom", before_kf, after_kf, frames_delta
        )
        zoom_value = lerp(zoom_start, zoom_end, t_eased)

        c0x, c0y, c1x, c1y = compute_segment_end_value_vec2(
            "center", before_kf, after_kf, frames_delta
        )
        center_x_value = lerp(c0x, c1x, t_eased)
        center_y_value = lerp(c0y, c1y, t_eased)

        angle_start, angle_end = compute_segment_end_value_scalar(
            "angle", before_kf, after_kf, frames_delta
        )
        angle_value = lerp(angle_start, angle_end, t_eased)

        p0x, p0y, p1x, p1y = compute_segment_end_value_vec2(
            "pan", before_kf, after_kf, frames_delta
        )
        pan_x_value = lerp(p0x, p1x, t_eased)
        pan_y_value = lerp(p0y, p1y, t_eased)

        t0x, t0y, t1x, t1y = compute_segment_end_value_vec2(
            "tilt", before_kf, after_kf, frames_delta
        )
        tilt_x_value = lerp(t0x, t1x, t_eased)
        tilt_y_value = lerp(t0y, t1y, t_eased)

        ds0, ds1 = compute_segment_end_value_scalar(
            "dolly_strength", before_kf, after_kf, frames_delta
        )
        dolly_strength_value = lerp(ds0, ds1, t_eased)

        dr0x, dr0y, dr1x, dr1y = compute_segment_end_value_vec2(
            "dolly_radius", before_kf, after_kf, frames_delta
        )
        dolly_radius_x_value = lerp(dr0x, dr1x, t_eased)
        dolly_radius_y_value = lerp(dr0y, dr1y, t_eased)

        df0, df1 = compute_segment_end_value_scalar(
            "dolly_feather", before_kf, after_kf, frames_delta
        )
        dolly_feather_value = lerp(df0, df1, t_eased)

        sphs0, sphs1 = compute_segment_end_value_scalar(
            "sphereize_strength", before_kf, after_kf, frames_delta
        )
        sphereize_strength_value = lerp(sphs0, sphs1, t_eased)

        sphr0x, sphr0y, sphr1x, sphr1y = compute_segment_end_value_vec2(
            "sphereize_radius", before_kf, after_kf, frames_delta
        )
        sphereize_radius_x_value = lerp(sphr0x, sphr1x, t_eased)
        sphereize_radius_y_value = lerp(sphr0y, sphr1y, t_eased)

        sphf0, sphf1 = compute_segment_end_value_scalar(
            "sphereize_feather", before_kf, after_kf, frames_delta
        )
        sphereize_feather_value = lerp(sphf0, sphf1, t_eased)

        dep0, dep1 = compute_segment_end_value_scalar(
            "depth_strength", before_kf, after_kf, frames_delta
        )
        depth_strength_value = lerp(dep0, dep1, t_eased)

        zoom_list[frame_index] = zoom_value
        center_x_list[frame_index] = center_x_value
        center_y_list[frame_index] = center_y_value
        angle_list[frame_index] = angle_value
        pan_x_list[frame_index] = pan_x_value
        pan_y_list[frame_index] = pan_y_value
        tilt_x_list[frame_index] = tilt_x_value
        tilt_y_list[frame_index] = tilt_y_value
        dolly_strength_list[frame_index] = dolly_strength_value
        dolly_radius_x_list[frame_index] = dolly_radius_x_value
        dolly_radius_y_list[frame_index] = dolly_radius_y_value
        dolly_feather_list[frame_index] = dolly_feather_value
        sphereize_strength_list[frame_index] = sphereize_strength_value
        sphereize_radius_x_list[frame_index] = sphereize_radius_x_value
        sphereize_radius_y_list[frame_index] = sphereize_radius_y_value
        sphereize_feather_list[frame_index] = sphereize_feather_value
        depth_strength_list[frame_index] = depth_strength_value
        dolly_mode_list[frame_index] = before_kf.get("dolly_mode", "radial")

    return (
        zoom_list,
        center_x_list,
        center_y_list,
        angle_list,
        pan_x_list,
        pan_y_list,
        tilt_x_list,
        tilt_y_list,
        dolly_strength_list,
        dolly_radius_x_list,
        dolly_radius_y_list,
        dolly_feather_list,
        sphereize_strength_list,
        sphereize_radius_x_list,
        sphereize_radius_y_list,
        sphereize_feather_list,
        depth_strength_list,
        dolly_mode_list,
    )


def build_camera_shake(
    num_frames: int,
    enable: bool,
    position_amplitude: float,
    rotation_amplitude: float,
    seed: int,
) -> Tuple[List[float], List[float], List[float]]:
    shake_x: List[float] = [0.0] * num_frames
    shake_y: List[float] = [0.0] * num_frames
    shake_angle: List[float] = [0.0] * num_frames

    if not enable or (position_amplitude <= 0.0 and rotation_amplitude <= 0.0):
        return shake_x, shake_y, shake_angle

    rng = random.Random(int(seed))

    current_x = 0.0
    current_y = 0.0
    current_angle = 0.0

    for i in range(1, num_frames):
        if position_amplitude > 0.0:
            step_x = rng.uniform(-position_amplitude / 3.0, position_amplitude / 3.0)
            step_y = rng.uniform(-position_amplitude / 3.0, position_amplitude / 3.0)
            current_x = clamp_value(current_x + step_x, -position_amplitude, position_amplitude)
            current_y = clamp_value(current_y + step_y, -position_amplitude, position_amplitude)

        if rotation_amplitude > 0.0:
            step_angle = rng.uniform(-rotation_amplitude / 3.0, rotation_amplitude / 3.0)
            current_angle = clamp_value(
                current_angle + step_angle,
                -rotation_amplitude,
                rotation_amplitude,
            )

        shake_x[i] = current_x
        shake_y[i] = current_y
        shake_angle[i] = current_angle

    return shake_x, shake_y, shake_angle


def build_base_grid(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return grid.unsqueeze(0)


def create_frame_grid(
    base_grid: torch.Tensor,
    center_x_norm: float,
    center_y_norm: float,
    zoom: float,
    angle_deg: float,
    pan_x_norm: float,
    pan_y_norm: float,
    tilt_x: float,
    tilt_y: float,
    dolly_strength: float,
    dolly_radius_x: float,
    dolly_radius_y: float,
    dolly_feather: float,
    sphereize_strength: float,
    sphereize_radius_x: float,
    sphereize_radius_y: float,
    sphereize_feather: float,
    dolly_mode: str,
    depth_grid: Optional[torch.Tensor],
    depth_strength: float,
    shake_x_norm: float,
    shake_y_norm: float,
) -> torch.Tensor:
    center_x_n = center_x_norm * 2.0 - 1.0
    center_y_n = center_y_norm * 2.0 - 1.0
    pan_x_n = pan_x_norm * 2.0
    pan_y_n = pan_y_norm * 2.0

    shake_x_n = shake_x_norm
    shake_y_n = shake_y_norm

    grid = base_grid
    x = grid[..., 0]
    y = grid[..., 1]

    x0 = x - center_x_n
    y0 = y - center_y_n

    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    if zoom <= 0.0:
        zoom = 1.0
    scale = 1.0 / zoom

    x1 = (cos_t * x0 - sin_t * y0) * scale
    y1 = (sin_t * x0 + cos_t * y0) * scale

    x2 = x1 + center_x_n + pan_x_n + shake_x_n
    y2 = y1 + center_y_n + pan_y_n + shake_y_n

    x_cam = x2
    y_cam = y2

    if tilt_y != 0.0:
        ty_factor = 1.0 + tilt_y * ((y2 + 1.0) * 0.5)
        x3 = x2 * ty_factor
    else:
        x3 = x2

    if tilt_x != 0.0:
        tx_factor = 1.0 + tilt_x * ((x2 + 1.0) * 0.5)
        y3 = y2 * tx_factor
    else:
        y3 = y2

    if dolly_strength != 0.0 and dolly_radius_x > 0.0 and dolly_radius_y > 0.0:
        rx = max(dolly_radius_x * 2.0, 1e-6)
        ry = max(dolly_radius_y * 2.0, 1e-6)

        dx = x3 - center_x_n
        dy = y3 - center_y_n

        mode = (dolly_mode or "radial").lower()

        if mode == "box":
            ax = torch.abs(dx) / rx
            ay = torch.abs(dy) / ry
            r = torch.max(ax, ay)
        elif mode == "aspect":
            r = torch.sqrt((dx / rx) ** 2 + (dy / ry) ** 2 + 1e-8)
            h, w = x3.shape[1], x3.shape[2]
            aspect = float(w) / float(h) if h > 0 else 1.0
            r = r * aspect
        else:
            r = torch.sqrt((dx / rx) ** 2 + (dy / ry) ** 2 + 1e-8)

        inner_mask = r <= 1.0
        if inner_mask.any():
            r_norm = torch.clamp(r[inner_mask], 0.0, 1.0)
            w = r_norm * r_norm * (3.0 - 2.0 * r_norm)
            gamma = 1.0 + 3.0 * (1.0 - clamp_value(dolly_feather, 0.0, 1.0))
            w = w ** gamma

            strength = float(dolly_strength)
            if strength > 0.0:
                s = 1.0 / (1.0 + strength * w)
            else:
                s = 1.0 + strength * w

            dx_new = dx.clone()
            dy_new = dy.clone()
            dx_new[inner_mask] = dx[inner_mask] * s
            dy_new[inner_mask] = dy[inner_mask] * s

            x4 = center_x_n + dx_new
            y4 = center_y_n + dy_new
        else:
            x4 = x3
            y4 = y3
    else:
        x4 = x3
        y4 = y3

    if sphereize_strength != 0.0 and sphereize_radius_x > 0.0 and sphereize_radius_y > 0.0:
        rx_sph = max(sphereize_radius_x * 2.0, 1e-6)
        ry_sph = max(sphereize_radius_y * 2.0, 1e-6)

        dx_sph = x4 - center_x_n
        dy_sph = y4 - center_y_n

        r_sph = torch.sqrt((dx_sph / rx_sph) ** 2 + (dy_sph / ry_sph) ** 2 + 1e-8)
        r_norm_sph = torch.clamp(r_sph, 0.0, 1.0)

        w_sph = r_norm_sph * r_norm_sph * (3.0 - 2.0 * r_norm_sph)
        gamma_sph = 1.0 + 3.0 * (1.0 - clamp_value(sphereize_feather, 0.0, 1.0))
        w_sph = w_sph ** gamma_sph

        k_sph = 1.0 + float(sphereize_strength) * w_sph

        dx_sph_new = dx_sph * k_sph
        dy_sph_new = dy_sph * k_sph

        x5 = center_x_n + dx_sph_new
        y5 = center_y_n + dy_sph_new
    else:
        x5 = x4
        y5 = y4

    if depth_grid is not None and depth_strength != 0.0:
        d = depth_grid
        if d.ndim == 3 and d.shape[0] == 1:
            d = d[0]
        d_min = d.amin()
        d_max = d.amax()
        if float(d_max - d_min) > 1e-6:
            d_norm = (d - d_min) / (d_max - d_min)
        else:
            d_norm = torch.zeros_like(d)

        alpha = clamp_value(abs(depth_strength), 0.0, 1.0)
        if depth_strength >= 0.0:
            m = d_norm
        else:
            m = 1.0 - d_norm
        depth_scale = 1.0 - alpha * m
        depth_scale = torch.clamp(depth_scale, 0.0, 1.0)

        x_final = x_cam + (x5 - x_cam) * depth_scale
        y_final = y_cam + (y5 - y_cam) * depth_scale
    else:
        x_final = x5
        y_final = y5

    transformed_grid = torch.stack((x_final, y_final), dim=-1)
    return transformed_grid


class WASCameraMotionTrajectory:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_frames": (
                    "INT",
                    {
                        "default": 60,
                        "min": 1,
                        "max": 2048,
                        "step": 1,
                    },
                ),
                "trajectory_spec": (
                    "STRING",
                    {
                        "default": DEFAULT_TRAJECTORY_SPEC,
                        "multiline": True,
                    },
                ),
                "edge_mode": (
                    ["border", "mirror", "wrap"],
                    {
                        "default": "mirror",
                    },
                ),
                "enable_camera_shake": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "shake_position_amplitude": (
                    "FLOAT",
                    {
                        "default": 0.03,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.001,
                    },
                ),
                "shake_rotation_amplitude": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 45.0,
                        "step": 0.1,
                    },
                ),
                "shake_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "depth_map": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("video", "frame_count")
    FUNCTION = "create_video"
    CATEGORY = "WAS/Video"

    def create_video(
        self,
        image: torch.Tensor,
        num_frames: int,
        trajectory_spec: str,
        edge_mode: str,
        enable_camera_shake: bool,
        shake_position_amplitude: float,
        shake_rotation_amplitude: float,
        shake_seed: int,
        depth_map: Optional[torch.Tensor] = None,
    ):
        if not isinstance(image, torch.Tensor):
            raise TypeError("Expected 'image' to be a torch.Tensor")

        if image.ndim != 4:
            raise ValueError(f"Expected 'image' shape [B, H, W, C], got {tuple(image.shape)}")

        batch_size, height, width, channels = image.shape

        config = load_trajectory_config(trajectory_spec)

        config_default_dolly_mode = str(config.get("dolly_mode", "radial")).lower()
        if config_default_dolly_mode not in ("radial", "aspect", "box"):
            config_default_dolly_mode = "radial"

        config_default_depth_strength = float(config.get("depth_strength", 0.0))

        keyframes, loop = normalize_keyframes(
            config=config,
            num_frames=num_frames,
            default_dolly_mode=config_default_dolly_mode,
            default_depth_strength=config_default_depth_strength,
        )

        (
            zoom_list,
            center_x_list,
            center_y_list,
            angle_list,
            pan_x_list,
            pan_y_list,
            tilt_x_list,
            tilt_y_list,
            dolly_strength_list,
            dolly_radius_x_list,
            dolly_radius_y_list,
            dolly_feather_list,
            sphereize_strength_list,
            sphereize_radius_x_list,
            sphereize_radius_y_list,
            sphereize_feather_list,
            depth_strength_list,
            dolly_mode_list,
        ) = interpolate_camera_paths(keyframes, num_frames, loop)

        shake_x_list, shake_y_list, shake_angle_list = build_camera_shake(
            num_frames=num_frames,
            enable=enable_camera_shake,
            position_amplitude=shake_position_amplitude,
            rotation_amplitude=shake_rotation_amplitude,
            seed=shake_seed,
        )

        device = image.device
        dtype = image.dtype

        image_chw = image.movedim(-1, 1).contiguous()
        base_grid = build_base_grid(height, width, device=device, dtype=dtype)

        depth_grid_base: Optional[torch.Tensor] = None
        if depth_map is not None:
            if not isinstance(depth_map, torch.Tensor) or depth_map.ndim != 4:
                raise ValueError(f"Expected 'depth_map' shape [B, H, W, C], got {tuple(depth_map.shape)}")
            depth_b, depth_h, depth_w, depth_c = depth_map.shape
            depth_t = depth_map.movedim(-1, 1).to(dtype)  # [B, C, H, W]

            if depth_h != height or depth_w != width:
                depth_t = F.interpolate(
                    depth_t,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )

            if depth_c == 3:
                depth_gray = (
                    0.299 * depth_t[:, 0:1] +
                    0.587 * depth_t[:, 1:2] +
                    0.114 * depth_t[:, 2:3]
                )
            else:
                depth_gray = depth_t[:, 0:1]
            depth_grid_base = depth_gray

        if edge_mode == "mirror":
            padding_mode = "reflection"
        elif edge_mode == "border":
            padding_mode = "border"
        elif edge_mode == "wrap":
            padding_mode = "border"
        else:
            padding_mode = "border"

        all_frames: List[torch.Tensor] = []

        for frame_index in range(num_frames):
            if batch_size > 1 and num_frames == batch_size:
                src_index = frame_index
            else:
                src_index = frame_index % batch_size

            zoom_value = float(zoom_list[frame_index])
            center_x_value = float(center_x_list[frame_index])
            center_y_value = float(center_y_list[frame_index])
            angle_value = float(angle_list[frame_index])
            pan_x_value = float(pan_x_list[frame_index])
            pan_y_value = float(pan_y_list[frame_index])
            tilt_x_value = float(tilt_x_list[frame_index])
            tilt_y_value = float(tilt_y_list[frame_index])
            dolly_strength_value = float(dolly_strength_list[frame_index])
            dolly_radius_x_value = float(dolly_radius_x_list[frame_index])
            dolly_radius_y_value = float(dolly_radius_y_list[frame_index])
            dolly_feather_value = float(dolly_feather_list[frame_index])
            sphereize_strength_value = float(sphereize_strength_list[frame_index])
            sphereize_radius_x_value = float(sphereize_radius_x_list[frame_index])
            sphereize_radius_y_value = float(sphereize_radius_y_list[frame_index])
            sphereize_feather_value = float(sphereize_feather_list[frame_index])
            shake_x_value = float(shake_x_list[frame_index])
            shake_y_value = float(shake_y_list[frame_index])
            shake_angle_value = float(shake_angle_list[frame_index])
            depth_strength_value = float(depth_strength_list[frame_index])
            dolly_mode_value = str(dolly_mode_list[frame_index])

            total_angle = angle_value + shake_angle_value

            depth_grid_frame: Optional[torch.Tensor] = None
            if depth_grid_base is not None:
                depth_src_index = src_index
                if depth_grid_base.shape[0] == 1:
                    depth_src_index = 0
                elif depth_grid_base.shape[0] != batch_size:
                    depth_src_index = src_index % depth_grid_base.shape[0]
                depth_grid_frame = depth_grid_base[depth_src_index, 0:1]

            frame_grid = create_frame_grid(
                base_grid=base_grid,
                center_x_norm=center_x_value,
                center_y_norm=center_y_value,
                zoom=zoom_value,
                angle_deg=total_angle,
                pan_x_norm=pan_x_value,
                pan_y_norm=pan_y_value,
                tilt_x=tilt_x_value,
                tilt_y=tilt_y_value,
                dolly_strength=dolly_strength_value,
                dolly_radius_x=dolly_radius_x_value,
                dolly_radius_y=dolly_radius_y_value,
                dolly_feather=dolly_feather_value,
                sphereize_strength=sphereize_strength_value,
                sphereize_radius_x=sphereize_radius_x_value,
                sphereize_radius_y=sphereize_radius_y_value,
                sphereize_feather=sphereize_feather_value,
                dolly_mode=dolly_mode_value,
                depth_grid=depth_grid_frame,
                depth_strength=depth_strength_value,
                shake_x_norm=shake_x_value,
                shake_y_norm=shake_y_value,
            )

            if edge_mode == "wrap":
                frame_grid = (frame_grid + 1.0) % 2.0 - 1.0

            src_frame = image_chw[src_index : src_index + 1]

            transformed = F.grid_sample(
                src_frame,
                frame_grid,
                mode="bilinear",
                padding_mode=padding_mode,
                align_corners=True,
            )
            all_frames.append(transformed)

        video_chw = torch.cat(all_frames, dim=0)
        video_bhwc = video_chw.movedim(1, 3).contiguous()
        return (video_bhwc, num_frames)


NODE_CLASS_MAPPINGS = {
    "WASCameraMotionTrajectory": WASCameraMotionTrajectory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WASCameraMotionTrajectory": "Camera Motion Trajectory from Images",
}
