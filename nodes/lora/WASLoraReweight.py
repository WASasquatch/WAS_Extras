import re
import hashlib
import torch

from pathlib import Path
from safetensors.torch import save_file, load_file
from typing import Dict, Tuple, Optional

from comfy.utils import load_torch_file
from comfy import sd as comfy_sd
from folder_paths import get_filename_list, get_full_path


_RX_BLOCK_FLEX = re.compile(
    r"(?:^|[._])(?:transformer(?:[._])?)?(?:blocks?|layers?|stages?|resblocks?|h)[._](\d+)(?:[._]|$)",
    re.IGNORECASE,
)

_RX_QWEN_BLOCK = re.compile(r"^transformer_blocks\.(\d+)\.", re.IGNORECASE)
_RX_WAN_BLOCK = re.compile(r"^lora_unet_blocks_(\d+)_", re.IGNORECASE)
_RX_FLUX_BLOCK = re.compile(r"^lora_unet_(?:double|single)_blocks_(\d+)_", re.IGNORECASE)
_RX_ZIMG_TURBO_BLOCK = re.compile(r"^diffusion_model\.layers\.(\d+)\.", re.IGNORECASE)
_RX_SD_UNET_BLOCK = re.compile(r"^lora_unet_(?:down|up)_blocks_(\d+)_", re.IGNORECASE)
_RX_SD_TE_BLOCK = re.compile(r"^lora_te_.*?_encoder_layers_(\d+)_", re.IGNORECASE)
_RX_SDXL_UNET_BLOCK = re.compile(r"^lora_unet_(?:input|output)_blocks_(\d+)_", re.IGNORECASE)
_RX_SDXL_MID_BLOCK = re.compile(r"^lora_unet_middle_block_", re.IGNORECASE)
_RX_SDXL_TE1_BLOCK = re.compile(r"^lora_te1_.*?_encoder_layers_(\d+)_", re.IGNORECASE)
_RX_SDXL_TE2_BLOCK = re.compile(r"^lora_te2_.*?_encoder_layers_(\d+)_", re.IGNORECASE)


def _infer_block_preset_from_lora_keys(keys) -> str:
    for k in keys:
        if isinstance(k, str) and k.startswith("diffusion_model.layers."):
            return "zimg-turbo"
        if isinstance(k, str) and (k.startswith("lora_unet_double_blocks_") or k.startswith("lora_unet_single_blocks_")):
            return "flux"
        if isinstance(k, str) and k.startswith("lora_unet_blocks_"):
            return "wan"
        if isinstance(k, str) and k.startswith("transformer_blocks."):
            return "qwen"
        if isinstance(k, str) and (k.startswith("lora_te1_") or k.startswith("lora_te2_")):
            return "sdxl"
        if isinstance(k, str) and k.startswith("lora_unet_"):
            return "sd"
    return "generic"


def _parse_block_index(key: str, preset: str) -> Optional[int]:
    p = (preset or "generic").lower().replace("_", "-")
    try:
        if p == "qwen":
            m = _RX_QWEN_BLOCK.search(key)
            return int(m.group(1)) if m else None
        if p == "wan":
            m = _RX_WAN_BLOCK.search(key)
            return int(m.group(1)) if m else None
        if p == "flux":
            m = _RX_FLUX_BLOCK.search(key)
            return int(m.group(1)) if m else None
        if p == "zimg-turbo":
            m = _RX_ZIMG_TURBO_BLOCK.search(key)
            return int(m.group(1)) if m else None
        if p == "sdxl":
            m = _RX_SDXL_TE1_BLOCK.search(key)
            if m:
                return int(m.group(1))
            m = _RX_SDXL_TE2_BLOCK.search(key)
            if m:
                return int(m.group(1))
            m = _RX_SDXL_UNET_BLOCK.search(key)
            if m:
                return int(m.group(1))
            if _RX_SDXL_MID_BLOCK.search(key):
                return None
            return None
        if p == "sd":
            m = _RX_SD_TE_BLOCK.search(key)
            if m:
                return int(m.group(1))
            m = _RX_SD_UNET_BLOCK.search(key)
            return int(m.group(1)) if m else None

        m = _RX_BLOCK_FLEX.search(key)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _detect_total_blocks_from_model(model) -> int:
    base = getattr(model, "model", None)
    if base is None:
        return 0
    max_idx = -1
    try:
        for name, _ in base.named_modules():
            m = _RX_BLOCK_FLEX.search(name)
            if m:
                bi = int(m.group(1))
                if bi > max_idx:
                    max_idx = bi
    except Exception:
        return 0
    return (max_idx + 1) if max_idx >= 0 else 0


def _detect_total_blocks_from_lora(state: Dict[str, torch.Tensor], preset: str) -> int:
    max_idx = -1
    for k in state.keys():
        bi = _parse_block_index(k, preset)
        if bi is not None and bi > max_idx:
            max_idx = bi
    return (max_idx + 1) if max_idx >= 0 else 0


def _compute_block_scale(i: Optional[int], total_blocks: int, g: float, s_front: float, s_mid: float, s_back: float, s_last: float):
    if total_blocks <= 0 or i is None:
        base = g
    else:
        third = max(1, total_blocks // 3)
        if i < third:
            base = g * s_front
        elif i < 2 * third:
            base = g * s_mid
        else:
            base = g * s_back
        if i == (total_blocks - 1):
            base *= s_last
    return base


def _which_lora_part(key: str):
    k = key.lower()
    if k.endswith(("lora.up.weight", "lora_up.weight", "loraa.weight", "lora_a.weight")):
        return "up"
    if k.endswith(("lora.down.weight", "lora_down.weight", "lorab.weight", "lora_b.weight")):
        return "down"
    return None


def _reweight_state_dict(
    state: Dict[str, torch.Tensor],
    total_blocks: int,
    g: float, s_front: float, s_mid: float, s_back: float, s_last: float,
    scale_target: str,
    filter_by_block_range: bool,
    filter_cutoff_blocks: int,
    preset: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    out: Dict[str, torch.Tensor] = {}
    changed = 0
    dropped = 0
    kept = 0
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        bi = _parse_block_index(k, preset)
        if filter_by_block_range and filter_cutoff_blocks > 0 and bi is not None and bi >= filter_cutoff_blocks:
            dropped += 1
            continue
        part = _which_lora_part(k)
        apply_scale = (
            (scale_target == "up_only" and part == "up") or
            (scale_target == "down_only" and part == "down") or
            (scale_target == "both" and (part == "up" or part == "down"))
        )
        if apply_scale:
            s = _compute_block_scale(bi, total_blocks, g, s_front, s_mid, s_back, s_last)
            if s != 1.0:
                v = v * s
                changed += 1
        out[k] = v
        kept += 1
    return out, {"changed": changed, "dropped": dropped, "kept": kept}


def _get_output_dir() -> Path:
    try:
        from folder_paths import get_output_directory
        return Path(get_output_directory())
    except Exception:
        return Path.cwd() / "output"


def _auto_name(src_name: str, scale_target: str, g: float, s_front: float, s_mid: float, s_back: float, s_last: float) -> str:
    stem = Path(src_name).stem
    return f"{stem}.reweighted.{scale_target}.g{g:.2f}.f{round(s_front,2)}.m{round(s_mid,2)}.b{round(s_back,2)}.L{round(s_last,2)}.safetensors"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class WASApplyReweightedLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (tuple(get_filename_list("loras")), {"tooltip": "LoRA filename from models/loras"}),
                "strength_model": ("FLOAT", {"default": 0.8, "min": -5.0, "max": 5.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 0.8, "min": -5.0, "max": 5.0, "step": 0.01}),
                "global_scale": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "front_scale": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "mid_scale": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "back_scale": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "last_block_scale": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "scale_target": (["up_only", "down_only", "both"], {"default": "up_only"}),
                "block_preset": (["auto", "wan", "qwen", "flux", "zimg-turbo", "sd", "sdxl", "generic"], {"default": "auto"}),
                "filter_by_block_range": ("BOOLEAN", {"default": True}),
                "save_reweighted": ("BOOLEAN", {"default": True}),
                "output_filename": ("STRING", {"default": "", "tooltip": "Optional; leave blank for auto-naming"}),
                "verify_roundtrip": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "DICT")
    RETURN_NAMES = ("model", "clip", "stats")
    FUNCTION = "apply"
    CATEGORY = "model/LoRA"
    DESCRIPTION = "WAN LoRA reweight in-memory. Saves the exact applied dict to output/. Optional roundtrip verification."

    def apply(
        self,
        model,
        clip,
        lora_name: str,
        strength_model: float,
        strength_clip: float,
        global_scale: float,
        front_scale: float,
        mid_scale: float,
        back_scale: float,
        last_block_scale: float,
        scale_target: str,
        block_preset: str,
        filter_by_block_range: bool,
        save_reweighted: bool,
        output_filename: str,
        verify_roundtrip: bool,
    ):
        src_path = Path(get_full_path("loras", lora_name))
        raw_state: Dict[str, torch.Tensor] = load_torch_file(str(src_path), safe_load=True)

        preset_norm = (block_preset or "auto").lower().replace("_", "-")
        if preset_norm == "auto":
            preset_norm = _infer_block_preset_from_lora_keys(raw_state.keys())

        model_blocks = _detect_total_blocks_from_model(model)
        lora_blocks = _detect_total_blocks_from_lora(raw_state, preset_norm)
        total_blocks = model_blocks if model_blocks > 0 else lora_blocks

        state_scaled, counters = _reweight_state_dict(
            raw_state,
            total_blocks,
            global_scale, front_scale, mid_scale, back_scale, last_block_scale,
            scale_target,
            filter_by_block_range,
            filter_cutoff_blocks=model_blocks,
            preset=preset_norm,
        )

        state_for_loader = state_scaled

        m = model.clone()
        c = clip.clone()
        m, c = comfy_sd.load_lora_for_models(m, c, state_for_loader, strength_model, strength_clip)

        saved_path = ""
        saved_sha = ""
        roundtrip_equal = None
        if save_reweighted:
            out_dir = _get_output_dir() / "loras"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = output_filename.strip() or _auto_name(
                src_path.name,
                scale_target,
                global_scale,
                front_scale,
                mid_scale,
                back_scale,
                last_block_scale,
            )
            save_path = out_dir / out_name
            save_file(
                state_for_loader,
                str(save_path),
                metadata={
                    "was_reweighted": "1",
                    "base_file": src_path.name,
                    "scale_target": scale_target,
                    "global_scale": f"{global_scale}",
                    "front_scale": f"{front_scale}",
                    "mid_scale": f"{mid_scale}",
                    "back_scale": f"{back_scale}",
                    "last_block_scale": f"{last_block_scale}",
                    "strength_model": f"{strength_model}",
                    "strength_clip": f"{strength_clip}",
                },
            )
            saved_path = str(save_path)
            saved_sha = _sha256(save_path)

            if verify_roundtrip:
                rt = load_file(str(save_path))

                keys_ok = (set(rt.keys()) == set(state_for_loader.keys()))
                all_ok = keys_ok
                if keys_ok:
                    for k in state_for_loader.keys():
                        a = state_for_loader[k]
                        b = rt[k]
                        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                                all_ok = False
                                break
                        else:
                            if a != b:
                                all_ok = False
                                break
                roundtrip_equal = bool(all_ok)

        stats = {
            "source": str(src_path),
            "block_preset": preset_norm,
            "total_blocks_detected": total_blocks,
            "model_blocks": model_blocks,
            "lora_blocks": lora_blocks,
            "changed": counters["changed"],
            "dropped_by_block_filter": counters["dropped"],
            "kept": counters["kept"],
            "strength_model": strength_model,
            "strength_clip": strength_clip,
            "global_scale": global_scale,
            "front_scale": front_scale,
            "mid_scale": mid_scale,
            "back_scale": back_scale,
            "last_block_scale": last_block_scale,
            "scale_target": scale_target,
            "saved_path": saved_path,
            "saved_sha256": saved_sha,
            "roundtrip_equal": roundtrip_equal,
        }
        return (m, c, stats)


NODE_CLASS_MAPPINGS = {
    "WASApplyReweightedLoRA": WASApplyReweightedLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WASApplyReweightedLoRA": "WAS Apply Reweighted LoRA",
}
