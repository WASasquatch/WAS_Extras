import os
import sys
import json
import re

from safetensors.torch import save_file
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm


import folder_paths
from comfy.utils import ProgressBar

from ..modules.cli import merge_loras_z
from nodes import LoraLoader

class WASProgress:
    def __init__(self, total: int, desc: str):
        self.total = int(total)
        self.comfy = ProgressBar(self.total)
        self.console = tqdm(
            total=self.total,
            desc=desc,
            unit="step",
            dynamic_ncols=True,
            miniters=1,
            mininterval=0.1,
            file=sys.stderr,
        )

    def set_total(self, total: int):
        self.total = int(total)
        try:
            self.console.total = self.total
        except Exception:
            pass

    def update_absolute(self, value: int):
        v = int(value)
        try:
            if hasattr(self.comfy, "update_absolute"):
                self.comfy.update_absolute(v, self.total)
            else:
                self.comfy.update(1)
        except Exception:
            pass

        try:
            delta = v - int(self.console.n)
            if delta > 0:
                self.console.update(delta)
                self.console.refresh()
        except Exception:
            pass

    def close(self):
        try:
            try:
                self.console.refresh()
            except Exception:
                pass
            self.console.close()
        except Exception:
            pass


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")


class FlexibleOptionalInputType(dict):
    def __init__(self, type, data: Optional[dict] = None):
        self.type = type
        self.data = data
        if self.data is not None:
            for k, v in self.data.items():
                self[k] = v

    def __getitem__(self, key):
        if self.data is not None and key in self.data:
            return self.data[key]
        return (self.type,)

    def __contains__(self, key):
        return True


class WASPowerLoraMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Base MODEL. The merged LoRA will be applied to this model after saving."}),
                "clip": ("CLIP", {"tooltip": "Base CLIP. The merged LoRA will be applied to this CLIP after saving."}),
                "output_filename": (
                    "STRING",
                    {
                        "default": "merged_lora.safetensors",
                        "tooltip": "Output filename (relative to ComfyUI models/loras). Must be a relative path. '.safetensors' is appended if missing.",
                    },
                ),
                "output_model_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Strength used when applying the newly-created LoRA to the output unet model.",
                    },
                ),
                "output_clip_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Strength used when applying the newly-created LoRA to the output clip model.",
                    },
                ),

                "mode": (
                    ["svd", "rebase", "add", "add-diff", "add-orth", "diff-export", "moe", "obfuscate", "block-mix"],
                    {
                        "default": "svd",
                        "tooltip": "Merge mode. svd=recompress merged delta via SVD; rebase=single-LoRA SVD recompress; add=exact linear stacking; add-diff=base + weighted diffs toward others; add-orth=base + orthogonalized contributions; diff-export=export only the difference between first two; moe=mixture-of-experts per module; obfuscate=stack-equivalent factor rebasis without SVD; block-mix=route modules to LoRA A or B using preset/recipe, merge via stack or svd.",
                    },
                ),
                "block_mix_recipe": (
                    [
                        "all_a",
                        "all_b",
                        "concept_a_style_b",
                        "concept_b_style_a",
                        "attn_a_ffn_b",
                        "attn_b_ffn_a",
                        "img_a_txt_b",
                        "img_b_txt_a",
                    ],
                    {
                        "default": "concept_a_style_b",
                        "tooltip": "block-mix mode only: routing recipe.",
                    },
                ),
            },
            "optional": FlexibleOptionalInputType(
                any_type,
                {
                    "options": (
                        "WAS_LORA_MERGE_OPTIONS",
                        {
                            "tooltip": "Optional advanced merge options dict (use a companion options node).",
                        },
                    ),
                },
            ),
            "hidden": {
                "was_lora_catalog": (["None"] + folder_paths.get_filename_list("loras"),),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "lora_path")
    FUNCTION = "merge"
    CATEGORY = "WAS Extras"

    def merge(
        self,
        model,
        clip,
        output_filename: str,
        output_model_strength: float,
        output_clip_strength: float,
        mode: str,
        block_mix_recipe: str,
        options: Any = None,
        **kwargs: Any,
    ):
        loras: List[Tuple[str, float]] = []

        def _to_bool(v: Any, default: bool = True) -> bool:
            if v is None:
                return default
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(int(v))
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("true", "1", "yes", "y", "on"):
                    return True
                if s in ("false", "0", "no", "n", "off", ""):
                    return False
                return default
            return bool(v)

        def _coerce_payload(v: Any) -> Optional[dict]:
            if isinstance(v, dict):
                return v
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return None
                try:
                    obj = json.loads(s)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
            return None

        payload_rows: List[dict] = []
        flat_enabled: Dict[int, bool] = {}
        flat_lora: Dict[int, Optional[str]] = {}
        flat_weight: Dict[int, float] = {}

        for key, value in kwargs.items():
            if not isinstance(key, str):
                continue
            k = key.lower()
            if not (k.startswith("lora_") or k.startswith("lora_payload_")):
                continue

            if k.startswith("lora_payload_"):
                payload = _coerce_payload(value)
                if payload is not None:
                    payload_rows.append(payload)
                continue

            m_enabled = re.fullmatch(r"lora_(\d+)_enabled", k)
            if m_enabled:
                idx = int(m_enabled.group(1))
                flat_enabled[idx] = _to_bool(value, default=True)
                continue

            m_weight = re.fullmatch(r"lora_(\d+)_weight", k)
            if m_weight:
                idx = int(m_weight.group(1))
                try:
                    flat_weight[idx] = float(value)
                except Exception:
                    flat_weight[idx] = 1.0
                continue

            m_lora = re.fullmatch(r"lora_(\d+)", k)
            if m_lora:
                idx = int(m_lora.group(1))
                if isinstance(value, str) and value and value != "None":
                    flat_lora[idx] = value
                else:
                    flat_lora[idx] = None
                continue

        if payload_rows:
            for payload in payload_rows:
                on = _to_bool(payload.get("on", True), default=True)
                lora_name = payload.get("lora", None)
                weight = float(payload.get("weight", 1.0))

                if not on:
                    continue
                if lora_name is None or lora_name == "" or lora_name == "None":
                    continue
                if weight == 0.0:
                    continue

                full_path = folder_paths.get_full_path("loras", lora_name)
                if full_path is None:
                    raise ValueError(f"LoRA not found: {lora_name}")
                loras.append((full_path, weight))
        else:
            all_indices = sorted(set(flat_enabled.keys()) | set(flat_lora.keys()) | set(flat_weight.keys()))
            for idx in all_indices:
                on = _to_bool(flat_enabled.get(idx, True), default=True)
                lora_name = flat_lora.get(idx, None)
                weight = float(flat_weight.get(idx, 1.0))

                if not on:
                    continue
                if lora_name is None or lora_name == "" or lora_name == "None":
                    continue
                if weight == 0.0:
                    continue

                full_path = folder_paths.get_full_path("loras", lora_name)
                if full_path is None:
                    raise ValueError(f"LoRA not found: {lora_name}")
                loras.append((full_path, weight))

        if not loras:
            raise ValueError("At least one LoRA must be provided.")

        progress = None
        progress_step = 0

        def progress_cb(stage: str, current: int, total: int, message: str | None = None):
            nonlocal progress, progress_step

            if progress is None:
                progress = WASProgress(1, desc="WAS LoRA Merger")

            stage_total = max(int(total), 1)
            stage_current = max(0, min(int(current), stage_total))

            try:
                progress.update_absolute(progress_step + stage_current)
            except Exception:
                pass

        if mode == "rebase" and len(loras) != 1:
            raise ValueError("rebase mode requires exactly one LoRA.")

        if mode == "block-mix" and len(loras) != 2:
            raise ValueError("block-mix mode requires exactly two LoRAs (A and B).")

        if mode in ("add-diff", "add-orth", "diff-export") and len(loras) < 2:
            raise ValueError(f"{mode} mode requires at least two LoRAs.")

        opt = options if isinstance(options, dict) else {}

        def _opt_bool(key: str, default: bool) -> bool:
            return _to_bool(opt.get(key, default), default=default)

        def _opt_int(key: str, default: int) -> int:
            try:
                return int(opt.get(key, default))
            except Exception:
                return int(default)

        def _opt_float(key: str, default: float) -> float:
            try:
                return float(opt.get(key, default))
            except Exception:
                return float(default)

        def _opt_str(key: str, default: str) -> str:
            v = opt.get(key, default)
            if v is None:
                return str(default)
            return str(v)

        rank = _opt_int("rank", 32)
        auto_rank_threshold = _opt_float("auto_rank_threshold", 0.99)
        preserve_norm = _opt_bool("preserve_norm", False)
        cap_mult_enable = _opt_bool("cap_mult_enable", False)
        cap_mult = _opt_float("cap_mult", 1.0)
        dtype = _opt_str("dtype", "bf16")
        compute_dtype = _opt_str("compute_dtype", "auto")
        cpu = _opt_bool("cpu", False)
        include_patterns = _opt_str("include_patterns", "")
        exclude_patterns = _opt_str("exclude_patterns", "")
        moe_temperature = _opt_float("moe_temperature", 1.0)
        moe_hard = _opt_bool("moe_hard", False)
        block_mix_method = _opt_str("block_mix_method", "svd")
        block_mix_preset = _opt_str("block_mix_preset", "auto")

        include_list = [p.strip() for p in include_patterns.split(",") if p.strip()]
        exclude_list = [p.strip() for p in exclude_patterns.split(",") if p.strip()]

        device = merge_loras_z.get_device(force_cpu=cpu)
        out_dtype = merge_loras_z.get_dtype(dtype)
        explicit_compute_dtype = merge_loras_z.get_compute_dtype(compute_dtype)

        loaded: List[Tuple[Dict[str, merge_loras_z.LoraPair], float]] = []
        ref_meta: Dict[str, str] = {}

        for idx, (path, weight) in enumerate(loras):
            pairs, meta = merge_loras_z.load_lora_pairs(path, device=device, progress_cb=progress_cb)
            if idx == 0:
                ref_meta = meta if isinstance(meta, dict) else {}
            loaded.append((pairs, weight))

            try:
                stats = merge_loras_z.summarize_pairs(
                    pairs,
                    weight=weight,
                    explicit_compute_dtype=explicit_compute_dtype,
                    include_patterns=include_list,
                    exclude_patterns=exclude_list,
                )
                print(
                    "WASPowerLoraMerger Report(input): "
                    + json.dumps(
                        {
                            "path": path,
                            "weight": float(weight),
                            "mode": mode,
                            "stats": stats,
                        },
                        indent=2,
                    )
                )
            except Exception as e:
                print(f"WASPowerLoraMerger Report(input) failed: {e}")

        all_prefixes = set()
        for pairs, _w in loaded:
            all_prefixes.update(pairs.keys())
        module_count = max(1, len(all_prefixes))

        progress_total = (module_count * 2) + 5
        if progress is None:
            progress = WASProgress(progress_total, desc="WAS LoRA Merger")
        else:
            progress.set_total(int(progress_total))
            try:
                progress.comfy = ProgressBar(progress.total)
            except Exception:
                pass

        progress_step = 0
        progress.update_absolute(progress_step)

        def merge_progress_cb(stage: str, current: int, total: int, message: str | None = None):
            nonlocal progress_step
            if progress is None:
                return
            t = max(int(total), 1)
            c = max(0, min(int(current), t))
            abs_value = min(module_count, int(round((c / t) * module_count)))
            progress.update_absolute(abs_value)

        if mode == "add":
            merged_pairs = merge_loras_z.merge_mode_add(loaded, include_list, exclude_list, explicit_compute_dtype, progress_cb=merge_progress_cb)
        elif mode == "block-mix":
            try:
                a_pairs, _a_weight = loaded[0]
                b_pairs, _b_weight = loaded[1]
                routing = merge_loras_z.block_mix_routing_report(
                    a_pairs=a_pairs,
                    b_pairs=b_pairs,
                    preset=block_mix_preset,
                    recipe=block_mix_recipe,
                    include_patterns=include_list,
                    exclude_patterns=exclude_list,
                )
                print("WASPowerLoraMerger Report(block-mix.routing): " + json.dumps(routing, indent=2))
            except Exception as e:
                print(f"WASPowerLoraMerger Report(block-mix.routing) failed: {e}")

            weighted = bool(opt.get("block_mix_weighted", False))
            if weighted:
                concept_mix = float(opt.get("block_mix_concept_mix", 0.5))
                style_mix = float(opt.get("block_mix_style_mix", 0.5))
                merged_pairs = merge_loras_z.merge_mode_block_mix_weighted(
                    loaded,
                    method=block_mix_method,
                    rank_value=rank,
                    preset=block_mix_preset,
                    recipe=block_mix_recipe,
                    concept_mix=concept_mix,
                    style_mix=style_mix,
                    include_patterns=include_list,
                    exclude_patterns=exclude_list,
                    auto_rank_threshold=auto_rank_threshold,
                    explicit_compute_dtype=explicit_compute_dtype,
                    progress_cb=merge_progress_cb,
                )
            else:
                merged_pairs = merge_loras_z.merge_mode_block_mix(
                    loaded,
                    method=block_mix_method,
                    rank_value=rank,
                    preset=block_mix_preset,
                    recipe=block_mix_recipe,
                    include_patterns=include_list,
                    exclude_patterns=exclude_list,
                    auto_rank_threshold=auto_rank_threshold,
                    explicit_compute_dtype=explicit_compute_dtype,
                    progress_cb=merge_progress_cb,
                )
        elif mode == "add-diff":
            merged_pairs = merge_loras_z.merge_mode_add_diff(
                loaded,
                rank_value=rank,
                include_patterns=include_list,
                exclude_patterns=exclude_list,
                auto_rank_threshold=auto_rank_threshold,
                explicit_compute_dtype=explicit_compute_dtype,
                progress_cb=merge_progress_cb,
            )
        elif mode == "add-orth":
            merged_pairs = merge_loras_z.merge_mode_add_orth(
                loaded,
                rank_value=rank,
                include_patterns=include_list,
                exclude_patterns=exclude_list,
                auto_rank_threshold=auto_rank_threshold,
                explicit_compute_dtype=explicit_compute_dtype,
                progress_cb=merge_progress_cb,
            )
        elif mode == "diff-export":
            merged_pairs = merge_loras_z.merge_mode_diff_export(
                loaded,
                rank_value=rank,
                include_patterns=include_list,
                exclude_patterns=exclude_list,
                auto_rank_threshold=auto_rank_threshold,
                explicit_compute_dtype=explicit_compute_dtype,
                progress_cb=merge_progress_cb,
            )
        elif mode == "moe":
            merged_pairs = merge_loras_z.merge_mode_moe(
                loaded,
                rank_value=rank,
                moe_temperature=moe_temperature,
                moe_hard=moe_hard,
                include_patterns=include_list,
                exclude_patterns=exclude_list,
                auto_rank_threshold=auto_rank_threshold,
                explicit_compute_dtype=explicit_compute_dtype,
                progress_cb=merge_progress_cb,
            )
        elif mode == "obfuscate":
            merged_pairs = merge_loras_z.merge_mode_obfuscate(
                loaded,
                include_patterns=include_list,
                exclude_patterns=exclude_list,
                explicit_compute_dtype=explicit_compute_dtype,
                progress_cb=merge_progress_cb,
            )
        elif mode == "rebase":
            merged_pairs = merge_loras_z.merge_mode_rebase(
                loaded[0],
                rank_value=rank,
                include_patterns=include_list,
                exclude_patterns=exclude_list,
                auto_rank_threshold=auto_rank_threshold,
                explicit_compute_dtype=explicit_compute_dtype,
                progress_cb=merge_progress_cb,
            )
        else:
            merged_pairs = merge_loras_z.merge_mode_svd(
                loaded,
                rank_value=rank,
                preserve_norm=preserve_norm,
                cap_mult=(cap_mult if cap_mult_enable else None),
                include_patterns=include_list,
                exclude_patterns=exclude_list,
                auto_rank_threshold=auto_rank_threshold,
                explicit_compute_dtype=explicit_compute_dtype,
                progress_cb=merge_progress_cb,
            )

        def build_progress_cb(stage: str, current: int, total: int, message: str | None = None):
            nonlocal progress_step
            if progress is None:
                return
            t = max(int(total), 1)
            c = max(0, min(int(current), t))
            abs_value = module_count + min(module_count, int(round((c / t) * module_count)))
            progress.update_absolute(abs_value)

        state, meta = merge_loras_z.build_state_dict(
            merged_pairs,
            metadata_ref=ref_meta,
            dtype=out_dtype,
            device=device,
            progress_cb=build_progress_cb,
        )

        try:
            merged_stats = merge_loras_z.summarize_pairs(
                merged_pairs,
                weight=1.0,
                explicit_compute_dtype=explicit_compute_dtype,
                include_patterns=include_list,
                exclude_patterns=exclude_list,
            )
            print("WASPowerLoraMerger Report(merged): " + json.dumps(merged_stats, indent=2))
        except Exception as e:
            print(f"WASPowerLoraMerger Report(merged) failed: {e}")

        if progress is not None:
            progress.update_absolute((module_count * 2) + 1)

        if mode == "obfuscate":
            note_str = "Created by WAS Merge Loras"
        else:
            note_str = f"Created by WAS Merge Loras - Merging Mode: {mode}"

        merged_from = [(p, w) for p, w in loras] if mode != "obfuscate" else None

        meta.update(
            {
                "merged_from": str(merged_from),
                "mode": mode,
                "rank": str(rank),
                "dtype": str(out_dtype).replace("torch.", ""),
                "compute_dtype": (
                    str(explicit_compute_dtype).replace("torch.", "")
                    if explicit_compute_dtype is not None
                    else "auto(bf16-if-mixed)"
                ),
                "preserve_norm": str(preserve_norm),
                "cap_mult": str(cap_mult) if cap_mult_enable else "None",
                "include_patterns": str(include_list),
                "exclude_patterns": str(exclude_list),
                "moe_temperature": str(moe_temperature),
                "moe_hard": str(moe_hard),
                "auto_rank_threshold": str(auto_rank_threshold),
                "tool": "WAS Merge Loras",
                "note": note_str,
            }
        )

        if mode == "obfuscate":
            meta.pop("mode", None)
            meta.pop("merged_from", None)
            meta.pop("include_patterns", None)
            meta.pop("exclude_patterns", None)

        try:
            from folder_paths import folder_names_and_paths

            lora_dirs = folder_names_and_paths.get("loras", [[], []])[0]
            lora_dir = lora_dirs[0] if lora_dirs else None
        except Exception:
            lora_dir = None

        if not lora_dir:
            raise RuntimeError("Unable to resolve the LoRA output directory.")

        rel_out = (output_filename or "").strip().replace("\\\\", "/")
        while rel_out.startswith("/"):
            rel_out = rel_out[1:]
        if not rel_out:
            rel_out = "merged_lora.safetensors"
        if rel_out.startswith("../") or "/../" in rel_out or rel_out == "..":
            raise ValueError("output_filename must not contain '..' path traversal.")
        if ":" in rel_out:
            raise ValueError("output_filename must be a relative path under models/loras.")

        if not rel_out.lower().endswith(".safetensors"):
            rel_out = rel_out + ".safetensors"

        lora_root_abs = os.path.abspath(lora_dir)
        out_path = os.path.abspath(os.path.join(lora_root_abs, *rel_out.split("/")))
        if os.path.commonpath([lora_root_abs, out_path]) != lora_root_abs:
            raise ValueError("`output_filename` resolves outside models/loras directory.\nFor security reasons, LoRA cannot be saved to this location.")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        save_file(state, out_path, metadata=meta)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"Saved merged LoRA to {out_path}")

        model, clip = LoraLoader().load_lora(model, clip, rel_out, output_model_strength, output_clip_strength)

        if progress is not None:
            progress.update_absolute(progress.total)
            progress.close()

        return (model, clip, rel_out)


class WASPowerLoraMergerOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rank": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Rank for SVD-based modes (svd, rebase, add-diff, add-orth, diff-export, moe). Set to 0 to use auto-rank (energy threshold).",
                    },
                ),
                "auto_rank_threshold": (
                    "FLOAT",
                    {
                        "default": 0.99,
                        "min": 0.5,
                        "max": 1.0,
                        "step": 0.0001,
                        "tooltip": "Auto-rank energy threshold for SVD-based modes when rank=0. Higher values keep more singular-value energy (larger rank).",
                    },
                ),
                "preserve_norm": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "svd mode only: preserve average per-module delta norm after merging (helps prevent overall strength drift).",
                    },
                ),
                "cap_mult_enable": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "svd mode only: enable capping merged per-module norm (cap_mult × mean(source_norms)).",
                    },
                ),
                "cap_mult": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "svd mode only (when cap_mult_enable is on): cap merged module norm to cap_mult × mean(source module norms).",
                    },
                ),
                "dtype": (
                    ["fp16", "fp32", "bf16"],
                    {
                        "default": "bf16",
                        "tooltip": "Output dtype for the saved LoRA tensors.",
                    },
                ),
                "compute_dtype": (
                    ["auto", "bf16", "fp16", "fp32"],
                    {
                        "default": "auto",
                        "tooltip": "Internal merge dtype alignment. auto => if mixed dtypes are encountered, prefer bf16; otherwise use existing dtype.",
                    },
                ),
                "cpu": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Force CPU even if CUDA is available (slower but avoids VRAM usage).",
                    },
                ),
                "include_patterns": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional filter: only merge modules whose prefix contains any of these substrings. Comma-separated list (whitespace is trimmed).",
                    },
                ),
                "exclude_patterns": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional filter: exclude modules whose prefix contains any of these substrings. Comma-separated list (whitespace is trimmed).",
                    },
                ),
                "moe_temperature": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 1e-6,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "moe mode only: softmax temperature for expert gating (lower = sharper selection).",
                    },
                ),
                "moe_hard": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "moe mode only: hard gating (pick a single best expert per module) instead of soft mixture.",
                    },
                ),
                "block_mix_method": (
                    ["svd", "stack"],
                    {
                        "default": "svd",
                        "tooltip": "block-mix mode only: svd (delta merge then SVD) or stack (exact rank stacking).",
                    },
                ),
                "block_mix_preset": (
                    ["auto", "zimg-turbo", "flux", "wan", "qwen", "sd", "sdxl", "generic"],
                    {
                        "default": "auto",
                        "tooltip": "block-mix mode only: routing preset family.",
                    },
                ),
                "block_mix_weighted": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable weighted block-mix (blend A/B per module using role-specific mix ratios).",
                    },
                ),
                "block_mix_concept_mix": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "When weighted block-mix is enabled: fraction of LoRA A to use for concept/attention modules (B = 1 - A).",
                    },
                ),
                "block_mix_style_mix": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "When weighted block-mix is enabled: fraction of LoRA A to use for style/FFN modules (B = 1 - A).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("WAS_LORA_MERGE_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "build"
    CATEGORY = "WAS Extras"

    def build(
        self,
        rank: int,
        auto_rank_threshold: float,
        preserve_norm: bool,
        cap_mult_enable: bool,
        cap_mult: float,
        dtype: str,
        compute_dtype: str,
        cpu: bool,
        include_patterns: str,
        exclude_patterns: str,
        moe_temperature: float,
        moe_hard: bool,
        block_mix_method: str,
        block_mix_preset: str,
        block_mix_weighted: bool,
        block_mix_concept_mix: float,
        block_mix_style_mix: float,
    ):
        options = {
            "rank": int(rank),
            "auto_rank_threshold": float(auto_rank_threshold),
            "preserve_norm": bool(preserve_norm),
            "cap_mult_enable": bool(cap_mult_enable),
            "cap_mult": float(cap_mult),
            "dtype": str(dtype),
            "compute_dtype": str(compute_dtype),
            "cpu": bool(cpu),
            "include_patterns": str(include_patterns),
            "exclude_patterns": str(exclude_patterns),
            "moe_temperature": float(moe_temperature),
            "moe_hard": bool(moe_hard),
            "block_mix_method": str(block_mix_method),
            "block_mix_preset": str(block_mix_preset),
            "block_mix_weighted": bool(block_mix_weighted),
            "block_mix_concept_mix": float(block_mix_concept_mix),
            "block_mix_style_mix": float(block_mix_style_mix),
        }
        return (options,)


NODE_CLASS_MAPPINGS = {
    "WASPowerLoraMerger": WASPowerLoraMerger,
    "WASPowerLoraMergerOptions": WASPowerLoraMergerOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WASPowerLoraMerger": "WAS Power LoRA Merger",
    "WASPowerLoraMergerOptions": "WAS Power LoRA Merger Options",
}
