import torch

import comfy.utils
from comfy import model_management


@torch.inference_mode()
def dynamic_tiled_upscale_with_custom_feather(
    samples,
    function,
    tile_size=512,
    overlap=32,
    output_device="cpu",
    pbar=None,
    feather=0,
    target_height=None,
    target_width=None,
    resample_method="lanczos",
    device=None,
):
    if samples.ndim != 4:
        raise ValueError("Expected samples with shape [B, C, H, W].")

    if target_height is None or target_width is None:
        raise ValueError("target_height and target_width must be provided.")

    tile_size = int(tile_size)
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")

    overlap = max(0, int(overlap))
    if overlap >= tile_size:
        overlap = tile_size - 1 if tile_size > 1 else 0

    tile_step = tile_size - overlap if tile_size > overlap else tile_size

    if device is None:
        device = model_management.get_torch_device()

    # Keep the full input on CPU and only move tiles to GPU.
    samples = samples.to(output_device)
    batch_size, channels, in_height, in_width = samples.shape

    scale_y_global = float(target_height) / float(in_height)
    scale_x_global = float(target_width) / float(in_width)

    blended_output = None

    for batch_index in range(batch_size):
        source = samples[batch_index : batch_index + 1]

        output_accumulator = None
        weight_map = None

        y_position = 0
        while y_position < in_height:
            x_position = 0
            while x_position < in_width:
                y_end = min(y_position + tile_size, in_height)
                x_end = min(x_position + tile_size, in_width)

                # CPU -> GPU per-tile
                tile_source_cpu = source[:, :, y_position:y_end, x_position:x_end]
                tile_source = tile_source_cpu.to(device, non_blocking=False)

                tile_output_native = function(tile_source)

                if blended_output is None:
                    out_channels = tile_output_native.shape[1]
                    blended_output = torch.zeros(
                        (batch_size, out_channels, target_height, target_width),
                        device=output_device,
                        dtype=tile_output_native.dtype,
                    )

                if output_accumulator is None:
                    output_accumulator = torch.zeros(
                        (1, tile_output_native.shape[1], target_height, target_width),
                        device=output_device,
                        dtype=tile_output_native.dtype,
                    )
                    # Single-channel weight map (broadcast to all channels)
                    weight_map = torch.zeros(
                        (1, 1, target_height, target_width),
                        device=output_device,
                        dtype=tile_output_native.dtype,
                    )

                out_y_start = int(round(y_position * target_height / in_height))
                out_y_end = int(round(y_end * target_height / in_height))
                out_x_start = int(round(x_position * target_width / in_width))
                out_x_end = int(round(x_end * target_width / in_width))

                tile_target_height = max(1, out_y_end - out_y_start)
                tile_target_width = max(1, out_x_end - out_x_start)

                if (
                    tile_output_native.shape[2] != tile_target_height
                    or tile_output_native.shape[3] != tile_target_width
                ):
                    tile_output_gpu = comfy.utils.common_upscale(
                        tile_output_native,
                        tile_target_width,
                        tile_target_height,
                        resample_method,
                        "disabled",
                    )
                else:
                    tile_output_gpu = tile_output_native

                tile_output = tile_output_gpu.to(output_device)

                if feather is None or feather <= 0:
                    feather_pixels_y = int(round(overlap * scale_y_global))
                    feather_pixels_x = int(round(overlap * scale_x_global))
                else:
                    feather_pixels_y = int(feather)
                    feather_pixels_x = int(feather)

                mask = torch.ones(
                    (1, 1, tile_output.shape[2], tile_output.shape[3]),
                    device=output_device,
                    dtype=tile_output.dtype,
                )

                if feather_pixels_y > 0:
                    max_vertical = tile_output.shape[2] // 2
                    feather_pixels_y = min(feather_pixels_y, max_vertical)
                    for t in range(feather_pixels_y):
                        weight_value = float(t + 1) / float(feather_pixels_y)
                        row_start = t
                        row_end = t + 1
                        inv_row_start = tile_output.shape[2] - 1 - t
                        inv_row_end = tile_output.shape[2] - t
                        mask[:, :, row_start:row_end, :].mul_(weight_value)
                        mask[:, :, inv_row_start:inv_row_end, :].mul_(weight_value)

                if feather_pixels_x > 0:
                    max_horizontal = tile_output.shape[3] // 2
                    feather_pixels_x = min(feather_pixels_x, max_horizontal)
                    for t in range(feather_pixels_x):
                        weight_value = float(t + 1) / float(feather_pixels_x)
                        col_start = t
                        col_end = t + 1
                        inv_col_start = tile_output.shape[3] - 1 - t
                        inv_col_end = tile_output.shape[3] - t
                        mask[:, :, :, col_start:col_end].mul_(weight_value)
                        mask[:, :, :, inv_col_start:inv_col_end].mul_(weight_value)

                out_y_end = out_y_start + tile_output.shape[2]
                out_x_end = out_x_start + tile_output.shape[3]

                output_accumulator[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += (
                    tile_output * mask
                )
                weight_map[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += mask

                if pbar is not None:
                    pbar.update(1)

                del tile_output_gpu, tile_output_native, tile_source
                torch.cuda.empty_cache()

                x_position += tile_step
            y_position += tile_step

        safe_weight_map = torch.where(
            weight_map == 0.0,
            torch.ones_like(weight_map),
            weight_map,
        )

        blended_output[batch_index : batch_index + 1] = (
            output_accumulator / safe_weight_map
        )

        del output_accumulator, weight_map
        torch.cuda.empty_cache()

    return blended_output.to(output_device)


class WASTiledImageUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL", {}),
                "image": ("IMAGE", {}),
                "upscale_factor": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 16.0,
                        "step": 0.1,
                        "tooltip": "Final scale relative to input image size. Output resolution ~= input * upscale_factor.",
                    },
                ),
                "tile_size": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Tile size in input pixels. Larger tiles are faster but use more VRAM.",
                    },
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "Tile overlap in input pixels. Higher overlap reduces seams but increases compute.",
                    },
                ),
                "feather": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Feather width in output pixels for tile blending. 0 = auto from overlap.",
                    },
                ),
                "resample_method": (
                    [
                        "nearest-exact",
                        "bilinear",
                        "area",
                        "bicubic",
                        "lanczos",
                    ],
                    {
                        "default": "lanczos",
                        "tooltip": "Resampling kernel used to reach the final upscale_factor resolution.",
                    },
                ),
                "clear_comfy_memory": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled, will unload all model, and do a soft cache dump followed by a hard cache dump. Aggressively frees VRAM by unloading other models.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(
        self,
        upscale_model,
        image,
        upscale_factor,
        tile_size,
        overlap,
        feather,
        resample_method,
        clear_comfy_memory,
    ):
        device = model_management.get_torch_device()

        if clear_comfy_memory:
            try:
                if hasattr(model_management, "unload_all_models"):
                    model_management.unload_all_models()
                    print("[WASTiledImageUpscaleWithModel] Cleared all models")
                else:
                    print("[WASTiledImageUpscaleWithModel] unload_all_models not found")
                if hasattr(model_management, "soft_empty_cache"):
                    model_management.soft_empty_cache(True)
                    print("[WASTiledImageUpscaleWithModel] Cleared soft cache")
                else:
                    print("[WASTiledImageUpscaleWithModel] soft_empty_cache not found")
                try:
                    torch.cuda.empty_cache()
                except Exception as exception:
                    print(f"[WASTiledImageUpscaleWithModel] Failed to empty PyTorch cache: {exception}")
            except Exception as exception:
                print(f"[WASTiledImageUpscaleWithModel] Failed to unload all comfy models and cache: {exception}")

        scale_estimate = getattr(upscale_model, "scale", 4.0)
        element_size = image.element_size()

        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (
            tile_size * tile_size * 3
        ) * element_size * max(scale_estimate, 1.0) * 384.0
        memory_required += image.nelement() * element_size

        model_management.free_memory(memory_required, device)

        upscale_model.to(device)

        batch_size, in_h, in_w, _ = image.shape

        upscale_factor = float(upscale_factor)
        if upscale_factor < 1.0:
            upscale_factor = 1.0

        target_height = max(1, int(round(in_h * upscale_factor)))
        target_width = max(1, int(round(in_w * upscale_factor)))

        input_image = image.movedim(-1, -3).to("cpu")

        current_tile_size = int(tile_size)
        minimum_tile_size = 64

        upscale_result = None

        oom = True
        last_exception = None

        while oom:
            try:
                steps = input_image.shape[0] * comfy.utils.get_tiled_scale_steps(
                    input_image.shape[3],
                    input_image.shape[2],
                    tile_x=current_tile_size,
                    tile_y=current_tile_size,
                    overlap=overlap,
                )
                progress = comfy.utils.ProgressBar(steps)

                upscale_result = dynamic_tiled_upscale_with_custom_feather(
                    samples=input_image,
                    function=lambda a: upscale_model(a),
                    tile_size=current_tile_size,
                    overlap=overlap,
                    output_device="cpu",
                    pbar=progress,
                    feather=feather,
                    target_height=target_height,
                    target_width=target_width,
                    resample_method=resample_method,
                    device=device,
                )

                oom = False
            except model_management.OOM_EXCEPTION as exception:
                last_exception = exception
                current_tile_size //= 2
                if current_tile_size < minimum_tile_size:
                    upscale_model.to("cpu")
                    raise last_exception

        upscale_model.to("cpu")

        upscale_result = torch.clamp(upscale_result, min=0.0, max=1.0)
        upscale_result = upscale_result.movedim(-3, -1)

        return (upscale_result,)


NODE_CLASS_MAPPINGS = {
    "WASTiledImageUpscaleWithModel": WASTiledImageUpscaleWithModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WASTiledImageUpscaleWithModel": "Tiled Image Upscale (With Model)",
}
