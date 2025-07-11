import torch
import torch.nn.functional as F

class VividSharpenV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "radius_highpass": ("FLOAT", {
                    "default": 5.0, "min": 0.01, "max": 64.0, "step": 0.01,
                    "tooltip": "Radius for invert+blur high-pass step"
                }),
                "radius_blur": ("FLOAT", {
                    "default": 2.5, "min": 0.01, "max": 64.0, "step": 0.01,
                    "tooltip": "Radius for secondary blur of inverted image"
                }),
                "blur_mode": (["gaussian", "box"], {
                    "default": "gaussian",
                    "tooltip": "Kernel type for both blurs"
                }),
                "hp_brightness": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01,
                    "tooltip": "Brightness multiplier on high-pass layer"
                }),
                "hp_contrast": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01,
                    "tooltip": "Contrast factor on high-pass layer"
                }),
                "vivid_opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Opacity for vivid-light blend"
                }),
                "overlay_opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Opacity for overlay blend"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01,
                    "tooltip": "Mix ratio: original vs sharpened (can exceed 1.0)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "sharpen"
    CATEGORY = "image/postprocessing"

    @staticmethod
    def make_kernel(radius, mode, C):
        int_r = max(1, int(radius))
        k     = int_r * 2 + 1
        if mode == "box":
            kern = torch.ones((k, k), dtype=torch.float32)
        else:
            coords = torch.arange(k, dtype=torch.float32) - int_r
            g      = torch.exp(-coords**2 / (2 * radius**2))
            kern   = g[:, None] * g[None, :]
        kern = kern / kern.sum()
        return kern.view(1, 1, k, k).repeat(C, 1, 1, 1)

    @staticmethod
    def blur(x, radius, mode):
        N, C, H, W = x.shape
        k = VividSharpenV2.make_kernel(radius, mode, C).to(x.device)
        pad = k.shape[-1] // 2
        return F.conv2d(F.pad(x, (pad,) * 4, mode="reflect"), k, groups=C)

    @staticmethod
    def adjust_bc(x, brightness, contrast):
        # contrast around 0.5 pivot
        return ((x - 0.5) * contrast + 0.5) * brightness

    @staticmethod
    def alpha_blend(base, blend, op):
        return base * (1 - op) + blend * op if op < 1.0 else blend

    @staticmethod
    def vivid(A, B, op):
        b  = torch.where(B > 0, 1 - (1 - A) / (2 * B), torch.zeros_like(A))
        d  = torch.where(B < 1, A / (2 * (1 - B)), torch.ones_like(A))
        vl = torch.where(B <= 0.5, b, d).clamp(0, 1)
        return VividSharpenV2.alpha_blend(A, vl, op)

    @staticmethod
    def ovl(A, B, op):
        d1 = 2 * A * B
        d2 = 1 - 2 * (1 - A) * (1 - B)
        ov = torch.where(A <= 0.5, d1, d2)
        return VividSharpenV2.alpha_blend(A, ov, op)

    def sharpen(
        self,
        images,
        radius_highpass, radius_blur, blur_mode,
        hp_brightness, hp_contrast,
        vivid_opacity, overlay_opacity,
        strength
    ):
        # [N,H,W,C] → [N,C,H,W]
        x   = images.permute(0, 3, 1, 2)
        inv = 1.0 - x

        # high-pass
        hp  = VividSharpenV2.blur(inv, radius_highpass, blur_mode)
        hp  = VividSharpenV2.adjust_bc(hp, hp_brightness, hp_contrast).clamp(0, 1)

        # second blur
        hp2 = VividSharpenV2.blur(hp, radius_blur, blur_mode)

        # blends
        vl  = VividSharpenV2.vivid(x, hp2, vivid_opacity)
        ov  = VividSharpenV2.ovl(x, vl, overlay_opacity)

        # mix & return
        res = x * (1 - strength) + ov * strength
        res = res.clamp(0, 1)
        return (res.permute(0, 2, 3, 1),)


# register the node
NODE_CLASS_MAPPINGS = {
    "VividSharpenV2": VividSharpenV2,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VividSharpenV2": "Vivid Sharpen (V2)",
}
