import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Vivid Light and Overlay methods adopted from layeris (an overlooked gem)
# https://github.com/subwaymatch/layer-is-python
def vivid_light(A, B, opacity=1.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        b = np.where(B > 0, 1 - (1 - A) / (2 * B), 0)
        d = np.where(B < 1, A / (2 * (1 - B)), 1)

    result = np.clip(np.where(B <= 0.5, b, d), 0, 1)
    return alpha_blend(A, result, opacity)

def overlay(A, B, opacity=1.0):
    B = rgb_float_if_hex(B)
    d1 = (2 * A) * B
    d2 = 1 - 2 * (1 - A) * (1 - B)
    result = np.where(A <= 0.5, d1, d2)
    return alpha_blend(A, result, opacity)

def alpha_blend(base, blend, opacity):
    if opacity < 1.0:
        return base * (1.0 - opacity) + blend * opacity
    return blend

def hex_to_rgb_float(hex_string):
    return np.array(list((int(hex_string.lstrip('#')[i:i + 2], 16) / 255) for i in (0, 2, 4)))

def rgb_float_if_hex(blend_data):
    if isinstance(blend_data, str):
        return hex_to_rgb_float(blend_data)
    return blend_data

def vivid_sharpen(image, radius=5, strength=1.0):
    original = image.copy()
    sg = Image.new('RGB', original.size, (255, 255, 255))
    sg.paste(original, (0, 0))
    sg = ImageOps.invert(sg)
    sg = sg.filter(ImageFilter.GaussianBlur(radius=radius))

    original_data = np.array(original).astype(float) / 255.0
    sg_data = np.array(sg).astype(float) / 255.0

    result_data = vivid_light(original_data, sg_data, 1.0)
    result_data = overlay(original_data, result_data, 1.0)

    result_image = Image.fromarray((result_data * 255).astype('uint8'))
    result_image = Image.blend(original, result_image, strength)
    
    return result_image

class VividSharpen:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "radius": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 64.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "sharpen"

    CATEGORY = "image/postprocessing"

    def sharpen(self, images, radius, strength):
    
        results = []
        if images.size(0) > 1:
            for image in images:
                image = tensor2pil(image)
                results.append(pil2tensor(vivid_sharpen(image, radius=radius, strength=strength)))
            results = torch.cat(results, dim=0)
        else:
            results = pil2tensor(vivid_sharpen(tensor2pil(images), radius=radius, strength=strength))
            
        return (results,)

NODE_CLASS_MAPPINGS = {
    "VividSharpen": VividSharpen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VividSharpen": "VividSharpen",
}
