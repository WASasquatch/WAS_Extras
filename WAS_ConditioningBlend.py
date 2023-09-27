import torch
import math

def normalize(latent, target_min=None, target_max=None):
    """
    Normalize a tensor `latent` between `target_min` and `target_max`.

    Args:
        latent (torch.Tensor): The input tensor to be normalized.
        target_min (float, optional): The minimum value after normalization. 
            - When `None` min will be tensor min range value.
        target_max (float, optional): The maximum value after normalization. 
            - When `None` max will be tensor max range value.

    Returns:
        torch.Tensor: The normalized tensor
    """
    min_val = latent.min()
    max_val = latent.max()
    
    if target_min is None:
        target_min = min_val
    if target_max is None:
        target_max = max_val
        
    normalized = (latent - min_val) / (max_val - min_val)
    scaled = normalized * (target_max - target_min) + target_min
    return scaled

def slerp(a, b, t):
    """
    Perform Spherical Linear Interpolation (SLERP) between two tensors.

    This function interpolates between two input tensors `a` and `b` using SLERP,
    which is a method for smoothly transitioning between orientations or vectors
    represented as tensors.

    Args:
        a (tensor): The first input tensor.
        b (tensor): The second input tensor.
        t (float): The blending factor, a value between 0 and 1 that controls the interpolation.

    Returns:
        tensor: The result of SLERP interpolation between `a` and `b`.

    Note:
        SLERP provides a smooth, shortest-path interpolation between two orientations or vectors
        represented as tensors. It's commonly used in applications like 3D graphics and robotics.
    """
    if a.shape != b.shape:
        raise ValueError("Input tensors a and b must have the same shape.")

    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)

    dot_product = torch.sum(a * b, dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(dot_product)

    slerp_result = (
        (a * torch.sin((1 - t) * angle) + b * torch.sin(t * angle)) /
        torch.sin(angle)
    )

    slerp_result = normalize(slerp_result)

    return slerp_result
    
def hslerp(a, b, t):
    """
    Perform Hybrid Spherical Linear Interpolation (HSLERP) between two tensors.

    This function combines two input tensors `a` and `b` using HSLERP, which is a specialized
    interpolation method for smooth transitions between orientations or colors.

    Args:
        a (tensor): The first input tensor.
        b (tensor): The second input tensor.
        t (float): The blending factor, a value between 0 and 1 that controls the interpolation.

    Returns:
        tensor: The result of HSLERP interpolation between `a` and `b`.

    Note:
        HSLERP provides smooth transitions between orientations or colors, particularly useful
        in applications like image processing and 3D graphics.
    """
    if a.shape != b.shape:
        raise ValueError("Input tensors a and b must have the same shape.")

    num_channels = a.size(1)
    
    interpolation_tensor = torch.zeros(1, num_channels, 1, 1, device=a.device, dtype=a.dtype)
    interpolation_tensor[0, 0, 0, 0] = 1.0

    result = (1 - t) * a + t * b

    if t < 0.5:
        result += (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor
    else:
        result -= (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor

    return result

import torch

blending_modes = {
    # Linearly combines the two input tensors a and b using the parameter t.
    'add': lambda a, b, t: (a * t + b * (1 - t)),

    # Interpolates between tensors a and b using normalized linear interpolation.
    'bislerp': lambda a, b, t: (a * (1 - t) + b * t),

    # Interpolates between tensors a and b using cosine interpolation.
    'cosine interp': lambda a, b, t: (a + b - (a - b) * torch.cos(t * torch.tensor(math.pi))) / 2,

    # Interpolates between tensors a and b using cubic interpolation.
    'cuberp': lambda a, b, t: a + (b - a) * (3 * t ** 2 - 2 * t ** 3),

    # Computes the absolute difference between tensors a and b, scaled by t.
    'difference': lambda a, b, t: (abs(a - b) * t),

    # Combines tensors a and b using an exclusion formula, scaled by t.
    'exclusion': lambda a, b, t: ((a + b - 2 * a * b) * t),

    # Interpolates between tensors a and b using normalized linear interpolation,
    # with a twist when t is greater than or equal to 0.5.
    'hslerp': lambda a, b, t: (a * (1 - t) + b * t) if t < 0.5 else (a * t + b * (1 - t)),

    # Adds tensor b to tensor a, scaled by t.
    'inject': lambda a, b, t: (a + b * t),

    # Interpolates between tensors a and b using linear interpolation.
    'lerp': lambda a, b, t: (a * (1 - t) + b * t),

    # Generates random values and combines tensors a and b with random weights, scaled by t.
    'random': lambda a, b, t: (a + (torch.rand_like(b) * b - a) * t),

    # Interpolates between tensors a and b using spherical linear interpolation (SLERP).
    'slerp': lambda a, b, t: (a * (1 - t) + b * t),

    # Subtracts tensor b from tensor a, scaled by t.
    'subtract': lambda a, b, t: (a * t - b * t),
}

class WAS_ConditioningBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_a": ("CONDITIONING", ),
                "conditioning_b": ("CONDITIONING", ),
                "blending_mode": (list(blending_modes.keys()), ),
                "blending_strength": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "combine"

    CATEGORY = "conditioning"

    def combine(self, conditioning_a, conditioning_b, blending_mode, blending_strength, seed):
    
        if seed > 0:
            torch.manual_seed(seed)
    
        a = conditioning_a[0][0].clone()
        b = conditioning_b[0][0].clone()
        
        pa = conditioning_a[0][1]["pooled_output"].clone()
        pb = conditioning_b[0][1]["pooled_output"].clone()

        cond = normalize(blending_modes[blending_mode](a, b, 1 - blending_strength))
        pooled = normalize(blending_modes[blending_mode](pa, pb, 1 - blending_strength))
        
        conditioning = [[cond, {"pooled_output": pooled}]]

        return (conditioning, )


NODE_CLASS_MAPPINGS = {
    "ConditioningBlend": WAS_ConditioningBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningBlend": "Conditioning (Blend)",
}
