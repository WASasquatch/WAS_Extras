# WAS_Extras
Experimental nodes, or other random extra helper nodes. 

# Installation

- Git clone the repo to `ComfyUI/custom_nodes`
- Or; install with [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
- Or; download the individual custom node `.py` files and put the in your `ComfyUI/custom_nodes` folder.


## Conditioning (Blend)
Blend prompt conditioning by different interpolation/operation methods. 

## DebugThis
Debug any input. It will print the object, as well as display a mapping of it if it's not a common time (like `str`, `int`, `float`, etc).

## Inpainting VAE Encode
[**Download Inpainting_Example_Workflow_And_Images.zip**](https://github.com/WASasquatch/WAS_Extras/files/12719211/Inpainting_Example_Workflow_And_Images.zip)

This node allows for mask offset erosion, and dilation for fine-tuning th emask. It also uses the mask result as the pixel masking for cleaner boundaries and ability to erode (shrink) the mask. 

## Vivid Sharpen
A python implementation of the Photoshop Vivid Sharpen technique

![image](https://github.com/WASasquatch/WAS_Extras/assets/1151589/ebc3a81b-abf2-436e-aa2a-495522554c16)
