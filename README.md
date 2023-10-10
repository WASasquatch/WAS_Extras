# WAS_Extras
Experimental nodes, or other random extra helper nodes. 

# Installation

- Git clone the repo to `ComfyUI/custom_nodes` and run the `requirements.txt` with your `/ComfyUI/python_embeded/python.exe` or env python.
- Or; install with [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
- Or; download the individual custom node `.py` files and put them in your `ComfyUI/custom_nodes` folder. **Note:** Still will need to install `requirements.txt` if using `ksampler_sequence.py`




## Conditioning (Blend)
Blend prompt conditioning by different interpolation/operation methods. 

## DebugThis
Debug any input. It will print the object, as well as display a mapping of it if it's not a common type (like `str`, `int`, `float`, etc).

## Inpainting VAE Encode
[**Download Inpainting_Example_Workflow_And_Images.zip**](https://github.com/WASasquatch/WAS_Extras/files/12719211/Inpainting_Example_Workflow_And_Images.zip)

This node allows for mask offset erosion, and dilation for fine-tuning th emask. It also uses the mask result as the pixel masking for cleaner boundaries and ability to erode (shrink) the mask. 

## Vivid Sharpen
A python implementation of the Photoshop Vivid Sharpen technique

![image](https://github.com/WASasquatch/WAS_Extras/assets/1151589/ebc3a81b-abf2-436e-aa2a-495522554c16)

## KSampler Sequence
A sequencer setup that allows you to sequence prompts in a looped Ksampler

### Provides
 -  CLIP Text Encode Sequence (Advanced)
    - **Note**: *CLIPTextEncode will use BLK Advanced CLIP Text Encode if available.*
 -  KSampler Sequence

[KSampler_Sequence_Workflow.zip](https://github.com/WASasquatch/WAS_Extras/files/12840983/KSampler_Sequence_Workflow.zip)
![image](https://github.com/WASasquatch/WAS_Extras/assets/1151589/83624414-4de8-4dcc-bf9e-2a1d1a8a2b10)

### BLVAEEncode

A VAE Encode that can store the latent in the workflow for easy sharing, potentially? Or something.

Workflow Example:
![ComfyUI_02554_](https://github.com/WASasquatch/WAS_Extras/assets/1151589/1a4f33e2-fb7a-49c7-90eb-727ecbef8ff7)
