# WAS_Extras
Experimental nodes, or other random extra helper nodes. 

# Installation

- Git clone the repo to `ComfyUI/custom_nodes`.
- Or; install with [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
- Or; download the individual custom node `.py` files and put them in your `ComfyUI/custom_nodes` folder. **Note:** Still will need to install `requirements.txt` if using `ksampler_sequence.py`


## Nodes

<details>
 <summary><code>BLVaeEncode.py</code></summary>

### Provides
 - VAEEncode (Bundle Latent)

Encodes an image to a latent like ComfyUI’s normal VAE encode, but can also store that latent directly into the workflow’s embedded metadata.
This makes it easier to share workflows that include a “starting latent” without requiring a separate latent file, and it can optionally load the previously-stored latent back out on subsequent runs.
Includes an optional tiled encode mode to reduce VRAM usage on large images.

Workflow Example:
![ComfyUI_02554_](https://github.com/WASasquatch/WAS_Extras/assets/1151589/1a4f33e2-fb7a-49c7-90eb-727ecbef8ff7)
</details>

<details>
 <summary><code>CameraMotionTrajectorySimple.py</code></summary>

### Provides
 - Camera Motion Trajectory from Images

Generates a motion “camera path” over a sequence of images using a keyframed trajectory (JSON) describing pan/tilt, rotation, zoom, and optional radial effects.
Use it to create animated video clips (Ken Burns-style moves, rotations, etc.) from stills or frame sequences while keeping everything inside the ComfyUI graph.
Includes easing and interpolation so you can smoothly transition between keyframes.
</details>

<details>
 <summary><code>ConditioningBlend.py</code></summary>

### Provides
 - Conditioning (Blend)

Combines two `CONDITIONING` inputs into a single conditioning using a selectable blend algorithm (lerp, cosine, slerp variants, difference/exclusion, and more).
This is useful for prompt morphing, soft transitions between prompts, or “mixing” two prompt embeddings without needing to re-encode prompts elsewhere.
Some modes can be stochastic; a seed input is provided for reproducibility.
</details>

<details>
 <summary><code>DebugThis.py</code></summary>

### Provides
 - Debug Input

Debug/inspect any incoming value in a workflow.
Prints the value to the console and, for non-primitive objects, also prints a directory-style listing so you can discover available fields and methods.
Useful when building custom graphs that pass around complex data structures (options dicts, custom objects, etc.).
</details>

<details>
 <summary><code>ksampler_sequence.py</code></summary>

### Provides
 - CLIP Text Encode Sequence (Advanced)
   - **Note**: *CLIPTextEncode will use BLK Advanced CLIP Text Encode if available.*
 - CLIP Text Encode Sequence (v2)
 - KSampler Sequence
 - KSampler Sequence (v2)

A prompt + sampling sequencer for running multiple KSampler passes in a controlled loop.
You can define prompt schedules (including per-frame keyframes), repeat for a number of loops, evolve seeds, and optionally interpolate between latents across iterations to smooth transitions.
Intended for “prompt progression” workflows where each loop nudges the generation while keeping continuity.

[KSampler_Sequence_Workflow.zip](https://github.com/WASasquatch/WAS_Extras/files/12840983/KSampler_Sequence_Workflow.zip)
![image](https://github.com/WASasquatch/WAS_Extras/assets/1151589/83624414-4de8-4dcc-bf9e-2a1d1a8a2b10)
</details>

<details>
 <summary><code>TiledUpscaleModel.py</code></summary>

### Provides
 - Tiled Image Upscale (With Model)

Runs an upscale model (ESRGAN-style / Comfy upscalers) on large images by splitting the input into tiles and stitching the results back together.
Designed to reduce VRAM usage and avoid OOM errors by only sending one tile to the GPU at a time, with optional overlap/feathering to minimize seams.
Includes adaptive behavior that can reduce tile size if a tile upscaling attempt runs out of memory.
</details>

<details>
 <summary><code>VAEEncodeForInpaint.py</code></summary>

### Provides
 - Inpainting VAE Encode (WAS)

[**Download Inpainting_Example_Workflow_And_Images.zip**](https://github.com/WASasquatch/WAS_Extras/files/12719211/Inpainting_Example_Workflow_And_Images.zip)

Encodes an image to an inpainting latent while generating a matching `noise_mask`, with a mask-offset control for dilation/erosion.
The mask is resized and applied to pixels before encoding, which can produce cleaner edges and reduce boundary artifacts when inpainting.
Use it when you need more control over mask growth/shrink than the default inpaint encode path.
</details>

<details>
 <summary><code>VividSharpen.py</code></summary>

### Provides
 - VividSharpen

An image post-processing node implementing a “Vivid Sharpen” technique inspired by Photoshop workflows.
It constructs an inverted/blur high-pass style layer and blends it back to increase perceived detail while keeping a configurable radius and overall strength.
Useful as a final-pass crispness/detail enhancement after upscaling or before saving.

![image](https://github.com/WASasquatch/WAS_Extras/assets/1151589/ebc3a81b-abf2-436e-aa2a-495522554c16)
</details>

<details>
 <summary><code>VividSharpenV2.py</code></summary>

### Provides
 - Vivid Sharpen (V2)

A torch-only reimplementation of Vivid Sharpen with more granular controls.
Lets you tune two blur radii, choose blur kernel type, adjust the high-pass brightness/contrast, and independently control vivid-light and overlay blend opacities.
Supports strength values above 1.0 for more aggressive sharpening.
</details>

<details>
 <summary><code>WASEdgeSafeLatentUpscale.py</code></summary>

### Provides
 - WAS Adaptive Difference Latent Upscale (Damped)

Latent upscaling that focuses enhancement away from strong edges to reduce ringing/halos.
It computes an edge/energy-based damp mask (with optional temporal smoothing for video latents) and uses that to modulate how strongly upscaling differences are applied.
Outputs the upscaled latent plus a mask and a mask preview image so you can see exactly where the node is “protecting” detail.
</details>

<details>
 <summary><code>WASHybridLatentUpscale.py</code></summary>

### Provides
 - Latent Hybrid Upscale

Hybrid latent upscaling that blends a decoded image-space edge mask back into latent-space upscaling, or providing a donor latent with.
It uses OpenCV (if available) to build an edge mask (Canny + dilation + feathering) and uses that mask to guide how the upscale blends across edges.
Includes special handling for video latents and optional tiled decode settings to keep VRAM usage under control.
</details>

<details>
 <summary><code>WASLatentContrastLimitedDetailBoost.py</code></summary>

### Provides
 - WAS Latent Detail Boost

Adds micro-detail to latents using a contrast-limited band-pass (Difference-of-Gaussians) enhancement.
Includes RMS-based limiting and optional edge protection to reduce haloing and hard outlines, even at higher gain.
Outputs both a mask (detail or edge mask) and a preview image so you can dial in settings visually.
</details>

<details>
 <summary><code>WASLoraReweight.py</code></summary>

### Provides
 - WAS Apply Reweighted LoRA

Reweights a LoRA’s internal tensors based on detected block indices (supports presets like WAN/Qwen/Flux/SD/SDXL and an auto-detect mode).
This lets you bias a LoRA’s influence toward “front/mid/back” blocks, optionally limit it to a subset of blocks, and choose whether to scale LoRA up weights, down weights, or both.
Can optionally save the reweighted LoRA as a new `.safetensors` file and then apply it to the current MODEL/CLIP.
</details>

<details>
 <summary><code>WASPowerLoraMerger.py</code></summary>

### Provides
 - WAS Power LoRA Merger
 - WAS Power LoRA Merger Options

An advanced LoRA merging node that can combine multiple LoRAs into a single output LoRA saved into your `models/loras` folder.
Supports multiple merge strategies (including SVD recompression, additive stacking, diff-based modes, mixture-of-experts routing, and block-mix recipes) via ComfyUI.
Optionally takes an existing MODEL/CLIP and applies the newly-created LoRA immediately, so the merge can be part of a single workflow run.
</details>

<details>
 <summary><code>WASWanExposureStabilizer.py</code></summary>

### Provides
 - WAN 2.2 Exposure Stabilizer

Stabilizes exposure across an image batch (typically video frames) by measuring per-frame log-luma statistics and applying per-frame gain correction.
Designed to reduce WAN 2.2 style exposure drift at the start and/or end of a sequence, with configurable anchoring (middle window vs tail reference) and settle detection.
Returns the corrected images plus a human-readable report string with the computed exposure deltas and gains.
</details>

<details>
 <summary><code>WanMoESampler.py</code></summary>

### Provides
 - Wan 2.2 MoE Sampler (WAS)
 - Wan 2.2 MoE Conditioning Append (WAS)

Utilities and sampler logic intended for WAN 2.2 workflows that need more control than a single straight KSampler pass.
Includes a MoE-oriented sampler context node and a conditioning “append” helper for building lists of conditionings that can be routed/combined.
Also contains a set of transition and latent upscaling utilities for smoother multi-stage sampling pipelines.
</details>

<details>
 <summary><code>WASLUT.py</code></summary>

Load and combine LUTs, and visualize RGB waveform with RGB parade view.

Provides a small LUT pipeline: create/load LUTs (including `.cube` files), blend/combine multiple LUTs, apply LUTs to images, and export LUTs back to `.cube`.
Also includes a waveform/parade visualization node to help evaluate color and channel balance while grading.
Intended for quick color grading inside ComfyUI without round-tripping to external tools.

### Provides
 - WAS Load LUT
   - Create a custom LUT, load a preset, or load *.cube files from your `ComfyUI/models/LUT` folder.
 - WAS LUT Blender
   - Combine multiple LUTs into a single LUT.
 - WAS Apply LUT
   - Apply a LUT to an image.
 - WAS Save LUT (.cube)
 - WAS Channel Waveform (Parade)
   - Visualize the RGB waveform with RGB parade view.
</details>
