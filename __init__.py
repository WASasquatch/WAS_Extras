import importlib
import traceback
import time

extras = [
    ".ConditioningBlend",
    ".DebugThis",
    ".VAEEncodeForInpaint",
    ".VividSharpen",
    ".VividSharpenV2",
    ".ksampler_sequence",
    ".BLVaeEncode"
]

PREFIX = '\33[31m\33[94m[WAS Extras]\33[0m '

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
module_timings = {}

print(f"{PREFIX}Loading extra custom nodes...")

for module_name in extras:
    start_time = time.time()

    success = True
    error = None
    try:
        module = importlib.import_module(module_name, package=__name__)
    except Exception as e:
        error = e
        success = False
        traceback.print_exc()

    end_time = time.time()
    timing = end_time - start_time

    if success:
        module_timings[module.__file__] = (timing, success, error)

    NODE_CLASS_MAPPINGS.update(getattr(module, 'NODE_CLASS_MAPPINGS', {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, 'NODE_DISPLAY_NAME_MAPPINGS', {}))


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"{PREFIX} Import times for extras:")
for module, (timing, success, error) in module_timings.items():
    print(f"   {timing:.1f} seconds{('' if success else ' (IMPORT FAILED)')}: {module}")
    if error:
        print("Error:", error)
