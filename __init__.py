import importlib
import traceback
import time

from cstr import cstr as cstr_instancer

cstr = cstr_instancer()
#! MESSAGE TEMPLATES
cstr.add_code("msg", f"{cstr.color.LIGHTBLUE}WAS Extras: {cstr.color.END}")
cstr.add_code("warning", f"{cstr.color.LIGHTBLUE}WAS Extras {cstr.color.LIGHTYELLOW}Warning: {cstr.color.END}")
cstr.add_code("error", f"{cstr.color.LIGHTRED}WAS Extras {cstr.color.END}Error: {cstr.color.END}")

extras = [
    ".ConditioningBlend",
    ".DebugThis",
    ".VAEEncodeForInpaint",
    ".VividSharpen",
    ".ksampler_sequence",
    ".BLVaeEncode"
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
module_timings = {}

cstr("Loading extra custom nodes...").msg.print()


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

cstr("Import times for extras:").msg.print()
for module, (timing, success, error) in module_timings.items():
    print(f"   {timing:.1f} seconds{('' if success else ' (IMPORT FAILED)')}: {module}")
    if error:
        print("Error:", error)
