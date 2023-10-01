import importlib
import time

extras = [
    ".ConditioningBlend",
    ".DebugThis",
    ".VAEEncodeForInpaint",
    ".VividSharpen",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
module_timings = {}

print("[\033[94m\033[1mWAS Extras\033[0m] Loading extra custom nodes...")

for module_name in extras:
    start_time = time.time()
    
    success = True
    try:
        module = importlib.import_module(module_name, package=__name__)
    except Exception:
        success = False
        pass
        
    end_time = time.time()
    timing = end_time - start_time
    
    module_timings[module.__file__] = (timing, success)

    NODE_CLASS_MAPPINGS.update(getattr(module, 'NODE_CLASS_MAPPINGS', {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, 'NODE_DISPLAY_NAME_MAPPINGS', {}))

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("[\033[94m\033[1mWAS Extras\033[0m] Import times for extras:")
for module, (timing, success) in module_timings.items():
    print(f"   {timing:.1f} seconds{('' if success else ' (IMPORT FAILED)')}: {module}")
