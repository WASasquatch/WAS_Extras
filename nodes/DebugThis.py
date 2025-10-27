from pprint import pprint

# Hack: string type that is always equal in not equal comparisons
# Borrowed from: https://github.com/M1kep/Comfy_KepListStuff/blob/main/utils.py
class AnyType(str): 
    def __ne__(self, __value): 
        return False

wildcard = AnyType("*")

class WAS_DebugThis:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"input": (wildcard, {"tooltip": "Any input to print/inspect in the console."})},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True

    FUNCTION = "debug"

    CATEGORY = "debug"

    def debug(self, input):
    
        print("Debug:")
        print(input)
        if isinstance(input, object) and not isinstance(input, (str, int, float, bool, list, dict, tuple)):
            print("Objects directory listing:")
            pprint(dir(input), indent=4)
		
        return ()

NODE_CLASS_MAPPINGS = {
    "DebugInput": WAS_DebugThis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DebugInput": "Debug Input",
}

