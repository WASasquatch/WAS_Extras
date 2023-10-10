import hashlib
import torch

import nodes

class BLVAEEncode:
    def __init__(self):
        self.VAEEncode = nodes.VAEEncode()
        self.VAEEncodeTiled = nodes.VAEEncodeTiled()
        self.last_hash = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "tiled": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),
                "store_or_load_latent": ("BOOLEAN", {"default": True}),
                "remove_latent_on_load": ("BOOLEAN", {"default": True}),
                "delete_workflow_latent": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "image": ("IMAGE",),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("latent", )
    
    FUNCTION = "encode"
    CATEGORY = "latent"

    def encode(self, vae, tiled, tile_size, store_or_load_latent, remove_latent_on_load, delete_workflow_latent, image=None, extra_pnginfo=None, unique_id=None):

        workflow_latent = None
        latent_key = f"latent_{unique_id}"

        if self.last_hash and torch.is_tensor(image):
            if self.last_hash is not self.sha256(image):
                delete_workflow_latent = True
        if torch.is_tensor(image):
            self.last_hash = self.sha256(image)

        if delete_workflow_latent:
            if extra_pnginfo['workflow']['extra'].__contains__(latent_key):
                try:
                    del extra_pnginfo['workflow']['extra'][latent_key]
                except Exception:
                    print(f"Unable to delete latent image from workflow node: {unqiue_id}")
                    pass

        if store_or_load_latent and unique_id:
            if latent_key in extra_pnginfo['workflow']['extra']:
                print(f"Loading latent image from workflow node: {unique_id}")
                try:
                    workflow_latent = self.deserialize(extra_pnginfo['workflow']['extra'][latent_key])
                except Exception as e:
                    print("There was an issue extracting the latent tensor from the workflow. Is it corrupted?")
                    workflow_latent = None
                    if not torch.is_tensor(image):
                        raise ValueError(f"Node {unique_id}: There was no image provided, and workflow latent missing. Unable to proceed.")
                
                if workflow_latent and remove_latent_on_load:
                    try:
                        del extra_pnginfo['workflow']['extra'][latent_key]
                    except Exception:
                        pass

        if workflow_latent:
            print(f"Loaded workflow latent from node: {unique_id}")
            return workflow_latent, { "extra_pnginfo": extra_pnginfo }

        if not torch.is_tensor(image):
            raise ValueError(f"Node {unique_id}: No workflow latent was loaded, and no image provided to encode. Unable to proceed. ")

        if tiled:
            encoded = self.VAEEncodeTiled.encode(pixels=image, tile_size=tile_size, vae=vae)
        else:
            encoded = self.VAEEncode.encode(pixels=image, vae=vae)

        if store_or_load_latent and unique_id:
            print(f"Saving latent to workflow node {unique_id}")
            new_workflow_latent = self.serialize(encoded[0])
            extra_pnginfo['workflow']['extra'][latent_key] = new_workflow_latent

        return encoded[0], { "extra_pnginfo": extra_pnginfo }

    def sha256(self, tensor):
        tensor_bytes = tensor.cpu().contiguous().numpy().tobytes()
        hash_obj = hashlib.sha256()
        hash_obj.update(tensor_bytes)
        return hash_obj.hexdigest()

    def serialize(self, obj):
        if isinstance(obj, dict):
            return {key: self.serialize(value) for key, value in obj.items()}
        elif torch.is_tensor(obj):
            return {"type": "latent", "data": obj.tolist(), "shape": list(obj.shape)}
        else:
            return obj

    def deserialize(self, serialized_obj):
        if isinstance(serialized_obj, dict):
            if serialized_obj.get("type") == "latent":
                data = serialized_obj["data"]
                shape = serialized_obj["shape"]
                return torch.tensor(data).view(*shape)
            else:
                return {key: self.deserialize(value) for key, value in serialized_obj.items()}
        else:
            return serialized_obj

NODE_CLASS_MAPPINGS = {
    "BLVAEEncode": BLVAEEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BLVAEEncode": "VAEEncode (Bundle Latent)",
}
