from nodes import ConditioningCombine, ConditioningSetMask
import torch
import torch.nn.functional as F

from .attention_couple import AttentionCouple

class LoopsComfyCouple:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive_1": ("CONDITIONING",),
                "positive_2": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "mask_1": ("MASK",),
                "mask_2": ("MASK",),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "CONDITIONING",
        "CONDITIONING",
    )

    FUNCTION = "process"
    CATEGORY = "loaders"

    def process(
            self,
            model,
            positive_1,
            positive_2,
            negative,
            mask_1,
            mask_2,
            debug=False,
    ):
        if debug:
            print(f"[ComfyCouple] Input mask_1 shape: {mask_1.shape}")
            print(f"[ComfyCouple] Input mask_2 shape: {mask_2.shape}")
            print(f"[ComfyCouple] Input positive_1 length: {len(positive_1)}")
            print(f"[ComfyCouple] Input positive_2 length: {len(positive_2)}")
            print(f"[ComfyCouple] Input negative length: {len(negative)}")
        
        # Ensure both masks have the same dimensions
        # Get the maximum dimensions from both masks
        h1, w1 = mask_1.shape[-2:]
        h2, w2 = mask_2.shape[-2:]
        
        if debug:
            print(f"[ComfyCouple] Original dimensions - mask_1: {h1}x{w1}, mask_2: {h2}x{w2}")
        
        target_h = max(h1, h2)
        target_w = max(w1, w2)
        
        if debug:
            print(f"[ComfyCouple] Target dimensions: {target_h}x{target_w}")
        
        # Resize masks to match the largest dimensions
        if mask_1.shape[-2:] != (target_h, target_w):
            if debug:
                print(f"[ComfyCouple] Resizing mask_1 from {mask_1.shape} to target {target_h}x{target_w}")
            mask_1 = F.interpolate(mask_1.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(0)
            if debug:
                print(f"[ComfyCouple] mask_1 after resize: {mask_1.shape}")
        
        if mask_2.shape[-2:] != (target_h, target_w):
            if debug:
                print(f"[ComfyCouple] Resizing mask_2 from {mask_2.shape} to target {target_h}x{target_w}")
            mask_2 = F.interpolate(mask_2.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(0)
            if debug:
                print(f"[ComfyCouple] mask_2 after resize: {mask_2.shape}")
        
        if debug:
            print(f"[ComfyCouple] Final mask_1 shape: {mask_1.shape}")
            print(f"[ComfyCouple] Final mask_2 shape: {mask_2.shape}")
        
        # Use the provided masks directly
        if debug:
            print(f"[ComfyCouple] Applying mask_2 to positive_1...")
        conditioning_mask_first = ConditioningSetMask().append(positive_1, mask_2, "default", 1.0)[0]
        if debug:
            print(f"[ComfyCouple] conditioning_mask_first length: {len(conditioning_mask_first)}")
            if len(conditioning_mask_first) > 0:
                print(f"[ComfyCouple] conditioning_mask_first[0] keys: {conditioning_mask_first[0][1].keys() if len(conditioning_mask_first[0]) > 1 else 'no metadata'}")
                if len(conditioning_mask_first[0]) > 1 and 'mask' in conditioning_mask_first[0][1]:
                    print(f"[ComfyCouple] conditioning_mask_first[0] mask shape: {conditioning_mask_first[0][1]['mask'].shape}")
        
        if debug:
            print(f"[ComfyCouple] Applying mask_1 to positive_2...")
        conditioning_mask_second = ConditioningSetMask().append(positive_2, mask_1, "default", 1.0)[0]
        if debug:
            print(f"[ComfyCouple] conditioning_mask_second length: {len(conditioning_mask_second)}")
            if len(conditioning_mask_second) > 0:
                print(f"[ComfyCouple] conditioning_mask_second[0] keys: {conditioning_mask_second[0][1].keys() if len(conditioning_mask_second[0]) > 1 else 'no metadata'}")
                if len(conditioning_mask_second[0]) > 1 and 'mask' in conditioning_mask_second[0][1]:
                    print(f"[ComfyCouple] conditioning_mask_second[0] mask shape: {conditioning_mask_second[0][1]['mask'].shape}")

        if debug:
            print(f"[ComfyCouple] Combining conditioning...")
        positive_combined = ConditioningCombine().combine(conditioning_mask_first, conditioning_mask_second)[0]
        if debug:
            print(f"[ComfyCouple] positive_combined length: {len(positive_combined)}")
            
            for i, cond in enumerate(positive_combined):
                print(f"[ComfyCouple] positive_combined[{i}] keys: {cond[1].keys() if len(cond) > 1 else 'no metadata'}")
                if len(cond) > 1 and 'mask' in cond[1]:
                    print(f"[ComfyCouple] positive_combined[{i}] mask shape: {cond[1]['mask'].shape}")

        if debug:
            print(f"[ComfyCouple] Calling AttentionCouple...")
        return AttentionCouple().attention_couple(model, positive_combined, negative, "Attention", debug)

NODE_CLASS_MAPPINGS = {
    "Loops Comfy Couple": LoopsComfyCouple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Loops Comfy Couple": "Loops Comfy Couple",
}
