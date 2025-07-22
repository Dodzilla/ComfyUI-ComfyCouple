from nodes import ConditioningCombine, ConditioningSetMask

from .attention_couple import AttentionCouple

class ComfyCouple:

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
    ):
        # Use the provided masks directly
        conditioning_mask_first = ConditioningSetMask().append(positive_1, mask_2, "default", 1.0)[0]
        conditioning_mask_second = ConditioningSetMask().append(positive_2, mask_1, "default", 1.0)[0]

        positive_combined = ConditioningCombine().combine(conditioning_mask_first, conditioning_mask_second)[0]

        return AttentionCouple().attention_couple(model, positive_combined, negative, "Attention")

NODE_CLASS_MAPPINGS = {
    "Comfy Couple": ComfyCouple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfy Couple": "Comfy Couple",
}
