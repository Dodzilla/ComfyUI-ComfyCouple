"""Anima LLLite bridge for the fleet's pinned ComfyUI and 40-block Nova.

Uses kohya-ss/ComfyUI-Anima-LLLite without modifying its module globals.
The mapping follows sparklingcoffee777's verified 28 -> 40 expansion.
GPL-3.0, distributed with ComfyUI-ComfyCouple.
"""
import logging
import re
import types

BASE_TO_NOVA = (0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19,
                20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38, 39)
BLOCK = re.compile(r"^(lllite_dit_blocks_)(\d+)(_.+)$")


def remap_modules(modules, source_count, host_count):
    if source_count == host_count:
        return modules
    if (source_count, host_count) != (28, 40):
        raise ValueError(f"Unsupported Anima LLLite block pairing: {source_count} -> {host_count}")
    reverse = {target: source for source, target in enumerate(BASE_TO_NOVA)}
    result = []
    for module in modules:
        match = BLOCK.match(module.lllite_name)
        if not match:
            raise ValueError(f"Unrecognized LLLite target: {module.lllite_name}")
        target = int(match[2])
        if target in reverse:
            module.lllite_name = f"{match[1]}{reverse[target]}{match[3]}"
            result.append(module)
    logging.info("[Furgen Anima LLLite] remapped %d modules from 28 to 40 blocks; inserted blocks excluded", len(result))
    return result


class FurgenAnimaLLLiteApply:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {
            "model": ("MODEL",), "lllite_name": (folder_paths.get_filename_list("controlnet"),),
            "image": ("IMAGE",),
            "strength": ("FLOAT", {"default": 1.0, "min": -10., "max": 10., "step": .01}),
            "start_percent": ("FLOAT", {"default": 0., "min": 0., "max": 1., "step": .001}),
            "end_percent": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": .001}),
            "preserve_wrapper": ("BOOLEAN", {"default": True}),
        }, "optional": {"mask": ("MASK",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders/Anima"

    def apply(self, model, lllite_name, image, strength, start_percent, end_percent,
              preserve_wrapper=True, mask=None):
        import folder_paths
        import nodes
        from safetensors import safe_open
        from comfy.ldm.anima.model import Anima
        if not isinstance(model.model.diffusion_model, Anima):
            raise ValueError("FurgenAnimaLLLiteApply requires an Anima model")
        path = folder_paths.get_full_path("controlnet", lllite_name)
        with safe_open(path, framework="pt", device="cpu") as weights:
            counts = {int(m[2]) for key in weights.keys() if (m := BLOCK.match(key.split('.')[0]))}
        if not counts:
            raise ValueError("Control checkpoint has no named Anima LLLite blocks")
        source_count = max(counts) + 1
        if counts != set(range(source_count)):
            raise ValueError("Partial or noncontiguous Anima control checkpoint")
        host_count = len(model.model.diffusion_model.blocks)
        if source_count != host_count and (source_count, host_count) != (28, 40):
            raise ValueError(f"Unsupported Anima control layout {source_count} -> {host_count}")
        upstream = nodes.NODE_CLASS_MAPPINGS.get("AnimaLLLiteApply_sdscripts")
        if upstream is None:
            raise RuntimeError("Pinned ComfyUI-Anima-LLLite bundle is missing")
        apply = upstream.apply
        scope = dict(apply.__globals__)
        original_type = scope["ControlNetLLLiteDiT"]
        original_load = scope["load_lllite_weights"]

        class MappedLLLite(original_type):
            def _create_modules(self, *args, **kwargs):
                return remap_modules(super()._create_modules(*args, **kwargs), source_count, host_count)

        def strict_load(lllite, filename, strict=False):
            return original_load(lllite, filename, strict=True)

        scope.update(ControlNetLLLiteDiT=MappedLLLite, load_lllite_weights=strict_load)
        # A function-local globals copy avoids modifying any other node or sampler.
        scoped_apply = types.FunctionType(apply.__code__, scope, apply.__name__, apply.__defaults__, apply.__closure__)
        return scoped_apply(upstream(), model, lllite_name, image, strength, start_percent,
                            end_percent, preserve_wrapper, mask)


NODE_CLASS_MAPPINGS = {"FurgenAnimaLLLiteApply": FurgenAnimaLLLiteApply}
NODE_DISPLAY_NAME_MAPPINGS = {"FurgenAnimaLLLiteApply": "Apply Anima LLLite (28/40 block compatible)"}
