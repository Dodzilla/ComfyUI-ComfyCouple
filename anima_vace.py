"""Anima VACE ControlNet (depth / canny) as a MODEL patch, mapped onto Nova's 40-block DiT.

Architecture and forward math follow TaihoC's sd-scripts trainer (networks/control_net_vace_anima.py) as ported to
ComfyUI-Advanced-ControlNet by PineCookie (fix/anima-vace-hardening, 18f1fc10, GPL-3.0). The control branch runs once
per model call inside a DIFFUSION_MODEL wrapper; forward hooks on the mapped DiT blocks add its hints. A 28-block
checkpoint lands on the 40-block host at the blocks our LLLite bridge uses (anima_lllite.BASE_TO_NOVA).
"""
import logging

import torch
import torch.nn as nn

BASE_TO_NOVA = (0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19,
                20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38, 39)
WRAPPER_KEY = "furgen_anima_vace"


def host_block(base_idx, trained_blocks, host_blocks):
    if trained_blocks == host_blocks:
        return base_idx
    if (trained_blocks, host_blocks) == (28, 40):
        return BASE_TO_NOVA[base_idx]
    raise ValueError(f"Unsupported Anima VACE block layout {trained_blocks} -> {host_blocks}")


def build_vace(sd, meta, operations):
    from comfy.ldm.cosmos.predict2 import Block, PatchEmbed

    channels = int(meta.get("vace.model_channels", "2048"))
    trained = int(meta.get("vace.num_blocks", "28"))
    every = int(meta.get("vace.block_every_n", "7"))
    if meta.get("vace.condition_strategy", "spaced") != "spaced" or meta.get("vace.use_after_proj", "true") != "true":
        raise ValueError("Only spaced VACE checkpoints with after_proj are supported")
    pad_mask = meta.get("vace.concat_padding_mask", "true") == "true"
    latent_ch = int(meta.get("vace.in_channels", "16"))

    class ControlBlock(nn.Module):
        def __init__(self, first):
            super().__init__()
            self.block = Block(x_dim=channels, context_dim=1024, num_heads=16, mlp_ratio=4.0, use_adaln_lora=True,
                               adaln_lora_dim=256, operations=operations)
            if first:
                self.before_proj = operations.Linear(channels, channels)
            self.after_proj = operations.Linear(channels, channels)

    class Vace(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = list(range(0, trained, every))
            self.trained = trained
            self.latent_ch = latent_ch
            self.pad_mask = pad_mask
            self.control_embedder = PatchEmbed(spatial_patch_size=2, temporal_patch_size=1,
                                               in_channels=latent_ch + (1 if pad_mask else 0), out_channels=channels,
                                               operations=operations)
            self.control_blocks = nn.ModuleList([ControlBlock(i == 0) for i in range(len(self.layers))])

        def forward(self, control_latent, x_base, emb, context, rope, adaln_lora, extra_pos, transformer_options):
            dt = control_latent.dtype
            x_base, emb, context = x_base.to(dt), emb.to(dt), context.to(dt)
            adaln_lora = adaln_lora.to(dt) if adaln_lora is not None else None
            extra_pos = extra_pos.to(dt) if extra_pos is not None else None
            if self.pad_mask and control_latent.shape[1] == self.latent_ch:
                control_latent = torch.cat([control_latent, torch.zeros_like(control_latent[:, :1])], dim=1)
            c = self.control_embedder(control_latent)
            if c.dtype == torch.float16:
                c, x_base = c.float(), x_base.float()
            kw = dict(emb_B_T_D=emb, crossattn_emb=context, rope_emb_L_1_1_D=rope.unsqueeze(1).unsqueeze(0),
                      adaln_lora_B_T_3D=adaln_lora, extra_per_block_pos_emb=extra_pos,
                      transformer_options=transformer_options)
            hints, stack = {}, []
            for i, (base_idx, cb) in enumerate(zip(self.layers, self.control_blocks)):
                if i == 0:
                    c = cb.before_proj(c.to(cb.before_proj.weight.dtype)).to(x_base.dtype) + x_base
                else:
                    c = stack.pop(-1)
                c = cb.block(c, **kw)
                skip = cb.after_proj(c.to(cb.after_proj.weight.dtype)).to(c.dtype)
                stack += [skip, c]
                hints[base_idx] = skip
            return hints

    vace = Vace()
    missing, unexpected = vace.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise ValueError(f"VACE checkpoint does not match: missing {missing[:4]}, unexpected {unexpected[:4]}")
    return vace


class FurgenAnimaVACEApply:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {
            "model": ("MODEL",), "vae": ("VAE",),
            "control_net_name": (folder_paths.get_filename_list("controlnet"),),
            "image": ("IMAGE",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders/Anima"

    def apply(self, model, vae, control_net_name, image, strength, start_percent, end_percent):
        import comfy.model_management as mm
        import comfy.model_patcher
        import comfy.ops
        import comfy.patcher_extension as pe
        import comfy.utils
        import folder_paths
        from comfy.ldm.anima.model import Anima
        from comfy.ldm.common_dit import pad_to_patch_size

        dit = model.model.diffusion_model
        if not isinstance(dit, Anima):
            raise ValueError("FurgenAnimaVACEApply requires an Anima model")
        if start_percent > end_percent:
            raise ValueError("start_percent must not exceed end_percent")
        path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
        sd, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        if not any(k.startswith("control_blocks.0.after_proj") for k in sd):
            raise ValueError(f"{control_net_name} is not an Anima VACE checkpoint")
        dtype = mm.unet_dtype()
        vace = build_vace(sd, meta or {}, comfy.ops.manual_cast).to(dtype)
        host = len(dit.blocks)
        targets = {b: host_block(b, vace.trained, host) for b in vace.layers}
        logging.info("[Furgen Anima VACE] %s: control blocks %s -> host %s of %d", control_net_name, vace.layers,
                     list(targets.values()), host)
        patcher = comfy.model_patcher.ModelPatcher(vace, load_device=mm.get_torch_device(), offload_device=mm.unet_offload_device())

        # The guide image is already at the canvas size in our graphs; VAE-encode it once, in the model's latent space.
        h8, w8 = image.shape[1] // 8 * 8, image.shape[2] // 8 * 8
        pixels = comfy.utils.common_upscale(image.movedim(-1, 1), w8, h8, "bilinear", "center").movedim(1, -1)
        latent = model.get_model_object("latent_format").process_in(vae.encode(pixels[:, :, :, :3]))
        sampling = model.get_model_object("model_sampling")
        sigma_hi, sigma_lo = sampling.percent_to_sigma(start_percent), sampling.percent_to_sigma(end_percent)

        def wrapper(executor, x, timesteps, context, *args, **kwargs):
            opts = kwargs.get("transformer_options", {})
            sigmas = opts.get("sigmas")
            if strength == 0 or (sigmas is not None and not (sigma_lo <= float(sigmas.max()) <= sigma_hi)):
                return executor(x, timesteps, context, *args, **kwargs)
            net = dit
            ctl = latent.to(device=x.device, dtype=dtype)
            if ctl.ndim == 4:
                ctl = ctl.unsqueeze(2)
            if ctl.shape[-2:] != x.shape[-2:]:
                ctl = torch.nn.functional.interpolate(ctl.squeeze(2), size=x.shape[-2:], mode="bilinear").unsqueeze(2)
            ctl = ctl.expand(x.shape[0], -1, -1, -1, -1)
            patch = (net.patch_temporal, net.patch_spatial, net.patch_spatial)
            xs, ctl = pad_to_patch_size(x, patch), pad_to_patch_size(ctl, patch)
            tokens, rope, extra_pos = net.prepare_embedded_sequence(xs, fps=None, padding_mask=None)
            t = timesteps.unsqueeze(1) if timesteps.ndim == 1 else timesteps
            emb, adaln_lora = net.t_embedder[1](net.t_embedder[0](t).to(tokens.dtype))
            emb = net.t_embedding_norm(emb)
            hints = vace(ctl, tokens, emb, context.to(dtype), rope, adaln_lora, extra_pos, opts)
            hooks = []
            try:
                for base_idx, hint in hints.items():
                    scaled = hint * strength
                    hooks.append(net.blocks[targets[base_idx]].register_forward_hook(
                        lambda _m, _a, out, h=scaled: out + h.to(out.dtype)))
                return executor(x, timesteps, context, *args, **kwargs)
            finally:
                for h in hooks:
                    h.remove()

        m = model.clone()
        m.set_additional_models(WRAPPER_KEY, [patcher])
        m.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, wrapper)
        return (m,)


NODE_CLASS_MAPPINGS = {"FurgenAnimaVACEApply": FurgenAnimaVACEApply}
NODE_DISPLAY_NAME_MAPPINGS = {"FurgenAnimaVACEApply": "Furgen Anima VACE ControlNet"}
