import torch
import torch.nn.functional as F
import copy
import comfy
from comfy.ldm.modules.attention import optimized_attention

def get_masks_from_q(masks, q, original_shape):

    if original_shape[2] * original_shape[3] == q.shape[1]:
        down_sample_rate = 1
    elif (original_shape[2] // 2) * (original_shape[3] // 2) == q.shape[1]:
        down_sample_rate = 2
    elif (original_shape[2] // 4) * (original_shape[3] // 4) == q.shape[1]:
        down_sample_rate = 4
    else:
        down_sample_rate = 8

    ret_masks = []
    for mask in masks:
        if isinstance(mask,torch.Tensor):
            size = (original_shape[2] // down_sample_rate, original_shape[3] // down_sample_rate)
            mask_downsample = F.interpolate(mask.unsqueeze(0), size=size, mode="nearest")
            mask_downsample = mask_downsample.view(1,-1, 1).repeat(q.shape[0], 1, q.shape[2])
            ret_masks.append(mask_downsample)
        else: # coupling処理なしの場合
            ret_masks.append(torch.ones_like(q))
    
    ret_masks = torch.cat(ret_masks, dim=0)
    return ret_masks

def set_model_patch_replace(model, patch, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    to["patches_replace"]["attn2"][key] = patch

class AttentionCouple:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "mode": (["Attention", "Latent"], ),
            }
        }
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "attention_couple"
    CATEGORY = "loaders"

    def attention_couple(self, model, positive, negative, mode):
        print(f"[AttentionCouple] Starting attention_couple with mode: {mode}")
        print(f"[AttentionCouple] Input positive length: {len(positive)}")
        print(f"[AttentionCouple] Input negative length: {len(negative)}")
        
        if mode == "Latent":
            return (model, positive, negative) # latent coupleの場合は何もしない
        
        self.negative_positive_masks = []
        self.negative_positive_conds = []
        
        new_positive = copy.deepcopy(positive)
        new_negative = copy.deepcopy(negative)
        
        print(f"[AttentionCouple] After deepcopy - new_positive length: {len(new_positive)}")
        print(f"[AttentionCouple] After deepcopy - new_negative length: {len(new_negative)}")
        
        dtype = model.model.diffusion_model.dtype
        device = comfy.model_management.get_torch_device()
        
        print(f"[AttentionCouple] Device: {device}, dtype: {dtype}")
        
        # maskとcondをリストに格納する
        for cond_idx, conditions in enumerate([new_negative, new_positive]):
            cond_name = "negative" if cond_idx == 0 else "positive"
            print(f"[AttentionCouple] Processing {cond_name} conditioning, length: {len(conditions)}")
            
            conditions_masks = []
            conditions_conds = []
            
            if len(conditions) != 1:
                print(f"[AttentionCouple] {cond_name} has multiple conditions ({len(conditions)}), processing masks...")
                
                # Log each condition before stacking
                for i, cond in enumerate(conditions):
                    print(f"[AttentionCouple] {cond_name}[{i}] structure: {type(cond)}, length: {len(cond) if hasattr(cond, '__len__') else 'N/A'}")
                    if len(cond) > 1:
                        print(f"[AttentionCouple] {cond_name}[{i}] metadata keys: {cond[1].keys()}")
                        if 'mask' in cond[1]:
                            mask_shape = cond[1]['mask'].shape
                            print(f"[AttentionCouple] {cond_name}[{i}] mask shape: {mask_shape}")
                            print(f"[AttentionCouple] {cond_name}[{i}] mask device: {cond[1]['mask'].device}")
                            print(f"[AttentionCouple] {cond_name}[{i}] mask dtype: {cond[1]['mask'].dtype}")
                        if 'mask_strength' in cond[1]:
                            print(f"[AttentionCouple] {cond_name}[{i}] mask_strength: {cond[1]['mask_strength']}")
                
                try:
                    print(f"[AttentionCouple] Attempting to stack {cond_name} masks...")
                    mask_list = [cond[1]["mask"].to(device, dtype=dtype) * cond[1]["mask_strength"] for cond in conditions]
                    print(f"[AttentionCouple] Created mask_list with {len(mask_list)} items")
                    for i, mask in enumerate(mask_list):
                        print(f"[AttentionCouple] mask_list[{i}] shape: {mask.shape}")
                    
                    mask_norm = torch.stack(mask_list)
                    print(f"[AttentionCouple] Successfully stacked {cond_name} masks, result shape: {mask_norm.shape}")
                    
                    mask_norm = mask_norm / mask_norm.sum(dim=0) # 合計が1になるように正規化(他が0の場合mask_strengthの効果がなくなる)
                    print(f"[AttentionCouple] Normalized {cond_name} masks")
                    
                    conditions_masks.extend([mask_norm[i] for i in range(mask_norm.shape[0])])
                    conditions_conds.extend([cond[0].to(device, dtype=dtype) for cond in conditions])
                    
                    print(f"[AttentionCouple] {cond_name} conditions_masks length: {len(conditions_masks)}")
                    print(f"[AttentionCouple] {cond_name} conditions_conds length: {len(conditions_conds)}")
                    
                    del conditions[0][1]["mask"] # latent coupleの無効化のため
                    del conditions[0][1]["mask_strength"]
                    print(f"[AttentionCouple] Cleaned up {cond_name} metadata")
                    
                except Exception as e:
                    print(f"[AttentionCouple] ERROR stacking {cond_name} masks: {e}")
                    raise e
                    
            else:
                print(f"[AttentionCouple] {cond_name} has single condition, using fallback")
                conditions_masks = [False]
                conditions_conds = [conditions[0][0].to(device, dtype=dtype)]
                
            self.negative_positive_masks.append(conditions_masks)
            self.negative_positive_conds.append(conditions_conds)
            
        self.conditioning_length = (len(new_negative), len(new_positive))
        print(f"[AttentionCouple] Final conditioning_length: {self.conditioning_length}")

        new_model = model.clone()
        self.sdxl = hasattr(new_model.model.diffusion_model, "label_emb")
        print(f"[AttentionCouple] SDXL model: {self.sdxl}")
        
        if not self.sdxl:
            for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.input_blocks[id][1].transformer_blocks[0].attn2), ("input", id))
            set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.middle_block[1].transformer_blocks[0].attn2), ("middle", 0))
            for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.output_blocks[id][1].transformer_blocks[0].attn2), ("output", id))
        else:
            for id in [4,5,7,8]: # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.input_blocks[id][1].transformer_blocks[index].attn2), ("input", id, index))
            for index in range(10):
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.middle_block[1].transformer_blocks[index].attn2), ("middle", id, index))
            for id in range(6): # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.output_blocks[id][1].transformer_blocks[index].attn2), ("output", id, index))
        
        print(f"[AttentionCouple] Model patching complete")
        return (new_model, [new_positive[0]], [new_negative[0]]) # pool outputは・・・後回し
    
    def make_patch(self, module):
        def patch(q, k, v, extra_options):
            
            len_neg, len_pos = self.conditioning_length # negative, positiveの長さ
            cond_or_uncond = extra_options["cond_or_uncond"] # 0: cond, 1: uncond
            q_list = q.chunk(len(cond_or_uncond), dim=0)
            b = q_list[0].shape[0] # batch_size
            
            masks_uncond = get_masks_from_q(self.negative_positive_masks[0], q_list[0], extra_options["original_shape"])
            masks_cond = get_masks_from_q(self.negative_positive_masks[1], q_list[0], extra_options["original_shape"])

            # Handle potential shape mismatches in conditioning tensors
            def safe_concat_conds(conds, cond_type):
                if len(conds) == 1:
                    return conds[0]
                
                # Check if all tensors have the same shape
                shapes = [cond.shape for cond in conds]
                print(f"[AttentionCouple] {cond_type} conditioning shapes: {shapes}")
                
                if len(set(shapes)) > 1:
                    # Shapes don't match - need to handle this
                    print(f"[AttentionCouple] Shape mismatch detected in {cond_type} conditioning")
                    
                    # Find the maximum sequence length (dimension 1)
                    max_seq_len = max(cond.shape[1] for cond in conds)
                    print(f"[AttentionCouple] Max sequence length: {max_seq_len}")
                    
                    # Pad shorter tensors to match the longest
                    padded_conds = []
                    for i, cond in enumerate(conds):
                        if cond.shape[1] < max_seq_len:
                            # Pad with zeros
                            pad_size = max_seq_len - cond.shape[1]
                            padding = torch.zeros(cond.shape[0], pad_size, cond.shape[2], 
                                                dtype=cond.dtype, device=cond.device)
                            padded_cond = torch.cat([cond, padding], dim=1)
                            print(f"[AttentionCouple] Padded {cond_type}[{i}] from {cond.shape} to {padded_cond.shape}")
                            padded_conds.append(padded_cond)
                        else:
                            padded_conds.append(cond)
                    
                    return torch.cat(padded_conds, dim=0)
                else:
                    # All shapes match, safe to concatenate
                    return torch.cat(conds, dim=0)
            
            context_uncond = safe_concat_conds(self.negative_positive_conds[0], "uncond")
            context_cond = safe_concat_conds(self.negative_positive_conds[1], "cond")
            
            k_uncond = module.to_k(context_uncond)
            k_cond = module.to_k(context_cond)
            v_uncond = module.to_v(context_uncond)
            v_cond = module.to_v(context_cond)

            out = []
            for i, c in enumerate(cond_or_uncond):
                if c == 0:
                    masks = masks_cond
                    k = k_cond
                    v = v_cond
                    length = len_pos
                else:
                    masks = masks_uncond
                    k = k_uncond
                    v = v_uncond
                    length = len_neg

                q_target = q_list[i].repeat(length, 1, 1)
                k = torch.cat([k[i].unsqueeze(0).repeat(b,1,1) for i in range(length)], dim=0)
                v = torch.cat([v[i].unsqueeze(0).repeat(b,1,1) for i in range(length)], dim=0)
                
                if k.dtype != q_target.dtype or v.dtype != q_target.dtype:
                    # Ensure all dtypes match
                    k = k.to(q_target.dtype)
                    v = v.to(q_target.dtype)
                qkv = optimized_attention(q_target, k, v, extra_options["n_heads"])
                qkv = qkv * masks
                qkv = qkv.view(length, b, -1, module.heads * module.dim_head).sum(dim=0)

                out.append(qkv)

            out = torch.cat(out, dim=0)
            return out
        return patch
        
NODE_CLASS_MAPPINGS = {
    "Attention couple": AttentionCouple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Attention couple": "Load Attention couple",
}
