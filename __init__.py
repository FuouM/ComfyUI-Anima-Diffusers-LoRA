"""
@author: fuoumarinas
@title: ComfyUI Anima Diffusers LoRA
@nickname: ComfyUI-Anima-Diffusers-LoRA
@description: Transparent ComfyUI patch to load Diffusers-format LoRAs for Anima models.
"""

import comfy.lora
import comfy.model_base
import comfy.utils

print("### Patching ComfyUI for Anima Diffusers-format LoRA support (PR #14182) ###")


def custom_anima_to_diffusers(mmdit_config, output_prefix=""):
    num_blocks = mmdit_config.get("num_blocks", 0)
    key_map = {}

    top_level_map = {
        "patch_embed.proj": "x_embedder.proj.1",
        "time_embed.t_embedder": "t_embedder.1",
        "time_embed.norm": "t_embedding_norm",
        "norm_out.linear_1": "final_layer.adaln_modulation.1",
        "norm_out.linear_2": "final_layer.adaln_modulation.2",
        "proj_out": "final_layer.linear",
    }
    for k, v in top_level_map.items():
        key_map["{}.weight".format(k)] = "{}{}.weight".format(output_prefix, v)
        key_map["{}.bias".format(k)] = "{}{}.bias".format(output_prefix, v)

    for i in range(num_blocks):
        prefix_from = "transformer_blocks.{}".format(i)
        prefix_to = "{}blocks.{}".format(output_prefix, i)

        block_sub_map = {
            "norm1.linear_1": "adaln_modulation_self_attn.1",
            "norm1.linear_2": "adaln_modulation_self_attn.2",
            "attn1.norm_q": "self_attn.q_norm",
            "attn1.norm_k": "self_attn.k_norm",
            "attn1.to_q": "self_attn.q_proj",
            "attn1.to_k": "self_attn.k_proj",
            "attn1.to_v": "self_attn.v_proj",
            "attn1.to_out.0": "self_attn.output_proj",
            "norm2.linear_1": "adaln_modulation_cross_attn.1",
            "norm2.linear_2": "adaln_modulation_cross_attn.2",
            "attn2.norm_q": "cross_attn.q_norm",
            "attn2.norm_k": "cross_attn.k_norm",
            "attn2.to_q": "cross_attn.q_proj",
            "attn2.to_k": "cross_attn.k_proj",
            "attn2.to_v": "cross_attn.v_proj",
            "attn2.to_out.0": "cross_attn.output_proj",
            "norm3.linear_1": "adaln_modulation_mlp.1",
            "norm3.linear_2": "adaln_modulation_mlp.2",
            "ff.net.0.proj": "mlp.layer1",
            "ff.net.2": "mlp.layer2",
        }

        for k, v in block_sub_map.items():
            key_map["{}.{}.weight".format(prefix_from, k)] = "{}.{}.weight".format(prefix_to, v)
            key_map["{}.{}.bias".format(prefix_from, k)] = "{}.{}.bias".format(prefix_to, v)

    return key_map


# Expose it in comfy.utils in case something expects it there
if not hasattr(comfy.utils, "anima_to_diffusers"):
    comfy.utils.anima_to_diffusers = custom_anima_to_diffusers

# Save original model_lora_keys_unet
orig_model_lora_keys_unet = comfy.lora.model_lora_keys_unet


def patched_model_lora_keys_unet(model, key_map=None):
    if key_map is None:
        key_map = {}

    key_map = orig_model_lora_keys_unet(model, key_map)

    if isinstance(model, comfy.model_base.Anima):
        diffusers_keys = comfy.utils.anima_to_diffusers(
            model.model_config.unet_config, output_prefix="diffusion_model."
        )
        for k in diffusers_keys:
            if k.endswith(".weight"):
                to = diffusers_keys[k]
                key_lora = k[: -len(".weight")]
                key_map["diffusion_model.{}".format(key_lora)] = to
                key_map["transformer.{}".format(key_lora)] = to
                key_map["lycoris_{}".format(key_lora.replace(".", "_"))] = to
                key_map["lora_transformer_{}".format(key_lora.replace(".", "_"))] = to
                key_map[key_lora] = to

    return key_map


comfy.lora.model_lora_keys_unet = patched_model_lora_keys_unet

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
