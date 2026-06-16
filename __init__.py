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

    for i in range(num_blocks):
        prefix_from = "transformer_blocks.{}".format(i)
        prefix_to = "{}blocks.{}".format(output_prefix, i)

        # attn1 is self attention, attn2 is cross attention
        for attn_from, attn_to in (("attn1", "self_attn"), ("attn2", "cross_attn")):
            block_map = {
                "{}.to_q.weight".format(attn_from): "{}.q_proj.weight".format(attn_to),
                "{}.to_k.weight".format(attn_from): "{}.k_proj.weight".format(attn_to),
                "{}.to_v.weight".format(attn_from): "{}.v_proj.weight".format(attn_to),
                "{}.to_out.0.weight".format(attn_from): "{}.output_proj.weight".format(
                    attn_to
                ),
            }
            for k, v in block_map.items():
                key_map["{}.{}".format(prefix_from, k)] = "{}.{}".format(prefix_to, v)

        key_map["{}.ff.net.0.proj.weight".format(prefix_from)] = (
            "{}.mlp.layer1.weight".format(prefix_to)
        )
        key_map["{}.ff.net.2.weight".format(prefix_from)] = (
            "{}.mlp.layer2.weight".format(prefix_to)
        )

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
                key_map[key_lora] = to

    return key_map


comfy.lora.model_lora_keys_unet = patched_model_lora_keys_unet

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
