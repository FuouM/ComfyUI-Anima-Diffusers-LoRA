# ComfyUI-Anima-Diffusers-LoRA

A transparent utility patch for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that adds on-the-fly support for Diffusers/OneTrainer-format LoRAs for the **Anima** model.

This custom node functions as a silent startup patch, allowing the native ComfyUI **Load LoRA** nodes to read and apply Diffusers-format Anima LoRAs without manual conversion and with **zero performance impact** on inference.

## Why is this needed?

Diffusers and training frameworks like OneTrainer export Anima LoRAs using key structures that differ from the legacy formats expected by ComfyUI's default loader. This node patches ComfyUI's internal key mapper to support both formats dynamically, aligning with the logic proposed in [ComfyUI PR #14182](https://github.com/Comfy-Org/ComfyUI/pull/14182).

## Features

* **Zero Workflow Impact:** No new node blocks are added to your editor. Keep using your standard LoRA loading workflows.
* **Dual Format Support:** Automatically detects and loads both converted (ComfyUI-native) and original Diffusers-format LoRAs.
* **Zero Inference Impact:** Key translation occurs entirely during the LoRA loading/binding phase, keeping model execution fully native and fast.

## Installation

Clone this repository directly into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/FuouM/ComfyUI-Anima-Diffusers-LoRA.git
```
