# Prompt Refiner for SD WebUI Forge Neo

Rewrite and expand image prompts using a supported text encoder you already have
in Forge Neo. The extension adds a **Prompt Refiner** tab and runs locally inside
Forge, without LM Studio, Ollama, or a separate language-model server.

## Install

In Forge Neo, open **Extensions > Install from URL**, paste this repository's URL,
install, and restart Forge. You can also clone the repository into
`<forge>/extensions/sd-webui-prompt-refiner/`.

Use an up-to-date **Forge Neo** installation. The extension reuses Forge's Python
packages; it does not install or upgrade them.

## Quick start

1. Place a supported `.safetensors` or `.gguf` text encoder in
   `<forge>/models/text_encoder/`. Subfolders, `models/text_encoders/`, and Forge's
   extra text encoder directory are also searched. Refresh the list after adding files.
2. Open **Prompt Refiner** and select the encoder. Leave **Model family** on **Auto**
   and the tokenizer directory empty unless setup requires an override.
3. Enter your idea, choose a system prompt preset, and click **Refine prompt**.
4. Review the result, then click **Send to txt2img** or **Send to img2img**.
   **Use result as input** lets you refine it again.

Expand **System prompt & presets** to edit instructions. **Save preset** updates
an existing name; change the name to save a copy. Presets can also be deleted,
and one backup of the previous version is kept.

Your model, preset selection, and options are remembered after restarting Forge.
Your idea and generated result are not saved. **Disable thinking via chat template**
is checked by default; dedicated thinking-only models cannot reliably disable reasoning.

## Supported encoders

| Encoder family | Common Forge Neo use |
| --- | --- |
| Qwen3 | Z-Image, Flux.2 Klein, Anima |
| Qwen3-VL | Krea2, Qwen-Image 2.1 |
| Qwen2.5-VL 7B | Qwen-Image / Qwen-Image-Edit |
| Qwen3.5 9B (dense) | MOSS-Image |

Supported layouts include BF16/FP16/FP32, scaled FP8, INT8/ConvRot, W4A8/W6A8,
INT4 ConvRot, NVFP4, and selected GGUF quantizations. Compatibility depends on the
actual model and weight layout. See the [technical reference](docs/TECHNICAL.md)
for exact formats, setup requirements, and testing coverage. MoE, CLIP, T5/UMT5,
and other model families are not supported.

## What to expect

- **Results depend on the encoder and instructions.** Instruct models usually suit
  rewriting better than base models. Review the result before generating an image;
  the encoder can add details or change the meaning of your idea.
- **Some encoders need an output head.** If Forge's encoder omits the weights needed
  to generate text, the status message explains what is missing. Expand **LM head**,
  select a known original model source, and download its matching head. This optional
  download is stored locally; do not substitute a head from a different model.
- **Speed and memory depend on your hardware and model.** The GPU path targets
  NVIDIA CUDA across different VRAM capacities. Choose an encoder that fits with
  room for inference; a small file does not guarantee low memory use or faster output.
  Quantization can reduce memory, but fast kernels vary by GPU and installed backend.
  CPU inference is available. AMD/ROCm is untested.
- **Models unload after refinement by default.** Enable **Keep model loaded** for
  repeated use if you have enough memory. **Model details** shows diagnostics and
  provides a manual unload button. Increase **Max new tokens** if output is cut short.

For tokenizer overrides, missing heads, memory troubleshooting, or choosing a
quantization, read the [technical reference](docs/TECHNICAL.md).

## Uninstall

1. Stop Forge Neo and delete `<forge>/extensions/sd-webui-prompt-refiner/`.
   This removes saved settings, presets, their backup, and tokenizers inside the
   extension. Back them up first if needed.
2. Delete downloaded LM heads in `<forge>/models/prompt_refiner/lm_heads/`.
   These are outside the extension folder and must be removed separately.
3. Remove any tokenizer or head files you placed elsewhere if no longer needed,
   then restart Forge.

Keep shared text encoders in Forge's model directories if you still use them for
image generation. No separate service or Python package uninstall is needed.
