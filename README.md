# Prompt Refiner (Forge Classic / Neo)

Adds a **Prompt Refiner** tab that rewrites prompts with a local LLM - using the same
Qwen3 / Qwen3-VL text encoder file you already use for image generation with Krea 2, Flux.2 Klein or Z-Image.  
No Ollama / LM Studio required.

## Install
In Forge Neo open **Extension** tab -> **Install from URL** and enter URL of this repository.  
Or clone the repository manually into `<forge>/extensions/` and restart Forge.

## Use
1. Put a Qwen3 / Qwen3-VL text encoder (e.g. `qwen3vl_4b_int8_convrot.safetensors`) in `models/text_encoder/`.
2. Open the **Prompt Refiner** tab, pick the encoder, type an idea, choose a system prompt preset, click *Refine prompt*.
3. *Send to txt2img / img2img* copies the result into the prompt box.

## Notes
- Supported: Qwen3 and Qwen3-VL text encoders (bf16 / fp16 / scaled fp8 / int8 / int8 ConvRot). Text only (no image input).
- Not supported yet: GGUF format, Qwen2.5-VL 7B (Qwen-Image), Qwen3-VL 8B (Qwen-Image 2.1), Qwen3.5-VL 9B (MOSS-Image).
- Support for other then Qwen models is not planned
- The encoder file must contain the language model weights; 4B models tie `lm_head` to the embeddings, so they work.
- By default the model is unloaded after each run to free VRAM.
- System prompt presets live in `presets.json` and are editable in the UI.
- I do not recommend using "thinking" models. They are slow - requiring 3-4 times more tokens and time than "instruct" models - yet the results are no better.
- If possible, use the int8 ConvRot format. It runs twice as fast as fp8.

## Low-VRAM mode (8 GB cards)
int8 / fp8 weights (incl. int8 ConvRot) are kept compressed in VRAM (~4.8 GB for Qwen3-VL 4B) instead of being expanded to bf16 (~8 GB).
A fused Triton kernel is used for single-token decoding when Triton is installed (self-checked on first use,
automatic fallback to plain PyTorch). The status line shows which path was used.

## Formats
- int8 ConvRot, scaled fp8 (Comfy), Qwen block-wise FP8 (`weight_scale_inv`, 128x128), bf16 / fp16.
- Block-wise FP8 is re-encoded to row-wise int8 on load so it stays small in VRAM.
- Optional "compress bf16/fp16 to int8" lets 8 GB cards run full-size files.
- Thinking models: `<think>` blocks are stripped from the result; raise "Max new tokens" for them (~2048).
