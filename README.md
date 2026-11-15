# Prompt Refiner (SD WebUI Forge Neo)

Adds a **Prompt Refiner** tab that rewrites image prompts using the language weights
of a text encoder you already use in Forge Neo. Runs inside Forge with local
PyTorch/Transformers inference; no Ollama, LM Studio, or separate server is needed.

## Install

In Forge Neo, open **Extensions → Install from URL**, enter this repository URL,
and restart Forge. Alternatively, clone it into `<forge>/extensions/`.
Use an up-to-date Neo installation with Qwen3 support in Transformers.
GGUF uses Neo's existing `gguf` package and GGUF tensor operations. W4A8 requires
Neo's Comfy-Kitchen version with `AsymW4A8Int8Layout` support. The extension does
not install or upgrade Forge's packages.

## Uninstall

1. Stop Forge Neo.
2. Delete the extension folder, normally
   `<forge>/extensions/sd-webui-prompt-refiner/`. This also removes saved presets
   (`presets.json`) and any tokenizers stored in its `tokenizer/<family>/` folders.
   Back up these files first if you want to keep them.
3. Delete downloaded output heads from
   `<forge>/models/prompt_refiner/lm_heads/`. These files are stored outside the
   extension and are not removed when its folder is deleted. You can remove
   `<forge>/models/prompt_refiner/` entirely if it contains only this extension's files.
4. If you supplied tokenizer directories or LM head files elsewhere, remove those
   separately when no longer needed.
5. Start Forge Neo again.

Text encoders in `models/text_encoder/`, `models/text_encoders/`, or a custom
text encoder directory are shared with image generation. Delete individual files
only if you no longer need them in Forge. The extension does not install a
separate service or additional Python packages, so no package uninstall is needed.

## Use

1. Put a supported `.safetensors` or `.gguf` encoder in `models/text_encoder/`.
   Subfolders, `models/text_encoders/`, and Forge's extra text encoder directory
   are also searched. Refresh the encoder list after adding files.
2. Select the encoder at the top. In **Advanced model settings**, keep
   **Model family** on **Auto**, or select the correct family for renamed/ambiguous
   checkpoints. The optional tokenizer directory is in the same collapsed section.
3. Enter an idea, select/edit the system prompt, and click **Refine prompt**.
4. **Send to txt2img / img2img** copies the result into the corresponding prompt.
   **Use result as input** supports another refinement pass.

The prompt and result are side by side, with refinement controls below the prompt
and transfer buttons below the result. Preset selection, **Max new tokens**, and
**Disable thinking via chat template** are available beside the prompt; disabling
thinking is checked by default for switchable Qwen3 models. Expand **System prompt
& presets** to edit instructions or manage presets. **Sampling options** and
**Advanced model settings** are collapsed by default. The short status appears
below the result; expand **Model details** for tokenizer, quantization, device,
and memory diagnostics or to unload the model manually.

The model is unloaded after each run by default. **Keep model loaded** avoids
reloading at the cost of occupied memory. **Unload model** releases it manually.
Loading from disk is reported whenever needed; retaining a model only avoids
reloading when the selected encoder and model settings still match.
Presets are editable in the UI and saved in `presets.json`.
An empty prompt or missing encoder produces a status message without loading a
model or raising an exception. The console logs skipped requests, the start of
refinement, completion/stopping, and failures; prompt text is not logged.

## Supported models

| Language model | Typical Neo image model | Notes |
| --- | --- | --- |
| Qwen3 | Z-Image, Flux.2 Klein, Anima | Known dense 0.6B / 1.7B / 4B / 8B / 14B / 32B profiles; base models may follow instructions poorly. |
| Qwen3-VL | Krea2, Qwen-Image 2.1 | Known dense 2B / 4B / 8B / 32B profiles; only language weights are loaded. |
| Qwen2.5-VL 7B | Qwen-Image / Qwen-Image-Edit | Preserves biased Q/K/V projections; requires its original output head if omitted. |

Inputs are text only. Vision weights and `mmproj` are not needed for refinement.
Qwen3.5 / MOSS-Image, MoE, Gemma, Ministral, CLIP, and T5/UMT5 are not supported
by this version. Selecting a file does not make an unsupported quantization work:
unknown layouts fail with an error before generation.

Prefer an instruction-tuned model for prompt rewriting. The extension cannot turn
base-model weights into an instruction-tuned model. Base checkpoints can work, but instruction following depends on the checkpoint
and chat template. Anima-identified Qwen3 files prefer Anima's own chat template
among Forge's bundled tokenizers. Base filenames produce an advisory status warning. **Disable thinking via chat
template** applies only to Qwen3; when needed, it adds the standard empty thinking
block. Completed and unfinished thinking blocks are hidden from displayed output. Thinking
models can require a larger output token budget. Dedicated Thinking-2507 models
are not switchable thinking/non-thinking models: the checkbox cannot reliably
disable their reasoning. For filenames identifying Thinking-2507, the extension
ignores that checkbox and reports the limitation in the status and console.
Prefer a standard or Instruct checkpoint for this task.

## Architecture configuration and tokenizers

Architecture dimensions are checked against the tensors. The model family comes
from an explicit selection, an adjacent Hugging Face config, GGUF architecture
metadata, or recognizable checkpoint signatures/names. Ambiguous names produce
an actionable error. Automatic RoPE uses the supplied config/GGUF metadata first,
then the family default (5M for Qwen3-VL and named Qwen3 2507 refreshes, 1M for
older Qwen3 and Qwen2.5-VL). The RoPE control remains available as an override.

For a renamed checkpoint or a configuration outside the known profiles, place its
original HF config beside it as `<encoder-stem>.config.json`; a neighboring
`config.json` is also accepted. Complete VL configs containing `text_config` work.
The config must match the tensors and specify `tie_word_embeddings` for unknown
profiles. Non-default RoPE scaling and sliding-window/hybrid attention configs are
rejected rather than silently ignored.

Tokenizer files are loaded locally; no tokenizer is downloaded automatically.
Selection order is:

1. **Tokenizer directory**, if explicitly entered.
2. A neighboring `<encoder-stem>.tokenizer/` or `tokenizer/` directory.
3. This extension's `tokenizer/qwen3/`, `tokenizer/qwen3_vl/`, or
   `tokenizer/qwen2_5_vl/` directory, according to the resolved family.
4. Matching tokenizer directories already shipped with Forge Neo: Flux.2 Klein
   for Qwen3, Krea2/Qwen-Image 2.1 for Qwen3-VL, Qwen-Image for Qwen2.5-VL.

Include `tokenizer_config.json` and either `tokenizer.json` or `vocab.json` plus
`merges.txt`, and any additional tokenizer assets from the same model.
Special token IDs and vocabulary bounds are checked. A supplied chat template is
used; Forge's encoding-only tokenizer configs fall back to standard text ChatML.
An explicit tokenizer override must still match the actual encoder.

## Safetensors formats and memory

- BF16 / FP16 / FP32 language weights.
- Plain or scaled FP8 E4M3 / E5M2, including mixed-precision files and
  `float8_e4m3fn_rowwise` / `float8_e5m2_rowwise` metadata (also `tensorwise`).
- Scalar, row, or explicit column scales; Comfy's `scale_weight` / `weight_scale`.
- HF block-wise FP8 with `weight_scale_inv`: fixed 128×128 blocks, including
  partial edge blocks. The stored value is the dequantization multiplier.
- Scaled INT8 and INT8 ConvRot with validated Hadamard group sizes.
- Neo W4A8 (`asym_w4a8_int8`): packed 4-bit weights, relative/group scales,
  channel scales, optional codebook/correction, and ConvRot metadata.
- Per-layer `.comfy_quant` metadata and header `_quantization_metadata` are read.
  Malformed metadata, missing INT8 scales, and unsupported layouts are rejected.

With **Keep weights compressed** enabled, row/tensor-scaled INT8 and FP8 linear
weights remain one byte per value. ConvRot rotates activations instead of expanding
weights. A self-checked Triton kernel accelerates eligible INT8 and row/tensor-scaled FP8 single-token
steps; PyTorch is the fallback. W4A8 keeps its original packed representation and
uses Comfy-Kitchen's backend, with a dense per-layer fallback if no kernel applies.
The status reports compressed layer counts, W4A8/GGUF counts, and INT8 conversion
counts. GEMV verification is separate for INT8, FP8 E4M3, and FP8 E5M2;
status reports only storage types present in the current model. This fused
single-token kernel reads compressed weights directly and does not use FP8
Tensor Core matrix multiplication. Prefill and unsupported devices use PyTorch.
W4A8 and GGUF use their own backend paths.

Block/column-scaled FP8 or INT8 linear weights that do not fit the resident
row-scale layout are decoded and converted to row-wise INT8 when compression is
on. This is another lossy quantization step. Disable **Keep weights compressed**
to decode into the chosen compute dtype without that conversion.
**Also compress bf16 / fp16 weights to int8** additionally converts dense linear
weights on load and only applies when **Keep weights compressed** is enabled.
Embeddings and normalization weights stay dense, so total memory is larger than
packed linear weights alone. Models must fit on the selected device; automatic
CPU/GPU layer offloading is not implemented.

NVFP4, MXFP8, ConvRot W4A4, W6A8, and unrelated INT4/AWQ/GPTQ layouts are not
supported by this version. A file's integer tensor dtype alone is not sufficient
to identify its quantization.

## GGUF

Single-file GGUF encoders with architectures `qwen3`, `qwen3vl`, or `qwen2vl`
(Qwen2.5-VL 7B) are accepted. Language tensor names are mapped to the corresponding
HF backbone; unknown language tensors are rejected. Use the matching local family
tokenizer as described above. Embedded GGUF tokenizer metadata is not imported.
Split/sharded GGUF sets must first be merged into one GGUF file outside the extension.

With compression enabled, linear weights stay packed using Neo's GGUF operations.
Only the active layer is dequantized for computation; the full model is not expanded
to BF16. Embeddings/norms are decoded on CPU during loading. Neo's baked linear
records use FP16 scales, even when the selected activation/compute dtype differs.
Resident record sizes can differ from file sizes. Each token still requires
dequantizing every active linear weight and running a dense PyTorch linear
operation; this is not a fused GGUF matrix-vector kernel. Different quantization
layouts have different decoding costs, so a smaller file need not generate faster.
Dense embeddings/output heads, temporary decoded layers, and the KV cache also
consume VRAM. If Windows spills GPU allocations into shared system memory,
generation can slow drastically even though loading succeeded. This path does not promise
llama.cpp performance and does not require llama.cpp or a separate process.

Regression-tested layouts are Q8_0, Q6_K, Q5_K, Q4_K, Q5_0, Q5_1, Q4_0, and Q4_1.
Q4_K_M/Q5_K_M files mix tensor types and are handled per tensor. F16/F32 and BF16
storage are also recognized. Other quantizations require support in Neo's GGUF
mapping and have not all been validated here; unsupported types fail explicitly.

## Missing LM heads

A diffusion encoder may omit `lm_head.weight`. Known Qwen3 0.6B / 1.7B / 4B and
Qwen3-VL 2B / 4B profiles tie this weight to embeddings and need no extra file.
Untied models, including Qwen2.5-VL 7B and Qwen3/VL 8B, need the exact original
output head. A model's hidden size alone is never used to infer weight tying.

Open **LM head**, enter the original checkpoint's Hugging Face repository, and
click **Download lm_head**. Only that tensor is fetched through HTTP range requests,
then stored in `models/prompt_refiner/lm_heads/`, as row-wise INT8 by default.
The token field is only needed for gated repositories. This is an explicit optional
weight download; inference itself stays local.

Choose the head explicitly. **Auto** uses an exact filename matching
`<encoder-stem>.safetensors`, or identical original-source metadata when available.
It no longer chooses a head merely because tensor dimensions match. Explicit
selection or renaming a head to the encoder stem binds it to that encoder; ensure
it came from the same original model/fine-tune. A same-sized head from a different
checkpoint can produce incorrect text. An explicit head overrides an embedded one.

Custom head files may contain `lm_head.weight` and optional `lm_head.weight_scale`.
Configuration, tokenizer, encoder, or head file changes invalidate the loaded-model
cache on the next refinement.
