Tokenizer files are optional when matching assets already exist in Forge Neo.

Use the matching family subfolder:
  qwen3/       - Qwen3 tokenizer files
  qwen3_vl/    - Qwen3-VL tokenizer files
  qwen2_5_vl/  - Qwen2.5-VL tokenizer files
  qwen3_5/     - Qwen3.5 / MOSS-Image tokenizer files (different vocabulary)

Include tokenizer_config.json and tokenizer.json, or vocab.json plus merges.txt,
and any other tokenizer assets from the same original model. Files are loaded
locally. The UI also accepts an explicit tokenizer directory.
