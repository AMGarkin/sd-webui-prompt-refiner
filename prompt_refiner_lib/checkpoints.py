"""Read checkpoint headers and expose one tensor interface for safetensors and GGUF.

GGUF data stays memory mapped until a tensor is used. Linear layers can retain the
original packed bytes; only embeddings and small non-linear weights become dense.
"""

import json
import re
from pathlib import Path

import torch


class TensorSlice:
    """Provide the shape/dtype subset of the safetensors slice API."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype

    def get_shape(self):
        return self.shape

    def get_dtype(self):
        return self.dtype


class SafeCheckpoint:
    """Expose both Comfy per-layer tensors and header-level quantization metadata."""

    is_gguf = False

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        from safetensors import safe_open

        self.file = safe_open(str(self.path), "pt", device="cpu")
        self.file.__enter__()
        try:
            self.info = self.file.metadata() or {}
            self.extra = {}
            keys = set(self.file.keys())
            if "_quantization_metadata" in self.info:
                metadata = json.loads(self.info["_quantization_metadata"])
                if not isinstance(metadata, dict):
                    raise ValueError("Quantization metadata must be an object.")
                layers = metadata.get("layers", {})
                if not isinstance(layers, dict):
                    raise ValueError("Quantization metadata layers must be an object.")
                for layer, config in layers.items():
                    matches = [
                        k[: -len(".weight")]
                        for k in keys
                        if k == layer + ".weight" or k.endswith("." + layer + ".weight")
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            f"Cannot bind quantization metadata to one layer: {layer}"
                        )
                    key = matches[0] + ".comfy_quant"
                    if key not in keys:
                        self.extra[key] = torch.tensor(
                            list(json.dumps(config).encode()), dtype=torch.uint8
                        )
            self.keyset = keys | set(self.extra)
            return self
        except Exception:
            self.file.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        return self.file.__exit__(*args)

    def keys(self):
        return self.keyset

    def metadata(self):
        return self.info

    def get_tensor(self, key):
        return self.extra[key] if key in self.extra else self.file.get_tensor(key)

    def get_slice(self, key):
        return self.file.get_slice(key)


def gguf_name(name):
    """Translate language-model GGUF names without touching vision tensors."""
    fixed = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    if name in fixed:
        return fixed[name]
    match = re.fullmatch(r"blk\.(\d+)\.([\w_]+)\.(weight|bias)", name)
    if not match:
        return None
    layer, part, suffix = match.groups()
    names = {
        "attn_norm": "input_layernorm",
        "ffn_norm": "post_attention_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj",
        "attn_q_norm": "self_attn.q_norm",
        "attn_k_norm": "self_attn.k_norm",
        "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
    }
    return f"model.layers.{layer}.{names[part]}.{suffix}" if part in names else None


class GGUFCheckpoint:
    """Adapt GGUFReader to the small safetensors API used by the model builder."""

    is_gguf = True

    def __init__(self, path):
        import gguf

        self.reader = gguf.GGUFReader(str(path))
        # Tokenizer arrays are large and are not needed: use Forge's local tokenizer.
        self.info = {
            k: v.contents()
            for k, v in self.reader.fields.items()
            if not k.startswith("tokenizer.")
        }
        arch = self.info.get("general.architecture")
        if int(self.info.get("split.count", 1)) > 1:
            raise ValueError(
                "Split GGUF checkpoints must be merged into one GGUF file before use."
            )
        if arch not in ("qwen3", "qwen3vl", "qwen2vl"):
            raise ValueError(
                f"Unsupported GGUF architecture '{arch}'. Supported: qwen3, qwen3vl, qwen2vl."
            )
        self.tensors = {}
        for tensor in self.reader.tensors:
            name = gguf_name(tensor.name)
            if name:
                if name in self.tensors:
                    raise ValueError(f"Duplicate GGUF tensor mapping: {name}")
                self.tensors[name] = tensor
            elif tensor.name.startswith("blk."):
                raise ValueError(
                    f"Unsupported language-model GGUF tensor: {tensor.name}"
                )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Modules copy packed bytes, so the reader can release its mmap after load.
        self.tensors.clear()
        self.reader = None

    def keys(self):
        return self.tensors.keys()

    def metadata(self):
        return self.info

    def shape(self, key):
        return tuple(int(v) for v in reversed(self.tensors[key].shape))

    def get_slice(self, key):
        return TensorSlice(self.shape(key), self.tensors[key].tensor_type.name)

    def packed(self, key, dtype=torch.float32):
        """Copy and bake one packed tensor using Forge's own quantization layouts."""
        import gguf
        from backend.operations_gguf import QUANTS_MAPPING, ParameterGGUF

        tensor = self.tensors[key]
        kind = tensor.tensor_type
        if kind in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            return None
        if kind not in QUANTS_MAPPING:
            raise ValueError(f"Unsupported GGUF quantization {kind.name} in {key}.")
        raw = torch.from_numpy(tensor.data.copy())
        weight = ParameterGGUF(
            raw, tensor_type=kind, tensor_shape=torch.Size(self.shape(key))
        )
        # Neo's baked Q4/Q8 records reserve two bytes for each scale. Float32
        # would change record widths and corrupt its fixed-offset decoders.
        weight.computation_dtype = torch.float16
        weight.gguf_cls.bake(weight)
        return weight

    def get_tensor(self, key):
        """Decode dense embeddings/norms on CPU using the unmodified GGUF layout."""
        import gguf

        tensor = self.tensors[key]
        array = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        return torch.from_numpy(array.copy()).reshape(self.shape(key))


def open_checkpoint(path):
    """Open only explicitly supported checkpoint containers, without pickle loading."""
    path = Path(path)
    if path.suffix.lower() == ".gguf":
        return GGUFCheckpoint(path)
    if path.suffix.lower() == ".safetensors":
        return SafeCheckpoint(path)
    raise ValueError(f"Unsupported checkpoint extension: {path.suffix}")


def read_config(path):
    """Prefer an encoder-specific HF config, then an adjacent config.json.

    A same-stem config makes renamed/repacked weights unambiguous without adding
    another model download. The complete HF config, including text_config, is accepted.
    """
    path = Path(path)
    for candidate in (path.with_suffix(".config.json"), path.parent / "config.json"):
        if candidate.is_file():
            config = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError(f"Model configuration must be an object: {candidate}")
            return config
    return {}


def is_supported(path):
    """Inspect headers only; do not accept Qwen3.5, MoE, CLIP, or T5 weights."""
    try:
        with open_checkpoint(path) as checkpoint:
            keys = set(checkpoint.keys())
            if any("linear_attn." in k or ".experts." in k for k in keys):
                return False
            anchor = next(
                (k for k in keys if k.endswith("layers.0.self_attn.q_proj.weight")),
                None,
            )
            if not anchor:
                return False
            prefix = anchor[: -len("layers.0.self_attn.q_proj.weight")]
            if prefix + "embed_tokens.weight" not in keys:
                return False
            if prefix + "layers.0.self_attn.q_norm.weight" in keys:
                return True
            # Qwen2.5-VL has biased Q/K/V projections and no Q/K normalization.
            shape = checkpoint.get_slice(prefix + "embed_tokens.weight").get_shape()
            return (
                tuple(shape) == (152064, 3584)
                and prefix + "layers.0.self_attn.q_proj.bias" in keys
            )
    except Exception:
        return False
