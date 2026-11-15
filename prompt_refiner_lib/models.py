"""Resolve Qwen architecture/tokenizer profiles and build text-only causal models.

For text-only input, all three MRoPE axes have the same positions. Qwen3-VL's
language weights therefore use Qwen3ForCausalLM; Qwen2.5-VL uses Qwen2ForCausalLM
with biased Q/K/V projections. Qwen3.5 uses a separate cached hybrid decoder.
No path allocates a vision tower.
"""

import re
from pathlib import Path

import torch

from .checkpoints import open_checkpoint, read_config
from .lm_heads import MissingLMHeadError
from .quantization import (
    GGUFLinear,
    QuantLinear,
    PackedLinear,
    W4A8Linear,
    packed_shape,
    _dequantize,
    _load_quantized,
    _read_comfy_quant,
    _requantize_rows,
)

FAMILIES = ["Auto", "Qwen3", "Qwen3-VL", "Qwen2.5-VL", "Qwen3.5"]
MODEL_TYPES = {
    "qwen3_5": "Qwen3.5",
    "qwen3_5_text": "Qwen3.5",
    "qwen3": "Qwen3",
    "qwen3_vl": "Qwen3-VL",
    "qwen3_vl_text": "Qwen3-VL",
    "qwen2_5_vl": "Qwen2.5-VL",
    "qwen2_5_vl_text": "Qwen2.5-VL",
}
# Exact published signatures: (hidden, layers, intermediate, Q heads, KV heads).
# Tying is a property of a model configuration, not a hidden-size threshold.
KNOWN = {
    "Qwen3": {
        (1024, 28, 3072, 16, 8): True,
        (2048, 28, 6144, 16, 8): True,
        (2560, 36, 9728, 32, 8): True,
        (4096, 36, 12288, 32, 8): False,
        (5120, 40, 17408, 40, 8): False,
        (5120, 64, 25600, 64, 8): False,
    },
    "Qwen3-VL": {
        (2048, 28, 6144, 16, 8): True,
        (2560, 36, 9728, 32, 8): True,
        (4096, 36, 12288, 32, 8): False,
        (5120, 64, 25600, 64, 8): False,
    },
    "Qwen2.5-VL": {(3584, 28, 18944, 28, 4): False},
}


class UnknownModelSignatureError(ValueError):
    """Signal an unfamiliar architecture that needs an original HF config."""


def logical_shape(f, keys, key):
    """Recover unpacked W4A8 dimensions before architecture detection."""
    shape = tuple(f.get_slice(key).get_shape())
    if not getattr(f, "is_gguf", False) and key.endswith(".weight"):
        info = _read_comfy_quant(f, keys, key[: -len(".weight")])
        if info and info["format"] in (
            "asym_w4a8_int8",
            "w6a8_int8",
            "convrot_w4a4",
            "nvfp4",
        ):
            return packed_shape(f, keys, key, info)
    return shape


def qwen35_profile(f, path, family="Auto"):
    """Infer the dense MOSS 9B hybrid profile and validate any explicit text config."""
    if family not in ("Auto", "Qwen3.5"):
        raise ValueError(
            "Hybrid Qwen3.5 weights conflict with the selected model family."
        )
    metadata = f.metadata() or {}
    gguf = getattr(f, "is_gguf", False)
    if gguf and metadata.get("general.architecture") != "qwen35":
        raise ValueError("This GGUF does not describe dense Qwen3.5.")
    if metadata.get("qwen35.rope.scaling.type", "none") not in ("none", "default", ""):
        raise ValueError("Non-default Qwen3.5 GGUF RoPE scaling is not supported.")
    keys = set(f.keys())
    config = read_config(path)
    text = dict(config.get("text_config", config))
    model_type = config.get("model_type", text.get("model_type"))
    if model_type and model_type not in ("qwen3_5", "qwen3_5_text"):
        raise ValueError("The adjacent configuration does not describe dense Qwen3.5.")
    anchors = [k for k in keys if k.endswith("layers.0.linear_attn.in_proj_qkv.weight")]
    if len(anchors) != 1:
        raise ValueError("Qwen3.5 requires a supported linear-attention first layer.")
    prefix = anchors[0][: -len("layers.0.linear_attn.in_proj_qkv.weight")]

    def shape(name):
        return logical_shape(f, keys, prefix + name)

    vocab, hidden = shape("embed_tokens.weight")
    layers = sorted(
        {
            int(m.group(1))
            for key in keys
            if key.startswith(prefix)
            for m in [re.match(r"layers\.(\d+)\.", key[len(prefix) :])]
            if m
        }
    )
    if layers != list(range(len(layers))) or not layers:
        raise ValueError(
            "Qwen3.5 language layers must be contiguous and start at zero."
        )
    kinds = []
    for i in layers:
        linear = prefix + f"layers.{i}.linear_attn.in_proj_qkv.weight" in keys
        full = prefix + f"layers.{i}.self_attn.q_proj.weight" in keys
        if linear == full:
            raise ValueError(f"Invalid Qwen3.5 attention type in layer {i}.")
        kinds.append("linear_attention" if linear else "full_attention")
    if "full_attention" not in kinds:
        raise ValueError("Qwen3.5 needs at least one full-attention layer.")
    first = kinds.index("full_attention")
    head_dim = shape(f"layers.{first}.self_attn.q_norm.weight")[0]
    qsize = shape(f"layers.{first}.self_attn.q_proj.weight")[0]
    ksize = shape(f"layers.{first}.self_attn.k_proj.weight")[0]
    if qsize % (2 * head_dim) or ksize % head_dim:
        raise ValueError("Invalid Qwen3.5 gated query/key dimensions.")
    intermediate = shape("layers.0.mlp.gate_proj.weight")[0]
    conv_shape = shape("layers.0.linear_attn.conv1d.weight")
    vheads = shape("layers.0.linear_attn.in_proj_a.weight")[0]
    vdim = shape("layers.0.linear_attn.norm.weight")[0]
    kdim = int(
        text.get("linear_key_head_dim", metadata.get("qwen35.ssm.state_size", 128))
    )
    ktotal = conv_shape[0] - vheads * vdim
    if len(conv_shape) != 3 or conv_shape[1] != 1 or ktotal <= 0 or ktotal % (2 * kdim):
        raise ValueError("Invalid Qwen3.5 causal convolution/key-head dimensions.")
    kheads = ktotal // (2 * kdim)
    expected = dict(
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=len(layers),
        intermediate_size=intermediate,
        num_attention_heads=qsize // (2 * head_dim),
        num_key_value_heads=ksize // head_dim,
        head_dim=head_dim,
        linear_num_key_heads=kheads,
        linear_num_value_heads=vheads,
        linear_key_head_dim=kdim,
        linear_value_head_dim=vdim,
        linear_conv_kernel_dim=conv_shape[-1],
        layer_types=kinds,
    )
    if gguf:
        for suffix, value in (
            ("ssm.group_count", kheads),
            ("ssm.time_step_rank", vheads),
            ("ssm.inner_size", vheads * vdim),
            ("ssm.state_size", kdim),
            ("ssm.conv_kernel", conv_shape[-1]),
        ):
            if (
                "qwen35." + suffix in metadata
                and int(metadata["qwen35." + suffix]) != value
            ):
                raise ValueError(
                    f"Qwen3.5 GGUF metadata {suffix} conflicts with tensor dimensions."
                )
    known = (
        vocab,
        hidden,
        len(layers),
        intermediate,
        expected["num_attention_heads"],
        expected["num_key_value_heads"],
        head_dim,
        kheads,
        vheads,
        kdim,
        vdim,
        conv_shape[-1],
    ) == (248320, 4096, 32, 12288, 16, 4, 256, 16, 32, 128, 128, 4)
    if not known and not (model_type and all(key in text for key in expected)):
        raise ValueError(
            "Only the dense Qwen3.5 9B profile is automatic; supply its complete original text config for other dimensions."
        )
    for key, value in expected.items():
        if key in text and text[key] != value:
            raise ValueError(
                f"Configuration {key} conflicts with Qwen3.5 checkpoint dimensions."
            )
    if (
        expected["num_attention_heads"] % expected["num_key_value_heads"]
        or vheads % kheads
    ):
        raise ValueError("Invalid Qwen3.5 grouped attention head counts.")
    if known and kinds != [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in layers
    ]:
        raise ValueError("Qwen3.5 9B requires full attention every fourth layer.")
    if text.get("attention_bias", False) or not text.get("attn_output_gate", True):
        raise ValueError("Unsupported Qwen3.5 attention bias/gate configuration.")
    if text.get("hidden_act", "silu") != "silu" or text.get(
        "use_sliding_window", False
    ):
        raise ValueError(
            "Unsupported Qwen3.5 activation or sliding-window configuration."
        )
    rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError("Non-default Qwen3.5 RoPE scaling is not supported.")
    partial = float(
        rope.get(
            "partial_rotary_factor",
            text.get(
                "partial_rotary_factor",
                float(metadata.get("qwen35.rope.dimension_count", head_dim * 0.25))
                / head_dim,
            ),
        )
    )
    rotary = int(head_dim * partial)
    if not 0 < partial <= 1 or rotary < 2 or rotary % 2:
        raise ValueError("Invalid Qwen3.5 partial rotary dimension.")
    tied = text.get("tie_word_embeddings", config.get("tie_word_embeddings", False))
    if tied is not False:
        raise ValueError("Dense Qwen3.5 requires its separate original output head.")
    return {
        **expected,
        "family": "Qwen3.5",
        "prefix": prefix,
        "tied": False,
        "rope_theta": float(
            rope.get(
                "rope_theta",
                text.get(
                    "rope_theta", metadata.get("qwen35.rope.freq_base", 10000000.0)
                ),
            )
        ),
        "partial_rotary_factor": partial,
        "max_position_embeddings": int(
            text.get(
                "max_position_embeddings", metadata.get("qwen35.context_length", 262144)
            )
        ),
        "rms_norm_eps": float(
            text.get(
                "rms_norm_eps",
                metadata.get("qwen35.attention.layer_norm_rms_epsilon", 1e-6),
            )
        ),
        "hidden_act": "silu",
        "source": config.get("_name_or_path", (f.metadata() or {}).get("source", "")),
    }


def profile(f, path, family="Auto"):
    """Resolve architecture from explicit choice, HF config, GGUF metadata, then signatures."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown model family: {family}")
    keys = set(f.keys())
    if any(".experts." in key for key in keys):
        raise ValueError("Mixture-of-experts encoders are not supported.")
    if any("linear_attn." in key for key in keys) or family == "Qwen3.5":
        return qwen35_profile(f, path, family)
    anchor = next(
        (k for k in keys if k.endswith("layers.0.self_attn.q_proj.weight")), None
    )
    if not anchor:
        raise ValueError(
            "The checkpoint has no supported Qwen language-model attention weights."
        )
    prefix = anchor[: -len("layers.0.self_attn.q_proj.weight")]
    config = read_config(path)
    text = dict(config.get("text_config", config))
    model_type = config.get("model_type", text.get("model_type"))
    declared = MODEL_TYPES.get(model_type)
    if model_type and not declared:
        raise ValueError(
            f"Unsupported model_type '{model_type}' in the adjacent configuration."
        )
    metadata = f.metadata() or {}
    arch = metadata.get("general.architecture")
    if arch and metadata.get(f"{arch}.rope.scaling.type", "none") not in (
        "none",
        "default",
        "",
    ):
        raise ValueError("Non-default GGUF RoPE scaling is not supported.")
    gguf_family = {
        "qwen3": "Qwen3",
        "qwen3vl": "Qwen3-VL",
        "qwen2vl": "Qwen2.5-VL",
    }.get(arch)
    declared = declared or gguf_family
    if family != "Auto" and declared and family != declared:
        raise ValueError(
            f"Selected family {family} conflicts with checkpoint configuration {declared}."
        )
    chosen = family if family != "Auto" else declared
    has_qnorm = prefix + "layers.0.self_attn.q_norm.weight" in keys
    vocab, hidden = logical_shape(f, keys, prefix + "embed_tokens.weight")
    layers = {
        int(m.group(1))
        for key in keys
        if key.startswith(prefix)
        for m in [re.match(r"layers\.(\d+)\.", key[len(prefix) :])]
        if m
    }
    if not layers or layers != set(range(max(layers) + 1)):
        raise ValueError("Language-model layers must be contiguous and start at zero.")
    if chosen is None:
        hint = (Path(path).stem + " " + str(metadata.get("source", ""))).lower()
        if not has_qnorm and vocab == 152064 and hidden == 3584:
            chosen = "Qwen2.5-VL"
        elif (
            "qwen3vl" in hint
            or re.search(r"qwen[_-]?3[_-]?vl", hint)
            or "language_model." in prefix
            or any(".visual." in k or k.startswith("visual.") for k in keys)
        ):
            chosen = "Qwen3-VL"
        elif re.search(r"qwen[_-]?3(?![.][5])", hint):
            chosen = "Qwen3"
        else:
            raise ValueError(
                "Model family is ambiguous. Select Qwen3 or Qwen3-VL, or supply an adjacent .config.json."
            )
    if (chosen == "Qwen2.5-VL") == has_qnorm:
        raise ValueError(f"Attention normalization weights do not match {chosen}.")
    head_dim = (
        logical_shape(f, keys, prefix + "layers.0.self_attn.q_norm.weight")[0]
        if has_qnorm
        else int(text.get("head_dim", 128))
    )
    qsize = logical_shape(f, keys, anchor)[0]
    ksize = logical_shape(f, keys, prefix + "layers.0.self_attn.k_proj.weight")[0]
    if head_dim <= 0 or qsize % head_dim or ksize % head_dim:
        raise ValueError(
            "Attention dimensions are incompatible with the configured head size."
        )
    heads, kv = qsize // head_dim, ksize // head_dim
    if not heads or not kv or heads % kv:
        raise ValueError(
            "Query head count must be a positive multiple of the KV head count."
        )
    intermediate = logical_shape(f, keys, prefix + "layers.0.mlp.gate_proj.weight")[0]
    signature = (hidden, len(layers), intermediate, heads, kv)
    tied = text.get("tie_word_embeddings", config.get("tie_word_embeddings"))
    if tied is None:
        tied = KNOWN[chosen].get(signature)
    if tied is None:
        raise UnknownModelSignatureError(
            f"Unknown {chosen} signature {signature}. Supply an HF .config.json with tie_word_embeddings."
        )
    if not isinstance(tied, bool):
        raise ValueError("Configured tie_word_embeddings must be a boolean.")
    expected = dict(
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=len(layers),
        intermediate_size=intermediate,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        head_dim=head_dim,
    )
    for key, value in expected.items():
        if key in text and text[key] != value:
            raise ValueError(
                f"Configuration {key}={text[key]} conflicts with checkpoint value {value}."
            )
    bias = chosen == "Qwen2.5-VL"
    if "attention_bias" in text and bool(text["attention_bias"]) != bias:
        raise ValueError(f"Configured attention_bias does not match {chosen}.")
    if text.get("use_sliding_window", False) or any(
        kind != "full_attention" for kind in text.get("layer_types", []) or []
    ):
        raise ValueError(
            "Sliding-window or hybrid attention configurations are not supported."
        )
    for layer in layers:
        for part in ("q", "k", "v"):
            name = prefix + f"layers.{layer}.self_attn.{part}_proj.bias"
            if (name in keys) != bias:
                raise ValueError(
                    f"Unexpected or missing attention bias for {chosen}: {name}"
                )
    rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError(
            "Non-default RoPE scaling is not supported; use an unscaled encoder configuration."
        )
    theta = rope.get("rope_theta", text.get("rope_theta"))
    if theta is None and arch:
        theta = metadata.get(f"{arch}.rope.freq_base")
    if theta is None:
        theta = (
            5_000_000.0
            if chosen == "Qwen3-VL" or "2507" in Path(path).name
            else 1_000_000.0
        )
    default_context = (
        262144
        if chosen == "Qwen3-VL" or "2507" in Path(path).name
        else (128000 if chosen == "Qwen2.5-VL" else 40960)
    )
    context = int(
        text.get(
            "max_position_embeddings",
            metadata.get(f"{arch}.context_length", default_context),
        )
    )
    return {
        **expected,
        "family": chosen,
        "prefix": prefix,
        "tied": bool(tied),
        "rope_theta": float(theta),
        "max_position_embeddings": context,
        "rms_norm_eps": float(
            text.get(
                "rms_norm_eps",
                metadata.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6),
            )
        ),
        "hidden_act": text.get("hidden_act", "silu"),
        "source": config.get("_name_or_path", metadata.get("source", "")),
    }


def inspect_profile(path, family="Auto"):
    """Inspect a checkpoint without loading full weights into RAM or VRAM."""
    with open_checkpoint(path) as f:
        return profile(f, path, family)


def find_tokenizer(path, resolved, forge_root, extension_dir, override=""):
    """Choose family-specific local tokenizer assets, with an explicit override first."""
    family = resolved["family"]
    slug = {
        "Qwen3": "qwen3",
        "Qwen3-VL": "qwen3_vl",
        "Qwen2.5-VL": "qwen2_5_vl",
        "Qwen3.5": "qwen3_5",
    }[family]
    path, extension_dir, forge_root = Path(path), Path(extension_dir), Path(forge_root)
    if override.strip():
        candidates = [Path(override.strip()).expanduser()]
    else:
        # Adjacent folders and the extension's family folder are intentional overrides.
        candidates = [
            path.with_suffix(".tokenizer"),
            path.parent / "tokenizer",
            extension_dir / slug,
        ]
        preferred = {
            "Qwen3": [
                "black-forest-labs/FLUX.2-klein-4B",
                "Tongyi-MAI/Z-Image-Turbo",
                "circlestone-labs/Anima",
            ],
            "Qwen3-VL": ["krea/Krea-2-Raw", "Qwen/Qwen-Image-2.1"],
            "Qwen2.5-VL": ["Qwen/Qwen-Image"],
            "Qwen3.5": ["OpenMOSS-Team/MOSS-Image-SFT"],
        }[family]
        # Preserve the source tokenizer's chat template when the encoder identifies
        # its Neo integration. Vocabulary compatibility alone does not select a template.
        if family == "Qwen3" and (
            "anima" in str(path).lower() or "qwen_3_06b_base" in path.stem.lower()
        ):
            preferred = ["circlestone-labs/Anima"] + [
                repo for repo in preferred if repo != "circlestone-labs/Anima"
            ]
        root = forge_root / "backend" / "huggingface"
        candidates.extend(root / repo / "tokenizer" for repo in preferred)
    for candidate in candidates:
        if (candidate / "tokenizer.json").is_file() or (
            (candidate / "vocab.json").is_file()
            and (candidate / "merges.txt").is_file()
        ):
            return candidate
    raise ValueError(
        f"No local {family} tokenizer found. Set 'Tokenizer directory' or place its tokenizer files in {extension_dir / slug}."
    )


def load_tokenizer(path, resolved):
    """Load locally and verify the special tokens used by the chat protocol."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    special = (
        (("<|im_start|>", 248045), ("<|im_end|>", 248046))
        if resolved["family"] == "Qwen3.5"
        else (("<|im_start|>", 151644), ("<|im_end|>", 151645))
    )
    for token, expected in special:
        if tokenizer.convert_tokens_to_ids(token) != expected:
            raise ValueError(
                f"Tokenizer has an incompatible {token} ID; select the matching {resolved['family']} tokenizer."
            )
    if max(tokenizer.get_vocab().values()) >= resolved["vocab_size"]:
        raise ValueError(
            "Tokenizer contains token IDs outside the checkpoint vocabulary."
        )
    if not tokenizer.chat_template:
        # Forge ships encoding-only tokenizer configs. These Qwen families use
        # ChatML for text, so a local fallback needs no additional download.
        tokenizer.chat_template = (
            "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + "
            "message['content'] + '<|im_end|>\n' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
    return tokenizer


def build_model(
    path,
    dtype,
    rope_choice,
    work_device,
    keep_quantized,
    compress,
    lm_choice,
    family,
    pick_head,
):
    """Construct on meta, assign validated tensors, then install compressed linear modules."""
    from transformers import (
        Qwen2Config,
        Qwen2ForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    with open_checkpoint(path) as f:
        p = profile(f, path, family)
        keys = set(f.keys())
        prefix = p["prefix"]
        theta = p["rope_theta"]
        if rope_choice.startswith("10,000,000"):
            theta = 10_000_000.0
        elif rope_choice.startswith("1"):
            theta = 1_000_000.0
        elif rope_choice.startswith("5"):
            theta = 5_000_000.0
        qwen35 = p["family"] == "Qwen3.5"
        if qwen35:
            from .qwen35 import Qwen35Config, Qwen35ForCausalLM

            fields = {
                key: value
                for key, value in p.items()
                if key not in ("family", "prefix", "tied", "source")
            }
            fields["rope_theta"] = theta
            config = Qwen35Config(**fields)
            config._attn_implementation = "sdpa"
            with torch.device("meta"):
                model = Qwen35ForCausalLM(config)
        else:
            qwen2 = p["family"] == "Qwen2.5-VL"
            config_cls, model_cls = (
                (Qwen2Config, Qwen2ForCausalLM)
                if qwen2
                else (Qwen3Config, Qwen3ForCausalLM)
            )
            config = config_cls(
                **{
                    key: p[key]
                    for key in (
                        "vocab_size",
                        "hidden_size",
                        "intermediate_size",
                        "num_hidden_layers",
                        "num_attention_heads",
                        "num_key_value_heads",
                        "head_dim",
                        "max_position_embeddings",
                        "rms_norm_eps",
                        "hidden_act",
                    )
                },
                rope_theta=theta,
                tie_word_embeddings=p["tied"],
                attention_bias=qwen2,
                use_sliding_window=False,
            )
            config._attn_implementation = "sdpa"
            if isinstance(getattr(config, "rope_parameters", None), dict):
                config.rope_parameters["rope_theta"] = theta
            with torch.device("meta"):
                model = model_cls(config)
            if qwen2:
                from transformers.models.qwen2.modeling_qwen2 import (
                    Qwen2RotaryEmbedding as Rotary,
                )
            else:
                from transformers.models.qwen3.modeling_qwen3 import (
                    Qwen3RotaryEmbedding as Rotary,
                )
            model.model.rotary_emb = Rotary(config=config, device="cpu")

        heads = [k for k in keys if k.endswith("lm_head.weight")]
        if len(heads) > 1:
            raise ValueError("Checkpoint contains multiple language output heads.")
        head = heads[0] if heads else None
        sidecar = None
        if lm_choice != "Auto" or (head is None and not p["tied"]):
            sidecar = pick_head(
                lm_choice, Path(path), p["vocab_size"], p["hidden_size"], p
            )
        if head is None and sidecar is None and not p["tied"]:
            raise MissingLMHeadError(p["family"])

        state, replacements = {}, {}
        gguf_count, w4_count, requant_count = 0, 0, 0
        w6_count, int4_count, nvfp4_count = 0, 0, 0

        def load_weight(source, source_keys, key, target):
            """Load one expected weight, preserving supported quantized linears."""
            nonlocal \
                gguf_count, \
                w4_count, \
                requant_count, \
                w6_count, \
                int4_count, \
                nvfp4_count
            module_name = target[: -len(".weight")]
            module = model.get_submodule(module_name)
            shape = logical_shape(source, source_keys, key)
            if tuple(module.weight.shape) != shape:
                raise ValueError(
                    f"Weight shape mismatch in {key}: expected {tuple(module.weight.shape)}, got {shape}."
                )
            if isinstance(module, torch.nn.Linear) and keep_quantized:
                if getattr(source, "is_gguf", False):
                    packed = source.packed(key, dtype)
                    if packed is not None:
                        permutation = getattr(source, "input_permutations", {}).get(key)
                        replacements[module_name] = GGUFLinear(packed, permutation)
                        gguf_count += 1
                        return
                    if compress:
                        quant = _requantize_rows(source.get_tensor(key), work_device)
                        replacements[module_name] = QuantLinear(*quant)
                        requant_count += 1
                        return
                else:
                    quant = _load_quantized(source, source_keys, key)
                    if isinstance(quant, torch.nn.Module):
                        replacements[module_name] = quant
                        if isinstance(quant, W4A8Linear):
                            if quant.bits == 6:
                                w6_count += 1
                            else:
                                w4_count += 1
                        elif isinstance(quant, PackedLinear):
                            if quant.kind == "nvfp4":
                                nvfp4_count += 1
                            else:
                                int4_count += 1
                        return
                    if quant is None:
                        small = source.get_slice(key).get_dtype() in (
                            "F8_E4M3",
                            "F8_E5M2",
                            "I8",
                        )
                        if small or compress:
                            quant = _requantize_rows(
                                _dequantize(
                                    source, source_keys, key, torch.float32, work_device
                                ),
                                work_device,
                            )
                            requant_count += 1
                    if quant is not None:
                        replacements[module_name] = QuantLinear(*quant)
                        return
            # Preserve FP32 norm parameters, whose forward path casts its result.
            # Embeddings and convolutions must use the selected activation dtype;
            # GGUF commonly stores their source weights as FP32 as well.
            target_dtype = (
                torch.float32
                if qwen35
                and module.weight.ndim == 1
                and source.get_slice(key).get_dtype() == "F32"
                else dtype
            )
            if getattr(source, "is_gguf", False):
                state[target] = source.get_tensor(key).to(target_dtype)
            else:
                state[target] = _dequantize(
                    source, source_keys, key, target_dtype, work_device
                )

        expected = model.state_dict()
        for target in expected:
            if target == "lm_head.weight":
                continue
            key = prefix + target[len("model.") :]
            if key not in keys:
                raise ValueError(f"Missing language-model weight: {key}")
            if target.endswith(".weight"):
                load_weight(f, keys, key, target)
            else:
                tensor = f.get_tensor(key)
                if tuple(tensor.shape) != tuple(expected[target].shape):
                    raise ValueError(f"Bias shape mismatch: {key}")
                state[target] = tensor.to(
                    torch.float32 if qwen35 and target.endswith(".A_log") else dtype
                )
        if sidecar:
            with open_checkpoint(sidecar) as sf:
                load_weight(sf, set(sf.keys()), "lm_head.weight", "lm_head.weight")
        elif head:
            load_weight(f, keys, head, "lm_head.weight")

        # Do not ignore unexpected language weights: a subtly different architecture
        # must fail rather than produce apparently valid but incorrect text.
        consumed = {
            prefix + k[len("model.") :] for k in expected if k.startswith("model.")
        }
        suffixes = (
            ".comfy_quant",
            ".scale_weight",
            ".weight_scale",
            ".weight_scale_inv",
            ".input_scale",
            ".scale_input",
            ".weight_scale_2",
            ".scaled_fp8",
            ".weight_s_rel",
            ".weight_s_channel",
            ".weight_codebook",
            ".weight_correction",
            ".pre_quant_scale",
        )
        extras = [
            k
            for k in keys
            if k.startswith(prefix)
            and re.match(r"(?:layers\.\d+|embed_tokens|norm)\.", k[len(prefix) :])
            and k not in consumed
            and not k.endswith(suffixes)
        ]
        if extras:
            raise ValueError(f"Unexpected language-model tensors: {extras[:5]}")

    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    tied = head is None and sidecar is None
    allowed_missing = {name + ".weight" for name in replacements}
    if tied:
        allowed_missing.add("lm_head.weight")
    if set(missing) - allowed_missing or unexpected:
        raise ValueError(
            f"Incomplete model load: missing {set(missing) - allowed_missing}, unexpected {unexpected}"
        )
    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    for name, replacement in replacements.items():
        bias_key = name + ".bias"
        if bias_key in state:
            replacement.bias = state[bias_key]
        parent, _, attr = name.rpartition(".")
        setattr(model.get_submodule(parent), attr, replacement)
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError("Model still contains uninitialized tensors after loading.")
    model.requires_grad_(False)
    model.eval()
    model._pr_quant_layers = len(replacements)
    model._pr_gguf_layers, model._pr_w4_layers = gguf_count, w4_count
    model._pr_w6_layers, model._pr_int4_layers, model._pr_nvfp4_layers = (
        w6_count,
        int4_count,
        nvfp4_count,
    )
    model._pr_profile = p
    info = (
        f"{p['family']}, {p['num_hidden_layers']} layers, hidden {p['hidden_size']}, "
        f"RoPE {theta:g}; {len(replacements)} compressed layers "
        f"({w4_count} W4A8, {w6_count} W6A8, {int4_count} INT4 ConvRot, {nvfp4_count} NVFP4, {gguf_count} GGUF, {requant_count} converted to INT8); "
        f"output head: {Path(sidecar).name if sidecar else ('tied embeddings' if tied else 'checkpoint')}"
    )
    if qwen35:
        info += (
            f"; hybrid attention: {p['layer_types'].count('linear_attention')} linear / "
            f"{p['layer_types'].count('full_attention')} full; recurrent state: FP32; vision/MTP skipped"
        )
    if re.search(r"(?:^|[_ .-])base(?:$|[_ .-])", Path(path).stem.lower()):
        info += (
            "; WARNING: base checkpoint, not instruction-tuned; chat instructions "
            "may produce repetition or unusable output. Prefer an instruction-tuned encoder."
        )
    return model, info
