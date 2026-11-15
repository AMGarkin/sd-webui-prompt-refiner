"""Reverse llama.cpp Qwen3.5 conversion without expanding resident linear weights.

Supports split and merged QKV/Z and beta/alpha projections. GGUF uses tiled V
heads, shifted norms and negative exponential decay; the text runtime uses HF's
original grouped V order, zero-centered norm weights and A_log. Row permutations
are applied to packed records; out-projection columns are handled on activations.
Reference: llama.cpp conversion/qwen.py and Forge Neo backend/loader_gguf.py.
"""

import re

import torch


def configure(checkpoint):
    """Build virtual HF tensor views using GGUF hybrid metadata and tensor headers."""
    info = checkpoint.info
    if any(key.startswith("prism.hadamard.") for key in info):
        raise ValueError("Hadamard-folded Qwen3.5 GGUF exports are not supported.")
    for field in (
        "ssm.group_count",
        "ssm.time_step_rank",
        "ssm.state_size",
        "ssm.inner_size",
    ):
        if int(info.get("qwen35." + field, 0)) <= 0:
            raise ValueError(f"Missing/invalid Qwen3.5 GGUF metadata qwen35.{field}.")
    nk, nv = (
        int(info["qwen35." + key]) for key in ("ssm.group_count", "ssm.time_step_rank")
    )
    kd = int(info["qwen35.ssm.state_size"])
    vd, remainder = divmod(int(info["qwen35.ssm.inner_size"]), nv)
    if remainder or nv % nk:
        raise ValueError("Invalid grouped Qwen3.5 GGUF value dimensions.")
    checkpoint.hybrid_dims = nk, nv, kd, vd
    checkpoint.row_slices = {}
    checkpoint.input_permutations = {}
    checkpoint.row_permutations = {}
    checkpoint.tensors = {}
    forward = torch.arange(nv * vd).reshape(nk, nv // nk, vd).transpose(0, 1).flatten()
    inverse = torch.argsort(forward)
    head_forward = torch.arange(nv).reshape(nk, nv // nk).transpose(0, 1).flatten()
    head_inverse = torch.argsort(head_forward)
    qk = 2 * nk * kd
    names = {
        "attn_norm.weight": "input_layernorm.weight",
        "post_attention_norm.weight": "post_attention_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
        "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight": "linear_attn.in_proj_z.weight",
        "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
        "ssm_beta.weight": "linear_attn.in_proj_b.weight",
        "ssm_out.weight": "linear_attn.out_proj.weight",
        "ssm_norm.weight": "linear_attn.norm.weight",
        "ssm_conv1d.weight": "linear_attn.conv1d.weight",
        "ssm_a": "linear_attn.A_log",
        "ssm_a.weight": "linear_attn.A_log",
        "ssm_dt.bias": "linear_attn.dt_bias",
    }
    fixed = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    blocks = int(info.get("qwen35.block_count", 0))
    mtp = int(info.get("qwen35.nextn_predict_layers", 0))
    if mtp < 0 or mtp >= blocks:
        raise ValueError("Invalid Qwen3.5 GGUF MTP layer count.")

    def add(key, tensor, row_slice=None):
        if key in checkpoint.tensors:
            raise ValueError(f"Duplicate Qwen3.5 GGUF mapping: {key}")
        checkpoint.tensors[key] = tensor
        if row_slice is not None:
            checkpoint.row_slices[key] = row_slice
        if key.endswith(
            ("linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight")
        ):
            checkpoint.row_permutations[key] = torch.cat(
                (torch.arange(qk), qk + inverse)
            )
        elif key.endswith("linear_attn.in_proj_z.weight"):
            checkpoint.row_permutations[key] = inverse
        elif key.endswith(
            (
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.A_log",
                "linear_attn.dt_bias",
            )
        ):
            checkpoint.row_permutations[key] = head_inverse
        elif key.endswith("linear_attn.out_proj.weight"):
            checkpoint.input_permutations[key] = forward

    for tensor in checkpoint.reader.tensors:
        name = tensor.name
        if name in fixed:
            add(fixed[name], tensor)
            continue
        match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
        if not match:
            continue  # Vision assets are not part of the text-only graph.
        layer, part = match.groups()
        if mtp and int(layer) >= blocks - mtp:
            continue
        prefix = f"model.layers.{layer}."
        if part == "attn_qkv_gate.weight":
            if int(tensor.shape[-1]) != qk + 2 * nv * vd:
                raise ValueError("Invalid merged Qwen3.5 QKV/gate row count.")
            add(prefix + "linear_attn.in_proj_qkv.weight", tensor, (0, qk + nv * vd))
            add(
                prefix + "linear_attn.in_proj_z.weight",
                tensor,
                (qk + nv * vd, qk + 2 * nv * vd),
            )
        elif part == "ssm_beta_alpha.weight":
            if int(tensor.shape[-1]) != 2 * nv:
                raise ValueError("Invalid merged Qwen3.5 beta/alpha row count.")
            add(prefix + "linear_attn.in_proj_b.weight", tensor, (0, nv))
            add(prefix + "linear_attn.in_proj_a.weight", tensor, (nv, 2 * nv))
        elif part in names:
            add(prefix + names[part], tensor)
        else:
            raise ValueError(f"Unsupported Qwen3.5 GGUF language tensor: {name}")


def shape(checkpoint, key):
    """Expose logical HF shapes for virtual slices and squeezed convolution weights."""
    result = list(reversed(checkpoint.tensors[key].shape))
    if key in checkpoint.row_slices:
        start, end = checkpoint.row_slices[key]
        if not 0 <= start < end <= result[0]:
            raise ValueError(f"Invalid merged Qwen3.5 projection dimensions: {key}")
        result[0] = end - start
    if key.endswith("linear_attn.conv1d.weight"):
        result.insert(1, 1)
    return tuple(int(v) for v in result)


def raw_rows(checkpoint, key):
    """Slice and reorder complete GGUF records before baking into Neo's layout."""
    tensor = checkpoint.tensors[key]
    data = tensor.data
    if len(tensor.shape) > 1:
        data = data.reshape(int(tensor.shape[-1]), -1)
    if key in checkpoint.row_slices:
        start, end = checkpoint.row_slices[key]
        data = data[start:end]
    if key in checkpoint.row_permutations:
        data = data[checkpoint.row_permutations[key].numpy()]
    return data


def dense(checkpoint, key):
    """Undo nonlinear GGUF transforms on CPU; packed out-projections stay unchanged."""
    import gguf
    import numpy as np

    tensor = checkpoint.tensors[key]
    data = gguf.quants.dequantize(raw_rows(checkpoint, key), tensor.tensor_type)
    result = torch.from_numpy(np.array(data, copy=True)).reshape(shape(checkpoint, key))
    if key in checkpoint.input_permutations:
        result = result[:, torch.argsort(checkpoint.input_permutations[key])]
    if key.endswith("norm.weight") and not key.endswith("linear_attn.norm.weight"):
        result = result.float() - 1
    if key.endswith("linear_attn.A_log"):
        if not torch.isfinite(result).all() or (result >= 0).any():
            raise ValueError("Qwen3.5 GGUF decay values must be finite and negative.")
        result = torch.log(-result.float())
    return result
