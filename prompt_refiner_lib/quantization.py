"""Validated low-memory INT8, FP8, ConvRot, W4A8, and GGUF linear operations."""

import json

import torch

_HADAMARD_CACHE = {}


def _hadamard(group_size: int, device):
    """Normalized 'regular' Hadamard matrix used by ConvRot: Kronecker power of H4, scaled by 1/sqrt(n).
    It is symmetric and orthonormal, so it is its own inverse."""
    key = (group_size, str(device))
    if key not in _HADAMARD_CACHE:
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=torch.float32,
        )
        h = torch.ones(1, 1)
        n = 1
        while n < group_size:
            h = torch.kron(h, h4)
            n *= 4
        if n != group_size:
            raise RuntimeError(
                f"Unsupported ConvRot group size {group_size} (expected a power of 4)."
            )
        _HADAMARD_CACHE[key] = (h / (group_size**0.5)).to(device)
    return _HADAMARD_CACHE[key]


def _read_comfy_quant(f, keyset, base):
    """Decode metadata strictly so malformed or unknown formats cannot become raw INT8."""
    name = base + ".comfy_quant"
    if name not in keyset:
        return None
    try:
        info = json.loads(bytes(f.get_tensor(name).tolist()).decode("utf-8"))
        if not isinstance(info, dict) or not isinstance(info.get("format"), str):
            raise ValueError("expected an object with a format string")
        params = info.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        info = {**params, **info}
    except Exception as exc:
        raise ValueError(f"Invalid quantization metadata in {name}: {exc}") from exc
    # Some mixed-precision exporters name the scaling granularity in the format.
    # Scale tensor shapes are still checked independently by _scale.
    aliases = {
        f"{kind}_{granularity}": kind
        for kind in ("float8_e4m3fn", "float8_e5m2")
        for granularity in ("rowwise", "tensorwise")
    }
    info["format"] = aliases.get(info["format"], info["format"])
    allowed = {
        "float8_e4m3fn",
        "float8_e5m2",
        "fp8",
        "int8",
        "int8_tensorwise",
        "asym_w4a8_int8",
    }
    if info["format"] not in allowed:
        raise ValueError(
            f"Unsupported quantization format '{info['format']}' in {base}."
        )
    return info


def _expand_block_scale(s, n, k, block_size=128):
    """Expand a fixed 128x128 FP8 scale grid, including partial edge blocks."""
    expected = ((n + block_size - 1) // block_size, (k + block_size - 1) // block_size)
    if tuple(s.shape) != expected:
        raise ValueError(
            f"Invalid block scale shape {tuple(s.shape)} for {(n, k)}; expected {expected}."
        )
    return s.repeat_interleave(block_size, 0)[:n].repeat_interleave(block_size, 1)[
        :, :k
    ]


def _scale(f, keyset, key, w):
    """Validate scalar, row, column, and explicitly block-wise dequantization scales."""
    base = key[: -len(".weight")]
    if base + ".weight_scale_2" in keyset:
        raise ValueError(f"Two-stage weight scaling is not supported: {key}")
    names = [
        base + suffix
        for suffix in (".scale_weight", ".weight_scale", ".weight_scale_inv")
        if base + suffix in keyset
    ]
    if len(names) > 1:
        raise ValueError(f"Ambiguous weight scales for {key}: {names}")
    if not names:
        if w.dtype == torch.int8:
            raise ValueError(f"INT8 weight has no dequantization scale: {key}")
        return None, False
    name = names[0]
    scale = f.get_tensor(name).float()
    if not torch.isfinite(scale).all() or (scale < 0).any():
        raise ValueError(f"Invalid weight scale values in {name}.")
    # HF weight_scale_inv stores the multiplier used to recover weights, despite its name.
    block = name.endswith(".weight_scale_inv") and scale.ndim == 2
    if block:
        expected = tuple((size + 127) // 128 for size in w.shape)
        if tuple(scale.shape) != expected:
            raise ValueError(
                f"Invalid 128x128 block scale shape in {name}: expected {expected}."
            )
    elif scale.numel() == 1:
        scale = scale.reshape(())
    elif w.ndim == 2 and (
        tuple(scale.shape) == (w.shape[0],) or tuple(scale.shape) == (w.shape[0], 1)
    ):
        scale = scale.reshape(-1, 1)
    elif w.ndim == 2 and tuple(scale.shape) == (1, w.shape[1]):
        pass
    else:
        raise ValueError(
            f"Unsupported scale shape {tuple(scale.shape)} for {key} {tuple(w.shape)}."
        )
    return scale, block


def _rotation(info, width):
    """Validate ConvRot instead of silently skipping an incompatible rotation."""
    if not info or not info.get("convrot", False):
        return 0
    group = int(info.get("convrot_groupsize", 256))
    if group < 4 or width % group:
        raise ValueError(
            f"ConvRot group size {group} does not divide input width {width}."
        )
    _hadamard(group, "cpu")
    return group


def _validate_layout(info, weight, key):
    """Reject packed/unknown integer layouts and metadata that contradicts storage."""
    supported = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.int8,
    )
    if weight.dtype not in supported:
        raise ValueError(
            f"Unsupported weight storage {weight.dtype} in {key}; packed layouts require explicit metadata."
        )
    if not info:
        return
    expected = {
        "int8": (torch.int8,),
        "int8_tensorwise": (torch.int8,),
        "float8_e4m3fn": (torch.float8_e4m3fn,),
        "float8_e5m2": (torch.float8_e5m2,),
        "fp8": (torch.float8_e4m3fn, torch.float8_e5m2),
    }
    if info["format"] in expected and weight.dtype not in expected[info["format"]]:
        raise ValueError(
            f"Quantization format {info['format']} contradicts {weight.dtype} storage in {key}."
        )


def _requantize_rows(w, work_device):
    """Dense fp32 [out,in] -> (int8 weight, fp32 per-row scale, 0). Lets any source format live as int8 in VRAM."""
    w = w.to(work_device, torch.float32)
    scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
    q = torch.round(w / scale).clamp_(-128, 127).to(torch.int8)
    return q.contiguous(), scale.reshape(-1).contiguous(), 0


def _dequantize(f, keyset, key, dtype, work_device="cpu"):
    """Decode one weight using its exact layout; never reinterpret packed codes as INT8."""
    info = _read_comfy_quant(f, keyset, key[: -len(".weight")])
    if info and info["format"] == "asym_w4a8_int8":
        return W4A8Linear.from_checkpoint(f, keyset, key).to(work_device).dense(dtype)
    w = f.get_tensor(key)
    _validate_layout(info, w, key)
    scale, block = _scale(f, keyset, key, w)
    w = w.to(device=work_device, dtype=torch.float32)
    if scale is not None:
        scale = scale.to(work_device)
        w = w * (_expand_block_scale(scale, *w.shape) if block else scale)
    group = _rotation(info, w.shape[1]) if w.ndim == 2 else 0
    if group:
        # TF32 truncation would introduce an additional approximation when
        # undoing the checkpoint's orthogonal rotation in the dense path.
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32 if w.is_cuda else None
        try:
            if w.is_cuda:
                torch.backends.cuda.matmul.allow_tf32 = False
            w = (w.reshape(w.shape[0], -1, group) @ _hadamard(group, w.device)).reshape(
                w.shape
            )
        finally:
            if previous_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return w.to(dtype)


# --------------------------------------------------------------------------------------
# Compressed (int8 / fp8) linear layer that stays small in VRAM
# --------------------------------------------------------------------------------------
# ConvRot stores W_rot = W_g @ H^T (per group of columns). Since H is symmetric and orthonormal,
#   y = x @ W^T = sum_g (x_g @ H) @ W_rot_g^T
# so instead of un-rotating the weights we rotate the (tiny) activations and keep the int8 weights as they are.

_ACT_H_CACHE = {}
_FP8_TRITON = {
    dtype: {"state": None} for dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
}
_TRITON = {"state": None}  # None = not tested yet, True = verified, False = disabled

try:
    import triton
    import triton.language as tl

    _GEMV_CONFIGS = [
        triton.Config(
            {"BLOCK_N": 8, "BLOCK_K": 512}, num_warps=4
        ),  # previous fixed setting
        triton.Config({"BLOCK_N": 4, "BLOCK_K": 512}, num_warps=4),
        triton.Config({"BLOCK_N": 4, "BLOCK_K": 1024}, num_warps=4),
        triton.Config({"BLOCK_N": 8, "BLOCK_K": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 8, "BLOCK_K": 1024}, num_warps=8),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 512}, num_warps=8),
        triton.Config({"BLOCK_N": 2, "BLOCK_K": 1024}, num_warps=2),
    ]

    def _autotune():
        try:  # newer Triton can remember the winner between sessions
            return triton.autotune(
                configs=_GEMV_CONFIGS, key=["N", "K"], cache_results=True
            )
        except TypeError:
            return triton.autotune(configs=_GEMV_CONFIGS, key=["N", "K"])

    @_autotune()
    @triton.jit
    def _gemv_quant_kernel(
        x_ptr, w_ptr, s_ptr, y_ptr, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        row_off = offs_n[:, None].to(tl.int64) * K
        acc = tl.zeros(
            [BLOCK_N, BLOCK_K], dtype=tl.float32
        )  # reduce once at the end, not every iteration
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            xv = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            wv = tl.load(
                w_ptr + row_off + offs_k[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += wv * xv[None, :]
        res = tl.sum(acc, axis=1)
        sc = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
        tl.store(y_ptr + offs_n, (res * sc).to(y_ptr.dtype.element_ty), mask=n_mask)

    _HAVE_TRITON = True
except Exception:
    _HAVE_TRITON = False
    _TRITON["state"] = False


def _act_hadamard(group_size, device, dtype):
    key = (group_size, str(device), dtype)
    if key not in _ACT_H_CACHE:
        _ACT_H_CACHE[key] = _hadamard(group_size, "cpu").to(device=device, dtype=dtype)
    return _ACT_H_CACHE[key]


class QuantLinear(torch.nn.Module):
    """y = (rotate(x) @ W_q^T) * row_scale, with W_q kept as int8 / fp8 in memory."""

    def __init__(self, weight, scale, group_size):
        super().__init__()
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("scale", scale, persistent=False)  # float32, shape [out]
        self.register_buffer("bias", None, persistent=False)
        self.group_size = int(group_size)
        self.out_features, self.in_features = weight.shape

    def _triton_gemv(self, x2):
        # Verify each storage dtype separately; INT8 success says nothing about FP8.
        state = (
            _TRITON
            if self.weight.dtype == torch.int8
            else _FP8_TRITON.get(self.weight.dtype)
        )
        if state is None or state["state"] is False or not _HAVE_TRITON:
            return None
        if not x2.is_cuda or x2.dtype not in (torch.bfloat16, torch.float16):
            return None
        try:
            x2 = x2.contiguous()
            y = torch.empty((1, self.out_features), dtype=x2.dtype, device=x2.device)

            def grid(meta):
                """Launch one program per output-row tile."""
                return (triton.cdiv(self.out_features, meta["BLOCK_N"]),)

            _gemv_quant_kernel[grid](
                x2, self.weight, self.scale, y, self.out_features, self.in_features
            )
            if (
                state["state"] is None
            ):  # one-time self-check against the plain PyTorch path
                ref = self._torch_linear(x2)
                err = (
                    (y.float() - ref.float()).norm() / (ref.float().norm() + 1e-9)
                ).item()
                state["state"] = err < 0.02
                if not state["state"]:
                    print(
                        f"[Prompt Refiner] Triton {self.weight.dtype} kernel gave a different result (rel. error {err:.3g}); using PyTorch path."
                    )
                    return ref
            return y
        except Exception as e:
            state["state"] = False
            print(
                f"[Prompt Refiner] Triton {self.weight.dtype} kernel disabled: {type(e).__name__}: {e}"
            )
            return None

    def _torch_linear(self, x2):
        y = torch.nn.functional.linear(x2, self.weight.to(x2.dtype))
        return (y.float() * self.scale).to(x2.dtype)

    def forward(self, x):
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        if self.group_size:
            g = self.group_size
            h = _act_hadamard(g, x2.device, x2.dtype)
            x2 = torch.matmul(x2.reshape(-1, self.in_features // g, g), h).reshape(
                -1, self.in_features
            )
        y = self._triton_gemv(x2) if x2.shape[0] == 1 else None
        if y is None:
            y = self._torch_linear(x2)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*lead, self.out_features)


def _load_quantized(f, keyset, key):
    """Return the compressed one-byte row-scaled representation, or request dense fallback."""
    info = _read_comfy_quant(f, keyset, key[: -len(".weight")])
    if info and info["format"] == "asym_w4a8_int8":
        return W4A8Linear.from_checkpoint(f, keyset, key)
    w = f.get_tensor(key)
    _validate_layout(info, w, key)
    if (
        w.dtype not in (torch.int8, torch.float8_e4m3fn, torch.float8_e5m2)
        or w.ndim != 2
    ):
        return None
    scale, block = _scale(f, keyset, key, w)
    group = _rotation(info, w.shape[1])
    if (
        scale is not None
        and scale.ndim == 2
        and scale.shape[0] == 1
        and scale.shape[1] > 1
    ):
        return None  # Explicit column scales must not become row scales on square matrices.
    if block or (scale is not None and scale.numel() not in (1, w.shape[0])):
        return None
    if scale is None:
        scale = torch.ones(w.shape[0], dtype=torch.float32)
    else:
        scale = scale.reshape(-1).expand(w.shape[0]).contiguous()
    return w.contiguous(), scale, group


class W4A8Linear(torch.nn.Module):
    """Keep Neo's packed W4A8 codes, group scales, channel scales, and codebook resident.

    Comfy-Kitchen selects its native backend. A dense per-layer fallback preserves
    correctness on devices without an eligible INT8 kernel; weights stay packed
    between calls and no second quantization is performed.
    """

    def __init__(
        self, weight, relative, channel, codebook, correction, group_size, rotation
    ):
        super().__init__()
        for name, tensor in (
            ("weight", weight),
            ("relative", relative),
            ("channel", channel),
            ("codebook", codebook),
            ("correction", correction),
        ):
            self.register_buffer(name, tensor, persistent=False)
        self.group_size, self.rotation = group_size, rotation
        self.out_features = weight.shape[0]
        self.in_features = relative.shape[1] * group_size
        self.register_buffer("bias", None, persistent=False)
        self._fallback = False
        self.params(torch.float32)  # validate before loading the rest of a model

    @classmethod
    def from_checkpoint(cls, f, keys, key):
        """Read the canonical W4A8 layout, including optional asymmetric correction."""
        base = key[: -len(".weight")]
        info = _read_comfy_quant(f, keys, base)
        group = int(info.get("group_size", 16))
        rotation = int(info.get("convrot_groupsize", 256))

        def read(suffix, required=False):
            name = key + suffix
            if name in keys:
                return f.get_tensor(name)
            if required:
                raise ValueError(f"Missing W4A8 tensor: {name}")
            return None

        weight = f.get_tensor(key)
        relative, channel = read("_s_rel", True), read("_s_channel", True)
        if relative.dtype == torch.uint8:
            relative = relative.view(torch.float8_e4m3fn)
        if relative.ndim != 2 or weight.ndim != 2 or weight.dtype != torch.int8:
            raise ValueError(f"Invalid W4A8 tensor layout in {key}.")
        width = relative.shape[1] * group
        if weight.shape != (relative.shape[0], width // 2):
            raise ValueError(
                f"W4A8 packed shape does not match its group scales: {key}"
            )
        if not info.get("convrot", True):
            raise ValueError(f"Unrotated W4A8 is not supported: {key}")
        _rotation({"convrot": True, "convrot_groupsize": rotation}, width)
        return cls(
            weight,
            relative,
            channel,
            read("_codebook"),
            read("_correction"),
            group,
            rotation,
        )

    def params(self, dtype):
        """Reconstruct layout parameters from registered buffers after device transfers."""
        try:
            from comfy_kitchen.tensor import AsymW4A8Int8Layout
        except ImportError as exc:
            raise RuntimeError(
                "W4A8 requires Forge Neo's Comfy-Kitchen with AsymW4A8Int8Layout support."
            ) from exc
        return AsymW4A8Int8Layout.Params(
            scale=self.relative,
            s_channel=self.channel,
            correction=self.correction,
            codebook=self.codebook,
            group_size=self.group_size,
            convrot_groupsize=self.rotation,
            orig_dtype=dtype,
            orig_shape=(self.out_features, self.in_features),
        )

    def dense(self, dtype):
        """Decode through the upstream layout, including codebook and rotation."""
        from comfy_kitchen.tensor import AsymW4A8Int8Layout

        return AsymW4A8Int8Layout.dequantize(self.weight, self.params(dtype))

    def forward(self, x):
        from comfy_kitchen.tensor import QuantizedTensor

        if not self._fallback:
            try:
                weight = QuantizedTensor(
                    self.weight, "AsymW4A8Int8Layout", self.params(x.dtype)
                )
                return torch.nn.functional.linear(x, weight, self.bias)
            except (NotImplementedError, RuntimeError) as exc:
                # Never hide an out-of-memory failure behind another allocation.
                if isinstance(exc, torch.OutOfMemoryError):
                    raise
                self._fallback = True
                print(f"[Prompt Refiner] W4A8 dense per-layer fallback: {exc}")
        return torch.nn.functional.linear(x, self.dense(x.dtype), self.bias)


class GGUFLinear(torch.nn.Module):
    """Use Forge's packed GGUF tensors and dequantize only the active linear layer."""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.out_features, self.in_features = weight.shape
        self.register_buffer("bias", None, persistent=False)

    def forward(self, x):
        from backend.operations_gguf import dequantize_tensor

        weight = (
            dequantize_tensor(self.weight, input=x)
            .as_subclass(torch.Tensor)
            .to(x.dtype)
        )
        return torch.nn.functional.linear(x, weight, self.bias)
