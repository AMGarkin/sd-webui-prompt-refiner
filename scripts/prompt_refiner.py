"""
Prompt Refiner - extension for Forge Classic / Neo

Adds a "Prompt Refiner" tab that rewrites / expands prompts with a local LLM.
The LLM is the very same text encoder file you already use for image generation
(e.g. qwen3vl_4b_fp8_scaled.safetensors from models/text_encoder).
No Ollama / LM Studio / external server is needed.

Supported (v1): Qwen3 and Qwen3-VL text encoders (bf16 / fp16 / scaled fp8 / int8 / int8 ConvRot .safetensors)
Not supported (v1): GGUF files, Gemma, Qwen2.5-VL, images as input.
"""

import gc
import json
import queue
import random
import re
import threading
import time
from pathlib import Path

import gradio as gr
import torch

from modules import script_callbacks, shared
from modules.paths import models_path, script_path

EXT_DIR = Path(__file__).resolve().parents[1]
PRESETS_FILE = EXT_DIR / "presets.json"
LOCAL_TOKENIZER_DIR = EXT_DIR / "tokenizer"

_RUN_LOCK = threading.Lock()
_STOP = threading.Event()

try:
    from transformers import StoppingCriteria, StoppingCriteriaList
    from transformers.generation.streamers import BaseStreamer
except Exception:  # keeps the tab loadable even if transformers changes its layout
    StoppingCriteria = StoppingCriteriaList = BaseStreamer = object

# --------------------------------------------------------------------------------------
# System prompt presets
# --------------------------------------------------------------------------------------

DEFAULT_PRESETS = {
    "Natural language (Krea / Flux / Qwen-Image / Z-Image)": (
        "You are an expert prompt writer for modern text-to-image models that understand natural language.\n"
        "Rewrite the user's input into one rich, vivid, well-structured description.\n"
        "Rules:\n"
        "- Start with the main subject and what it is doing, then describe environment, composition, lighting, "
        "materials/textures, colors, mood, and camera/lens details where it makes sense.\n"
        "- Keep every detail the user gave; never contradict it. Invent only fitting details.\n"
        "- Use flowing prose, no bullet points, no headings, no negations ('no', 'without').\n"
        "- If the user wants text in the image, keep it in \"quotation marks\" and say where it is placed.\n"
        "- Write 2-3 paragraphs at most.\n"
        "Respond ONLY with the final prompt. No introduction, no explanation, no quotes around it."
    ),
    "Short idea -> detailed prompt": (
        "You turn a short idea into a detailed image generation prompt.\n"
        "Expand the idea with a concrete subject, setting, composition, lighting, style and mood. "
        "Stay faithful to the idea. Output a single paragraph of plain prose.\n"
        "Respond ONLY with the final prompt."
    ),
    "Danbooru tags (SDXL / Anima / anime)": (
        "You convert and improve prompts for anime-style image models that use Danbooru tags.\n"
        "Output a single comma-separated line of lowercase tags: subject count, character, "
        "appearance, clothing, pose, expression, setting, lighting, composition, quality tags.\n"
        "Keep all concepts from the user's input. Do not write sentences.\n"
        "Respond ONLY with the tag line."
    ),
    "Comma-separated keywords (SD 1.5 / SDXL)": (
        "You improve prompts for Stable Diffusion 1.5 / SDXL.\n"
        "Output one line of comma-separated keywords and short phrases: subject, details, setting, "
        "lighting, camera, style, quality boosters. Keep everything from the user's input.\n"
        "Respond ONLY with the prompt line."
    ),
    "Light polish (keep wording)": (
        "You lightly polish an image generation prompt: fix grammar and spelling, remove redundancy, "
        "and add at most a few precise visual details. Keep the user's wording, order and language style "
        "as much as possible.\n"
        "Respond ONLY with the final prompt."
    ),
}


def load_presets():
    if PRESETS_FILE.is_file():
        try:
            data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return data
        except Exception:
            pass
    save_presets(DEFAULT_PRESETS)
    return dict(DEFAULT_PRESETS)


def save_presets(data):
    PRESETS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Finding text encoders and tokenizer
# --------------------------------------------------------------------------------------

_ENCODER_PATHS = {}
_SIGNATURE_CACHE = {}


def _encoder_dirs():
    dirs = [Path(models_path) / "text_encoder", Path(models_path) / "text_encoders"]
    extra = getattr(shared.cmd_opts, "text_encoder_dir", None)
    if extra:
        dirs.append(Path(extra))
    return [d for d in dirs if d.is_dir()]


def _is_qwen3_file(path: Path) -> bool:
    """Cheap check (header only): Qwen3 has q_norm / k_norm in every attention block."""
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return False
    if key in _SIGNATURE_CACHE:
        return _SIGNATURE_CACHE[key]
    ok = False
    try:
        from safetensors import safe_open

        with safe_open(str(path), "pt", device="cpu") as f:
            ok = any(k.endswith("layers.0.self_attn.q_norm.weight") for k in f.keys())
    except Exception:
        ok = False
    _SIGNATURE_CACHE[key] = ok
    return ok


def list_encoders():
    _ENCODER_PATHS.clear()
    for d in _encoder_dirs():
        for p in sorted(d.rglob("*.safetensors")):
            if _is_qwen3_file(p):
                _ENCODER_PATHS.setdefault(str(p.relative_to(d)), p)
    return list(_ENCODER_PATHS.keys())


def find_tokenizer_dir() -> Path:
    """Prefer a tokenizer shipped with the extension, otherwise reuse one shipped with Forge."""
    if (LOCAL_TOKENIZER_DIR / "tokenizer.json").is_file() or (LOCAL_TOKENIZER_DIR / "vocab.json").is_file():
        return LOCAL_TOKENIZER_DIR

    root = Path(script_path) / "backend" / "huggingface"
    candidates = []
    if root.is_dir():
        for tj in root.rglob("tokenizer.json"):
            candidates.append(tj.parent)
        for vj in root.rglob("vocab.json"):
            if (vj.parent / "merges.txt").is_file() and vj.parent not in candidates:
                candidates.append(vj.parent)

    def score(p: Path):
        s = str(p).lower()
        return (("qwen3" in s) * 2 + ("qwen" in s), -len(s))

    candidates = [c for c in candidates if "qwen" in str(c).lower()]
    if candidates:
        return sorted(candidates, key=score, reverse=True)[0]

    raise RuntimeError(
        "No Qwen tokenizer found. Download tokenizer.json, tokenizer_config.json, vocab.json and merges.txt "
        "from https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct and put them into:\n" + str(LOCAL_TOKENIZER_DIR)
    )


# --------------------------------------------------------------------------------------
# Loading a Qwen3 text encoder file into a HF Qwen3ForCausalLM
# --------------------------------------------------------------------------------------

_LAYER_RE = re.compile(r"^(embed_tokens|layers\.\d+|norm)\.")
_SCALE_SUFFIXES = (".scale_weight", ".weight_scale", ".weight_scale_inv", ".scale_input", ".input_scale", ".weight_scale_2", ".comfy_quant")


def _pick_dtype(name: str, device: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    # auto
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.bfloat16


_HADAMARD_CACHE = {}


def _hadamard(group_size: int, device):
    """Normalized 'regular' Hadamard matrix used by ConvRot: Kronecker power of H4, scaled by 1/sqrt(n).
    It is symmetric and orthonormal, so it is its own inverse."""
    key = (group_size, str(device))
    if key not in _HADAMARD_CACHE:
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
        h = torch.ones(1, 1)
        n = 1
        while n < group_size:
            h = torch.kron(h, h4)
            n *= 4
        if n != group_size:
            raise RuntimeError(f"Unsupported ConvRot group size {group_size} (expected a power of 4).")
        _HADAMARD_CACHE[key] = (h / (group_size ** 0.5)).to(device)
    return _HADAMARD_CACHE[key]


def _read_comfy_quant(f, keyset, base):
    name = base + ".comfy_quant"
    if name not in keyset:
        return None
    try:
        return json.loads(bytes(f.get_tensor(name).tolist()).decode("utf-8"))
    except Exception:
        return None


def _expand_block_scale(s, n, k):
    """Expand a [ceil(n/bn), ceil(k/bk)] block-scale grid to [n, k]."""
    bn = -(-n // s.shape[0])
    bk = -(-k // s.shape[1])
    return s.repeat_interleave(bn, 0)[:n].repeat_interleave(bk, 1)[:k]


def _requantize_rows(w, work_device):
    """Dense fp32 [out,in] -> (int8 weight, fp32 per-row scale, 0). Lets any source format live as int8 in VRAM."""
    w = w.to(work_device, torch.float32)
    scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
    q = torch.round(w / scale).clamp_(-128, 127).to(torch.int8)
    return q.contiguous(), scale.reshape(-1).contiguous(), 0


def _dequantize(f, keyset, key: str, dtype, work_device="cpu"):
    """Load f[key] as a dense tensor.
    Handles: plain bf16/fp16, scaled fp8, row/tensor-scaled int8 and int8 + ConvRot (Hadamard-rotated)."""
    w = f.get_tensor(key)
    base = key[: -len(".weight")]
    cq = _read_comfy_quant(f, keyset, base)
    fmt = str(cq.get("format", "")) if cq else ""
    if cq and not any(x in fmt for x in ("float8", "int8", "fp8")):
        raise RuntimeError(f"Unsupported quantization format '{fmt}' in {key} (supported: fp8 / int8 / int8 ConvRot).")

    scale = None
    for suffix in (".scale_weight", ".weight_scale", ".weight_scale_inv"):
        if base + suffix in keyset:
            scale = f.get_tensor(base + suffix)
            break

    if scale is None and w.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8):
        return w.to(dtype)  # ordinary bf16/fp16 weight

    prev_tf32 = None
    if torch.cuda.is_available():
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False  # keep the un-rotation exact
    try:
        w = w.to(work_device, torch.float32)
        if scale is not None:
            s = scale.to(work_device, torch.float32)
            if s.ndim == 1 and s.numel() > 1 and w.ndim == 2:
                s = s.view(-1, 1) if s.numel() == w.shape[0] else s.view(1, -1)
            elif s.ndim == 2 and w.ndim == 2 and not (s.shape[0] == w.shape[0] and s.shape[1] == 1):
                s = _expand_block_scale(s, w.shape[0], w.shape[1])  # e.g. Qwen FP8: one scale per 128x128 block
            w = w * s
        if cq and cq.get("convrot") and w.ndim == 2:
            gs = int(cq.get("convrot_groupsize", 256))
            out_f, in_f = w.shape
            if in_f % gs == 0:
                w = (w.view(out_f, in_f // gs, gs) @ _hadamard(gs, w.device)).reshape(out_f, in_f)
        return w.to(dtype)
    finally:
        if prev_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32


# --------------------------------------------------------------------------------------
# Compressed (int8 / fp8) linear layer that stays small in VRAM
# --------------------------------------------------------------------------------------
# ConvRot stores W_rot = W_g @ H^T (per group of columns). Since H is symmetric and orthonormal,
#   y = x @ W^T = sum_g (x_g @ H) @ W_rot_g^T
# so instead of un-rotating the weights we rotate the (tiny) activations and keep the int8 weights as they are.

_ACT_H_CACHE = {}
_TRITON = {"state": None}  # None = not tested yet, True = verified, False = disabled

try:
    import triton
    import triton.language as tl

    _GEMV_CONFIGS = [
        triton.Config({"BLOCK_N": 8, "BLOCK_K": 512}, num_warps=4),   # previous fixed setting
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
            return triton.autotune(configs=_GEMV_CONFIGS, key=["N", "K"], cache_results=True)
        except TypeError:
            return triton.autotune(configs=_GEMV_CONFIGS, key=["N", "K"])

    @_autotune()
    @triton.jit
    def _gemv_int8_kernel(x_ptr, w_ptr, s_ptr, y_ptr, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        row_off = offs_n[:, None].to(tl.int64) * K
        acc = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)  # reduce once at the end, not every iteration
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            xv = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + row_off + offs_k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0).to(tl.float32)
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
        self.group_size = int(group_size)
        self.out_features, self.in_features = weight.shape

    def _triton_gemv(self, x2):
        if _TRITON["state"] is False or not _HAVE_TRITON:
            return None
        if self.weight.dtype != torch.int8 or not x2.is_cuda or x2.dtype not in (torch.bfloat16, torch.float16):
            return None
        try:
            x2 = x2.contiguous()
            y = torch.empty((1, self.out_features), dtype=x2.dtype, device=x2.device)
            grid = lambda meta: (triton.cdiv(self.out_features, meta["BLOCK_N"]),)
            _gemv_int8_kernel[grid](x2, self.weight, self.scale, y, self.out_features, self.in_features)
            if _TRITON["state"] is None:  # one-time self-check against the plain PyTorch path
                ref = self._torch_linear(x2)
                err = ((y.float() - ref.float()).norm() / (ref.float().norm() + 1e-9)).item()
                _TRITON["state"] = err < 0.02
                if not _TRITON["state"]:
                    print(f"[Prompt Refiner] Triton kernel gave a different result (rel. error {err:.3g}); using PyTorch path.")
                    return ref
            return y
        except Exception as e:
            _TRITON["state"] = False
            print(f"[Prompt Refiner] Triton kernel disabled: {type(e).__name__}: {e}")
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
            x2 = torch.matmul(x2.reshape(-1, self.in_features // g, g), h).reshape(-1, self.in_features)
        y = self._triton_gemv(x2) if x2.shape[0] == 1 else None
        if y is None:
            y = self._torch_linear(x2)
        return y.reshape(*lead, self.out_features)


def _load_quantized(f, keyset, key):
    """Return (weight_1byte, row_scale_fp32[out], convrot_group) for a layer we can keep compressed, else None."""
    w = f.get_tensor(key)
    if w.dtype not in (torch.int8, torch.float8_e4m3fn, torch.float8_e5m2) or w.ndim != 2:
        return None
    base = key[: -len(".weight")]
    scale = None
    for suffix in (".scale_weight", ".weight_scale", ".weight_scale_inv"):
        if base + suffix in keyset:
            scale = f.get_tensor(base + suffix)
            break
    if scale is None:
        return None
    cq = _read_comfy_quant(f, keyset, base)
    if cq and not any(x in str(cq.get("format", "")) for x in ("float8", "int8", "fp8")):
        raise RuntimeError(f"Unsupported quantization format '{cq.get('format')}' in {key}.")
    out_f, in_f = w.shape
    if scale.ndim == 2 and scale.shape[1] != 1:
        return None  # block-wise grid (e.g. Qwen FP8 128x128): handled by re-quantization instead
    s = scale.to(torch.float32).reshape(-1)
    if s.numel() == 1:
        s = s.expand(out_f).contiguous()
    elif s.numel() != out_f:
        return None  # block-wise scales: fall back to full dequantization
    gs = 0
    if cq and cq.get("convrot"):
        g = int(cq.get("convrot_groupsize", 256))
        gs = g if in_f % g == 0 else 0
        if gs:
            _hadamard(gs, "cpu")  # validates the group size early
    return w.contiguous(), s, gs


_PROJ_RE = re.compile(r"^layers\.\d+\.(self_attn|mlp)\.\w+_proj\.weight$")


def build_model(path: Path, dtype, rope_choice: str, work_device="cpu", keep_quantized=True, compress=False):
    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    with safe_open(str(path), "pt", device="cpu") as f:
        keys = list(f.keys())
        keyset = set(keys)

        # Detect the key prefix (Comfy files: "model.", Qwen3-VL HF layout: "model.language_model.", bare: "")
        anchor = next((k for k in keys if k.endswith("layers.0.self_attn.q_proj.weight")), None)
        if anchor is None:
            raise RuntimeError("This file does not look like a Qwen3 text encoder.")
        prefix = anchor[: -len("layers.0.self_attn.q_proj.weight")]

        def shape(k):
            return list(f.get_slice(k).get_shape())

        vocab, hidden = shape(prefix + "embed_tokens.weight")
        layer_ids = {int(m.group(1)) for k in keys if k.startswith(prefix) for m in [re.match(r"layers\.(\d+)\.", k[len(prefix):])] if m}
        n_layers = max(layer_ids) + 1
        head_dim = shape(prefix + "layers.0.self_attn.q_norm.weight")[0]
        n_heads = shape(prefix + "layers.0.self_attn.q_proj.weight")[0] // head_dim
        n_kv = shape(prefix + "layers.0.self_attn.k_proj.weight")[0] // head_dim
        inter = shape(prefix + "layers.0.mlp.gate_proj.weight")[0]

        has_lm_head = "lm_head.weight" in keyset
        is_vl = (any(".visual." in k or k.startswith("visual.") for k in keys)
                 or "vl" in path.name.lower() or "2507" in path.name)  # Qwen3-VL and the 2507 refreshes use 5M
        if rope_choice.startswith("1"):
            theta = 1_000_000.0
        elif rope_choice.startswith("5"):
            theta = 5_000_000.0
        else:
            theta = 5_000_000.0 if is_vl else 1_000_000.0

        cfg = Qwen3Config(
            vocab_size=vocab,
            hidden_size=hidden,
            intermediate_size=inter,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv,
            head_dim=head_dim,
            max_position_embeddings=40960,
            rms_norm_eps=1e-6,
            rope_theta=theta,
            tie_word_embeddings=not has_lm_head,
            attention_bias=False,
        )
        # newer transformers keep the theta inside rope_parameters
        rp = getattr(cfg, "rope_parameters", None)
        if isinstance(rp, dict):
            rp["rope_theta"] = theta

        if not has_lm_head and n_layers >= 36 and hidden >= 4096:
            raise RuntimeError(
                "This encoder file has no 'lm_head' weights and a model of this size does not tie them to the "
                "embeddings, so it cannot generate text. Use a Qwen3 / Qwen3-VL 4B (or smaller) file, "
                "or a full LLM checkpoint."
            )

        with torch.device("meta"):
            model = Qwen3ForCausalLM(cfg)
        # rotary buffers must be real tensors (they are not part of the checkpoint)
        model.model.rotary_emb = Qwen3RotaryEmbedding(config=cfg, device="cpu")

        sd = {}
        quant = {}  # module path -> (weight, scale, group)
        for k in keys:
            if not k.startswith(prefix) or k.endswith(_SCALE_SUFFIXES) or k.endswith("scaled_fp8"):
                continue
            rest = k[len(prefix):]
            if not _LAYER_RE.match(rest):
                continue
            if keep_quantized and _PROJ_RE.match(rest):
                q = _load_quantized(f, keyset, k)
                if q is None:
                    one_byte = f.get_slice(k).get_dtype() in ("F8_E4M3", "F8_E5M2", "I8")
                    if one_byte or compress:  # block-scaled fp8, or user asked to shrink bf16/fp16 weights
                        q = _requantize_rows(_dequantize(f, keyset, k, torch.float32, work_device), work_device)
                if q is not None:
                    quant["model." + rest[: -len(".weight")]] = q
                    continue
            if k.endswith(".weight") and len(shape(k)) == 2:
                sd["model." + rest] = _dequantize(f, keyset, k, dtype, work_device)
            else:
                sd["model." + rest] = f.get_tensor(k).to(dtype)
        if has_lm_head:
            sd["lm_head.weight"] = _dequantize(f, keyset, "lm_head.weight", dtype, work_device)

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [m for m in missing if (m != "lm_head.weight" or has_lm_head) and m[: -len(".weight")] not in quant]
    if missing:
        raise RuntimeError(f"Missing weights in encoder file: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if not has_lm_head:
        model.lm_head.weight = model.model.embed_tokens.weight
    for mod_path, (qw, qs, qg) in quant.items():
        parent_path, _, attr = mod_path.rpartition(".")
        setattr(model.get_submodule(parent_path), attr, QuantLinear(qw, qs, qg))
    model._pr_quant_layers = len(quant)
    model.eval()
    info = f"Qwen3 {n_layers} layers, hidden {hidden}, heads {n_heads}/{n_kv}, rope_theta {theta:g}"
    if quant:
        info += f", {len(quant)} layers kept compressed"
    return model, info


# --------------------------------------------------------------------------------------
# Runtime state
# --------------------------------------------------------------------------------------

_STATE = {"key": None, "model": None, "tokenizer": None, "info": ""}


def unload_model():
    _STATE.update(key=None, model=None, tokenizer=None, info="")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "Model unloaded."


def _free_forge_vram():
    try:
        from backend import memory_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_model(encoder_name, device_choice, dtype_choice, rope_choice, free_vram, keep_q=True, compress=False):
    path = _ENCODER_PATHS.get(encoder_name)
    if path is None:
        list_encoders()
        path = _ENCODER_PATHS.get(encoder_name)
    if path is None:
        raise RuntimeError("Select a text encoder first (put Qwen3 / Qwen3-VL files into models/text_encoder).")

    device = "cuda" if (device_choice.startswith("auto") and torch.cuda.is_available()) or device_choice == "cuda" else "cpu"
    dtype = _pick_dtype(dtype_choice, device)
    key = (str(path), device, str(dtype), rope_choice, bool(keep_q), bool(compress))

    if _STATE["key"] == key and _STATE["model"] is not None:
        return _STATE["model"], _STATE["tokenizer"], device

    unload_model()
    if device == "cuda" and free_vram:
        _free_forge_vram()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(find_tokenizer_dir()))
    model, info = build_model(path, dtype, rope_choice, device, keep_q, compress)
    model.to(device)  # no dtype argument on purpose: keeps rotary buffers in fp32

    _STATE.update(key=key, model=model, tokenizer=tokenizer, info=info)
    return model, tokenizer, device


def _build_chat(system: str, user: str, empty_think: bool) -> str:
    text = ""
    if system.strip():
        text += f"<|im_start|>system\n{system.strip()}<|im_end|>\n"
    text += f"<|im_start|>user\n{user.strip()}<|im_end|>\n<|im_start|>assistant\n"
    if empty_think:
        text += "<think>\n\n</think>\n\n"
    return text


def _clean_output(text: str) -> str:
    if "</think>" in text and "<think>" not in text:  # template already opened the block (Thinking-2507)
        text = "<think>" + text
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    stripped = re.sub(r"<think>.*$", "", stripped, flags=re.DOTALL)  # unfinished thinking
    stripped = stripped.strip()
    if not stripped:
        stripped = text.replace("<think>", "").replace("</think>", "").strip()
    stripped = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", stripped).strip()
    if len(stripped) > 1 and stripped[0] == stripped[-1] and stripped[0] in "\"'":
        stripped = stripped[1:-1].strip()
    return stripped


class _TokenStreamer(BaseStreamer):
    """Collects generated token ids so the UI thread can show live progress."""

    def __init__(self, q):
        self.q = q
        self.ids = []
        self._skip_prompt = True
        self.t_first = None

    def put(self, value):
        if self._skip_prompt:  # first call carries the prompt itself
            self._skip_prompt = False
            return
        if self.t_first is None:
            self.t_first = time.time()
        self.ids.extend(value.reshape(-1).tolist())
        self.q.put("tok")

    def end(self):
        self.q.put("end")


class _StopFlag(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), _STOP.is_set(), dtype=torch.bool, device=input_ids.device)


def _bar(frac, text, busy=False):
    pct = 100 if frac is None else max(0, min(100, int(frac * 100)))
    anim = "animation:prpulse 1.1s ease-in-out infinite;" if busy else ""
    return (
        "<style>@keyframes prpulse{0%,100%{opacity:.35}50%{opacity:1}}</style>"
        f'<div style="margin:4px 0"><div style="font-size:0.9em;margin-bottom:3px">{text}</div>'
        '<div style="height:12px;border-radius:6px;overflow:hidden;background:var(--background-fill-secondary);'
        'border:1px solid var(--border-color-primary)">'
        f'<div style="height:100%;width:{pct}%;background:var(--color-accent);transition:width .2s;{anim}"></div>'
        "</div></div>"
    )


def _describe(model, device):
    params = list(model.parameters())
    p = params[0]
    s = f"device {p.device}, {str(p.dtype).replace('torch.', '')}, attention: {getattr(model.config, '_attn_implementation', '?')}"
    nq = getattr(model, "_pr_quant_layers", 0)
    if nq:
        k = {True: "Triton (verified)", False: "PyTorch fallback", None: "not used yet"}[_TRITON["state"]]
        s += f", {nq} compressed layers, GEMV kernel: {k}"
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        s += f", GPU {torch.cuda.get_device_name(0)}, VRAM in use {(total - free) / 1e9:.1f}/{total / 1e9:.1f} GB"
        n_cpu = sum(1 for q in params if q.device.type == "cpu")
        if n_cpu:
            s += f" - WARNING: {n_cpu} weight tensors are still on the CPU"
    return s


def run_refine(
    encoder_name, prompt, system_prompt, temperature, top_p, top_k, min_p, rep_penalty,
    max_new_tokens, seed, device_choice, dtype_choice, rope_choice, keep_loaded, free_vram, empty_think,
    keep_q=True, compress=False,
):
    """Generator: streams the result text, a status line and a progress bar while generating."""
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt first.")
    if not encoder_name:
        raise gr.Error("No Qwen3 / Qwen3-VL text encoder found in models/text_encoder.")

    max_new_tokens = int(max_new_tokens)
    with _RUN_LOCK:
        _STOP.clear()
        t0 = time.time()
        try:
            cached = _STATE["model"] is not None
            msg = "Loading model..." if cached else "Loading model (reading file, decoding weights - first run only)..."
            yield gr.update(), msg, _bar(None, msg, busy=True)

            model, tokenizer, device = _get_model(encoder_name, device_choice, dtype_choice, rope_choice, free_vram, keep_q, compress)
            t_load = time.time() - t0

            text = _build_chat(system_prompt, prompt, empty_think)
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
            n_prompt = enc["input_ids"].shape[1]

            seed = int(seed)
            if seed < 0:
                seed = random.randint(0, 2**31 - 1)
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed)

            eos_ids = [i for i in (tokenizer.convert_tokens_to_ids("<|im_end|>"),
                                   tokenizer.convert_tokens_to_ids("<|endoftext|>")) if isinstance(i, int) and i >= 0]
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_ids or None,
                pad_token_id=eos_ids[-1] if eos_ids else 0,
                repetition_penalty=float(rep_penalty),
            )
            if temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=float(top_p), top_k=int(top_k))
                if min_p > 0:
                    gen_kwargs["min_p"] = float(min_p)
            else:
                gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

            q = queue.Queue()
            streamer = _TokenStreamer(q)
            gen_kwargs["streamer"] = streamer
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_StopFlag()])
            errors = []

            def worker():
                try:
                    with torch.inference_mode():
                        model.generate(**enc, **gen_kwargs)
                except BaseException as e:  # reported in the UI thread
                    errors.append(e)
                finally:
                    q.put("end")

            t1 = time.time()
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            def snapshot(final=False):
                ids = list(streamer.ids)
                n = len(ids)
                rate = (n - 1) / (time.time() - streamer.t_first) if streamer.t_first and n > 1 else 0.0
                shown = _clean_output(tokenizer.decode(ids, skip_special_tokens=True)) if ids else ""
                if streamer.t_first is None:
                    label = f"Reading prompt ({n_prompt} tokens)..."
                else:
                    label = f"Generating... {n}/{max_new_tokens} tokens - {rate:.1f} tok/s"
                return shown, n, rate, label

            last = 0.0
            finished = False
            while not finished:
                try:
                    finished = q.get(timeout=0.2) == "end"
                except queue.Empty:
                    pass
                now = time.time()
                if not finished and now - last >= 0.25:
                    shown, n, rate, label = snapshot()
                    yield shown, label, _bar(n / max_new_tokens if n else None, label, busy=(n == 0))
                    last = now

            thread.join()
            if errors:
                raise errors[0]

            result, n, rate, _ = snapshot(final=True)
            t_gen = time.time() - t1
            t_first = (streamer.t_first - t1) if streamer.t_first else t_gen
            stopped = " (stopped by user)" if _STOP.is_set() else ""
            limit_note = ""
            if n >= max_new_tokens and not _STOP.is_set():
                limit_note = " Token limit reached - raise 'Max new tokens' (thinking models often need 2000+)."
            status = (f"Done{stopped}.{limit_note} {n} tokens in {t_gen:.1f}s ({rate:.1f} tok/s), prompt processing {t_first:.1f}s, "
                      f"model load {t_load:.1f}s, seed {seed}.\n{_STATE['info']}; {_describe(model, device)}")
            yield result, status, _bar(1.0, f"Done{stopped} - {n} tokens, {rate:.1f} tok/s")
        except gr.Error:
            raise
        except Exception as e:
            unload_model()
            raise gr.Error(f"{type(e).__name__}: {e}")
        finally:
            if not keep_loaded:
                unload_model()


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------

SEND_JS = """
(text) => {
    const ta = gradioApp().querySelector('#TAB_prompt textarea');
    if (ta) {
        ta.value = text;
        ta.dispatchEvent(new Event('input', {bubbles: true}));
    }
    if (typeof switch_to_TAB === 'function') { switch_to_TAB(); }
    return [];
}
"""


def on_ui_tabs():
    presets = load_presets()
    preset_names = list(presets.keys())
    encoders = list_encoders()

    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(
            "Rewrite or expand prompts with your own text encoder (Qwen3 / Qwen3-VL). "
            "Runs locally inside Forge - no Ollama or LM Studio needed."
        )
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    encoder = gr.Dropdown(
                        label="Text encoder (models/text_encoder)", choices=encoders,
                        value=encoders[0] if encoders else None, scale=10,
                    )
                    refresh_btn = gr.Button("🔄", elem_classes=["tool"], scale=1, min_width=40)
                prompt = gr.Textbox(label="Your prompt / idea", lines=6, placeholder="a cat astronaut on mars, cinematic")
                preset = gr.Dropdown(label="System prompt preset", choices=preset_names,
                                     value=preset_names[0] if preset_names else None)
                system_prompt = gr.Textbox(label="System prompt (instructions for the LLM)", lines=9,
                                           value=presets[preset_names[0]] if preset_names else "")
                with gr.Row():
                    preset_name = gr.Textbox(label="Preset name", scale=3, placeholder="My preset")
                    save_btn = gr.Button("Save preset", scale=1)
                    delete_btn = gr.Button("Delete selected", scale=1)

            with gr.Column(scale=1):
                with gr.Row():
                    run_btn = gr.Button("Refine prompt", variant="primary", scale=4)
                    stop_btn = gr.Button("Stop", scale=1)
                progress_html = gr.HTML(value="")
                output = gr.Textbox(label="Result", lines=14, interactive=True, show_copy_button=True)
                status = gr.Textbox(label="Status", interactive=False, lines=2)
                with gr.Row():
                    send_txt = gr.Button("Send to txt2img")
                    send_img = gr.Button("Send to img2img")
                    reuse_btn = gr.Button("Use result as input")
                    unload_btn = gr.Button("Unload model")

        with gr.Accordion("Sampling & model options", open=False):
            with gr.Row():
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature (0 = greedy)")
                top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p")
                top_k = gr.Slider(0, 200, value=64, step=1, label="Top-k (0 = off)")
                min_p = gr.Slider(0.0, 1.0, value=0.05, step=0.01, label="Min-p (0 = off)")
            with gr.Row():
                rep_penalty = gr.Slider(1.0, 2.0, value=1.05, step=0.01, label="Repetition penalty")
                max_new = gr.Slider(32, 4096, value=768, step=16, label="Max new tokens")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            with gr.Row():
                device = gr.Dropdown(["auto (GPU if available)", "cuda", "cpu"], value="auto (GPU if available)", label="Device")
                dtype = gr.Dropdown(["auto", "bfloat16", "float16", "float32"], value="auto", label="Compute dtype")
                rope = gr.Dropdown(["Auto (Qwen3-VL / 2507: 5M, older Qwen3: 1M)", "1,000,000 (Qwen3)", "5,000,000 (Qwen3-VL)"],
                                   value="Auto (Qwen3-VL / 2507: 5M, older Qwen3: 1M)", label="RoPE theta")
            with gr.Row():
                keep_loaded = gr.Checkbox(value=False, label="Keep model loaded after generation (uses VRAM!)")
                free_vram = gr.Checkbox(value=True, label="Unload Forge's image models from VRAM before loading the LLM")
                keep_q = gr.Checkbox(value=True, label="Keep int8 / fp8 weights compressed in VRAM (recommended for 8 GB cards)")
                compress = gr.Checkbox(value=False, label="Also compress bf16 / fp16 weights to int8 on load (lets 8 GB cards run full-size files; tiny quality loss)")
                empty_think = gr.Checkbox(value=False, label="Add empty <think></think> block (only for Qwen3 hybrid-thinking models)")

        # ---- events ----
        def _refresh(current):
            names = list_encoders()
            return gr.update(choices=names, value=current if current in names else (names[0] if names else None))

        refresh_btn.click(_refresh, inputs=[encoder], outputs=[encoder])

        preset.change(lambda n: load_presets().get(n, ""), inputs=[preset], outputs=[system_prompt])

        def _save(name, text):
            name = (name or "").strip()
            if not name:
                raise gr.Error("Enter a preset name.")
            data = load_presets()
            data[name] = text
            save_presets(data)
            return gr.update(choices=list(data.keys()), value=name)

        save_btn.click(_save, inputs=[preset_name, system_prompt], outputs=[preset])

        def _delete(name):
            data = load_presets()
            data.pop(name, None)
            if not data:
                data = dict(DEFAULT_PRESETS)
            save_presets(data)
            first = next(iter(data))
            return gr.update(choices=list(data.keys()), value=first), data[first]

        delete_btn.click(_delete, inputs=[preset], outputs=[preset, system_prompt])

        run_btn.click(
            run_refine,
            inputs=[encoder, prompt, system_prompt, temperature, top_p, top_k, min_p, rep_penalty,
                    max_new, seed, device, dtype, rope, keep_loaded, free_vram, empty_think, keep_q, compress],
            outputs=[output, status, progress_html],
            show_progress="hidden",
        )
        stop_btn.click(lambda: _STOP.set(), inputs=None, outputs=None)
        reuse_btn.click(lambda t: t, inputs=[output], outputs=[prompt])
        unload_btn.click(unload_model, outputs=[status])

        send_txt.click(fn=None, inputs=[output], outputs=None, js=SEND_JS.replace("TAB", "txt2img"))
        send_img.click(fn=None, inputs=[output], outputs=None, js=SEND_JS.replace("TAB", "img2img"))

    return [(ui, "Prompt Refiner", "prompt_refiner")]


script_callbacks.on_ui_tabs(on_ui_tabs)
