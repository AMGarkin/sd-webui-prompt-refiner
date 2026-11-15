"""Text-only Qwen3.5 hybrid inference for Forge's existing Transformers 4.x runtime.

Implements Qwen3.5's published gated attention and Gated DeltaNet equations.
References: Hugging Face transformers/models/qwen3_5 (Apache-2.0) and Neo's
backend/nn/llm/llama.py. Vision and multi-token prediction are not instantiated.
GenerationMixin supplies sampling, repetition penalties, streaming, and stopping;
request-local caches preserve full-attention KV and linear convolution/FP32 state.
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class Qwen35Config(PretrainedConfig):
    """Dense 9B text defaults; explicit smaller configs are useful for regression tests."""

    model_type = "prompt_refiner_qwen35"

    def __init__(self, **kwargs):
        defaults = dict(
            vocab_size=248320,
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=32,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=256,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            partial_rotary_factor=0.25,
            max_position_embeddings=262144,
            rms_norm_eps=1e-6,
            rope_theta=10000000.0,
            hidden_act="silu",
            tie_word_embeddings=False,
            eos_token_id=248046,
            pad_token_id=248044,
            use_cache=True,
        )
        defaults.update(kwargs)
        defaults.setdefault(
            "layer_types",
            [
                "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                for i in range(defaults["num_hidden_layers"])
            ],
        )
        super().__init__(**defaults)


class HybridCache:
    """Keep independent per-request state; no recurrent state survives a generation call."""

    is_compileable = False

    def __init__(self):
        self.length = 0
        self.kv = {}
        self.conv = {}
        self.recurrent = {}

    def get_seq_length(self, *args):
        return self.length


class RMSNorm(nn.Module):
    """Qwen3.5 uses zero-centered weights except in its gated DeltaNet output norm."""

    def __init__(self, width, eps, additive=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(width) if additive else torch.ones(width)
        )
        self.eps, self.additive = eps, additive

    def forward(self, x, gate=None):
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        if self.additive:
            return (normalized * (1 + self.weight.float())).to(x.dtype)
        output = self.weight * normalized.to(x.dtype)
        return (output * F.silu(gate.float())).to(x.dtype)


def chunk_delta_rule(query, key, value, decay, beta, initial, chunk=32):
    """Solve causal delta updates within small chunks, then scan only chunk states.

    Inputs are already normalized FP32 [batch, heads, tokens, width]. The lower
    triangular system is the parallel form of the recurrent correction equation.
    Padding uses zero updates/decay, preserving the exact final recurrent state.
    """
    length = query.shape[2]
    pad = (-length) % chunk
    q, k, v = [F.pad(t, (0, 0, 0, pad)) for t in (query, key, value)]
    b, g = [F.pad(t, (0, pad)) for t in (beta, decay)]
    batch, heads, total, _ = k.shape
    q, k, v = [
        t.reshape(batch, heads, total // chunk, chunk, t.shape[-1]) for t in (q, k, v)
    ]
    b, g = [t.reshape(batch, heads, total // chunk, chunk) for t in (b, g)]
    cumulative = g.cumsum(-1)
    difference = cumulative[..., :, None] - cumulative[..., None, :]
    # Mask before exp: upper-triangular entries can otherwise overflow for strong decay.
    lower = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    decay_matrix = difference.masked_fill(~lower, -torch.inf).exp()
    kb = k * b[..., None]
    system = (kb @ k.transpose(-1, -2) * decay_matrix).tril(-1)
    system = system + torch.eye(chunk, device=q.device)
    new_values = torch.linalg.solve_triangular(
        system, v * b[..., None], upper=False, unitriangular=True
    )
    carried_keys = torch.linalg.solve_triangular(
        system, kb * cumulative.exp()[..., None], upper=False, unitriangular=True
    )
    intra = q @ k.transpose(-1, -2) * decay_matrix
    state = initial
    if state is None:
        state = v.new_zeros(batch, heads, k.shape[-1], v.shape[-1])
    outputs = []
    for i in range(total // chunk):
        residual = new_values[:, :, i] - carried_keys[:, :, i] @ state
        outputs.append(
            (q[:, :, i] * cumulative[:, :, i].exp()[..., None]) @ state
            + intra[:, :, i] @ residual
        )
        final_decay = cumulative[:, :, i, -1]
        decayed_key = (
            k[:, :, i] * (final_decay[..., None] - cumulative[:, :, i]).exp()[..., None]
        )
        state = (
            state * final_decay.exp()[..., None, None]
            + decayed_key.transpose(-1, -2) @ residual
        )
    return torch.cat(outputs, dim=2)[:, :, :length], state


def delta_rule(query, key, value, decay, beta, initial=None):
    """FP32 recurrent gated delta update; also a reference path for prompt processing."""
    original_dtype = query.dtype
    if query.shape[2] != value.shape[2]:
        repeats = value.shape[2] // query.shape[2]
        query, key = (t.repeat_interleave(repeats, dim=2) for t in (query, key))
    query, key, value = (t.transpose(1, 2).float() for t in (query, key, value))
    query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    query = query * (query.shape[-1] ** -0.5)
    beta, decay = beta.transpose(1, 2).float(), decay.transpose(1, 2).float()
    if query.shape[2] > 1:
        out, state = chunk_delta_rule(query, key, value, decay, beta, initial)
        return out.transpose(1, 2).to(original_dtype), state
    state = initial
    if state is None:
        state = value.new_zeros(
            value.shape[0], value.shape[1], key.shape[-1], value.shape[-1]
        )
    outputs = []
    for i in range(query.shape[2]):
        state = state * decay[:, :, i].exp()[..., None, None]
        k, q = key[:, :, i], query[:, :, i]
        residual = (value[:, :, i] - (state * k[..., None]).sum(-2)) * beta[
            :, :, i, None
        ]
        state = state + k[..., None] * residual[..., None, :]
        outputs.append((state * q[..., None]).sum(-2))
    return torch.stack(outputs, dim=2).transpose(1, 2).to(original_dtype), state


class GatedDeltaNet(nn.Module):
    """Causal depthwise convolution followed by gated recurrent linear attention."""

    def __init__(self, config, index):
        super().__init__()
        self.config, self.index = config, index
        self.key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        self.value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        width = 2 * self.key_dim + self.value_dim
        self.in_proj_qkv = nn.Linear(config.hidden_size, width, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_a = nn.Linear(
            config.hidden_size, config.linear_num_value_heads, bias=False
        )
        self.in_proj_b = nn.Linear(
            config.hidden_size, config.linear_num_value_heads, bias=False
        )
        self.conv1d = nn.Conv1d(
            width, width, config.linear_conv_kernel_dim, groups=width, bias=False
        )
        self.A_log = nn.Parameter(
            torch.zeros(config.linear_num_value_heads, dtype=torch.float32)
        )
        self.dt_bias = nn.Parameter(torch.zeros(config.linear_num_value_heads))
        self.norm = RMSNorm(
            config.linear_value_head_dim, config.rms_norm_eps, additive=False
        )
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)

    def forward(self, x, cache):
        batch, length, _ = x.shape
        config = self.config
        projected = self.in_proj_qkv(x).transpose(1, 2)
        previous = cache.conv.get(self.index) if cache else None
        if previous is None:
            previous = projected.new_zeros(
                batch, projected.shape[1], config.linear_conv_kernel_dim - 1
            )
        combined = torch.cat((previous, projected), dim=-1)
        if cache is not None:
            cache.conv[self.index] = combined[
                :, :, -(config.linear_conv_kernel_dim - 1) :
            ].clone()
        mixed = F.silu(self.conv1d(combined)).transpose(1, 2)
        q, k, v = mixed.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)
        q, k = (
            t.reshape(
                batch, length, config.linear_num_key_heads, config.linear_key_head_dim
            )
            for t in (q, k)
        )
        v = v.reshape(
            batch, length, config.linear_num_value_heads, config.linear_value_head_dim
        )
        decay = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(x).float() + self.dt_bias.float()
        )
        beta = self.in_proj_b(x).sigmoid()
        out, state = delta_rule(
            q, k, v, decay, beta, cache.recurrent.get(self.index) if cache else None
        )
        if cache is not None:
            cache.recurrent[self.index] = state
        z = self.in_proj_z(x).reshape_as(out)
        out = self.norm(out, z).reshape(batch, length, self.value_dim)
        return self.out_proj(out)


class GatedAttention(nn.Module):
    """Full causal GQA with per-head output gates and partial text-only rotary positions."""

    def __init__(self, config, index):
        super().__init__()
        self.config, self.index = config, index
        inner = config.num_attention_heads * config.head_dim
        kv = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, 2 * inner, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv, bias=False)
        self.o_proj = nn.Linear(inner, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(self, x, positions, cache):
        batch, length, _ = x.shape
        c = self.config
        q, gate = (
            self.q_proj(x)
            .reshape(batch, length, c.num_attention_heads, 2 * c.head_dim)
            .chunk(2, -1)
        )
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(x).reshape(batch, length, c.num_key_value_heads, c.head_dim)
        ).transpose(1, 2)
        v = (
            self.v_proj(x)
            .reshape(batch, length, c.num_key_value_heads, c.head_dim)
            .transpose(1, 2)
        )
        rotary = int(c.head_dim * c.partial_rotary_factor)
        # All three multimodal position axes coincide for text-only input.
        freq = positions.float()[..., None] / (
            c.rope_theta
            ** (torch.arange(0, rotary, 2, device=x.device).float() / rotary)
        )
        angles = torch.cat((freq, freq), dim=-1).unsqueeze(1)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)

        def rotate(tensor):
            part, tail = tensor[..., :rotary], tensor[..., rotary:]
            left, right = part.chunk(2, -1)
            rotated = part * cos + torch.cat((-right, left), dim=-1) * sin
            return torch.cat((rotated, tail), dim=-1)

        q, k = rotate(q), rotate(k)
        past_length = 0
        if cache is not None:
            if self.index in cache.kv:
                old_k, old_v = cache.kv[self.index]
                past_length = old_k.shape[2]
                k, v = torch.cat((old_k, k), 2), torch.cat((old_v, v), 2)
            cache.kv[self.index] = k, v
        repeat = c.num_attention_heads // c.num_key_value_heads
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        mask = None
        if past_length and length > 1:
            mask = torch.arange(k.shape[2], device=x.device)[None, :] <= (
                past_length + torch.arange(length, device=x.device)[:, None]
            )
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=not past_length and length > 1
        )
        out = (
            out.transpose(1, 2).reshape(batch, length, -1)
            * gate.reshape(batch, length, -1).sigmoid()
        )
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU feed-forward block shared by both layer types."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Pre-norm hybrid decoder layer, retaining Hugging Face checkpoint tensor names."""

    def __init__(self, config, index):
        super().__init__()
        self.kind = config.layer_types[index]
        if self.kind == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, index)
        else:
            self.self_attn = GatedAttention(config, index)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, positions, cache):
        norm = self.input_layernorm(x)
        out = (
            self.linear_attn(norm, cache)
            if self.kind == "linear_attention"
            else self.self_attn(norm, positions, cache)
        )
        x = x + out
        return x + self.mlp(self.post_attention_layernorm(x))


class TextModel(nn.Module):
    """Language backbone only; deliberately omits vision and MTP weights."""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class Qwen35ForCausalLM(PreTrainedModel, GenerationMixin):
    """Bridge the cached hybrid decoder to Forge's existing GenerationMixin API."""

    config_class = Qwen35Config
    base_model_prefix = "model"
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def _supports_default_dynamic_cache(cls):
        # A KV-only DynamicCache cannot represent the convolution and delta state.
        return False

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, past_key_values.length :]
        return dict(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=kwargs.get("use_cache", True),
            logits_to_keep=1,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
        logits_to_keep=0,
        position_ids=None,
        return_dict=True,
        **kwargs,
    ):
        if (
            past_key_values is None
            and attention_mask is not None
            and not bool(attention_mask.all())
        ):
            raise ValueError("Qwen3.5 refinement requires an unpadded text prompt.")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        cache = past_key_values if use_cache else None
        if use_cache and cache is None:
            cache = HybridCache()
        offset = cache.length if cache else 0
        x = self.model.embed_tokens(input_ids)
        positions = position_ids
        if positions is None:
            positions = torch.arange(offset, offset + x.shape[1], device=x.device)[
                None, :
            ]
        for layer in self.model.layers:
            x = layer(x, positions, cache)
        if cache is not None:
            cache.length += input_ids.shape[1]
        x = self.model.norm(x)
        if logits_to_keep:
            x = x[:, -logits_to_keep:]
        logits = self.lm_head(x)
        return CausalLMOutputWithPast(logits=logits, past_key_values=cache)
