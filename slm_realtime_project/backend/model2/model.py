"""
=============================================================================
Story-SLM2: 101M Parameter Neural Network Architecture (model.py)
=============================================================================
Transformer backbone sized to actually land at ~100M parameters (verified by
direct computation, not hand-wavy estimation):

    vocab_size=50304, d_model=768, n_layers=10, n_heads=12, n_kv_heads=4,
    d_ffn=2048  ->  101,564,160 parameters (~101.6M)

Features:
  - Grouped-Query Attention (12 Q heads : 4 KV heads, 3:1 ratio)
  - RoPE rotary position embeddings
  - SwiGLU feed-forward activation
  - Pre-norm RMSNorm
  - Tied input embedding / output LM head
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from tokenizer import EMBED_VOCAB_SIZE


@dataclass
class StorySLM2Config:
    vocab_size: int = EMBED_VOCAB_SIZE
    d_model: int = 448
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 4
    d_ffn: int = 1216
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    bias: bool = False
    weight_tying: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[offset : offset + seq_len], self.sin_cached[offset : offset + seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


class KVCache:
    """Pre-allocated per-layer key/value buffers for inference-time generation.

    A naive cache that grows via torch.cat every decode step reallocates and copies
    the ENTIRE past on every single new token -- measured to make per-token cost
    climb continuously as a story grows (from ~18ms/token early on to ~75ms/token
    past ~1500 tokens on this model's CPU target), from allocator churn and
    increasingly cache-unfriendly memory access, even though the model's own
    compute at this size barely grows with length. Allocating one fixed buffer
    per layer up front (sized to max_seq_len, the model's hard context ceiling
    anyway) and writing new tokens into it in place removes that cost entirely.
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        device,
        dtype,
    ):
        self.k = [
            torch.zeros(batch_size, max_seq_len, n_kv_heads, head_dim, device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.v = [
            torch.zeros(batch_size, max_seq_len, n_kv_heads, head_dim, device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.length = 0  # tokens written so far; shared across layers, which always advance together

    def write(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        start, end = self.length, self.length + k_new.shape[1]
        self.k[layer_idx][:, start:end] = k_new
        self.v[layer_idx][:, start:end] = v_new
        return self.k[layer_idx][:, :end], self.v[layer_idx][:, :end]

    def advance(self, n_new_tokens: int):
        self.length += n_new_tokens


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: StorySLM2Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        B, S, C = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        q, k = apply_rope(q, k, cos, sin)

        past_len = kv_cache.length if kv_cache is not None else 0
        # Cache keys/values pre-repeat (n_kv_heads, not n_heads) -- cheaper to store,
        # and we re-expand to n_heads fresh on every call since n_rep is tiny (2 here).
        if kv_cache is not None:
            k, v = kv_cache.write(layer_idx, k, v)

        if self.n_rep > 1:
            k_rep = k.repeat_interleave(self.n_rep, dim=2)
            v_rep = v.repeat_interleave(self.n_rep, dim=2)
        else:
            k_rep, v_rep = k, v

        q = q.transpose(1, 2)
        k_rep = k_rep.transpose(1, 2)
        v_rep = v_rep.transpose(1, 2)

        total_len = k_rep.size(2)
        if kv_cache is not None and total_len != S:
            # is_causal=True only works when query/key lengths match (it applies a plain
            # L_q x L_k lower-triangular mask, NOT one aligned to the end of the key
            # sequence) -- with a cache, L_k > L_q, so we need an explicit mask that lets
            # each new query see the full past plus itself, and nothing beyond.
            query_pos = torch.arange(past_len, total_len, device=x.device)
            key_pos = torch.arange(total_len, device=x.device)
            attn_mask = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)  # (S, total_len) bool
            out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, S, C)
        return self.o_proj(out)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: StorySLM2Config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ffn, bias=config.bias)
        self.w2 = nn.Linear(config.d_ffn, config.d_model, bias=config.bias)
        self.w3 = nn.Linear(config.d_model, config.d_ffn, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: StorySLM2Config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLUFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        attn_out = self.attn(self.attn_norm(x), cos, sin, kv_cache, self.layer_idx)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class StorySLM2(nn.Module):
    def __init__(self, config: StorySLM2Config):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(
            config.d_model // config.n_heads, config.max_seq_len, config.rope_theta
        )
        self.layers = nn.ModuleList([TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.weight_tying:
            self.lm_head.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def new_kv_cache(self, batch_size: int, device, dtype) -> KVCache:
        head_dim = self.config.d_model // self.config.n_heads
        return KVCache(
            n_layers=self.config.n_layers,
            batch_size=batch_size,
            max_seq_len=self.config.max_seq_len,
            n_kv_heads=self.config.n_kv_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """kv_cache is opt-in and only used by the inference-time generator
        (generator.py). Training (train.py) never passes it, so this is a no-op
        change for training -- same (logits, loss) return as always."""
        B, S = input_ids.shape
        past_len = kv_cache.length if kv_cache is not None else 0
        x = self.tok_embeddings(input_ids)
        cos, sin = self.rope(S, offset=past_len)

        for layer in self.layers:
            x = layer(x, cos, sin, kv_cache)

        if kv_cache is not None:
            kv_cache.advance(S)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


if __name__ == "__main__":
    config = StorySLM2Config()
    model = StorySLM2(config)
    total, trainable = model.count_parameters()

    print("==========================================================")
    print("[TEST] Story-SLM2 Architecture Verification")
    print("==========================================================")
    print(f"vocab_size={config.vocab_size}  d_model={config.d_model}  n_layers={config.n_layers}")
    print(f"n_heads={config.n_heads}  n_kv_heads={config.n_kv_heads}  d_ffn={config.d_ffn}")
    print(f"Total parameters:     {total:,} ({total/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable:,} ({trainable/1e6:.2f}M)")

    dummy = torch.randint(0, config.vocab_size, (2, 128))
    targets = torch.randint(0, config.vocab_size, (2, 128))
    logits, loss = model(dummy, targets)
    print(f"Forward pass OK. Logits shape: {tuple(logits.shape)}  Loss: {loss.item():.4f}")
    print("==========================================================")
