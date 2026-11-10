"""The winning configuration from the numpy-llm study, in PyTorch:
NorMuon-trained GPT with SwiGLU, LayerNorm (pre-norm) and RoPE.

It mirrors nllm/nn.py exactly (same layout, same init), so a model trained
with the old framework can be loaded here and gives the same outputs
(see minigpt/convert.py). Differences from nllm are only in *how* it runs:
PyTorch autograd, GPU kernels and fused attention.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 81
    ctx: int = 128          # context length (tokens)
    n_layer: int = 2
    d_model: int = 64
    n_head: int = 4
    rope_base: float = 10000.0


class LayerNorm(nn.Module):
    """Same as nllm: mean/variance over the last dim, learned gain and bias."""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.gain, self.bias, self.eps)


def rope_tables(T, head_dim, base, device, dtype):
    """cos/sin tables (T, head_dim), 'rotate-half' layout, as in nllm/positional.py."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(T, device=device, dtype=torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin):
    """Rotate pairs (x[i], x[i + D/2]) by a position-dependent angle."""
    h = x.shape[-1] // 2
    rotated = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head, self.hd = cfg.n_head, cfg.d_model // cfg.n_head
        self.rope_base = cfg.rope_base
        self.wqkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.wqkv(x).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, self.hd).transpose(1, 2) for t in (q, k, v))  # (B, H, T, D)
        cos, sin = rope_tables(T, self.hd, self.rope_base, x.device, q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # fused softmax(q k^T / sqrt(D) + causal mask) v; uses FlashAttention-style kernels when available
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    """FFN with a SiLU gate. Hidden width 8d/3 keeps the parameter count of a 4d plain FFN."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        h = int(round(8 * cfg.d_model / 3))
        self.w_gate = nn.Linear(cfg.d_model, h, bias=False)
        self.w_up = nn.Linear(cfg.d_model, h, bias=False)
        self.w_down = nn.Linear(h, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1, self.attn = LayerNorm(cfg.d_model), Attention(cfg)
        self.norm2, self.ffn = LayerNorm(cfg.d_model), SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))          # pre-norm residual blocks
        return x + self.ffn(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        """GPT-2 init, as in nllm: std 0.02, layers writing into the residual stream 0.02/sqrt(2L)."""
        resid_std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
        for name, p in self.named_parameters():
            if p.ndim == 2:
                std = resid_std if name.endswith(("wo.weight", "w_down.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.norm_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8, top_k=20, generator=None):
        """Plain sampling (no KV-cache yet: that comes in the serving step)."""
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.cfg.ctx:])
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
