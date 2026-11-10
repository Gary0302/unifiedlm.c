"""
Ablation variants for the unification experiment. Not part of the standard model.

    S  model.UnifiedBlock          y = mu(r w/c) + fast(r)    shared address, evidence attenuates the static read
    D  DecoupledBlock              y = mu(r)     + fast(r)    shared address, no attenuation
    P  SeparateBlock(m_s, m_f)     y = mu(r_s)   + fast(r_f)  one address per path, two slot sets
    A  SeparateBlock alternating   y = fast(r_f), then y = mu(r_s)
    T  llama2.c Transformer        attention with RoPE, then a SwiGLU feed-forward of hidden_dim units, per block pair

fast(r) is unified_memory(r, v, g, w): read address r, write address r shifted by one, normalized by
c = w + decayed evidence. Variants only train and evaluate; forward_recurrent, export and run.c stay
with the standard model until a variant wins.
"""
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import model
from kernel import unified_memory
from model import RMSNorm, UnifiedBlock, _hi


class DecoupledBlock(UnifiedBlock):

    def forward(self, x):
        h = self.norm(x)
        u = self.conv(h.transpose(1, 2))[..., : h.size(1)].transpose(1, 2)
        r = self.code(u)
        _, fast = unified_memory(r, self.val(h), -self.log_rate.exp(), self._w())
        return x + self.dropout(self.mu(r.type_as(x)) + fast.type_as(x))

    def step(self, x, s):
        raise NotImplementedError("the inherited token-by-token form computes S, not D")


class SeparateBlock(nn.Module):
    """m_s static slots and m_f context slots with independent addresses; either count may be 0."""

    def __init__(self, args, m_s, m_f):
        super().__init__()
        d = args.dim
        self.norm = RMSNorm(d, eps=args.norm_eps)
        self.conv = nn.Conv1d(d, d, args.conv_size, groups=d, padding=args.conv_size - 1, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        if m_s:
            self.key_s = nn.Linear(d, m_s, bias=True)
            self.log_eps_s = nn.Parameter(torch.zeros(()))
            self.mu = nn.Linear(m_s, d, bias=False)
        if m_f:
            self.key_f = nn.Linear(d, m_f, bias=True)
            self.log_eps_f = nn.Parameter(torch.zeros(()))
            self.val = nn.Linear(d, d, bias=False)
            self.log_rate = nn.Parameter(torch.tensor(math.log(1.0 / (4 * args.max_seq_len))))
            self.log_w = nn.Parameter(torch.full((m_f,), math.log(args.prior)))
        self.m_s, self.m_f = m_s, m_f

    @staticmethod
    def _code(key, log_eps, u):
        z = F.relu(_hi(key(u))).square()
        return z / (log_eps.clamp(-10, 10).exp() + z.sum(-1, keepdim=True))

    def forward(self, x):
        h = self.norm(x)
        u = self.conv(h.transpose(1, 2))[..., : h.size(1)].transpose(1, 2)
        y = 0
        if self.m_s:
            y = y + self.mu(self._code(self.key_s, self.log_eps_s, u).type_as(x))
        if self.m_f:
            r = self._code(self.key_f, self.log_eps_f, u)
            w = self.log_w.clamp(-7.0, 12.0).exp()
            _, fast = unified_memory(r, self.val(h), -self.log_rate.exp(), w)
            y = y + fast.type_as(x)
        return x + self.dropout(y)


HEAD_DIM = 48  # llama2.c 15M: 6 heads of 48 at dim 288


def _rope(x):
    T, hd = x.size(-2), x.size(-1)
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=x.device).float() / hd))
    ang = torch.outer(torch.arange(T, device=x.device).float(), inv)
    c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2).type_as(x)


class AttentionHalf(nn.Module):
    """The attention half of a llama2.c layer. The output projection is named mu so that UnifiedLM's init scales
    it to 0.02/sqrt(n_layers), which with two blocks per layer is GPT-2's scaling of residual projections."""

    def __init__(self, args):
        super().__init__()
        d = args.dim
        self.h = d // HEAD_DIM
        self.norm = RMSNorm(d, eps=args.norm_eps)
        self.wq, self.wk, self.wv, self.mu = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        B, T, d = x.shape
        a = self.norm(x)
        q, k, v = (w(a).view(B, T, self.h, HEAD_DIM).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        y = F.scaled_dot_product_attention(_rope(q), _rope(k), v, is_causal=True)
        return x + self.dropout(self.mu(y.transpose(1, 2).reshape(B, T, d)))


class FeedForwardHalf(nn.Module):
    """The SwiGLU feed-forward half of a llama2.c layer; mu is its output projection (w2 in llama2.c)."""

    def __init__(self, args):
        super().__init__()
        d, hidden = args.dim, args.hidden_dim
        self.norm = RMSNorm(d, eps=args.norm_eps)
        self.w1, self.w3 = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.mu = nn.Linear(hidden, d, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        f = self.norm(x)
        return x + self.dropout(self.mu(F.silu(self.w1(f)) * self.w3(f)))


def install(variant, m_s=0, m_f=0):
    """UnifiedLM builds its layers through the global model.UnifiedBlock, so swapping it is enough."""
    if variant == "S":
        model.UnifiedBlock = UnifiedBlock
    elif variant == "D":
        model.UnifiedBlock = DecoupledBlock
    elif variant == "P":
        model.UnifiedBlock = lambda args: SeparateBlock(args, m_s, m_f)
    elif variant == "A":
        i = itertools.count()
        model.UnifiedBlock = lambda args: SeparateBlock(args, 0, m_f) if next(i) % 2 == 0 else SeparateBlock(args, m_s, 0)
    elif variant == "SG":  # S with a per-slot, per-token forget gate read off the address, scripts/gated.py
        import gated
        model.UnifiedBlock = gated.GatedBlock
    elif variant == "SR":  # SG's ablation: per-slot rates, no gate
        import gated
        model.UnifiedBlock = gated.RateBlock
    elif variant == "T":
        i = itertools.count()
        model.UnifiedBlock = lambda args: AttentionHalf(args) if next(i) % 2 == 0 else FeedForwardHalf(args)
    else:
        raise ValueError(variant)
