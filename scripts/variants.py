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


class CausalConv(nn.Module):
    """SD's causal depthwise conv over time (size 4, no activation). UnifiedLM's init starts it as the identity."""

    def __init__(self, d, size=4):
        super().__init__()
        self.conv = nn.Conv1d(d, d, size, groups=d, padding=size - 1, bias=False)

    def forward(self, x):
        return self.conv(x.transpose(1, 2))[..., : x.size(1)].transpose(1, 2)


class AttentionHalf(nn.Module):
    """The attention half of a llama2.c layer. The output projection is named mu so that UnifiedLM's init scales
    it to 0.02/sqrt(n_layers), which with two blocks per layer is GPT-2's scaling of residual projections.
    conv=True (TC) passes the normed input through SD's causal conv before q, k, v."""

    def __init__(self, args, conv=False):
        super().__init__()
        d = args.dim
        self.h = d // HEAD_DIM
        self.norm = RMSNorm(d, eps=args.norm_eps)
        self.mix = CausalConv(d) if conv else nn.Identity()
        self.wq, self.wk, self.wv, self.mu = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        B, T, d = x.shape
        a = self.mix(self.norm(x))
        q, k, v = (w(a).view(B, T, self.h, HEAD_DIM).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        y = F.scaled_dot_product_attention(_rope(q), _rope(k), v, is_causal=True)
        return x + self.dropout(self.mu(y.transpose(1, 2).reshape(B, T, d)))


class WindowAttentionHalf(AttentionHalf):
    """AttentionHalf where each query sees only the last `window` keys (itself included): SDA's local attention.
    With RoPE and a band this short, it behaves the same at any sequence length."""

    def __init__(self, args, window=128):
        super().__init__(args)
        self.window = window

    def forward(self, x):
        B, T, d = x.shape
        a = self.norm(x)
        q, k, v = (w(a).view(B, T, self.h, HEAD_DIM).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        i = torch.arange(T, device=x.device)
        band = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - self.window)
        y = F.scaled_dot_product_attention(_rope(q), _rope(k), v, attn_mask=band)
        return x + self.dropout(self.mu(y.transpose(1, 2).reshape(B, T, d)))


class GatedDeltaHalf(nn.Module):
    """Gated DeltaNet (Yang, Kautz, Hatamizadeh, ICLR 2025, arXiv 2412.06464) in place of attention: the GDN
    baseline, a recurrent (SSM-family) Transformer. Per head, with unit q_t, k_t (q also scaled by 1/sqrt(dk)):
        S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T,   o_t = S_t^T q_t
        alpha_t = exp(-exp(A) softplus(W_a h + dt)),  beta_t = sigmoid(W_b h)
    then o -> RMSNorm per head -> * SiLU(W_g h) -> output projection, as in the paper and fla's GatedDeltaNet.
    q, k, v pass a causal depthwise conv (size 4) and SiLU. delta.delta_memory computes the recurrence (chunk 32);
    log alpha is clamped at -2 per step so its within-chunk exp(-cumsum) stays finite in fp32. soft=True (GDNs)
    bounds it smoothly instead, 2 expm1(log alpha / 2) in (-2, 0], so a head near the bound still gets a gradient."""

    def __init__(self, args, soft=False):
        super().__init__()
        self.soft = soft
        d = args.dim
        self.h = d // HEAD_DIM
        self.norm = RMSNorm(d, eps=args.norm_eps)
        self.wq, self.wk, self.wv, self.wg, self.mu = (nn.Linear(d, d, bias=False) for _ in range(5))
        self.conv = nn.Conv1d(3 * d, 3 * d, 4, groups=3 * d, padding=3, bias=False)
        self.wa, self.wb = nn.Linear(d, self.h, bias=False), nn.Linear(d, self.h, bias=False)
        self.A_log = nn.Parameter(torch.empty(self.h).uniform_(1, 16).log())          # fla's init
        dt = torch.exp(torch.rand(self.h) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_shift = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))                # softplus^-1(dt)
        self.onorm = RMSNorm(HEAD_DIM, eps=args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        import delta
        B, T, d = x.shape
        H, hd = self.h, HEAD_DIM
        a = self.norm(x)
        qkv = torch.cat([self.wq(a), self.wk(a), self.wv(a)], -1).transpose(1, 2)
        q, k, v = F.silu(self.conv(qkv)[..., :T]).transpose(1, 2).chunk(3, -1)
        heads = lambda z: z.reshape(B, T, H, hd).transpose(1, 2).reshape(B * H, T, hd)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = (heads(_hi(z)) for z in (q, k, v))
            q = F.normalize(q, dim=-1) / math.sqrt(hd)
            k = F.normalize(k, dim=-1)
            beta = torch.sigmoid(_hi(self.wb(a))).transpose(1, 2).reshape(B * H, T)
            la = -self.A_log.exp() * F.softplus(_hi(self.wa(a)) + self.dt_shift)
            la = (2 * torch.expm1(la / 2) if self.soft else la.clamp(min=-2.0)).transpose(1, 2).reshape(B * H, T, 1).expand(B * H, T, hd)
            o = delta.delta_memory(k, v, beta, la, q, chunk=32)
        o = self.onorm(o.reshape(B, H, T, hd).transpose(1, 2)).reshape(B, T, d).type_as(x)
        return x + self.dropout(self.mu(o * F.silu(self.wg(a))))


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
    elif variant == "SG64":  # SG with a rank-64 gate bottleneck
        import gated
        model.UnifiedBlock = lambda args: gated.GatedBlock(args, rank=64)
    elif variant == "SGh":  # SG with the gate read off the block input h instead of the address r
        import gated
        model.UnifiedBlock = lambda args: gated.GatedBlock(args, gate_from="h")
    elif variant == "SGs":  # SG with a SiLU inside the rank-16 gate (a two-layer MLP gate)
        import gated
        model.UnifiedBlock = lambda args: gated.GatedBlock(args, silu=True)
    elif variant == "SR":  # SG's ablation: per-slot rates, no gate
        import gated
        model.UnifiedBlock = gated.RateBlock
    elif variant == "SH":  # control: S with its forgetting rate started at 1/64
        import gated
        model.UnifiedBlock = gated.ShortBlock
    elif variant == "SD":  # S with a delta-rule memory anchored at the FFN, scripts/delta.py
        import delta
        model.UnifiedBlock = delta.DeltaBlock
    elif variant == "SDp":  # SD with a learnable address power per layer (sharpness), from 2
        import delta
        model.UnifiedBlock = delta.PowerDeltaBlock
    elif variant == "SD2":  # SD plus a slow memory and a read mix
        import delta
        model.UnifiedBlock = delta.TwoDeltaBlock
    elif variant == "SDA":  # hybrid: after every 4 SD blocks a local attention half (window 128); n_layers counts both
        import delta
        i = itertools.count()
        model.UnifiedBlock = lambda args: WindowAttentionHalf(args) if next(i) % 5 == 4 else delta.DeltaBlock(args)
    elif variant == "GDN":  # baseline: T with Gated DeltaNet in place of attention
        import delta  # before UnifiedBlock is replaced: delta imports gated, which subclasses it
        i = itertools.count()
        model.UnifiedBlock = lambda args: GatedDeltaHalf(args) if next(i) % 2 == 0 else FeedForwardHalf(args)
    elif variant == "GDNs":  # GDN with the soft bound on log alpha; run with large heads (--head_dim=144)
        import delta
        i = itertools.count()
        model.UnifiedBlock = lambda args: GatedDeltaHalf(args, soft=True) if next(i) % 2 == 0 else FeedForwardHalf(args)
    elif variant == "T":
        i = itertools.count()
        model.UnifiedBlock = lambda args: AttentionHalf(args) if next(i) % 2 == 0 else FeedForwardHalf(args)
    elif variant == "TC":  # control: T with SD's causal conv before attention
        i = itertools.count()
        model.UnifiedBlock = lambda args: AttentionHalf(args, conv=True) if next(i) % 2 == 0 else FeedForwardHalf(args)
    else:
        raise ValueError(variant)
