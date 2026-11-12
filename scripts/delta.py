"""
SD and SD2: the unified block with a delta-rule fast memory, anchored at the FFN. Not part of the standard model.

S keeps a running mean per slot, (w mu_a + sum k v) / (w + sum k): every write adds evidence, so a long memory
dilutes the static content mu (the trade-off SG's gates show: layers that remember longer keep less mu). SD stores
only what the FFN does not already predict, and overwrites by content instead of averaging:

    memory read   y_t = mu r_t + E_t^T q_t                    q_t = r_t / |r_t|  (unit read address)
    write target  u_t = v_t - mu k_t                          k_t = r_{t-1}, the residual the FFN leaves
    update        E_t = (I - beta_t kh_t kh_t^T) Lam_t E_{t-1} + beta_t kh_t u_t^T,   kh_t = k_t / |k_t|

E (m x d) is the deviation of the slots from mu, so the memory decays towards the FFN, as in S. The update is the
delta rule: it first looks the write address up in the memory (Lam E^T kh, what the slots predict there) and
writes beta times the error, so slots that already hold the value are not rewritten and what changes is only the
content being replaced. beta_t = sigmoid(W_beta h_t + b) is the write strength (how much this token matters);
Lam_t is SG's per-slot gate read off r_t, without SG's beta * k term (the delta rule does that job).

SD2 adds a second, slow memory with per-slot constant rates (SR's form, started at 1/(16 L)) and its own write
strength, read through a mixing gate rho_t = sigmoid(W_rho h_t): y = mu r + (1 - rho) E_fast^T q + rho E_slow^T q.
Tokens the write gates mark as important can live in the slow memory, away from the fast memory's turnover.

delta_memory computes the recurrence chunk by chunk (PyTorch autograd, fp32 with TF32): within a chunk the
pseudo-values delta = beta (u - (Lam E)^T kh) solve a unit lower-triangular system, across chunks a loop carries E.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gated import CHUNK, GatedBlock
from model import _hi


def delta_memory(k, u, beta, logl, q, chunk=CHUNK):
    """k: (B,T,m) unit write keys, u: (B,T,d) write targets, beta: (B,T) in [0, 1], logl: (B,T,m) log decay applied
    at step t (in [-1, 0]), q: (B,T,m) read addresses. With E_0 = 0,
        E_t = (I - beta_t k_t k_t^T) diag(lam_t) E_{t-1} + beta_t k_t u_t^T,   o_t = E_t^T q_t.
    Equivalently E_t = diag(lam_t) E_{t-1} + k_t delta_t^T with delta_t = beta_t (u_t - E_{t-1}^T diag(lam_t) k_t)."""
    B, T, m = k.shape
    d = u.size(-1)
    pad = (-T) % chunk
    if pad:
        k, u, logl, q = (F.pad(x, (0, 0, 0, pad)) for x in (k, u, logl, q))
        beta = F.pad(beta, (0, pad))
    N = (T + pad) // chunk
    kc, uc, lc, qc = (x.reshape(B, N, chunk, -1) for x in (k, u, logl, q))
    bc = beta.reshape(B, N, chunk)
    b = lc.cumsum(2)                          # log decay from the state entering the chunk to step i (inclusive)
    bl = b[:, :, -1]
    kd = kc * torch.exp(b)                    # D_i k_i
    ki = kc * torch.exp(-b)                   # D_j^-1 k_j
    qd = qc * torch.exp(b)                    # D_i q_i
    ks = kc * torch.exp(bl[:, :, None] - b)   # D_last D_j^-1 k_j
    eye = torch.eye(chunk, device=k.device, dtype=k.dtype)
    A = (bc[..., None] * (kd @ ki.transpose(-1, -2))).tril(-1)   # A_ij = beta_i (D_i k_i . D_j^-1 k_j), j < i
    M = torch.linalg.solve_triangular(eye + A, eye.expand_as(A), upper=False, unitriangular=True)
    W = M * bc[:, :, None, :]                 # (I + A)^-1 diag(beta): delta = W (U - KD E_0)
    PU, PK = W @ uc, W @ kd
    Q = (qd @ ki.transpose(-1, -2)).tril()    # Q_ij = D_i q_i . D_j^-1 k_j, j <= i
    el = torch.exp(bl)
    E = k.new_zeros(B, m, d)
    out = []
    for n in range(N):
        delta = PU[:, n] - PK[:, n] @ E
        out.append(qd[:, n] @ E + Q[:, n] @ delta)
        E = el[:, n, :, None] * E + ks[:, n].transpose(-1, -2) @ delta
    return torch.stack(out, 1).reshape(B, N * chunk, d)[:, :T]


def _lin(layer, h):
    """layer(h) in the slot precision, inside the autocast-free region."""
    return F.linear(_hi(h), _hi(layer.weight), None if layer.bias is None else _hi(layer.bias))


def _unit(x):
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-12)


class DeltaBlock(GatedBlock):
    """SD."""

    def __init__(self, args):
        super().__init__(args)
        del self.log_w, self.log_beta         # no prior mass and no erase term: the delta rule does both
        self.beta = nn.Linear(args.dim, 1, bias=True)

    def log_decay(self, r):
        rn = r * torch.rsqrt(r.square().mean(-1, keepdim=True) + 1e-12)
        g = F.softplus(_hi(self.gate_p(self.gate_q(rn)))) / math.log(2.0)
        return torch.expm1(-self.log_rate.exp() * g)

    def _inputs(self, x):
        h = self.norm(x)
        u = self.conv(h.transpose(1, 2))[..., : h.size(1)].transpose(1, 2)
        r = self.code(u)
        return h, r, self.val(h)

    def forward(self, x):
        h, r, v = self._inputs(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            r, v = _hi(r), _hi(v)
            k = F.pad(r[:, :-1], (0, 0, 1, 0))
            target = v - F.linear(k, _hi(self.mu.weight))
            beta = torch.sigmoid(_lin(self.beta, h)).squeeze(-1)
            fast = delta_memory(_unit(k), target, beta, self.log_decay(r), _unit(r))
        return x + self.dropout(self.mu(r.type_as(x)) + fast.type_as(x))

    def step(self, x, s):
        raise NotImplementedError("token-by-token SD lives in scripts/check_delta.py until SD wins")


class PowerDeltaBlock(DeltaBlock):
    """SDp: SD with a learnable address sharpness per layer, r = relu(K u + b)^p / (eps + sum), p = 1 + e^a from 2.
    relu^2 over its sum is scale-invariant, so SD's sharpness is fixed at p = 2; here the layer can learn it."""

    def __init__(self, args):
        super().__init__(args)
        self.log_pm1 = nn.Parameter(torch.zeros(()))

    def code(self, u):
        s = F.relu(_hi(self.key(u)))
        p = 1.0 + self.log_pm1.clamp(-5.0, 2.0).exp()
        z = torch.where(s > 0, s.clamp_min(1e-30).pow(p), torch.zeros_like(s))
        return z / (self.log_eps.clamp(-10, 10).exp() + z.sum(-1, keepdim=True))


class TwoDeltaBlock(DeltaBlock):
    """SD2: SD plus a slow memory with constant per-slot rates, its own write strength, and a read mix."""

    def __init__(self, args):
        super().__init__(args)
        m = args.hidden_dim
        self.log_rate_slow = nn.Parameter(torch.full((m,), math.log(1.0 / (16 * args.max_seq_len))))
        self.beta_slow = nn.Linear(args.dim, 1, bias=True)
        self.rho = nn.Linear(args.dim, 1, bias=True)   # UnifiedLM zeroes Linear biases: every gate starts at 1/2

    def forward(self, x):
        h, r, v = self._inputs(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            r, v = _hi(r), _hi(v)
            k = F.pad(r[:, :-1], (0, 0, 1, 0))
            target = v - F.linear(k, _hi(self.mu.weight))
            kh, qh = _unit(k), _unit(r)
            fast = delta_memory(kh, target, torch.sigmoid(_lin(self.beta, h)).squeeze(-1), self.log_decay(r), qh)
            slow_logl = torch.expm1(-self.log_rate_slow.exp()).expand_as(r)
            slow = delta_memory(kh, target, torch.sigmoid(_lin(self.beta_slow, h)).squeeze(-1), slow_logl, qh)
            rho = torch.sigmoid(_lin(self.rho, h))
            mem = (1 - rho) * fast + rho * slow
        return x + self.dropout(self.mu(r.type_as(x)) + mem.type_as(x))


class TwoFastDeltaBlock(DeltaBlock):
    """SD2c, SD2's control: the second memory is another copy of the first (its own gated per-slot decay started at
    1/(4 L), its own write strength), read through the same mix rho. Same state as SD2, no fast/slow split."""

    def __init__(self, args):
        super().__init__(args)
        m, rank = args.hidden_dim, self.gate_q.out_features
        self.log_rate2 = nn.Parameter(torch.full((m,), math.log(1.0 / (4 * args.max_seq_len))))
        self.gate_q2 = nn.Linear(m, rank, bias=False)
        self.gate_p2 = nn.Linear(rank, m, bias=True)
        self.beta2 = nn.Linear(args.dim, 1, bias=True)
        self.rho = nn.Linear(args.dim, 1, bias=True)

    def log_decay2(self, r):
        rn = r * torch.rsqrt(r.square().mean(-1, keepdim=True) + 1e-12)
        g = F.softplus(_hi(self.gate_p2(self.gate_q2(rn)))) / math.log(2.0)
        return torch.expm1(-self.log_rate2.exp() * g)

    def forward(self, x):
        h, r, v = self._inputs(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            r, v = _hi(r), _hi(v)
            k = F.pad(r[:, :-1], (0, 0, 1, 0))
            target = v - F.linear(k, _hi(self.mu.weight))
            kh, qh = _unit(k), _unit(r)
            one = delta_memory(kh, target, torch.sigmoid(_lin(self.beta, h)).squeeze(-1), self.log_decay(r), qh)
            two = delta_memory(kh, target, torch.sigmoid(_lin(self.beta2, h)).squeeze(-1), self.log_decay2(r), qh)
            rho = torch.sigmoid(_lin(self.rho, h))
            mem = (1 - rho) * one + rho * two
        return x + self.dropout(self.mu(r.type_as(x)) + mem.type_as(x))


class FixedPowerDeltaBlock(DeltaBlock):
    """SD with the address power fixed at p instead of 2: r = relu(K u + b)^p / (eps + sum). SDr uses p = 1 (plain
    ReLU, a blunter, denser address) in the first blocks, where the fast memory lives, and SD's p = 2 above."""

    def __init__(self, args, p):
        super().__init__(args)
        self.p = p

    def code(self, u):
        z = F.relu(_hi(self.key(u)))
        z = z if self.p == 1 else z.pow(self.p)
        return z / (self.log_eps.clamp(-10, 10).exp() + z.sum(-1, keepdim=True))


class WideRateDeltaBlock(DeltaBlock):
    """SDw: SD with the per-slot base forgetting rates started log-uniformly between 1/(L/4) and 1/(16 L) instead
    of all at 1/(4 L), so one memory starts with fast and slow slots (SD2's slow memory sits at 1/(16 L))."""

    def __init__(self, args):
        super().__init__(args)
        L, m = args.max_seq_len, args.hidden_dim
        with torch.no_grad():
            self.log_rate.copy_(torch.linspace(math.log(4.0 / L), math.log(1.0 / (16 * L)), m))
