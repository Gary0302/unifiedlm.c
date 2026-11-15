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
from model import RMSNorm, _hi


def delta_memory(k, u, beta, logl, q, chunk=CHUNK, reset=None, E0=None, final=False):
    """k: (B,T,m) unit write keys, u: (B,T,d) write targets, beta: (B,T) in [0, 1], logl: (B,T,m) log decay applied
    at step t (in [-1, 0]), q: (B,T,m) read addresses. With E_0 = 0,
        E_t = (I - beta_t k_t k_t^T) diag(lam_t) E_{t-1} + beta_t k_t u_t^T,   o_t = E_t^T q_t.
    Equivalently E_t = diag(lam_t) E_{t-1} + k_t delta_t^T with delta_t = beta_t (u_t - E_{t-1}^T diag(lam_t) k_t).
    reset (B,T) bool, SDb: lam_t = 0 where set (E_{t-1} is dropped), by masking the terms that cross a reset.
    E0 (B,m,d), SDulc: the state entering step 0 instead of 0; final=True also returns E_T."""
    B, T, m = k.shape
    d = u.size(-1)
    pad = (-T) % chunk
    if pad:
        k, u, logl, q = (F.pad(x, (0, 0, 0, pad)) for x in (k, u, logl, q))
        beta = F.pad(beta, (0, pad))
        reset = None if reset is None else F.pad(reset, (0, pad))
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
    Q = (qd @ ki.transpose(-1, -2)).tril()    # Q_ij = D_i q_i . D_j^-1 k_j, j <= i
    el = torch.exp(bl)
    kd0 = kd
    if reset is not None:
        seg = reset.reshape(B, N, chunk).long().cumsum(2)              # resets so far in the chunk, step i included
        same = (seg[..., :, None] == seg[..., None, :]).to(k.dtype)    # no reset in (j, i]
        first = (seg == 0).to(k.dtype)[..., None]                      # step i still sees the state entering the chunk
        A, Q = A * same, Q * same
        kd0, qd = kd * first, qd * first
        ks = ks * (seg == seg[..., -1:]).to(k.dtype)[..., None]       # steps carried into the next chunk
        el = el * (seg[..., -1] == 0).to(k.dtype)[..., None]
    M = torch.linalg.solve_triangular(eye + A, eye.expand_as(A), upper=False, unitriangular=True)
    W = M * bc[:, :, None, :]                 # (I + A)^-1 diag(beta): delta = W (U - KD E_0)
    PU, PK = W @ uc, W @ kd0
    E = k.new_zeros(B, m, d) if E0 is None else E0.to(k.dtype)
    out = []
    for n in range(N):
        delta = PU[:, n] - PK[:, n] @ E
        out.append(qd[:, n] @ E + Q[:, n] @ delta)
        E = el[:, n, :, None] * E + ks[:, n].transpose(-1, -2) @ delta
    o = torch.stack(out, 1).reshape(B, N * chunk, d)[:, :T]
    return (o, E) if final else o


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


class FastRateDeltaBlock(DeltaBlock):
    """SDl: SD with log_rate stored as log_rate / 10, so Adam (step ~ lr in the stored units, no weight decay on 1-D
    tensors) moves log_rate 10x as fast; same start (all at 1/(4 L)), same everything else.
    SDl100 and SDl1000 use 100 and 1000 instead of 10."""
    FACTOR = 10.0

    def __init__(self, args):
        super().__init__(args)
        r = self.log_rate.detach().clone()
        del self.log_rate
        self.log_rate_s = nn.Parameter(r / self.FACTOR)

    @property
    def log_rate(self):
        return self.log_rate_s * self.FACTOR


class FastRate100DeltaBlock(FastRateDeltaBlock):
    FACTOR = 100.0


class FastRate1000DeltaBlock(FastRateDeltaBlock):
    FACTOR = 1000.0


class RawFFNDeltaBlock(DeltaBlock):
    """SDu: SD whose static half reads the unnormalized code z = relu(K u + b)^2 (a plain ReLU^2 FFN, which keeps how
    strongly a token fires), y = mu z + E^T q; the memory keeps the normalized address r = z / (eps + sum z) and the
    write target v - mu k with k = r_{t-1}, as in SD."""

    def forward(self, x):
        h = self.norm(x)
        u = self.conv(h.transpose(1, 2))[..., : h.size(1)].transpose(1, 2)
        s, v = self.key(u), self.val(h)
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = F.relu(_hi(s)).square()
            r, v = z / (self.log_eps.clamp(-10, 10).exp() + z.sum(-1, keepdim=True)), _hi(v)
            k = F.pad(r[:, :-1], (0, 0, 1, 0))
            target = v - F.linear(k, _hi(self.mu.weight))
            beta = torch.sigmoid(_lin(self.beta, h)).squeeze(-1)
            fast = delta_memory(_unit(k), target, beta, self.log_decay(r), _unit(r))
        return x + self.dropout(self.mu(z.type_as(x)) + fast.type_as(x))


class RawFFNFastRate1000DeltaBlock(RawFFNDeltaBlock, FastRate1000DeltaBlock):
    """SDul: SDu (static half on the unnormalized code) with SDl1000's log_rate learning 1000x as fast."""



class AnchorRawFFNDeltaBlock(RawFFNFastRate1000DeltaBlock):
    """SDula: SDul whose write target is anchored at the FFN output the block actually gives, v - mu z_{t-1}, instead
    of SDu's v - mu r_{t-1} (z = relu(K u + b)^2, r = z / (eps + sum z)); everything else as in SDul."""

    def forward(self, x):
        h = self.norm(x)
        u = self.conv(h.transpose(1, 2))[..., : h.size(1)].transpose(1, 2)
        s, v = self.key(u), self.val(h)
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = F.relu(_hi(s)).square()
            r, v = z / (self.log_eps.clamp(-10, 10).exp() + z.sum(-1, keepdim=True)), _hi(v)
            k = F.pad(r[:, :-1], (0, 0, 1, 0))
            target = v - F.linear(F.pad(z[:, :-1], (0, 0, 1, 0)), _hi(self.mu.weight))
            beta = torch.sigmoid(_lin(self.beta, h)).squeeze(-1)
            fast = delta_memory(_unit(k), target, beta, self.log_decay(r), _unit(r))
        return x + self.dropout(self.mu(z.type_as(x)) + fast.type_as(x))


class ReadGateAnchorDeltaBlock(AnchorRawFFNDeltaBlock):
    """SDulag: SDula with a gate on the memory read, one number per token, o_t = 2 sigmoid(w . h_t + b) in (0, 2), near 1
    at initialisation (b = 0, w at std 0.02: about 0.65 to 1.35) with dim + 1 parameters per block: the block can turn
    its read down where the memory holds nothing useful for this token, y = x + mu z + o_t E^T q. Else as in SDula."""

    def __init__(self, args):
        super().__init__(args)
        self.read = nn.Linear(args.dim, 1, bias=True)

    def forward(self, x):
        h = self.norm(x)
        u = self.conv(h.transpose(1, 2))[..., : h.size(1)].transpose(1, 2)
        s, v = self.key(u), self.val(h)
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = F.relu(_hi(s)).square()
            r, v = z / (self.log_eps.clamp(-10, 10).exp() + z.sum(-1, keepdim=True)), _hi(v)
            k = F.pad(r[:, :-1], (0, 0, 1, 0))
            target = v - F.linear(F.pad(z[:, :-1], (0, 0, 1, 0)), _hi(self.mu.weight))
            beta = torch.sigmoid(_lin(self.beta, h)).squeeze(-1)
            fast = self.read_out(delta_memory(_unit(k), target, beta, self.log_decay(r), _unit(r)), h)
        return x + self.dropout(self.mu(z.type_as(x)) + fast.type_as(x))

    def read_out(self, o, h):
        return o * (2 * torch.sigmoid(_lin(self.read, h)))


class ReadScaleAnchorDeltaBlock(ReadGateAnchorDeltaBlock):
    """SDulas: the control for SDulag, one learned read scale per block, 2 sigmoid(b) (1 at initialisation), in place of
    the per-token gate: SDulag's gate came out nearly the same on every token of a layer (readstats.py)."""

    def __init__(self, args):
        super().__init__(args)
        del self.read
        self.read_b = nn.Parameter(torch.zeros(()))

    def read_out(self, o, h):
        return o * (2 * torch.sigmoid(self.read_b.float()))


class ReadNormGateAnchorDeltaBlock(ReadGateAnchorDeltaBlock):
    """SDulagn: SDulag with the read RMS-normalised before the gate, o_t RMSNorm(E^T q), as GatedDeltaNet does: the read
    no longer grows as the memory fills (readstats.py: |E^T q| rises with position in the long-memory layers), and its
    size is left to the gate. The per-channel gain starts at 1 / sqrt(dim), so the read starts at norm 1."""

    def __init__(self, args):
        super().__init__(args)
        self.read_norm = RMSNorm(args.dim, args.norm_eps)
        with torch.no_grad():
            self.read_norm.weight.fill_(args.dim ** -0.5)

    def read_out(self, o, h):
        return super().read_out(self.read_norm(o), h)



class NoMemoryDeltaBlock(RawFFNDeltaBlock):
    """SDn: SDu with the memory removed, the control for what the memory buys: y = x + mu z, z = relu(K u + b)^2 on
    the conv output. The memory's parameters (val, beta, the gate, log_rate, log_eps) are dropped."""

    def __init__(self, args):
        super().__init__(args)
        del self.val, self.beta, self.gate_q, self.gate_p, self.log_rate, self.log_eps

    def forward(self, x):
        h = self.norm(x)
        u = self.conv(h.transpose(1, 2))[..., : h.size(1)].transpose(1, 2)
        s = self.key(u)
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = F.relu(_hi(s)).square()
        return x + self.dropout(self.mu(z.type_as(x)))

CARRY = {"calls": 0, "accum": 1}   # SDulc: training forwards so far, micro-batches per optimizer step


class CarryRawFFNDeltaBlock(RawFFNFastRate1000DeltaBlock):
    """SDulc: SDul whose memory, in training, starts each window from the state the same batch row ended its previous
    window with (gradient stopped), so the memory sees a stream longer than L. The rows read consecutive windows of one
    stream each (carry_stream); micro-batch j of a step continues micro-batch j of the step before, so a block keeps one
    state per micro-batch. The first block counts the training forwards. Evaluation starts every window empty, as in
    SDul. The conv still starts each window with zeros."""

    def __init__(self, args, first=False):
        super().__init__(args)
        self.first = first
        self.carry = {}

    def forward(self, x):
        if not self.training:
            return super().forward(x)
        if self.first:
            CARRY["calls"] += 1
        j = (CARRY["calls"] - 1) % CARRY["accum"]
        h = self.norm(x)
        u = self.conv(h.transpose(1, 2))[..., : h.size(1)].transpose(1, 2)
        s, v = self.key(u), self.val(h)
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = F.relu(_hi(s)).square()
            r, v = z / (self.log_eps.clamp(-10, 10).exp() + z.sum(-1, keepdim=True)), _hi(v)
            k = F.pad(r[:, :-1], (0, 0, 1, 0))
            target = v - F.linear(k, _hi(self.mu.weight))
            beta = torch.sigmoid(_lin(self.beta, h)).squeeze(-1)
            fast, E = delta_memory(_unit(k), target, beta, self.log_decay(r), _unit(r), E0=self.carry.get(j), final=True)
            self.carry[j] = E.detach()
        return x + self.dropout(self.mu(z.type_as(x)) + fast.type_as(x))


def carry_stream(orig):
    """PretokDataset.__iter__ for SDulc: the training split (shards end to end) cut into A * B equal stretches, one
    stream of consecutive windows each (B the batch size, A the micro-batches per step). Sample n goes to row n % B of
    micro-batch (n // B) % A of step n // (A B) and is window `step` of stream (micro-batch, row). The validation
    split is unchanged."""
    import bisect
    import glob
    import os
    import numpy as np
    import fineweb

    def it(self):
        if self.split != "train":
            yield from orig(self)
            return
        B, A, L = CARRY["batch"], CARRY["accum"], self.max_seq_len
        ms = [np.memmap(f, dtype=np.uint16, mode="r") for f in sorted(glob.glob(os.path.join(fineweb.DATA_DIR, "*.bin")))[:-1]]
        ends = np.cumsum([len(m) for m in ms]).tolist()
        R = ends[-1] // (A * B)        # tokens per stream
        W = (R - 1) // L               # windows per stream before it wraps

        def read(p, n):
            i = bisect.bisect_right(ends, p)
            a = p - (ends[i - 1] if i else 0)
            c = ms[i][a:a + n]
            return c if len(c) == n else np.concatenate([c, ms[i + 1][:n - len(c)]])

        n = 0
        while True:
            b, row = divmod(n, B)
            step, j = divmod(b, A)
            p = (j * B + row) * R + (step % W) * L
            chunk = torch.from_numpy(read(p, L + 1).astype(np.int64))
            yield chunk[:-1], chunk[1:]
            n += 1
    return it


BOS = 1   # llama2 tokenizer: every document starts with BOS


class DocResetDeltaBlock(DeltaBlock):
    """SDb: SD whose memory is emptied at every BOS (lam = 0 and beta = 0 there), so nothing written in one
    document is read in the next. doc_reset_forward (installed as UnifiedLM.forward) passes the BOS mask."""

    def forward(self, x, reset=None):
        h, r, v = self._inputs(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            r, v = _hi(r), _hi(v)
            k = F.pad(r[:, :-1], (0, 0, 1, 0))
            target = v - F.linear(k, _hi(self.mu.weight))
            beta = torch.sigmoid(_lin(self.beta, h)).squeeze(-1)
            if reset is not None:
                beta = beta * (~reset).to(beta.dtype)
            fast = delta_memory(_unit(k), target, beta, self.log_decay(r), _unit(r), reset=reset)
        return x + self.dropout(self.mu(r.type_as(x)) + fast.type_as(x))


def doc_reset_forward(self, tokens, targets=None):
    """UnifiedLM.forward with the BOS mask passed to every block (SDb)."""
    reset = tokens == BOS
    h = self.dropout(self.tok_embeddings(tokens))
    for layer in self.layers:
        h = layer(h, reset)
    h = self.norm(h)
    if targets is not None:
        logits = self.output(h)
        self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    else:
        logits = self.output(h[:, [-1], :])
        self.last_loss = None
    return logits


class DiverseRateDeltaBlock(DeltaBlock):
    """SDv: per-slot base horizons log-uniform from 4 to 64 L tokens, and a bounded gate g = 2 sigmoid(.) in (0, 2)
    instead of softplus (unbounded), so the gate can slow a slot down freely but speed it up at most 2x: the spread
    survives training (SDw's gate sped its slow slots up ~25x). g = 1 at the start, as in SD."""

    def __init__(self, args):
        super().__init__(args)
        L, m = args.max_seq_len, args.hidden_dim
        with torch.no_grad():
            self.log_rate.copy_(torch.linspace(math.log(1.0 / 4), math.log(1.0 / (64 * L)), m))

    def log_decay(self, r):
        rn = r * torch.rsqrt(r.square().mean(-1, keepdim=True) + 1e-12)
        g = 2.0 * torch.sigmoid(_hi(self.gate_p(self.gate_q(rn))))
        return torch.expm1(-self.log_rate.exp() * g)


class WideRateDeltaBlock(DeltaBlock):
    """SDw: SD with the per-slot base forgetting rates started log-uniformly between 1/(L/4) and 1/(16 L) instead
    of all at 1/(4 L), so one memory starts with fast and slow slots (SD2's slow memory sits at 1/(16 L))."""

    def __init__(self, args):
        super().__init__(args)
        L, m = args.max_seq_len, args.hidden_dim
        with torch.no_grad():
            self.log_rate.copy_(torch.linspace(math.log(4.0 / L), math.log(1.0 / (16 * L)), m))
