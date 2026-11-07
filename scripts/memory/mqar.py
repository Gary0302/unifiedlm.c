"""
Synthetic in-context memory tasks for scripts/reports/memory-plan.md. One run trains one model on a
mixture of key-value counts and writes out_dir/eval.json with the answer accuracy per count. Run from the
repository root:

$ python scripts/memory/mqar.py --model=S --dim=128 --lr=1e-3 --out_dir=runs/mem/s_d128

  m1  N key-value pairs, then every key again in random order, each followed by its value (MQAR). Keys
      come from the lower half of the vocabulary, values from the upper half, token 0 pads the sequence.
      Loss and accuracy count only the positions whose next token is the value after a query.

Models: S (model.UnifiedLM) and P (separate addresses with m slots each, scripts/variants.py) with 4
blocks, and T, a llama2.c-style Transformer with 2 layers. One Transformer layer counts as two blocks, and
the slot count m gives S the parameter count of T. Without CUDA the memory falls back to the dense
reference (scripts/dense_memory.py), so the script can be checked on a laptop.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
if not torch.cuda.is_available():
    from dense_memory import dense_memory
    sys.modules["kernel"] = types.SimpleNamespace(CHUNK=64, unified_memory=dense_memory)
import model as ulm  # noqa: E402
import variants  # noqa: E402

HEAD_DIM = 32


def ffn_hidden(d):  # llama2.c's rule: 2/3 of 4d, rounded up to a multiple of 32
    h = int(2 * 4 * d / 3)
    return 32 * ((h + 31) // 32)


def slots(d):
    """m such that two UnifiedBlocks have the parameters of one Transformer layer."""
    t_layer = 4 * d * d + 3 * d * ffn_hidden(d) + 2 * d
    return round((t_layer / 2 - d * d - 5 * d - 2) / (2 * d + 2))


def rope(x, T):
    hd = x.size(-1)
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=x.device).float() / hd))
    ang = torch.outer(torch.arange(T, device=x.device).float(), inv)
    c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2).type_as(x)


class TBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.h = d // HEAD_DIM
        self.attention_norm = ulm.RMSNorm(d, eps=1e-5)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.ffn_norm = ulm.RMSNorm(d, eps=1e-5)
        hidden = ffn_hidden(d)
        self.w1, self.w2, self.w3 = nn.Linear(d, hidden, bias=False), nn.Linear(hidden, d, bias=False), nn.Linear(d, hidden, bias=False)

    def forward(self, x):
        B, T, d = x.shape
        a = self.attention_norm(x)
        q, k, v = (w(a).view(B, T, self.h, HEAD_DIM).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        y = F.scaled_dot_product_attention(rope(q, T), rope(k, T), v, is_causal=True)
        x = x + self.wo(y.transpose(1, 2).reshape(B, T, d))
        f = self.ffn_norm(x)
        return x + self.w2(F.silu(self.w1(f)) * self.w3(f))


class Transformer(nn.Module):
    """The same attribute names as UnifiedLM (tok_embeddings, layers, norm, output), tied embeddings."""

    def __init__(self, d, n_layers, vocab):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([TBlock(d) for _ in range(n_layers)])
        self.norm = ulm.RMSNorm(d, eps=1e-5)
        self.output = nn.Linear(d, vocab, bias=False)
        self.tok_embeddings.weight = self.output.weight
        for pn, p in self.named_parameters():
            if p.dim() >= 2:
                std = 0.02 / math.sqrt(2 * n_layers) if pn.endswith(("w3.weight", "wo.weight")) else 0.02
                nn.init.normal_(p, mean=0.0, std=std)


class Body(nn.Module):
    """Embedding, blocks and final norm; the output head runs only at the answer positions."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, tokens):
        h = self.net.tok_embeddings(tokens)
        for layer in self.net.layers:
            h = layer(h)
        return self.net.norm(h)


def build(kind, d, vocab, seq):
    if kind == "T":
        return Transformer(d, 2, vocab), 0
    m = slots(d)
    variants.install(kind, m, m)  # S ignores the counts; P gets m static and m context slots
    return ulm.UnifiedLM(ulm.ModelArgs(dim=d, n_layers=4, hidden_dim=m, vocab_size=vocab, max_seq_len=seq, dropout=0.0)), m


def m1_batch(B, N, T, V, g, device):
    half = V // 2
    keys = torch.rand(B, half - 1, generator=g, device=device).argsort(1)[:, :N] + 1  # distinct within a row
    vals = torch.randint(half, V, (B, N), generator=g, device=device)
    perm = torch.rand(B, N, generator=g, device=device).argsort(1)
    qk, qv = keys.gather(1, perm), vals.gather(1, perm)
    x = torch.zeros(B, T, dtype=torch.long, device=device)
    x[:, 0:2 * N:2], x[:, 1:2 * N:2] = keys, vals
    x[:, 2 * N:4 * N:2], x[:, 2 * N + 1:4 * N:2] = qk, qv
    y = torch.full((B, T), -1, dtype=torch.long, device=device)
    y[:, 2 * N:4 * N:2] = qv
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["S", "P", "T"], required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--task", choices=["m1"], default="m1")
    ap.add_argument("--counts", default="8,16,32,64,128")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--eval_size", type=int, default=1024)
    ap.add_argument("--compile", type=int, default=1)
    a = ap.parse_args()
    counts = [int(c) for c in a.counts.split(",")]
    assert 4 * max(counts) <= a.seq, "4N tokens must fit in the sequence"
    os.makedirs(a.out_dir, exist_ok=True)

    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    torch.manual_seed(a.seed)
    net, m = build(a.model, a.dim, a.vocab, a.seq)
    net.to(dev)
    params = sum(p.numel() for p in net.parameters())
    body = Body(net)
    fwd = torch.compile(body) if (cuda and a.compile) else body
    decay = [p for p in net.parameters() if p.dim() >= 2]
    other = [p for p in net.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": other, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.99), **({"fused": True} if cuda else {}))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else torch.autocast("cpu", enabled=False)
    print(f"model {a.model} d={a.dim} m={m} params {params:,} device {dev}", flush=True)

    def loss_acc(x, y):
        with amp:
            h = fwd(x)
            sel = y != -1
            logits = net.output(h[sel]).float()
        t = y[sel]
        return F.cross_entropy(logits, t), (logits.argmax(-1) == t).float().mean()

    @torch.no_grad()
    def evaluate(split):  # fixed sets: the same sequences for every model and run
        net.eval()
        out = {}
        for N in counts:
            g = torch.Generator(device=dev).manual_seed({"val": 1, "test": 2}[split] * 100003 + N)
            accs = [loss_acc(*m1_batch(a.batch, N, a.seq, a.vocab, g, dev))[1].item()
                    for _ in range(max(1, a.eval_size // a.batch))]
            out[N] = sum(accs) / len(accs)
        net.train()
        return out

    g = torch.Generator(device=dev).manual_seed(a.seed)
    curve, times, diverged = [], [], False
    t0 = time.time()
    tl = time.time()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            v = evaluate("val")
            curve.append((step, v))
            print(f"step {step}: val acc " + " ".join(f"N{N} {acc:.4f}" for N, acc in v.items()), flush=True)
            tl = time.time()  # evaluation time is not training time
        if step == a.steps:
            break
        lr = a.lr * (step + 1) / a.warmup if step < a.warmup else \
            a.lr * 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / (a.steps - a.warmup)))
        for group in opt.param_groups:
            group["lr"] = lr
        loss, acc = loss_acc(*m1_batch(a.batch, counts[step % len(counts)], a.seq, a.vocab, g, dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 99:
            loss_v = loss.item()
            dt = (time.time() - tl) / 100 * 1000
            tl = time.time()
            times.append(dt)
            print(f"{step + 1} | loss {loss_v:.4f} | acc {acc.item():.4f} | lr {lr:.3e} | {dt:.2f}ms", flush=True)
            if not math.isfinite(loss_v):
                diverged = True
                print(f"diverged at step {step + 1}", flush=True)
                break
    result = {"task": a.task, "model": a.model, "dim": a.dim, "slots": m, "params": params, "lr": a.lr, "seed": a.seed,
              "steps": a.steps, "batch": a.batch, "seq": a.seq, "vocab": a.vocab, "counts": counts, "diverged": diverged,
              "val": evaluate("val") if diverged else curve[-1][1], "test": evaluate("test"), "val_curve": curve,
              "ms_per_iter": statistics.median(times[1:]) if len(times) > 1 else None,
              "train_seconds": round(time.time() - t0)}
    with open(os.path.join(a.out_dir, "eval.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("test acc " + " ".join(f"N{N} {acc:.4f}" for N, acc in result["test"].items()), flush=True)


if __name__ == "__main__":
    main()
