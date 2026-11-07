"""
The real Triton kernel against the dense float64 definition, for every variant and slot width the experiment
uses: loss and every parameter gradient, in fp32 and bf16, eager and compiled, plus one optimizer update.
Needs CUDA. Run from the repository root:

$ python scripts/check_kernel.py

The tolerance is test_all.py's. S runs first as the yardstick for what the kernel normally achieves.
"""
import copy
import math
import os
import sys

import torch

sys.path.insert(0, os.getcwd())

import kernel                      # noqa: E402
import model                       # noqa: E402
import variants                    # noqa: E402
from dense_memory import dense_memory  # noqa: E402
from model import ModelArgs, UnifiedLM  # noqa: E402

TOL = 1e-2
#         variant, m_s, m_f, d, m, T
CONFIGS = [
    ("S", 0, 0, 64, 140, 512), ("D", 0, 0, 64, 140, 512), ("P", 140, 140, 64, 140, 512), ("P", 69, 140, 64, 140, 512),
    ("P", 93, 93, 64, 140, 512), ("P", 105, 70, 64, 140, 512), ("A", 207, 207, 64, 140, 512),
    ("S", 0, 0, 288, 712, 256), ("D", 0, 0, 288, 712, 256), ("P", 712, 712, 288, 712, 256), ("P", 355, 712, 288, 712, 256),
    ("P", 474, 474, 288, 712, 256), ("P", 534, 356, 288, 712, 256), ("A", 1045, 1045, 288, 712, 256),
    # partial last chunk
    ("S", 0, 0, 64, 140, 200), ("D", 0, 0, 64, 140, 200), ("P", 105, 70, 64, 140, 200), ("A", 207, 207, 64, 140, 200),
]
MODES = [(torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)]


def set_memory(fn):
    model.unified_memory = fn
    variants.unified_memory = fn


def build(variant, m_s, m_f, d, m, T):
    variants.install(variant, m_s, m_f)
    torch.manual_seed(0)
    net = UnifiedLM(ModelArgs(dim=d, n_layers=4, hidden_dim=m, vocab_size=512, max_seq_len=T))
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name.endswith("conv.weight"):
                p.normal_(std=0.5)
            elif name.endswith(("key.bias", "key_s.bias", "key_f.bias")):
                p.normal_(std=0.5)
            elif name.endswith("val.weight"):
                p.normal_(std=0.2)
            elif name.endswith("log_rate"):
                p.fill_(math.log(0.05))
            elif name.endswith("log_w"):
                p.normal_(std=1.0)
    return net


def check(config, dtype, compiled):
    net = build(*config)
    ref = copy.deepcopy(net).cuda().double().train()
    net = net.cuda().train()
    x = torch.randint(0, 512, (2, config[-1]), device="cuda")

    set_memory(kernel.unified_memory)
    run = torch.compile(net) if compiled else net
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        run(x, x)
        loss = run.last_loss
    loss.backward()

    set_memory(dense_memory)
    ref(x, x)
    ref.last_loss.backward()
    set_memory(kernel.unified_memory)

    errs = {"loss": abs(loss.item() - ref.last_loss.item()) / abs(ref.last_loss.item())}
    for (name, p), (_, q) in zip(net.named_parameters(), ref.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        key = name.split(".", 2)[-1]
        errs[key] = max(errs.get(key, 0.0), ((p.grad.double() - q.grad).norm() / q.grad.norm().clamp(min=1e-30)).item())

    # one update through the real optimizer must stay finite and move the weights
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, fused=True)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    assert all(torch.isfinite(p).all() for p in net.parameters())
    assert any((p != before[n]).any() for n, p in net.named_parameters())
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        run(x, x)
    assert math.isfinite(run.last_loss.item())
    return errs


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    failed = 0
    for config in CONFIGS:
        for dtype, compiled in MODES:
            errs = check(config, dtype, compiled)
            key, worst = max(errs.items(), key=lambda kv: kv[1])
            ok = worst < TOL
            failed += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {config[0]} m_s={config[1]:<5} m_f={config[2]:<5} d={config[3]:<4} T={config[5]:<4} "
                  f"{dtype.__name__ if hasattr(dtype, '__name__') else str(dtype).split('.')[-1]:<8} compiled={compiled!s:<5} "
                  f"worst {worst:.2e} ({key}), loss {errs['loss']:.2e}", flush=True)
    print(f"{failed} failures")
    sys.exit(1 if failed else 0)
