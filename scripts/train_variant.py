"""
Train one ablation variant with the unmodified training loop. Run from the repository root:

$ python scripts/train_variant.py --variant=P --m_s=105 --m_f=70 --out_dir=runs/x [train.py flags]

Writes out_dir/variant.json, which scripts/eval_nll.py requires: D has the same state dict as S, so the
checkpoint alone cannot tell them apart. Resuming is refused, because train.py restarts the data stream on
resume while the learning rate continues; an interrupted run is rerun from scratch.
"""
import json
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())

own = {"variant": "S", "m_s": 0, "m_f": 0, "dataset": "tinystories"}
argv = []
for arg in sys.argv[1:]:
    key, _, val = arg[2:].partition("=")
    if key in own:
        own[key] = type(own[key])(val)
    else:
        argv.append(arg)
out_dir = next((a.split("=", 1)[1] for a in argv if a.startswith("--out_dir=")), None)
if out_dir is None:
    raise SystemExit("--out_dir is required")
if "--init_from=resume" in argv:
    raise SystemExit("resuming is refused: rerun from scratch")
sys.argv = [sys.argv[0]] + argv

import export    # noqa: E402
import variants  # noqa: E402

variants.install(own["variant"], own["m_s"], own["m_f"])
if own["dataset"] == "babylm":  # train.py imports its Task from tinystories
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "babylm"))
    import babylm  # noqa: E402
    sys.modules["tinystories"] = babylm
    seed = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--seed=")), 1337)
    babylm.DATA_SEED += seed - 1337  # a new seed changes the data order as well as the initialization
elif own["dataset"] != "tinystories":
    raise SystemExit(f"unknown dataset {own['dataset']}")
export.model_export = lambda *a, **k: None       # the run.c format belongs to the standard model

os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "variant.json"), "w") as f:
    json.dump({**own, "argv": argv}, f, indent=2)

runpy.run_path("train.py", run_name="__main__")
