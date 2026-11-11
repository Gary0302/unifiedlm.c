#!/usr/bin/env bash
# SG versus T: long_eval.py (16 L) and samples_fineweb.py for every finished SG and T run, into /workspace/runs/sgvt/<run>.
# Skips runs already done and runs not published yet (rerun later to fill them in). Run from the repo root.
set -euo pipefail
R=/workspace/runs
RUNS="fw15m/fw15m_t fw15m_seeds/fw15m_t_seed1338 fw15m_seeds/fw15m_t_seed1339 fw42m/fw42m_t
      fwsg/fwsg15m fwsg/fwsg15m_seed1338 fwsg/fwsg15m_seed1339 fwsg/fwsg42m fwsg/fwsg42m_seed1338 fwsgx/fwsg42m_seed1339
      fwsh/fwsh15m fwsh2/fwsh15m_seed1338 fwsh2/fwsh15m_seed1339 fwsh2/fwsh42m
      fwsd/fwsd15m fwsd/fwsd15m_seed1338 fwsd/fwsd15m_seed1339 fwsd/fwsd2_15m"
for p in $RUNS; do
    name=${p#*/}
    dest=$R/sgvt/$name
    [ -e "$dest/long_eval.json" ] && continue
    [ -e "$R/$p/eval.json" ] || { echo "skip $name: not published"; continue; }
    tmp=/scratch/sgvt/$name
    rm -rf "$tmp" && mkdir -p "$tmp"
    flag=""; [[ $name == *_t* ]] && flag="--window"
    python scripts/fineweb/samples_fineweb.py --out_dir="$R/$p" > "$tmp/samples.jsonl"
    [ -n "$flag" ] && python scripts/fineweb/samples_fineweb.py --out_dir="$R/$p" --window >> "$tmp/samples.jsonl"
    python scripts/fineweb/long_eval.py --out_dir="$R/$p" --dest="$tmp" $flag > "$tmp/long_eval.log" 2>&1
    mkdir -p $R/sgvt && rm -rf "$dest.tmp" && cp -r "$tmp" "$dest.tmp" && mv "$dest.tmp" "$dest"
    echo "done $name"
done
echo SGVT DONE
