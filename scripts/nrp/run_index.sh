#!/usr/bin/env bash
# Runs line JOB_COMPLETION_INDEX of a plan file, "<name> <flags...>". Trains on local disk, then publishes the
# finished run to the PVC with a rename, so a run directory on the PVC is always complete.
set -euo pipefail

PLAN=$1
LINE=$(sed -n "$((JOB_COMPLETION_INDEX + 1))p" "scripts/nrp/$PLAN")
NAME=${LINE%% *}
FLAGS=${LINE#* }
OUT=/workspace/runs/${PLAN%.txt}/$NAME
if [ -e "$OUT/eval.json" ]; then
    echo "$NAME is already complete at $OUT"
    exit 0
fi

LOCAL=/scratch/out
rm -rf "$LOCAL"  # a pod may run several plan lines; start each from an empty directory
mkdir -p "$LOCAL" "$(dirname "$OUT")"
nvidia-smi -L || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 > "$LOCAL/gpu.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

START=$SECONDS
if [[ $PLAN == mem* ]]; then  # synthetic memory tasks write their own eval.json
    python scripts/memory/mqar.py --out_dir="$LOCAL" $FLAGS 2>&1 | tee "$LOCAL/train.log"
    TRAIN_SECONDS=$((SECONDS - START))
else
    python scripts/train_variant.py --out_dir="$LOCAL" $FLAGS 2>&1 | tee "$LOCAL/train.log"
    TRAIN_SECONDS=$((SECONDS - START))
    python scripts/eval_nll.py --out_dir="$LOCAL" 2>&1 | tee "$LOCAL/eval.log"
fi
echo "{\"name\": \"$NAME\", \"node\": \"${NODE_NAME:-unknown}\", \"train_seconds\": $TRAIN_SECONDS}" > "$LOCAL/timing.json"

rm -rf "$OUT.tmp"
cp -r "$LOCAL" "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
echo "published $OUT after ${TRAIN_SECONDS}s of training"
