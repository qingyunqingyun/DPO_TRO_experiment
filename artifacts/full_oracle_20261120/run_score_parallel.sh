#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/hezhisheng/su/oracle_full_20261120
PY=/mnt/hezhisheng/distributionally_robust_dpo-2/.venv/bin/python
RM=/mnt/hezhisheng/su/experiment_handoff_v5/models/Skywork-Reward-V2-Llama-3.1-8B
V5=/mnt/hezhisheng/su/1030-lastest/experiment_handoff_v5
run_one() {
  local part=$1 gpu=$2
  local out="$ROOT/scores/train_full_skywork.part${part}.jsonl"
  if [ ! -s "$out" ] || [ "$(wc -l < "$out")" -ne "$([ "$part" = 00 ] && echo 30484 || echo 30521)" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/score_skywork.py" --model "$RM" --input "$ROOT/data/train_full.part${part}" --output "$out" --mode pairs --batch-size 2 --max-length 16384 --device cuda:0 > "$ROOT/logs/score_train_full_part${part}.log" 2>&1
  fi
}
run_one 00 0 & p0=$!
run_one 01 6 & p1=$!
wait "$p0"
wait "$p1"
cat "$ROOT/scores/train_full_skywork.part00.jsonl" "$ROOT/scores/train_full_skywork.part01.jsonl" > "$ROOT/scores/train_full_skywork.jsonl"
"$PY" - <<'PY'
import json
from pathlib import Path
src=Path('/mnt/hezhisheng/su/1030-lastest/experiment_handoff_v5/data/raw/train_filtered.jsonl')
out=Path('/mnt/hezhisheng/su/oracle_full_20261120/scores/train_full_skywork.jsonl')
a=[json.loads(x)['pair_id'] for x in src.open() if x.strip()]
b=[json.loads(x)['pair_id'] for x in out.open() if x.strip()]
assert len(a)==61005 and len(b)==61005 and a==b, (len(a),len(b),a[:2],b[:2])
print('full score audit PASS', len(b))
PY
"$PY" "$ROOT/build_full_oracle.py" --pairs "$V5/data/raw/train_filtered.jsonl" --scores "$ROOT/scores/train_full_skywork.jsonl" --output "$ROOT/data/oracle_train_full.jsonl" --dataset-output "$ROOT/datasets/full_oracle" > "$ROOT/logs/build_oracle.log" 2>&1
