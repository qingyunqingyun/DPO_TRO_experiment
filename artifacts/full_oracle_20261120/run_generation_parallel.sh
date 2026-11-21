#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/hezhisheng/su/oracle_full_20261120
V5=/mnt/hezhisheng/su/1030-lastest/experiment_handoff_v5
PY=/mnt/hezhisheng/distributionally_robust_dpo-2/.venv/bin/python
MODEL=$ROOT/outputs/full_oracle/final_hf
run_one() {
  local i=$1 gpu=$2
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$V5/generate_eval.py" --model "$MODEL" --pairs "$ROOT/data/test_parts/part${i}.jsonl" --output "$ROOT/eval/full_test_generations.part${i}.jsonl" --k 10 --seed 401 --max-new-tokens 1024 --max-model-len 4096 --gpu-memory-utilization 0.5 > "$ROOT/logs/generate_full_test_part${i}.log" 2>&1
}
run_one 0 0 & p0=$!
run_one 1 4 & p1=$!
run_one 2 6 & p2=$!
wait "$p0"
wait "$p1"
wait "$p2"
cat "$ROOT/eval/full_test_generations.part0.jsonl" "$ROOT/eval/full_test_generations.part1.jsonl" "$ROOT/eval/full_test_generations.part2.jsonl" > "$ROOT/eval/full_test_generations.jsonl"
"$PY" - <<'PY'
import json
from pathlib import Path
src=Path('/mnt/hezhisheng/su/1030-lastest/experiment_handoff_v5/data/raw/test_filtered.jsonl')
out=Path('/mnt/hezhisheng/su/oracle_full_20261120/eval/full_test_generations.jsonl')
a=[json.loads(x)['pair_id'] for x in src.open() if x.strip()]
b=[json.loads(x)['pair_id'] for x in out.open() if x.strip()]
assert len(a)==1995 and len(b)==19950 and b[::10]==a, (len(a),len(b))
print('generation audit PASS',len(b))
PY
