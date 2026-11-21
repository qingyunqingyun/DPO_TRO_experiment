#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/hezhisheng/su/oracle_full_20261120
PY=/mnt/hezhisheng/distributionally_robust_dpo-2/.venv/bin/python
RM=/mnt/hezhisheng/su/experiment_handoff_v5/models/Skywork-Reward-V2-Llama-3.1-8B
run_one() {
  local i=$1 gpu=$2
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/score_skywork.py" --model "$RM" --input "$ROOT/eval/full_test_generations.part${i}.jsonl" --output "$ROOT/eval/full_test_scored.part${i}.jsonl" --mode responses --batch-size 2 --max-length 4096 --device cuda:0 > "$ROOT/logs/score_full_test_part${i}.log" 2>&1
}
run_one 0 0 & p0=$!
run_one 1 4 & p1=$!
run_one 2 6 & p2=$!
wait "$p0"
wait "$p1"
wait "$p2"
cat "$ROOT/eval/full_test_scored.part0.jsonl" "$ROOT/eval/full_test_scored.part1.jsonl" "$ROOT/eval/full_test_scored.part2.jsonl" > "$ROOT/eval/full_test_scored.jsonl"
"$PY" "$ROOT/summarize_full_oracle.py"
