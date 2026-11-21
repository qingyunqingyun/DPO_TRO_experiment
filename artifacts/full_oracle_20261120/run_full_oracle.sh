#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/hezhisheng/su/oracle_full_20261120
V5=/mnt/hezhisheng/su/1030-lastest/experiment_handoff_v5
PY=/mnt/hezhisheng/distributionally_robust_dpo-2/.venv/bin/python
RM=/mnt/hezhisheng/su/experiment_handoff_v5/models/Skywork-Reward-V2-Llama-3.1-8B
if [ ! -s "$ROOT/scores/train_full_skywork.jsonl" ] || [ "$(wc -l < "$ROOT/scores/train_full_skywork.jsonl")" -ne 61005 ]; then
  CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/score_skywork.py" --model "$RM" --input "$V5/data/raw/train_filtered.jsonl" --output "$ROOT/scores/train_full_skywork.jsonl" --mode pairs --batch-size 2 --max-length 16384 --device cuda:0 > "$ROOT/logs/score_train_full.log" 2>&1
fi
"$PY" "$ROOT/build_full_oracle.py" --pairs "$V5/data/raw/train_filtered.jsonl" --scores "$ROOT/scores/train_full_skywork.jsonl" --output "$ROOT/data/oracle_train_full.jsonl" --dataset-output "$ROOT/datasets/full_oracle" > "$ROOT/logs/build_oracle.log" 2>&1
