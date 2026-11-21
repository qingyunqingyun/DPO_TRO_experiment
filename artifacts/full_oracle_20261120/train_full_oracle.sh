#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/hezhisheng/su/oracle_full_20261120
REPO=/mnt/hezhisheng/distributionally_robust_dpo-2
PY=$REPO/.venv/bin/python
MODEL=$REPO/models/Llama1b
OUT=$ROOT/outputs/full_oracle
if [ -f "$OUT/final_hf/model.safetensors" ]; then
  echo "checkpoint already exists"; exit 0
fi
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=4,6 DS_SKIP_CUDA_CHECK=1 MASTER_PORT=29920 "$PY" -m deepspeed.launcher.runner --include localhost:4,6 --master_port 29920 "$REPO/src/train_alignment.py" \
  --save_path "$OUT" --ckpt_path "$OUT" --save_hf_ckpt --disable_ds_ckpt --save_steps=-1 --eval_steps=-1 \
  --micro_train_batch_size=1 --train_batch_size=128 --gradient_checkpointing --zero_stage=2 --bf16 --no-flash_attn \
  --torch_adam --learning_rate=5e-7 --lr_warmup_ratio=0.1 --max_norm=1.0 --adam_betas 0.9 0.95 --l2=0 --zpg=2 \
  --train_task tro --tro_method oracle_soft_dpo --max_epochs 2 --beta 0.01 \
  --pretrain "$MODEL" --ref_pretrain "$MODEL" --dataset "$ROOT/datasets/full_oracle" \
  --prompt_key prompt --chosen_key chosen --rejected_key rejected --apply_chat_template --max_len 2048 \
  --dataset_num_processors 8 --train_split train --eval_split dev --seed 1101 \
  > "$ROOT/logs/train_full_oracle.log" 2>&1
