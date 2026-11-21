# Full eligible-set oracle result (2026-11-20)

This run trains one `oracle_soft_dpo` Llama-3.2-1B-Instruct policy on all 61,005 length-eligible, prompt-deduplicated training pairs. The 1,995 eligible `test_filtered` prompts are held out for evaluation.

## Training and scoring

| item | value |
|---|---|
| train pairs | 61005 |
| test prompts | 1995 |
| responses per prompt | 10 |
| epochs | 2 |
| max policy length | 2048 |
| effective batch | 128 |
| learning rate | 5e-07 |
| DPO beta | 0.01 |
| generation temperature | 0.7 |
| top-p | 0.9 |
| max new tokens | 1024 |
| RM | Skywork-Reward-V2-Llama-3.1-8B |

## Result

- Mean within-prompt RM reward `R`: **7.157362**
- Across-prompt standard deviation: `10.475243`
- Scored rows: `19950` (`1995 × 10`)

`R` is the mean of the ten-response mean reward for each held-out prompt. This is an oracle-policy reference result; no competing full-data policy was trained in this run.

## Artifacts

- `data/oracle_train_full.jsonl`: full p* training manifest
- `datasets/full_oracle`: HF training dataset
- `outputs/full_oracle/final_hf`: final policy checkpoint
- `eval/full_test_generations.jsonl`: all generated responses
- `eval/full_test_scored.jsonl`: RM scores
- `report/full_oracle_metrics.json` and `report/full_oracle_prompt_means.csv`: summaries
