# Section 7 uniform vote growth experiment (2026-11-19)

This artifact contains the compact report, complete repeat-level and prompt-level tables, figures, and input/checkpoint audits for the protocol Section 7 experiment.

- Matrix: `N={1,5,11,51,101}` × 3 repeats; 15 empirical soft DPO models and 3 matched oracle soft DPO models.
- Evaluation: 500 frozen `test_diagnostic` prompts, 10 responses per model/prompt.
- Decode: temperature 0.7, top-p 0.9, max-new-tokens 1024, context 4096; generation seeds 401/402/403.
- RM: same frozen `Skywork-Reward-V2-Llama-3.1-8B` for every score.
- All 18 previously trained checkpoints were reused; no retraining was required. The input audit verifies p*, the pair list, training commands, repeat/oracle matching, and nested vote prefixes.

Report files are under `report/`. `growth_results_complete.csv` keeps each repeat; `growth_prompt_metrics.csv` keeps each prompt-level metric; `growth_results_by_N.csv` is the cross-repeat summary. The two required plots are `N_vs_mean_RM_reward.png` and `N_vs_oracle_shortfall.png` (PDF copies are included).

Raw 18 generation JSONL files (493 MB) and scored JSONL files (501 MB) remain on the experiment server at `/mnt/hezhisheng/su/convergence_eval_20261119/{generations,scored}`. Their per-file SHA256, row counts, and prompt-set checks are in `audit/post_audit.json`. The exact checkpoint and training/data audit is in `audit/growth_audit.json`.

The historical `test_diagnostic` split is disclosed in the report because it is the frozen 500-prompt evaluation split available in the v5 handoff. During r3 scoring, GPU0 was externally full; the same RM and scoring settings were rerun with batch=1 on GPU6. No model generation was rerun.
