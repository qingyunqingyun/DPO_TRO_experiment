# Full eligible-set oracle run (2026-11-20)

This run uses all 61,005 length-eligible, prompt-deduplicated training pairs from the v5 handoff (`data/raw/train_filtered.jsonl`). The independent evaluation split is all 1,995 eligible pairs in `data/raw/test_filtered.jsonl`; it is not used for training.

The oracle probability is recomputed with Skywork-Reward-V2-Llama-3.1-8B scores and the protocol temperature rule (median non-zero absolute score gap on the full training manifest). The policy is Llama-3.2-1B-Instruct, max length 2048, 2 epochs, effective batch 128, learning rate 5e-7, DPO beta 0.01, and the `oracle_soft_dpo` trainer.
