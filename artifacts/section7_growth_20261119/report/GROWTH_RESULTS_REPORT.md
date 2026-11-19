# Section 7 uniform vote growth results

- Evaluation: `test_diagnostic.jsonl`, 500 frozen prompts; 10 responses per prompt. This is the frozen diagnostic/test split available in the v5 handoff; its historical reuse is disclosed.
- Models: 15 empirical soft DPO checkpoints + 3 matched oracle soft DPO checkpoints; 3 repeats.
- Decode: temperature 0.7, top_p 0.9, max_new_tokens 1024, context 4096; generation seeds 401/402/403.
- RM: Skywork-Reward-V2-Llama-3.1-8B, max length 16384, same frozen copy for all rows. Scoring used batch=2 on GPU0 for r1/r2 and batch=1 on GPU6 for r3 after an external GPU0 OOM; model and scores are otherwise identical.
- `R`: mean of within-prompt RM means. `S_O`: mean positive prompt-level shortfall relative to the oracle from the same repeat.

## Repeat-level complete table

| repeat | N | R | oracle R | signed gap | S_O |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 5.482117 | 5.461911 | 0.020206 | 0.968559 |
| 1 | 5 | 5.446741 | 5.461911 | -0.015170 | 1.016287 |
| 1 | 11 | 5.562856 | 5.461911 | 0.100945 | 0.889522 |
| 1 | 51 | 5.543719 | 5.461911 | 0.081808 | 0.966119 |
| 1 | 101 | 5.369932 | 5.461911 | -0.091979 | 0.978641 |
| 2 | 1 | 5.450717 | 5.430402 | 0.020315 | 0.968383 |
| 2 | 5 | 5.494575 | 5.430402 | 0.064173 | 0.910204 |
| 2 | 11 | 5.435913 | 5.430402 | 0.005512 | 0.902922 |
| 2 | 51 | 5.302043 | 5.430402 | -0.128359 | 0.964012 |
| 2 | 101 | 5.504637 | 5.430402 | 0.074235 | 0.915877 |
| 3 | 1 | 5.391730 | 5.387562 | 0.004168 | 0.986608 |
| 3 | 5 | 5.567675 | 5.387562 | 0.180113 | 0.862262 |
| 3 | 11 | 5.405604 | 5.387562 | 0.018042 | 1.004927 |
| 3 | 51 | 5.438528 | 5.387562 | 0.050966 | 0.942415 |
| 3 | 101 | 5.500482 | 5.387562 | 0.112919 | 0.962473 |
| 1 | oracle | 5.461911 | 5.461911 | 0.000000 | 0.000000 |
| 2 | oracle | 5.430402 | 5.430402 | 0.000000 | 0.000000 |
| 3 | oracle | 5.387562 | 5.387562 | 0.000000 | 0.000000 |

## Cross-repeat summary

| N | R mean ± SD | S_O mean ± SD |
|---:|---:|---:|
| 1 | 5.441521 ± 0.045890 | 0.974516 ± 0.010472 |
| 5 | 5.502997 ± 0.060905 | 0.929584 ± 0.078820 |
| 11 | 5.468125 ± 0.083428 | 0.932457 ± 0.063117 |
| 51 | 5.428097 ± 0.121175 | 0.957515 ± 0.013120 |
| 101 | 5.458350 ± 0.076601 | 0.952330 ± 0.032588 |

The highest cross-repeat mean R is at N=5; the lowest cross-repeat mean S_O is at N=5. These descriptive summaries do not require a monotone curve and should be interpreted with the three-repeat variation shown in the plots.

Files: `growth_results_complete.csv` (repeat-level rows), `growth_prompt_metrics.csv` (prompt-level rows), `growth_results_by_N.csv`, `N_vs_mean_RM_reward.png`, `N_vs_oracle_shortfall.png`, and matching PDF files.
