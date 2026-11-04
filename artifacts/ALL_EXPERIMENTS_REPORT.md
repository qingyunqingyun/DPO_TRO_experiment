# All Experiments Report

生成时间（UTC）：2026-11-04T11:11:47Z

本包汇总了 v3 两波、旧 v5 低异质性试验、以及 1030-lastest 下的最新 v5 试验。压缩包只包含可复现实验参数、脚本、日志、指标 JSON 和轻量数据；模型权重、完整生成结果和大体积原始分数保留在原目录，并在 `ALL_EXPERIMENTS_MANIFEST.json` 中记录路径、文件数和大小。

## 1. 实验索引

- **v3 / N={5,10,20,100}**：ARMORm reward；结果见 `historical/v3/report_n051020100.md` 和 `historical/v3/eval/pilot_metrics.json`。
- **v3 / N={1,3,5,7,9}**：ARMORm reward；结果见 `historical/v3/report_n13579.md` 和 `historical/v3/eval/pilot_n13579_metrics.json`。
- **旧 v5 / N={1,3,5,7,9}**：Skywork Reward V2 低异质性 pilot；结果见 `historical/v5_old/report_low_heterogeneous_skywork.md` 和 `historical/v5_old/eval/low_heterogeneous_metrics.json`。
- **最新 v5 / N={1,3,5}**：1B policy + Skywork 8B RM，2 epoch，screening、devB、3 次 final repeat、convergence training。

## 2. 最新 v5 参数

权威参数来自 `current_v5/protocol/resolved_runtime_manifest.json` 和 `experiment_config.json`：

- policy：本地 `Llama-3.2-1B-Instruct`，reference 等于初始 policy，max length 2048，2 epochs，lr `5e-7`，DPO beta `0.01`，BF16，gradient checkpointing，ZeRO-2，2 GPU，effective batch 128，AdamW，warmup 0.1，clip norm 1.0。
- reward model：本地 `Skywork/Skywork-Reward-V2-Llama-3.1-8B`，BF16，max length 16384，冻结，`T_pref=9.2548828125`，score field `logits_scalar`。
- data：`HuggingFaceH4/ultrafeedback_binarized` revision `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`；train/dev/test-diagnostic 分别 2000/500/500；按 full chat-template 两侧 max length ≤2048 过滤。
- annotation：`N_i ∈ {1,3,5}`，balanced random allocation seed 1731；screening vote seed 101；final repeats 使用独立训练/生成 seed。
- search：DRO radii `{0.005,0.05,0.1}`；TRO c `{1,3,5}`；15 个 screening objectives；按 dev 平均 prompt reward shortfall 选择。生成 max 1024，context 4096，K=10，temperature 0.7，top-p 0.9，bootstrap 2000。

## 3. 最新 v5 运行完成度

- screening A：15/15 训练完成并完成 dev 评估。
- devB：5/5 训练和评估完成；最终选择 DRO `rho=0.1,c=1`，TRO `rho=0.005,c=1`。
- final repeats：3 个 repeat × 5 方法 = 15/15 checkpoint 完成，最终 test 生成与评分 15/15 完成。
- convergence：18/18 个 checkpoint 训练完成，但当前目录中没有为 convergence checkpoint 生成并评分，因此本包不报告 convergence 曲线；只保留训练日志和脚本。
- final test 使用 `test_diagnostic` 的 500 个 prompt（每方法 5000 samples）；该 split 是 diagnostic/reused test prompt split，不应解读为额外独立保留测试集。

## 4. 最新 v5 final test 汇总

指标为 3 个 repeat 的均值 ± repeat 标准差；shortfall 越低越好。

| 方法 | mean reward | oracle shortfall |
|---|---:|---:|
| soft_dpo | 5.5638 ± 0.1476 | 0.9056 ± 0.1169 |
| tro | 5.4848 ± 0.1388 | 0.9477 ± 0.0802 |
| hard_dpo | 5.3917 ± 0.1317 | 0.9670 ± 0.0870 |
| label_dro | 5.4719 ± 0.0534 | 0.9672 ± 0.0868 |

逐 repeat 原始指标见 `current_v5/metrics/final_test_metrics_r1.json`、`..._r2.json`、`..._r3.json` 和合并文件 `final_test_metrics.json`。

## 5. 最新 v5 screening / devB 结果

screening A 的完整 15 项结果见 `current_v5/metrics/screening_seed101_metrics.json`；devB 的 4 个候选和 oracle 见 `current_v5/metrics/devB_metrics.json`。两者均以 prompt-level positive oracle shortfall 为主选择指标。

## 6. 历史结果

v3 和旧 v5 的原始指标、参数、报告、校验信息分别保存在 `historical/v3/` 与 `historical/v5_old/`；本报告不重新换算不同 reward 标度，因此跨实验的绝对 reward 数值不应直接横向比较。

## 7. 归档内容与大文件说明

- `current_v5/logs/` 保留当前实验日志；`current_v5/scripts/` 保留训练、生成、评分、队列和收尾脚本。
- 未打包：模型 checkpoints/weights、完整 datasets、raw 数据、完整 score blobs、final_test 生成明细。这些文件可能占用数百 GB，原始位置和大小见 `ALL_EXPERIMENTS_MANIFEST.json`。
- `ALL_EXPERIMENTS_MANIFEST.json` 记录主要已打包文件的 SHA256，`checksums.sha256` 记录本归档内全部文件，便于校验。
