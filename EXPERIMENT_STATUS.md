# 实验状态与执行回传

更新日期：2026-11-11

当前纲要：[2026-11-10-v7.2-dev-grid](EXPERIMENT_PROTOCOL.md)

本页只跟踪当前方案。框架负责人维护纲要，实验执行者回填冻结配置、任务进度和结果。本轮 v7.2 的 34 个配置已完成训练、生成、评分和分析；数据、损失与完整评分核验通过。训练存在末尾 32 对梯度未提交等已披露偏差，不能标记为严格无偏差复现。结果入口见下方第 4 节。

## 1. 配置冻结表

| 项目 | 当前方案或建议 | 冻结 / 执行情况 |
|---|---|---|
| $\lambda_i$ 定义 | $\Pr(\lvert p_i-h_i\rvert\le\varepsilon\mid N_i,K_i)$ | 已实现，float64 离线缓存；参考表与实际 trainer 梯度核验通过 |
| 名义标签 / 球中心 | 原始经验比例 $h_i=K_i/N_i$ | 使用原始 h，跨配置/epoch 共享冻结投票 |
| 公共半径 | 各分支内所有题同一半径；KL 方向经验到候选；W1 标签间成本为 1 | KL 方向和 W1 成本按纲要实现，端点核验通过 |
| 主先验 | 建议 Beta(0.5,0.5) | 已冻结 Beta(0.5,0.5) |
| 先验敏感性 | 建议 Beta(1,1) | 未运行；属后续独立预算 |
| $\varepsilon$ 网格 | 建议 0.025、0.05、0.1；实验上限固定为 0.1 | 已冻结 0.025/0.05/0.1 |
| 首轮 KL 半径网格 | 0.005、0.05、0.1、0.3 | DRO 和 TRO 使用相同的四档半径 |
| 首轮 W1 半径网格 | 0.01、0.05、0.15、0.3 | DRO 和 TRO 使用相同的四档 W1 半径 |
| 数据与模型 | UltraFeedback 回答对；Llama-3.2-1B-Instruct；Skywork Reward V2 Llama-3.1-8B | 沿用 v5 已冻结完整回答对与 RM 评分；模型/tokenizer 内容 hash 已登记 |
| 数据规模与划分 | 2,000 训练对，500 道开发题 | 2000 train + 500 dev，无 prompt 重叠；输入 hash 已登记 |
| 主实验投票人数 | $N_i=1,3,5$ 按题数 4:3:3；2,000 对为 800/600/600，共 5,600 票 | 800/600/600，5600 票；allocation=1731，vote=2101，人数映射/投票已保存 |
| 训练设置 | 2048 tokens、2 epochs、有效 batch 128、lr=5e-7、DPO beta=0.01 | 已完成，DP=2，microbatch=1，累积64；尾批/更新数偏差见报告 |
| 参数网格 | KL 和 W1 各 12 组 TRO、4 组 DRO，共享经验 soft DPO 和 oracle soft DPO | 34/34 模型完成；全部候选保留 |
| 开发集评测 | 全部 34 个配置使用同一批 500 道开发题，每模型每题 10 条。逐行报告 R、S_O 与配对差值 | 34/34 配置完成，170000 条评分；逐题和配对比较已交付 |
| 代码与依赖 | trainer、投票、后验计算及评测代码 | 基于 d6dc68b 的修改工作区，确切源码快照/依赖与 hash 见 artifacts/v7_dev_grid_20261110/audit/ |
| 后续预算 | 论文对照候选为 DPO、DrDPO 和 DPO-PRO。与重训、机制对照和人数增长各自登记预算 | 见纲要第 5.1、6.3 节 |

## 2. 任务与交付标准

| 顺序 | 工作项 | 当前状态 | 完成时需要的材料 |
|---|---|---|---|
| 1 | 冻结训练数据、完整参数网格与开发题清单 | 已完成 | 划分与 hash、票数映射、seed、模型版本和实际参数 |
| 2 | 核验 loss 与开发集评测流程 | 已核验，偏差见报告 | 公式检查、生成与评分核验、实际运行成本 |
| 3 | 完成 34 组训练与开发集评测 | 分析完成 | 完整参数表、各组 R 与 S_O、逐题评分，以及相对同距离同半径 DRO 和经验 soft DPO 的配对差值 |
| 4 | 后续分析与复跑 | 待计划 | 重训、机制对照、人数增长或敏感性实验单列范围与预算 |

状态分别标记为“待执行、进行中、训练完成、生成完成、评分完成、分析完成、材料待补、失败”。复用产物时核验代码、数据、投票、模型与配置，回填来源与 hash。

## 3. 每阶段回传模板

每个阶段添加记录，保留全部候选和失败项：

| run_id | 纲要 / 实现 commit | 阶段与方法 | 先验 | epsilon | rho | allocation/vote/train/gen seed | 数据 hash | 训练状态 | 生成/评分状态 | R | S_O | 日志与产物 |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---|
| 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待执行 | 待执行 | — | — | 待上传 |

每条记录另存 ambiguity_type（KL / W1），注明 KL 方向或 W1 标签间成本，半径与 epsilon 分栏。开发集结果表按距离展示全部固定参数配置。

另附人数实际名额、映射 hash、边界标签比例、有效区间宽度、输入长度、microbatch / 累积、实际 DP 数量、训练更新数、实际 G、generation_id/前缀 hash、稳定性子集 hash、GPU 时间、失败与重试原因，以及各对照的配对差值区间。产物复用时注明对应 run_id；大文件提供持久位置、大小、hash 和访问说明。

## 4. v7_dev_grid_20261110 实际执行

绑定纲要 commit：`7beb6b46328c33731ac5684dd5939fc1463fd4fe`。策略模型使用 1B，RM 为 Skywork Reward V2 Llama-3.1-8B；先验 Beta(0.5,0.5)，N=1/3/5 按 4:3:3。训练 seed=1101，allocation/vote/generation seed=1731/2101/2101。

- [完整实验报告](artifacts/v7_dev_grid_20261110/reports/V7_DEV_GRID_REPORT.md)：参数、34 行结果、48 行配对比较、异常和解释范围。
- [完整结果 CSV](artifacts/v7_dev_grid_20261110/reports/dev_grid_results.csv)、[配对比较 CSV](artifacts/v7_dev_grid_20261110/reports/tro_baseline_comparisons.csv)、[结果图](artifacts/v7_dev_grid_20261110/reports/dev_grid_heatmap.png)。
- [运行配置](artifacts/v7_dev_grid_20261110/run_metadata.json)、[全网格清单](artifacts/v7_dev_grid_20261110/v7_grid_manifest.json)、[内容清单与 hash](artifacts/v7_dev_grid_20261110/ARTIFACT_MANIFEST.json)。
- [逐题分数均值](artifacts/v7_dev_grid_20261110/reports/prompt_scores.csv)、[输入/数学核验](artifacts/v7_dev_grid_20261110/audit/input_and_math_checks.json)、[实际 trainer 核验](artifacts/v7_dev_grid_20261110/audit/trainer_loss_checks.json)、[完整评分核验](artifacts/v7_dev_grid_20261110/audit/eval_checks.json)。

训练、生成、评分均为 34/34 完成。完整训练从 2026-11-10 09:14:49 UTC 开始，最后评分于 14:36:47 UTC 写入，共约 5 小时 22 分钟（包含初始化、显存竞争、失败及补跑）。

平均奖励最高为 W1-DRO rho=0.05：R=4.964061；除 oracle 参照外，短缺最小为 KL-TRO rho=0.005、epsilon=0.1：R=4.903892，S_O=0.882258。经验 soft DPO：R=4.761189，S_O=0.978941。最佳短缺 TRO 相对 soft DPO 的配对差值区间跨 0，不能断言稳定提升；TRO 也没有在全网格一致胜出。

执行偏差：原 trainer 的末尾 32 对梯度未提交（31 个完整更新；scheduler 计划30个）；dev 内部 loss 数据额外生成了票数字段，正式生成指标不依赖这些字段；原生成日志未保存 finish_reason，无法精确统计达到生成上限比例；显存/调度问题触发 7 个配置从初始化补跑。完整说明见报告。所有评分按完整 RM 输入复核，无截断。人数增长、先验敏感性和独立 final test 未纳入本轮 34 配置预算。

全部逐条生成及评分、原始日志、实际输入和模型配置已加入仓库的分卷归档；恢复方法、大小和 hash 见[交付目录说明](artifacts/v7_dev_grid_20261110/README.md)。模型权重与优化器 checkpoint 保留在共享服务器 `/mnt/hezhisheng/su/experiment_v7_20261110`，最终权重位置及 hash 已登记。
