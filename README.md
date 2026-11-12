# DPO / TRO：有限投票下的偏好学习实验

研究目标：每个回答对只有少量、不同数量的投票时，根据逐题后验可信程度分配经验损失和鲁棒损失，检验训练后的生成表现能否更接近 oracle soft DPO。

## 协作入口

| 文档 | 内容 | 主要维护者 |
|---|---|---|
| [实验纲要](EXPERIMENT_PROTOCOL.md) | 当前数学模型、参数组合、训练与评测流程、交付标准 | 框架负责人 |
| [实验状态与执行回传](EXPERIMENT_STATUS.md) | 本轮配置、任务进度、实际参数及结果链接 | 实验执行者 |

**当前版本为 2026-11-11-v7.3-epsilon-extension。新增 16 组已完成训练、生成、评分和分析；合并后共 50 组、250,000 条评分。**

本批交付：[完整扩展报告](artifacts/v73_epsilon_extension_20261111/reports/V73_EPSILON_EXTENSION_REPORT.md)、[50 组参数结果表](artifacts/v73_epsilon_extension_20261111/reports/combined_results.csv)、[80 行配对比较](artifacts/v73_epsilon_extension_20261111/reports/combined_comparisons.csv)、[完整归档与恢复说明](artifacts/v73_epsilon_extension_20261111/README.md)。新增 80,000 条生成与评分、原始日志和实际输入均包含在分卷归档中，完成状态见[状态表第5节](EXPERIMENT_STATUS.md#5-v73-epsilon-扩展补跑)。

已有 34 组交付：[完整报告](artifacts/v7_dev_grid_20261110/reports/V7_DEV_GRID_REPORT.md)、[34 组参数与结果](artifacts/v7_dev_grid_20261110/reports/dev_grid_results.csv)、[归档与恢复说明](artifacts/v7_dev_grid_20261110/README.md)。归档包括全部 170,000 条生成及评分、原始日志、冻结输入和源码快照；模型权重另存共享服务器并登记 hash。执行偏差与统计解释范围见报告。

可信权重为固定 Beta 先验下的后验概率：

$$
\lambda_i=\Pr\left(\left|p_i-\frac{K_i}{N_i}\right|\le\varepsilon\mid N_i,K_i\right).
$$

经验标签为 $K_i/N_i$。各配置使用预先登记的公共 DRO 半径 $\rho$ 和全局 $\varepsilon$，先验固定并报告。$\varepsilon$ 网格为 $\{0.025,0.05,0.1,0.15,0.2\}$，本次补充后两档。

首轮同时比较 KL 和 Wasserstein-1。KL 半径取 **0.005、0.05、0.1、0.3**，Wasserstein-1 半径取 **0.01、0.05、0.15、0.3**。

主实验的 $N_i=1,3,5$ 按题目数量固定为 **4:3:3**：2,000 对分别分配 800、600、600 对，总计 5,600 票、平均每题 2.8 票；随机分配人数，同一重复内所有方法共享数据，外层每题仍等权。

每种距离共 20 组 TRO 和 4 组 DRO，两种距离共享经验 soft DPO 和 oracle soft DPO，共 50 个配置。已有 34 组复用，本次补跑 8 组 KL-TRO 和 8 组 W1-TRO，新增 80,000 条生成与评分。全部配置使用同一批 500 道开发题，每题生成 10 条，逐行报告平均奖励和 oracle 短缺。

执行者按[纲要第 6.5 节的 16 组清单](EXPERIMENT_PROTOCOL.md#65-本次-epsilon-扩展补跑的-16-组)补跑。沿用原四档 KL / W1 半径、冻结数据和投票，以及所有训练和评测设置，allocation/vote/train/generation seed=1731/2101/1101/2101。进度回填[状态第 5 节](EXPERIMENT_STATUS.md#5-v73-epsilon-扩展补跑)。

本轮结果标注为开发集实验。论文对照候选为 DPO、DrDPO 和 DPO-PRO，复现设置与预算见纲要第 5.1 节。以后需要独立验证时，再划分测试集并登记评测方案。

## 如何协作

1. 框架负责人维护实验纲要，明确公式、参数范围和评测规则。
2. 实验执行者在状态表登记冻结配置与纲要 commit，按阶段实施。
3. 每阶段回填实际参数、指标、日志与产物，分别标记训练、生成、评分和分析状态。
4. 后续直接修改这两个固定文档，每次运行引用其实际采用的版本。
