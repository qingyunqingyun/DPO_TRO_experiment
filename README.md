# DPO / TRO：有限投票下的偏好学习实验

研究目标：每个回答对只有少量、不同数量的投票时，利用逐题可信程度分配经验损失和鲁棒损失，检验训练后的生成表现能否更接近 oracle soft DPO。

## 当前协作入口

| 文档 | 用途 | 主要维护者 |
|---|---|---|
| [实验纲要](EXPERIMENT_PROTOCOL.md) | 研究问题、数学模型、参数、实验矩阵、评测与验收规则 | 框架负责人 |
| [实验状态与执行回传](EXPERIMENT_STATUS.md) | 已完成工作、待办、参数冻结、结果链接与实际偏离 | 实验执行者 |
| [历史实验总报告](artifacts/ALL_EXPERIMENTS_REPORT.md) | 2026-11-04 归档中的旧版结果 | 保留历史记录 |

**当前纲要版本：2026-11-09-v6-posterior-confidence。新方案尚未实施。**

本版将旧的信任规则 $\lambda_i=N_i/(N_i+c)$ 改为：

$$
\lambda_i=\Pr\!\left(\left|p_i-\frac{K_i}{N_i}\right|\le\varepsilon\mid N_i,K_i\right).
$$

在固定 Beta 先验下，根据每题人数和投票比例计算这个后验概率。经验标签仍为 $K_i/N_i$；所有题共用一个 DRO 半径 $\rho$。主要开发搜索参数是全局 $\varepsilon$ 和 $\rho$，先验也是需要公开报告的模型设定。

## 如何协作

1. 框架负责人直接修改实验纲要，记录版本与变更原因。
2. 实验执行者在状态表登记拟执行配置、所依据的纲要 commit，再分阶段运行。
3. 每阶段回传实际参数、指标、日志和原始结果位置；把“训练完成”和“评测完成”分开记录。
4. 改变标签、模型、数据清单、输入长度或评测规则后，重新检查哪些旧结果可以复用。

后续维护这两个固定文档即可；历史版本由 Git 保存。不要把实验方案只放在压缩包里，也不要覆盖旧版结果来冒充新方案的结果。

## 历史归档

- [旧实验归档压缩包](artifacts/experiment_bundle_20261104.tar.gz)
- [归档校验值](artifacts/experiment_bundle_20261104.tar.gz.sha256)
- [历史参数与文件清单](artifacts/ALL_EXPERIMENTS_MANIFEST.json)

压缩包内旧纲要位于 current_v5/protocol/EXPERIMENT_PROTOCOL.md，版本为 2026-10-30-v5-2048-convergence。旧版已完成主实验的训练和诊断评测，但统一人数收敛实验只完成训练；具体缺口见[状态表](EXPERIMENT_STATUS.md)。
