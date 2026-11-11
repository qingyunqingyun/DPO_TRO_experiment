# v7.2 开发集实验交付

34 个配置的训练、生成、RM 评分与结果分析已完成。入口：[完整报告](reports/V7_DEV_GRID_REPORT.md)。

本目录包含完整参数表、48 个配对比较、逐题均值、数据/投票、确切训练源码快照、依赖版本和权重/评分文件 hash。`raw_results/` 分卷归档进一步包含全部 34 组逐条生成及评分（各 170,000 行）、全部原始日志、各方法实际输入、训练最终及初次尝试状态、最终模型的配置和 tokenizer。原始文件相对路径、大小、SHA256、评分行数与分卷校验见 [RAW_RESULTS_MANIFEST.json](RAW_RESULTS_MANIFEST.json)；目录全部交付文件见 [ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json)。

模型权重、优化器 checkpoint 和重复的 Hugging Face 序列化数据集保留在共享服务器 `/mnt/hezhisheng/su/experiment_v7_20261110`。其中模型产物约 395 GiB，不纳入 Git 仓库；`audit/model_exports.json` 给出 34 个最终模型权重的位置、字节大小与 SHA256，访问需要同一服务器/共享盘权限。所有用于重建输入的 JSONL 已包含在归档。

分卷均不超过 40 MiB。克隆仓库后，在本目录执行：

```bash
sha256sum -c raw_results/SHA256SUMS
cat raw_results/experiment_v7_complete.tar.gz.part-* > experiment_v7_complete.tar.gz
sha256sum -c COMPLETE_ARCHIVE.sha256
mkdir -p restored
tar -xzf experiment_v7_complete.tar.gz -C restored
```

恢复后，`restored/eval/<config>/generations.jsonl` 与 `scored.jsonl` 各含 5,000 行；`restored/logs/` 保留成功与失败尝试日志，`restored/data/` 为实际输入。分卷是普通 Git 文件，不依赖 Git LFS。

本轮为单一投票/训练 seed 的 500 题 dev 网格，非独立 final test。尾批梯度预算、失败补跑及其他偏差见报告。

源码注意：`audit/*_executed.py` 是运行时调度代码；目录根部的调度器已修复固定 GPU 排队及续跑只补评分的逻辑。准备数据用系统 Python + scipy；训练/生成/评分用 `audit/dependencies.json` 记录的 venv。训练源码在 `audit/`，需要 OpenRLHF 等依赖；脚本中的本机路径需按新环境改写。
