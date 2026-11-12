# v7.3 epsilon 扩展实验交付

[完整报告](reports/V73_EPSILON_EXTENSION_REPORT.md)。新增 16 组已完成训练、生成、RM 评分和核验，共 80,000 条；合并原 34 组后为 50 组、250,000 条。报告、参数结果、逐题均值、80 行配对比较及图表均在 `reports/`，保留来源批次和纲要 commit。

`raw_results/` 包含本轮全部 16 组逐条生成及评分（各 80,000 行）、原始训练/生成/评分和重试日志、各方法实际输入、训练状态、最终模型配置和 tokenizer，按不超过 40 MiB 的普通 Git 文件分卷保存，不依赖 Git LFS。源文件相对路径、字节数、SHA256 与行数见 [RAW_RESULTS_MANIFEST.json](RAW_RESULTS_MANIFEST.json)，目录全部交付文件清单见 [ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json)。

克隆仓库后，在本目录执行：

```bash
sha256sum -c raw_results/SHA256SUMS
cat raw_results/experiment_v73_complete.tar.gz.part-* > experiment_v73_complete.tar.gz
sha256sum -c COMPLETE_ARCHIVE.sha256
mkdir -p restored
tar -xzf experiment_v73_complete.tar.gz -C restored
```

恢复后，`restored/eval/<config>/generations.jsonl` 和 `scored.jsonl` 各 5,000 行；`restored/logs/` 包含原始成功与失败尝试，`restored/data/` 包含冻结回答对、oracle、投票及逐方法实际输入。旧 34 组完整结果在 [v7.2 交付目录](../v7_dev_grid_20261110/README.md)，无需重复上传，`combined_*.csv` 已包含两批所有配置的汇总。

模型权重、优化器 checkpoint 和重复的 Hugging Face 序列化数据集保存在共享服务器 `/mnt/hezhisheng/su/experiment_v73_20261111`。访问需共享盘权限；16 个最终模型权重的路径、字节数、tensor 数量与 SHA256 见 `audit/model_exports.json`。输入 JSONL 已归档，可用准备脚本重建序列化数据集，脚本中的本机路径需按环境调整。

运行配置在 `run_metadata.json` / `v7_grid_manifest.json`，结束阶段状态在 `pipeline_status.json`。绑定纲要 commit 为 `33136d47511f477cc0189a82c1e25ae98e805a71`；确切 trainer、actor、数学实现、生成/评分脚本和依赖快照在 `audit/`，实际执行调度脚本在本目录。源码以归档内容为准；训练代码来自修改工作区，基础源码 commit 不足以单独确定实现。`audit/EXPERIMENT_PROTOCOL.md`、`audit/EXPERIMENT_STATUS.md` 是运行前冻结的文档，当前完成状态见仓库根目录状态表。

北京时间 2026-11-11 15:07:18 开始，18:12:59 完成最后评分，18:14:24 完成分析。实际训练更新数和尾批偏差、一次生成初始化 OOM 及重试、输入长度和评分核验均在报告与 `audit/` 中披露。`analysis_pipeline_preflight.json` 记录隔离目录中用旧结果验证分析程序的预检，不是本轮实验指标。

本次是在观察首批开发集结果后登记的网格扩展，单一训练 seed，全部使用开发集，没有独立 final test。
