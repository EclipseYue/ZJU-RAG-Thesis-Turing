# Overall TODO

本清单依据 [overall_modify_guide.md](/Users/eclipse/code/RAG/Rererank_v1/overall_modify_guide.md) 和当前确定的 [Route A 架构与迁移蓝图](/Users/eclipse/code/RAG/Rererank_v1/docs/ROUTE_A_ARCHITECTURE.md) 整理。目标是把仓库从“旧版自研 baseline + 伪异构诊断”平滑迁移到“成熟 baseline + 真实异构任务”的正式路线。

GPU 服务器迁移与真实 API 运行细则见：

- [GPU 服务器迁移与真实 API 实验运行手册](/Users/eclipse/code/RAG/Rererank_v1/docs/GPU_SERVER_RUNBOOK.md)

## P0 必做

- 已完成：调整论文中对“CoVe”的表述，统一为“基于 CoVe 思想的启发式后验验证”或明确区分“启发式近似”和“真实 LLM 验证”。
- 已完成：在实验框架中补齐可配置的真实 LLM 生成/验证接口，支持 OpenAI 兼容后端切换。
- 已完成：在论文正文前段前置说明“伪异构数据”的控制变量设计意图，避免第 3 章才暴露实验前提。
- 已完成：统一实验文档入口，避免 `README`、`docs/project_guide.md`、`docs/experiment_overview.md`、`experiments/README.md` 之间重复和过期。
- 已完成：增加 `--real-cove` 开关，支持在同一实验命令下切换“类 CoVe”和“真实 CoVe”。
- 已完成：清理旧科研可视化网站入口，避免 Flask 看板继续作为仓库主线。
- 待做：以 LlamaIndex 为底座建立可信纯文本 baseline，优先覆盖 HotpotQA。
- 已完成：建立 Route A 的 `baselines/`、`adapters/`、`modules/` 目录骨架。
- 已完成：新增 LlamaIndex 文本 baseline 包装器和 Route A 实验预设。
- 已完成：新增 `experiments/run_route_a_baseline.py`，用于驱动 Route A 文本 baseline 小样本试跑。
- 已完成：在 `.venv` 中安装 LlamaIndex，并跑通 `route_a_hotpotqa_smoke_2` 小样本验证。
- 已完成：在 GPU 服务器上完成 Route A 的 `20 -> 100` 样本真实 API 递进实验，并整理为规范化批次结果。
- 已完成：新增 `run_verification_feedback_study.py`，用于比较硬拒答、软验证与验证反馈补检索。
- 已完成：新增 `plot_tradeoff_calibration.py`，用于生成 F1--拒答率、F1--延迟与后续校准图。
- 已完成：在 GPU 服务器上运行 `verification_feedback_study_hotpotqa_50_v2/v3`，验证反馈闭环可以缓解 CoVe 崩溃。
- 已完成：对 `verification_feedback` 做 repeated-run 稳定性验证，确认真实 API 波动范围。
- 已完成：新增短答案抽取/答案格式约束实验配置，优先解决 Route A 与反馈实验中的生成格式瓶颈。
- 待做：在服务器运行 Route A short-answer 与 verification-feedback short-answer 两组 50 样本实验。
- 待做：规划真实表格/图谱数据小样本补充实验，优先用静态数据文件，谨慎引入 Neo4j 服务依赖。

## P1 强烈建议

- 已完成：为主实验入口补充 `generator_backend`、`verifier_backend`、`generator_model`、`verifier_model` 等配置项。
- 已完成：新增 API 环境配置文档与本地私有覆盖文件。
- 已完成：在实验说明中明确哪些脚本仍使用启发式生成，哪些脚本已经具备真实 LLM 接口。
- 已完成：在 `src/rererank_v1/` 下建立 `baselines/`、`adapters/`、`modules/` 新骨架，避免后续继续把新逻辑堆进旧原型文件。
- 待做：将 `Adaptive PRF`、`CoVe`、`Evidence Chain` 设计成可插拔模块，挂接到成熟 baseline。
- 待做：将 soft verification / feedback retrieval 与 short-answer extractor 迁移到 Route A baseline，避免长期停留在旧原型壳。
- 待做：将 `dataset_loader.py` 中的伪异构逻辑降级为 `legacy path`，不再作为默认实验路线。
- 待做：将旧 `run_all.py` 主消融壳明确收缩为“小样本真实 API 诊断线”，只保留关键配置 smoke 与 verifier 诊断任务。

## P2 后续可做

- 待做：引入 `HybridQA` 或 `OTT-QA` 作为真实 text+table 异构任务。
- 待做：为真实异构任务补充独立实验预设，例如 `route_a_hybridqa.json`。
- 待做：若继续使用图谱数据，优先导出为 JSONL 三元组并复用统一 EvidenceUnit，而不是直接依赖在线 Neo4j。
- 待做：补齐 `run_bucket_gain_study.py` 与其他补充脚本的统一后端开关，使所有补实验都能复用真实验证策略。
- 待做：增加真实 LLM 生成与启发式生成的小规模对照配置，但不在当前阶段实际运行。
- 待做：在新 baseline 跑稳后，更新论文主线，将旧“伪异构诊断”结果重定位为原型阶段分析。

## 交付物

- 一份 Route A 迁移蓝图。
- 一份统一实验总指南。
- 一份 API 配置说明。
- 一套支持真实 LLM 生成/验证、可迁移到成熟 baseline 的实验框架改动。
- 一套用于验证崩溃诊断、软置信度验证与反馈补检索对比的实验配置。
- 一组当前批次的权衡曲线绘图脚本与论文图表。
- 一轮仓库历史遗留清理。
