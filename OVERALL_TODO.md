# Overall TODO

本清单依据 [overall_modify_guide.md](/Users/eclipse/code/RAG/Rererank_v1/overall_modify_guide.md) 整理，目标是把后续工作拆成可执行任务，但不在本文档中直接跑实验。

## P0 必做

- 已完成：调整论文中对“CoVe”的表述，统一为“基于 CoVe 思想的启发式后验验证”或明确区分“启发式近似”和“真实 LLM 验证”。
- 已完成：在实验框架中补齐可配置的真实 LLM 生成/验证接口，支持 Moonshot/OpenAI 兼容后端，但默认不启用。
- 已完成：在论文正文前段前置说明“伪异构数据”的控制变量设计意图，避免第 3 章才暴露实验前提。
- 已完成：统一实验文档入口，避免 `README`、`docs/project_guide.md`、`docs/experiment_overview.md`、`experiments/README.md` 之间重复和过期。
- 已完成：增加 `--real-cove` 开关，支持在同一实验命令下切换“类 CoVe”和“真实 CoVe”。

## P1 强烈建议

- 已完成：为主实验入口补充 `generator_backend`、`verifier_backend`、`generator_model`、`verifier_model` 等配置项。
- 已完成：新增 API 环境配置文档，明确 Moonshot / OpenAI / SiliconFlow 三类 OpenAI 兼容接口的变量命名。
- 已完成：在论文摘要、系统设计、总结部分同步润色，使实验假设、验证近似和结论边界前后一致。
- 已完成：在实验说明中明确哪些脚本仍使用启发式生成，哪些脚本已经具备真实 LLM 接口。
- 已完成：新增本地私有 API 覆盖文件 `experiments/configs/local_api_overrides.json`，并加入 `.gitignore`。

## P2 后续可做

- 已完成：在 `run_verifier_comparison.py` 与 `run_false_rejection_diagnostics.py` 中接入统一的真实验证后端。
- 待做：补齐 `run_bucket_gain_study.py` 与其他补充脚本的统一后端开关，使所有补实验都能复用 `--real-cove`。
- 待做：增加真实 LLM 生成与启发式生成的小规模对照配置，但不在当前阶段实际运行。
- 待做：补充基于真实表格库或真实图谱库的异构扩展实验文档。
- 待做：在真实 CoVe 跑完后，分别回填“类 CoVe”和“真 CoVe”两套结果，保留双版本分析。

## 交付物

- 一份统一实验总指南。
- 一份 API 配置说明。
- 一套支持真实 LLM 生成/验证但默认关闭的实验框架改动。
- 一轮论文措辞与结构边界的精修。
- 一份本地私有 API 覆盖配置。
