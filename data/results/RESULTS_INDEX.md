# Results Index

本文档用于整理 `data/results/` 下当前仍在使用或有参考价值的结果文件，并给出推荐引用顺序。

原则：

- 根目录原始文件全部保留，不做破坏性清理。
- `data/results/batches/` 下提供按批次整理后的规范化别名。
- 写论文、做后续实验分析时，优先引用 `batches/` 下的规范化文件。

## 1. 当前批次划分

### Batch A: 2026-04-28 Route A 服务器实验

目录：

- [2026-04-28-route-a-server](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-28-route-a-server)

用途：

- Route A 文本 baseline
- 服务器环境下的真实 API 与启发式对照

推荐引用文件：

- [route_a_hotpotqa_heuristic_smoke_matrix.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-28-route-a-server/route_a_hotpotqa_heuristic_smoke_matrix.json)
- [route_a_hotpotqa_realapi_smoke_latest_matrix.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-28-route-a-server/route_a_hotpotqa_realapi_smoke_latest_matrix.json)
- [route_a_hotpotqa_realapi_100_matrix.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-28-route-a-server/route_a_hotpotqa_realapi_100_matrix.json)

三次 `realapi smoke` 报告已按运行顺序归档：

- [route_a_hotpotqa_realapi_smoke_run1_report.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-28-route-a-server/route_a_hotpotqa_realapi_smoke_run1_report.json)
- [route_a_hotpotqa_realapi_smoke_run2_report.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-28-route-a-server/route_a_hotpotqa_realapi_smoke_run2_report.json)
- [route_a_hotpotqa_realapi_smoke_run3_report.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-28-route-a-server/route_a_hotpotqa_realapi_smoke_run3_report.json)

### Batch B: 2026-04-28 旧消融服务器 smoke

目录：

- [2026-04-28-legacy-server-smoke](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-28-legacy-server-smoke)

用途：

- 旧 `run_all.py` 壳在服务器上的小样本诊断
- 只用于过渡期诊断，不建议作为后续主线 baseline

推荐引用文件：

- [legacy_a_baseline_smoke_matrix.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-28-legacy-server-smoke/legacy_a_baseline_smoke_matrix.json)
- [legacy_a3_cove_smoke_matrix.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-28-legacy-server-smoke/legacy_a3_cove_smoke_matrix.json)

### Batch C: 2026-04-28 Route A 本地历史 smoke

目录：

- [2026-04-28-local-route-a-history](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-28-local-route-a-history)

用途：

- Route A 在本地初期接线阶段的 heuristic smoke
- 主要用于排查与回顾，不建议作为论文正式结果引用

## 2. 当前结果摘要

### 2.1 Route A 服务器批次

启发式检索 smoke：

- `SupportRecall@K = 75.0`
- `SupportAllHit@K = 50.0`
- `ExactMatch = 5.0`
- `F1 = 10.08`

含义：

- 检索链可运行
- 启发式生成很弱，只适合做排障基线

真实 API smoke（最新保留矩阵）：

- `SupportRecall@K = 75.0`
- `SupportAllHit@K = 50.0`
- `ExactMatch = 15.0`
- `F1 = 23.82`

100 样本真实 API：

- `SupportRecall@K = 75.5`
- `SupportAllHit@K = 53.0`
- `ExactMatch = 8.0`
- `F1 = 21.12`

结论：

- 检索侧基本稳定
- 真实 API 生成链已打通
- 当前 `deepseek-v4-flash` 生成质量仍明显偏弱，尚不足以直接当作最终正式 baseline

### 2.2 旧消融服务器批次

`A_Baseline` smoke：

- `ExactMatch = 15.0`
- `F1 = 28.24`

`A3_Baseline_CoVe` smoke：

- `ExactMatch = 15.0`
- `F1 = 15.0`
- `No_Answer_Rate = 75.0`

结论：

- 旧实验壳在服务器上已经稳定可跑
- 但真实 CoVe 下拒答率仍然很高
- 这批结果更适合用作诊断和论文中的“问题暴露”材料

## 3. 当前推荐引用顺序

如果你现在要继续实验或写文档，建议按以下优先级引用结果：

1. Route A 服务器 `realapi_100`
2. Route A 服务器 `realapi_smoke_latest`
3. Route A 服务器 `heuristic_smoke`
4. 旧消融服务器 `legacy_a_baseline_smoke`
5. 旧消融服务器 `legacy_a3_cove_smoke`

## 3.1 当前批次对比图

已生成的可视化图表：

- [route_a_quality_comparison.png](/Users/eclipse/code/RAG/Rererank_v1/paper/zjuthesis/figures/route_a_quality_comparison.png)
- [current_server_batch_comparison.png](/Users/eclipse/code/RAG/Rererank_v1/paper/zjuthesis/figures/current_server_batch_comparison.png)
- [current_batch_latency_noanswer.png](/Users/eclipse/code/RAG/Rererank_v1/paper/zjuthesis/figures/current_batch_latency_noanswer.png)
- [tradeoff_f1_rejection.png](/Users/eclipse/code/RAG/Rererank_v1/paper/zjuthesis/figures/tradeoff_f1_rejection.png)
- [tradeoff_f1_latency.png](/Users/eclipse/code/RAG/Rererank_v1/paper/zjuthesis/figures/tradeoff_f1_latency.png)

用途：

- `route_a_quality_comparison.png`
  对比 Route A 在 heuristic / real API smoke / real API 100 样本下的回答质量与检索稳定性
- `current_server_batch_comparison.png`
  对比 Route A 与旧消融服务器 smoke 的 EM/F1 水平
- `current_batch_latency_noanswer.png`
  对比当前各批次的延迟与拒答行为
- `tradeoff_f1_rejection.png`
  展示当前批次的 F1--拒答率权衡，用于支撑 verification collapse 讨论
- `tradeoff_f1_latency.png`
  展示当前批次的 F1--延迟权衡，用于支撑成本-效果讨论

后续若存在 `data/results/verification_feedback_study_hotpotqa*.json`，运行 `experiments/plot_tradeoff_calibration.py` 后会将 hard/soft/feedback 变体自动加入权衡图，并在逐样本记录含验证置信度时生成：

- `paper/zjuthesis/figures/verifier_calibration.png`

## 4. 命名规范

后续建议统一采用：

- `route_a_<dataset>_<backend>_<samples>_matrix.json`
- `route_a_<dataset>_<backend>_<samples>_report.json`
- `legacy_<config>_<purpose>_matrix.json`
- `legacy_<config>_<purpose>_report.json`

示例：

- `route_a_hotpotqa_realapi_100_matrix.json`
- `route_a_hotpotqa_realapi_100_report.json`
- `legacy_a_baseline_smoke_matrix.json`
- `legacy_a3_cove_smoke_report.json`

## 5. 下一步建议

- Route A 继续做误差分析和模型对照，而不是立刻扩更大样本。
- 旧消融壳只保留诊断角色，不再作为后续主线的主要结果来源。
- 下一批优先运行 `verification_feedback_study_hotpotqa_50`，观察软验证与反馈补检索能否降低 CoVe 造成的过度拒答。
