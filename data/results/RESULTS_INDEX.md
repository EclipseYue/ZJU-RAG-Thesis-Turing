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

### Batch D: 2026-04-29 Verification Feedback 真实 CoVe 实验

目录：

- [2026-04-29-verification-feedback](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-29-verification-feedback)

用途：

- 比较 hard reject、soft accept、claim-concat feedback 与 targeted feedback 四种验证策略
- 支撑“verification collapse 可以通过软判定与反馈补检索缓解，但收益受生成格式和反馈查询质量约束”的论文论点

推荐引用文件：

- [verification_feedback_hotpotqa_50_v2.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-29-verification-feedback/verification_feedback_hotpotqa_50_v2.json)
- [verification_feedback_hotpotqa_50_v3.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-29-verification-feedback/verification_feedback_hotpotqa_50_v3.json)
- [verification_feedback_hotpotqa_50_v3_run2.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-29-verification-feedback/verification_feedback_hotpotqa_50_v3_run2.json)
- [verification_feedback_hotpotqa_50_v3_run3.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-29-verification-feedback/verification_feedback_hotpotqa_50_v3_run3.json)
- [verification_feedback_policy_smoke.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-29-verification-feedback/verification_feedback_policy_smoke.json)
- [verification_feedback_targeted_smoke.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-29-verification-feedback/verification_feedback_targeted_smoke.json)

说明：

- `verification_feedback_study_hotpotqa_50.json` 是修复 verifier 解析与置信度口径前的污染版，只保留作排障记录，不建议引用。
- `verification_feedback_study_hotpotqa_50_v2.json` 是第一版可引用版本。
- `verification_feedback_study_hotpotqa_50_v3.json` 是当前推荐引用版本。

### Batch E: 2026-04-29 Short-answer Constraint 实验

目录：

- [2026-04-29-short-answer](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-29-short-answer)

用途：

- 检查严格短答案提示是否能修复 Route A 与反馈闭环中的答案格式问题。
- 作为“生成格式控制不能只靠 prompt 压缩”的负结果证据。

推荐引用文件：

- [route_a_hotpotqa_realapi_50_short_answer_matrix.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-29-short-answer/route_a_hotpotqa_realapi_50_short_answer_matrix.json)
- [route_a_hotpotqa_realapi_50_short_answer_report.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-29-short-answer/route_a_hotpotqa_realapi_50_short_answer_report.json)
- [verification_feedback_hotpotqa_50_short_answer.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-29-short-answer/verification_feedback_hotpotqa_50_short_answer.json)

### Batch F: 2026-04-30 Real-CoVe Follow-up 实验

目录：

- [2026-04-30-real-cove-followup](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-30-real-cove-followup)

用途：

- 复核真实 LLM verifier 条件下 hard reject、soft accept、feedback 与 targeted feedback 的差异。
- 用 200 样本 verifier comparison 支撑“阈值越严格，拒答率越高，不安全接受率越低”的 tradeoff 结论。
- 注意：源文件 `verification_feedback_real_cove_100.json` 的文件名包含 100，但内部配置 `samples=50`，论文和索引均按真实样本数 N=50 引用。

推荐引用文件：

- [verification_feedback_real_cove_50_followup.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-30-real-cove-followup/verification_feedback_real_cove_50_followup.json)
- [verifier_comparison_real_cove_200.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-30-real-cove-followup/verifier_comparison_real_cove_200.json)

### Batch G: 2026-04-30 Real LLM Ablation 复核实验

目录：

- [2026-04-30-real-llm-ablation](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-30-real-llm-ablation)

用途：

- 在 \texttt{deepseek-v4-flash} 真实生成/验证条件下复核 A/A2/A3/B/D 的核心消融趋势。
- 将真实 LLM 反馈闭环扩展到 N=300，检查 hard reject、soft accept、verification feedback 与 targeted feedback 的方向是否稳定。
- 支撑论文中的“大规模启发式诊断 + 真实 LLM 复核”分层实验设计。

推荐引用文件：

- [real_llm_text_ablation_100.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-30-real-llm-ablation/real_llm_text_ablation_100.json)
- [real_llm_hetero_ablation_100.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-30-real-llm-ablation/real_llm_hetero_ablation_100.json)
- [real_llm_full_cove_100.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-30-real-llm-ablation/real_llm_full_cove_100.json)
- [real_llm_verification_feedback_100.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/batches/2026-04-30-real-llm-ablation/real_llm_verification_feedback_100.json)
- [automated_ablation_real_llm_300.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/automated_ablation_real_llm_300.json)
- [verification_feedback_study_hotpotqa_300_real_cove_rerun.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/verification_feedback_study_hotpotqa_300_real_cove_rerun.json)

### Batch H: 2026-05-08 评阅注释后补充实验

用途：

- 回应“部分样本量偏小”的评阅意见。
- 补充 Route A N=300、HybridQA text-table smoke，并记录无效批次，防止误引用。

推荐引用文件：

- [route_a_hotpotqa_realapi_300.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/route_a_hotpotqa_realapi_300.json)
- [route_a_hybridqa_text_table_smoke_50.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/route_a_hybridqa_text_table_smoke_50.json)

不建议引用文件：

- [automated_ablation_real_llm_300.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/automated_ablation_real_llm_300.json)：矩阵中 `MockMode=true`，属于 mock fallback 后的合成指标。
- [verification_feedback_study_hotpotqa_300_real_cove.json](/Users/eclipse/code/RAG/Rererank_v1/data/results/verification_feedback_study_hotpotqa_300_real_cove.json)：四个策略完全同分、反馈率为 0、延迟异常低，判断真实 API/验证链未按预期生效。

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

300 样本真实 API：

- `SupportRecall@K = 72.83`
- `SupportAllHit@K = 48.67`
- `ExactMatch = 1.33`
- `F1 = 6.19`

结论：

- 检索侧基本稳定
- 真实 API 生成链已打通
- 当前 `deepseek-v4-flash` 生成质量仍明显偏弱，尚不足以直接当作最终正式 baseline
- N=300 进一步显示主要瓶颈在答案 span 抽取和答案格式，而不是单纯检索不可用

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

### 2.3 Verification Feedback 批次

#### v2 结果

`hard_reject`：

- `ExactMatch = 6.0`
- `F1 = 13.15`
- `No_Answer_Rate = 40.0`
- `Avg_Latency_ms = 266.7`

`soft_accept`：

- `ExactMatch = 10.0`
- `F1 = 19.19`
- `No_Answer_Rate = 34.0`
- `Avg_Latency_ms = 261.67`

`verification_feedback`：

- `ExactMatch = 12.0`
- `F1 = 20.47`
- `No_Answer_Rate = 20.0`
- `Feedback_Rate = 34.0`
- `Avg_Latency_ms = 356.37`
- `Avg_Retrieval_Calls = 2.68`

结论：

- soft accept 相比 hard reject 明显降低了过度拒答，并带来 F1 提升。
- verification feedback 进一步把拒答率降到 20.0%，F1 小幅提升到 20.47，但增加了约 36% 的端到端延迟。
- 该批结果适合用于论文中的 tradeoff 分析，而不是作为“最终强性能系统”声明。

#### v3 结果

`hard_reject`：

- `ExactMatch = 0.0`
- `F1 = 8.78`
- `No_Answer_Rate = 44.0`
- `Avg_Latency_ms = 264.64`

`soft_accept`：

- `ExactMatch = 6.0`
- `F1 = 13.58`
- `No_Answer_Rate = 44.0`
- `Avg_Latency_ms = 259.83`

`verification_feedback`：

- `ExactMatch = 14.0`
- `F1 = 25.35`
- `No_Answer_Rate = 12.0`
- `Feedback_Rate = 30.0`
- `Avg_Latency_ms = 345.86`
- `Avg_Retrieval_Calls = 2.60`

`targeted_feedback`：

- `ExactMatch = 10.0`
- `F1 = 20.98`
- `No_Answer_Rate = 16.0`
- `Feedback_Rate = 24.0`
- `Avg_Latency_ms = 338.83`
- `Avg_Retrieval_Calls = 2.48`

结论：

- v3 中 claim-concat `verification_feedback` 是当前最佳配置，F1 达到 25.35，拒答率降至 12.0%。
- `targeted_feedback` 没有超过 claim-concat feedback，说明当前简单的实体/标题增强查询并未稳定提升补检索质量。
- v2 与 v3 的 hard/soft 结果存在波动，说明真实 API 生成与验证具有一定非确定性；后续若要写更稳结论，需要做 repeated-run 稳定性验证。

#### v3 repeated-run 稳定性

三次 50 样本 repeated-run 的均值如下：

`hard_reject`：

- `F1 = 11.77 ± 2.64`
- `No_Answer_Rate = 42.67 ± 1.89`

`soft_accept`：

- `F1 = 15.36 ± 1.68`
- `No_Answer_Rate = 34.00 ± 7.12`

`verification_feedback`：

- `F1 = 23.51 ± 1.43`
- `No_Answer_Rate = 12.67 ± 0.94`
- `Feedback_Rate = 30.00 ± 3.27`
- `Avg_Latency_ms = 345.40 ± 6.04`

`targeted_feedback`：

- `F1 = 20.87 ± 0.88`
- `No_Answer_Rate = 18.00 ± 1.63`
- `Feedback_Rate = 28.00 ± 3.27`
- `Avg_Latency_ms = 349.80 ± 10.54`

结论：

- `verification_feedback` 在三次运行中稳定优于 `hard_reject`、`soft_accept` 与 `targeted_feedback`。
- claim-concat 反馈不是偶然单次峰值，而是当前最稳的反馈闭环配置。
- 论文可以用 repeated-run 均值报告该部分结果，并把单次 v3 最优值作为补充说明。

### 2.4 Real-CoVe Follow-up 批次

#### 真实 LLM verifier 反馈复核（N=50）

`hard_reject`：

- `ExactMatch = 4.0`
- `F1 = 13.19`
- `No_Answer_Rate = 48.0`

`soft_accept`：

- `ExactMatch = 12.0`
- `F1 = 22.81`
- `No_Answer_Rate = 30.0`

`verification_feedback`：

- `ExactMatch = 12.0`
- `F1 = 21.68`
- `No_Answer_Rate = 16.0`
- `Feedback_Rate = 34.0`

`targeted_feedback`：

- `ExactMatch = 14.0`
- `F1 = 25.87`
- `No_Answer_Rate = 16.0`
- `Feedback_Rate = 24.0`

结论：

- 真实 LLM verifier 下，反馈闭环仍显著优于硬拒答。
- 本轮 `targeted_feedback` 高于 claim-concat，说明 targeted query 不是无效方向，但稳定性仍需更多批次确认。
- 论文中应将稳定结论写成“验证失败后的反馈补检索有效”，而不是过度声称某一种 query 构造稳定最优。

#### 真实 verifier 阈值对比（N=200）

`cove_soft`：

- `F1 = 15.05`
- `No_Answer_Rate / FRR = 29.5`
- `Unsafe_Accept_Rate = 33.0`

`cove_standard`：

- `F1 = 15.23`
- `No_Answer_Rate / FRR = 28.0`
- `Unsafe_Accept_Rate = 33.0`

`cove_strict`：

- `F1 = 12.01`
- `No_Answer_Rate / FRR = 59.5`
- `Unsafe_Accept_Rate = 13.0`

`overlap_soft`：

- `F1 = 12.23`
- `No_Answer_Rate / FRR = 59.5`
- `Unsafe_Accept_Rate = 15.5`

结论：

- 提高阈值可以降低不安全接受率，但会明显推高错误拒答率并压低 F1。
- 真实 LLM verifier 在 soft 阈值下比 overlap-soft 更可用，但仍没有形成理想的安全性-可用性折中。

### 2.5 Short-answer Constraint 批次

`Route A strict-short`：

- `ExactMatch = 0.0`
- `F1 = 2.66`
- `SupportRecall@K = 78.0`
- `SupportAllHit@K = 56.0`

`soft_accept_short_answer`：

- `ExactMatch = 0.0`
- `F1 = 1.21`
- `No_Answer_Rate = 32.0`

`verification_feedback_short_answer`：

- `ExactMatch = 0.0`
- `F1 = 2.01`
- `No_Answer_Rate = 16.0`
- `Feedback_Rate = 44.0`

结论：

- 严格短答案 prompt 显著退化，说明答案格式问题不能仅靠 prompt 约束解决。
- 样本检查显示模型仍频繁输出 “We need to extract...” 这类元推理句，只是被截断得更短。
- 后续若继续优化，应实现候选答案抽取器或 span reranker，而不是继续压缩 prompt。

### 2.6 Real LLM Ablation 复核批次

#### A/A2/A3/B/D 核心消融（N=300）

`A Text`：

- `ExactMatch = 6.0`
- `F1 = 18.40`
- `SupportRecall@K = 76.5`
- `SupportAllHit@K = 56.0`
- `No_Answer_Rate = 0.0`

`A2 Text+Adaptive`：

- `ExactMatch = 6.0`
- `F1 = 19.70`
- `SupportRecall@K = 76.5`
- `SupportAllHit@K = 56.0`
- `No_Answer_Rate = 0.0`

`A3 Text+CoVe`：

- `ExactMatch = 2.0`
- `F1 = 9.61`
- `SupportRecall@K = 73.83`
- `SupportAllHit@K = 50.67`
- `No_Answer_Rate = 57.0`

`B Hetero`：

- `ExactMatch = 5.33`
- `F1 = 16.51`
- `SupportRecall@K = 70.83`
- `SupportAllHit@K = 47.0`
- `No_Answer_Rate = 0.0`

`D Full`：

- `ExactMatch = 2.33`
- `F1 = 9.56`
- `SupportRecall@K = 71.33`
- `SupportAllHit@K = 47.33`
- `No_Answer_Rate = 61.33`

结论：

- 真实 LLM 条件下，Adaptive 只带来小幅 F1 增益，仍不能被表述为端到端优化器。
- CoVe hard reject 在 A3/D 中继续显著推高拒答率，验证崩溃不是启发式后端独有现象。
- 当前伴生式异构序列化仍低于纯文本配置，论文应将其定位为格式噪声诊断，而非真实异构知识收益证明。

#### 真实 LLM 反馈闭环（N=300）

`hard_reject`：

- `ExactMatch = 1.0`
- `F1 = 12.48`
- `No_Answer_Rate = 39.67`
- `Feedback_Rate = 0.0`

`soft_accept`：

- `ExactMatch = 3.33`
- `F1 = 15.05`
- `No_Answer_Rate = 27.67`
- `Feedback_Rate = 0.0`

`verification_feedback`：

- `ExactMatch = 4.67`
- `F1 = 18.15`
- `No_Answer_Rate = 14.33`
- `Feedback_Rate = 24.67`
- `SupportAllHit@K = 56.67`

`targeted_feedback`：

- `ExactMatch = 4.67`
- `F1 = 17.56`
- `No_Answer_Rate = 14.67`
- `Feedback_Rate = 28.33`
- `SupportAllHit@K = 56.67`

结论：

- 反馈闭环在 N=300 上继续显著降低 hard reject 的过度拒答。
- 普通反馈与定向反馈效果接近，前者 F1 略高，后者拒答率略低。
- 反馈闭环的作用应表述为“缓解验证崩溃”，而不是“超过无验证纯生成配置”。

## 3. 当前推荐引用顺序

如果你现在要继续实验或写文档，建议按以下优先级引用结果：

1. Real LLM Ablation N=300 复核批次：`automated_ablation_real_llm_300.json`
2. VAR Real-CoVe N=300 扩样本批次：`verification_feedback_study_hotpotqa_300_real_cove_rerun.json`
3. Route A 服务器 `realapi_300`
4. HybridQA text-table smoke `route_a_hybridqa_text_table_smoke_50`
5. Verifier threshold N=200：`verifier_comparison_hotpotqa.json`
6. Verification Feedback repeated-run `hotpotqa_50_v3/run2/run3`
7. Short-answer Constraint 批次
8. Route A 服务器 `realapi_smoke_latest`
9. 旧消融服务器 `legacy_a_baseline_smoke`
10. 旧消融服务器 `legacy_a3_cove_smoke`

## 3.1 当前批次对比图

已生成的可视化图表：

- [route_a_quality_comparison.png](/Users/eclipse/code/RAG/Rererank_v1/paper/zjuthesis/figures/route_a_quality_comparison.png)
- [current_server_batch_comparison.png](/Users/eclipse/code/RAG/Rererank_v1/paper/zjuthesis/figures/current_server_batch_comparison.png)
- [current_batch_latency_noanswer.png](/Users/eclipse/code/RAG/Rererank_v1/paper/zjuthesis/figures/current_batch_latency_noanswer.png)
- [tradeoff_f1_rejection.png](/Users/eclipse/code/RAG/Rererank_v1/paper/zjuthesis/figures/tradeoff_f1_rejection.png)
- [tradeoff_f1_latency.png](/Users/eclipse/code/RAG/Rererank_v1/paper/zjuthesis/figures/tradeoff_f1_latency.png)
- [verifier_calibration.png](/Users/eclipse/code/RAG/Rererank_v1/paper/zjuthesis/figures/verifier_calibration.png)
- [verifier_threshold_tradeoff.png](/Users/eclipse/code/RAG/Rererank_v1/paper/zjuthesis/figures/verifier_threshold_tradeoff.png)
- [short_answer_ablation.png](/Users/eclipse/code/RAG/Rererank_v1/paper/zjuthesis/figures/short_answer_ablation.png)
- [real_llm_ablation_followup.png](/Users/eclipse/code/RAG/Rererank_v1/paper/zjuthesis/figures/real_llm_ablation_followup.png)
- [real_llm_feedback_followup.png](/Users/eclipse/code/RAG/Rererank_v1/paper/zjuthesis/figures/real_llm_feedback_followup.png)

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
- `verifier_calibration.png`
  展示验证置信度与实际回答正确性的关系，用于支撑 calibration 分析
- `verifier_threshold_tradeoff.png`
  展示真实 verifier 阈值在错误拒答与不安全接受之间的权衡
- `short_answer_ablation.png`
  展示严格短答案约束带来的退化，用于支撑答案抽取瓶颈分析
- `real_llm_ablation_followup.png`
  展示真实 LLM 条件下 A/A2/A3/B/D 的 N=300 F1、覆盖率与拒答率复核
- `real_llm_feedback_followup.png`
  展示 N=300 真实 LLM 反馈闭环在 F1、拒答率与证据覆盖率上的变化

当前 `experiments/plot_tradeoff_calibration.py` 会优先读取 `verification_feedback_study_hotpotqa_300_real_cove_rerun.json` 和 `verifier_comparison_hotpotqa.json`，用于生成最新权衡曲线与校准图。

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
- repeated-run 与短答案约束实验已完成；下一步若继续实验，应实现候选答案抽取器或 span reranker。
- 真实异构数据可以补充，但应作为独立 smoke：表格优先 HybridQA/OTT-QA，图谱优先静态 JSONL 图谱问答文件，不建议正式实验依赖在线 Neo4j 服务。
