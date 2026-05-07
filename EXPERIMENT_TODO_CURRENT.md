# 当前实验待办清单

本文档面向 `2026-04-29` 这一阶段的实际推进情况整理。当前目标已经从“直接补真实 CoVe”进一步调整为：

- 用 **Route A** 建立可信 baseline
- 把旧消融壳降级为“小样本真实 API 诊断线”
- 把 CoVe 失败从“现象描述”推进到“软验证/反馈检索”的方法对照
- 增加 F1--拒答率、F1--延迟、验证置信度校准等论文型图表
- 在 v3 结果基础上，把下一轮重点转向短答案生成/答案抽取与 repeated-run 稳定性验证
- 在 GPU 服务器上稳定推进，不再重复之前的长时全损问题

总原则：

- 保留现有历史结果，但不再把旧 baseline 当成后续主线。
- Route A baseline 结果优先级高于旧消融壳的大样本补跑。
- 真实大模型实验只做**小样本补充验证**，不追求全量 7405。
- GPU 服务器优先用于本地检索、嵌入、重排和 baseline 建设。
- 生成与验证走真实 API，但必须控制请求速率、样本量和配置范围。

推荐搭配阅读：

- [GPU 服务器迁移与真实 API 实验运行手册](/Users/eclipse/code/RAG/Rererank_v1/docs/GPU_SERVER_RUNBOOK.md)

建议先在服务器设置限速环境变量：

```bash
export RERERANK_LLM_MIN_INTERVAL_SEC=3.5
export RERERANK_LLM_MAX_RETRIES=8
export RERERANK_LLM_BACKOFF_BASE_SEC=8
export RERERANK_LLM_BACKOFF_MAX_SEC=90
```

## P0 必做

### 0. 当前结论与下一轮优先级

已完成：

- Route A 真实 API 100 样本文本基线。
- Verification Feedback v2/v3 小样本真实 CoVe 对照。
- F1--拒答率、F1--延迟与 verifier calibration 图。

当前最佳反馈配置：

- `verification_feedback_study_hotpotqa_50_v3.json` 中的 `verification_feedback`
- `F1 = 25.35`
- `No_Answer_Rate = 12.0%`
- `Avg_Latency_ms = 345.86`
- `Avg_Retrieval_Calls = 2.60`

关键判断：

- `targeted_feedback` 没有超过 claim-concat `verification_feedback`，暂时不继续沿这个方向微调。
- 当前主要瓶颈已经从“是否能缓解 CoVe collapse”转向“生成器能否稳定输出 HotpotQA 短答案”。
- 下一轮 P0 是答案抽取/短答案约束与 repeated-run 稳定性验证。

下一轮建议顺序：

1. 已完成 `verification_feedback` repeated-run 稳定性验证。
2. 已完成 `Route A + strict_short` 与 `verification_feedback + strict_short` 的 50 样本对照。
3. 下一步若继续优化，应实现候选答案抽取器或 span reranker，而不是继续压缩 prompt。
4. 真实表格/图谱数据作为独立 smoke 规划，不并入当前主线大实验。

### 1. GPU 服务器准备与 Route A 环境验证

任务：

- 在 GPU 服务器上重建 `.venv`
- 安装 Route A 依赖
- 同步 `local_api_overrides.json` 和离线数据

命令：

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  -r requirements-route-a.txt
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
.venv/bin/python -c "import llama_index; import llama_index.core; print('llamaindex ok')"
ls experiments/configs/local_api_overrides.json
```

通过标准：

- `.venv` 可用
- `llama_index` 可导入
- `local_api_overrides.json` 存在
- 离线数据目录存在

### 2. Route A 检索 smoke

任务：

- 先在服务器上确认 Route A baseline 能正常跑
- 先只检查检索、嵌入、落盘链路

命令：

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 20 \
  --generator-backend heuristic \
  --output-name route_a_hotpotqa_heuristic_server_smoke.json
```

预期产物：

- `data/results/route_a_hotpotqa_heuristic_server_smoke.json`

### 3. Route A 真实生成 smoke

任务：

- 验证 Route A + 真实 API 生成链是否稳定
- 检查空答案、长句复述和极端慢速是否仍存在
- 当前这一条默认就是**真实 API**

命令：

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 20 \
  --output-name route_a_hotpotqa_realapi_smoke.json
```

### 4. Route A 100 样本小规模 baseline

任务：

- 获取第一版服务器上的 Route A 候选 baseline 结果

命令：

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 100 \
  --output-name route_a_hotpotqa_realapi_100.json
```

说明：

- 这一步优先级高于旧版大样本消融
- 只要这一步还没稳，旧 baseline 就不应该继续扩样本
- 该命令默认使用 `deepseek / deepseek-v4-flash`，除非你显式改回 heuristic

### 5. 旧主消融单配置 smoke test

任务：

- 验证旧实验壳在服务器上是否能稳定落盘
- 只检查最基础的 `A_Baseline`

命令：

```bash
.venv/bin/python experiments/run_all.py \
  --config experiments/configs/ablation_with_controls.json \
  --samples 20 \
  --only-configs A_Baseline \
  --checkpoint-every 2 \
  --output-name automated_ablation_smoke_a_server.json
```

预期产物：

- `data/results/automated_ablation_smoke_a_server.json`

通过标准：

- 脚本正常结束
- 输出 JSON 正常落盘
- 没有再次出现长时间无结果文件的情况

### 6. 真实 CoVe 关键配置 smoke

任务：

- 只运行真正依赖 CoVe 的关键配置
- 避免一上来扫全量 A/A2/A3/B/C/D

命令：

```bash
.venv/bin/python experiments/run_all.py \
  --config experiments/configs/ablation_with_controls.json \
  --samples 20 \
  --real-cove \
  --only-configs A3_Baseline_CoVe D_CoVe_Full \
  --checkpoint-every 2 \
  --output-name automated_ablation_smoke_cove_server.json
```

预期产物：

- `data/results/automated_ablation_smoke_cove_server.json`

### 7. Verification-aware 反馈闭环实验

任务：

- 比较硬拒答、软置信度接受、失败声明拼接反馈、定向补检索四种策略
- 把“verification collapse”从负结果推进为可修复的方法问题
- 该实验默认使用真实生成端与真实 CoVe 验证端，样本量先从 50 开始

命令：

```bash
.venv/bin/python experiments/run_verification_feedback_study.py \
  --config experiments/configs/verification_feedback_study.json \
  --samples 50 \
  --real-cove \
  --output-name verification_feedback_study_hotpotqa_50_v3.json
```

预期产物：

- `data/results/verification_feedback_study_hotpotqa_50_v3.json`

用于论文：

- 对比 `hard_reject / soft_accept / verification_feedback / targeted_feedback`
- 报告 F1、拒答率、平均验证置信度、反馈触发率、平均检索次数与延迟
- 若 `targeted_feedback` 相比 `verification_feedback` 提升 F1 或降低拒答率，可作为“定向反馈闭环”贡献

当前结果：

- v3 中 `verification_feedback` 优于 `targeted_feedback`，因此后续默认保留 `verification_feedback` 作为反馈闭环代表。
- `targeted_feedback` 作为负结果保留，用于说明“更复杂的 query 构造不必然带来收益”。

### 7.1 Repeated-run 稳定性验证

任务：

- 检查真实 API 下 `verification_feedback` 结果是否稳定。
- 避免单次 50 样本结果受生成器波动影响过大。

当前状态：

- 已完成 `verification_feedback_study_hotpotqa_50_v3`、`v3_run2`、`v3_run3`。
- `verification_feedback` 三次 F1 均值为 `23.51 ± 1.43`，拒答率为 `12.67 ± 0.94`。
- `targeted_feedback` 三次 F1 均值为 `20.87 ± 0.88`，稳定但低于 claim-concat 反馈。
- 该部分已经达到论文可写程度。

推荐命令：

```bash
.venv/bin/python experiments/run_verification_feedback_study.py \
  --config experiments/configs/verification_feedback_study.json \
  --samples 50 \
  --real-cove \
  --output-name verification_feedback_study_hotpotqa_50_v3_run2.json
```

如果成本允许，再跑 run3：

```bash
.venv/bin/python experiments/run_verification_feedback_study.py \
  --config experiments/configs/verification_feedback_study.json \
  --samples 50 \
  --real-cove \
  --output-name verification_feedback_study_hotpotqa_50_v3_run3.json
```

论文写法：

- 可以写成“反馈闭环稳定降低拒答率并提升 F1，但仍未解决短答案生成瓶颈”。
- 不建议再继续扩同类反馈实验；边际收益不高。

### 7.2 短答案抽取/答案格式约束实验

任务：

- 针对 Route A 误差分析中 74% 冗长推理式输出的问题，增加短答案抽取或更严格 answer-only 生成。
- 目标是提高 EM/F1，而不是继续降低拒答率。
- 这是论文定稿前最值得补的一组实验。

当前状态：

- 已完成 Route A strict-short 50 样本：`F1 = 2.66`，`EM = 0.0`。
- 已完成 verification-feedback strict-short 50 样本：`F1 = 2.01`，`EM = 0.0`，`No_Answer_Rate = 16.0%`。
- 结果表明单纯 prompt 压缩会显著退化，论文应将其写成负结果。

建议实验名：

```text
route_a_hotpotqa_realapi_50_short_answer.json
verification_feedback_study_hotpotqa_50_short_answer.json
```

服务器命令：

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa_short_answer.json \
  --samples 50 \
  --output-name route_a_hotpotqa_realapi_50_short_answer.json
```

```bash
.venv/bin/python experiments/run_verification_feedback_study.py \
  --config experiments/configs/verification_feedback_short_answer.json \
  --samples 50 \
  --real-cove \
  --output-name verification_feedback_study_hotpotqa_50_short_answer.json
```

预期判断：

- 若短答案约束能显著提升 Route A F1，则论文中应把“生成格式控制”列为主要瓶颈和后续改进方向。
- 若短答案约束不提升，则需要回到检索证据链质量和 answer extraction 逻辑。

最低完成标准：

- 已达成：50 样本 Route A short-answer 对照。
- 已达成：50 样本 verification-feedback short-answer 对照。
- 已达成：短答案约束前后 F1/EM 对比图。

后续不建议：

- 不建议继续只调 prompt。
- 不建议扩到 100 样本，因为方向已经明确退化。

后续建议：

- 若还要提升 F1，新增候选答案抽取器：从 top-k 证据句中抽实体/日期/数值候选，再用问题类型和验证分数排序。
- 或实现 span reranker：输入问题、候选 span 和证据句，输出最可能短答案。

### 7.3 真实表格/图谱数据路线

任务：

- 补足“伪异构数据外部效度不足”的短板。
- 先做数据接入设计和小样本 smoke，不要把它和当前 HotpotQA 主线混成一个大实验。

推荐路线：

- 表格：优先用 `HybridQA` 或 `OTT-QA` 的 text+table 样本，先做 20--50 条 smoke。
- 图谱：优先用本地 JSONL 三元组文件模拟 graph evidence，不建议论文最后阶段强依赖 Neo4j 服务。
- Neo4j：可以作为工程展示或附录方案，但正式实验建议导出为 `subject, relation, object, source_title` 的静态 JSONL，保证可复现。

建议文件结构：

```text
data/datasets/hybridqa/validation.json
data/datasets/graph_triples/hotpotqa_wikidata_like.jsonl
experiments/presets/route_a_hybridqa.json
experiments/presets/route_a_graph_triples.json
```

最低完成标准：

- 表格或图谱二选一先跑通 20 条。
- 只报告检索覆盖率、回答 F1 和案例，不追求大样本。
- 论文写作中明确这是“真实异构接入的可行性补充”，不是替代当前 HotpotQA 主结果。

当前行文要求：

- 前序异构实验必须继续表述为“伴生式序列化异构证据”，不能写成真实表格库/真实 Neo4j 图谱实验。
- B/C/D 的退化结论只说明当前序列化接入方案存在格式噪声，不外推为“异构数据无效”。
- HybridQA/OTT-QA/Neo4j 应写作后续真实异构验证路线，而不是已经完成的主实验。

### 8. 权衡曲线与校准图生成

任务：

- 将当前 Route A 与旧消融 smoke 结果整理成论文图
- 若第 7 步结果已存在，自动加入 soft/feedback 变体并生成验证校准图

命令：

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python experiments/plot_tradeoff_calibration.py
```

预期产物：

- `paper/zjuthesis/figures/tradeoff_f1_rejection.png`
- `paper/zjuthesis/figures/tradeoff_f1_latency.png`
- `paper/zjuthesis/figures/verifier_calibration.png`，仅当反馈实验报告含逐样本置信度时生成

## P1 强烈建议

### 9. 真实 CoVe 验证器对比实验

任务：

- 比较 `cove_soft / cove_standard / cove_strict / overlap_soft`
- 观察真实 verifier 条件下 FRR 与 unsafe accept 的变化
- 这一步继续保留旧实验壳，但定位为“诊断线”

命令：

```bash
.venv/bin/python experiments/run_verifier_comparison.py \
  --config experiments/configs/verifier_comparison.json \
  --samples 100 \
  --real-cove \
  --output-name verifier_comparison_real_cove_100.json
```

预期产物：

- `data/results/verifier_comparison_real_cove_100.json`

用于论文：

- 更新 verifier 诊断表
- 讨论类 CoVe 与真实 CoVe 的行为差异

### 10. 真实 CoVe 错误拒答诊断

任务：

- 在真实 verifier 条件下诊断 false rejection 来源
- 为 4.2 节及相关诊断表提供补充证据

命令：

```bash
.venv/bin/python experiments/run_false_rejection_diagnostics.py \
  --config experiments/configs/false_rejection_diagnostics.json \
  --samples 100 \
  --real-cove \
  --output-name false_rejection_diagnostics_real_cove_100.json
```

如果前面很稳定，可扩到 200：

```bash
.venv/bin/python experiments/run_false_rejection_diagnostics.py \
  --config experiments/configs/false_rejection_diagnostics.json \
  --samples 200 \
  --real-cove \
  --output-name false_rejection_diagnostics_real_cove_200.json
```

预期产物：

- `data/results/false_rejection_diagnostics_real_cove_100.json`
- 或 `data/results/false_rejection_diagnostics_real_cove_200.json`

用于论文：

- 补真实 verifier 下的拒答率、FRR、不安全接受率

### 11. 拉回结果并重画图

任务：

- 把服务器结果同步回本地
- 先分析再决定哪些进入正文主线

拉回结果示例：

```bash
scp root@<server-ip>:/root/home/ZJU-RAG-Thesis-Turing/data/results/*route_a*json /Users/eclipse/code/RAG/Rererank_v1/data/results/
scp root@<server-ip>:/root/home/ZJU-RAG-Thesis-Turing/data/results/*real_cove*json /Users/eclipse/code/RAG/Rererank_v1/data/results/
```

重画现有旧图命令：

```bash
python3 experiments/plot_results.py
```

说明：

- Route A 图表建议单独新增，不直接覆盖旧图
- 旧图仍主要对应历史消融主线

## P2 可选补充

### 12. 旧消融壳的完整真实 API 小样本实验

任务：

- 把生成端和 verifier 都维持真实 API
- 作为“旧实验壳在真实条件下”的补充实验

推荐先做 50 样本：

```bash
.venv/bin/python experiments/run_all.py \
  --config experiments/configs/ablation_with_controls.json \
  --samples 50 \
  --real-cove \
  --output-name automated_ablation_real_llm_50.json
```

说明：

- 只有当 Route A 100 样本已经稳定后，才建议做这一步
- 如果 `generator_api_key` 未配置，这条命令仍会回退到启发式生成

### 13. provider 对照实验

任务：

- 比较不同真实生成器对整体结果的影响
- 例如 DeepSeek 生成与其他 provider 的差异

说明：

- 当前项目尚未把“多 provider 对照”单独整理成公共配置模板
- 建议只做 50 样本以内的小规模实验

## 暂不建议做

### 14. 不建议直接跑全量 7405 的真实 API 实验

原因：

- RPM 和 TPD 风险高
- 成本不可控
- 对本科论文的边际收益有限

### 15. 不建议在本阶段切换到全新真实异构数据集

原因：

- Route A 的第一优先级仍是可信文本 baseline
- 需要先确认新 baseline 和真实生成链已经稳定

## 结果回填任务

### 16. 论文回填

任务：

- Route A 结果优先作为后续主线候选
- 旧消融壳结果优先写进诊断段落
- 保留类 CoVe 历史结果，但不要继续把它包装成最终 baseline

涉及文件：

- `paper/zjuthesis/body/undergraduate/final/2-body.tex`
- `paper/zjuthesis/body/undergraduate/final/abstract.tex`
- `paper/zjuthesis/body/undergraduate/final/5-conclusion.tex`

### 17. 文档轻微更新

任务：

- 在完成服务器实验后，同步更新待办与总指南

涉及文件：

- `OVERALL_TODO.md`
- `docs/EXPERIMENT_MASTER_GUIDE.md`
- `docs/GPU_SERVER_RUNBOOK.md`

## 建议执行顺序

1. 准备 GPU 服务器环境
2. 跑 Route A heuristic smoke
3. 跑 Route A 真实生成 smoke
4. 跑 Route A 100 样本 baseline
5. 跑旧 `A_Baseline` 20 样本 smoke
6. 跑旧 `A3_Baseline_CoVe / D_CoVe_Full` 20 样本 smoke
7. 已完成：跑 `run_verification_feedback_study.py` 的 50 样本真实 CoVe 闭环实验
8. 已完成：跑 `run_verifier_comparison.py` 的 200 样本真实 CoVe 阈值对比
9. 可选：跑 `run_false_rejection_diagnostics.py` 的 100 样本真实 CoVe，若论文篇幅足够再补
10. 拉回结果
11. 运行 `plot_tradeoff_calibration.py` 重画权衡曲线与校准图
12. 回填论文
13. 更新 TODO 和 guide

## 评阅注释后的补充实验计划

本节对应 2026-05-07 评阅意见中的“部分样本量偏小”问题。原则上不建议把所有历史实验都盲目扩到大样本，而应优先扩展直接支撑核心结论的配置。

### P0：真实 LLM 核心消融扩样本

目标：

- 将当前 N=100 的真实 LLM A/A2/A3/B/C/D 复核扩展到 N=300。
- 重点验证三个结论是否稳定：Adaptive 主要作用于召回、CoVe hard reject 触发高拒答、伴生式异构序列化仍引入格式噪声。

命令：

```bash
.venv/bin/python experiments/run_all.py \
  --config experiments/configs/ablation_with_controls.json \
  --samples 300 \
  --real-cove \
  --only-configs A_Baseline A2_Baseline_Adaptive A3_Baseline_CoVe B_Hetero C_Hetero_Adaptive D_CoVe_Full \
  --checkpoint-every 5 \
  --output-name automated_ablation_real_llm_300.json
```

### P0：VAR 真实 CoVe 扩样本

目标：

- 将当前 N=100 的 VAR 矩阵扩展到 N=300。
- 检查 hard reject、soft accept、VAR、Targeted VAR 的 F1、拒答率与覆盖率排序是否稳定。

命令：

```bash
.venv/bin/python experiments/run_verification_feedback_study.py \
  --config experiments/configs/verification_feedback_study.json \
  --samples 300 \
  --real-cove \
  --output-name verification_feedback_study_hotpotqa_300_real_cove.json
```

### P1：verifier 阈值对比扩样本

目标：

- 将当前 N=200 的 verifier threshold study 扩展到 N=500。
- 用更稳定样本支持“阈值越严格，不安全接受率越低，但错误拒答和 F1 损失越高”的安全-可用性权衡。

命令：

```bash
.venv/bin/python experiments/run_verifier_comparison.py \
  --config experiments/configs/verifier_comparison.json \
  --samples 500 \
  --real-cove \
  --output-name verifier_comparison_real_cove_500.json
```

### P1：Route A 文本基线扩样本

目标：

- 将 Route A baseline 从 N=100 扩展到 N=300。
- 检查“检索已较稳定，但答案表达层限制 EM/F1”的结论是否稳定。

命令：

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 300 \
  --output-name route_a_hotpotqa_realapi_300.json
```

### P2：真实表格与图谱数据 smoke

目标：

- 不在最终提交前强行替换主线结论，而是准备后续扩展路线。
- 表格侧优先使用 `experiments/presets/route_a_hybridqa.json` 做 text-table smoke。
- 图谱侧建议先把 Wikidata/Neo4j 导出为静态 JSONL 三元组，再通过 `EvidenceUnit` 接口接入，避免把在线图查询、实体链接和检索机制混在一起。

建议命令：

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hybridqa.json \
  --samples 50 \
  --output-name route_a_hybridqa_text_table_smoke_50.json
```

当前论文写法注意：

- 已有异构实验只能表述为“伴生式序列化异构证据的诊断结果”。
- 不能写成“真实多源异构知识融合整体无效”。
- 若后续补 HybridQA 或 Neo4j/Wikidata，只作为独立扩展实验，不覆盖当前 HotpotQA 主线。
