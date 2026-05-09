# 实验总指南

本文件统一整合仓库中的实验入口、配置方式、离线数据说明、LLM/API 后端说明、结果落盘约定与论文更新流程。后续如需执行实验，请优先以本文档为准。

当前仓库已明确选择 **Route A**：

- 保留现有实验壳与论文联动机制
- 用成熟框架重建可信 baseline
- 将真实异构任务作为后续主线

路线总览见：
- [Route A 架构与迁移蓝图](/Users/eclipse/code/RAG/Rererank_v1/docs/ROUTE_A_ARCHITECTURE.md)
- [GPU 服务器迁移与真实 API 实验运行手册](/Users/eclipse/code/RAG/Rererank_v1/docs/GPU_SERVER_RUNBOOK.md)

## 1. 目标与范围

本项目当前的实验分为三层：

- 迁移层：清理旧原型、搭建新 baseline 接口、验证实验壳可复用性。
- 原型/诊断层：保留旧 HotpotQA / 2Wiki 消融与失败分析，作为历史阶段参考。
- 正式主线层：未来在成熟 baseline 与真实异构任务上重跑关键实验，并回填论文。

本文档只负责说明如何配置和执行，不替你自动跑实验。

## 2. 关键实验入口

说明：

- 现有入口大多仍服务于“旧 baseline + 诊断实验”。
- Route A 下，这些入口优先被视为 **实验壳**，不是最终 baseline 本身。

### 2.1 主消融入口

文件：`experiments/run_all.py`

用途：
- 跑主消融矩阵
- 支持 A / A2 / A3 / B / C / D 控制组
- 支持离线数据加载
- 支持启发式或真实 LLM 生成/验证后端切换

常用命令：

```bash
python experiments/run_all.py --config experiments/configs/ablation_with_controls.json
python experiments/run_all.py --dataset hotpotqa --samples 100 --device cuda --include-controls
python experiments/run_all.py --config experiments/configs/ablation_with_controls.json --real-cove
```

### 2.2 补充诊断入口

- `experiments/run_false_rejection_diagnostics.py`
  用于错误拒答来源诊断
- `experiments/run_bucket_gain_study.py`
  用于 bridge / comparison 分桶分析
- `experiments/run_verifier_comparison.py`
  用于验证器模式与阈值对比
- `experiments/run_supplementary_study.py`
  用于小规模补充对照

推荐命令：

```bash
python experiments/run_false_rejection_diagnostics.py --config experiments/configs/false_rejection_diagnostics.json
python experiments/run_false_rejection_diagnostics.py --config experiments/configs/false_rejection_diagnostics.json --real-cove
python experiments/run_verifier_comparison.py --config experiments/configs/verifier_comparison.json
python experiments/run_verifier_comparison.py --config experiments/configs/verifier_comparison.json --real-cove
```

### 2.3 Route A baseline 入口

文件：`experiments/run_route_a_baseline.py`

用途：
- 运行成熟框架底座的 Route A baseline
- 当前支持 `llamaindex_text`
- 默认读取 `experiments/presets/route_a_hotpotqa.json`

默认小样本试跑命令：

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 20 \
  --output-name route_a_hotpotqa_realapi_smoke.json
```

说明：

- 当前 `route_a_hotpotqa.json` 已显式声明 `generator_backend=deepseek`、`generator_model=deepseek-v4-flash`
- 若 `experiments/configs/local_api_overrides.json` 中也存在生成端配置，则会继续以该私有配置为准
- 只有在你显式传入 `--generator-backend heuristic` 时，才会改走启发式

如果只是排查检索链，不希望消耗真实 API：

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 20 \
  --generator-backend heuristic \
  --output-name route_a_hotpotqa_heuristic_smoke.json
```

若本地 PyPI 证书或代理不稳定，Route A 依赖建议安装到仓库 `.venv`：

```bash
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  -r requirements-route-a.txt
```

兼容普通环境的命令如下：

```bash
python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 20 \
  --output-name route_a_hotpotqa_realapi_smoke.json
```

迁移到 GPU 服务器后的推荐执行顺序，见：

- [GPU 服务器迁移与真实 API 实验运行手册](/Users/eclipse/code/RAG/Rererank_v1/docs/GPU_SERVER_RUNBOOK.md)

### 2.4 Verification-aware 反馈闭环入口

文件：`experiments/run_verification_feedback_study.py`

用途：
- 比较 `hard_reject`、`soft_accept`、`verification_feedback`、`targeted_feedback` 四类验证策略
- 检查 CoVe 式验证从“一票否决”改为“软置信度聚合”后，拒答率与 F1 的变化
- 检查验证失败后触发定向补检索是否能缓解 verification collapse

配置文件：

```text
experiments/configs/verification_feedback_study.json
```

服务器推荐命令：

```bash
.venv/bin/python experiments/run_verification_feedback_study.py \
  --config experiments/configs/verification_feedback_study.json \
  --samples 50 \
  --real-cove \
  --output-name verification_feedback_study_hotpotqa_50_v3.json
```

输出指标：

- `ExactMatch`
- `F1_Score`
- `SupportRecall@K`
- `SupportAllHit@K`
- `No_Answer_Rate_Percent`
- `Feedback_Rate_Percent`
- `Avg_Verify_Confidence`
- `Avg_Latency_ms`
- `Avg_Retrieval_Calls`

论文定位：

- `hard_reject` 对应旧式严格 CoVe 的崩溃风险
- `soft_accept` 对应软置信度判别
- `verification_feedback` 对应失败声明拼接式最小反馈闭环
- `targeted_feedback` 对应基于问题实体、已检索标题与失败声明的定向补检索闭环

当前结论：

- `verification_feedback_study_hotpotqa_50_v3.json` 中，claim-concat `verification_feedback` 是该批次最佳配置。
- 三次 repeated-run 显示，`verification_feedback` 的 F1 为 `23.51 ± 1.43`，拒答率为 `12.67 ± 0.94`，该结果已经足够进入论文。
- 2026-04-30 独立真实 CoVe 复核中，`targeted_feedback` 达到 `F1=25.87`、`No_Answer_Rate=16.0`，说明定向补检索并非无效，但稳定性仍需更多批次确认。
- 2026-04-30 的 100 样本真实 LLM 反馈复核中，`verification_feedback` 达到 `F1=18.80`、`No_Answer_Rate=18.0`，`targeted_feedback` 达到 `F1=18.60`、`No_Answer_Rate=15.0`；因此论文应把 feedback 表述为“缓解验证崩溃”，而不是“超过无验证纯生成配置”。
- 200 样本 verifier comparison 显示，阈值越严格，不安全接受率越低，但错误拒答率和 F1 损失越明显。
- 后续主线转为 short-answer generation / answer extraction。

推荐 repeated-run 命令：

```bash
.venv/bin/python experiments/run_verification_feedback_study.py \
  --config experiments/configs/verification_feedback_study.json \
  --samples 50 \
  --real-cove \
  --output-name verification_feedback_study_hotpotqa_50_v3_run2.json
```

当前 repeated-run 已完成；除非需要更多统计置信度，否则不建议继续重复同一实验。

### 2.5 真实 LLM 核心消融复核

目的：
- 检查旧启发式消融得到的模块趋势能否在真实 LLM 生成/验证后端下复现。
- 不追求全量 7405 样本，而是用 N=300 验证方向性结论。
- 支撑论文中的分层实验设计：大规模启发式诊断负责定位机制，真实 LLM 复核负责排除后端伪影。

已完成批次：

```text
data/results/automated_ablation_real_llm_300.json
```

关键结果：
- A Text：`F1=18.40`，`No_Answer_Rate=0.0`
- A2 Text+Adaptive：`F1=19.70`，`No_Answer_Rate=0.0`
- A3 Text+CoVe：`F1=9.61`，`No_Answer_Rate=57.0`
- B Hetero：`F1=16.51`，`SupportAllHit@K=47.0`
- D Full：`F1=9.56`，`No_Answer_Rate=61.33`

结论：
- Adaptive 在真实 LLM 条件下仍主要表现为小幅增益，不能写成强端到端优化器。
- CoVe hard reject 仍会造成高拒答，验证崩溃不是启发式后端独有现象。
- 当前伴生式异构序列化仍不应被写成真实异构知识收益，只能支持“格式噪声会影响检索/生成”的诊断结论。

绘图命令：

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python experiments/plot_real_llm_followup.py
```

### 2.6 评阅注释后的扩样本优先级

评阅意见指出部分补充实验样本量偏小。后续不建议把所有历史配置平均扩样本，而应优先扩展直接支撑论文核心结论的实验：

- P0：真实 LLM 核心消融从 N=100 扩展到 N=300，验证 Adaptive、CoVe 与异构序列化的关键趋势。
- P0：VAR 真实 CoVe 矩阵从 N=100 扩展到 N=300，验证 hard reject、soft accept、VAR 与 Targeted VAR 的排序稳定性。
- P1：真实 verifier 阈值对比从 N=200 扩展到 N=500，增强安全-可用性权衡结论的统计稳定性。
- P1：Route A 文本基线从 N=100 扩展到 N=300，检查“检索稳定但答案表达受限”的误差归因是否稳定。
- P2：HybridQA text-table 与 Neo4j/Wikidata JSONL graph 只作为独立 smoke，不覆盖当前 HotpotQA 主线。

当前执行状态：

- Route A N=300 已完成并回填论文：`F1=6.19`，`SupportRecall@K=72.83`，说明答案抽取瓶颈比 N=100 更明显。
- 真实 LLM 核心消融 N=300 已完成有效批次：A/A2/A3/B/D 均为 `MockMode=false`，其中 A2 相比 A 仅提升 `1.30` F1，A3/D 拒答率分别为 `57.0%` 和 `61.33%`。
- VAR 真实 CoVe N=300 已完成有效批次：VAR 将 hard reject 的 `39.67%` 拒答率降至 `14.33%`，F1 从 `12.48` 提升到 `18.15`。
- HybridQA text-table N=50 已完成并回填论文：`F1=2.51`，`SupportAllHit@K=0.00`，只作为真实异构接入边界实验。
- verifier threshold N=200 已更新：CoVe-strict 将不安全接受率降至 `13.0%`，但拒答率升至 `59.5%`。
- Neo4j/Wikidata graph smoke 已补本地 JSONL loader、preset 与 `data/datasets/wikidata_graph/validation.jsonl`；该文件由 2WikiMultihopQA 证据三元组转写为 Wikidata-style 静态图谱问答数据。

完整命令见：

- [当前实验待办清单](/Users/eclipse/code/RAG/Rererank_v1/EXPERIMENT_TODO_CURRENT.md)

输出：

```text
paper/zjuthesis/figures/real_llm_ablation_followup.png
paper/zjuthesis/figures/real_llm_feedback_followup.png
```

### 2.7 短答案约束实验

目的：
- 检查 Route A 与反馈闭环的剩余瓶颈是否来自冗长推理式输出。
- 检索与验证配置保持不变，只把生成端切到 `answer_mode=strict_short`。

Route A 短答案命令：

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa_short_answer.json \
  --samples 50 \
  --output-name route_a_hotpotqa_realapi_50_short_answer.json
```

Feedback 短答案命令：

```bash
.venv/bin/python experiments/run_verification_feedback_study.py \
  --config experiments/configs/verification_feedback_short_answer.json \
  --samples 50 \
  --real-cove \
  --output-name verification_feedback_study_hotpotqa_50_short_answer.json
```

判断：
- 若 F1/EM 明显提升，论文应把答案格式控制列为主要工程瓶颈。
- 若 F1/EM 不提升，说明剩余瓶颈更可能在证据链缺跳或验证器校准。

当前结果：
- Route A strict-short：`F1 = 2.66`，`EM = 0.0`。
- Verification Feedback strict-short：`F1 = 2.01`，`EM = 0.0`。
- 该结果说明单纯 prompt 压缩会退化，后续应实现候选答案抽取器或 span reranker。

### 2.8 真实异构数据补充路线

当前建议：
- 不在最后阶段把 Neo4j 服务作为正式实验硬依赖。
- 若需要图谱实验，优先把 Neo4j/Wikidata 结果导出为静态 JSONL 图谱问答文件，保证可复现。
- 若需要表格实验，优先选择 `HybridQA` 或 `OTT-QA` 小样本 smoke。

表格 smoke 命令：

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hybridqa.json \
  --samples 100 \
  --output-name route_a_hybridqa_text_table_100.json
```

图谱 smoke 数据格式：

```json
{"id":"graph_001","question":"...","answer":"...","triples":[{"head":"...","relation":"...","tail":"...","source":"wikidata","qid":"Q..."}],"supporting_triples":[0]}
```

图谱 smoke 命令：

```bash
ls data/datasets/wikidata_graph/validation.jsonl
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_wikidata_graph.json \
  --samples 50 \
  --output-name route_a_wikidata_graph_smoke_50.json
```

推荐定位：
- 作为“真实异构数据接入的可行性补充实验”。
- 不替代当前 HotpotQA + Route A + Verification Feedback 主线。

写作约束：
- 当前论文已完成的 B/C/D 异构实验是伴生式序列化异构证据，不是真实 HybridQA/Neo4j 实验。
- 这些结果只能支持“当前序列化接入方案存在格式噪声”，不能写成“真实异构知识无效”。
- 若后续加入 Neo4j，正式结果建议使用导出的静态 JSONL 图谱问答文件，避免在线服务影响可复现性。

### 2.9 权衡曲线与校准图入口

文件：`experiments/plot_tradeoff_calibration.py`

用途：
- 汇总当前 Route A 与旧消融 smoke 结果
- 绘制 `F1--拒答率` 与 `F1--延迟` 权衡图
- 若存在 `verification_feedback_study_hotpotqa*.json` 且包含逐样本置信度，自动绘制 verifier calibration 图

命令：

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python experiments/plot_tradeoff_calibration.py
```

输出：

```text
paper/zjuthesis/figures/tradeoff_f1_rejection.png
paper/zjuthesis/figures/tradeoff_f1_latency.png
paper/zjuthesis/figures/verifier_calibration.png
```

## 3. 数据与离线模式

### 3.1 默认数据目录

推荐把离线导出的数据集放到：

```text
data/datasets/hotpotqa/validation.json
data/datasets/2wikimultihopqa/validation.json
```

离线目录细节可参考：
- `experiments/configs/README_offline.md`

### 3.2 离线优先策略

当前加载器顺序：

1. 本地 `json/jsonl`
2. 本地 Hugging Face cache
3. 在线下载

如果配置了 `offline=true`，则只走前两步。

## 4. 生成与验证后端

### 4.1 当前默认行为

- 生成端默认 `generator_backend=auto`
- 验证端默认 `verifier_backend=heuristic`

这意味着：
- 生成端若检测到 OpenAI 兼容接口配置，会优先使用真实 API
- 若没有检测到可用 API，则回退到启发式生成
- 验证端默认仍然使用启发式 CoVe 近似，不会自动调用外部大模型

### 4.2 支持的后端

`run_all.py` 当前支持以下后端参数：

```text
--generator-backend auto|heuristic|openai|deepseek|moonshot|siliconflow
--verifier-backend heuristic|openai|deepseek|moonshot|siliconflow
--generator-model <model_name>
--verifier-model <model_name>
--real-cove
```

`--real-cove` 的约定是：

- 不改变实验矩阵本身，只改变 CoVe 相关配置实际使用的验证后端。
- 不传入时，默认沿用“类 CoVe / 启发式近似验证”。
- 传入后，优先使用 `verifier_backend` 和 `verifier_model`。
- 若传入 `--real-cove` 但 `verifier_backend` 仍为 `heuristic`，主入口会自动回退到 `deepseek` 作为默认真实 CoVe 后端。

### 4.3 环境变量

支持三类 OpenAI 兼容接口：

- OpenAI 风格：
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`
- DeepSeek：
  - `DEEPSEEK_API_KEY`
  - `DEEPSEEK_BASE_URL`，默认 `https://api.deepseek.com`
- Moonshot / Kimi（历史可选）：
  - `MOONSHOT_API_KEY` 或 `KIMI_API_KEY`
  - `MOONSHOT_BASE_URL`，默认 `https://api.moonshot.cn/v1`
- SiliconFlow：
  - `SILICONFLOW_API_KEY`
  - `SILICONFLOW_BASE_URL`，默认 `https://api.siliconflow.cn/v1`

若 `generator_backend=auto`，系统会优先尝试 `OPENAI_*`，再尝试 DeepSeek，再尝试 Moonshot，最后尝试 SiliconFlow。

### 4.4 本地私有覆盖配置

为避免把私钥写进公共配置文件，项目支持本地私有覆盖文件：

```text
experiments/configs/local_api_overrides.json
```

该文件会在以下脚本中被自动加载：

- `experiments/run_all.py`
- `experiments/run_false_rejection_diagnostics.py`
- `experiments/run_verifier_comparison.py`

推荐把真实 CoVe 所需的 `verifier_backend`、`verifier_model`、`verifier_api_key`、`verifier_base_url` 放在这里。

### 4.5 示例

使用 DeepSeek 作为真实验证后端，但不实际运行示例：

```bash
export DEEPSEEK_API_KEY="sk-..."
python experiments/run_all.py \
  --config experiments/configs/ablation_with_controls.json \
  --generator-backend deepseek \
  --generator-model deepseek-v4-flash \
  --verifier-backend deepseek \
  --verifier-model deepseek-v4-flash \
  --real-cove
```

## 4.6 迁移到 GPU 服务器后的建议

当前推荐做法不是“把所有实验都切成真实 API 全量跑”，而是：

- Route A baseline 作为正式主线
- 旧 `run_all.py` / `run_verifier_comparison.py` / `run_false_rejection_diagnostics.py` 作为小样本诊断线
- 真实 API 只在关键实验上开启
- 旧消融壳优先使用 `--only-configs` 与 `--checkpoint-every`

推荐优先顺序：

1. Route A heuristic smoke
2. Route A 真实生成 smoke
3. Route A 100 样本真实生成 baseline
4. 旧 `A_Baseline` 20 样本 smoke
5. 旧 `A3_Baseline_CoVe / D_CoVe_Full` 20 样本 smoke
6. verifier comparison / false rejection diagnostics 小样本真实 CoVe

详细命令和迁移步骤见：

- [GPU 服务器迁移与真实 API 实验运行手册](/Users/eclipse/code/RAG/Rererank_v1/docs/GPU_SERVER_RUNBOOK.md)

## 5. 配置文件

当前主配置位于 `experiments/configs/`：

- `ablation_with_controls.json`
- `false_rejection_diagnostics.json`
- `bucket_gain_study.json`
- `verifier_comparison.json`
- `verification_feedback_study.json`

Route A 的新预设位于 `experiments/presets/`：

- `route_a_hotpotqa.json`
- `route_a_hybridqa.json`

建议做法：
- 把数据路径、设备、离线模式写进配置文件
- 把是否使用真实后端放在命令行覆盖，避免误跑高成本实验
- 把私钥与网关地址放在 `local_api_overrides.json`，不要写进公共配置

## 6. 输出文件

主要结果统一落在：

```text
data/results/
```

常见文件包括：

- `automated_ablation_with_controls.json`
- `*_report_时间戳.json`
- `false_rejection_diagnostics_*.json`
- `bucket_gain_study_*.json`
- `verifier_comparison_*.json`

研究历史追踪：

```text
data/research_history.json
```

## 7. 论文联动

实验结果更新后，通常有三步：

1. 更新 `data/results/` 中的主结果 JSON
2. 运行 `experiments/plot_results.py` 重绘图表
3. 修改 `paper/zjuthesis/body/undergraduate/final/` 中的正文和摘要

论文主文件位置：

- `paper/zjuthesis/body/undergraduate/final/1-introduction.tex`
- `paper/zjuthesis/body/undergraduate/final/2-body.tex`
- `paper/zjuthesis/body/undergraduate/final/abstract.tex`
- `paper/zjuthesis/body/undergraduate/final/5-conclusion.tex`

编译命令：

```bash
cd paper/zjuthesis
latexmk -g
```

## 8. 当前边界

需要明确：

- 当前“异构数据”主要是伴生式规则构造，不是真实独立表格库/图谱库。
- 当前 CoVe 默认是启发式近似验证，不应在论文里无条件表述为“完整复现的真实 LLM CoVe”。
- 真实 LLM 生成/验证接口已经可配置，但是否启用是实验选择，不是默认行为。
- 现有 `run_all.py` 等入口适合做迁移前的诊断与小规模补充验证，但不应继续被视为最终可信 baseline 的唯一来源。

## 9. 推荐执行顺序

如果后续准备正式跑一轮新实验，建议顺序如下：

1. 先确认本地/远程离线数据在位
2. 先跑一轮类 CoVe 小样本试跑
3. 再用 `--real-cove` 跑同口径版本
4. 跑 `verification_feedback_study`，比较硬拒答、软判定与反馈补检索
5. 再跑错误拒答诊断和验证器对比
6. 视情况补跑 bridge / comparison 分桶
7. 运行 `plot_tradeoff_calibration.py` 重绘权衡曲线与校准图
8. 回填论文

## 10. 相关旧文档

以下文档已被本文档吸收，保留仅作历史参考：

- `experiments/README.md`
- `docs/experiment_overview.md`
- `docs/project_guide.md`
- `README.md` 中的实验部分
