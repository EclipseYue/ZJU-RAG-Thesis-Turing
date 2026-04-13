# RAG 系统多阶段实验：总览与执行规划 (Experiment Overview)

## 1. 核心实验规划与当前进度

目前，系统架构已完成代码工程化，并在小规模样本上跑通了数据流。接下来需要进入**全量测试**和**系统性评估**阶段。

| 实验编号 | 验证目标 | 对应代码 | 当前进度 | 待办事项 (Next Steps) |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1: 异构知识融合 (Phase 2)** | 验证表格/知识图谱统一建模后的跨模态检索优势，对比纯文本在延迟和召回率上的差距。 | `experiments/phase2_experiment.py` | 🟢 **已完成 (小样本)** | 1. 在 HotpotQA/2WikiMultihopQA 全量测试集上跑通。<br>2. 补充召回率指标 (Recall@K, MRR)。 |
| **Exp 2: 证据链推理 (Phase 3)** | 验证 Graph of Thoughts 证据链能否在多跳查询中为 LLM 减负，对比扁平化 Prompt 的生成效果。 | `experiments/phase3_experiment.py` | 🟢 **已完成 (代码与小样本)** | 1. 接入真实的生成大模型（如 GPT-4 / Qwen）。<br>2. 计算生成回答的 F1 / EM (Exact Match) 指标。 |
| **Exp 3: 自适应扩展 (Phase 1)** | 定量分析 LLM-judge 按需检索（Adaptive Retrieval）带来的 Token/延迟开销与多跳推理成功率的权衡（Trade-off）。 | `experiments/phase4_experiment.py` (Ablation) | 🟢 **已完成 (消融验证)** | 1. 生成帕累托前沿曲线 (Pareto Frontier) 图表。 |
| **Exp 4: CoVe 防幻觉 (Phase 4)** | 验证 CoVe 自我验证在应对“知识盲区”和“诱导性幻觉”时的拒答率（No-Answer Rate）和安全兜底能力。 | `experiments/phase4_experiment.py` (Ablation) | 🟢 **已完成 (拦截率验证)** | 1. 大规模验证：构造500条幻觉触发器进行压力测试。 |

---

## 2. 自动化跑实验方案 (Automated Runner)

为了避免每次手动执行单一脚本，我们将建立一个主控脚本 `experiments/run_all.py`。
* **执行逻辑**：
  1. 读取统一的参数配置文件（如 `config.yaml` 或直接使用命令行 argparse）。
  2. 串行/并行调用 Phase 2 ~ Phase 4 的测试用例。
  3. 将所有日志、中间链条（Chain）、结果指标统一保存至 `data/results/` 和 `data/research_history.json`。
* **测试用例扩充**：利用 `datasets` 从 HuggingFace 加载 100~500 条标准的验证集数据进行批量评估。

---

## 3. 实验可视化网站跟进 (Experiment Visualization)

在进行全量数据测试时，单纯看控制台输出难以直观分析指标波动。我们计划引入 **Weights & Biases (WandB)** 或 **MLflow** 作为实验看板：
* **核心监控指标**：
  * 💰 **Cost Tracking**: `total_tokens`, `avg_tokens_per_query`
  * ⏱️ **Latency**: `avg_latency_ms`, `reranker_calls`
  * 🛡️ **Safety**: `no_answer_rate` (CoVe 拒答率)
  * 🎯 **Accuracy**: `exact_match`, `f1_score` (后续引入)
* **操作流程**：
  1. `pip install wandb`
  2. 在 `run_all.py` 中初始化 `wandb.init(project="rag-rererank-study")`。
  3. 将每一轮的 Ablation Config（模型变体A/B/C/D）作为一个 Run 记录到云端看板。
  4. 自动生成对比折线图、柱状图，可直接截图贴入最终论文。

## 4. 结果落盘结构

实验产生的所有物料均严格按照 `experiment-runner` Skill 要求落盘：
```text
Rererank_v1/
├── data/
│   ├── results/
│   │   ├── phase2_results.json    # 异构检索对比结果
│   │   ├── phase3_results.json    # 证据链生成结果
│   │   └── ablation_matrix.json   # Phase 4 最终消融对比大表
│   └── research_history.json      # 自动化执行的全局历史追踪
└── experiments/
    └── run_all.py                 # (新增) 自动化集成主控脚本
```