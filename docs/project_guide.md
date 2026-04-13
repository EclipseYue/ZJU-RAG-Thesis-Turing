# 🚀 RAG 系统四阶段工程：全链路操作与实验指挥白皮书

本指南是本项目的**最高级别架构说明与操作手册**。它专为人类研究员（实验指挥官）设计，不仅深度解析了系统的数据来源、各模块功能及核心创新点，更是一份**指挥 AI Agent（如 Trae 等）完成后续“真实数据实验 $\rightarrow$ 结果可视化 $\rightarrow$ 论文更新”全流程的 Prompt 字典。**

---

## 1. 项目数据来源与规模 (Data Sources & Scale)

本系统的实验数据分为两类，分别用于快速原型验证与大规模真实评测：

### 1.1 内部原型异构数据集 (Synthetic Data)
- **位置**：`src/rererank_v1/hetero_data.py`
- **规模**：数十条手工构造的关联数据。
- **作用**：包含纯文本（Text）、表格（Table 行序列化）、知识图谱（Graph 三元组序列化），用于在 CPU/Mock 模式下秒级跑通代码数据流，验证代码的跨模态融合逻辑是否无误。

### 1.2 真实开源多跳基准测试集 (Real Benchmark Data)
- **位置**：`src/rererank_v1/dataset_loader.py`
- **来源**：HuggingFace `datasets` 库中的经典多跳问答数据集（如 **HotpotQA**, **2WikiMultihopQA**）。
- **规模**：包含数万到数十万条需要多跳推理的复杂 QA 对及相应的文档上下文。
- **实验抽取**：在我们的实验脚本（如 `run_all.py`）中，通常会采样 **100~500 条验证集数据** 进行消融实验的批量运行，以确保统计结果的显著性，同时将 API 成本控制在合理范围内。

---

## 2. 系统四大模块解析与创新点分布 (Architecture & Innovations)

本项目针对传统 RAG 在面对复杂问题时“搜不准、搜不全、易幻觉”的痛点，提出了四个阶段的创新解决方案：

| 模块 (Phase) | 代码位置 | 核心功能 | 🌟 论文创新点 (Innovation) |
| :--- | :--- | :--- | :--- |
| **Phase 1: Adaptive Retrieval (自适应检索)** | `rag_pipeline.py` | 拦截重排序低置信度的结果，使用 LLM 提取关键词进行 Pseudo-Relevance Feedback (PRF) 并执行二次扩充检索。 | 提出**基于置信度门控的主动检索**。打破了一次性检索在多跳问题上的局限，将 Token 成本、延迟与多跳成功率纳入帕累托最优权衡。 |
| **Phase 2: Heterogeneous Knowledge (异构知识融合)** | `hetero_data.py` & `rag_pipeline.py` | 统一处理文本段落、结构化表格（行序列化）和知识图谱（三元组），混合存入向量空间。 | 设计了统一的 **`EvidenceUnit` 抽象**，并在 Reranker 阶段显式注入模态特征（如 `[source=table]`），使交叉编码器能精准给结构化信息打分。 |
| **Phase 3: Evidence Chain (证据链构造)** | `evidence_chain.py` | 将检索到的独立文档碎片，基于共现关系和得分连接成树状/图状的上下文传递给大模型。 | **Graph-of-Thoughts 检索版**。相比于传统的扁平化文档列表拼接，层级化的证据链大幅降低了 LLM 生成时在长上下文中的“迷失”概率。 |
| **Phase 4: Chain-of-Verification (自我验证)** | `cove_verifier.py` | 在生成答案后，强制抽取原子声明 (Claims)，并与底层的 Evidence Chain 进行事实交叉比对。 | **No-Answer 安全兜底机制**。在遇到知识盲区或诱导性谬误时，精准切断幻觉生成，触发“拒答”，极大提升系统的工业级安全性。 |

---

## 3. 实验全流程与自动化管线 (Experimental Workflow)

项目的所有实验入口均收敛于 `experiments/` 目录下。

### 3.1 实验脚本清单
- `phase2_experiment.py`：单跑异构知识对比（文本 vs 混合），验证召回效率。
- `phase3_experiment.py`：单跑 HotpotQA，观察打印出的 Evidence Chain 结构。
- `phase4_experiment.py`：跑带有幻觉触发器的验证集，观察 CoVe 的拦截表现。
- **`run_all.py`**：**核心主控脚本**。一键串行运行 A (Baseline)、B (+Hetero)、C (+Adaptive)、D (+CoVe Full) 四个消融变体。

### 3.2 结果指标含义解读 (Metrics Interpretation)
当 `run_all.py` 跑完后，会在 `data/results/automated_ablation.json` 生成实验矩阵，包含以下核心指标：
- **Avg_Tokens (Token 成本)**：反映了引入自适应检索（Adaptive）后增加的 API 成本，用于论证“以算力换取多跳准确率”的代价。
- **Avg_Latency_ms (延迟)**：反映了异构检索（Hetero）的巨大优势。直接命中表格/图谱相比于盲目多轮纯文本检索，延迟会断崖式下降。
- **No_Answer_Rate (拒答率)**：核心安全指标。反映 CoVe 机制在遇到系统外知识时，成功抑制幻觉并回答“不知道”的比例（越高代表系统越安全）。

---

## 4. 实验数据可视化与工具链 (Visualization Tools)

### 4.1 云端看板：Weights & Biases (WandB)
- **优势**：工业界最主流的实验追踪平台，实时生成精美的折线图与对比柱状图，论文直接截图可用。
- **操作**：终端执行 `wandb login` 后，让 Agent 在运行 `experiments/run_all.py` 时开启 `use_wandb=True`。

### 4.2 开源替代方案：MLflow / TensorBoard
- 如果不方便使用外网云服务，本项目架构亦兼容本地开源看板。你可以指挥 Agent 引入 `mlflow`，它会在本地起一个 `localhost:5000` 的网页，记录每一轮消融实验的超参数与 Metrics。

### 4.3 本地脚本直绘 (Matplotlib / Seaborn)
- **最推荐的论文绘图方式**：让 Agent 读取 `automated_ablation.json`，直接使用 Python 画出符合 IEEE / SCI 论文标准的矢量图（PDF/SVG）。

---

## 5. 终极指挥官 Prompt 字典 (Agent Command Templates)

作为人类实验指挥官，你不需要亲自写样板代码。请直接复制以下 Prompt 丢给对话框，Agent 会调用相应的 Skill 自动完成所有脏活累活。

### 🛠️ 任务 A：开始全量真实数据跑批
> **Prompt**: "Agent，请调用 `experiment-runner` 技能。修改 `experiments/run_all.py` 以加载 100 条真实的 HotpotQA 数据（不要用 mock），将 `use_wandb` 设为 `False`，在后台静默运行这四个消融模型。跑完后把结果保存在 `data/results/automated_ablation.json` 中并向我汇报核心结论。"

### 📊 任务 B：数据可视化与本地画图
> **Prompt**: "Agent，我们刚刚跑完了最新的 ablation json 数据。请编写一个新脚本 `experiments/plot_results.py`，使用 `matplotlib` 和 `seaborn` 读取 `data/results/automated_ablation.json`。我需要两张图：
> 1. 一个双轴柱状图（X轴为四个模型，左Y轴为 Token 成本，右Y轴为 Latency 延迟）。
> 2. 一个折线图展示 No-Answer Rate 的攀升。
> 请使用学术论文风格（高分辨率，清晰的图例），并将图片保存在 `docs/images/` 目录下。"

### 📝 任务 C：论文内容自动更新 (LaTeX 联动)
> **Prompt**: "Agent，请调用 `thesis-maintainer` 技能。我们刚画出了最新的消融实验对比图（在 `docs/images/`），且 json 数据也有了更新。
> 1. 请把最新的表格数据硬编码更新到 `paper/zjuthesis/body/undergraduate/final/2-body.tex` 的消融矩阵中。
> 2. 把我们刚画的柱状图 `\includegraphics` 插入到论文里。
> 3. 结合新图表，重写一小段对实验结果的深度分析。
> 4. 进入 `paper/zjuthesis` 执行 `rm -rf out/ && make`，确保编译成功后打开 PDF 给我看。"

### 🚀 任务 D：开发下一个创新模块 (Graph-RAG 升级)
> **Prompt**: "Agent，请查阅 `docs/project_guide.md` 的 Phase 3 章节。目前的 `evidence_chain.py` 还是线性拼接。我希望你重构这个文件，引入 `networkx` 库，基于实体共现矩阵（Entity Co-occurrence）构建一个真正的 Graph，并用 PageRank 算法抽出得分最高的 Top-3 推理路径。重构完成后，写一个小脚本测试一下并告诉我效果。"