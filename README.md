# RAG Pipeline with Reranking and Pseudo-Relevance Feedback (PRF)

This project implements an advanced Retrieval-Augmented Generation (RAG) pipeline in Python. It features a two-stage retrieval process (Vector Search + Cross-Encoder Reranking), enhanced by Conditional Pseudo-Relevance Feedback (PRF) and Reciprocal Rank Fusion (RRF).

## 🚀 Key Features & Innovation Points (基于开题报告)

本项目的研究目标是构建一个面向**多源异构知识（Heterogeneous Knowledge）**的通用 RAG 框架，解决异构融合困难、复杂跨文档推理以及推理过程缺乏可信验证的问题。核心创新点包括：

1. **统一证据单元入口**：建立涵盖文本、表格、知识图谱（三元组）的统一检索算子接口与可追溯的证据空间。
2. **成本约束下的自适应检索**：通过置信度与证据充分性触发按需检索（Active Retrieval），将 Token 消耗与延迟（p50/p95）作为硬约束指标。
3. **基于证据链的复杂推理与验证闭环**：基于结构化证据链（借鉴 TRACE 等思想）进行多跳推理，并在系统层面引入 CoVe 自检与事实性评估（FactScore）机制，保障系统的“无答案安全（No-Answer Safety）”。

### 当前系统能力
1.  **Vector Retrieval**: Uses `SentenceTransformers` (`all-MiniLM-L6-v2`) for efficient initial retrieval based on cosine similarity.
2.  **Reranking**: Integrates `CrossEncoder` (`BAAI/bge-reranker-base`) to accurately rescore and reorder the retrieved candidates.
3.  **Pseudo-Relevance Feedback (PRF)**:
    *   Automatically triggers when the top reranked document has a high confidence score (> 0.8).
    *   Extracts keywords from the top document to generate an expanded query.
    *   Performs a second retrieval with the expanded query.
4.  **Reciprocal Rank Fusion (RRF)**: Robustly combines results from the original query and the expanded query to produce the final ranking.
5.  **Robust Error Handling**: Includes a "Mock Mode" fallback mechanism that allows the pipeline to run (with simulated data) even if model download fails due to network issues.

## 🛠️ Installation

Ensure you have Python 3.8+ installed.

1.  Clone the repository (if applicable) or navigate to the project directory.
2.  Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Requirements:**
*   `sentence-transformers>=2.2.2`
*   `torch>=2.0.0`
*   `numpy>=1.24.0`
*   `scikit-learn>=1.3.0`

## 🏃 Usage

Run the main script to see the pipeline in action with sample data:

```bash
python rag_pipeline.py
```

## 📁 Project Layout

- `paper/`: LaTeX thesis sources (see `paper/README.md`)
- `src/`: Python package (`rererank_v1`)
- `experiments/`: experiment scripts (see `experiments/README.md`)
- `data/`: local artifacts (history/results) (see `data/README.md`)

### Custom Usage

You can easily integrate this pipeline into your own application:

```python
from rag_pipeline import RAGPipeline

# Initialize the pipeline
rag = RAGPipeline()

# Add your documents
documents = [
    "Doc 1 text...",
    "Doc 2 text...",
    # ...
]
rag.add_documents(documents)

# Search
results = rag.search("your query here", top_k=5)

# Print results
for res in results:
    print(f"Rank: {res.get('rank')}, Text: {res['text']}")
```

## 🧪 Feasible Experimental Schemes (下一阶段实验方案)

结合开题报告的“分阶段+组合验证”规划，下一步可开展以下 4 个阶段的递进实验，并在本项目的 `experiments/` 目录下落地：

### Phase 1: 纯文本多跳推理与自适应检索验证 (Baseline)
- **数据集**: HotpotQA / 2WikiMultiHopQA
- **实验设计**:
  - 引入复杂多跳查询测试用例（如包含两步以上的组合条件）。
  - 改进当前的 PRF 逻辑，引入基于 LLM 的 **Self-RAG/Active RAG** 判断机制（替代单一的置信度 > 0.8 阈值），以决定“是否需要二次检索”。
  - **核心指标**: 记录调用次数、端到端延迟（p50/p95）、Token 消耗，对比“静态一次性检索”与“自适应多轮检索”的效果-成本 Pareto 曲线。

### Phase 2: 多源异构知识的统一证据入口 (Heterogeneous)
- **数据集**: 补充 HybridQA 或 WikiTableQuestions (包含表格)，构建局部知识图谱。
- **实验设计**:
  - 在 `src/rererank_v1/` 中抽象出统一的 `EvidenceUnit` 类，兼容文本 Chunk、表格行（序列化）与 KG 三元组。
  - 引入混合检索（Elasticsearch 用于精确匹配，FAISS 用于语义检索，Neo4j 用于关系跳数遍历）。
  - **核心指标**: 在异构语料库下，重排序模型对跨模态（如自然语言问题 vs 表格列序列）证据的相关性评分区分度（Gap）、召回覆盖率。

### Phase 3: 证据链构造与复杂推理闭环 (Reasoning)
- **数据集**: 基于 Phase 2 的多源数据集。
- **实验设计**:
  - 实现基于图结构（Graph of Thoughts, GoT）的推理链控制器，将碎片化证据组织为推理子图。
  - 在生成阶段注入来源追踪（Source Citation）机制，确保输出可溯源到具体证据。
  - **核心指标**: 答案准确率（EM/F1）、事实性评分（如 FactScore 覆盖度）、推理链分支规模与逻辑连贯性。

### Phase 4: 验证机制与系统级消融分析 (Verification & Ablation)
- **实验设计**:
  - 实现 Chain-of-Verification (CoVe) 模块，对生成结论进行自检，拦截并修正高置信度的错误幻觉。
  - 设计包含强干扰项、知识缺失的鲁棒性测试集，验证“无答案（No-Answer）”场景下的拒答率与安全性。
  - **消融实验（Ablation Study）**: 对比 `纯文本` vs `异构 (+表格/图谱)` vs `异构 & 自适应`，明确单点机制带来的提升比例。

## ⚙️ Configuration

The `RAGPipeline` class accepts several parameters during initialization:

*   `embedding_model_name`: Hugging Face model ID for embeddings (default: `all-MiniLM-L6-v2`).
*   `reranker_model_name`: Hugging Face model ID for reranker (default: `BAAI/bge-reranker-base`).
*   `device`: Computation device (`'cpu'`, `'cuda'`, `'mps'`, or `None` for auto-detection).

## 🧩 Architecture Flow

1.  **Input Query** -> **Vector Search** -> *Initial Candidates*
2.  *Initial Candidates* -> **Reranker** -> *Scored Candidates*
3.  **Check Confidence**:
    *   **If High (> 0.8)**:
        *   Extract Keywords -> **New Query**
        *   **Vector Search (New Query)** -> **Reranker** -> *Secondary Candidates*
        *   *Scored Candidates* + *Secondary Candidates* -> **RRF Fusion** -> **Final Results**
    *   **If Low**:
        *   Return *Scored Candidates* directly.

## ⚠️ Network Issues / Mock Mode

If the script cannot connect to Hugging Face to download the models (e.g., due to SSL/connection errors), it will automatically switch to **Mock Mode**. In this mode, it uses random vectors and scores to demonstrate the *logic flow* without performing actual semantic retrieval.

To use real models in a restricted network environment, consider setting a mirror:
```bash
export HF_ENDPOINT=https://hf-mirror.com
python rag_pipeline.py
```
