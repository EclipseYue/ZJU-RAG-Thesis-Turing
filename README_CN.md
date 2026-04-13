# RAG Pipeline with Reranking and Pseudo-Relevance Feedback (PRF)

该项目使用 Python 实现了一个先进的检索增强生成（RAG）流程。它具有两阶段检索过程（向量搜索 + Cross-Encoder 重排序），并增强了条件伪相关反馈（PRF）和倒数排名融合（RRF）。

## 🚀 核心功能

1.  **向量检索**：使用 `SentenceTransformers` (`all-MiniLM-L6-v2`) 基于余弦相似度进行高效的初始检索。
2.  **重排序**：集成 `CrossEncoder` (`BAAI/bge-reranker-base`) 对检索到的候选文档进行精确的重新评分和排序。
3.  **伪相关反馈 (PRF)**：
    *   当排名靠前的文档具有高置信度分数 (> 0.8) 时自动触发。
    *   从排名第一的文档中提取关键词以生成扩展查询。
    *   使用扩展查询执行第二次检索。
4.  **倒数排名融合 (RRF)**：稳健地结合原始查询和扩展查询的结果以生成最终排名。
5.  **稳健的错误处理**：包含“模拟模式（Mock Mode）”回退机制，即使因网络问题导致模型下载失败，流程也能运行（使用模拟数据）。

## 🛠️ 安装

确保您已安装 Python 3.8+。

1.  克隆存储库（如果适用）或导航到项目目录。
2.  安装所需的依赖项：

```bash
pip install -r requirements.txt
```

**依赖项：**
*   `sentence-transformers>=2.2.2`
*   `torch>=2.0.0`
*   `numpy>=1.24.0`
*   `scikit-learn>=1.3.0`

## 🏃 使用方法

运行主脚本以查看使用示例数据的流程：

```bash
python rag_pipeline.py
```

### 自定义使用

您可以轻松地将此流程集成到您自己的应用程序中：

```python
from rag_pipeline import RAGPipeline

# 初始化流程
rag = RAGPipeline()

# 添加您的文档
documents = [
    "Doc 1 text...",
    "Doc 2 text...",
    # ...
]
rag.add_documents(documents)

# 搜索
results = rag.search("your query here", top_k=5)

# 打印结果
for res in results:
    print(f"Rank: {res.get('rank')}, Text: {res['text']}")
```

## ⚙️ 配置

`RAGPipeline` 类在初始化期间接受几个参数：

*   `embedding_model_name`：用于嵌入的 Hugging Face 模型 ID（默认值：`all-MiniLM-L6-v2`）。
*   `reranker_model_name`：用于重排序的 Hugging Face 模型 ID（默认值：`BAAI/bge-reranker-base`）。
*   `device`：计算设备（`'cpu'`、`'cuda'`、`'mps'` 或 `None` 用于自动检测）。

## 🧩 架构流程

1.  **输入查询** -> **向量搜索** -> *初始候选文档*
2.  *初始候选文档* -> **重排序** -> *评分后的候选文档*
3.  **检查置信度**：
    *   **如果高 (> 0.8)**：
        *   提取关键词 -> **新查询**
        *   **向量搜索（新查询）** -> **重排序** -> *二次候选文档*
        *   *评分后的候选文档* + *二次候选文档* -> **RRF 融合** -> **最终结果**
    *   **如果低**：
        *   直接返回 *评分后的候选文档*。

## 📊 运行结果分析

运行脚本后，您将看到两个典型测试案例的输出，展示了 Pipeline 如何处理不同类型的查询。

### 案例 1：相关性强的查询
**Query**: `"What is BGE Reranker?"`

1.  **初步检索**：成功检索到包含 "BGE Reranker" 的文档。
2.  **重排序**：Cross-Encoder 给出了极高的置信度分数（例如 `0.9933`），确认文档高度相关。
3.  **触发 PRF**：由于分数超过阈值（`0.8`），系统自动触发伪相关反馈。
    *   **提取关键词**：从文档中提取出 `bge`, `reranker`, `powerful` 等词。
    *   **生成新查询**：`"What is BGE Reranker? bge reranker powerful model text"`
4.  **最终融合**：通过 RRF 融合两次检索结果，**目标文档稳居第一**。

```text
Final Results for Query 1:
1. The BGE Reranker is a powerful model for text retrieval. (Score: 0.0328)
...
```

### 案例 2：低相关性/模糊查询
**Query**: `"food recipes"`（语料库中不存在相关文档）

1.  **初步检索**：检索到一些无关文档。
2.  **重排序**：Cross-Encoder 给出了很低的分数（例如 `0.1192`）。
3.  **跳过 PRF**：由于分数低于阈值，系统**拒绝**进行查询扩展，防止引入更多噪音。
4.  **直接返回**：返回初始排序结果，避免了错误的优化。

---

## ⚠️ 网络问题 / 模拟模式 (Mock Mode)

如果脚本无法连接到 Hugging Face 下载模型（例如，由于 SSL/连接错误），它将自动切换到 **模拟模式 (Mock Mode)**。在此模式下，它使用确定性的模拟向量和逻辑来演示完整的 RAG 流程。

### 强制使用模拟模式
如果您想在不下载模型的情况下快速测试流程逻辑，可以使用环境变量 `FORCE_MOCK=1`：

```bash
FORCE_MOCK=1 python rag_pipeline.py
```

要在受限网络环境中使用真实模型，请考虑设置镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python rag_pipeline.py
```
