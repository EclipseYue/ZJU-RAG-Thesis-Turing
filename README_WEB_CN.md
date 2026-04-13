# 📊 科研可视化看板 (Reranker Research Dashboard)

本项目包含一个轻量级的 Web 仪表盘，用于可视化您的科研进度、实验结果以及随时间推移的性能趋势。

## 🌟 核心功能

*   **实时指标 (Real-time Metrics)**：展示基准测试中最新的 MRR 和 NDCG@5 分数。
*   **趋势分析 (Trend Analysis)**：交互式折线图，直观展示不同迭代版本中 "Vector Only"（基线）与 "Full Pipeline"（改进模型）的性能差异。
*   **科研计划 (Research Plan)**：追踪科研阶段的状态（Planned / In Progress / Completed）。
*   **迭代历史 (Iteration History)**：详细记录每次基准运行的时间戳和性能提升百分比。

## 🚀 快速开始

### 1. 运行基准测试 (Run Benchmarks)
首先，运行基准测试脚本生成数据。脚本已集成自动保存功能，结果将写入 `research_history.json`。

```bash
python reranker_study.py
```

### 2. 启动看板 (Start Dashboard)
启动 Flask Web 应用程序：

```bash
python app.py
```

### 3. 浏览器访问 (View in Browser)
打开浏览器并访问：
**http://localhost:5001**

## 🔄 科研工作流

1.  **修改代码**：更新 `rag_pipeline.py` 或 `benchmark_data.py` 以测试新的想法（如处理多跳查询或否定句）。
2.  **运行实验**：执行 `python reranker_study.py`。
3.  **可视化验证**：刷新网页，查看新的数据点和趋势图变化。

## 📁 文件结构

*   `app.py`: Flask 后端服务。
*   `templates/dashboard.html`: 前端 HTML/JS 模板（使用 Chart.js）。
*   `research_history.json`: 基准测试指标的持久化存储（自动生成）。
