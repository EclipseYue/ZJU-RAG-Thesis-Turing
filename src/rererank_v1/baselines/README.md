# Baselines

本目录用于存放 Route A 下的可信 baseline 实现。

优先级：

1. `llamaindex_text.py`
   HotpotQA / 2Wiki 纯文本 baseline
2. `llamaindex_hybridqa.py`
   HybridQA / OTT-QA 真实 text+table baseline

当前已经提供：

- `llamaindex_text.py`
  Route A Phase A1 的文本 baseline 包装器，采用延迟导入，未安装 LlamaIndex 时不会影响仓库其他代码。
- `llamaindex_hybridqa.py`
  Route A Phase A3 的真实异构任务占位入口。

后续新增 baseline 优先放在这里，而不是继续直接扩写旧 `rag_pipeline.py`。
