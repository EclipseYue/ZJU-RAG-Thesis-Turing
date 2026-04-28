# Route A 架构与迁移蓝图

本文档给出当前仓库从“旧版自研 baseline + 伪异构数据”迁移到 **Route A** 的具体方案。

## 1. 为什么选择 Route A

当前系统最强的部分在于：

- 主消融与诊断实验组织完整
- 论文联动、结果回填、图表生成链路成熟
- Adaptive PRF、CoVe、evidence chain 这些模块具备研究增量潜力

当前系统最弱的部分在于：

- baseline 可信度不足，难以判断低分究竟来自方法本身还是底层实现问题
- 伪异构数据更像“格式噪声注入”，而不是真实异构知识融合
- 真实 LLM 生成还在为 QA 答案归一化付出额外代价

因此 Route A 的原则是：

**保留上层研究壳，替换下层 baseline 和异构知识源。**

## 2. 保留、替换、归档

### 2.1 保留的资产

- `experiments/run_all.py`
- `experiments/run_verifier_comparison.py`
- `experiments/run_false_rejection_diagnostics.py`
- `experiments/plot_results.py`
- `experiments/configs/*.json`
- `src/rererank_v1/cove_verifier.py`
- `src/rererank_v1/llm_generator.py`
- `src/rererank_v1/llm_backends.py`
- `src/rererank_v1/evidence_chain.py`
- 论文正文、图表和结果回填机制

### 2.2 替换的资产

- 旧版自研 baseline 检索主线
- 当前 `dataset_loader.py` 中的伪异构构造默认路线
- 以“统一序列化伪 table / 伪 graph”为主的异构实验论证

### 2.3 归档/清理的资产

- 已清理：旧科研可视化 Flask 看板入口
- 历史文档中与看板强绑定的内容保留在 `docs/research_docs/`，仅作历史记录，不再作为当前路线说明

## 3. 新架构分层

### 3.1 Baseline 层

目标：先得到可信、稳定、可解释的文本 RAG baseline。

建议底座：

- `LlamaIndex` 作为首选
- `LangChain` 可作为辅助编排，不建议作为唯一 baseline 标准

首批任务：

- HotpotQA 纯文本 baseline
- 2WikiMultihopQA 纯文本 baseline
- 结构化短答案输出，避免自由文本拉低 EM/F1

### 3.2 Plugin 层

目标：将你的创新模块变成可插拔增量，而不是与旧 baseline 绑死。

计划迁移的模块：

- `Adaptive PRF`
- `CoVe / verifier`
- `Evidence Chain`
- 成本、延迟、安全性统计

建议定义统一接口：

- `RetrieverAdapter`
- `GeneratorAdapter`
- `VerifierAdapter`
- `ExperimentPreset`

### 3.3 Heterogeneous 层

目标：让“异构”变成真实异构，而不是文本模板变体。

优先顺序：

1. `HybridQA`
   真实 `text + table`
2. `OTT-QA`
   开放域 `text + table`
3. `Wikipedia + Wikidata/Neo4j`
   作为更进一步的图结构异构扩展

## 4. 目录改造建议

建议新增并逐步填充以下结构：

```text
src/rererank_v1/baselines/
  llamaindex_text.py
  llamaindex_hybridqa.py

src/rererank_v1/adapters/
  retriever.py
  generator.py
  verifier.py

src/rererank_v1/modules/
  adaptive_prf.py
  cove_plugin.py
  evidence_chain_plugin.py

experiments/presets/
  route_a_hotpotqa.json
  route_a_hybridqa.json
```

当前阶段不要求一次性重构完，但后续新增代码优先放入上述新目录，不再继续把新逻辑堆进旧原型文件。

## 5. 阶段任务

### Phase A1：可信文本 baseline

目标：

- 用 LlamaIndex 建立 HotpotQA baseline
- 输出稳定的 EM/F1 / evidence recall / latency / token 指标

完成标准：

- 能独立运行
- 能接入现有结果落盘格式
- 能被 `run_all.py` 或其兼容入口复用

### Phase A2：插件回挂

目标：

- 将 Adaptive 和 CoVe 接到新 baseline 上

最小对比矩阵：

- Baseline
- Baseline + Adaptive
- Baseline + CoVe
- Baseline + Adaptive + CoVe

### Phase A3：真实异构任务

目标：

- 先在 `HybridQA` 或 `OTT-QA` 上建立真实 text+table baseline
- 将“异构退化”从伪噪声问题转化为真实任务问题

### Phase A4：论文主线更新

目标：

- 将旧“伪异构诊断”收口为历史阶段发现
- 将新 baseline 和真实异构实验写成正式主线

## 6. 与当前论文的兼容策略

当前论文里最有价值的“诊断型发现”不需要直接删除，但需要重新定位：

- 旧结果可保留为“原型阶段诊断实验”
- 新 Route A 结果作为“正式 baseline 与扩展实验”

这样不会浪费已有分析，也不会让主结论继续绑在低可信度 baseline 上。

## 7. 当前仓库的近期行动

P0：

- 清理旧网站入口与文档误导
- 更新 README 和总指南，明确 Route A 为当前主线

P1：

- 搭建 `src/rererank_v1/baselines/` 和 `adapters/` 新骨架
- 设计第一版 LlamaIndex baseline 接口
- 新增 `experiments/presets/route_a_hotpotqa.json` 作为 Phase A1 配置目标

P2：

- 引入 HybridQA / OTT-QA 的数据接入说明和实验预设
- 将旧 `dataset_loader.py` 中的伪异构逻辑下沉为“legacy path”
