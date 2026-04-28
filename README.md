# Route A RAG Thesis Workspace

本仓库当前采用 **Route A** 作为主线：  
保留现有实验编排、论文联动和诊断分析框架，逐步将底层 baseline 替换为成熟 RAG 标准件，并把“伪异构”升级为真实异构任务或真实异构知识源。

## 当前定位

这已经不再是一个“从零自写所有底层组件”的原型仓库，而是一个面向毕业论文交付的研究工作区，包含三类资产：

- `paper/`：论文与答辩材料
- `experiments/`：实验入口、配置、绘图与结果回填脚本
- `src/rererank_v1/`：当前研究代码，包括旧版自研 baseline、Adaptive/CoVe 等可迁移模块

## Route A 的核心判断

当前仓库里最值得保留的，是上层研究框架，而不是旧版 baseline 本身。

保留：

- 主消融与诊断实验壳
- 结果落盘、checkpoint、配置覆盖机制
- Adaptive PRF、CoVe、evidence chain 等可插拔研究模块
- 图表生成与论文回填链路

替换：

- 低可信度的旧版自研文本 baseline
- “由文本机械派生 table/graph”的伪异构构造方式
- 不适配 HotpotQA 风格 EM/F1 的宽松生成出口

## 新路线架构

Route A 推荐分三层推进：

1. **可信文本 baseline 层**
   使用 LlamaIndex 等成熟框架先建立 HotpotQA / 2Wiki 的可复现实验底座。

2. **研究模块层**
   将 Adaptive PRF、CoVe、evidence chain 作为插件挂载到成熟 baseline 之上，验证真实增量。

3. **真实异构层**
   优先接入 `HybridQA` / `OTT-QA` 这样的真实 text+table 任务；若时间允许，再扩展到 Wikidata/Neo4j 类图结构知识源。

详细迁移方案见：

- [Route A 架构与迁移蓝图](/Users/eclipse/code/RAG/Rererank_v1/docs/ROUTE_A_ARCHITECTURE.md)
- [实验总指南](/Users/eclipse/code/RAG/Rererank_v1/docs/EXPERIMENT_MASTER_GUIDE.md)
- [Overall TODO](/Users/eclipse/code/RAG/Rererank_v1/OVERALL_TODO.md)

## 仓库结构

- `src/rererank_v1/`
  当前研究代码与待迁移模块
- `experiments/`
  主消融、诊断实验、配置文件、绘图脚本
- `data/`
  本地数据与实验结果
- `docs/`
  研究路线、实验规范、历史计划
- `paper/`
  论文源文件、图表与编译输出

## 当前建议的工作顺序

1. 用成熟框架重建纯文本 baseline
2. 在新 baseline 上接回 Adaptive / CoVe
3. 将异构实验切换到真实任务（优先 HybridQA / OTT-QA）
4. 把真实 LLM 小样本实验作为补充验证，而不是全量主结果
5. 再更新论文中的实验主线和结论边界

Route A baseline 试跑入口：

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 20 \
  --generator-backend heuristic \
  --output-name route_a_hotpotqa_smoke.json
```

Route A 额外依赖：

```bash
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  -r requirements-route-a.txt
```

## 说明

- 历史上的科研可视化 Flask 看板已从当前主线移除，不再作为仓库入口维护。
- 当前仓库仍保留一部分旧版实验和原型代码，作为迁移阶段参考，不代表后续正式 baseline。
