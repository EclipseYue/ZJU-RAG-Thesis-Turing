# 本地评估与问题修复记录

## 1. 本次检查的核心发现

本轮检查针对仓库结构、实验脚本、环境依赖和本机硬件进行了逐项核验，发现以下问题：

1. 论文正文中的部分结果与当前仓库可复现实验不一致。
2. `experiments/run_large_scale.py` 原先输出的是手工构造数据，不应直接作为论文实验结果。
3. `phase4_experiment.py` 与 `run_all.py` 在统计 token 与 latency 时使用了累计值，导致平均值被放大。
4. Windows + GBK 控制台下，实验脚本中的 emoji 会导致运行中断。
5. `rag-thesis` 环境未安装完整深度学习依赖，且依赖范围过宽，导致装包解析不稳定。
6. 当前机器 GPU 为 `Intel Arc 130T`，与文档里记录的 `RTX 3060 6GB` 不一致。

## 2. 已完成修复

- 将 `src/rererank_v1/rag_pipeline.py` 改为懒加载重型依赖，保证 mock 模式在未装齐深度学习栈时仍可运行。
- 修复 `experiments/phase4_experiment.py` 与 `experiments/run_all.py` 的统计逻辑，改为按 query 重置计数器。
- 将 `experiments/run_large_scale.py` 改为容量规划脚本，输出 `data/results/large_scale_plan.json`。
- 为 `experiments/phase2_experiment.py` 增加代理指标输出，至少可以形成异构检索的初步量化记录。
- 清除实验脚本中的 emoji 控制台输出，保证 Windows 环境可运行。
- 更新 `README.md`、`THESIS_CONTENT.md`、`docs/experiment_overview.md`，区分本地可复现实验与大规模规划。

## 3. 本机复现实验结果

### 3.1 Phase 1

- 查询数：17
- MRR：0.3422
- NDCG@5：0.4923
- Precision@3：0.3529
- Avg Tokens：181.3
- p50 Latency：0.002s
- p95 Latency：0.004s

### 3.2 Phase 4 Ablation

| Model | Hetero | Adaptive | CoVe | Avg Tokens | Avg Latency (ms) | No-Answer Rate |
|------|--------|----------|------|------------|------------------|----------------|
| A Baseline | no | no | no | 117 | 0.67 | 0% |
| B +Hetero | yes | no | no | 94 | 0.00 | 0% |
| C +Adaptive | yes | yes | no | 115 | 0.17 | 0% |
| D +CoVe (Full) | yes | yes | yes | 115 | 0.33 | 100% |

说明：该表来自 3 条诊断查询的小规模 mock 评估，适合用于展示机制趋势，不适合作为论文最终大样本主结果表。

## 4. 本机硬件评估

- GPU：Intel(R) Arc(TM) 130T GPU
- CUDA：不可用
- 结论：当前机器适合进行 CPU/mock 验证、脚本联调、案例分析，不适合承担论文最终版的真实模型大规模消融。

## 5. 服务器建议

如果要完成 100~500 条 HotpotQA / 2WikiMultihopQA 的真实模型实验，建议：

- 首选：租用带 NVIDIA GPU 的云服务器，如 RTX 4090 / A10 / L4 / A100。
- 最低建议：显存 16GB 以上，便于稳定运行嵌入模型和 reranker。
- 推荐任务分层：
  - 本机：debug、mock 验证、论文插图案例
  - 服务器：真实模型、批量评测、最终统计图表

## 6. 后续优先级

1. 先用 `environment.rag-thesis.yml` 重建稳定环境。
2. 在服务器上跑 100 条真实样本，形成第一版真实主表。
3. 再扩展到 300/500 条，补齐消融图和成本曲线。
4. 最后统一回写论文正文、附录硬件配置与实验章节。
