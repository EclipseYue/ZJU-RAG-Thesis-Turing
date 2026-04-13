---
name: "experiment-runner"
description: "运行与改造实验脚本（reranker_study/v6_experiment），整理结果与对比表。用户要求跑实验、加指标、保存结果或复现实验时调用。"
---

# Experiment Runner

## 目标

- 维护 `experiments/` 下的实验脚本与实验入口
- 统一结果落盘到 `data/`（history 与 results）
- 输出可复现的对比结果（指标、表格、报告）

## 约定

- 入口脚本从仓库根目录运行：`python reranker_study.py`、`python v6_experiment.py`
- 结果文件放入 `data/research_history.json` 与 `data/results/`

