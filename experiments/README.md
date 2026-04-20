# 实验

建议从仓库根目录运行入口脚本：

```bash
python reranker_study.py
python v6_experiment.py
python experiments/run_all.py --dataset hotpotqa --samples 100 --device cuda
python experiments/run_large_scale.py --dataset hotpotqa --samples 500 --device cuda
```

输出位置：

- `data/research_history.json`：实验迭代记录（Dashboard 与报告复用）。
- `data/results/`：实验结果 JSON（如 `v6_experiment_results_*.json`）。

真实基准评测说明：

- `run_all.py`：真实基准消融入口，支持 `HotpotQA` / `2WikiMultihopQA`、样本规模参数、`cpu/cuda/mock` 模式切换。
- `run_large_scale.py`：大规模包装脚本，默认按 `N=500` 调用 `run_all.py`，不再写入预设假结果。
- 推荐先做小规模试跑再扩展：

```bash
python experiments/run_all.py --dataset hotpotqa --samples 20 --device cpu
python experiments/run_all.py --dataset hotpotqa --samples 100 --device cuda
python experiments/run_large_scale.py --dataset 2wiki --samples 300 --device cuda
```

离线运行说明：

- 远程服务器无法访问 `huggingface.co` 时，优先将数据集导出为 `json/jsonl` 放到 `data/datasets/`。
- 具体目录结构见 [experiments/configs/README_offline.md](/Users/eclipse/code/RAG/Rererank_v1/experiments/configs/README_offline.md)。
- 当前四个实验配置文件已经默认开启 `offline=true`，并优先读取 `data/datasets/`。
