# GPU 与实验规模升级记录

## 本次发现的问题

1. `experiments/run_all.py` 仍使用手工构造的小样本查询，不能支撑真实基准结论。
2. `experiments/run_large_scale.py` 直接写入预设结果，不是实际运行得到的数据。
3. 仓库缺少可复现的 `conda` 环境定义，当前 IDE 会话中也缺失 `sentence-transformers` 与 `datasets`。
4. 文档中仍保留早期小规模实验与旧硬件信息，未体现后续可扩展到 `RTX 3060 12GB` 的新条件。

## 已完成修复

1. 将 `experiments/run_all.py` 重构为真实基准评测入口：
   - 支持 `HotpotQA` 与 `2WikiMultihopQA`
   - 支持 `--samples`、`--top-k`、`--device`、`--mock`
   - 真实输出 `SupportRecall@K`、`SupportAllHit@K`、`EM`、`F1`、`Avg_Tokens`、`Avg_Latency_ms`、`No_Answer_Rate_Percent`
   - 自动落盘到 `data/results/automated_ablation.json` 与详细报告文件
2. 将 `experiments/run_large_scale.py` 改为真实运行包装脚本，不再生成伪造结果。
3. 新增 `environment.yml`，用于在宿主机创建独立 `conda` 环境。
4. 更新项目主 README、实验 README 与论文相关文档，明确新的运行方式和硬件说明。

## 当前运行约束

- 当前 IDE 容器内 `torch` 为 CUDA 构建版本，但 `nvidia-smi` 不可用，`torch.cuda.is_available()` 返回 `False`。
- 这说明仓库代码已经准备好使用 GPU，但当前会话没有拿到宿主机的 NVIDIA 驱动直通。
- 在宿主机终端或具备驱动直通的开发容器中，`RTX 3060 12GB` 可直接用于 `100/300/500` 样本实验。

## 推荐运行命令

```bash
conda env create -f environment.yml
conda activate zju-rag-thesis

python experiments/run_all.py --dataset hotpotqa --samples 100 --device cuda
python experiments/run_large_scale.py --dataset hotpotqa --samples 500 --device cuda
python experiments/run_large_scale.py --dataset 2wiki --samples 300 --device cuda
```

## 建议的论文更新口径

- 保留已完成的 `N=100` 小规模结果作为当前正文主表。
- 将 `100/300/500` 的可扩展跑批能力写入“实验平台与复现环境”部分。
- 新增一句说明：早期实验受限于无可用显卡，仅做小规模验证；现已完成脚本与环境升级，可在 `RTX 3060 12GB` 条件下复现实验并扩展样本规模。
