# 离线数据放置方式

将导出的数据集文件放到仓库内的 `data/datasets/` 下，支持 `json` 或 `jsonl`。

推荐目录结构：

```text
data/datasets/
  hotpotqa/
    validation.json
  2wikimultihopqa/
    validation.json
```

也支持这种平铺命名：

```text
data/datasets/hotpotqa_validation.json
data/datasets/2wikimultihopqa_validation.json
```

本地文件中的每条记录应保留 Hugging Face 原始字段，至少包括：

- `id`
- `question`
- `answer`
- `context`
- `supporting_facts`

其中 `context` 需要包含：

- `title`
- `sentences`

其中 `supporting_facts` 需要包含：

- `title`

离线运行示例：

```bash
python3 experiments/run_all.py --config experiments/configs/ablation_with_controls.json
python3 experiments/run_false_rejection_diagnostics.py --config experiments/configs/false_rejection_diagnostics.json
python3 experiments/run_bucket_gain_study.py --config experiments/configs/bucket_gain_study.json
python3 experiments/run_verifier_comparison.py --config experiments/configs/verifier_comparison.json
```
