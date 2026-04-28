# 实验总指南

本文件统一整合仓库中的实验入口、配置方式、离线数据说明、LLM/API 后端说明、结果落盘约定与论文更新流程。后续如需执行实验，请优先以本文档为准。

当前仓库已明确选择 **Route A**：

- 保留现有实验壳与论文联动机制
- 用成熟框架重建可信 baseline
- 将真实异构任务作为后续主线

路线总览见：
- [Route A 架构与迁移蓝图](/Users/eclipse/code/RAG/Rererank_v1/docs/ROUTE_A_ARCHITECTURE.md)
- [GPU 服务器迁移与真实 API 实验运行手册](/Users/eclipse/code/RAG/Rererank_v1/docs/GPU_SERVER_RUNBOOK.md)

## 1. 目标与范围

本项目当前的实验分为三层：

- 迁移层：清理旧原型、搭建新 baseline 接口、验证实验壳可复用性。
- 原型/诊断层：保留旧 HotpotQA / 2Wiki 消融与失败分析，作为历史阶段参考。
- 正式主线层：未来在成熟 baseline 与真实异构任务上重跑关键实验，并回填论文。

本文档只负责说明如何配置和执行，不替你自动跑实验。

## 2. 关键实验入口

说明：

- 现有入口大多仍服务于“旧 baseline + 诊断实验”。
- Route A 下，这些入口优先被视为 **实验壳**，不是最终 baseline 本身。

### 2.1 主消融入口

文件：`experiments/run_all.py`

用途：
- 跑主消融矩阵
- 支持 A / A2 / A3 / B / C / D 控制组
- 支持离线数据加载
- 支持启发式或真实 LLM 生成/验证后端切换

常用命令：

```bash
python experiments/run_all.py --config experiments/configs/ablation_with_controls.json
python experiments/run_all.py --dataset hotpotqa --samples 100 --device cuda --include-controls
python experiments/run_all.py --config experiments/configs/ablation_with_controls.json --real-cove
```

### 2.2 补充诊断入口

- `experiments/run_false_rejection_diagnostics.py`
  用于错误拒答来源诊断
- `experiments/run_bucket_gain_study.py`
  用于 bridge / comparison 分桶分析
- `experiments/run_verifier_comparison.py`
  用于验证器模式与阈值对比
- `experiments/run_supplementary_study.py`
  用于小规模补充对照

推荐命令：

```bash
python experiments/run_false_rejection_diagnostics.py --config experiments/configs/false_rejection_diagnostics.json
python experiments/run_false_rejection_diagnostics.py --config experiments/configs/false_rejection_diagnostics.json --real-cove
python experiments/run_verifier_comparison.py --config experiments/configs/verifier_comparison.json
python experiments/run_verifier_comparison.py --config experiments/configs/verifier_comparison.json --real-cove
```

### 2.3 Route A baseline 入口

文件：`experiments/run_route_a_baseline.py`

用途：
- 运行成熟框架底座的 Route A baseline
- 当前支持 `llamaindex_text`
- 默认读取 `experiments/presets/route_a_hotpotqa.json`

默认小样本试跑命令：

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 20 \
  --output-name route_a_hotpotqa_realapi_smoke.json
```

说明：

- 当前 `route_a_hotpotqa.json` 已显式声明 `generator_backend=deepseek`、`generator_model=deepseek-v4-flash`
- 若 `experiments/configs/local_api_overrides.json` 中也存在生成端配置，则会继续以该私有配置为准
- 只有在你显式传入 `--generator-backend heuristic` 时，才会改走启发式

如果只是排查检索链，不希望消耗真实 API：

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 20 \
  --generator-backend heuristic \
  --output-name route_a_hotpotqa_heuristic_smoke.json
```

若本地 PyPI 证书或代理不稳定，Route A 依赖建议安装到仓库 `.venv`：

```bash
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  -r requirements-route-a.txt
```

兼容普通环境的命令如下：

```bash
python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 20 \
  --output-name route_a_hotpotqa_realapi_smoke.json
```

迁移到 GPU 服务器后的推荐执行顺序，见：

- [GPU 服务器迁移与真实 API 实验运行手册](/Users/eclipse/code/RAG/Rererank_v1/docs/GPU_SERVER_RUNBOOK.md)

## 3. 数据与离线模式

### 3.1 默认数据目录

推荐把离线导出的数据集放到：

```text
data/datasets/hotpotqa/validation.json
data/datasets/2wikimultihopqa/validation.json
```

离线目录细节可参考：
- `experiments/configs/README_offline.md`

### 3.2 离线优先策略

当前加载器顺序：

1. 本地 `json/jsonl`
2. 本地 Hugging Face cache
3. 在线下载

如果配置了 `offline=true`，则只走前两步。

## 4. 生成与验证后端

### 4.1 当前默认行为

- 生成端默认 `generator_backend=auto`
- 验证端默认 `verifier_backend=heuristic`

这意味着：
- 生成端若检测到 OpenAI 兼容接口配置，会优先使用真实 API
- 若没有检测到可用 API，则回退到启发式生成
- 验证端默认仍然使用启发式 CoVe 近似，不会自动调用外部大模型

### 4.2 支持的后端

`run_all.py` 当前支持以下后端参数：

```text
--generator-backend auto|heuristic|openai|deepseek|moonshot|siliconflow
--verifier-backend heuristic|openai|deepseek|moonshot|siliconflow
--generator-model <model_name>
--verifier-model <model_name>
--real-cove
```

`--real-cove` 的约定是：

- 不改变实验矩阵本身，只改变 CoVe 相关配置实际使用的验证后端。
- 不传入时，默认沿用“类 CoVe / 启发式近似验证”。
- 传入后，优先使用 `verifier_backend` 和 `verifier_model`。
- 若传入 `--real-cove` 但 `verifier_backend` 仍为 `heuristic`，主入口会自动回退到 `deepseek` 作为默认真实 CoVe 后端。

### 4.3 环境变量

支持三类 OpenAI 兼容接口：

- OpenAI 风格：
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`
- DeepSeek：
  - `DEEPSEEK_API_KEY`
  - `DEEPSEEK_BASE_URL`，默认 `https://api.deepseek.com`
- Moonshot / Kimi（历史可选）：
  - `MOONSHOT_API_KEY` 或 `KIMI_API_KEY`
  - `MOONSHOT_BASE_URL`，默认 `https://api.moonshot.cn/v1`
- SiliconFlow：
  - `SILICONFLOW_API_KEY`
  - `SILICONFLOW_BASE_URL`，默认 `https://api.siliconflow.cn/v1`

若 `generator_backend=auto`，系统会优先尝试 `OPENAI_*`，再尝试 DeepSeek，再尝试 Moonshot，最后尝试 SiliconFlow。

### 4.4 本地私有覆盖配置

为避免把私钥写进公共配置文件，项目支持本地私有覆盖文件：

```text
experiments/configs/local_api_overrides.json
```

该文件会在以下脚本中被自动加载：

- `experiments/run_all.py`
- `experiments/run_false_rejection_diagnostics.py`
- `experiments/run_verifier_comparison.py`

推荐把真实 CoVe 所需的 `verifier_backend`、`verifier_model`、`verifier_api_key`、`verifier_base_url` 放在这里。

### 4.5 示例

使用 DeepSeek 作为真实验证后端，但不实际运行示例：

```bash
export DEEPSEEK_API_KEY="sk-..."
python experiments/run_all.py \
  --config experiments/configs/ablation_with_controls.json \
  --generator-backend deepseek \
  --generator-model deepseek-v4-flash \
  --verifier-backend deepseek \
  --verifier-model deepseek-v4-flash \
  --real-cove
```

## 4.6 迁移到 GPU 服务器后的建议

当前推荐做法不是“把所有实验都切成真实 API 全量跑”，而是：

- Route A baseline 作为正式主线
- 旧 `run_all.py` / `run_verifier_comparison.py` / `run_false_rejection_diagnostics.py` 作为小样本诊断线
- 真实 API 只在关键实验上开启
- 旧消融壳优先使用 `--only-configs` 与 `--checkpoint-every`

推荐优先顺序：

1. Route A heuristic smoke
2. Route A 真实生成 smoke
3. Route A 100 样本真实生成 baseline
4. 旧 `A_Baseline` 20 样本 smoke
5. 旧 `A3_Baseline_CoVe / D_CoVe_Full` 20 样本 smoke
6. verifier comparison / false rejection diagnostics 小样本真实 CoVe

详细命令和迁移步骤见：

- [GPU 服务器迁移与真实 API 实验运行手册](/Users/eclipse/code/RAG/Rererank_v1/docs/GPU_SERVER_RUNBOOK.md)

## 5. 配置文件

当前主配置位于 `experiments/configs/`：

- `ablation_with_controls.json`
- `false_rejection_diagnostics.json`
- `bucket_gain_study.json`
- `verifier_comparison.json`

Route A 的新预设位于 `experiments/presets/`：

- `route_a_hotpotqa.json`
- `route_a_hybridqa.json`

建议做法：
- 把数据路径、设备、离线模式写进配置文件
- 把是否使用真实后端放在命令行覆盖，避免误跑高成本实验
- 把私钥与网关地址放在 `local_api_overrides.json`，不要写进公共配置

## 6. 输出文件

主要结果统一落在：

```text
data/results/
```

常见文件包括：

- `automated_ablation_with_controls.json`
- `*_report_时间戳.json`
- `false_rejection_diagnostics_*.json`
- `bucket_gain_study_*.json`
- `verifier_comparison_*.json`

研究历史追踪：

```text
data/research_history.json
```

## 7. 论文联动

实验结果更新后，通常有三步：

1. 更新 `data/results/` 中的主结果 JSON
2. 运行 `experiments/plot_results.py` 重绘图表
3. 修改 `paper/zjuthesis/body/undergraduate/final/` 中的正文和摘要

论文主文件位置：

- `paper/zjuthesis/body/undergraduate/final/1-introduction.tex`
- `paper/zjuthesis/body/undergraduate/final/2-body.tex`
- `paper/zjuthesis/body/undergraduate/final/abstract.tex`
- `paper/zjuthesis/body/undergraduate/final/5-conclusion.tex`

编译命令：

```bash
cd paper/zjuthesis
latexmk -g
```

## 8. 当前边界

需要明确：

- 当前“异构数据”主要是伴生式规则构造，不是真实独立表格库/图谱库。
- 当前 CoVe 默认是启发式近似验证，不应在论文里无条件表述为“完整复现的真实 LLM CoVe”。
- 真实 LLM 生成/验证接口已经可配置，但是否启用是实验选择，不是默认行为。
- 现有 `run_all.py` 等入口适合做迁移前的诊断与小规模补充验证，但不应继续被视为最终可信 baseline 的唯一来源。

## 9. 推荐执行顺序

如果后续准备正式跑一轮新实验，建议顺序如下：

1. 先确认本地/远程离线数据在位
2. 先跑一轮类 CoVe 小样本试跑
3. 再用 `--real-cove` 跑同口径版本
4. 再跑错误拒答诊断和验证器对比
5. 视情况补跑 bridge / comparison 分桶
6. 重绘图表
7. 回填论文

## 10. 相关旧文档

以下文档已被本文档吸收，保留仅作历史参考：

- `experiments/README.md`
- `docs/experiment_overview.md`
- `docs/project_guide.md`
- `README.md` 中的实验部分
