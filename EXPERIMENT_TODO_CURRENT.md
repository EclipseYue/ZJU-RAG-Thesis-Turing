# 当前实验待办清单

本文档面向 `2026-04-28` 这一阶段的实际推进情况整理。当前目标已经从“直接补真实 CoVe”进一步调整为：

- 用 **Route A** 建立可信 baseline
- 把旧消融壳降级为“小样本真实 API 诊断线”
- 在 GPU 服务器上稳定推进，不再重复之前的长时全损问题

总原则：

- 保留现有历史结果，但不再把旧 baseline 当成后续主线。
- Route A baseline 结果优先级高于旧消融壳的大样本补跑。
- 真实大模型实验只做**小样本补充验证**，不追求全量 7405。
- GPU 服务器优先用于本地检索、嵌入、重排和 baseline 建设。
- 生成与验证走真实 API，但必须控制请求速率、样本量和配置范围。

推荐搭配阅读：

- [GPU 服务器迁移与真实 API 实验运行手册](/Users/eclipse/code/RAG/Rererank_v1/docs/GPU_SERVER_RUNBOOK.md)

建议先在服务器设置限速环境变量：

```bash
export RERERANK_LLM_MIN_INTERVAL_SEC=3.5
export RERERANK_LLM_MAX_RETRIES=8
export RERERANK_LLM_BACKOFF_BASE_SEC=8
export RERERANK_LLM_BACKOFF_MAX_SEC=90
```

## P0 必做

### 1. GPU 服务器准备与 Route A 环境验证

任务：

- 在 GPU 服务器上重建 `.venv`
- 安装 Route A 依赖
- 同步 `local_api_overrides.json` 和离线数据

命令：

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  -r requirements-route-a.txt
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
.venv/bin/python -c "import llama_index; import llama_index.core; print('llamaindex ok')"
ls experiments/configs/local_api_overrides.json
```

通过标准：

- `.venv` 可用
- `llama_index` 可导入
- `local_api_overrides.json` 存在
- 离线数据目录存在

### 2. Route A 检索 smoke

任务：

- 先在服务器上确认 Route A baseline 能正常跑
- 不急着打开真实生成

命令：

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 20 \
  --generator-backend heuristic \
  --output-name route_a_hotpotqa_server_smoke.json
```

预期产物：

- `data/results/route_a_hotpotqa_server_smoke.json`

### 3. Route A 真实生成 smoke

任务：

- 验证 Route A + 真实 API 生成链是否稳定
- 检查空答案、长句复述和极端慢速是否仍存在

命令：

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 20 \
  --output-name route_a_hotpotqa_realapi_smoke.json
```

### 4. Route A 100 样本小规模 baseline

任务：

- 获取第一版服务器上的 Route A 候选 baseline 结果

命令：

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 100 \
  --output-name route_a_hotpotqa_realapi_100.json
```

说明：

- 这一步优先级高于旧版大样本消融
- 只要这一步还没稳，旧 baseline 就不应该继续扩样本

### 5. 旧主消融单配置 smoke test

任务：

- 验证旧实验壳在服务器上是否能稳定落盘
- 只检查最基础的 `A_Baseline`

命令：

```bash
.venv/bin/python experiments/run_all.py \
  --config experiments/configs/ablation_with_controls.json \
  --samples 20 \
  --only-configs A_Baseline \
  --checkpoint-every 2 \
  --output-name automated_ablation_smoke_a_server.json
```

预期产物：

- `data/results/automated_ablation_smoke_a_server.json`

通过标准：

- 脚本正常结束
- 输出 JSON 正常落盘
- 没有再次出现长时间无结果文件的情况

### 6. 真实 CoVe 关键配置 smoke

任务：

- 只运行真正依赖 CoVe 的关键配置
- 避免一上来扫全量 A/A2/A3/B/C/D

命令：

```bash
.venv/bin/python experiments/run_all.py \
  --config experiments/configs/ablation_with_controls.json \
  --samples 20 \
  --real-cove \
  --only-configs A3_Baseline_CoVe D_CoVe_Full \
  --checkpoint-every 2 \
  --output-name automated_ablation_smoke_cove_server.json
```

预期产物：

- `data/results/automated_ablation_smoke_cove_server.json`

## P1 强烈建议

### 7. 真实 CoVe 验证器对比实验

任务：

- 比较 `cove_soft / cove_standard / cove_strict / overlap_soft`
- 观察真实 verifier 条件下 FRR 与 unsafe accept 的变化
- 这一步继续保留旧实验壳，但定位为“诊断线”

命令：

```bash
.venv/bin/python experiments/run_verifier_comparison.py \
  --config experiments/configs/verifier_comparison.json \
  --samples 100 \
  --real-cove \
  --output-name verifier_comparison_real_cove_100.json
```

预期产物：

- `data/results/verifier_comparison_real_cove_100.json`

用于论文：

- 更新 verifier 诊断表
- 讨论类 CoVe 与真实 CoVe 的行为差异

### 8. 真实 CoVe 错误拒答诊断

任务：

- 在真实 verifier 条件下诊断 false rejection 来源
- 为 4.2 节及相关诊断表提供补充证据

命令：

```bash
.venv/bin/python experiments/run_false_rejection_diagnostics.py \
  --config experiments/configs/false_rejection_diagnostics.json \
  --samples 100 \
  --real-cove \
  --output-name false_rejection_diagnostics_real_cove_100.json
```

如果前面很稳定，可扩到 200：

```bash
.venv/bin/python experiments/run_false_rejection_diagnostics.py \
  --config experiments/configs/false_rejection_diagnostics.json \
  --samples 200 \
  --real-cove \
  --output-name false_rejection_diagnostics_real_cove_200.json
```

预期产物：

- `data/results/false_rejection_diagnostics_real_cove_100.json`
- 或 `data/results/false_rejection_diagnostics_real_cove_200.json`

用于论文：

- 补真实 verifier 下的拒答率、FRR、不安全接受率

### 9. 拉回结果并重画图

任务：

- 把服务器结果同步回本地
- 先分析再决定哪些进入正文主线

拉回结果示例：

```bash
scp root@<server-ip>:/root/home/ZJU-RAG-Thesis-Turing/data/results/*route_a*json /Users/eclipse/code/RAG/Rererank_v1/data/results/
scp root@<server-ip>:/root/home/ZJU-RAG-Thesis-Turing/data/results/*real_cove*json /Users/eclipse/code/RAG/Rererank_v1/data/results/
```

重画现有旧图命令：

```bash
python3 experiments/plot_results.py
```

说明：

- Route A 图表建议单独新增，不直接覆盖旧图
- 旧图仍主要对应历史消融主线

## P2 可选补充

### 10. 旧消融壳的完整真实 API 小样本实验

任务：

- 把生成端和 verifier 都维持真实 API
- 作为“旧实验壳在真实条件下”的补充实验

推荐先做 50 样本：

```bash
.venv/bin/python experiments/run_all.py \
  --config experiments/configs/ablation_with_controls.json \
  --samples 50 \
  --real-cove \
  --output-name automated_ablation_real_llm_50.json
```

说明：

- 只有当 Route A 100 样本已经稳定后，才建议做这一步
- 如果 `generator_api_key` 未配置，这条命令仍会回退到启发式生成

### 11. provider 对照实验

任务：

- 比较不同真实生成器对整体结果的影响
- 例如 DeepSeek 生成与其他 provider 的差异

说明：

- 当前项目尚未把“多 provider 对照”单独整理成公共配置模板
- 建议只做 50 样本以内的小规模实验

## 暂不建议做

### 12. 不建议直接跑全量 7405 的真实 API 实验

原因：

- RPM 和 TPD 风险高
- 成本不可控
- 对本科论文的边际收益有限

### 13. 不建议在本阶段切换到全新真实异构数据集

原因：

- Route A 的第一优先级仍是可信文本 baseline
- 需要先确认新 baseline 和真实生成链已经稳定

## 结果回填任务

### 14. 论文回填

任务：

- Route A 结果优先作为后续主线候选
- 旧消融壳结果优先写进诊断段落
- 保留类 CoVe 历史结果，但不要继续把它包装成最终 baseline

涉及文件：

- `paper/zjuthesis/body/undergraduate/final/2-body.tex`
- `paper/zjuthesis/body/undergraduate/final/abstract.tex`
- `paper/zjuthesis/body/undergraduate/final/5-conclusion.tex`

### 15. 文档轻微更新

任务：

- 在完成服务器实验后，同步更新待办与总指南

涉及文件：

- `OVERALL_TODO.md`
- `docs/EXPERIMENT_MASTER_GUIDE.md`
- `docs/GPU_SERVER_RUNBOOK.md`

## 建议执行顺序

1. 准备 GPU 服务器环境
2. 跑 Route A heuristic smoke
3. 跑 Route A 真实生成 smoke
4. 跑 Route A 100 样本 baseline
5. 跑旧 `A_Baseline` 20 样本 smoke
6. 跑旧 `A3_Baseline_CoVe / D_CoVe_Full` 20 样本 smoke
7. 跑 `run_verifier_comparison.py` 的 100 样本真实 CoVe
8. 跑 `run_false_rejection_diagnostics.py` 的 100 样本真实 CoVe
9. 拉回结果
10. 重画图
11. 回填论文
12. 更新 TODO 和 guide
