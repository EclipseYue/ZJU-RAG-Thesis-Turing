# 当前实验待办清单

本文档面向 `2026-04-26` 这一阶段的实际推进情况整理，目标是在**不推翻现有主论文主线**的前提下，补充“真实大模型 API 条件下”的关键实验。

总原则：

- 保留现有大样本主结果，继续作为正文主表主图来源。
- 真实大模型实验只做**小样本补充验证**，不追求全量 7405。
- 优先使用服务器显卡完成检索、重排和本地 RAG 前半段。
- 生成与验证走真实 API，但要控制请求速率和样本量。

建议先在服务器设置限速环境变量：

```bash
export RERERANK_LLM_MIN_INTERVAL_SEC=3.5
export RERERANK_LLM_MAX_RETRIES=8
export RERERANK_LLM_BACKOFF_BASE_SEC=8
export RERERANK_LLM_BACKOFF_MAX_SEC=90
```

## P0 必做

### 1. 真实 CoVe 主实验 smoke test

任务：
- 验证服务器上的真实 verifier 配置可用
- 验证 `--real-cove` 链路不会因为 RPM/TPD 或配置问题直接失败

命令：

```bash
python3 experiments/run_all.py \
  --config experiments/configs/ablation_with_controls.json \
  --samples 20 \
  --real-cove \
  --output-name automated_ablation_real_cove_smoke.json
```

预期产物：
- `data/results/automated_ablation_real_cove_smoke.json`

通过标准：
- 脚本正常结束
- 输出 JSON 正常落盘
- 日志中没有持续性 429 死循环

### 2. 真实 CoVe 小规模主消融

任务：
- 在小样本条件下补一版“真实 verifier”实验
- 用于和现有类 CoVe 主结果形成对照

命令：

```bash
python3 experiments/run_all.py \
  --config experiments/configs/ablation_with_controls.json \
  --samples 100 \
  --real-cove \
  --output-name automated_ablation_real_cove_100.json
```

预期产物：
- `data/results/automated_ablation_real_cove_100.json`

用于论文：
- 新增“真实 CoVe 条件下的小规模复现实验”
- 重画真实 CoVe 版本的 F1 / Safety / Cost 图

### 3. 真实 CoVe 验证器对比实验

任务：
- 比较 `cove_soft / cove_standard / cove_strict / overlap_soft`
- 观察真实 verifier 条件下 FRR 与 unsafe accept 的变化

命令：

```bash
python3 experiments/run_verifier_comparison.py \
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

## P1 强烈建议

### 4. 真实 CoVe 错误拒答诊断

任务：
- 在真实 verifier 条件下诊断 false rejection 来源
- 为 4.2 节及相关诊断表提供补充证据

命令：

```bash
python3 experiments/run_false_rejection_diagnostics.py \
  --config experiments/configs/false_rejection_diagnostics.json \
  --samples 100 \
  --real-cove \
  --output-name false_rejection_diagnostics_real_cove_100.json
```

如果前面很稳定，可扩到 200：

```bash
python3 experiments/run_false_rejection_diagnostics.py \
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

### 5. 拉回结果并重画图

任务：
- 把服务器真实 CoVe 结果同步回本地
- 基于新 JSON 重画真实 CoVe 图表

拉回结果示例：

```bash
scp root@<server-ip>:/root/home/ZJU-RAG-Thesis-Turing/data/results/*real_cove*.json /Users/eclipse/code/RAG/Rererank_v1/data/results/
```

现有主图重画命令：

```bash
python3 experiments/plot_results.py
```

说明：
- 当前 `plot_results.py` 默认仍读 `automated_ablation_with_controls.json`
- 如果要为真实 CoVe 单独出图，建议下一步补一个真实 CoVe 图输出入口

## P2 可选补充

### 6. 真实生成 + 真实 CoVe 的完整小样本实验

任务：
- 把生成端也切到真实大模型 API
- 作为“完整真实 RAG 流程”的补充实验

前提：
- 当前 provider 配额允许
- 当前本地私有配置已补上 generator 的真实 API 配置

推荐先做 50 样本：

```bash
python3 experiments/run_all.py \
  --config experiments/configs/ablation_with_controls.json \
  --samples 50 \
  --real-cove \
  --output-name automated_ablation_real_llm_50.json
```

说明：
- 如果 `generator_api_key` 未配置，这条命令仍会回退到启发式生成
- 因此运行前应确认 `experiments/configs/local_api_overrides.json` 中生成端是否已经真实启用

### 7. provider 对照实验

任务：
- 比较不同真实生成器对整体结果的影响
- 例如 Qwen 生成 vs Kimi 生成

说明：
- 当前项目尚未把“多 provider 对照”单独整理成公共配置模板
- 建议只做 50 样本以内的小规模实验

## 暂不建议做

### 8. 不建议直接跑全量 7405 的真实 API 实验

原因：
- RPM 和 TPD 风险高
- 成本不可控
- 对本科论文的边际收益有限

### 9. 不建议在本阶段切换到全新真实异构数据集

原因：
- 会改变现有实验前提
- 需要重写 loader、解释口径和结果分析

## 结果回填任务

### 10. 论文回填

任务：
- 新增真实 CoVe 小样本结果小节
- 保留类 CoVe 主线不动
- 明确说明真实 API 补充实验是“真实性验证”，不是主结果替代

涉及文件：

- `paper/zjuthesis/body/undergraduate/final/2-body.tex`
- `paper/zjuthesis/body/undergraduate/final/abstract.tex`
- `paper/zjuthesis/body/undergraduate/final/5-conclusion.tex`

### 11. 文档轻微更新

任务：
- 在完成真实 CoVe 后，同步更新待办与总指南

涉及文件：

- `OVERALL_TODO.md`
- `docs/EXPERIMENT_MASTER_GUIDE.md`

## 建议执行顺序

1. 设置限速环境变量
2. 跑 `smoke test`
3. 跑 `run_all.py` 的 100 样本真实 CoVe
4. 跑 `run_verifier_comparison.py` 的 100 样本真实 CoVe
5. 跑 `run_false_rejection_diagnostics.py` 的 100 样本真实 CoVe
6. 拉回结果
7. 重画图
8. 回填论文
9. 更新 TODO 和 guide
