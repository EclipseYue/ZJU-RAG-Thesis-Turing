# GPU 服务器迁移与真实 API 实验运行手册

本文档面向当前 **Route A + 历史消融壳并行** 的状态，目标是在迁移到有 GPU 的服务器后，用尽量稳定、可控的方式继续推进实验。

适用场景：

- 服务器负责本地检索、嵌入、重排等 GPU / CPU 密集步骤
- 生成与 CoVe 验证继续调用真实外部 API
- 需要同时兼顾 `Route A baseline` 和 `旧版主消融/诊断实验`

## 1. 总体原则

当前推荐把实验分成两条线并行推进：

### 1.1 Route A 正式主线

用途：

- 建立可信文本 baseline
- 逐步过渡到真实异构任务
- 作为后续论文正式主线的候选结果来源

当前入口：

- `experiments/run_route_a_baseline.py`

### 1.2 旧消融壳诊断线

用途：

- 复用现有 A / A2 / A3 / B / C / D 实验矩阵
- 小样本验证真实 API 条件下的 Adaptive / CoVe 行为
- 为论文已有章节补充对照与诊断

当前入口：

- `experiments/run_all.py`
- `experiments/run_verifier_comparison.py`
- `experiments/run_false_rejection_diagnostics.py`

结论：

- **Route A** 负责“可信 baseline 建设”
- **旧消融壳** 负责“模块行为诊断和论文过渡期补充实验”

## 2. 迁移前要同步什么

上服务器前，至少同步以下内容：

```text
src/rererank_v1/baselines/
src/rererank_v1/adapters/
experiments/run_route_a_baseline.py
experiments/presets/
experiments/configs/
docs/
requirements-route-a.txt
```

特别注意：

- `experiments/configs/local_api_overrides.json` 被 `.gitignore` 忽略，**不会自动跟随 git 同步**
- 离线数据目录也要单独同步，例如：
  - `data/datasets/hotpotqa/validation.json`
  - `data/datasets/2wikimultihopqa/validation.json`

推荐同步方式：

```bash
git pull
scp /本地路径/experiments/configs/local_api_overrides.json <server>:/path/to/repo/experiments/configs/local_api_overrides.json
scp -r /本地路径/data/datasets <server>:/path/to/repo/data/
```

## 3. 服务器初始化

进入仓库后，先做这几步：

```bash
cd /path/to/Rererank_v1
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  -r requirements-route-a.txt
```

检查 GPU 和 PyTorch：

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

检查 LlamaIndex：

```bash
.venv/bin/python -c "import llama_index; import llama_index.core; print('llamaindex ok')"
```

检查私有 API 配置是否存在：

```bash
ls experiments/configs/local_api_overrides.json
```

## 4. 真实 API 运行约束

服务器有 GPU，不代表可以无脑扩大真实 API 实验。

真实 API 仍会受以下约束：

- RPM / TPM / TPD 限制
- 长时重试带来的总时长放大
- 真实生成答案与 EM/F1 评测口径不匹配

因此建议：

- 先小样本 smoke，再扩大
- 所有真实 API 实验都显式保留结果文件名
- 旧消融入口优先使用 `--only-configs` 和 `--checkpoint-every`

建议先设置限速环境变量：

```bash
export RERERANK_LLM_MIN_INTERVAL_SEC=3.5
export RERERANK_LLM_MAX_RETRIES=8
export RERERANK_LLM_BACKOFF_BASE_SEC=8
export RERERANK_LLM_BACKOFF_MAX_SEC=90
```

## 5. 推荐执行顺序

## 5.1 第一阶段：验证 Route A 在服务器可跑

### Step 1. Route A 检索 smoke

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 20 \
  --generator-backend heuristic \
  --output-name route_a_hotpotqa_server_smoke.json
```

目标：

- 确认离线数据正常
- 确认 LlamaIndex / embedding / 检索链能在服务器上跑通

### Step 2. Route A 真实生成 smoke

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 20 \
  --output-name route_a_hotpotqa_realapi_smoke.json
```

说明：

- 若 `local_api_overrides.json` 已配置真实生成端，会直接走真实 API
- 若没有，则会回退到默认逻辑

目标：

- 检查真实生成是否仍有空答案、长句复述或极端慢速问题

### Step 3. Route A 真实生成小规模实验

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_hotpotqa.json \
  --samples 100 \
  --output-name route_a_hotpotqa_realapi_100.json
```

目标：

- 形成第一版 Route A 正式 baseline 候选结果

## 5.2 第二阶段：旧消融壳做小样本诊断

这一阶段不要再直接跑整套全量矩阵。

### Step 4. 旧 baseline 单配置 smoke

```bash
.venv/bin/python experiments/run_all.py \
  --config experiments/configs/ablation_with_controls.json \
  --samples 20 \
  --only-configs A_Baseline \
  --checkpoint-every 2 \
  --output-name automated_ablation_smoke_a_server.json
```

目标：

- 验证旧实验壳在服务器上结果落盘稳定
- 复查真实生成是否仍出现异常答案格式

### Step 5. 真实 CoVe 关键配置 smoke

```bash
.venv/bin/python experiments/run_all.py \
  --config experiments/configs/ablation_with_controls.json \
  --samples 20 \
  --real-cove \
  --only-configs A3_Baseline_CoVe D_CoVe_Full \
  --checkpoint-every 2 \
  --output-name automated_ablation_smoke_cove_server.json
```

目标：

- 只测试真正依赖 CoVe 的关键配置
- 不要在 smoke 阶段扫完整个 A/A2/A3/B/C/D

### Step 6. 真实 CoVe 小规模诊断

```bash
.venv/bin/python experiments/run_verifier_comparison.py \
  --config experiments/configs/verifier_comparison.json \
  --samples 100 \
  --real-cove \
  --output-name verifier_comparison_real_cove_100.json
```

```bash
.venv/bin/python experiments/run_false_rejection_diagnostics.py \
  --config experiments/configs/false_rejection_diagnostics.json \
  --samples 100 \
  --real-cove \
  --output-name false_rejection_diagnostics_real_cove_100.json
```

目标：

- 生成真实 verifier 条件下的 FRR / unsafe accept 诊断数据

## 6. 针对“刚才消融实验”的具体后续建议

结合最近的 `smoke_a / smoke_a_v2 / smoke_a_v3` 结果，当前更合理的推进方式是：

### 6.1 不要立刻扩大旧主消融样本量

原因：

- 最近已经暴露出真实生成答案格式仍不够稳定
- 继续扩大旧矩阵样本量，收益不如先把 Route A baseline 做稳

### 6.2 旧消融只保留“诊断角色”

优先保留的配置：

- `A_Baseline`
- `A3_Baseline_CoVe`
- `D_CoVe_Full`

用途：

- `A_Baseline`：检查真实生成输出质量
- `A3_Baseline_CoVe`：检查 verifier 单独影响
- `D_CoVe_Full`：检查完整路径下的拒答/误拒答

### 6.3 新主线优先转向 Route A

建议你把后续主要精力放在：

1. Route A 文本 baseline
2. Route A + 真实生成
3. Route A 上回挂 Adaptive / CoVe

而不是继续在旧自研 baseline 上深挖分数。

## 7. 结果文件建议命名

为避免混淆，建议在服务器上统一使用如下命名：

### Route A

- `route_a_hotpotqa_server_smoke.json`
- `route_a_hotpotqa_realapi_smoke.json`
- `route_a_hotpotqa_realapi_100.json`

### 旧消融壳

- `automated_ablation_smoke_a_server.json`
- `automated_ablation_smoke_cove_server.json`
- `verifier_comparison_real_cove_100.json`
- `false_rejection_diagnostics_real_cove_100.json`

## 8. 拉回本地后的动作

服务器实验完成后，建议按以下顺序处理：

1. 拉回 `data/results/*.json`
2. 先做简要分析，不急着全量改论文
3. 只有当 Route A 结果达到可解释区间时，再决定是否回填正文主线
4. 旧消融壳结果优先用于：
   - 诊断表
   - 失败案例
   - 真实 verifier 对比段落

## 9. 近期推荐路线

如果时间和 API 配额都有限，当前最优先顺序是：

1. `run_route_a_baseline.py` 20 条 heuristic smoke
2. `run_route_a_baseline.py` 20 条真实生成 smoke
3. `run_route_a_baseline.py` 100 条真实生成小规模实验
4. `run_all.py` 仅 `A_Baseline` 20 条
5. `run_all.py` 仅 `A3_Baseline_CoVe / D_CoVe_Full` 20 条
6. `run_verifier_comparison.py` 100 条真实 CoVe
7. `run_false_rejection_diagnostics.py` 100 条真实 CoVe

一句话总结：

**GPU 服务器优先用来把 Route A baseline 跑稳，旧消融壳只做小样本真实 API 诊断，不再当主线全量推进。**
