# 运行完整多跳QA基准测试与独立消融验证 Spec

## Why
为将论文从工程报告提升为具备“研究范式意识”的学术论文，需要严谨的实验设计。先前实验虽然建立了框架，但缺乏对模块（Adaptive, CoVe）的独立贡献证明，也缺乏衡量大模型生成准确度的核心指标（如 F1 Score / Exact Match）。此外，需要真实的推断数据支撑定量分析与定性案例（Case Study）。

## What Changes
- 修改实验管线 `run_large_scale.py`（或新建真实运行脚本 `run_real_evaluation.py`），接入真实的 LLM 评测逻辑，计算 F1 / EM。
- 增加控制组：Baseline + Adaptive, Baseline + CoVe。
- 运行真实评估并输出结果至 `data/results/real_ablation.json`。
- 更新可视化绘图脚本 `plot_results.py` 适配 6 组变体和 F1 指标。
- 将生成的客观图表与真实案例同步更新至 LaTeX 论文。

## Impact
- Affected specs: N/A
- Affected code: `experiments/run_real_evaluation.py`, `experiments/plot_results.py`, `paper/zjuthesis/body/undergraduate/final/2-body.tex`

## ADDED Requirements
### Requirement: 独立消融控制组与准确率评测
系统必须在实验中包含 A(Baseline), A2(Base+Adaptive), A3(Base+CoVe), B(Hetero), C(Hetero+Adaptive), D(Full) 共6组配置，并在最后环节计算与标准答案的 F1/EM 匹配度。

#### Scenario: 成功生成多指标数据
- **WHEN** 运行评估脚本
- **THEN** 输出各配置下的 Token、延迟、拒答率及 F1 Score 矩阵。