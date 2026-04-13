# Tasks
- [x] Task 1: 编写真实评测脚本 `experiments/run_real_evaluation.py`：加载真实的 HotpotQA 数据（或小规模 N=50 以控制API成本/时间），引入 F1/EM 计算逻辑，涵盖6种消融变体。
- [x] Task 2: 运行评估脚本：执行 `run_real_evaluation.py` 并确保结果落盘至 `data/results/automated_ablation.json`。
- [x] Task 3: 更新可视化脚本 `experiments/plot_results.py`：读取新生成的数据并输出更新后的双轴图和折线图，包含 F1 指标的展示。
- [x] Task 4: 更新LaTeX论文 `paper/zjuthesis/body/undergraduate/final/2-body.tex`：将新跑出的数据和真实生成的 Case Study 补充到论文中。
- [x] Task 5: 编译论文：执行 `make` 确保 PDF 成功生成且无致命错误。

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 3
- Task 5 depends on Task 4