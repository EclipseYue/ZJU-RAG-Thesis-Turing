# Experiment Presets

本目录用于存放 Route A 之后的新实验预设。

建议后续新增：

- `route_a_hotpotqa.json`
  成熟文本 baseline 的主实验预设
- `route_a_hybridqa.json`
  真实 text+table 异构任务预设

当前已经提供上述两个预设。它们先作为 Route A 的配置目标存在，不代表实验已经运行。

设计原则：

- 旧 `experiments/configs/*.json` 继续服务于历史原型与诊断实验
- 新 `experiments/presets/*.json` 服务于 Route A 正式主线
