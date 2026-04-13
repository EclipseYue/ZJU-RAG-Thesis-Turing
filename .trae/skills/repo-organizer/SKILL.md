---
name: "repo-organizer"
description: "重构仓库目录与解耦模块（paper/src/experiments/apps/docs/data）。用户要求项目结构调整、模块化、迁移文件与修复引用时调用。"
---

# Repo Organizer

## 目标

- 按职责划分目录：论文、代码、实验、数据、文档、应用
- 保持可运行：保留兼容入口脚本，修复 import 与路径
- 为每个目录补 README，降低上手成本

