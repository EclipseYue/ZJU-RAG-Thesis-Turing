---
name: "thesis-maintainer"
description: "维护论文 LaTeX（版式/引用/编译）。当用户要求修改开题/综述/译文、修复 LaTeX 报错或生成 PDF 时调用。"
---

# Thesis Maintainer

## 目标

- 修改 `paper/zjuthesis/` 下的论文内容与版式
- 修复编译问题（引用、目录、溢出、空白页等）
- 生成可交付 PDF

## 工作方式

- 优先在 `paper/zjuthesis/` 内定位文件与宏包配置
- 变更前先读取目标 `.tex/.bib` 文件的上下文，保持模板风格
- 修改后全量编译并检查日志中的报错与关键 warning

## 常用输出

- 修改点与对应文件位置
- 最新 PDF 路径

