# Beamer 答辩 Slides

本科毕业设计答辩用 LaTeX Beamer 工程，复用论文 `paper/zjuthesis/figures/` 下的实验图。

## 编译

```bash
# 推荐：latexmk 一键编译
make

# 或手动
xelatex slides.tex
xelatex slides.tex   # 第二次以解决交叉引用 / 目录
```

需要 TeX Live / MiKTeX。若本地存在 `beamerthemeFormal.sty`，将优先使用 `Formal` 主题；否则回退到 `metropolis` / `Madrid`。

```bash
# 安装 metropolis（如未使用 Formal 主题）
tlmgr install beamertheme-metropolis pgfopts
```

## 目录结构

```
slides/
├── slides.tex         # 主文件（preamble + \input 各章）
├── Makefile           # 一键编译 / 讲义 / 清理
├── README.md
├── figures/           # 本工程独有图（如有）
└── sections/
    ├── 01_background.tex     # 研究背景与动机
    ├── 02_problem.tex        # 研究目标与创新点
    ├── 03_method.tex         # 方法与系统架构
    ├── 05_experiment.tex     # 实验设计
    ├── 06_results.tex        # 主结果
    ├── 07_analysis.tex       # 失败原因与案例分析
    ├── 08_conclusion.tex     # 结论与未来工作
    └── 09_qa.tex             # Q&A
```

## 字体说明

- 默认优先使用 Windows 系统字体：`Times New Roman`、`Aptos/Segoe UI`、`SimSun`、`Microsoft YaHei`。
- 如出现 `Font ... not found`，可在 `slides.tex` 显式指定 `\setmainfont`、`\setsansfont`、`\setCJKmainfont`。
