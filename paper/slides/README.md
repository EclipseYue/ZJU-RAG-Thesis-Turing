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

需要 TeX Live / MiKTeX，并安装 `metropolis` 主题（可选，缺失时自动回退到 Madrid 主题）。

```bash
# 安装 metropolis（一次性）
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
    ├── 02_problem.tex        # 研究问题、目标、创新点
    ├── 03_method.tex         # 方法与系统架构
    ├── 04_implementation.tex # 关键实现细节
    ├── 05_experiment.tex     # 实验设计
    ├── 06_results.tex        # 主消融与跨数据集结果
    ├── 07_analysis.tex       # 阈值/验证器/错误分析与案例
    ├── 08_conclusion.tex     # 结论与未来工作
    ├── 09_qa.tex             # 答辩感谢
    └── zz_appendix.tex       # 附录（备用 slide）
```

## 字体说明

- 主类 `ctex` 自动选择字体集（Windows 用宋体/黑体；macOS 用 PingFang）。
- 如出现 `Font ... not found`，可在 `slides.tex` 显式指定 `\setCJKmainfont{...}`。
