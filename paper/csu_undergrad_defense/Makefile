# =============================================================
#  Beamer Slides 构建脚本（XeLaTeX）
#  用法：
#    make           # 编译为 slides.pdf
#    make handout   # 编译为讲义版（无 overlay）
#    make clean     # 清理临时文件
#    make distclean # 清理所有产物（含 PDF）
# =============================================================

LATEX     = xelatex
LATEXMK   = latexmk
MAIN      = slides
ENGINE    = -xelatex
LATEXOPTS = -interaction=nonstopmode -halt-on-error -shell-escape

.PHONY: all pdf handout clean distclean watch

all: pdf

pdf:
	$(LATEXMK) $(ENGINE) $(LATEXOPTS) $(MAIN).tex

# 讲义版：把 \documentclass 添加 handout 选项
handout:
	$(LATEX) $(LATEXOPTS) "\PassOptionsToClass{handout}{beamer}\input{$(MAIN).tex}"
	$(LATEX) $(LATEXOPTS) "\PassOptionsToClass{handout}{beamer}\input{$(MAIN).tex}"
	mv $(MAIN).pdf $(MAIN)-handout.pdf

watch:
	$(LATEXMK) $(ENGINE) $(LATEXOPTS) -pvc $(MAIN).tex

clean:
	$(LATEXMK) -c $(MAIN).tex
	-rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	-rm -f $(MAIN).pdf $(MAIN)-handout.pdf
