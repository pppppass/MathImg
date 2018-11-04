PNGOBJS1 = Figure1barbara0.png Figure1barbara1.png Figure1barbara2.png\
	Figure1boats0.png Figure1boats1.png Figure1boats2.png Figure1gradient0.png\
	Figure1gradient1.png Figure1gradient2.png Figure1lena0.png Figure1lena1.png\
	Figure1lena2.png Figure1triangle0.png Figure1triangle1.png Figure1triangle2.png\
	Figure1tsukasa0.png Figure1tsukasa1.png Figure1tsukasa2.png
PNGOBJS2 = Figure2lena0.png Figure2lena1.png Figure2lena2.png Figure2lena3.png\
	Figure2lena4.png Figure2lena5.png Figure2tsukasa0.png Figure2tsukasa1.png\
	Figure2tsukasa2.png Figure2tsukasa3.png Figure2tsukasa4.png Figure2tsukasa5.png
PNGOBJS3 = Figure3lena1.png Figure3lena2.png Figure3lena3.png Figure3tsukasa1.png\
	Figure3tsukasa2.png Figure3tsukasa3.png
PNGOBJS4 = Figure4lena1.png Figure4lena2.png Figure4tsukasa1.png Figure4tsukasa2.png
PNGOBJS5 = Figure5lena1.png Figure5lena2.png Figure5lena3.png Figure5lena4.png\
	Figure5lena5.png Figure5tsukasa1.png Figure5tsukasa2.png Figure5tsukasa3.png\
	Figure5tsukasa4.png Figure5tsukasa5.png
PNGOBJS = $(PNGOBJS1) $(PNGOBJS2) $(PNGOBJS3) $(PNGOBJS4) $(PNGOBJS5)
TBLOBJS = Table11.tbl Table12.tbl Table21.tbl Table22.tbl Table51.tbl Table52.tbl
RUNOBJS = Problem21.run Problem22.run Problem23.run Problem24.run Problem25.run

.PHONY: all
all: Answer.pdf

Answer.pdf: Answer.tex Answer.bbl $(PNGOBJS) $(TBLOBJS) | Answer.aux
	lualatex Answer.tex

Answer.aux: Answer.tex Answer.bbl
	lualatex Answer.tex

Answer.bbl: Answer.tex
	lualatex Answer.tex
	biber Answer

$(TBLOBJS): $(wildcard $(RUNOBJS)) $(if $(wildcard Result.dir), Plot.py)
	python Plot.py

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py tv.py
	python $<
	touch $@

.PHONY: report
report: Answer.pdf

.PHONY: run
run: $(RUNOBJS)
