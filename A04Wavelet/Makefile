PNGOBJS = Figure1barbara0.png Figure1barbara1.png Figure1barbara2.png Figure1boats0.png Figure1boats1.png Figure1boats2.png Figure1gradient0.png Figure1gradient1.png Figure1gradient2.png Figure1lena0.png Figure1lena1.png Figure1lena2.png Figure1triangle0.png Figure1triangle1.png Figure1triangle2.png Figure1tsukasa0.png Figure1tsukasa1.png Figure1tsukasa2.png Figure2barbara0.png Figure2barbara1.png Figure2barbara2.png Figure2boats0.png Figure2boats1.png Figure2boats2.png Figure2gradient0.png Figure2gradient1.png Figure2gradient2.png Figure2lena0.png Figure2lena1.png Figure2lena2.png Figure2triangle0.png Figure2triangle1.png Figure2triangle2.png Figure2tsukasa0.png Figure2tsukasa1.png Figure2tsukasa2.png Figure3lena0.png Figure3lena1.png Figure3lena2.png Figure3lena3.png Figure3lena4.png Figure3lena5.png Figure3tsukasa0.png Figure3tsukasa1.png Figure3tsukasa2.png Figure3tsukasa3.png Figure3tsukasa4.png Figure3tsukasa5.png Figure4lena0.png Figure4lena1.png Figure4lena2.png Figure4lena3.png Figure4lena4.png Figure4lena5.png Figure4tsukasa0.png Figure4tsukasa1.png Figure4tsukasa2.png Figure4tsukasa3.png Figure4tsukasa4.png Figure4tsukasa5.png Figure5lena1.png Figure5lena2.png Figure5lena3.png Figure5lena4.png Figure5lena5.png Figure5lena6.png Figure5tsukasa1.png Figure5tsukasa2.png Figure5tsukasa3.png Figure5tsukasa4.png Figure5tsukasa5.png Figure5tsukasa6.png Figure6lena0.png Figure6lena1.png Figure6lena2.png Figure6lena3.png Figure6lena4.png Figure6lena5.png Figure6tsukasa0.png Figure6tsukasa1.png Figure6tsukasa2.png Figure6tsukasa3.png Figure6tsukasa4.png Figure6tsukasa5.png Figure7lena0.png Figure7lena1.png Figure7lena2.png Figure7lena3.png Figure7lena4.png Figure7lena5.png Figure7tsukasa0.png Figure7tsukasa1.png Figure7tsukasa2.png Figure7tsukasa3.png Figure7tsukasa4.png Figure7tsukasa5.png Figure8lena0.png Figure8lena1.png Figure8lena2.png Figure8tsukasa0.png Figure8tsukasa1.png Figure8tsukasa2.png Figure9barbara2.png Figure9boats2.png Figure9gradient2.png Figure9lena2.png Figure9triangle2.png Figure9tsukasa2.png
TBLOBJS = Table11.tbl Table12.tbl Table21.tbl Table22.tbl Table31.tbl Table32.tbl Table41.tbl Table42.tbl Table51.tbl Table52.tbl Table61.tbl Table62.tbl Table71.tbl Table72.tbl
RUNOBJS = Problem1Part1.run Problem1Part2.run Problem1Part3.run Problem1Part4.run Problem1Part5.run Problem2Part1.run Problem2Part2.run Problem2Part3.run Test1.run Test2.run Test3.run

.PHONY: all
all: Answer.pdf

Answer.pdf: Answer.tex $(PNGOBJS) $(TBLOBJS) | Answer.aux
	lualatex Answer.tex

Answer.aux: Answer.tex
	lualatex Answer.tex

.SECONDEXPANSION:
$(PGFOBJS) $(TBLOBJS): $(wildcard $(RUNOBJS)) $(if $(wildcard Result.dir), Plot.py)
	python Plot.py

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py analysis.py balance.py utils/degrade.py
	python $<
	touch $@

.PHONY: report
report: Answer.pdf

.PHONY: run
run: $(RUNOBJS)
