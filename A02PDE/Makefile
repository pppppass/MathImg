PNGOBJS = Figure01baboon0.png Figure01baboon1.png Figure01baboon2.png Figure01barbara0.png Figure01barbara1.png Figure01barbara2.png Figure01gradient0.png Figure01gradient1.png Figure01gradient2.png Figure01lena0.png Figure01lena1.png Figure01lena2.png Figure01radial0.png Figure01radial1.png Figure01radial2.png Figure01tsukasa0.png Figure01tsukasa1.png Figure01tsukasa2.png Figure02barbara0.png Figure02barbara1.png Figure02barbara2.png Figure02barbara3.png Figure02barbara4.png Figure02barbara5.png Figure02lena0.png Figure02lena1.png Figure02lena2.png Figure02lena3.png Figure02lena4.png Figure02lena5.png Figure02tsukasa0.png Figure02tsukasa1.png Figure02tsukasa2.png Figure02tsukasa3.png Figure02tsukasa4.png Figure02tsukasa5.png Figure03baboon0.png Figure03baboon1.png Figure03baboon2.png Figure03barbara0.png Figure03barbara1.png Figure03barbara2.png Figure03lena0.png Figure03lena1.png Figure03lena2.png Figure03tsukasa0.png Figure03tsukasa1.png Figure03tsukasa2.png Figure04barbara0.png Figure04barbara1.png Figure04barbara2.png Figure04barbara3.png Figure04barbara4.png Figure04barbara5.png Figure04lena0.png Figure04lena1.png Figure04lena2.png Figure04lena3.png Figure04lena4.png Figure04lena5.png Figure04tsukasa0.png Figure04tsukasa1.png Figure04tsukasa2.png Figure04tsukasa3.png Figure04tsukasa4.png Figure04tsukasa5.png Figure05baboon0.png Figure05baboon1.png Figure05baboon2.png Figure05barbara0.png Figure05barbara1.png Figure05barbara2.png Figure05lena0.png Figure05lena1.png Figure05lena2.png Figure05tsukasa0.png Figure05tsukasa1.png Figure05tsukasa2.png Figure06barbara0.png Figure06barbara1.png Figure06barbara2.png Figure06barbara3.png Figure06barbara4.png Figure06barbara5.png Figure06lena0.png Figure06lena1.png Figure06lena2.png Figure06lena3.png Figure06lena4.png Figure06lena5.png Figure06tsukasa0.png Figure06tsukasa1.png Figure06tsukasa2.png Figure06tsukasa3.png Figure06tsukasa4.png Figure06tsukasa5.png Figure07barbara0.png Figure07barbara1.png Figure07barbara2.png Figure07gradient0.png Figure07gradient1.png Figure07gradient2.png Figure07lena0.png Figure07lena1.png Figure07lena2.png Figure07peppers0.png Figure07peppers1.png Figure07peppers2.png Figure07triangle0.png Figure07triangle1.png Figure07triangle2.png Figure07tsukasa0.png Figure07tsukasa1.png Figure07tsukasa2.png Figure08lena0.png Figure08lena1.png Figure08lena2.png Figure08lena3.png Figure08lena4.png Figure08lena5.png Figure08tsukasa0.png Figure08tsukasa1.png Figure08tsukasa2.png Figure08tsukasa3.png Figure08tsukasa4.png Figure08tsukasa5.png Figure09lena1.png Figure09lena2.png Figure09tsukasa1.png Figure09tsukasa2.png Figure10lena1.png Figure10lena2.png Figure10tsukasa1.png Figure10tsukasa2.png Figure11barbara0.png Figure11barbara1.png Figure11barbara2.png Figure11gradient0.png Figure11gradient1.png Figure11gradient2.png Figure11lena0.png Figure11lena1.png Figure11lena2.png Figure11tsukasa0.png Figure11tsukasa1.png Figure11tsukasa2.png
PGFOBJS = Figure12.pgf
TBLOBJS = Table11.tbl Table12.tbl Table21.tbl Table22.tbl Table23.tbl Table31.tbl Table32.tbl Table41.tbl Table42.tbl Table43.tbl Table51.tbl Table52.tbl Table61.tbl Table62.tbl Table63.tbl Table71.tbl Table72.tbl Table81.tbl Table82.tbl Table91.tbl Table92.tbl
RUNOBJS = Problem1Part1.run Problem1Part2.run Problem2Part1.run Problem2Part2.run Problem2Part3.run Problem2Part4.run Problem2Part5.run Problem2Part6.run Problem3Part1.run Problem3Part2.run Problem3Part3.run Problem3Part4.run

.PHONY: all
all: Answer.pdf

Answer.pdf: Answer.tex Answer.bbl $(PNGOBJS) $(PGFOBJS) $(TBLOBJS) | Answer.aux
	lualatex Answer.tex

Answer.aux: Answer.tex Answer.bbl
	lualatex Answer.tex

Answer.bbl: Answer.tex
	lualatex Answer.tex
	biber Answer

.SECONDEXPANSION:
$(PGFOBJS) $(TBLOBJS): $(wildcard $(RUNOBJS)) $(if $(wildcard Result.dir), Plot.py)
	python Plot.py
	sed -i 's/<LABEL1~*>/$$ \\text{\\textsf{PSNR}} \/ 30 $$ (\\Si{dB})/' Figure12.pgf
	sed -i 's/<LABEL21~*>/\\texttt{tsukasa}/' Figure12.pgf
	sed -i 's/<LABEL22~*>/\\texttt{lena}/' Figure12.pgf

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py heat.py pm.py shock.py utils/degrade.py
	python $<
	touch $@

.PHONY: report
report: Answer.pdf

.PHONY: run
run: $(RUNOBJS)
