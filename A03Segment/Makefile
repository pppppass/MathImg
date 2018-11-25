PNGOBJS = Figure01bird01.png Figure01bird02.png Figure01bird03.png Figure01bird04.png Figure01bird05.png Figure01bird06.png Figure01bird07.png Figure01bird08.png Figure01bird09.png Figure01bird10.png Figure01cells01.png Figure01cells02.png Figure01cells03.png Figure01cells04.png Figure01cells05.png Figure01cells06.png Figure01cells07.png Figure01cells08.png Figure01cells09.png Figure01cells10.png Figure01konata01.png Figure01konata02.png Figure01konata03.png Figure01konata04.png Figure01konata05.png Figure01konata06.png Figure01konata07.png Figure01konata08.png Figure01konata09.png Figure01konata10.png Figure01lena01.png Figure01lena02.png Figure01lena03.png Figure01lena04.png Figure01lena05.png Figure01lena06.png Figure01lena07.png Figure01lena08.png Figure01lena09.png Figure01lena10.png Figure01objects01.png Figure01objects02.png Figure01objects03.png Figure01objects04.png Figure01objects05.png Figure01objects06.png Figure01objects07.png Figure01objects08.png Figure01objects09.png Figure01objects10.png Figure01triangle01.png Figure01triangle02.png Figure01triangle03.png Figure01triangle04.png Figure01triangle05.png Figure01triangle06.png Figure01triangle07.png Figure01triangle08.png Figure01triangle09.png Figure01triangle10.png Figure02bird.png Figure02cells.png Figure02konata.png Figure02lena.png Figure02objects.png Figure02triangle.png Figure03bird01.png Figure03bird02.png Figure03bird03.png Figure03bird04.png Figure03bird05.png Figure03bird06.png Figure03bird07.png Figure03bird08.png Figure03bird09.png Figure03bird10.png Figure03triangle01.png Figure03triangle02.png Figure03triangle03.png Figure03triangle04.png Figure03triangle05.png Figure03triangle06.png Figure03triangle07.png Figure03triangle08.png Figure03triangle09.png Figure03triangle10.png Figure04bird01.png Figure04bird02.png Figure04bird03.png Figure04bird04.png Figure04bird05.png Figure04bird06.png Figure04bird07.png Figure04bird08.png Figure04bird09.png Figure04bird10.png Figure04triangle01.png Figure04triangle02.png Figure04triangle03.png Figure04triangle04.png Figure04triangle05.png Figure04triangle06.png Figure04triangle07.png Figure04triangle08.png Figure04triangle09.png Figure04triangle10.png Figure05bird.png Figure05triangle.png Figure06bird.png Figure06triangle.png
TBLOBJS = Table1.tbl Table2.tbl
RUNOBJS = Problem1.run Problem2.run Problem3.run Problem4.run

.PHONY: all
all: Answer.pdf

Answer.pdf: Answer.tex Answer.bbl $(PNGOBJS) $(TBLOBJS) | Answer.aux
	lualatex Answer.tex

Answer.aux: Answer.tex Answer.bbl
	lualatex Answer.tex

Answer.bbl: Answer.tex
	lualatex Answer.tex
	biber Answer

.SECONDEXPANSION:
$(PGFOBJS) $(TBLOBJS): $(wildcard $(RUNOBJS)) $(if $(wildcard Result.dir), Plot.py)
	python Plot.py

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py gac.py cv.py utils/degrade.py utils/downsamp.py
	python $<
	touch $@

.PHONY: report
report: Answer.pdf

.PHONY: run
run: $(RUNOBJS)
