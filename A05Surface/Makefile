PNGOBJS = Figure1cndragonback.png Figure1cndragonbackrot.png Figure1cndragonfront.png Figure1lucyback.png Figure1lucyfront.png Figure1statuetteback.png Figure1statuettefront.png Figure1statuettefrontrot.png Figure2bird.png Figure2cells.png Figure2konata.png Figure2lena.png Figure2objects.png Figure2triangle.png Figure30.png Figure3coif1.png Figure3coif2.png Figure3coif3.png Figure3coif4.png Figure3coif5.png Figure3db1.png Figure3db2.png Figure3db3.png Figure3db4.png Figure3db5.png Figure3haar1.png Figure3haar2.png Figure3haar3.png Figure3haar4.png Figure3haar5.png Figure3tv1.png Figure3tv2.png Figure3tv3.png Figure3tv4.png Figure3tv5.png Figure4bird.png Figure4cells.png Figure4konata.png Figure4lena.png Figure4objects.png Figure4triangle.png
TBLOBJS = Table1.tbl Table2.tbl Table3.tbl
RUNOBJS = Problem1.run Problem2.run Problem3Part1.run Problem3Part2.run Problem3Part3.run

.PHONY: all
all: Answer.pdf

Answer.pdf: Answer.tex Answer.bbl $(PNGOBJS) $(TBLOBJS) | Answer.aux
	lualatex Answer.tex

Answer.aux: Answer.tex Answer.bbl
	lualatex Answer.tex

Answer.bbl: Answer.tex
	lualatex Answer.tex
	biber Answer

Figure1cndragonback.png: $(wildcard Result1cndragon.mat)
	matlab -nodisplay -r "Isosurface Result1cndragon Figure1cndragonback 90.0 80.0 1.0 1.0 -1.0 1200 600; exit;"

Figure1cndragonfront.png: $(wildcard Result1cndragon.mat)
	matlab -nodisplay -r "Isosurface Result1cndragon Figure1cndragonfront 90.0 -80.0 1.0 1.0 -1.0 1200 600; exit;"

Figure1lucyback.png: $(wildcard Result1lucy.mat)
	matlab -nodisplay -r "Isosurface Result1lucy Figure1lucyback 270.0 10.0 1.0 1.0 -1.0 600 1200; exit;"

Figure1lucyfront.png: $(wildcard Result1lucy.mat)
	matlab -nodisplay -r "Isosurface Result1lucy Figure1lucyfront 90.0 10.0 1.0 1.0 -1.0 600 1200; exit;"

Figure1statuetteback.png: $(wildcard Result1statuette.mat)
	matlab -nodisplay -r "Isosurface Result1statuette Figure1statuetteback 90.0 -80.0 1.0 1.0 -1.0 600 1200; exit;"

Figure1statuettefront.png: $(wildcard Result1statuette.mat)
	matlab -nodisplay -r "Isosurface Result1statuette Figure1statuettefront 90.0 80.0 1.0 1.0 -1.0 600 1200; exit;"

.PRECIOUS: %rot.png
%rot.png: %.png
	convert -rotate "180" $*.png $*rot.png    

.SECONDEXPANSION:
$(TBLOBJS): $(wildcard $(RUNOBJS)) $(if $(wildcard Result.dir), Plot.py)
	python Plot.py

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py wcv.py surf.py utils/degrade.py
	python $<
	touch $@

.PHONY: report
report: Answer.pdf

.PHONY: run
run: $(RUNOBJS)
