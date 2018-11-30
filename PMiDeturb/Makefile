.PHONY: all
all: Report.pdf

Report.pdf: Report.tex Report.bbl | Report.aux
	lualatex Report.tex

Report.aux: Report.tex Report.bbl
	lualatex Report.tex

Report.bbl: Report.tex Biblio.bib
	lualatex Report.tex
	biber Report

.PHONY: report
report: Report.pdf
