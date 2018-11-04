RUNOBJS = Problem21.run Problem22.run Problem23.run Problem24.run Problem25.run

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py tv.py
	python $<
	touch $@

.PHONY: run
run: $(RUNOBJS)
