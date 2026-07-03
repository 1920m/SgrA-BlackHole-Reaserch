# Compile the research paper: pdflatex + BibTeX.
TEX ?= pdflatex
BIB ?= bibtex

.PHONY: all clean

all: main.pdf

main.pdf: main.tex refs.bib
	$(TEX) -interaction=nonstopmode -halt-on-error main.tex
	$(BIB) main
	$(TEX) -interaction=nonstopmode -halt-on-error main.tex
	$(TEX) -interaction=nonstopmode -halt-on-error main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.pdf
