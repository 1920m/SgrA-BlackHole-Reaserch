# Compile the research paper: pdflatex + BibTeX.
# Plain '=' on purpose: GNU make predefines TEX=tex as a builtin default,
# so '?=' would silently run plain TeX (which cannot read \documentclass).
TEX = pdflatex
BIB = bibtex

.PHONY: all thesis extension clean

all: main.pdf

thesis: thesis.pdf

main.pdf: main.tex refs.bib numbers.tex
	$(TEX) -interaction=nonstopmode -halt-on-error main.tex
	$(BIB) main
	$(TEX) -interaction=nonstopmode -halt-on-error main.tex
	$(TEX) -interaction=nonstopmode -halt-on-error main.tex

thesis.pdf: thesis.tex frontmatter.tex refs.bib numbers.tex $(wildcard chapters/*.tex)
	$(TEX) -interaction=nonstopmode -halt-on-error thesis.tex
	$(BIB) thesis
	$(TEX) -interaction=nonstopmode -halt-on-error thesis.tex
	$(TEX) -interaction=nonstopmode -halt-on-error thesis.tex

# Extension paper (separate document; bibliography = ../refs.bib + refs_ext.bib)
extension: extension/extension.pdf

EXT_DEPS = extension/extension.tex extension/extension-frontmatter.tex            extension/refs_ext.bib extension/numbers_ext.tex refs.bib numbers.tex            $(wildcard extension/chapters/*.tex) $(wildcard extension/tab_*.tex)
extension/extension.pdf: $(EXT_DEPS)
	cd extension && $(TEX) -interaction=nonstopmode -halt-on-error extension.tex
	cd extension && $(BIB) extension
	cd extension && $(TEX) -interaction=nonstopmode -halt-on-error extension.tex
	cd extension && $(TEX) -interaction=nonstopmode -halt-on-error extension.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.pdf \
	      thesis.aux thesis.bbl thesis.blg thesis.log thesis.out thesis.toc \
	      thesis.lof thesis.lot thesis.pdf
