LATEXMK := latexmk
UV := uv
VENV := .venv313
PYTHON := $(VENV)/bin/python
DATA_DIR := data

SLIDES_MAIN := slides/main.tex
POSTER_MAIN := poster/poster.tex
DIMENSIONAL_SCALING_DATA := $(DATA_DIR)/SDSS_dimensional_scaling.parquet
COMPOSITION_VARIATION_DATA := $(DATA_DIR)/composition_variation_experiment.parquet
DATA_FILES := $(DIMENSIONAL_SCALING_DATA) $(COMPOSITION_VARIATION_DATA)
SLIDES_GENERATED_FIGURES_DIR := slides/figures/generated
POSTER_GENERATED_FIGURES_DIR := poster/figures/generated

.PHONY: all slides poster watch watch-slides watch-poster clean clean-data venv data figures

all: figures slides poster

slides:
	$(LATEXMK) -cd -xelatex -interaction=nonstopmode -halt-on-error $(SLIDES_MAIN)

poster:
	rsvg-convert -f pdf -o $(POSTER_GENERATED_FIGURES_DIR)/BMFTR_Logo.pdf poster/figures/logo/BMFTR_Logo.svg
	rsvg-convert -f pdf -o $(POSTER_GENERATED_FIGURES_DIR)/lamarr.pdf poster/figures/logo/lamarr.svg
	$(LATEXMK) -cd -xelatex -interaction=nonstopmode -halt-on-error $(POSTER_MAIN)

venv: $(PYTHON)

$(PYTHON):
	$(UV) venv --python 3.13 $(VENV)
	$(UV) pip install --python $@ matplotlib seaborn pandas numpy pyarrow

data: $(DATA_FILES)

$(DIMENSIONAL_SCALING_DATA):
	mkdir -p $(@D)
	wget -O $@ https://raw.githubusercontent.com/maximilianberens/TeamIndex/main/data/SDSS_dimensional_scaling.parquet

$(COMPOSITION_VARIATION_DATA):
	mkdir -p $(@D)
	wget -O $@ https://raw.githubusercontent.com/maximilianberens/TeamIndex/main/data/composition_variation_experiment.parquet

figures: venv data
	$(PYTHON) slides/figures/python/create_figures.py
	$(PYTHON) poster/figures/python/create_figures.py

watch: watch-slides

watch-slides:
	$(LATEXMK) -cd -pvc -xelatex -interaction=nonstopmode -halt-on-error $(SLIDES_MAIN)

watch-poster:
	$(LATEXMK) -cd -pvc -xelatex -interaction=nonstopmode -halt-on-error $(POSTER_MAIN)

clean:
	$(LATEXMK) -cd -C $(SLIDES_MAIN)
	$(LATEXMK) -cd -C $(POSTER_MAIN)
	rm -rvf $(SLIDES_GENERATED_FIGURES_DIR) $(POSTER_GENERATED_FIGURES_DIR)

clean-data:
	rm -v $(DATA_FILES)
