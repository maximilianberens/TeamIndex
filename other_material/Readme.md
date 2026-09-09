# Index Intersection for High-Dimensional Range Queries

Presentation slides and poster for the regular research paper ["Index Intersection for High-Dimensional Range Queries"](https://www.vldb.org/pvldb/vol19/p767-berens.pdf) held at the VLDB 2026 ([Program](https://vldb.org/2026/program.html)).


## Requirements

- XeLaTeX and `latexmk`
- `uv` (pip alternative, [link](https://docs.astral.sh/uv/))
- `rsvg-convert` command
- `wget`
- Python 3.13 (used via venv)

`make venv` automatically creates the shared Python environment at the repository root and installs the `matplotlib`, `seaborn`, `pandas`, `numpy`, and `pyarrow` Python packages.

## Build procedure

The Python environment (`.venv313`) and downloaded datasets (`data/`) are shared by the slides and poster projects at the repository root. Run all build commands from the repository root:

```sh
make venv
make data
make all
```

`make venv` creates `.venv313` with Python 3.13 and the required Python packages. `make data` creates `data/` and downloads the necessary datasets. `make figures` regenerates the Python figures supported by those datasets and imported by the slides and poster. `make slides` or `make poster` builds only that document. `make all` builds the figures, slides, and poster. `make clean` removes generated figures and both documents' build artifacts; `make clean-data` removes the shared downloaded datasets. `make watch-slides` and `make watch-poster` continuously rebuild their respective documents.
