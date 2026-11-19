# Higgs diffusion

**Open [report.html](report.html).** It is the complete onboarding guide: dataset,
graph construction, IC/LT mechanics, results and graphs, limitations, and the
implementation plan for GNNs and prompted agents. No server or setup is needed to
read it. The report is intentionally tracked so collaborators can read it immediately.

## Run

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required to regenerate results.

```sh
uv sync --locked
uv run python run.py
```

The runner executes the notebook in memory and rebuilds `report.html`; it keeps
notebook outputs empty to avoid storing two copies of the figures. One process,
one numerical-library thread by default, no GPU or API key. The first run reads
the full sparse social graph; unchanged samples are cached under `.cache/`.

To export slide-ready PNGs, CSVs, simulation draws, and a reproducibility manifest:

```sh
uv run python run.py --export
```

Exports go to ignored `results/` and are overwritten on the next export. Ordinary
runs only update the report and hidden cache. Copy the report/exports before
comparing experiments. Remove `.cache/` whenever you want; it regenerates.

## Where things live

```text
report.html                 Read this first; complete standalone guide and results
notebooks/analysis.ipynb     Edit experiment settings and plots here
diffusion/                 Three core modules (data, models, experiment)
data/                       Original compressed datasets (not tracked)
docs/                       Original proposal PDF and Word document
run.py                      Rebuild the report; optional --export
pyproject.toml + uv.lock     Reproducible Python environment
```

To work directly in the notebook, select `.venv/bin/python` as the kernel. It
locates the project from either the root or `notebooks/`. Change the configuration
cell to try another sample size/seed or simulation count. Read `diffusion/models.py`
for the IC/LT implementations, `experiment.py` for evaluation, and `data.py` for
sampling and labels.

## Dataset setup for a collaborator

Download these three files from [SNAP](https://snap.stanford.edu/data/higgs-twitter.html)
into `data/`, keeping the `.gz` compression:

- `higgs-activity_time.txt.gz`
- `higgs-social_network.edgelist.gz`
- `higgs-retweet_network.edgelist.gz`

They are already present in this working copy. They cover July 1–7, **2012**;
2015 is the dataset update date. Raw data, environments, caches, and optional
exports are excluded from Git. The test suite was removed to keep this starter
small; the report runner is the practical end-to-end verification command.

Implemented: IC/LT and simple controls. Next: a small GNN, then synthetic prompted
agents. The report explains what can be reused and what must change for each.
