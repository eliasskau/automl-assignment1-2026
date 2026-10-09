# AutoML Assignment 1 - Hyperparameter optimisation of random forests

Compares an untuned random forest with forests tuned by Random Search, SMBO (Optuna TPE)
and Hyperband on five OpenML datasets, plus TabPFN on covertype.

## Setup

Needs Python 3.14 and uv. In the repo folder:

```
uv sync
```

TabPFN needs a one-time licence: make an account at https://ux.priorlabs.ai/account, accept the
licence, and save your API key in `~/.cache/tabpfn/auth_token` (the browser login does not work on Windows).

## Files

- `data_loading.py` - downloads the OpenML datasets, 60/20/20 stratified split, preprocessing
- `random_forest.py` - search space, sampler, metrics (balanced accuracy, accuracy, log loss), validation objective
- `random_search.py` - Random Search
- `smbo.py` - SMBO with Optuna's TPE sampler
- `hyperband.py` - Hyperband, number of trees is the resource
- `tabular_foundation.py` - TabPFN on covertype
- `experiment.py` - runs everything and saves results to `results/`
- `foundation_budget.py` - forests trained on the same 1000 rows as TabPFN
- `make_figures.py` - tables and figures for the report

Settings: search space and `N_JOBS` in `random_forest.py`, budgets per profile in `experiment.py`,
TabPFN context size in `tabular_foundation.py`.

## Pipeline check

```
uv run python experiment.py --dataset breast-w --profile smoke --seed 0
```

## Main study

Course profile: all rows, max 81 trees, 16 trials for Random Search and SMBO, Hyperband min 3 trees
and reduction factor 3, seeds 0 1 2.

```
uv run python experiment.py --dataset all --profile course --seed 0
uv run python experiment.py --dataset all --profile course --seed 1
uv run python experiment.py --dataset all --profile course --seed 2
```

Every run writes `results/run_course_<timestamp>.jsonl` (one line per dataset and method with the
config, search history, search time and test metrics) and a `meta_*.json` with the settings.

TabPFN on covertype (about 35 min per seed on CPU):

```
uv run python experiment.py --dataset covertype --profile course --methods foundation --seed 0
uv run python foundation_budget.py --seeds 0 1 2
```

## Tables and figures

```
uv run python make_figures.py
```

Prints the tables (mean and std over seeds) and writes `figures/progress.pdf`. The tables in the report were typed from that output; the foundation table from the TabPFN run output and `results/foundation_budget.jsonl`.

## Hardware

Forest study and TabPFN: AMD Ryzen 5 7600X (6 cores / 12 threads), 15 GB RAM, Windows 11,
`N_JOBS = 4` per forest, the three seeds were run at the same time.
