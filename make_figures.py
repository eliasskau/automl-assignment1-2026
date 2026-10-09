"""Prints the result tables (mean +- std over seeds) and saves the progress figure.

    python make_figures.py
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["default", "random", "smbo", "hyperband"]
NAMES = {"default": "Untuned", "random": "Random Search", "smbo": "SMBO (TPE)", "hyperband": "Hyperband"}
COLORS = {"default": "grey", "random": "tab:blue", "smbo": "tab:orange", "hyperband": "tab:green"}
DATASETS = ["breast-w", "credit-g", "phoneme", "electricity", "covertype"]
MAX_TREES = 81

os.makedirs("figures", exist_ok=True)

# load every run of the course profile
runs = []
for path in sorted(glob.glob("results/run_course_*.jsonl")):
    for line in open(path):
        r = json.loads(line)
        if r["method"] in METHODS:
            runs.append(r)

# one row per run, then mean and std over the seeds
rows = []
for r in runs:
    rows.append({"dataset": r["dataset"], "method": r["method"], "seed": r["seed"],
                 "test_bal_acc": r["final_result"]["metrics"]["balanced_accuracy"],
                 "test_acc": r["final_result"]["metrics"]["accuracy"],
                 "search_sec": r["search_seconds"]})
df = pd.DataFrame(rows)
df.to_csv("figures/summary_per_seed.csv", index=False)

pd.set_option("display.width", 200)
for column in ["test_bal_acc", "test_acc", "search_sec"]:
    print("\n==", column, "(mean / std over seeds) ==")
    table = df.pivot_table(index="dataset", columns="method", values=column, aggfunc=["mean", "std"])
    print(table.reindex(DATASETS).round(3))


def best_so_far(run):
    # x = trees trained so far (in full forests), y = best validation score so far
    hist = run["history"]
    # older smbo results only have "value" per trial, every trial was a full forest
    scores = [h["objective"] if "objective" in h else h["value"] for h in hist]
    trees = [h.get("n_trees", MAX_TREES) for h in hist]
    return np.cumsum(trees) / MAX_TREES, np.maximum.accumulate(scores)


# progress figure: best validation score against trees trained, one panel per dataset
fig, axes = plt.subplots(1, len(DATASETS), figsize=(15.5, 2.9))
for ax, d in zip(axes, DATASETS):
    for m in METHODS:
        seeds = [r for r in runs if r["dataset"] == d and r["method"] == m]
        if m == "default":
            base = np.mean([r["history"][0]["objective"] for r in seeds])
            ax.axhline(base, color=COLORS[m], ls="--", lw=1.5, label=NAMES[m])
            continue
        curves = [best_so_far(r) for r in seeds]
        grid = np.linspace(min(x[0] for x, y in curves), max(x[-1] for x, y in curves), 200)
        ys = np.array([np.interp(grid, x, y, left=np.nan) for x, y in curves])
        ax.plot(grid, np.nanmean(ys, axis=0), color=COLORS[m], lw=2, label=NAMES[m])
        ax.fill_between(grid, np.nanmin(ys, axis=0), np.nanmax(ys, axis=0), color=COLORS[m], alpha=0.15, lw=0)
    ax.set_title(d, fontsize=10)
    ax.set_xlabel("full-forest equivalents trained")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("best validation balanced acc.")
axes[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("figures/progress.pdf")
fig.savefig("figures/progress.png", dpi=150)
