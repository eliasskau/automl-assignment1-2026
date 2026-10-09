"""Trains forests on the same 1000 covertype rows that TabPFN gets as context,
so TabPFN and the forests can be compared with the same amount of labelled data.

    python foundation_budget.py --seeds 0 1 2
"""

import argparse
import json
import os
from time import perf_counter

import numpy as np

from data_loading import load_and_split, make_preprocessor
from random_forest import make_classifier, predictive_metrics
from tabular_foundation import N_CONTEXT

# best config found on covertype in the main study
TUNED = {"max_depth": None, "max_features": 1.0, "min_samples_leaf": 1}
MAX_TREES = 81


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--out", default="results/foundation_budget.jsonl")
    args = parser.parse_args()

    splits = load_and_split("covertype", args.cache_dir, None, args.split_seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f = open(args.out, "w")
    for seed in args.seeds:
        # same rows as tabular_foundation.py picks for this seed
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(splits.X_train), size=N_CONTEXT, replace=False)
        X_ctx = splits.X_train.iloc[idx]
        y_ctx = splits.y_train[idx]
        pre = make_preprocessor(X_ctx)
        X_fit = np.asarray(pre.fit_transform(X_ctx), dtype=np.float32)
        X_test = np.asarray(pre.transform(splits.X_test), dtype=np.float32)

        for name, config in [("default", {}), ("tuned", TUNED)]:
            start = perf_counter()
            model = make_classifier(config, MAX_TREES, seed)
            model.fit(X_fit, y_ctx)
            metrics = predictive_metrics(model, X_test, splits.y_test)
            elapsed = perf_counter() - start
            row = {"dataset": "covertype", "method": "forest_" + name + "_" + str(N_CONTEXT), "seed": seed,
                   "configuration": config, "n_context": N_CONTEXT, "n_test": len(splits.y_test),
                   "metrics": metrics, "elapsed_sec": elapsed}
            f.write(json.dumps(row) + "\n")
            print("seed", seed, name, "bal_acc", round(metrics["balanced_accuracy"], 4),
                  "acc", round(metrics["accuracy"], 4), round(elapsed, 1), "s")
    f.close()


if __name__ == "__main__":
    main()
