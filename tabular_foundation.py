"""Pre-trained tabular foundation model (TabPFN) on the largest dataset."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from tabpfn import TabPFNClassifier

from data_loading import DataSplits


N_CONTEXT = 1_000    # labelled rows given to TabPFN as context
N_TEST = 5_000       # test rows subsampled for feasibility


def run_foundation_model(splits: DataSplits, seed: int) -> Any:
    """In-context TabPFN with a fixed labelled-data budget and a subsampled test set."""
    rng = np.random.default_rng(seed)

    X_train, y_train = splits.X_train, splits.y_train
    X_test, y_test = splits.X_test, splits.y_test

    # Context subsample
    n_ctx = min(N_CONTEXT, len(X_train))
    ctx_idx = rng.choice(len(X_train), size=n_ctx, replace=False)
    X_ctx = X_train.iloc[ctx_idx].to_numpy()
    y_ctx = y_train[ctx_idx]

    # Test subsample
    n_test = min(N_TEST, len(X_test))
    test_idx = rng.choice(len(X_test), size=n_test, replace=False)
    X_te = X_test.iloc[test_idx].to_numpy()
    y_te = y_test[test_idx]

    clf = TabPFNClassifier(device="cpu")

    start = perf_counter()
    clf.fit(X_ctx, y_ctx)
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)
    elapsed = perf_counter() - start

    return {
        "metrics": {
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
            "log_loss": float(log_loss(y_te, y_proba)),
        },
        "elapsed_sec": elapsed,
        "n_context": n_ctx,
        "n_test": n_test,
        "test_idx": test_idx.tolist(),   # so you can re-evaluate forests on the same subset
    }