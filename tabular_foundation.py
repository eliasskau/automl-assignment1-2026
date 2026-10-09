"""Pre-trained tabular foundation model (TabPFN) on the largest dataset."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from tabpfn import TabPFNClassifier

from data_loading import DataSplits


N_CONTEXT = 1_000    # labelled rows given to TabPFN as context


def run_foundation_model(splits: DataSplits, seed: int) -> Any:
    """In-context TabPFN with a fixed labelled-data budget, evaluated on the complete test set."""
    rng = np.random.default_rng(seed)

    X_train, y_train = splits.X_train, splits.y_train
    X_test, y_test = splits.X_test, splits.y_test

    # Context subsample
    n_ctx = min(N_CONTEXT, len(X_train))
    ctx_idx = rng.choice(len(X_train), size=n_ctx, replace=False)
    X_ctx = X_train.iloc[ctx_idx].to_numpy()
    y_ctx = y_train[ctx_idx]

    # Full heldout test set, same as the forest methods
    X_te = X_test.to_numpy()
    y_te = y_test

    clf = TabPFNClassifier(device="cpu", random_state=seed)

    start = perf_counter()
    clf.fit(X_ctx, y_ctx)
    fit_sec = perf_counter() - start
    y_proba = clf.predict_proba(X_te)
    y_pred = np.argmax(y_proba, axis=1)
    elapsed = perf_counter() - start

    return {
        "metrics": {
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
            "log_loss": float(log_loss(y_te, y_proba)),
        },
        "elapsed_sec": elapsed,
        "fit_sec": fit_sec,
        "predict_sec": elapsed - fit_sec,
        "n_context": n_ctx,
        "n_test": int(len(y_te)),
    }
