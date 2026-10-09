"""Optional shared forest helpers and student-completed predictive evaluation.

Reuse or replace these interfaces, including with package-native scoring and
search spaces. Unused helpers may be removed. Preserve the common
comparison and record the metric definitions and objective direction you use.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

Config = dict[str, Any]
Evaluator = Callable[[Config, int, int], dict[str, Any]]

# Example parallelism per forest; -1 uses all available CPU cores.
# Choose this for your hardware and report it when comparing runtimes.
N_JOBS = 4

# Optional starting point. Choose and justify a shared space and sampling rules,
# or use your optimiser package's search-space tools. Keep tree count separate.
SEARCH_SPACE = {
    "max_depth": (None, 4, 16, 32),
    "max_features": ("sqrt", 0.5, 1.0),
    "min_samples_leaf": (1, 2, 4, 8),
}


def sample_configuration(rng: np.random.Generator) -> Config:
    """TODO if using this helper: sample a legal configuration using rng.

    Return the hyperparameters to pass to RandomForestClassifier. Account for
    dependencies between parameters if you extend the example search space.
    """
    config = { 
        "max_depth" :SEARCH_SPACE["max_depth"]
[rng.integers(len(SEARCH_SPACE["max_depth"]))],
        "max_features": SEARCH_SPACE["max_features"]
[rng.integers(len(SEARCH_SPACE["max_features"]))],
        "min_samples_leaf": SEARCH_SPACE["min_samples_leaf"]
[rng.integers(len(SEARCH_SPACE["min_samples_leaf"]))],
    }
    return config


def make_classifier(config: Config, n_estimators: int, seed: int) -> RandomForestClassifier:
    """Use {} for the untuned baseline; omitted parameters keep library defaults.

    Tree count, seed, and parallelism are set here, outside the search space.
    Invalid configurations are left for scikit-learn to reject during fitting.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=N_JOBS,
        **config,
    )


def predictive_metrics(
    model: Any, X: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    y_pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
    }
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
        metrics["log_loss"] = float(log_loss(y, y_proba))
    return metrics


def validation_objective(metrics: dict[str, float]) -> float:
    """Return balanced accuracy as the objective. Higher is better."""
    return metrics["balanced_accuracy"]

def make_evaluator(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> Evaluator:
    """Evaluate on one split; timing includes model construction, fit, and scoring.

    The returned dictionary is an optional in-memory interface.
    """

    def evaluate(config: Config, n_trees: int, seed: int) -> dict[str, Any]:
        start = perf_counter()
        model = make_classifier(config, n_trees, seed)
        model.fit(X_train, y_train)
        metrics = predictive_metrics(model, X_valid, y_valid)
        objective = validation_objective(metrics)
        return {
            "configuration": dict(config),
            "metrics": metrics,
            "objective": float(objective),
            "n_trees": int(n_trees),
            "elapsed_sec": float(perf_counter() - start),
        }

    return evaluate


def final_test_evaluation(
    config: Config,
    n_estimators: int,
    seed: int,
    X_train_valid: np.ndarray,
    y_train_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Fit on all non-test data and score the test set, with the same timing scope."""

    start = perf_counter()
    model = make_classifier(config, n_estimators, seed)
    model.fit(X_train_valid, y_train_valid)
    metrics = predictive_metrics(model, X_test, y_test)
    return {
        "metrics": metrics,
        "elapsed_sec": float(perf_counter() - start),
    }
