"""SMBO for the shared forest search space, using Optuna's TPE sampler."""

from __future__ import annotations

from typing import Any

import optuna
from optuna.samplers import TPESampler

from random_forest import Config, Evaluator


def optimise_smbo(
    evaluator: Evaluator,
    n_trials: int,
    n_trees: int,
    seed: int,
) -> tuple[Config, Any]:
    """Use SMBO (Optuna TPE) to choose configurations from past evaluations.

    Trains each candidate with n_trees trees, uses up to n_trials evaluations,
    and selects the best configuration by the validation objective (balanced
    accuracy, higher is better). Returns the best config and the trial history.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        config = {
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 4, 16, 32]
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", 0.5, 1.0]
            ),
            "min_samples_leaf": trial.suggest_categorical(
                "min_samples_leaf", [1, 2, 4, 8]
            ),
        }
        result = evaluator(config, n_trees, seed)
        return result["objective"]

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_config = dict(study.best_params)

    history = [
        {
            "trial": t.number,
            "params": dict(t.params),
            "value": t.value,
        }
        for t in study.trials
    ]

    return best_config, history