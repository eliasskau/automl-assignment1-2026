"""Optional Hyperband interface for random forests.

Implement the method, connect a suitable package, or replace this interface.
Choose and justify the allocation schedule and how you retain search results.
"""

from __future__ import annotations

from typing import Any

from random_forest import Config, Evaluator, sample_configuration

import numpy as np
import math


def optimise_hyperband(
    evaluator: Evaluator,
    min_trees: int,
    max_trees: int,
    seed: int,
    reduction_factor: int = 3,
) -> tuple[Config, Any]:
    """TODO: implement or configure multiple successive-halving brackets.

    Use the shared search space. Start brackets with different numbers of
    configurations and trees per forest, between min_trees and max_trees.
    At each stage, keep the better configurations and give them more trees,
    keeping other settings fixed. Use reduction_factor for the decrease in
    configuration count and increase in trees.
    Compare validation objectives consistently: respect whether higher or lower
    values are better. Explain your schedule, refitting or warm starts, and
    how validation results determine the final selection.
    Return the selected configuration and results needed for your analysis.
    """

    rng = np.random.default_rng(seed)
    eta = reduction_factor
    history = []
    s_max = math.floor(math.log(max_trees / min_trees, eta))

    for s in range(s_max, -1, -1):
        n = math.ceil((s_max + 1) / (s + 1) * eta ** s)
        r = max_trees / eta ** s
        configs = [sample_configuration(rng) for x in range(n)]

        for i in range (s+1):
            n_i = math.floor(n / eta ** i)
            r_i = round (r * eta ** i)
            results = []
            for config in configs:
                result = evaluator(config, r_i, seed)
                result["bracket"] = s
                result["stage"] = i
                results.append(result)
                history.append(result)
            results.sort(key=lambda res: res["objective"], reverse=True)
            keep = math.floor(n_i / eta)
            configs = [res["configuration"] for res in results[:keep]]
    final = [h for h in history if h["n_trees"] == max_trees]
    best = max(final, key=lambda h: h["objective"])
    return best["configuration"], history