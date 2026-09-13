import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from evaluation.metrics import MetricRegistry

logger = logging.getLogger(__name__)


class EnsembleBlender:
    """
    Caruana-style Forward Ensemble Selection with Replacement (Hill Climbing).
    Finds optimal mixture weights for verified candidate models.
    """

    @staticmethod
    def hill_climbing(
        oof_dict: dict[str, np.ndarray],
        y_true: np.ndarray,
        metric_name: str = "roc_auc",
        n_iterations: int = 100,
        min_delta: float = 0.0001,
    ) -> tuple[dict[str, float], float]:
        metric_fn, direction = MetricRegistry.get_metric(metric_name)
        maximize = direction == "maximize"

        model_names = list(oof_dict.keys())
        m = len(model_names)
        if m == 0:
            raise ValueError("OOF candidate pool cannot be empty for ensembling")

        oof_matrix = np.column_stack([oof_dict[name] for name in model_names])

        # 1. Select the single best model as initial state
        single_scores = [metric_fn(y_true, oof_matrix[:, i]) for i in range(m)]
        best_single_idx = int(np.argmax(single_scores) if maximize else np.argmin(single_scores))

        current_ensemble_pred = oof_matrix[:, best_single_idx].copy()
        current_score = single_scores[best_single_idx]
        selected_indices = [best_single_idx]

        # 2. Greedy forward hill-climbing with replacement
        for step in range(1, n_iterations):
            best_step_score = current_score
            best_candidate_idx = None

            for i in range(m):
                trial_pred = (current_ensemble_pred * step + oof_matrix[:, i]) / (step + 1)
                trial_score = metric_fn(y_true, trial_pred)

                is_improved = (
                    (trial_score > best_step_score + min_delta)
                    if maximize
                    else (trial_score < best_step_score - min_delta)
                )

                if is_improved:
                    best_step_score = trial_score
                    best_candidate_idx = i

            if best_candidate_idx is None:
                # No candidate improves the ensemble score further
                break

            selected_indices.append(best_candidate_idx)
            current_ensemble_pred = (current_ensemble_pred * step + oof_matrix[:, best_candidate_idx]) / (step + 1)
            current_score = best_step_score

        # 3. Calculate normalized weights
        total_selected = len(selected_indices)
        weights = {}
        for i, name in enumerate(model_names):
            cnt = selected_indices.count(i)
            if cnt > 0:
                weights[name] = round(cnt / total_selected, 4)

        return weights, float(current_score)

    @classmethod
    def blend_test_predictions(
        cls,
        test_dict: dict[str, np.ndarray],
        weights: dict[str, float],
    ) -> np.ndarray:
        blend_pred = None
        for name, w in weights.items():
            if name in test_dict:
                pred = test_dict[name] * w
                blend_pred = pred if blend_pred is None else (blend_pred + pred)
            else:
                logger.warning("Model %s in blend weights not present in test predictions", name)

        if blend_pred is None:
            raise ValueError("No matching models found between blend weights and test dict")

        return blend_pred

    @classmethod
    def evaluate_gate_5(
        cls,
        best_single_cv: float,
        blended_cv: float,
        metric_name: str = "roc_auc",
        min_gain: float = 0.0005,
    ) -> tuple[bool, float]:
        _, direction = MetricRegistry.get_metric(metric_name)
        maximize = direction == "maximize"

        gain = (blended_cv - best_single_cv) if maximize else (best_single_cv - blended_cv)
        passed = gain >= min_gain
        return passed, gain
