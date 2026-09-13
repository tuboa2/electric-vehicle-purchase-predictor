import logging
from typing import Any, Callable
import numpy as np
import pandas as pd
from evaluation.metrics import MetricRegistry

logger = logging.getLogger(__name__)


class AblationHarness:
    """
    Controlled Component Ablation Engine.
    Systematically isolates and removes components (features, transforms, architectures)
    to quantify true marginal contribution to out-of-fold CV.
    """

    def __init__(self, metric_name: str = "roc_auc"):
        self.metric_name = metric_name
        self.metric_fn, self.direction = MetricRegistry.get_metric(metric_name)
        self.maximize = self.direction == "maximize"

    def run_feature_ablation(
        self,
        evaluate_fn: Callable[[list[str]], float],
        feature_blocks: dict[str, list[str]],
        all_features: list[str],
    ) -> dict[str, Any]:
        """
        Conducts leave-one-block-out ablation:
        For each block, drops it from all_features and measures score delta.
        """
        # Baseline with full feature set
        logger.info("Computing full feature baseline score...")
        full_score = evaluate_fn(all_features)
        logger.info("Full pipeline CV: %.5f", full_score)

        results = []
        for block_name, block_cols in feature_blocks.items():
            ablated_cols = [c for c in all_features if c not in block_cols]
            if not ablated_cols:
                logger.warning("Skipping block %s: cannot ablate all features", block_name)
                continue

            logger.info("Evaluating ablation of block: %s (%d features)", block_name, len(block_cols))
            ablated_score = evaluate_fn(ablated_cols)

            # Marginal contribution: how much score drops when block is removed
            marginal_contrib = (
                (full_score - ablated_score) if self.maximize else (ablated_score - full_score)
            )

            status = "ESSENTIAL" if marginal_contrib > 0.0005 else ("NEUTRAL" if marginal_contrib >= -0.0002 else "HARMFUL")

            results.append({
                "block_name": block_name,
                "num_features": len(block_cols),
                "ablated_score": round(ablated_score, 5),
                "marginal_contribution": round(marginal_contrib, 5),
                "status": status,
            })

        return {
            "baseline_score": round(full_score, 5),
            "metric_name": self.metric_name,
            "direction": self.direction,
            "ablation_results": sorted(results, key=lambda x: x["marginal_contribution"], reverse=True),
        }
