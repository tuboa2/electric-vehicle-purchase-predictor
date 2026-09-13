import logging
from typing import Any, Callable
from evaluation.metrics import MetricRegistry

logger = logging.getLogger(__name__)


class CrossCompetitionTransferEvaluator:
    """
    Evaluates the portability and generalization of strategic heuristics
    across distinct competition environments.
    """

    def __init__(self, metric_name: str = "roc_auc"):
        self.metric_name = metric_name
        self.metric_fn, self.direction = MetricRegistry.get_metric(metric_name)
        self.maximize = self.direction == "maximize"

    def evaluate_strategy_transfer(
        self,
        strategy_name: str,
        source_domain: str,
        target_domain: str,
        baseline_score: float,
        transferred_score: float,
    ) -> dict[str, Any]:
        gain = (
            (transferred_score - baseline_score)
            if self.maximize
            else (baseline_score - transferred_score)
        )
        is_effective = gain >= 0.0005

        transfer_status = "POSITIVE_TRANSFER" if is_effective else ("NEGATIVE_TRANSFER" if gain < -0.0005 else "NEUTRAL")

        logger.info(
            "Transfer result for '%s' (%s -> %s): delta=%.5f, status=%s",
            strategy_name,
            source_domain,
            target_domain,
            gain,
            transfer_status,
        )

        return {
            "strategy_name": strategy_name,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "baseline_score": round(baseline_score, 5),
            "transferred_score": round(transferred_score, 5),
            "gain": round(gain, 5),
            "status": transfer_status,
            "recommend_promotion": is_effective,
        }
