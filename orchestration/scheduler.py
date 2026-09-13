import logging
from typing import Any
from orchestration.blackboard import FilesystemBlackboard
from orchestration.budget import BudgetManager
from schemas.experiment import HypothesisSpec

logger = logging.getLogger(__name__)


class EVScheduler:
    """
    Expected Value (EV) Hypothesis Scheduler.
    Prioritizes and filters experimental hypotheses based on:
    EV = (Expected_Gain * Confidence * Info_Gain) / (Compute_Cost * (1 + Risk))
    """

    def __init__(
        self,
        blackboard: FilesystemBlackboard,
        budget_manager: BudgetManager,
        min_ev_threshold: float = 0.15,
        max_concurrent_runners: int = 2,
    ):
        self.blackboard = blackboard
        self.budget_manager = budget_manager
        self.min_ev_threshold = min_ev_threshold
        self.max_concurrent_runners = max_concurrent_runners

    def add_hypothesis(self, hypothesis: HypothesisSpec) -> None:
        queue = self.blackboard.read_queue()
        item = hypothesis.model_dump(mode="json")
        item["ev_score"] = hypothesis.ev.ev_score

        # Avoid duplicates
        existing_ids = {q.get("hypothesis_id") for q in queue}
        if hypothesis.hypothesis_id not in existing_ids:
            queue.append(item)
            self._sort_and_save(queue)
            logger.info("Hypothesis %s queued with EV: %.4f", hypothesis.hypothesis_id, item["ev_score"])
        else:
            logger.warning("Hypothesis %s already in queue", hypothesis.hypothesis_id)

    def _sort_and_save(self, queue: list[dict[str, Any]]) -> None:
        # Sort descending by EV score
        queue.sort(key=lambda x: x.get("ev_score", 0.0), reverse=True)
        self.blackboard.write_queue(queue)

    def pop_next_candidate(self, require_gpu: bool = True) -> dict[str, Any] | None:
        queue = self.blackboard.read_queue()
        if not queue:
            return None

        # Filter and pop the highest eligible candidate
        eligible_idx = None
        for i, item in enumerate(queue):
            ev_score = item.get("ev_score", 0.0)
            if ev_score < self.min_ev_threshold:
                logger.info("Skipping hypothesis %s: EV %.4f below threshold %.4f", item.get("hypothesis_id"), ev_score, self.min_ev_threshold)
                continue

            cost_hours = item.get("ev", {}).get("compute_cost", 0.5)
            if require_gpu and not self.budget_manager.can_allocate_gpu(cost_hours):
                logger.warning("Skipping hypothesis %s: insufficient GPU budget for %.2fh", item.get("hypothesis_id"), cost_hours)
                continue

            eligible_idx = i
            break

        if eligible_idx is not None:
            candidate = queue.pop(eligible_idx)
            self.blackboard.write_queue(queue)
            return candidate

        return None

    def get_queue_depth(self) -> int:
        return len(self.blackboard.read_queue())
