import json
import os
import time
from pathlib import Path
from typing import Any
from schemas.state import BudgetState


class BudgetManager:
    """
    Manages Kaggle compute resources and submission quotas.
    Guarantees hard protection against quota exhaustion and session timeouts.
    """

    def __init__(
        self,
        budget_file: str | Path = "experiments/blackboard/budget.json",
        max_weekly_gpu_hours: float = 28.0,
        max_session_hours: float = 10.5,
        daily_submission_limit: int = 5,
        submission_safety_margin: int = 1,
    ):
        self.budget_file = Path(budget_file)
        self.max_weekly_gpu_hours = max_weekly_gpu_hours
        self.max_session_hours = max_session_hours
        self.daily_submission_limit = daily_submission_limit
        self.submission_safety_margin = submission_safety_margin
        self.session_start_epoch = time.time()

        self.state = self._load_or_init()

    def _load_or_init(self) -> BudgetState:
        if self.budget_file.exists():
            try:
                with open(self.budget_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return BudgetState.model_validate(data)
            except Exception:
                pass

        state = BudgetState(
            weekly_gpu_limit_hours=self.max_weekly_gpu_hours,
            used_weekly_gpu_hours=0.0,
            session_timeout_hours=self.max_session_hours,
            current_session_hours=0.0,
            daily_submissions_used=0,
            daily_submission_limit=self.daily_submission_limit,
        )
        self._save(state)
        return state

    def _save(self, state: BudgetState) -> None:
        self.budget_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.budget_file.with_suffix(f".tmp.{os.getpid()}")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state.model_dump(mode="json"), f, indent=2)
        os.replace(tmp_path, self.budget_file)

    def elapsed_session_hours(self) -> float:
        return (time.time() - self.session_start_epoch) / 3600.0

    def can_allocate_gpu(self, hours: float) -> bool:
        if self.state.used_weekly_gpu_hours + hours > self.max_weekly_gpu_hours:
            return False
        if self.elapsed_session_hours() + hours > self.max_session_hours:
            return False
        return True

    def record_gpu_usage(self, hours: float) -> None:
        self.state.used_weekly_gpu_hours += hours
        self.state.current_session_hours = self.elapsed_session_hours()
        self._save(self.state)

    def can_submit(self) -> bool:
        max_allowed = self.daily_submission_limit - self.submission_safety_margin
        return self.state.daily_submissions_used < max_allowed

    def record_submission(self) -> None:
        self.state.daily_submissions_used += 1
        self._save(self.state)

    def is_session_expired(self) -> bool:
        return self.elapsed_session_hours() >= self.max_session_hours

    def get_remaining_gpu_hours(self) -> float:
        return max(0.0, self.max_weekly_gpu_hours - self.state.used_weekly_gpu_hours)
