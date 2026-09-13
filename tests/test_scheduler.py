from pathlib import Path
import pytest
from orchestration.blackboard import FilesystemBlackboard
from orchestration.budget import BudgetManager
from orchestration.scheduler import EVScheduler
from schemas.experiment import HypothesisSpec, ModelFamily, EVSpec


def test_ev_calculation():
    # EV = (Gain * Conf * Info) / (Cost * (1 + Risk))
    ev = EVSpec(
        expected_gain=0.010,
        confidence=0.8,
        info_gain=0.5,
        compute_cost=0.5,
        risk=0.1,
    )
    # Expected: (0.010 * 0.8 * 0.5) / (0.5 * 1.1) = 0.004 / 0.55 = ~0.007272
    assert pytest.approx(ev.ev_score, rel=1e-3) == 0.007272


def test_ev_scheduler_priority_and_filtering(tmp_path: Path):
    bb = FilesystemBlackboard(root_dir=tmp_path / "bb")
    budget = BudgetManager(budget_file=tmp_path / "budget.json", max_weekly_gpu_hours=2.0)
    scheduler = EVScheduler(bb, budget, min_ev_threshold=0.10)

    # 1. High EV candidate
    h1 = HypothesisSpec(
        hypothesis_id="h1_high",
        title="High EV trial",
        description="test",
        category="feature_engineering",
        model_family=ModelFamily.LIGHTGBM,
        ev=EVSpec(expected_gain=0.05, confidence=0.9, info_gain=0.9, compute_cost=0.1, risk=0.05),
    )

    # 2. Low EV candidate (below threshold)
    h2 = HypothesisSpec(
        hypothesis_id="h2_low",
        title="Low EV trial",
        description="test",
        category="model_tuning",
        model_family=ModelFamily.LIGHTGBM,
        ev=EVSpec(expected_gain=0.001, confidence=0.2, info_gain=0.1, compute_cost=1.0, risk=0.5),
    )

    scheduler.add_hypothesis(h2)
    scheduler.add_hypothesis(h1)

    # h1 should be popped first because its EV is higher
    cand = scheduler.pop_next_candidate(require_gpu=False)
    assert cand is not None
    assert cand["hypothesis_id"] == "h1_high"

    # h2 should be filtered out by min_ev_threshold
    cand_next = scheduler.pop_next_candidate(require_gpu=False)
    assert cand_next is None


def test_budget_manager_limits(tmp_path: Path):
    budget = BudgetManager(
        budget_file=tmp_path / "budget.json",
        max_weekly_gpu_hours=5.0,
        daily_submission_limit=5,
        submission_safety_margin=1,
    )

    assert budget.can_allocate_gpu(4.0) is True
    assert budget.can_allocate_gpu(6.0) is False

    budget.record_gpu_usage(3.5)
    assert budget.get_remaining_gpu_hours() == 1.5
    assert budget.can_allocate_gpu(2.0) is False

    # Submission safety margin test
    assert budget.can_submit() is True
    budget.record_submission() # 1
    budget.record_submission() # 2
    budget.record_submission() # 3
    budget.record_submission() # 4 (max allowed is 5 - 1 = 4)
    assert budget.can_submit() is False
