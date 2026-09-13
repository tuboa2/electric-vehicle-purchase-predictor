from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class SystemPhase(str, Enum):
    INITIALIZED = "INITIALIZED"
    RECONNAISSANCE = "RECONNAISSANCE"
    INGESTION = "INGESTION"
    EDA = "EDA"
    ADVERSARIAL_VAL = "ADVERSARIAL_VAL"
    CV_FORMULATION = "CV_FORMULATION"
    LEAKAGE_AUDIT = "LEAKAGE_AUDIT"
    BASELINE = "BASELINE"
    HYPOTHESIS_QUEUE = "HYPOTHESIS_QUEUE"
    ENSEMBLING = "ENSEMBLING"
    PRE_SUBMISSION_AUDIT = "PRE_SUBMISSION_AUDIT"
    KAGGLE_SUBMISSION = "KAGGLE_SUBMISSION"
    POSTMORTEM = "POSTMORTEM"
    TERMINATED = "TERMINATED"


class QualityGate(str, Enum):
    GATE_1_DATA_INTEGRITY = "GATE_1_DATA_INTEGRITY"
    GATE_2_VALIDATION_VETO = "GATE_2_VALIDATION_VETO"
    GATE_3_PIPELINE_SMOKE = "GATE_3_PIPELINE_SMOKE"
    GATE_4_EV_COMPUTE = "GATE_4_EV_COMPUTE"
    GATE_5_BLENDING_METRIC = "GATE_5_BLENDING_METRIC"
    GATE_6_SUBMISSION_VETO = "GATE_6_SUBMISSION_VETO"


class BudgetState(BaseModel):
    weekly_gpu_limit_hours: float = 30.0
    used_weekly_gpu_hours: float = 0.0
    session_timeout_hours: float = 12.0
    current_session_hours: float = 0.0
    daily_submissions_used: int = 0
    daily_submission_limit: int = 5


class BlackboardState(BaseModel):
    competition_id: str
    phase: SystemPhase = SystemPhase.INITIALIZED
    current_iteration: int = 0
    cleared_gates: list[QualityGate] = Field(default_factory=list)
    budget: BudgetState = Field(default_factory=BudgetState)
    active_locks: list[str] = Field(default_factory=list)
    best_cv_score: float | None = None
    best_run_id: str | None = None
    public_lb_score: float | None = None
    last_updated: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
