from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    TIMEOUT = "TIMEOUT"
    OOM = "OOM"
    VETOED = "VETOED"


class FoldScore(BaseModel):
    fold: int
    score: float
    train_loss: float | None = None
    val_loss: float | None = None
    num_samples: int | None = None


class MetricResult(BaseModel):
    metric_name: str
    overall_score: float
    direction: str = "maximize" # maximize or minimize
    fold_scores: list[FoldScore] = Field(default_factory=list)
    std_score: float = 0.0


class RunResult(BaseModel):
    run_id: str
    status: ExecutionStatus
    metrics: MetricResult | None = None
    execution_time_seconds: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    cpu_memory_peak_mb: float = 0.0
    artifacts_created: list[str] = Field(default_factory=list)
    error_message: str | None = None


class AuditCertificate(BaseModel):
    status: str = "CERTIFIED"
    candidate_path: str
    sample_path: str
    sha256: str
    row_count: int
    columns: list[str]
    timestamp: str


class VetoDocument(BaseModel):
    veto_id: str
    issuing_role: str
    phase: str
    reason: str
    violating_feature: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    timestamp: str
