from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ModelFamily(str, Enum):
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    PYTORCH_TABULAR = "pytorch_tabular"
    SCIKIT_LEARN = "scikit-learn"
    ENSEMBLE = "ensemble"


class EVSpec(BaseModel):
    expected_gain: float = Field(..., ge=0.0, description="Estimated metric delta")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in hypothesis (0 to 1)")
    info_gain: float = Field(..., ge=0.0, le=1.0, description="Novelty/information gain")
    compute_cost: float = Field(..., gt=0.0, description="Estimated execution time in hours")
    risk: float = Field(default=0.1, ge=0.0, le=1.0, description="Estimated probability of failure/leakage")

    @property
    def ev_score(self) -> float:
        return (self.expected_gain * self.confidence * self.info_gain) / (self.compute_cost * (1.0 + self.risk))


class HypothesisSpec(BaseModel):
    hypothesis_id: str
    title: str
    description: str
    category: str = Field(..., description="feature_engineering | model_tuning | architecture | ensembling")
    model_family: ModelFamily
    parameters: dict[str, Any] = Field(default_factory=dict)
    feature_transforms: list[str] = Field(default_factory=list)
    ev: EVSpec


class ExperimentSpec(BaseModel):
    run_id: str
    hypothesis: HypothesisSpec
    worktree_path: str
    folds_path: str
    train_data_path: str
    test_data_path: str
    seed: int = 42
    timeout_seconds: int = 1800 # default 30 minutes
