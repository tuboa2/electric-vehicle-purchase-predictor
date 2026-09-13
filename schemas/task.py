from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    COMMANDER = "commander"
    SCOUT = "scout"
    DATA_FORENSICS = "data_forensics"
    VALIDATION_ARCHITECT = "validation_architect"
    LEAKAGE_COMPLIANCE = "leakage_compliance"
    FEATURE_MODEL_STRATEGIST = "feature_model_strategist"
    EXPERIMENT_MANAGER = "experiment_manager"
    RUNNER = "runner"
    BLENDER = "blender"
    ARTIFACT_ANALYST = "artifact_analyst"
    KAGGLE_EXECUTOR = "kaggle_executor"
    MEMORY_CURATOR = "memory_curator"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    VETOED = "VETOED"


class TaskSpec(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    assigned_role: AgentRole = Field(..., description="Target agent role")
    phase: str = Field(..., description="Lifecycle phase")
    description: str = Field(..., description="Actionable instruction")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Input parameters and file paths")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current execution state")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    completed_at: str | None = Field(default=None, description="ISO 8601 completion timestamp")
