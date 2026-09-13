from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    DATA_PARQUET = "DATA_PARQUET"
    FOLDS_PARQUET = "FOLDS_PARQUET"
    MODEL_WEIGHTS = "MODEL_WEIGHTS"
    OOF_PREDICTIONS = "OOF_PREDICTIONS"
    TEST_PREDICTIONS = "TEST_PREDICTIONS"
    METRICS_JSON = "METRICS_JSON"
    SUBMISSION_CSV = "SUBMISSION_CSV"
    AUDIT_CERTIFICATE = "AUDIT_CERTIFICATE"
    VETO_NOTICE = "VETO_NOTICE"


class ArtifactMetadata(BaseModel):
    artifact_id: str
    artifact_type: ArtifactType
    file_path: str
    sha256: str
    size_bytes: int
    row_count: int | None = None
    column_names: list[str] | None = None
    created_by_role: str
    created_at: str
    extra_metadata: dict[str, Any] = Field(default_factory=dict)
