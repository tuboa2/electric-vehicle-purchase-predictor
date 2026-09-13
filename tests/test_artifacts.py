from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from evaluation.metrics import MetricRegistry


def verify_submission(candidate_csv: Path, sample_csv: Path) -> dict:
    cand_df = pd.read_csv(candidate_csv)
    samp_df = pd.read_csv(sample_csv)

    if len(cand_df) != len(samp_df):
        raise ValueError(f"Row count mismatch: Candidate={len(cand_df)}, Sample={len(samp_df)}")

    if list(cand_df.columns) != list(samp_df.columns):
        raise ValueError(f"Columns mismatch: Candidate={list(cand_df.columns)}, Sample={list(samp_df.columns)}")

    target_cols = [c for c in samp_df.columns if c != samp_df.columns[0]]
    for col in target_cols:
        if cand_df[col].isnull().any() or not np.isfinite(cand_df[col]).all():
            raise ValueError(f"Column '{col}' contains NaN or infinite values")
        if cand_df[col].nunique() <= 1:
            raise ValueError(f"Column '{col}' is constant")

    id_col = samp_df.columns[0]
    if not (cand_df[id_col] == samp_df[id_col]).all():
        raise ValueError("ID column values or ordering does not match sample submission")

    return {"status": "CERTIFIED", "rows": len(cand_df)}


def test_artifact_analyst_valid_submission(tmp_path: Path):
    sample_csv = tmp_path / "sample.csv"
    cand_csv = tmp_path / "cand.csv"

    sample_df = pd.DataFrame({"id": [1, 2, 3], "target": [0.0, 0.0, 0.0]})
    cand_df = pd.DataFrame({"id": [1, 2, 3], "target": [0.12, 0.85, 0.44]})

    sample_df.to_csv(sample_csv, index=False)
    cand_df.to_csv(cand_csv, index=False)

    res = verify_submission(cand_csv, sample_csv)
    assert res["status"] == "CERTIFIED"
    assert res["rows"] == 3


def test_artifact_analyst_row_mismatch(tmp_path: Path):
    sample_csv = tmp_path / "sample.csv"
    cand_csv = tmp_path / "cand.csv"

    sample_df = pd.DataFrame({"id": [1, 2, 3], "target": [0.0, 0.0, 0.0]})
    cand_df = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})

    sample_df.to_csv(sample_csv, index=False)
    cand_df.to_csv(cand_csv, index=False)

    with pytest.raises(ValueError, match="Row count mismatch"):
        verify_submission(cand_csv, sample_csv)


def test_artifact_analyst_nan_rejection(tmp_path: Path):
    sample_csv = tmp_path / "sample.csv"
    cand_csv = tmp_path / "cand.csv"

    sample_df = pd.DataFrame({"id": [1, 2, 3], "target": [0.0, 0.0, 0.0]})
    cand_df = pd.DataFrame({"id": [1, 2, 3], "target": [0.1, np.nan, 0.5]})

    sample_df.to_csv(sample_csv, index=False)
    cand_df.to_csv(cand_csv, index=False)

    with pytest.raises(ValueError, match="contains NaN or infinite"):
        verify_submission(cand_csv, sample_csv)
