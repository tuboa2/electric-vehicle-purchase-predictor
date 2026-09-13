#!/usr/bin/env python3
"""
Runner Agent: Baseline Training Script for playground-series-s6e9
Executes 5-fold LightGBM baseline via ExperimentRunner, verifies artifacts,
records results in SQLite, and clears Gate 3 (GATE_3_PIPELINE_SMOKE).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
from experiments.registry import RunRegistry
from experiments.runner import ExperimentRunner
from experiments.worktree import GitWorktreeManager
from memory.sqlite_store import SQLiteExperimentStore
from orchestration.blackboard import FilesystemBlackboard
from orchestration.state_machine import PipelineStateMachine
from schemas.experiment import ModelFamily
from schemas.state import SystemPhase, QualityGate

COMPETITION_ID = "playground-series-s6e9"
RUN_ID = "baseline_lgbm_001"
DATA_DIR = Path(f"data/processed/{COMPETITION_ID}")
ARTIFACT_DIR = Path(f"experiments/artifacts/{RUN_ID}")


def main():
    print("=" * 70)
    print(f"RUNNER AGENT: EXECUTING 5-FOLD BASELINE {RUN_ID}")
    print("=" * 70)

    bb = FilesystemBlackboard()
    fsm = PipelineStateMachine(bb)
    fsm.transition_to(SystemPhase.BASELINE, "runner")

    # 1. Load Datasets
    print("[1] Loading processed Parquet data and certified fold assignments...")
    train_df = pl.read_parquet(DATA_DIR / "train.parquet").to_pandas()
    test_df = pl.read_parquet(DATA_DIR / "test.parquet").to_pandas()
    folds_df = pl.read_parquet(DATA_DIR / "folds.parquet").to_pandas()

    id_col = "id"
    target_col = "Will_Buy_EV"
    features = [c for c in test_df.columns if c != id_col]
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}, Features ({len(features)}): {features}")

    # 2. Setup Runner Harness
    worktree_mgr = GitWorktreeManager()
    registry = RunRegistry()
    runner = ExperimentRunner(worktree_mgr, registry)

    # 3. Model Parameters
    params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
    }

    print("[2] Executing 5-fold cross-validation with LightGBM...")
    run_result = runner.execute_tabular_gbdt(
        run_id=RUN_ID,
        train_df=train_df,
        test_df=test_df,
        folds_df=folds_df,
        features=features,
        target_col=target_col,
        id_col=id_col,
        model_family=ModelFamily.LIGHTGBM,
        params=params,
        metric_name="roc_auc",
        is_classification=True,
    )

    if run_result.status.value != "SUCCESS":
        print(f"[!] Baseline execution failed: {run_result.error_message}")
        sys.exit(1)

    # 4. Verify Physical Artifacts
    print("[3] Verifying generated artifacts in", ARTIFACT_DIR)
    expected_artifacts = [
        ARTIFACT_DIR / "metrics.json",
        ARTIFACT_DIR / "oof_preds.parquet",
        ARTIFACT_DIR / "test_preds.parquet",
        ARTIFACT_DIR / "feature_importance.json",
    ]

    for art in expected_artifacts:
        if not art.exists():
            print(f"[!] MISSING ARTIFACT: {art}")
            sys.exit(1)
        size_kb = art.stat().st_size / 1024
        print(f"[+] Verified artifact: {art.name} ({size_kb:.1f} KB)")

    # Verify OOF and Test schemas
    oof_check = pl.read_parquet(ARTIFACT_DIR / "oof_preds.parquet")
    test_check = pl.read_parquet(ARTIFACT_DIR / "test_preds.parquet")

    assert len(oof_check) == len(train_df), f"OOF length mismatch: {len(oof_check)} vs {len(train_df)}"
    assert len(test_check) == len(test_df), f"Test length mismatch: {len(test_check)} vs {len(test_df)}"
    assert id_col in oof_check.columns and "pred" in oof_check.columns and target_col in oof_check.columns
    assert id_col in test_check.columns and "pred" in test_check.columns

    # 5. Record Trial in SQLite Memory
    print("[4] Persisting trial record to SQLite Project Memory (memory/experiments.db)...")
    store = SQLiteExperimentStore("memory/experiments.db")
    created_at = datetime.now(timezone.utc).isoformat()

    store.record_run(
        run_result=run_result,
        competition_id=COMPETITION_ID,
        model_family="lightgbm",
        hypothesis_id="baseline_lgbm",
        parameters=params,
        created_at=created_at,
    )

    # Record feature importances in SQLite
    with open(ARTIFACT_DIR / "feature_importance.json", "r") as f:
        feat_imp = json.load(f)
    store.record_feature_importances(RUN_ID, feat_imp)
    print("[+] Recorded run and feature importances in SQLite memory.")

    # 6. Update Blackboard State & Clear Gate 3
    print("[5] Updating Blackboard state...")
    state = bb.read_state()
    cv_score = run_result.metrics.overall_score
    cv_std = run_result.metrics.std_score

    state.best_cv_score = cv_score
    state.best_run_id = RUN_ID
    bb.write_state(state, updated_by="runner")

    fsm.mark_gate_cleared(QualityGate.GATE_3_PIPELINE_SMOKE, "runner")
    print("[+] QualityGate.GATE_3_PIPELINE_SMOKE marked CLEARED in state.json.")

    print("=" * 70)
    print(f"BASELINE MODEL {RUN_ID} VERIFIED SUCCESSFULLY!")
    print(f"Overall ROC-AUC: {cv_score:.6f}")
    print(f"Fold Std Dev:    {cv_std:.6f}")
    print("Fold Scores:")
    for fs in run_result.metrics.fold_scores:
        print(f"  Fold {fs.fold}: {fs.score:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
