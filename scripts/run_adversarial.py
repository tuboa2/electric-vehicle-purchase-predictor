#!/usr/bin/env python3
"""
Adversarial Validation Script for playground-series-s6e9
Evaluates covariate shift between train and test using evaluation/adversarial.py.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
from evaluation.adversarial import AdversarialValidator
from orchestration.blackboard import FilesystemBlackboard
from orchestration.state_machine import PipelineStateMachine
from schemas.state import SystemPhase, QualityGate

DATA_DIR = Path("data/processed/playground-series-s6e9")
ARTIFACTS_DIR = Path("experiments/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("[1] Loading processed Parquet datasets...")
    train_df = pl.read_parquet(DATA_DIR / "train.parquet").to_pandas()
    test_df = pl.read_parquet(DATA_DIR / "test.parquet").to_pandas()

    print(f"Loaded train ({len(train_df)} rows) and test ({len(test_df)} rows)")

    # Exclude ID and target
    drop_cols = ["id", "Will_Buy_EV"]
    print(f"[2] Running Adversarial Validation excluding {drop_cols}...")

    # Transition state to ADVERSARIAL_VAL
    bb = FilesystemBlackboard()
    fsm = PipelineStateMachine(bb)
    fsm.transition_to(SystemPhase.ADVERSARIAL_VAL, "data_forensics")

    result = AdversarialValidator.evaluate(
        train_df=train_df,
        test_df=test_df,
        drop_cols=drop_cols,
        seed=42,
    )

    print("=" * 60)
    print(f"ADVERSARIAL VALIDATION RESULT: AUC = {result['adversarial_auc']:.4f}")
    print(f"SEVERITY: {result['severity']}")
    print("=" * 60)
    print("Top Feature Importances:")
    for item in result["top_drift_features"]:
        print(f"  - {item['feature']}: {item['importance']:.2f}")

    # Write report
    report_path = ARTIFACTS_DIR / "adversarial_report.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[+] Saved report to {report_path}")

    # Integrity check and Gate 1 clearance
    auc = result["adversarial_auc"]
    if auc < 0.70:
        print(f"[+] Adversarial AUC {auc:.4f} < 0.70 threshold. Covariate shift is safe.")
        fsm.mark_gate_cleared(QualityGate.GATE_1_DATA_INTEGRITY, "data_forensics")
        print("[+] QualityGate.GATE_1_DATA_INTEGRITY marked CLEARED in state.json.")
    else:
        print(f"[!] Warning: Adversarial AUC {auc:.4f} exceeds safety threshold.")

if __name__ == "__main__":
    main()
