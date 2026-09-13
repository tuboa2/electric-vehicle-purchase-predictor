from pathlib import Path
import pytest
from scripts.run_pipeline import AutonomousPipeline
from schemas.state import SystemPhase, QualityGate


def test_full_pipeline_e2e(tmp_path: Path):
    pipeline = AutonomousPipeline(
        competition_id="test-comp-e2e",
        data_dir=str(tmp_path / "data"),
        mock_data=True,
    )

    success = pipeline.run()
    assert success is True

    # Verify final blackboard state
    state = pipeline.blackboard.read_state()
    assert state.phase == SystemPhase.TERMINATED
    assert state.best_cv_score is not None
    assert state.best_cv_score > 0.50

    # Verify that all 6 quality gates were cleared
    cleared_gate_names = {g.value for g in state.cleared_gates}
    assert QualityGate.GATE_1_DATA_INTEGRITY.value in cleared_gate_names
    assert QualityGate.GATE_2_VALIDATION_VETO.value in cleared_gate_names
    assert QualityGate.GATE_3_PIPELINE_SMOKE.value in cleared_gate_names
    assert QualityGate.GATE_4_EV_COMPUTE.value in cleared_gate_names
    assert QualityGate.GATE_5_BLENDING_METRIC.value in cleared_gate_names
    assert QualityGate.GATE_6_SUBMISSION_VETO.value in cleared_gate_names

    # Verify certified submission exists
    sub_csv = Path("experiments/submissions/sub_latest/submission.csv")
    assert sub_csv.exists()

    cert_json = Path("experiments/artifacts/audit_cert.json")
    assert cert_json.exists()
