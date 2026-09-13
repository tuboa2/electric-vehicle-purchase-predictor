from pathlib import Path
import pytest
from orchestration.blackboard import FilesystemBlackboard
from schemas.state import BlackboardState, SystemPhase
from schemas.result import VetoDocument


def test_blackboard_lifecycle(tmp_path: Path):
    bb = FilesystemBlackboard(root_dir=tmp_path)

    # 1. Read default state
    state = bb.read_state()
    assert state.phase == SystemPhase.INITIALIZED

    # 2. Write state transition
    state.phase = SystemPhase.CV_FORMULATION
    state.current_iteration = 2
    bb.write_state(state, updated_by="test_actor")

    # 3. Re-read and verify persistence
    loaded_state = bb.read_state()
    assert loaded_state.phase == SystemPhase.CV_FORMULATION
    assert loaded_state.current_iteration == 2

    # 4. Verify queue operations
    items = [{"hypothesis_id": "hypo_01", "ev_score": 0.25}]
    bb.write_queue(items)
    assert bb.read_queue() == items

    # 5. Verify veto registration
    veto = VetoDocument(
        veto_id="veto_999",
        issuing_role="validation_architect",
        phase="CV_FORMULATION",
        reason="Detected data leakage",
        timestamp="2026-09-13T00:00:00Z",
    )
    veto_path = bb.register_veto(veto)
    assert veto_path.exists()
