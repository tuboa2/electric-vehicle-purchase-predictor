import logging
from typing import Any
from orchestration.blackboard import FilesystemBlackboard
from schemas.state import BlackboardState, SystemPhase, QualityGate
from schemas.result import VetoDocument

logger = logging.getLogger(__name__)


class PipelineStateMachine:
    """
    Orchestrates the 12-step autonomous lifecycle and enforces the 6 quality gates.
    Halts immediately upon receiving an authorized veto from validation or leakage agents.
    """

    PHASE_ORDER = [
        SystemPhase.INITIALIZED,
        SystemPhase.RECONNAISSANCE,
        SystemPhase.INGESTION,
        SystemPhase.EDA,
        SystemPhase.ADVERSARIAL_VAL,
        SystemPhase.CV_FORMULATION,
        SystemPhase.LEAKAGE_AUDIT,
        SystemPhase.BASELINE,
        SystemPhase.HYPOTHESIS_QUEUE,
        SystemPhase.ENSEMBLING,
        SystemPhase.PRE_SUBMISSION_AUDIT,
        SystemPhase.KAGGLE_SUBMISSION,
        SystemPhase.POSTMORTEM,
        SystemPhase.TERMINATED,
    ]

    GATE_REQUIREMENTS = {
        SystemPhase.BASELINE: QualityGate.GATE_2_VALIDATION_VETO,
        SystemPhase.HYPOTHESIS_QUEUE: QualityGate.GATE_3_PIPELINE_SMOKE,
        SystemPhase.ENSEMBLING: QualityGate.GATE_4_EV_COMPUTE,
        SystemPhase.PRE_SUBMISSION_AUDIT: QualityGate.GATE_5_BLENDING_METRIC,
        SystemPhase.KAGGLE_SUBMISSION: QualityGate.GATE_6_SUBMISSION_VETO,
    }

    def __init__(self, blackboard: FilesystemBlackboard):
        self.blackboard = blackboard

    def get_current_phase(self) -> SystemPhase:
        return self.blackboard.read_state().phase

    def check_gate(self, gate: QualityGate) -> bool:
        state = self.blackboard.read_state()
        return gate in state.cleared_gates

    def mark_gate_cleared(self, gate: QualityGate, actor_role: str) -> None:
        state = self.blackboard.read_state()
        if gate not in state.cleared_gates:
            state.cleared_gates.append(gate)
            self.blackboard.write_state(state, updated_by=actor_role)
            self.blackboard.append_ledger(
                role=actor_role,
                action="GATE_CLEARED",
                details={"gate": gate.value},
            )
            logger.info("Quality gate %s cleared by %s", gate.value, actor_role)

    def can_transition_to(self, target_phase: SystemPhase) -> tuple[bool, str]:
        state = self.blackboard.read_state()
        current_phase = state.phase

        if target_phase == current_phase:
            return True, "Already in target phase"

        # Check required gate for entering phase
        required_gate = self.GATE_REQUIREMENTS.get(target_phase)
        if required_gate and required_gate not in state.cleared_gates:
            return False, f"Cannot enter {target_phase}: Required gate {required_gate.value} has not been cleared."

        return True, "Eligible"

    def transition_to(self, target_phase: SystemPhase, actor_role: str) -> bool:
        can_move, reason = self.can_transition_to(target_phase)
        if not can_move:
            logger.error("Phase transition rejected: %s", reason)
            return False

        state = self.blackboard.read_state()
        prev_phase = state.phase
        state.phase = target_phase
        state.current_iteration += 1
        self.blackboard.write_state(state, updated_by=actor_role)
        logger.info("Transitioned phase from %s to %s by %s", prev_phase.value, target_phase.value, actor_role)
        return True

    def handle_veto(self, veto: VetoDocument) -> None:
        """
        Processes a formal veto from validation or leakage compliance agents.
        Rolls back the state machine to CV_FORMULATION or PRE_SUBMISSION_AUDIT.
        """
        self.blackboard.register_veto(veto)
        state = self.blackboard.read_state()

        logger.critical("VETO RECEIVED from %s: %s", veto.issuing_role, veto.reason)

        if veto.issuing_role in ["validation_architect", "leakage_compliance"]:
            # Roll back to CV_FORMULATION and revoke Gate 2
            if QualityGate.GATE_2_VALIDATION_VETO in state.cleared_gates:
                state.cleared_gates.remove(QualityGate.GATE_2_VALIDATION_VETO)
            state.phase = SystemPhase.CV_FORMULATION
        elif veto.issuing_role == "artifact_analyst":
            # Revoke Gate 6 and stay in PRE_SUBMISSION_AUDIT
            if QualityGate.GATE_6_SUBMISSION_VETO in state.cleared_gates:
                state.cleared_gates.remove(QualityGate.GATE_6_SUBMISSION_VETO)
            state.phase = SystemPhase.PRE_SUBMISSION_AUDIT

        self.blackboard.write_state(state, updated_by=veto.issuing_role)
