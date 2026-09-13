import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml

from schemas.state import BlackboardState, SystemPhase
from schemas.result import VetoDocument


class FilesystemBlackboard:
    """
    Decoupled Filesystem Blackboard managing global state, active runs,
    queue, and immutable ledger with atomic write-then-rename guarantees.
    """

    def __init__(self, root_dir: str | Path = "experiments/blackboard"):
        self.root = Path(root_dir)
        self.state_file = self.root / "state.json"
        self.queue_file = self.root / "queue.yaml"
        self.ledger_file = self.root / "ledger.jsonl"
        self.active_runs_dir = self.root / "active_runs"
        self.vetos_dir = self.root / "vetos"

        self._initialize_directories()

    def _initialize_directories(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.active_runs_dir.mkdir(parents=True, exist_ok=True)
        self.vetos_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_write_json(self, path: Path, data: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(f".tmp.{os.getpid()}")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, path)

    def _atomic_write_yaml(self, path: Path, data: Any) -> None:
        tmp_path = path.with_suffix(f".tmp.{os.getpid()}")
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False)
        os.replace(tmp_path, path)

    def append_ledger(self, role: str, action: str, details: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "action": action,
            "details": details,
        }
        with open(self.ledger_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def read_state(self) -> BlackboardState:
        if not self.state_file.exists():
            default_state = BlackboardState(
                competition_id="uninitialized",
                phase=SystemPhase.INITIALIZED,
                last_updated=datetime.now(timezone.utc).isoformat(),
            )
            self.write_state(default_state, updated_by="init")
            return default_state

        with open(self.state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return BlackboardState.model_validate(data)

    def write_state(self, state: BlackboardState, updated_by: str = "system") -> None:
        state.last_updated = datetime.now(timezone.utc).isoformat()
        state_dict = state.model_dump(mode="json")
        self._atomic_write_json(self.state_file, state_dict)
        self.append_ledger(
            role=updated_by,
            action="STATE_TRANSITION",
            details={"phase": state.phase, "iteration": state.current_iteration},
        )

    def read_queue(self) -> list[dict[str, Any]]:
        if not self.queue_file.exists():
            return []
        with open(self.queue_file, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            return content if isinstance(content, list) else []

    def write_queue(self, queue_items: list[dict[str, Any]]) -> None:
        self._atomic_write_yaml(self.queue_file, queue_items)

    def register_veto(self, veto: VetoDocument) -> Path:
        veto_path = self.vetos_dir / f"veto_{veto.veto_id}.json"
        self._atomic_write_json(veto_path, veto.model_dump(mode="json"))
        self.append_ledger(
            role=veto.issuing_role,
            action="VETO_ISSUED",
            details={"veto_id": veto.veto_id, "reason": veto.reason, "phase": veto.phase},
        )
        return veto_path

    def register_active_run(self, run_id: str, spec: dict[str, Any]) -> Path:
        run_dir = self.active_runs_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        spec_path = run_dir / "spec.yaml"
        self._atomic_write_yaml(spec_path, spec)

        sentinel = {
            "run_id": run_id,
            "status": "INITIALIZED",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
        }
        self._atomic_write_json(run_dir / "run_sentinel.json", sentinel)
        return run_dir

    def update_run_sentinel(self, run_id: str, updates: dict[str, Any]) -> None:
        sentinel_path = self.active_runs_dir / f"run_{run_id}" / "run_sentinel.json"
        if sentinel_path.exists():
            with open(sentinel_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.update(updates)
            data["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
            self._atomic_write_json(sentinel_path, data)

    def complete_active_run(self, run_id: str) -> None:
        self.update_run_sentinel(run_id, {"status": "COMPLETED"})
