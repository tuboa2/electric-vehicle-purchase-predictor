import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RunRegistry:
    """
    Registry for tracking active experiment processes, sentinels,
    resource usage, and termination signals.
    """

    def __init__(self, artifacts_base_dir: str | Path = "experiments/artifacts"):
        self.artifacts_base_dir = Path(artifacts_base_dir)
        self.artifacts_base_dir.mkdir(parents=True, exist_ok=True)

    def init_run_artifact_dir(self, run_id: str) -> Path:
        dir_name = run_id if (run_id.startswith("run_") or run_id.startswith("baseline_")) else f"run_{run_id}"
        run_p = self.artifacts_base_dir / dir_name
        run_p.mkdir(parents=True, exist_ok=True)
        return run_p

    def write_sentinel(self, run_id: str, status: str, pid: int | None = None) -> None:
        run_p = self.init_run_artifact_dir(run_id)
        sentinel_file = run_p / "run_sentinel.json"
        data = {
            "run_id": run_id,
            "status": status,
            "pid": pid or os.getpid(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(sentinel_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def is_run_active(self, run_id: str) -> bool:
        sentinel_file = self.init_run_artifact_dir(run_id) / "run_sentinel.json"
        if not sentinel_file.exists():
            return False
        try:
            with open(sentinel_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("status") in ["INITIALIZED", "RUNNING"]
        except Exception:
            return False
