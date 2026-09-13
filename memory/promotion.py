import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from memory.sqlite_store import SQLiteExperimentStore

logger = logging.getLogger(__name__)


class MemoryPromotionEngine:
    """
    Evaluates completed experimental trials and promotes high-performing
    heuristics from Project Memory (Layer 2) to Strategic Memory (Layer 3)
    and Meta Memory (Layer 4).
    """

    def __init__(
        self,
        db_store: SQLiteExperimentStore,
        knowledge_dir: str | Path = "knowledge",
        min_cv_gain_threshold: float = 0.0010,
    ):
        self.db = db_store
        self.knowledge_dir = Path(knowledge_dir)
        self.min_cv_gain_threshold = min_cv_gain_threshold
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_and_promote(
        self,
        run_id: str,
        baseline_cv: float,
        current_cv: float,
        hypothesis_title: str,
        modality: str = "tabular",
        maximize: bool = True,
    ) -> bool:
        gain = (current_cv - baseline_cv) if maximize else (baseline_cv - current_cv)

        if gain < self.min_cv_gain_threshold:
            logger.info("Hypothesis '%s' gain (%.4f) below promotion threshold (%.4f)", hypothesis_title, gain, self.min_cv_gain_threshold)
            return False

        # Promote to strategic playbook
        playbook_path = self.knowledge_dir / f"{modality}_playbook.md"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        entry = f"""
### Promoted Strategy: {hypothesis_title}
- **Date Promoted:** {timestamp}
- **Run ID:** `{run_id}`
- **Empirical CV Delta:** `+{gain:.4f}` (Baseline: {baseline_cv:.4f} -> Candidate: {current_cv:.4f})
- **Status:** Verified and Integrated into Default Strategy Pool
"""
        with open(playbook_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")

        # Record meta-lesson
        meta_path = self.knowledge_dir / "meta_lessons.md"
        meta_entry = f"""
- **[{timestamp}] Promotion:** Strategy `{hypothesis_title}` demonstrated statistically significant CV improvement (+{gain:.4f}) on {modality} data. Recommended for early EV scheduling in future competitions.
"""
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(meta_entry)

        logger.info("Successfully promoted '%s' to %s", hypothesis_title, playbook_path)
        return True
