from pathlib import Path
from typing import Any


class KnowledgeRetrievalEngine:
    """
    Retrieves domain heuristics and prior winning patterns from Strategic Memory (Layer 3).
    """

    def __init__(self, knowledge_dir: str | Path = "knowledge"):
        self.knowledge_dir = Path(knowledge_dir)

    def get_playbook(self, modality: str = "tabular") -> str:
        playbook_file = self.knowledge_dir / f"{modality}_playbook.md"
        if playbook_file.exists():
            return playbook_file.read_text(encoding="utf-8")
        return f"# {modality.capitalize()} Playbook\nNo recorded heuristics yet."

    def get_meta_lessons(self) -> str:
        meta_file = self.knowledge_dir / "meta_lessons.md"
        if meta_file.exists():
            return meta_file.read_text(encoding="utf-8")
        return "# Meta Lessons\nNo lessons recorded yet."

    def get_default_hyperparameters(self, model_family: str) -> dict[str, Any]:
        """Returns production-grade Kaggle baseline defaults for GBDT families."""
        defaults = {
            "lightgbm": {
                "n_estimators": 1000,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            },
            "catboost": {
                "iterations": 1000,
                "learning_rate": 0.04,
                "depth": 6,
                "random_seed": 42,
                "verbose": 0,
                "thread_count": -1,
            },
            "xgboost": {
                "n_estimators": 1000,
                "learning_rate": 0.03,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
            },
        }
        return defaults.get(model_family.lower(), {})
