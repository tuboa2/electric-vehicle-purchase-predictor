from pathlib import Path
from typing import Any
from memory.sqlite_store import SQLiteExperimentStore
from memory.promotion import MemoryPromotionEngine
from memory.retrieval import KnowledgeRetrievalEngine
from schemas.result import RunResult


class UnifiedMemoryEngine:
    """
    Coordinates all four layers of the KAMAS Memory Hierarchy:
    - Layer 1: Working Memory (active ephemeral runs)
    - Layer 2: Project Memory (relational SQLite store)
    - Layer 3: Strategic Memory (curated domain playbooks)
    - Layer 4: Meta Memory (longitudinal system self-reflection)
    """

    def __init__(
        self,
        db_path: str | Path = "memory/experiments.db",
        knowledge_dir: str | Path = "knowledge",
    ):
        self.project_store = SQLiteExperimentStore(db_path)
        self.retrieval = KnowledgeRetrievalEngine(knowledge_dir)
        self.promotion = MemoryPromotionEngine(self.project_store, knowledge_dir)

    def log_trial(
        self,
        run_result: RunResult,
        competition_id: str,
        model_family: str,
        hypothesis_id: str,
        parameters: dict[str, Any],
        created_at: str,
        feature_importances: dict[str, float] | None = None,
    ) -> None:
        self.project_store.record_run(
            run_result=run_result,
            competition_id=competition_id,
            model_family=model_family,
            hypothesis_id=hypothesis_id,
            parameters=parameters,
            created_at=created_at,
        )
        if feature_importances:
            self.project_store.record_feature_importances(run_result.run_id, feature_importances)

    def consider_promotion(
        self,
        run_id: str,
        baseline_cv: float,
        current_cv: float,
        hypothesis_title: str,
        modality: str = "tabular",
        maximize: bool = True,
    ) -> bool:
        return self.promotion.evaluate_and_promote(
            run_id=run_id,
            baseline_cv=baseline_cv,
            current_cv=current_cv,
            hypothesis_title=hypothesis_title,
            modality=modality,
            maximize=maximize,
        )
