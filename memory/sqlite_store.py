import json
import sqlite3
from pathlib import Path
from typing import Any
from schemas.result import RunResult, FoldScore


class SQLiteExperimentStore:
    """
    Project Memory (Layer 2): Relational SQLite store for all experimental trials,
    fold breakdowns, feature importances, submissions, and tracebacks.
    """

    def __init__(self, db_path: str | Path = "memory/experiments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    run_id TEXT PRIMARY KEY,
                    competition_id TEXT,
                    model_family TEXT,
                    hypothesis_id TEXT,
                    overall_cv REAL,
                    std_cv REAL,
                    direction TEXT DEFAULT 'maximize',
                    execution_time REAL,
                    parameters TEXT,
                    status TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS fold_scores (
                    run_id TEXT,
                    fold INTEGER,
                    score REAL,
                    train_loss REAL,
                    val_loss REAL,
                    PRIMARY KEY (run_id, fold),
                    FOREIGN KEY (run_id) REFERENCES experiments(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS feature_importances (
                    run_id TEXT,
                    feature_name TEXT,
                    importance REAL,
                    PRIMARY KEY (run_id, feature_name),
                    FOREIGN KEY (run_id) REFERENCES experiments(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS submissions (
                    submission_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    cv_score REAL,
                    public_lb REAL,
                    sha256 TEXT,
                    status TEXT,
                    created_at TEXT,
                    FOREIGN KEY (run_id) REFERENCES experiments(run_id)
                );

                CREATE TABLE IF NOT EXISTS errors (
                    error_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    phase TEXT,
                    error_type TEXT,
                    message TEXT,
                    traceback TEXT,
                    created_at TEXT
                );
                """
            )

    def record_run(
        self,
        run_result: RunResult,
        competition_id: str,
        model_family: str,
        hypothesis_id: str,
        parameters: dict[str, Any],
        created_at: str,
    ) -> None:
        with self._get_conn() as conn:
            cv = run_result.metrics.overall_score if run_result.metrics else None
            std = run_result.metrics.std_score if run_result.metrics else None
            direction = run_result.metrics.direction if run_result.metrics else "maximize"

            conn.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    run_id, competition_id, model_family, hypothesis_id,
                    overall_cv, std_cv, direction, execution_time, parameters, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_result.run_id,
                    competition_id,
                    model_family,
                    hypothesis_id,
                    cv,
                    std,
                    direction,
                    run_result.execution_time_seconds,
                    json.dumps(parameters),
                    run_result.status.value,
                    created_at,
                ),
            )

            if run_result.metrics and run_result.metrics.fold_scores:
                for fs in run_result.metrics.fold_scores:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO fold_scores (run_id, fold, score, train_loss, val_loss)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (run_result.run_id, fs.fold, fs.score, fs.train_loss, fs.val_loss),
                    )

    def record_feature_importances(self, run_id: str, importances: dict[str, float]) -> None:
        with self._get_conn() as conn:
            for feat, imp in importances.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO feature_importances (run_id, feature_name, importance)
                    VALUES (?, ?, ?)
                    """,
                    (run_id, feat, imp),
                )

    def record_submission(
        self,
        submission_id: str,
        run_id: str,
        cv_score: float,
        sha256: str,
        status: str,
        created_at: str,
        public_lb: float | None = None,
    ) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO submissions (
                    submission_id, run_id, cv_score, public_lb, sha256, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (submission_id, run_id, cv_score, public_lb, sha256, status, created_at),
            )

    def update_public_lb(self, submission_id: str, public_lb: float) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE submissions SET public_lb = ?, status = 'SCORED' WHERE submission_id = ?",
                (public_lb, submission_id),
            )

    def record_error(
        self,
        error_id: str,
        run_id: str | None,
        phase: str,
        error_type: str,
        message: str,
        traceback: str,
        created_at: str,
    ) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO errors (
                    error_id, run_id, phase, error_type, message, traceback, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (error_id, run_id, phase, error_type, message, traceback, created_at),
            )

    def get_best_run(self, competition_id: str, maximize: bool = True) -> dict[str, Any] | None:
        order = "DESC" if maximize else "ASC"
        with self._get_conn() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM experiments
                WHERE competition_id = ? AND overall_cv IS NOT NULL AND status = 'SUCCESS'
                ORDER BY overall_cv {order} LIMIT 1
                """,
                (competition_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_top_runs(self, competition_id: str, limit: int = 5, maximize: bool = True) -> list[dict[str, Any]]:
        order = "DESC" if maximize else "ASC"
        with self._get_conn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM experiments
                WHERE competition_id = ? AND overall_cv IS NOT NULL AND status = 'SUCCESS'
                ORDER BY overall_cv {order} LIMIT ?
                """,
                (competition_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_feature_importances(self, run_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT feature_name, importance FROM feature_importances
                WHERE run_id = ? ORDER BY importance DESC LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
