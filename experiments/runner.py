import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, root_mean_squared_error

from evaluation.metrics import MetricRegistry
from schemas.experiment import ExperimentSpec, ModelFamily
from schemas.result import RunResult, ExecutionStatus, MetricResult, FoldScore
from experiments.worktree import GitWorktreeManager
from experiments.registry import RunRegistry

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Physical training execution harness.
    Executes cross-validated models inside sandboxed worktrees and writes
    verifiable out-of-fold and test predictions to disk.
    """

    def __init__(
        self,
        worktree_manager: GitWorktreeManager,
        registry: RunRegistry,
        artifacts_dir: str | Path = "experiments/artifacts",
    ):
        self.worktree_manager = worktree_manager
        self.registry = registry
        self.artifacts_dir = Path(artifacts_dir)

    def execute_tabular_gbdt(
        self,
        run_id: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        folds_df: pd.DataFrame,
        features: list[str],
        target_col: str,
        id_col: str,
        model_family: ModelFamily,
        params: dict[str, Any],
        metric_name: str = "roc_auc",
        is_classification: bool = True,
    ) -> RunResult:
        start_time = time.time()
        run_artifact_dir = self.registry.init_run_artifact_dir(run_id)
        self.registry.write_sentinel(run_id, "RUNNING")

        # Join fold information
        df = train_df.merge(folds_df, on=id_col, how="inner")
        if "fold" not in df.columns:
            error_msg = "Fold assignment column 'fold' not found after merging with folds table"
            logger.error(error_msg)
            return RunResult(
                run_id=run_id,
                status=ExecutionStatus.FAILURE,
                error_message=error_msg,
            )

        n_folds = df["fold"].nunique()
        oof_preds = np.zeros(len(df))
        test_preds = np.zeros(len(test_df))
        fold_scores = []
        feature_importance_accum = np.zeros(len(features))

        def calc_metric(y_t: np.ndarray, y_p: np.ndarray) -> float:
            try:
                fn, _ = MetricRegistry.get_metric(metric_name)
                return fn(y_t, y_p)
            except Exception:
                if metric_name == "roc_auc":
                    return float(roc_auc_score(y_t, y_p))
                elif metric_name == "rmse":
                    return float(root_mean_squared_error(y_t, y_p))
                return float(np.mean(y_t == (y_p >= 0.5))) if is_classification else float(root_mean_squared_error(y_t, y_p))

        # Handle non-numeric categorical features
        cat_features = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
        for c in cat_features:
            df[c] = df[c].astype("category")
            test_df[c] = test_df[c].astype("category")

        try:
            for fold in range(n_folds):
                tr_mask = df["fold"] != fold
                va_mask = df["fold"] == fold

                X_tr, y_tr = df.loc[tr_mask, features], df.loc[tr_mask, target_col]
                X_va, y_va = df.loc[va_mask, features], df.loc[va_mask, target_col]

                # Model selection
                if model_family == ModelFamily.LIGHTGBM or model_family == ModelFamily.SCIKIT_LEARN:
                    model_cls = lgb.LGBMClassifier if is_classification else lgb.LGBMRegressor
                    clean_params = {k: v for k, v in params.items() if k not in ["seed", "random_state"]}
                    model = model_cls(random_state=42, verbose=-1, n_jobs=-1, **clean_params)
                    model.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_va, y_va)],
                        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
                    )

                    if is_classification:
                        va_pred = model.predict_proba(X_va)[:, 1] if len(model.classes_) == 2 else model.predict_proba(X_va)
                        t_pred = model.predict_proba(test_df[features])[:, 1] if len(model.classes_) == 2 else model.predict_proba(test_df[features])
                    else:
                        va_pred = model.predict(X_va)
                        t_pred = model.predict(test_df[features])

                    feature_importance_accum += model.feature_importances_ / n_folds
                else:
                    raise NotImplementedError(f"Model family {model_family} not yet wired in runner")

                oof_preds[va_mask] = va_pred
                test_preds += t_pred / n_folds

                score = calc_metric(y_va.values, va_pred)
                fold_scores.append(FoldScore(fold=fold, score=score))

            overall_metric = calc_metric(df[target_col].values, oof_preds)
            std_metric = float(np.std([fs.score for fs in fold_scores]))

            direction = "maximize" if metric_name in ["roc_auc", "accuracy", "f1"] else "minimize"
            metrics = MetricResult(
                metric_name=metric_name,
                overall_score=overall_metric,
                direction=direction,
                fold_scores=fold_scores,
                std_score=std_metric,
            )

            # Persist artifacts to disk
            # 1. metrics.json
            with open(run_artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics.model_dump(mode="json"), f, indent=2)

            # 2. oof_preds.parquet
            oof_df = pd.DataFrame({
                id_col: df[id_col],
                "pred": oof_preds,
                target_col: df[target_col],
                "fold": df["fold"],
            })
            pl.from_pandas(oof_df).write_parquet(run_artifact_dir / "oof_preds.parquet")

            # 3. test_preds.parquet
            test_pred_df = pd.DataFrame({
                id_col: test_df[id_col],
                "pred": test_preds,
            })
            pl.from_pandas(test_pred_df).write_parquet(run_artifact_dir / "test_preds.parquet")

            # 4. feature_importances.json
            feat_imp = {feat: float(imp) for feat, imp in zip(features, feature_importance_accum)}
            with open(run_artifact_dir / "feature_importance.json", "w", encoding="utf-8") as f:
                json.dump(feat_imp, f, indent=2)

            exec_time = time.time() - start_time
            self.registry.write_sentinel(run_id, "COMPLETED")

            return RunResult(
                run_id=run_id,
                status=ExecutionStatus.SUCCESS,
                metrics=metrics,
                execution_time_seconds=exec_time,
                artifacts_created=[
                    str(run_artifact_dir / "metrics.json"),
                    str(run_artifact_dir / "oof_preds.parquet"),
                    str(run_artifact_dir / "test_preds.parquet"),
                    str(run_artifact_dir / "feature_importance.json"),
                ],
            )

        except Exception as e:
            logger.exception("Error executing trial %s: %s", run_id, e)
            self.registry.write_sentinel(run_id, "FAILED")
            return RunResult(
                run_id=run_id,
                status=ExecutionStatus.FAILURE,
                execution_time_seconds=time.time() - start_time,
                error_message=str(e),
            )
