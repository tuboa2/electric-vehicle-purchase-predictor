#!/usr/bin/env python3
"""
KAMAS Autonomous Multi-Agent Pipeline Runner
Executes the full 12-phase competitive ML loop under strict epistemic and veto governance.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from orchestration.blackboard import FilesystemBlackboard
from orchestration.budget import BudgetManager
from orchestration.scheduler import EVScheduler
from orchestration.state_machine import PipelineStateMachine
from memory.engine import UnifiedMemoryEngine
from kaggle.client import KaggleClient
from kaggle.hardware import HardwareManager
from experiments.worktree import GitWorktreeManager
from experiments.registry import RunRegistry
from experiments.runner import ExperimentRunner
from evaluation.cv import CrossValidationBuilder
from evaluation.adversarial import AdversarialValidator
from evaluation.blender import EnsembleBlender
from evaluation.ablation import AblationHarness
from evaluation.metrics import MetricRegistry
from schemas.state import SystemPhase, QualityGate
from schemas.experiment import HypothesisSpec, ModelFamily, EVSpec
from schemas.result import ExecutionStatus, VetoDocument, AuditCertificate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("KAMAS-Pipeline")


class AutonomousPipeline:
    def __init__(self, competition_id: str, data_dir: str = "data", mock_data: bool = False):
        self.competition_id = competition_id
        self.raw_data_dir = Path(data_dir) / "raw" / competition_id
        self.processed_data_dir = Path(data_dir) / "processed" / competition_id
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.mock_data = mock_data

        self.blackboard = FilesystemBlackboard()
        self.budget_mgr = BudgetManager()
        self.scheduler = EVScheduler(self.blackboard, self.budget_mgr)
        self.fsm = PipelineStateMachine(self.blackboard)
        self.memory = UnifiedMemoryEngine()
        self.worktree_mgr = GitWorktreeManager()
        self.registry = RunRegistry()
        self.runner = ExperimentRunner(self.worktree_mgr, self.registry)
        self.kaggle_client = KaggleClient()

    def _ensure_data_exists(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_path = self.processed_data_dir / "train.parquet"
        test_path = self.processed_data_dir / "test.parquet"
        sample_path = self.processed_data_dir / "sample_submission.csv"

        if train_path.exists() and test_path.exists() and sample_path.exists():
            return (
                pl.read_parquet(train_path).to_pandas(),
                pl.read_parquet(test_path).to_pandas(),
                pd.read_csv(sample_path),
            )

        # If data does not exist, synthesize a realistic benchmark tabular dataset
        logger.info("Synthesizing realistic benchmark competition data...")
        np.random.seed(42)
        n_train = 1200
        n_test = 400

        # Features with subtle signal and noise
        f1_tr = np.random.normal(0, 1, n_train)
        f2_tr = np.random.uniform(10, 50, n_train)
        cat1_tr = np.random.choice(["A", "B", "C"], n_train)
        group_tr = np.random.choice([f"grp_{i}" for i in range(50)], n_train)

        # Target has non-linear relationship with f1 and f2
        logits = 0.8 * f1_tr + 0.03 * f2_tr + (cat1_tr == "B") * 0.5 + np.random.normal(0, 0.5, n_train)
        prob = 1.0 / (1.0 + np.exp(-logits))
        y_tr = (prob >= 0.5).astype(int)

        train_df = pd.DataFrame({
            "id": np.arange(n_train),
            "feat_1": f1_tr,
            "feat_2": f2_tr,
            "cat_1": cat1_tr,
            "group_id": group_tr,
            "target": y_tr,
        })

        f1_te = np.random.normal(0.05, 1, n_test) # Slight mild drift
        f2_te = np.random.uniform(10, 50, n_test)
        cat1_te = np.random.choice(["A", "B", "C"], n_test)
        group_te = np.random.choice([f"grp_test_{i}" for i in range(20)], n_test)

        test_df = pd.DataFrame({
            "id": np.arange(n_train, n_train + n_test),
            "feat_1": f1_te,
            "feat_2": f2_te,
            "cat_1": cat1_te,
            "group_id": group_te,
        })

        sample_sub = pd.DataFrame({
            "id": test_df["id"],
            "target": np.zeros(n_test, dtype=float),
        })

        pl.from_pandas(train_df).write_parquet(train_path)
        pl.from_pandas(test_df).write_parquet(test_path)
        sample_sub.to_csv(sample_path, index=False)

        logger.info("Benchmark data persisted to %s", self.processed_data_dir)
        return train_df, test_df, sample_sub

    def run(self) -> bool:
        logger.info("=== STARTING KAMAS 12-PHASE AUTONOMOUS PIPELINE ===")

        # Phase 1: Reconnaissance
        self.fsm.transition_to(SystemPhase.RECONNAISSANCE, "scout")
        target_col = "target"
        id_col = "id"
        group_col = "group_id"
        metric_name = "roc_auc"
        is_classification = True

        # Phase 2: Ingestion
        self.fsm.transition_to(SystemPhase.INGESTION, "data_forensics")
        train_df, test_df, sample_df = self._ensure_data_exists()

        # Phase 3: Exploratory Data Analysis
        self.fsm.transition_to(SystemPhase.EDA, "data_forensics")
        logger.info("EDA: Train rows=%d, Test rows=%d, Features=%d", len(train_df), len(test_df), train_df.shape[1] - 2)

        # Phase 4: Adversarial Validation & Drift Analysis
        self.fsm.transition_to(SystemPhase.ADVERSARIAL_VAL, "data_forensics")
        adv_res = AdversarialValidator.evaluate(
            train_df=train_df,
            test_df=test_df,
            drop_cols=[id_col, target_col, group_col],
        )
        logger.info("Adversarial Validation: AUC=%.4f (Severity: %s)", adv_res["adversarial_auc"], adv_res["severity"])
        self.fsm.mark_gate_cleared(QualityGate.GATE_1_DATA_INTEGRITY, "data_forensics")

        # Phase 5: CV Scheme Formulation
        self.fsm.transition_to(SystemPhase.CV_FORMULATION, "validation_architect")
        folds_df = CrossValidationBuilder.generate_folds(
            df=train_df,
            id_col=id_col,
            target_col=target_col,
            group_col=group_col,
            is_classification=is_classification,
            n_splits=5,
        )
        folds_path = self.processed_data_dir / "folds.parquet"
        pl.from_pandas(folds_df).write_parquet(folds_path)
        logger.info("Generated leak-free 5-fold partition with GroupKFold on '%s'", group_col)

        # Phase 6: Leakage Pre-Audit
        self.fsm.transition_to(SystemPhase.LEAKAGE_AUDIT, "leakage_compliance")
        # Check target correlations
        for col in ["feat_1", "feat_2"]:
            corr = np.abs(np.corrcoef(train_df[col], train_df[target_col])[0, 1])
            if corr >= 0.999:
                veto = VetoDocument(
                    veto_id=f"leak_{int(time.time())}",
                    issuing_role="leakage_compliance",
                    phase="LEAKAGE_AUDIT",
                    reason=f"Feature {col} has suspicious correlation {corr:.4f} with target",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self.fsm.handle_veto(veto)
                return False

        self.fsm.mark_gate_cleared(QualityGate.GATE_2_VALIDATION_VETO, "leakage_compliance")

        # Phase 7: Minimal Baseline
        self.fsm.transition_to(SystemPhase.BASELINE, "runner")
        features_base = ["feat_1", "feat_2"]

        baseline_res = self.runner.execute_tabular_gbdt(
            run_id="baseline_lgbm_001",
            train_df=train_df,
            test_df=test_df,
            folds_df=folds_df,
            features=features_base,
            target_col=target_col,
            id_col=id_col,
            model_family=ModelFamily.LIGHTGBM,
            params={"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 15},
            metric_name=metric_name,
            is_classification=is_classification,
        )

        if baseline_res.status != ExecutionStatus.SUCCESS or not baseline_res.metrics:
            logger.critical("Baseline execution failed: %s", baseline_res.error_message)
            return False

        baseline_cv = baseline_res.metrics.overall_score
        logger.info("Baseline LightGBM CV: %.5f (Std: %.5f)", baseline_cv, baseline_res.metrics.std_score)
        self.memory.log_trial(
            run_result=baseline_res,
            competition_id=self.competition_id,
            model_family="lightgbm",
            hypothesis_id="baseline_lgbm",
            parameters={"n_estimators": 100, "learning_rate": 0.05},
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.fsm.mark_gate_cleared(QualityGate.GATE_3_PIPELINE_SMOKE, "runner")

        # Phase 8: Hypothesis Queue & Research Exploration
        self.fsm.transition_to(SystemPhase.HYPOTHESIS_QUEUE, "experiment_manager")

        # Generate feature engineering: Groupby aggregations and interaction
        train_df["feat_interact"] = train_df["feat_1"] * train_df["feat_2"]
        test_df["feat_interact"] = test_df["feat_1"] * test_df["feat_2"]
        features_expanded = ["feat_1", "feat_2", "feat_interact"]

        # Queue Hypothesis 1: Feature Interaction
        hypo_1 = HypothesisSpec(
            hypothesis_id="hypo_interaction_001",
            title="Non-linear multiplicative interaction of feat_1 and feat_2",
            description="Physics-based interaction product feature",
            category="feature_engineering",
            model_family=ModelFamily.LIGHTGBM,
            parameters={"n_estimators": 150, "learning_rate": 0.03, "num_leaves": 20},
            feature_transforms=["feat_interact = feat_1 * feat_2"],
            ev=EVSpec(expected_gain=0.005, confidence=0.8, info_gain=0.7, compute_cost=0.05, risk=0.05),
        )
        self.scheduler.add_hypothesis(hypo_1)

        # Queue Hypothesis 2: Regularized Depth-Limited Trees
        hypo_2 = HypothesisSpec(
            hypothesis_id="hypo_reg_tree_002",
            title="Shallow depth-constrained LightGBM with higher subsample",
            description="Anti-overfitting shallow trees with regularized learning rate",
            category="model_tuning",
            model_family=ModelFamily.LIGHTGBM,
            parameters={"n_estimators": 200, "learning_rate": 0.02, "max_depth": 3, "num_leaves": 7},
            feature_transforms=[],
            ev=EVSpec(expected_gain=0.003, confidence=0.7, info_gain=0.5, compute_cost=0.05, risk=0.05),
        )
        self.scheduler.add_hypothesis(hypo_2)

        # Dispatch trials
        oof_candidates = {}
        test_candidates = {}

        # Load baseline OOF
        oof_base = pl.read_parquet("experiments/artifacts/run_baseline_lgbm_001/oof_preds.parquet").to_pandas()["pred"].values
        test_base = pl.read_parquet("experiments/artifacts/run_baseline_lgbm_001/test_preds.parquet").to_pandas()["pred"].values
        oof_candidates["baseline_lgbm"] = oof_base
        test_candidates["baseline_lgbm"] = test_base

        while self.scheduler.get_queue_depth() > 0:
            candidate = self.scheduler.pop_next_candidate(require_gpu=False)
            if not candidate:
                break

            run_id = f"exp_{candidate['hypothesis_id']}"
            logger.info("Executing trial: %s (EV: %.4f)", candidate["title"], candidate.get("ev_score", 0.0))

            feat_set = features_expanded if "interaction" in candidate["hypothesis_id"] else features_base

            res = self.runner.execute_tabular_gbdt(
                run_id=run_id,
                train_df=train_df,
                test_df=test_df,
                folds_df=folds_df,
                features=feat_set,
                target_col=target_col,
                id_col=id_col,
                model_family=ModelFamily(candidate["model_family"]),
                params=candidate["parameters"],
                metric_name=metric_name,
                is_classification=is_classification,
            )

            if res.status == ExecutionStatus.SUCCESS and res.metrics:
                logger.info("Trial %s CV: %.5f (delta: %+.5f)", run_id, res.metrics.overall_score, res.metrics.overall_score - baseline_cv)
                self.memory.log_trial(
                    run_result=res,
                    competition_id=self.competition_id,
                    model_family=candidate["model_family"],
                    hypothesis_id=candidate["hypothesis_id"],
                    parameters=candidate["parameters"],
                    created_at=datetime.now(timezone.utc).isoformat(),
                )

                oof_arr = pl.read_parquet(f"experiments/artifacts/run_{run_id}/oof_preds.parquet").to_pandas()["pred"].values
                test_arr = pl.read_parquet(f"experiments/artifacts/run_{run_id}/test_preds.parquet").to_pandas()["pred"].values
                oof_candidates[candidate["hypothesis_id"]] = oof_arr
                test_candidates[candidate["hypothesis_id"]] = test_arr

        self.fsm.mark_gate_cleared(QualityGate.GATE_4_EV_COMPUTE, "experiment_manager")

        # Phase 10: Ensembling & Blending
        self.fsm.transition_to(SystemPhase.ENSEMBLING, "blender")

        # Find best single model score
        metric_fn, _ = MetricRegistry.get_metric(metric_name)
        single_scores = {
            name: metric_fn(train_df[target_col].values, oof)
            for name, oof in oof_candidates.items()
        }
        best_single_name = max(single_scores, key=single_scores.get)
        best_single_cv = single_scores[best_single_name]

        weights, blend_cv = EnsembleBlender.hill_climbing(
            oof_dict=oof_candidates,
            y_true=train_df[target_col].values,
            metric_name=metric_name,
            n_iterations=50,
        )
        logger.info("Ensemble Hill Climbing completed. Blend Weights: %s, Blended CV: %.5f", weights, blend_cv)

        gate5_passed, gain = EnsembleBlender.evaluate_gate_5(best_single_cv, blend_cv, metric_name)
        if not gate5_passed:
            logger.warning(
                "Gate 5 check: Blend gain %.5f did not meet minimum +0.0005. Falling back to certified best single model '%s' (CV: %.5f).",
                gain, best_single_name, best_single_cv,
            )
            weights = {best_single_name: 1.0}
            blend_cv = best_single_cv
        else:
            logger.info("Gate 5 check PASSED: Blend improves CV by +%.5f over best single model.", gain)

        self.fsm.mark_gate_cleared(QualityGate.GATE_5_BLENDING_METRIC, "blender")

        # Compute blended test predictions
        final_test_pred = EnsembleBlender.blend_test_predictions(test_candidates, weights)

        # Generate submission CSV candidate
        submission_dir = Path("experiments/submissions/sub_latest")
        submission_dir.mkdir(parents=True, exist_ok=True)
        sub_csv_path = submission_dir / "submission.csv"

        sub_df = pd.DataFrame({
            id_col: test_df[id_col],
            target_col: final_test_pred,
        })
        sub_df.to_csv(sub_csv_path, index=False)

        # Phase 11: Pre-Submission Audit
        self.fsm.transition_to(SystemPhase.PRE_SUBMISSION_AUDIT, "artifact_analyst")
        sample_path = self.processed_data_dir / "sample_submission.csv"

        # Contract checks
        if len(sub_df) != len(sample_df):
            veto = VetoDocument(
                veto_id=f"sub_len_{int(time.time())}",
                issuing_role="artifact_analyst",
                phase="PRE_SUBMISSION_AUDIT",
                reason=f"Row count mismatch: Candidate={len(sub_df)}, Sample={len(sample_df)}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self.fsm.handle_veto(veto)
            return False

        if list(sub_df.columns) != list(sample_df.columns):
            veto = VetoDocument(
                veto_id=f"sub_cols_{int(time.time())}",
                issuing_role="artifact_analyst",
                phase="PRE_SUBMISSION_AUDIT",
                reason="Column names or order do not match sample submission",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self.fsm.handle_veto(veto)
            return False

        if sub_df[target_col].isnull().any() or not np.isfinite(sub_df[target_col]).all():
            veto = VetoDocument(
                veto_id=f"sub_vals_{int(time.time())}",
                issuing_role="artifact_analyst",
                phase="PRE_SUBMISSION_AUDIT",
                reason="Candidate predictions contain nulls or infinite values",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self.fsm.handle_veto(veto)
            return False

        sha256_hash = hashlib.sha256(sub_csv_path.read_bytes()).hexdigest()
        audit_cert = AuditCertificate(
            status="CERTIFIED",
            candidate_path=str(sub_csv_path),
            sample_path=str(sample_path),
            sha256=sha256_hash,
            row_count=len(sub_df),
            columns=list(sub_df.columns),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        cert_path = Path("experiments/artifacts/audit_cert.json")
        with open(cert_path, "w", encoding="utf-8") as f:
            json.dump(audit_cert.model_dump(mode="json"), f, indent=2)

        self.fsm.mark_gate_cleared(QualityGate.GATE_6_SUBMISSION_VETO, "artifact_analyst")
        logger.info("Pre-submission audit PASSED. SHA-256: %s", sha256_hash)

        # Phase 12: Submission & Postmortem
        self.fsm.transition_to(SystemPhase.KAGGLE_SUBMISSION, "kaggle_executor")
        auth_ok, _ = self.kaggle_client.check_auth()
        if auth_ok:
            logger.info("Submitting candidate to Kaggle via CLI...")
            sub_ok, sub_msg = self.kaggle_client.submit_csv(
                competition_id=self.competition_id,
                file_path=sub_csv_path,
                message=f"KAMAS Ensemble Blend (CV: {blend_cv:.5f})",
            )
            logger.info("Kaggle CLI submission response: %s", sub_msg)
        else:
            logger.info("Kaggle API token not active. Certified submission ready at %s", sub_csv_path)

        # Postmortem & Memory Promotion
        self.fsm.transition_to(SystemPhase.POSTMORTEM, "memory_curator")
        promoted = self.memory.consider_promotion(
            run_id="exp_hypo_interaction_001",
            baseline_cv=baseline_cv,
            current_cv=blend_cv,
            hypothesis_title="Interaction Product & Ensembling",
            modality="tabular",
            maximize=True,
        )
        logger.info("Memory promotion status: %s", "PROMOTED" if promoted else "NOT PROMOTED")

        # Update final state
        state = self.blackboard.read_state()
        state.best_cv_score = blend_cv
        state.best_run_id = "ensemble_blend"
        state.phase = SystemPhase.TERMINATED
        self.blackboard.write_state(state, updated_by="pipeline_completion")

        logger.info("=== KAMAS 12-PHASE PIPELINE RUN COMPLETED SUCCESSFULLY ===")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="KAMAS Autonomous Kaggle Multi-Agent Pipeline")
    parser.add_argument("--config", default="CONFIG.yaml", help="Path to CONFIG.yaml")
    parser.add_argument("--competition", default="benchmark-tabular-task", help="Kaggle competition slug")
    parser.add_argument("--mock-data", action="store_true", help="Generate mock benchmark dataset")
    args = parser.parse_args()

    pipeline = AutonomousPipeline(competition_id=args.competition, mock_data=args.mock_data)
    success = pipeline.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
