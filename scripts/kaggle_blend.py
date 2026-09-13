#!/usr/bin/env python3
"""
Kaggle Notebook Phase 10 Blending & Ensembling Engine for playground-series-s6e9.
Auto-discovers all trained models in /kaggle/working/models/, computes:
1. Out-of-fold correlation matrix between candidate models
2. Rank-averaged ensemble
3. Scipy Nelder-Mead optimal weight blend on ROC-AUC
4. Generates certified final submission.csv with Gate 5 metric validation.

Usage in Kaggle notebook:
    !python scripts/kaggle_blend.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from kaggle.paths import resolve_output_dir


def rank_average(pred_matrix: np.ndarray) -> np.ndarray:
    """Computes normalized percentile rank average across model predictions."""
    ranks = np.zeros_like(pred_matrix)
    for col_idx in range(pred_matrix.shape[1]):
        ranks[:, col_idx] = rankdata(pred_matrix[:, col_idx]) / len(pred_matrix)
    return np.mean(ranks, axis=1)


def optimize_weights(oof_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Optimizes linear blend weights using Nelder-Mead to maximize ROC-AUC."""
    n_models = oof_matrix.shape[1]
    init_weights = np.ones(n_models) / n_models

    def loss(weights):
        # Softmax normalize to ensure weights are non-negative and sum to 1
        w = np.exp(weights) / np.sum(np.exp(weights))
        blend_oof = oof_matrix @ w
        return -roc_auc_score(y_true, blend_oof)

    res = minimize(loss, init_weights, method="Nelder-Mead", options={"maxiter": 500})
    best_w = np.exp(res.x) / np.sum(np.exp(res.x))
    return best_w


def main():
    parser = argparse.ArgumentParser(description="Ensemble Blending Engine")
    parser.add_argument("--models-dir", type=Path, default=None, help="Directory containing model subfolders")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output destination")
    args = parser.parse_args()

    base_out = args.output_dir if args.output_dir else resolve_output_dir()
    models_dir = args.models_dir if args.models_dir else (base_out / "models")

    print("=" * 70)
    print("PHASE 10: ENSEMBLE & HILL-CLIMBING BLENDER")
    print(f"Scanning for trained models in: {models_dir}")
    print("=" * 70)

    if not models_dir.exists():
        print(f"[!] No models directory found at {models_dir}")
        print("    Train models first using: !python scripts/kaggle_train.py --model lgbm")
        sys.exit(1)

    model_subdirs = [d for d in models_dir.iterdir() if d.is_dir() and (d / "oof_preds.parquet").exists()]
    if not model_subdirs:
        print(f"[!] No valid model artifacts with oof_preds.parquet found in {models_dir}")
        sys.exit(1)

    print(f"[+] Found {len(model_subdirs)} candidate model(s): {[d.name for d in model_subdirs]}")

    model_names = []
    oof_list = []
    test_list = []
    y_true = None
    test_ids = None

    for d in sorted(model_subdirs):
        oof_df = pl.read_parquet(d / "oof_preds.parquet").to_pandas()
        test_df = pl.read_parquet(d / "test_preds.parquet").to_pandas()

        if y_true is None:
            y_true = (oof_df["Will_Buy_EV"] == "Yes").to_numpy().astype(int)
            test_ids = test_df["id"].to_numpy()

        oof_list.append(oof_df["pred"].to_numpy())
        test_list.append(test_df["pred"].to_numpy())
        model_names.append(d.name)

        single_auc = roc_auc_score(y_true, oof_df["pred"].to_numpy())
        print(f"  - Model: {d.name:<25} | Single OOF ROC-AUC: {single_auc:.6f}")

    oof_matrix = np.column_stack(oof_list)
    test_matrix = np.column_stack(test_list)

    # 1. Pearson correlation between model predictions
    if len(model_names) > 1:
        corr_df = pd.DataFrame(oof_matrix, columns=model_names).corr()
        print("\n[*] Out-of-Fold Prediction Correlation Matrix:")
        print(corr_df.to_string())

    best_single_auc = max(roc_auc_score(y_true, oof_matrix[:, i]) for i in range(len(model_names)))

    if len(model_names) == 1:
        print("\n[*] Only 1 model available. Writing single model predictions to submission.csv.")
        final_test_preds = test_matrix[:, 0]
        ensemble_auc = best_single_auc
        chosen_strategy = "single_model"
        weights_dict = {model_names[0]: 1.0}
    else:
        # Strategy A: Rank Averaging
        rank_oof = rank_average(oof_matrix)
        rank_auc = roc_auc_score(y_true, rank_oof)
        print(f"\n[*] Strategy A (Rank Average)   OOF ROC-AUC: {rank_auc:.6f} (Δ: {rank_auc - best_single_auc:+.6f})")

        # Strategy B: Optimized Weighted Blend
        opt_weights = optimize_weights(oof_matrix, y_true)
        opt_oof = oof_matrix @ opt_weights
        opt_auc = roc_auc_score(y_true, opt_oof)
        print(f"[*] Strategy B (Optimized Blend) OOF ROC-AUC: {opt_auc:.6f} (Δ: {opt_auc - best_single_auc:+.6f})")
        print(f"    Optimal Weights: {dict(zip(model_names, [round(float(w), 4) for w in opt_weights]))}")

        if opt_auc >= rank_auc:
            chosen_strategy = "optimized_weighted_blend"
            ensemble_auc = opt_auc
            final_test_preds = test_matrix @ opt_weights
            weights_dict = dict(zip(model_names, [float(w) for w in opt_weights]))
        else:
            chosen_strategy = "rank_average"
            ensemble_auc = rank_auc
            final_test_preds = rank_average(test_matrix)
            weights_dict = {m: 1.0 / len(model_names) for m in model_names}

    delta_cv = ensemble_auc - best_single_auc
    print("=" * 70)
    print(f"[+] CHOSEN ENSEMBLE STRATEGY: {chosen_strategy.upper()}")
    print(f"[+] ENSEMBLE OOF ROC-AUC:      {ensemble_auc:.6f}")
    print(f"[+] DELTA OVER BEST SINGLE:    {delta_cv:+.6f}")
    print("=" * 70)

    # Output Ensemble Submission
    ensemble_dir = base_out / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    sub_df = pd.DataFrame({
        "id": test_ids,
        "Will_Buy_EV": final_test_preds,
    })

    # Save to ensemble dir
    sub_df.to_csv(ensemble_dir / "submission.csv", index=False)

    # Save directly to /kaggle/working/submission.csv for 1-click submit
    root_sub = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else base_out / "submission.csv"
    sub_df.to_csv(root_sub, index=False)
    print(f"[+] Submission file written to: {root_sub}")

    # Verify submission integrity
    assert len(sub_df) == 286571, f"Expected 286,571 rows, got {len(sub_df)}"
    assert not sub_df["Will_Buy_EV"].isnull().any(), "Submission contains NaN values!"
    assert (sub_df["Will_Buy_EV"] >= 0.0).all() and (sub_df["Will_Buy_EV"] <= 1.0).all(), "Predictions out of [0, 1]!"

    # Save ensemble metadata
    meta = {
        "competition_id": "playground-series-s6e9",
        "ensemble_strategy": chosen_strategy,
        "candidate_models": model_names,
        "weights": weights_dict,
        "best_single_auc": float(best_single_auc),
        "ensemble_auc": float(ensemble_auc),
        "cv_delta": float(delta_cv),
    }
    with open(ensemble_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[+] Verification PASSED. Submission ready for Kaggle!")


if __name__ == "__main__":
    main()
