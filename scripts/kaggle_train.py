#!/usr/bin/env python3
"""
Kaggle Notebook Execution Runner for playground-series-s6e9.
Self-contained, robust pipeline entrypoint designed for flawless execution inside Kaggle Notebooks
or local environments.

Usage inside Kaggle Notebook cell:
    !python scripts/kaggle_train.py
or
    !python scripts/kaggle_train.py --data-dir /kaggle/input/playground-series-s6e9 --output-dir /kaggle/working
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
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from evaluation.metrics import MetricRegistry
from kaggle.paths import resolve_data_dir, resolve_output_dir


def load_data(data_dir: Path | str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, Path]:
    """Loads train, test, and sample submission from Parquet or CSV with auto-discovery."""
    resolved_dir = resolve_data_dir(data_dir)
    print(f"[*] Ingesting data from: {resolved_dir}")

    # Load Train
    if (resolved_dir / "train.parquet").exists():
        train_df = pl.read_parquet(resolved_dir / "train.parquet").to_pandas()
    elif (resolved_dir / "train.csv").exists():
        train_df = pl.read_csv(resolved_dir / "train.csv").to_pandas()
    else:
        raise FileNotFoundError(f"Neither train.parquet nor train.csv found in {resolved_dir}")

    # Load Test
    if (resolved_dir / "test.parquet").exists():
        test_df = pl.read_parquet(resolved_dir / "test.parquet").to_pandas()
    elif (resolved_dir / "test.csv").exists():
        test_df = pl.read_csv(resolved_dir / "test.csv").to_pandas()
    else:
        raise FileNotFoundError(f"Neither test.parquet nor test.csv found in {resolved_dir}")

    # Load Sample Submission (optional reference)
    sample_df = None
    if (resolved_dir / "sample_submission.parquet").exists():
        sample_df = pl.read_parquet(resolved_dir / "sample_submission.parquet").to_pandas()
    elif (resolved_dir / "sample_submission.csv").exists():
        sample_df = pd.read_csv(resolved_dir / "sample_submission.csv")

    print(f"[+] Loaded train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df, sample_df, resolved_dir


def get_or_create_folds(train_df: pd.DataFrame, data_dir: Path, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Retrieves certified fold assignments or generates deterministic StratifiedKFold."""
    folds_parquet = data_dir / "folds.parquet"
    if folds_parquet.exists():
        print(f"[*] Loading pre-computed certified folds from {folds_parquet}")
        f_df = pl.read_parquet(folds_parquet).to_pandas()
        merged = train_df[["id"]].merge(f_df, on="id", how="left")
        return merged["fold"].to_numpy()

    print(f"[*] Generating deterministic {n_splits}-fold StratifiedKFold (seed={seed})...")
    y_raw = train_df["Will_Buy_EV"]
    y_bin = (y_raw == "Yes").to_numpy().astype(int) if y_raw.dtype == object else y_raw.to_numpy().astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_arr = np.empty(len(train_df), dtype=np.int32)
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, y_bin)):
        fold_arr[val_idx] = fold_idx

    return fold_arr


def train_and_predict(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    n_splits: int = 5,
    seed: int = 42,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    n_estimators: int = 1000,
) -> dict:
    resolved_out = output_dir if output_dir else resolve_output_dir()
    resolved_out.mkdir(parents=True, exist_ok=True)
    train_df, test_df, sample_df, resolved_data_dir = load_data(data_dir)

    target_col = "Will_Buy_EV"
    id_col = "id"
    features = [c for c in test_df.columns if c != id_col]

    # Assign folds
    train_df["fold"] = get_or_create_folds(train_df, resolved_data_dir, n_splits=n_splits, seed=seed)

    # Convert categoricals to pandas category dtype for LightGBM
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(train_df[c])]
    for c in cat_cols:
        train_df[c] = train_df[c].astype("category")
        test_df[c] = test_df[c].astype("category")

    y_train = (train_df[target_col] == "Yes").to_numpy().astype(int)

    oof_preds = np.zeros(len(train_df), dtype=np.float64)
    test_preds = np.zeros(len(test_df), dtype=np.float64)
    fold_scores = []
    feature_importances = np.zeros(len(features), dtype=np.float64)

    print(f"[*] Training {n_splits}-fold LightGBM on {len(features)} features...")
    print(f"    Categorical features ({len(cat_cols)}): {cat_cols}")

    for fold in range(n_splits):
        tr_mask = train_df["fold"] != fold
        va_mask = train_df["fold"] == fold

        X_tr, y_tr = train_df.loc[tr_mask, features], y_train[tr_mask]
        X_va, y_va = train_df.loc[va_mask, features], y_train[va_mask]

        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            random_state=seed + fold,
            verbose=-1,
            n_jobs=-1,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )

        val_prob = model.predict_proba(X_va)[:, 1]
        test_prob = model.predict_proba(test_df[features])[:, 1]

        oof_preds[va_mask] = val_prob
        test_preds += test_prob / n_splits
        feature_importances += model.feature_importances_ / n_splits

        fold_auc = MetricRegistry.roc_auc(y_va, val_prob)
        fold_scores.append(fold_auc)
        print(f"  [Fold {fold+1}/{n_splits}] Val ROC-AUC: {fold_auc:.6f}")

    overall_auc = MetricRegistry.roc_auc(y_train, oof_preds)
    std_auc = float(np.std(fold_scores))
    print("=" * 60)
    print(f"[+] OVERALL 5-FOLD OOF ROC-AUC: {overall_auc:.6f} (+/- {std_auc:.6f})")
    print("=" * 60)

    # 1. Write Submission File (Gate 6 Strict Compliance)
    sub_path = resolved_out / "submission.csv"
    sub_df = pd.DataFrame({
        id_col: test_df[id_col],
        target_col: test_preds,
    })
    sub_df.to_csv(sub_path, index=False)
    print(f"[+] Submission file written to: {sub_path} ({sub_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # In Kaggle notebooks, also ensure /kaggle/working/submission.csv is directly accessible
    if Path("/kaggle/working").exists() and sub_path != Path("/kaggle/working/submission.csv"):
        try:
            sub_df.to_csv("/kaggle/working/submission.csv", index=False)
            print("[+] Also mirrored submission directly to /kaggle/working/submission.csv")
        except Exception:
            pass

    # Validate Submission
    assert len(sub_df) == len(test_df), f"Row count mismatch: {len(sub_df)} vs {len(test_df)}"
    assert list(sub_df.columns) == ["id", "Will_Buy_EV"], f"Invalid columns: {sub_df.columns}"
    assert not sub_df["Will_Buy_EV"].isnull().any(), "Submission contains null/NaN values!"
    assert (sub_df["Will_Buy_EV"] >= 0.0).all() and (sub_df["Will_Buy_EV"] <= 1.0).all(), "Predictions out of [0, 1] bounds!"
    print("[+] Submission verification PASSED: 0 nulls, correct headers, valid probability bounds.")

    # 2. Write OOF Predictions & Test Predictions
    oof_out = resolved_out / "oof_preds.parquet"
    pl.DataFrame({
        id_col: train_df[id_col],
        "pred": oof_preds,
        target_col: train_df[target_col],
        "fold": train_df["fold"],
    }).write_parquet(oof_out, compression="zstd")

    test_out = resolved_out / "test_preds.parquet"
    pl.DataFrame({
        id_col: test_df[id_col],
        "pred": test_preds,
    }).write_parquet(test_out, compression="zstd")
    print(f"[+] Test predictions written to: {test_out}")

    # 3. Write Metrics Summary
    metrics_summary = {
        "competition_id": "playground-series-s6e9",
        "metric_name": "roc_auc",
        "overall_cv": float(overall_auc),
        "std_cv": float(std_auc),
        "fold_scores": [float(s) for s in fold_scores],
        "n_samples_train": len(train_df),
        "n_samples_test": len(test_df),
    }
    with open(resolved_out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # 4. Write Feature Importances
    feat_imp_dict = {f: float(imp) for f, imp in sorted(zip(features, feature_importances), key=lambda x: x[1], reverse=True)}
    with open(resolved_out / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(feat_imp_dict, f, indent=2)

    print(f"[+] All artifacts successfully generated in: {resolved_out}")
    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Kaggle Notebook GBDT Training Runner")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to raw/processed data")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for submissions & artifacts")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV splits")
    parser.add_argument("--n-estimators", type=int, default=1000, help="Max trees per fold")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else resolve_data_dir()
    output_dir = Path(args.output_dir) if args.output_dir else resolve_output_dir()

    print(f"Data Dir:   {data_dir}")
    print(f"Output Dir: {output_dir}")

    train_and_predict(
        data_dir=data_dir,
        output_dir=output_dir,
        n_splits=args.n_splits,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
