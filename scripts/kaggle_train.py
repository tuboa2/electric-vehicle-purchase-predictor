#!/usr/bin/env python3
"""
Kaggle Notebook Multi-Model Training & Prediction Runner for playground-series-s6e9.
Supports Phase 8 (Feature Engineering) & Phase 9 (Diverse Model Exploration):
- LightGBM (leaf-wise GBDT)
- CatBoost (symmetric oblivious trees with native categorical handling & GPU support)
- XGBoost (histogram-based depth-wise GBDT with GPU support)

Usage in Kaggle notebook:
    !python scripts/kaggle_train.py --model lgbm --features domain
    !python scripts/kaggle_train.py --model catboost --features domain
    !python scripts/kaggle_train.py --model xgboost --features domain
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features.domain_features import generate_domain_features
from kaggle.paths import resolve_data_dir, resolve_output_dir


def detect_gpu() -> bool:
    """Checks if NVIDIA GPU is available for acceleration."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[+] GPU detected: {torch.cuda.get_device_name(0)}")
            return True
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        print("[+] GPU detected via nvidia-smi")
        return True
    print("[-] No GPU detected. Running on CPU.")
    return False


def load_data(data_dir: Path | str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, Path]:
    """Loads train, test, and sample submission with auto-discovery."""
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

    # Load Sample Submission
    sample_df = None
    if (resolved_dir / "sample_submission.parquet").exists():
        sample_df = pl.read_parquet(resolved_dir / "sample_submission.parquet").to_pandas()
    elif (resolved_dir / "sample_submission.csv").exists():
        sample_df = pd.read_csv(resolved_dir / "sample_submission.csv")

    print(f"[+] Loaded train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df, sample_df, resolved_dir


def get_or_create_folds(train_df: pd.DataFrame, data_dir: Path, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Uses precomputed certified folds if available, else builds StratifiedKFold."""
    folds_path = data_dir / "folds.parquet"
    if folds_path.exists():
        folds_df = pl.read_parquet(folds_path).to_pandas()
        if "fold" in folds_df.columns and len(folds_df) == len(train_df):
            print(f"[+] Using certified folds from {folds_path}")
            return folds_df["fold"].to_numpy()

    print(f"[*] Generating {n_splits}-fold StratifiedKFold (seed={seed})...")
    y_bin = (train_df["Will_Buy_EV"] == "Yes").astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_arr = np.empty(len(train_df), dtype=np.int32)
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, y_bin)):
        fold_arr[val_idx] = fold_idx
    return fold_arr


def train_lgbm(X_tr, y_tr, X_va, y_va, X_te, cat_cols, seed, fold):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        random_state=seed + fold,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )
    val_prob = model.predict_proba(X_va)[:, 1]
    te_prob = model.predict_proba(X_te)[:, 1]
    return val_prob, te_prob, model.feature_importances_


def train_catboost(X_tr, y_tr, X_va, y_va, X_te, cat_cols, seed, fold, use_gpu):
    from catboost import CatBoostClassifier
    task_type = "GPU" if use_gpu else "CPU"
    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=seed + fold,
        task_type=task_type,
        verbose=False,
        cat_features=cat_cols,
        early_stopping_rounds=50,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    val_prob = model.predict_proba(X_va)[:, 1]
    te_prob = model.predict_proba(X_te)[:, 1]
    feat_imp = model.get_feature_importance()
    return val_prob, te_prob, feat_imp


def train_xgboost(X_tr, y_tr, X_va, y_va, X_te, cat_cols, seed, fold, use_gpu):
    import xgboost as xgb
    # For XGBoost, convert categoricals to pandas category dtype or enable categorical support
    device = "cuda" if use_gpu else "cpu"
    model = xgb.XGBClassifier(
        n_estimators=1200,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed + fold,
        tree_method="hist",
        device=device,
        enable_categorical=True,
        early_stopping_rounds=40,
        eval_metric="auc",
        n_jobs=-1 if not use_gpu else 1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    val_prob = model.predict_proba(X_va)[:, 1]
    te_prob = model.predict_proba(X_te)[:, 1]
    return val_prob, te_prob, model.feature_importances_


def train_and_predict(
    model_name: str = "lgbm",
    features_type: str = "domain",
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    start_time = time.time()
    has_gpu = detect_gpu()

    # Destination output directory
    base_out = output_dir if output_dir else resolve_output_dir()
    model_dir = base_out / "models" / f"{model_name}_{features_type}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_df, test_df, sample_df, resolved_data_dir = load_data(data_dir)
    target_col = "Will_Buy_EV"
    id_col = "id"

    # 2. Feature Engineering
    if features_type == "domain":
        print("[*] Applying Domain Feature Transformations (Hypothesis 1)...")
        train_df, test_df, new_cols = generate_domain_features(train_df, test_df)
        print(f"[+] Injected {len(new_cols)} engineered features: {new_cols}")
    else:
        print("[*] Using Raw Features (Baseline mode).")

    features = [c for c in test_df.columns if c != id_col]
    train_df["fold"] = get_or_create_folds(train_df, resolved_data_dir, n_splits=n_splits, seed=seed)

    # Convert categoricals
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(train_df[c])]
    print(f"[*] Total Features: {len(features)} | Categorical: {cat_cols}")

    if model_name in ["lgbm", "xgboost"]:
        for c in cat_cols:
            train_df[c] = train_df[c].astype("category")
            test_df[c] = test_df[c].astype("category")
    elif model_name == "catboost":
        for c in cat_cols:
            train_df[c] = train_df[c].astype(str)
            test_df[c] = test_df[c].astype(str)

    y_train = (train_df[target_col] == "Yes").to_numpy().astype(int)

    oof_preds = np.zeros(len(train_df), dtype=np.float64)
    test_preds = np.zeros(len(test_df), dtype=np.float64)
    fold_scores = []
    feature_importances = np.zeros(len(features), dtype=np.float64)

    print("=" * 65)
    print(f"[*] TRAINING {n_splits}-FOLD {model_name.upper()} ({features_type.upper()} FEATURES)")
    print(f"    Accelerator: {'GPU' if has_gpu and model_name in ['catboost', 'xgboost'] else 'CPU'}")
    print("=" * 65)

    for fold in range(n_splits):
        f_start = time.time()
        tr_mask = train_df["fold"] != fold
        va_mask = train_df["fold"] == fold

        X_tr, y_tr = train_df.loc[tr_mask, features], y_train[tr_mask]
        X_va, y_va = train_df.loc[va_mask, features], y_train[va_mask]
        X_te = test_df[features]

        if model_name == "lgbm":
            v_prob, t_prob, imp = train_lgbm(X_tr, y_tr, X_va, y_va, X_te, cat_cols, seed, fold)
        elif model_name == "catboost":
            v_prob, t_prob, imp = train_catboost(X_tr, y_tr, X_va, y_va, X_te, cat_cols, seed, fold, has_gpu)
        elif model_name == "xgboost":
            v_prob, t_prob, imp = train_xgboost(X_tr, y_tr, X_va, y_va, X_te, cat_cols, seed, fold, has_gpu)
        else:
            raise ValueError(f"Unknown model: {model_name}. Supported: lgbm, catboost, xgboost")

        oof_preds[va_mask] = v_prob
        test_preds += t_prob / n_splits
        feature_importances += imp / n_splits

        fold_auc = roc_auc_score(y_va, v_prob)
        fold_scores.append(fold_auc)
        f_dur = time.time() - f_start
        print(f"  [Fold {fold+1}/{n_splits}] ROC-AUC: {fold_auc:.6f} ({f_dur:.1f}s)")

    overall_auc = roc_auc_score(y_train, oof_preds)
    std_auc = float(np.std(fold_scores))
    total_time = time.time() - start_time

    print("=" * 65)
    print(f"[+] OVERALL 5-FOLD OOF ROC-AUC: {overall_auc:.6f} (+/- {std_auc:.6f})")
    print(f"[+] Training duration: {total_time:.1f}s ({total_time / 60:.2f} min)")
    print("=" * 65)

    # 1. Write Submission File to model_dir AND /kaggle/working/submission.csv
    sub_df = pd.DataFrame({
        id_col: test_df[id_col],
        target_col: test_preds,
    })
    sub_path = model_dir / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    # Always ensure root /kaggle/working/submission.csv has the latest submission for 1-click UI
    root_sub = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else base_out / "submission.csv"
    sub_df.to_csv(root_sub, index=False)
    print(f"[+] Updated primary submission file: {root_sub}")

    # 2. Write OOF & Test Predictions (.parquet)
    oof_out = model_dir / "oof_preds.parquet"
    pl.DataFrame({
        id_col: train_df[id_col],
        "pred": oof_preds,
        target_col: train_df[target_col],
        "fold": train_df["fold"],
    }).write_parquet(oof_out, compression="zstd")

    test_out = model_dir / "test_preds.parquet"
    pl.DataFrame({
        id_col: test_df[id_col],
        "pred": test_preds,
    }).write_parquet(test_out, compression="zstd")

    # 3. Write Metrics Summary
    metrics_summary = {
        "competition_id": "playground-series-s6e9",
        "model_name": model_name,
        "features_type": features_type,
        "metric_name": "roc_auc",
        "overall_cv": float(overall_auc),
        "std_cv": float(std_auc),
        "fold_scores": [float(s) for s in fold_scores],
        "n_samples_train": len(train_df),
        "n_samples_test": len(test_df),
        "features": features,
        "execution_time_seconds": total_time,
    }
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # 4. Write Feature Importances
    feat_imp_dict = {f: float(imp) for f, imp in sorted(zip(features, feature_importances), key=lambda x: x[1], reverse=True)}
    with open(model_dir / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(feat_imp_dict, f, indent=2)

    print(f"[+] Model artifacts saved in: {model_dir}")
    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Phase 8/9 Multi-Model Trainer for playground-series-s6e9")
    parser.add_argument("--model", type=str, default="lgbm", choices=["lgbm", "catboost", "xgboost"],
                        help="Model architecture to train (lgbm, catboost, xgboost)")
    parser.add_argument("--features", type=str, default="domain", choices=["domain", "raw"],
                        help="Feature set: 'domain' (engineered interactions) or 'raw'")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Optional explicit path to dataset")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional explicit output directory")
    parser.add_argument("--splits", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    train_and_predict(
        model_name=args.model,
        features_type=args.features,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_splits=args.splits,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
