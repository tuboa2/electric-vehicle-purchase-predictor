"""
Grandmaster Training Pipeline for playground-series-s6e9.
Implements the 0.9463+ synthetic artifact exploit pipeline:
- Digit Decomposition (10^-4 to 10^3)
- Smooth Keys (Floor-binned numerics)
- Hard Boundary Flags (Millionaire cliff, Dead zone, 30k spike, Env hater)
- Original Dataset Target Mean Priors
- Global Frequency Encodings
- Triple Sklearn Target Encoding (auto, smooth=10.0, smooth=100.0)
- High-Resolution Gradient Boosted Trees (max_bin=1024, colsample=0.3)
- Automatic GPU Acceleration & Single-Command Turnkey Execution
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

# Tree libraries are imported dynamically inside train_single_model

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.grandmaster_features import build_grandmaster_features, TARGET


def detect_gpu() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[+] GPU detected: {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        pass
    print("[-] No CUDA GPU detected. Running on CPU.")
    return False


def locate_data_dir() -> Path:
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        dirs = [d for d in kaggle_input.iterdir() if d.is_dir()]
        for d in dirs:
            if d.name == "competitions":
                comp_dir = d / "playground-series-s6e9"
                if comp_dir.exists():
                    return comp_dir
            if "playground-series-s6e9" in d.name.lower():
                return d

    candidates = [
        PROJECT_ROOT / "data" / "processed" / "playground-series-s6e9",
        PROJECT_ROOT / "data" / "raw" / "playground-series-s6e9",
        PROJECT_ROOT / "data" / "raw",
    ]
    for c in candidates:
        if (c / "train.csv").exists() or (c / "train.parquet").exists():
            return c

    raise FileNotFoundError("Could not find competition dataset in /kaggle/input or project data/")


def load_dataset(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    # Train
    if (data_dir / "train.parquet").exists():
        train_df = pl.read_parquet(data_dir / "train.parquet").to_pandas()
    elif (data_dir / "train.csv").exists():
        train_df = pd.read_csv(data_dir / "train.csv")
    else:
        raise FileNotFoundError(f"train data not found in {data_dir}")

    # Test
    if (data_dir / "test.parquet").exists():
        test_df = pl.read_parquet(data_dir / "test.parquet").to_pandas()
    elif (data_dir / "test.csv").exists():
        test_df = pd.read_csv(data_dir / "test.csv")
    else:
        raise FileNotFoundError(f"test data not found in {data_dir}")

    # Original Dataset
    orig_candidates = [
        PROJECT_ROOT / "data" / "original" / "EV_Adoption_and_Range_Anxiety_Dataset.csv",
        Path("/kaggle/input/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"),
    ]
    orig_df = None
    for cand in orig_candidates:
        if cand.exists():
            orig_df = pd.read_csv(cand)
            print(f"[+] Ground-Truth Original Dataset loaded from: {cand} ({len(orig_df)} samples)")
            break

    return train_df, test_df, orig_df


def train_single_model(
    model_type: str,
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    features: List[str],
    te_cols: List[str],
    n_splits: int,
    seed: int,
    has_gpu: bool,
    output_dir: Path,
    primary_sub_path: Path,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model_name = f"{model_type}_grandmaster"
    print(f"\n=================================================================")
    print(f"[*] TRAINING {n_splits}-FOLD GRANDMASTER: {model_name.upper()}")
    print(f"    Accelerator: {'GPU' if has_gpu else 'CPU'}")
    print(f"=================================================================")

    X = train_feat[features].copy()
    y = train_feat[TARGET].values
    X_test = test_feat[features].copy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X), dtype=np.float64)
    test_preds = np.zeros(len(X_test), dtype=np.float64)

    start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        f_start = time.time()
        X_tr = X.iloc[train_idx].copy()
        y_tr = y[train_idx]
        X_va = X.iloc[val_idx].copy()
        y_va = y[val_idx]
        X_te = X_test.copy()

        # Triple Target Encoding on Categoricals & Smooth Bins
        if te_cols:
            te_auto = TargetEncoder(shuffle=True, cv=n_splits, smooth="auto", random_state=seed)
            te_10 = TargetEncoder(shuffle=True, cv=n_splits, smooth=10.0, random_state=seed)
            te_100 = TargetEncoder(shuffle=True, cv=n_splits, smooth=100.0, random_state=seed)

            tr_auto = te_auto.fit_transform(X_tr[te_cols], y_tr)
            va_auto = te_auto.transform(X_va[te_cols])
            te_auto_arr = te_auto.transform(X_te[te_cols])

            tr_10 = te_10.fit_transform(X_tr[te_cols], y_tr)
            va_10 = te_10.transform(X_va[te_cols])
            te_10_arr = te_10.transform(X_te[te_cols])

            tr_100 = te_100.fit_transform(X_tr[te_cols], y_tr)
            va_100 = te_100.transform(X_va[te_cols])
            te_100_arr = te_100.transform(X_te[te_cols])

            te_data_tr = {}
            te_data_va = {}
            te_data_te = {}
            for idx, col in enumerate(te_cols):
                te_data_tr[f"{col}_TE_auto"] = tr_auto[:, idx].astype("float32")
                te_data_va[f"{col}_TE_auto"] = va_auto[:, idx].astype("float32")
                te_data_te[f"{col}_TE_auto"] = te_auto_arr[:, idx].astype("float32")

                te_data_tr[f"{col}_TE_10"] = tr_10[:, idx].astype("float32")
                te_data_va[f"{col}_TE_10"] = va_10[:, idx].astype("float32")
                te_data_te[f"{col}_TE_10"] = te_10_arr[:, idx].astype("float32")

                te_data_tr[f"{col}_TE_100"] = tr_100[:, idx].astype("float32")
                te_data_va[f"{col}_TE_100"] = va_100[:, idx].astype("float32")
                te_data_te[f"{col}_TE_100"] = te_100_arr[:, idx].astype("float32")

            X_tr = pd.concat([X_tr.drop(columns=te_cols), pd.DataFrame(te_data_tr, index=X_tr.index)], axis=1)
            X_va = pd.concat([X_va.drop(columns=te_cols), pd.DataFrame(te_data_va, index=X_va.index)], axis=1)
            X_te = pd.concat([X_te.drop(columns=te_cols), pd.DataFrame(te_data_te, index=X_te.index)], axis=1)

        if model_type == "lgbm":
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(
                n_estimators=20000,
                learning_rate=0.02,
                max_depth=5,
                num_leaves=32,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.3,
                reg_alpha=0.071,
                reg_lambda=2.0,
                max_bin=1024,
                random_state=seed,
                feature_pre_filter=False,
                metric="auc",
                n_jobs=-1,
                verbose=-1,
            )
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=500, verbose=False),
                    lgb.log_evaluation(period=1000),
                ],
            )
            val_probs = clf.predict_proba(X_va)[:, 1]
            test_probs_fold = clf.predict_proba(X_te)[:, 1]

        elif model_type == "catboost":
            import catboost as cb
            clf = cb.CatBoostClassifier(
                iterations=10000,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=4.0,
                eval_metric="AUC",
                random_seed=seed,
                task_type="GPU" if has_gpu else "CPU",
                verbose=1000,
            )
            clf.fit(
                X_tr, y_tr,
                eval_set=(X_va, y_va),
                early_stopping_rounds=400,
                verbose=1000,
            )
            val_probs = clf.predict_proba(X_va)[:, 1]
            test_probs_fold = clf.predict_proba(X_te)[:, 1]

        elif model_type == "xgboost":
            import xgboost as xgb
            clf = xgb.XGBClassifier(
                n_estimators=15000,
                learning_rate=0.02,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.3,
                reg_alpha=0.071,
                reg_lambda=2.0,
                max_bin=1024,
                tree_method="hist",
                device="cuda" if has_gpu else "cpu",
                eval_metric="auc",
                early_stopping_rounds=500,
                random_state=seed,
            )
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=1000,
            )
            val_probs = clf.predict_proba(X_va)[:, 1]
            test_probs_fold = clf.predict_proba(X_te)[:, 1]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        oof_preds[val_idx] = val_probs
        test_preds += test_probs_fold / n_splits
        fold_auc = roc_auc_score(y_va, val_probs)
        print(f"  [Fold {fold}/{n_splits}] ROC-AUC: {fold_auc:.6f} ({time.time() - f_start:.1f}s)")

    overall_auc = roc_auc_score(y, oof_preds)
    duration = time.time() - start_time
    print(f"=================================================================")
    print(f"[+] OVERALL 5-FOLD OOF ROC-AUC: {overall_auc:.6f}")
    print(f"[+] Total training duration: {duration:.1f}s ({duration/60:.2f} min)")
    print(f"=================================================================")

    # Save artifacts
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    oof_df = pd.DataFrame({"id": train_feat["id"], "oof_pred": oof_preds, "target": y})
    oof_df.to_parquet(model_dir / "oof_preds.parquet", index=False)

    sub_df = pd.DataFrame({"id": test_feat["id"], TARGET: test_preds})
    sub_df.to_parquet(model_dir / "test_preds.parquet", index=False)
    sub_df.to_csv(model_dir / "submission.csv", index=False)

    meta = {
        "model_name": model_name,
        "model_type": model_type,
        "n_splits": n_splits,
        "seed": seed,
        "oof_auc": float(overall_auc),
        "duration_seconds": float(duration),
    }
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Update primary submission
    primary_sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(primary_sub_path, index=False)
    print(f"[+] Primary submission updated at: {primary_sub_path}")

    return overall_auc, oof_preds, test_preds


def blend_grandmaster_models(
    models_oof: Dict[str, np.ndarray],
    models_test: Dict[str, np.ndarray],
    y_true: np.ndarray,
    test_ids: pd.Series,
    output_dir: Path,
    primary_sub_path: Path,
):
    print(f"\n======================================================================")
    print(f"[*] EXECUTING GRANDMASTER OPTIMIZED BLEND")
    print(f"======================================================================")
    model_names = list(models_oof.keys())
    m = len(model_names)
    if m <= 1:
        print("[!] Only 1 model trained; skipping blend.")
        return

    oof_matrix = np.column_stack([models_oof[k] for k in model_names])
    test_matrix = np.column_stack([models_test[k] for k in model_names])

    # Objective: Minimize negative ROC-AUC with Nelder-Mead
    def loss_func(weights):
        w = np.array(weights)
        if np.sum(np.abs(w)) == 0:
            return 0.0
        w = w / np.sum(w)
        blend = np.dot(oof_matrix, w)
        return -roc_auc_score(y_true, blend)

    init_w = np.ones(m) / m
    res = minimize(
        loss_func,
        init_w,
        method="Nelder-Mead",
        bounds=[(0.0, 1.0)] * m,
        options={"maxiter": 500, "disp": False},
    )
    raw_w = np.clip(res.x, 0.0, None)
    best_weights = raw_w / np.sum(raw_w)
    weights_dict = {model_names[i]: float(best_weights[i]) for i in range(m)}

    blend_oof = np.dot(oof_matrix, best_weights)
    blend_test = np.dot(test_matrix, best_weights)
    blend_auc = roc_auc_score(y_true, blend_oof)

    best_single_auc = max(roc_auc_score(y_true, models_oof[k]) for k in model_names)
    delta_auc = blend_auc - best_single_auc

    print(f"[+] Optimal Blend Weights: {weights_dict}")
    print(f"[+] Best Single Model AUC:  {best_single_auc:.6f}")
    print(f"[+] Ensembled Grandmaster:  {blend_auc:.6f} (Δ: {delta_auc:+.6f})")

    # Save ensemble artifacts
    ensemble_dir = output_dir.parent / "ensemble_grandmaster"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    blend_sub = pd.DataFrame({"id": test_ids, TARGET: blend_test})
    blend_sub.to_csv(ensemble_dir / "submission.csv", index=False)
    blend_sub.to_csv(primary_sub_path, index=False)

    meta = {
        "ensemble_type": "grandmaster_nelder_mead",
        "models": model_names,
        "weights": weights_dict,
        "best_single_auc": float(best_single_auc),
        "ensemble_auc": float(blend_auc),
        "delta_auc": float(delta_auc),
    }
    with open(ensemble_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[+] Grandmaster blend submission written to: {primary_sub_path}")


def main():
    parser = argparse.ArgumentParser(description="Grandmaster EV Purchase Training")
    parser.add_argument("--model", type=str, choices=["lgbm", "catboost", "xgboost", "all"], default="lgbm")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sub-path", type=str, default=None)
    args = parser.parse_args()

    has_gpu = detect_gpu()
    data_dir = locate_data_dir()
    print(f"[+] Training data directory: {data_dir}")

    is_kaggle = Path("/kaggle/working").exists()
    default_out = Path("/kaggle/working/models") if is_kaggle else PROJECT_ROOT / "experiments" / "artifacts" / "models"
    default_sub = Path("/kaggle/working/submission.csv") if is_kaggle else PROJECT_ROOT / "submission.csv"

    output_dir = Path(args.output_dir) if args.output_dir else default_out
    sub_path = Path(args.sub_path) if args.sub_path else default_sub

    train_df, test_df, orig_df = load_dataset(data_dir)
    print(f"[+] Ingested train: {train_df.shape}, test: {test_df.shape}")

    print("[*] Generating Grandmaster Features (Digit Decomposition, Smooth Keys, Hard Boundaries)...")
    train_feat, test_feat, features, te_cols = build_grandmaster_features(train_df, test_df, orig_df)
    print(f"[+] Feature generation complete. Total features: {len(features)} | Target-encode cols: {len(te_cols)}")

    models_oof: Dict[str, np.ndarray] = {}
    models_test: Dict[str, np.ndarray] = {}

    models_to_run = ["lgbm", "catboost", "xgboost"] if args.model == "all" else [args.model]

    for m in models_to_run:
        auc, oof, test_p = train_single_model(
            model_type=m,
            train_feat=train_feat,
            test_feat=test_feat,
            features=features,
            te_cols=te_cols,
            n_splits=args.folds,
            seed=args.seed,
            has_gpu=has_gpu,
            output_dir=output_dir,
            primary_sub_path=sub_path,
        )
        models_oof[m] = oof
        models_test[m] = test_p

    if len(models_to_run) > 1:
        blend_grandmaster_models(
            models_oof=models_oof,
            models_test=models_test,
            y_true=train_feat[TARGET].values,
            test_ids=test_feat["id"],
            output_dir=output_dir,
            primary_sub_path=sub_path,
        )


if __name__ == "__main__":
    main()
