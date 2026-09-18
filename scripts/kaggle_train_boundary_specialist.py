"""
Boundary Specialist Training Script
Targets the specific boundary fracture region where the latent deterministic linear formula breaks down.
Train an XGBoost model explicitly on samples where |Buy Score - 5.61235| < 0.4.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.grandmaster_features import TARGET, build_grandmaster_features


def locate_data_dir() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/playground-series-s6e9"),
        Path("/kaggle/input/playground-series-s6e9"),
        PROJECT_ROOT / "data" / "processed" / "playground-series-s6e9",
        PROJECT_ROOT / "data" / "raw" / "playground-series-s6e9",
        PROJECT_ROOT / "data" / "raw",
    ]
    for c in candidates:
        if c.exists() and (c / "train.csv").exists():
            return c
    raise FileNotFoundError("Could not locate Kaggle dataset 'playground-series-s6e9'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=10, help="Number of CV folds")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=10, help="Number of CV folds")
    parser.add_argument(
        "--boundary_threshold",
        type=float,
        default=0.4,
        help="Absolute distance to boundary to include in training",
    )
    args = parser.parse_args()

    data_dir = locate_data_dir()
    print(f"[*] Loading data from {data_dir}")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    orig_path = data_dir / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
    orig = pd.read_csv(orig_path) if orig_path.exists() else None

    print("[*] Engineering Features via Grandmaster Module...")
    X_train_full, X_test_full, features, cat_cols = build_grandmaster_features(
        train, test, orig
    )
    y_train_full = (
        train[TARGET].map({"Yes": 1, "No": 0, "1": 1, "0": 0}).astype(int).values
    )

    print(
        f"[*] Extracting Boundary Subset (abs(feat_recipe_dist_to_boundary) < {args.boundary_threshold})"
    )

    # We must filter based on the engineered feature
    if "feat_recipe_abs_dist" not in X_train_full.columns:
        raise ValueError("feat_recipe_abs_dist not found in engineered features!")

    boundary_mask = X_train_full["feat_recipe_abs_dist"] < args.boundary_threshold

    X_train = X_train_full[boundary_mask].reset_index(drop=True)
    y_train = y_train_full[boundary_mask]

    print(
        f"[+] Full Train Size: {len(X_train_full)} | Boundary Train Size: {len(X_train)} ({(len(X_train) / len(X_train_full)) * 100:.1f}%)"
    )

    # For OOF, we only generate predictions for the boundary mask subset.
    # The final ensemble script will merge these back in.
    oof_preds = np.zeros(len(X_train))

    test_preds = np.zeros(len(X_test_full))

    import xgboost as xgb

    # Fast GPU detection
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except:
        has_gpu = False

    tree_method = "hist"
    device = "cuda" if has_gpu else "cpu"

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.03,
        "max_depth": 5,
        "colsample_bytree": 0.5,
        "subsample": 0.8,
        "min_child_weight": 20,
        "tree_method": tree_method,
        "device": device,
        "n_estimators": 500,  # slightly reduced for speed on local test
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": -1,
    }

    # Need to handle target encoding inside CV for the boundary subset
    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    models = []

    print(f"[*] Starting Cross-Validation on Boundary Subset on {device}...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"\n--- Fold {fold + 1}/{args.folds} ---")
        X_tr, y_tr = X_train.iloc[train_idx].copy(), y_train[train_idx]
        X_va, y_va = X_train.iloc[val_idx].copy(), y_train[val_idx]

        # Target Encode
        if cat_cols:
            te = TargetEncoder(target_type="binary", random_state=42, cv=3)
            X_tr[cat_cols] = te.fit_transform(X_tr[cat_cols], y_tr)
            X_va[cat_cols] = te.transform(X_va[cat_cols])

            if fold == 0:
                X_te = X_test_full.copy()
            else:
                X_te_fold = X_test_full.copy()

            X_te_fold = X_test_full.copy()
            X_te_fold[cat_cols] = te.transform(X_test_full[cat_cols])
        else:
            X_te_fold = X_test_full.copy()

        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=100)

        preds_va = clf.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = preds_va

        fold_auc = roc_auc_score(y_va, preds_va)
        print(f"Fold {fold + 1} AUC: {fold_auc:.5f}")

        test_preds += clf.predict_proba(X_te_fold)[:, 1] / args.folds
        models.append(clf)

    overall_auc = roc_auc_score(y_train, oof_preds)
    print(f"\n[+] Boundary Specialist OOF AUC: {overall_auc:.5f}")

    # Save artifacts
    out_dir = PROJECT_ROOT / "models"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Save the masked OOF predictions along with their original training indices
    # so we can perfectly reconstruct the full OOF array later
    oof_df = pd.DataFrame(
        {
            "original_index": X_train_full[boundary_mask].index,
            "boundary_oof_pred": oof_preds,
        }
    )
    oof_df.to_parquet(out_dir / "boundary_specialist_oof.parquet")

    test_df = pd.DataFrame({"id": test["id"], "boundary_test_pred": test_preds})
    test_df.to_parquet(out_dir / "boundary_specialist_test.parquet")

    # Quick metrics record
    with open(out_dir / "boundary_metrics.json", "w") as f:
        json.dump(
            {
                "boundary_threshold": args.boundary_threshold,
                "subset_size": len(X_train),
                "oof_auc": overall_auc,
            },
            f,
            indent=4,
        )

    print(f"[*] Artifacts saved to {out_dir}/")

