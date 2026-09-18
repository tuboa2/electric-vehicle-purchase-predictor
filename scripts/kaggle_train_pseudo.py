"""
Phase 2: Extreme Pseudo-Labeling Training Script
Injects high-confidence test predictions from submission (3).csv into the training folds
to tighten the tree leaves using the unseen test distribution.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.grandmaster_features import build_grandmaster_features, TARGET

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
    parser.add_argument("--pos_thresh", type=float, default=0.990, help="Threshold for positive pseudo-labels")
    parser.add_argument("--neg_thresh", type=float, default=0.008, help="Threshold for negative pseudo-labels")
    parser.add_argument("--pseudo_weight", type=float, default=0.60, help="Sample weight for pseudo-labels")
    args = parser.parse_args()

    data_dir = locate_data_dir()
    print(f"[*] Loading data from {data_dir}")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    orig_path = data_dir / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
    orig = pd.read_csv(orig_path) if orig_path.exists() else None

    # Load submission (3).csv or fallback to top1 champion
    sub_path = PROJECT_ROOT / "models" / "submission (3).csv"
    if not sub_path.exists():
        sub_path = PROJECT_ROOT / "models" / "submission_top1_champion.csv"
        
    if not sub_path.exists():
        print(f"[-] Could not find submission file for pseudo-labels at {sub_path}. Aborting.")
        sys.exit(1)
        
    print(f"[*] Loading test predictions for pseudo-labeling from {sub_path.name}")
    sub_df = pd.read_csv(sub_path)
    
    # Identify high confidence samples
    high_conf_mask = (sub_df[TARGET] > args.pos_thresh) | (sub_df[TARGET] < args.neg_thresh)
    test_pseudo = test[high_conf_mask].copy()
    test_pseudo_y = (sub_df.loc[high_conf_mask, TARGET] > 0.5).astype(int).values
    
    print(f"[+] Found {len(test_pseudo)} high-confidence test samples ({(len(test_pseudo)/len(test))*100:.1f}%)")
    
    print("[*] Engineering Features via Grandmaster Module...")
    X_train_full, X_test_full, features, cat_cols = build_grandmaster_features(train, test, orig)
    
    # We must drop TARGET from features!
    if TARGET in X_train_full.columns:
        X_train_full = X_train_full.drop(columns=[TARGET])
    if TARGET in X_test_full.columns:
        X_test_full = X_test_full.drop(columns=[TARGET])

    y_train_full = train[TARGET].map({"Yes": 1, "No": 0, "1": 1, "0": 0}).astype(int).values
    
    # Extract the pre-engineered features for the pseudo-label subset
    X_pseudo = X_test_full.loc[high_conf_mask].copy()
    y_pseudo = test_pseudo_y
    
    oof_preds = np.zeros(len(X_train_full))
    test_preds = np.zeros(len(X_test_full))
    
    # Fast GPU detection
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except:
        has_gpu = False
        
    device = "cuda" if has_gpu else "cpu"
    tree_method = "hist"
    
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
        "n_estimators": 1000,
        "early_stopping_rounds": 100,
        "random_state": 42,
        "n_jobs": -1
    }
    
    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    
    print(f"[*] Starting Cross-Validation with Pseudo-Label Injection on {device}...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train_full)):
        print(f"\n--- Fold {fold+1}/{args.folds} ---")
        
        # Base train/val split
        X_tr, y_tr = X_train_full.iloc[train_idx].copy(), y_train_full[train_idx]
        X_va, y_va = X_train_full.iloc[val_idx].copy(), y_train_full[val_idx]
        
        # Inject Pseudo-Labels INTO THE TRAINING SPLIT ONLY
        X_tr = pd.concat([X_tr, X_pseudo], ignore_index=True)
        y_tr = np.concatenate([y_tr, y_pseudo])
        
        # Create sample weights: 1.0 for real data, 0.60 for pseudo-labels
        w_tr = np.ones(len(y_tr))
        w_tr[len(train_idx):] = args.pseudo_weight
        
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
            
        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        dtest = xgb.DMatrix(X_te_fold)
        
        evals = [(dval, "validation")]
        
        clf = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=100,
            verbose_eval=100
        )
        
        preds_va = clf.predict(dval)
        oof_preds[val_idx] = preds_va
        
        fold_auc = roc_auc_score(y_va, preds_va)
        print(f"Fold {fold+1} AUC: {fold_auc:.5f}")
        
        test_preds += clf.predict(dtest) / args.folds
        
    overall_auc = roc_auc_score(y_train_full, oof_preds)
    print(f"\n[+] Pseudo-Label OOF AUC: {overall_auc:.6f}")
    
    out_dir = PROJECT_ROOT / "models"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    oof_df = pd.DataFrame({"id": train["id"], "pseudo_oof_pred": oof_preds})
    oof_df.to_parquet(out_dir / "pseudo_label_oof.parquet")
    
    test_df = pd.DataFrame({"id": test["id"], "pseudo_test_pred": test_preds})
    test_df.to_parquet(out_dir / "pseudo_label_test.parquet")
    
    with open(out_dir / "pseudo_metrics.json", "w") as f:
        json.dump({
            "pos_thresh": args.pos_thresh,
            "neg_thresh": args.neg_thresh,
            "pseudo_weight": args.pseudo_weight,
            "injected_samples": len(test_pseudo),
            "oof_auc": overall_auc
        }, f, indent=4)
        
    print(f"[*] Artifacts saved to {out_dir}/")

if __name__ == "__main__":
    main()
