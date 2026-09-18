"""
Phase 5: Knowledge Distillation for Production Deployment
Trains a single, ultra-fast LightGBM Student model to mimic the complex 160-model Mega-Blend.
Saves a single lightweight artifact for instant inference.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score

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
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save the distilled model")
    args = parser.parse_args()

    data_dir = locate_data_dir()
    print(f"[*] Loading data from {data_dir}")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    orig_path = data_dir / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
    orig = pd.read_csv(orig_path) if orig_path.exists() else None

    # Load Ground Truth
    y_true = train[TARGET].map({"Yes": 1, "No": 0, "1": 1, "0": 0}).astype(int).values

    print("[*] Engineering Features via Grandmaster Module...")
    X_train_full, X_test_full, features, cat_cols = build_grandmaster_features(train, test, orig)
    
    if TARGET in X_train_full.columns:
        X_train_full = X_train_full.drop(columns=[TARGET])
        
    print("[*] Assembling the 160-Model Mega-Blend Teacher Probabilities...")
    models_dir = PROJECT_ROOT / "models"
    
    # Load Teacher OOFs
    try:
        oof_bb = pd.read_parquet(models_dir / "oof_preds_top1.parquet")
        preds_bb = oof_bb["pred_blend"].values if "pred_blend" in oof_bb.columns else oof_bb.iloc[:, 1].values
        
        oof_pseudo = pd.read_parquet(models_dir / "pseudo_label_oof.parquet")
        preds_pseudo = oof_pseudo["pseudo_oof_pred"].values
        
        oof_nn = pd.read_parquet(models_dir / "nn_tabular/oof_preds.parquet")
        nn_col = [c for c in oof_nn.columns if c != "id"][0]
        preds_nn = oof_nn[nn_col].values
        
        oof_bound = pd.read_parquet(models_dir / "boundary_specialist_oof.parquet")
        bound_idx = oof_bound["original_index"].values
        preds_bound = oof_bound["boundary_oof_pred"].values
    except Exception as e:
        print(f"[-] Could not load one of the OOF parquets from {models_dir}. Make sure all models were run!")
        print(e)
        sys.exit(1)

    # Apply Optimal Nelder-Mead Weights
    w_bb, w_pseudo, w_nn, w_bound = 1.1173, 0.2013, 0.0420, 0.0404
    eps = 1e-7
    
    logit_bb = logit(np.clip(preds_bb, eps, 1-eps))
    logit_pseudo = logit(np.clip(preds_pseudo, eps, 1-eps))
    logit_nn = logit(np.clip(preds_nn, eps, 1-eps))
    logit_bound = logit(np.clip(preds_bound, eps, 1-eps))

    teacher_logits = w_bb * logit_bb + w_pseudo * logit_pseudo + w_nn * logit_nn
    
    # Apply boundary override
    merged_boundary_logit = (w_bb * logit_bb[bound_idx] + 
                             w_pseudo * logit_pseudo[bound_idx] + 
                             w_nn * logit_nn[bound_idx] + 
                             w_bound * logit_bound)
    teacher_logits[bound_idx] = merged_boundary_logit
    
    teacher_probs = expit(teacher_logits)
    
    teacher_auc = roc_auc_score(y_true, teacher_probs)
    print(f"[+] Reconstructed Teacher AUC: {teacher_auc:.6f}")
    
    print("\n[*] Training Distilled Student (Single LightGBM)...")
    # Convert categoricals for LightGBM
    for c in cat_cols:
        if c in X_train_full.columns:
            X_train_full[c] = X_train_full[c].astype('category')
            
    # Train directly on the continuous teacher probabilities using cross-entropy (regression objective works best)
    dtrain = lgb.Dataset(X_train_full, label=teacher_probs, free_raw_data=False)
    
    params = {
        "objective": "regression", # We want to exactly match the teacher's soft probabilities
        "metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 7,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42
    }
    
    # We train on the ENTIRE dataset (no validation splits) because we are distilling
    student_model = lgb.train(
        params,
        dtrain,
        num_boost_round=600
    )
    
    # Evaluate fidelity
    student_preds = student_model.predict(X_train_full)
    student_auc = roc_auc_score(y_true, student_preds)
    
    print(f"[+] Distilled Student AUC: {student_auc:.6f} (Fidelity to Teacher: {student_auc/teacher_auc*100:.2f}%)")
    
    # Save the deployment artifact
    out_dir = PROJECT_ROOT / args.save_dir
    out_dir.mkdir(exist_ok=True, parents=True)
    
    model_path = out_dir / "distilled_student.txt"
    student_model.save_model(str(model_path))
    
    print(f"[+] Deployment artifact successfully saved to: {model_path}")
    print("[*] Ready for production inference!")

if __name__ == "__main__":
    main()
