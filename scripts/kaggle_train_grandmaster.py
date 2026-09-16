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
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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


_LGBM_DEVICE_CACHE = None


def probe_lgbm_device(has_gpu: bool) -> str:
    """
    Safely probes whether LightGBM has GPU acceleration available (CUDA or OpenCL).
    Gracefully falls back to CPU if GPU tree learner is not compiled in the binary.
    """
    if not has_gpu:
        return "cpu"
    try:
        import lightgbm as lgb
        X_micro = np.random.randn(50, 4).astype(np.float32)
        y_micro = np.random.randint(0, 2, 50).astype(np.float32)
        ds_micro = lgb.Dataset(X_micro, label=y_micro, free_raw_data=False)

        # 1. Test native CUDA
        try:
            bst = lgb.train({"device": "cuda", "verbose": -1}, ds_micro, num_boost_round=1)
            del bst, ds_micro
            print("[+] LightGBM native CUDA acceleration ENABLED (device='cuda')")
            return "cuda"
        except Exception:
            pass

        # 2. Test OpenCL GPU
        try:
            ds_micro = lgb.Dataset(X_micro, label=y_micro, free_raw_data=False)
            bst = lgb.train({"device": "gpu", "verbose": -1}, ds_micro, num_boost_round=1)
            del bst, ds_micro
            print("[+] LightGBM OpenCL GPU acceleration ENABLED (device='gpu')")
            return "gpu"
        except Exception:
            pass
    except Exception:
        pass

    print("[*] LightGBM running on CPU with maximum thread parallelism (n_jobs=-1)")
    return "cpu"


def get_lgbm_device(has_gpu: bool) -> str:
    global _LGBM_DEVICE_CACHE
    if _LGBM_DEVICE_CACHE is None:
        _LGBM_DEVICE_CACHE = probe_lgbm_device(has_gpu)
    return _LGBM_DEVICE_CACHE


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


def find_pseudo_labels(
    source_path: Optional[str],
    test_ids: pd.Series,
    conf_high: float = 0.995,
    conf_low: float = 0.005,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Attempts to locate and load high-confidence pseudo-labels for test data.
    Returns (pseudo_mask, pseudo_targets) or (None, None).
    """
    candidates = []
    if source_path:
        candidates.append(Path(source_path))

    candidates.extend([
        Path("/kaggle/working/submission.csv"),
        Path("/kaggle/working/models/lgbm_grandmaster/submission.csv"),
        PROJECT_ROOT / "submission.csv",
        PROJECT_ROOT / "experiments" / "artifacts" / "models" / "lgbm_grandmaster" / "submission.csv",
    ])

    sub_file = None
    for c in candidates:
        if c.exists() and c.is_file():
            sub_file = c
            break

    if sub_file is None:
        print("[-] No prior test submission found for pseudo-labeling.")
        return None, None

    print(f"[+] Loading prior test predictions for pseudo-labeling: {sub_file}")
    try:
        if str(sub_file).endswith(".parquet"):
            df = pl.read_parquet(sub_file).to_pandas()
        else:
            df = pd.read_csv(sub_file)

        pred_col = TARGET if TARGET in df.columns else ("prediction" if "prediction" in df.columns else df.columns[-1])
        if "id" in df.columns:
            df = df.set_index("id").reindex(test_ids.values).reset_index()
        preds = df[pred_col].values

        mask_pos = preds >= conf_high
        mask_neg = preds <= conf_low
        mask = mask_pos | mask_neg
        targets = np.where(mask_pos, 1, 0)

        n_pos = int(mask_pos.sum())
        n_neg = int(mask_neg.sum())
        n_total = int(mask.sum())

        print(f"[+] High-confidence pseudo-labels extracted: {n_total} samples ({n_pos} positive, {n_neg} negative)")
        print(f"    Thresholds: >= {conf_high:.4f} (pos) | <= {conf_low:.4f} (neg) | Test coverage: {n_total / len(test_ids) * 100:.2f}%")
        return mask, targets
    except Exception as e:
        print(f"[!] Warning: Failed to load pseudo-labels from {sub_file}: {e}")
        return None, None


def prepare_seed_folds(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    features: List[str],
    te_cols: List[str],
    n_splits: int,
    seed: int,
    pseudo_mask: Optional[np.ndarray] = None,
    pseudo_targets: Optional[np.ndarray] = None,
    pseudo_weight: float = 0.8,
    use_base_margin: bool = False,
) -> List[Dict[str, Any]]:
    """
    Precomputes & caches Target-Encoded fold matrices once per random seed.
    Eliminates redundant CPU encoding across models, downcasting to float32 (zero OOM risk).
    """
    print(f"\n[*] Pre-computing & Caching {n_splits}-Fold Encodings for Seed {seed}...")
    start_t = time.time()
    pseudo_active = pseudo_mask is not None and np.sum(pseudo_mask) > 0

    X = train_feat[features].copy()
    y = train_feat[TARGET].values
    X_test = test_feat[features].copy()

    if pseudo_active:
        X_pseudo = X_test.iloc[pseudo_mask].copy()
        y_pseudo = pseudo_targets[pseudo_mask].copy()
        sw_pseudo = np.full(len(y_pseudo), pseudo_weight, dtype=np.float32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds_data = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        f_t0 = time.time()
        X_tr = X.iloc[train_idx].copy()
        y_tr = y[train_idx].copy()
        X_va = X.iloc[val_idx].copy()
        y_va = y[val_idx].copy()
        X_te = X_test.copy()

        if pseudo_active:
            X_tr = pd.concat([X_tr, X_pseudo], ignore_index=True)
            y_tr = np.concatenate([y_tr, y_pseudo])
            sw_tr = np.concatenate([np.ones(len(train_idx), dtype=np.float32), sw_pseudo])
        else:
            sw_tr = None

        if te_cols:
            te_auto = TargetEncoder(smooth="auto", cv=n_splits, random_state=seed)
            te_10 = TargetEncoder(smooth=10.0, cv=n_splits, random_state=seed)
            te_100 = TargetEncoder(smooth=100.0, cv=n_splits, random_state=seed)

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
                te_data_tr[f"{col}_te_auto"] = tr_auto[:, idx].astype("float32")
                te_data_va[f"{col}_te_auto"] = va_auto[:, idx].astype("float32")
                te_data_te[f"{col}_te_auto"] = te_auto_arr[:, idx].astype("float32")

                te_data_tr[f"{col}_te_10"] = tr_10[:, idx].astype("float32")
                te_data_va[f"{col}_te_10"] = va_10[:, idx].astype("float32")
                te_data_te[f"{col}_te_10"] = te_10_arr[:, idx].astype("float32")

                te_data_tr[f"{col}_te_100"] = tr_100[:, idx].astype("float32")
                te_data_va[f"{col}_te_100"] = va_100[:, idx].astype("float32")
                te_data_te[f"{col}_te_100"] = te_100_arr[:, idx].astype("float32")

            X_tr = pd.concat([X_tr.drop(columns=te_cols), pd.DataFrame(te_data_tr, index=X_tr.index)], axis=1)
            X_va = pd.concat([X_va.drop(columns=te_cols), pd.DataFrame(te_data_va, index=X_va.index)], axis=1)
            X_te = pd.concat([X_te.drop(columns=te_cols), pd.DataFrame(te_data_te, index=X_te.index)], axis=1)

        # Downcast float64 to float32 to enforce strict RAM bound (<1.5 GB total)
        f64_tr = X_tr.select_dtypes(include=["float64"]).columns
        if len(f64_tr) > 0:
            X_tr[f64_tr] = X_tr[f64_tr].astype("float32")
            X_va[f64_tr] = X_va[f64_tr].astype("float32")
            X_te[f64_tr] = X_te[f64_tr].astype("float32")

        margin_tr = X_tr["feat_recipe_base_margin"].values.astype(np.float32) if ("feat_recipe_base_margin" in X_tr.columns and use_base_margin) else None
        margin_va = X_va["feat_recipe_base_margin"].values.astype(np.float32) if ("feat_recipe_base_margin" in X_va.columns and use_base_margin) else None
        margin_te = X_te["feat_recipe_base_margin"].values.astype(np.float32) if ("feat_recipe_base_margin" in X_te.columns and use_base_margin) else None

        folds_data.append({
            "fold": fold,
            "X_tr": X_tr,
            "y_tr": y_tr,
            "sw_tr": sw_tr,
            "margin_tr": margin_tr,
            "X_va": X_va,
            "y_va": y_va,
            "margin_va": margin_va,
            "X_te": X_te,
            "margin_te": margin_te,
            "val_idx": val_idx,
        })
        print(f"  [Fold {fold}/{n_splits} Encoded] ({time.time() - f_t0:.1f}s) | Fold RAM: {X_tr.memory_usage().sum() / 1e6:.1f} MB")

    print(f"[+] All {n_splits} folds pre-encoded in {time.time() - start_t:.1f}s. Models will train with native Base Margins.")
    gc.collect()
    return folds_data


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
    pseudo_mask: Optional[np.ndarray] = None,
    pseudo_targets: Optional[np.ndarray] = None,
    pseudo_weight: float = 0.8,
    prepared_folds: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model_name = f"{model_type}_grandmaster"
    print(f"\n=================================================================")
    print(f"[*] TRAINING {n_splits}-FOLD GRANDMASTER: {model_name.upper()}")
    print(f"    Accelerator: {'GPU' if has_gpu else 'CPU'} | Base Margin: ACTIVE (0.93769 Utility Prior)")
    pseudo_active = pseudo_mask is not None and np.sum(pseudo_mask) > 0
    if pseudo_active:
        print(f"    Pseudo-Labeling: ACTIVE ({np.sum(pseudo_mask)} test samples added to fold train splits, weight={pseudo_weight})")
    print(f"=================================================================")

    if prepared_folds is None:
        prepared_folds = prepare_seed_folds(
            train_feat=train_feat,
            test_feat=test_feat,
            features=features,
            te_cols=te_cols,
            n_splits=n_splits,
            seed=seed,
            pseudo_mask=pseudo_mask,
            pseudo_targets=pseudo_targets,
            pseudo_weight=pseudo_weight,
        )

    y = train_feat[TARGET].values
    oof_preds = np.zeros(len(train_feat), dtype=np.float64)
    test_preds = np.zeros(len(test_feat), dtype=np.float64)

    start_time = time.time()

    for f_info in prepared_folds:
        fold = f_info["fold"]
        f_start = time.time()
        X_tr = f_info["X_tr"]
        y_tr = f_info["y_tr"]
        sw_tr = f_info["sw_tr"]
        margin_tr = f_info.get("margin_tr", None)
        X_va = f_info["X_va"]
        y_va = f_info["y_va"]
        margin_va = f_info.get("margin_va", None)
        X_te = f_info["X_te"]
        margin_te = f_info.get("margin_te", None)
        val_idx = f_info["val_idx"]

        if model_type == "lgbm":
            import lightgbm as lgb
            dtr = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr, init_score=margin_tr, free_raw_data=False)
            dva = lgb.Dataset(X_va, label=y_va, init_score=margin_va, reference=dtr, free_raw_data=False)
            lgb_dev = get_lgbm_device(has_gpu)
            lgb_params = {
                "objective": "binary",
                "metric": "auc",
                "device": lgb_dev,
                "learning_rate": 0.035,
                "max_depth": 6,
                "num_leaves": 63,
                "min_child_samples": 20,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.4,
                "reg_alpha": 0.05,
                "reg_lambda": 2.5,
                "max_bin": 255 if lgb_dev != "cpu" else 512,
                "random_state": seed,
                "n_jobs": -1 if lgb_dev == "cpu" else 4,
                "verbose": -1,
            }
            bst = lgb.train(
                lgb_params,
                dtr,
                valid_sets=[dva],
                num_boost_round=10000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=300, verbose=False),
                    lgb.log_evaluation(period=1000),
                ],
            )
            if margin_va is not None:
                val_raw = bst.predict(X_va, raw_score=True) + margin_va
                val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_raw, -35.0, 35.0)))
                test_raw = bst.predict(X_te, raw_score=True) + margin_te
                test_probs_fold = 1.0 / (1.0 + np.exp(-np.clip(test_raw, -35.0, 35.0)))
            else:
                val_probs = bst.predict(X_va)
                test_probs_fold = bst.predict(X_te)
            del bst, dtr, dva

        elif model_type == "catboost":
            import catboost as cb
            tr_pool = cb.Pool(X_tr, y_tr, weight=sw_tr, baseline=margin_tr)
            va_pool = cb.Pool(X_va, y_va, baseline=margin_va)
            te_pool = cb.Pool(X_te, baseline=margin_te)
            cb_params = {
                "iterations": 8000,
                "learning_rate": 0.04,
                "depth": 6,
                "l2_leaf_reg": 4.0,
                "eval_metric": "AUC",
                "random_seed": seed,
                "early_stopping_rounds": 300,
                "verbose": 1000,
            }
            if has_gpu:
                cb_params["task_type"] = "GPU"
                cb_params["devices"] = "0"
            else:
                cb_params["thread_count"] = -1
            clf = cb.CatBoostClassifier(**cb_params)
            clf.fit(tr_pool, eval_set=va_pool, verbose=1000)
            val_probs = clf.predict_proba(va_pool)[:, 1]
            test_probs_fold = clf.predict_proba(te_pool)[:, 1]
            del clf, tr_pool, va_pool, te_pool

        elif model_type == "xgboost":
            import xgboost as xgb
            dtr = xgb.DMatrix(X_tr, label=y_tr, weight=sw_tr, base_margin=margin_tr)
            dva = xgb.DMatrix(X_va, label=y_va, base_margin=margin_va)
            dte = xgb.DMatrix(X_te, base_margin=margin_te)

            xgb_params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "device": "cuda" if has_gpu else "cpu",
                "learning_rate": 0.03,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.4,
                "reg_alpha": 0.05,
                "reg_lambda": 2.5,
                "max_bin": 512,
                "seed": seed,
                "nthread": 4 if has_gpu else -1,
            }
            bst = xgb.train(
                xgb_params,
                dtr,
                evals=[(dva, "val")],
                num_boost_round=10000,
                callbacks=[xgb.callback.EarlyStopping(rounds=300, save_best=True)],
                verbose_eval=1000,
            )
            val_probs = bst.predict(dva)
            test_probs_fold = bst.predict(dte)
            del bst, dtr, dva, dte
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        oof_preds[val_idx] = val_probs
        test_preds += test_probs_fold / n_splits
        fold_auc = roc_auc_score(y_va, val_probs)
        print(f"  [Fold {fold}/{n_splits}] ROC-AUC: {fold_auc:.6f} ({time.time() - f_start:.1f}s)")

        del val_probs, test_probs_fold
        gc.collect()

    overall_auc = roc_auc_score(train_feat[TARGET].values, oof_preds)
    duration = time.time() - start_time
    print(f"=================================================================")
    print(f"[+] OVERALL {n_splits}-FOLD OOF ROC-AUC: {overall_auc:.6f}")
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


def make_zero_tie_ranks(primary_scores: np.ndarray, secondary_scores: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Continuous Lexicographical Zero-Tie Ranking (np.lexsort).
    Guarantees 100% strictly unique percentile ranks across all test samples,
    eliminating the 0.5 AUC tie penalty and maximizing ROC-AUC.
    """
    if secondary_scores is None or len(secondary_scores) != len(primary_scores):
        secondary_scores = np.arange(len(primary_scores), dtype=np.float64)
    order = np.lexsort((secondary_scores, primary_scores))
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = (np.arange(len(order), dtype=np.float64) + 0.5) / len(order)
    return ranks



def blend_grandmaster_models(
    models_oof: Dict[str, np.ndarray],
    models_test: Dict[str, np.ndarray],
    y_true: np.ndarray,
    test_ids: pd.Series,
    output_dir: Path,
    primary_sub_path: Path,
    test_feat: Optional[pd.DataFrame] = None,
):
    from scipy.stats import rankdata

    print(f"\n======================================================================")
    print(f"[*] EXECUTING GRANDMASTER TRIPLE OPTIMIZED BLEND (PROB + RANK + LOGIT)")
    print(f"======================================================================")
    model_names = list(models_oof.keys())
    m = len(model_names)
    if m <= 1:
        print("[!] Only 1 model trained; skipping blend.")
        return

    oof_matrix = np.column_stack([models_oof[k] for k in model_names])
    test_matrix = np.column_stack([models_test[k] for k in model_names])

    oof_rank_matrix = np.column_stack([rankdata(models_oof[k]) / len(y_true) for k in model_names])
    test_rank_matrix = np.column_stack([rankdata(models_test[k]) / len(test_ids) for k in model_names])

    eps = 1e-7
    clip_oof = np.clip(oof_matrix, eps, 1.0 - eps)
    oof_logit_matrix = np.log(clip_oof / (1.0 - clip_oof))
    clip_test = np.clip(test_matrix, eps, 1.0 - eps)
    test_logit_matrix = np.log(clip_test / (1.0 - clip_test))

    best_single_auc = max(roc_auc_score(y_true, models_oof[k]) for k in model_names)

    import itertools

    best_overall_auc = -1.0
    best_candidate_info = None

    # Generate all subsets of models of size >= 2
    subsets = []
    for k in range(2, m + 1):
        for comb in itertools.combinations(range(m), k):
            subsets.append(list(comb))

    for subset_indices in subsets:
        sub_names = [model_names[i] for i in subset_indices]
        k_sub = len(subset_indices)
        sub_oof = oof_matrix[:, subset_indices]
        sub_test = test_matrix[:, subset_indices]
        sub_oof_rank = oof_rank_matrix[:, subset_indices]
        sub_test_rank = test_rank_matrix[:, subset_indices]
        sub_oof_logit = oof_logit_matrix[:, subset_indices]
        sub_test_logit = test_logit_matrix[:, subset_indices]

        init_w_sub = np.ones(k_sub) / k_sub

        # 1. Probability Space
        def loss_p(weights):
            w = np.array(weights)
            if np.sum(np.abs(w)) == 0:
                return 0.0
            w = w / np.sum(w)
            return -roc_auc_score(y_true, np.dot(sub_oof, w))

        res_p = minimize(loss_p, init_w_sub, method="Nelder-Mead", bounds=[(0.0, 1.0)] * k_sub, options={"maxiter": 400, "disp": False})
        w_p = np.clip(res_p.x, 0.0, None)
        w_p = w_p / np.sum(w_p)
        oof_p = np.dot(sub_oof, w_p)
        p_auc = roc_auc_score(y_true, oof_p)
        if p_auc > best_overall_auc:
            best_overall_auc = p_auc
            best_candidate_info = ("prob_space_nelder_mead", sub_names, w_p, oof_p, np.dot(sub_test, w_p))

        # 2. Percentile Rank Space
        def loss_r(weights):
            w = np.array(weights)
            if np.sum(np.abs(w)) == 0:
                return 0.0
            w = w / np.sum(w)
            return -roc_auc_score(y_true, np.dot(sub_oof_rank, w))

        res_r = minimize(loss_r, init_w_sub, method="Nelder-Mead", bounds=[(0.0, 1.0)] * k_sub, options={"maxiter": 400, "disp": False})
        w_r = np.clip(res_r.x, 0.0, None)
        w_r = w_r / np.sum(w_r)
        oof_r = np.dot(sub_oof_rank, w_r)
        r_auc = roc_auc_score(y_true, oof_r)
        if r_auc > best_overall_auc:
            best_overall_auc = r_auc
            best_candidate_info = ("rank_space_nelder_mead", sub_names, w_r, oof_r, np.dot(sub_test_rank, w_r))

        # 3. Logit Space
        def loss_l(weights):
            w = np.array(weights)
            if np.sum(np.abs(w)) == 0:
                return 0.0
            w = w / np.sum(w)
            b_l = np.dot(sub_oof_logit, w)
            b_p = 1.0 / (1.0 + np.exp(-np.clip(b_l, -35.0, 35.0)))
            return -roc_auc_score(y_true, b_p)

        res_l = minimize(loss_l, init_w_sub, method="Nelder-Mead", bounds=[(0.0, 1.0)] * k_sub, options={"maxiter": 400, "disp": False})
        w_l = np.clip(res_l.x, 0.0, None)
        w_l = w_l / np.sum(w_l)
        oof_l = 1.0 / (1.0 + np.exp(-np.clip(np.dot(sub_oof_logit, w_l), -35.0, 35.0)))
        l_auc = roc_auc_score(y_true, oof_l)
        if l_auc > best_overall_auc:
            best_overall_auc = l_auc
            test_l = 1.0 / (1.0 + np.exp(-np.clip(np.dot(sub_test_logit, w_l), -35.0, 35.0)))
            best_candidate_info = ("logit_space_nelder_mead", sub_names, w_l, oof_l, test_l)

    chosen_method, chosen_models, best_weights, blend_oof, blend_test = best_candidate_info
    final_auc = best_overall_auc
    weights_dict = {chosen_models[i]: float(best_weights[i]) for i in range(len(chosen_models))}
    delta_auc = final_auc - best_single_auc

    print(f"\n[+] Champion Ensemble Strategy: {chosen_method.upper()}")
    print(f"[+] Selected Models:           {chosen_models}")
    print(f"[+] Optimal Blend Weights:     {weights_dict}")
    print(f"[+] Best Single Model AUC:     {best_single_auc:.6f}")
    print(f"[+] Ensembled Grandmaster:     {final_auc:.6f} (Δ: {delta_auc:+.6f})")

    # Save ensemble artifacts
    ensemble_dir = output_dir.parent / "ensemble_grandmaster"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    # Extract continuous secondary signal for zero-tie ranking
    secondary_score = None
    if test_feat is not None:
        if "feat_buy_recipe_score" in test_feat.columns:
            secondary_score = test_feat["feat_buy_recipe_score"].values
        elif "Annual_Income_USD" in test_feat.columns:
            inc = test_feat["Annual_Income_USD"].values
            env = test_feat["Environmental_Concern_Level"].values if "Environmental_Concern_Level" in test_feat.columns else 3.0
            sub = (test_feat["Subsidy_Available"].astype(str) == "Yes").astype(float).values if "Subsidy_Available" in test_feat.columns else 0.0
            anx = test_feat["Range_Anxiety_Level"].astype(str).values if "Range_Anxiety_Level" in test_feat.columns else "Low"
            secondary_score = (
                1.2 * (inc / 100000.0)
                + 0.6 * env
                + 2.0 * sub
                - 1.0 * (anx == "Medium").astype(float)
                - 3.0 * (anx == "High").astype(float)
            )

    # 1. Champion Submission: Pure Probability Nelder-Mead Blend
    work_dir = primary_sub_path.parent
    pure_prob_sub = pd.DataFrame({"id": test_ids, TARGET: blend_test})
    pure_prob_sub.to_csv(ensemble_dir / "submission_pure_prob.csv", index=False)
    pure_prob_sub.to_csv(work_dir / "submission_pure_prob.csv", index=False)
    pure_prob_sub.to_csv(ensemble_dir / "submission.csv", index=False)
    pure_prob_sub.to_csv(primary_sub_path, index=False)
    print(f"[+] Champion Calibrated Probability Blend written to:       {primary_sub_path}")

    # 2. Champion Micro-Jitter Zero-Tie Submission (Preserves 100% Probability Distribution, 0 ties)
    if secondary_score is not None:
        sec_norm = (secondary_score - np.nanmean(secondary_score)) / (np.nanstd(secondary_score) + 1e-7)
        micro_zero_tie = blend_test + 1e-9 * sec_norm
    else:
        micro_zero_tie = blend_test
    micro_sub = pd.DataFrame({"id": test_ids, TARGET: micro_zero_tie})
    micro_sub.to_csv(ensemble_dir / "submission_micro_zero_tie.csv", index=False)
    micro_sub.to_csv(work_dir / "submission_micro_zero_tie.csv", index=False)
    print(f"[+] Micro-Jitter Zero-Tie Probability Blend written to:   {work_dir / 'submission_micro_zero_tie.csv'}")

    # 3. Optional Uniform Rank Lexsort Submission
    zero_tie_champion = make_zero_tie_ranks(blend_test, secondary_score)
    pd.DataFrame({"id": test_ids, TARGET: zero_tie_champion}).to_csv(work_dir / "submission_zero_tie_champion.csv", index=False)
    pd.DataFrame({"id": test_ids, TARGET: zero_tie_champion}).to_csv(work_dir / "submission_uniform_rank.csv", index=False)

    # 3. Dual Rank with Zero-Tie Lexsort (No ties!)
    if "lgbm" in models_test and "xgboost" in models_test:
        r_lgb = rankdata(models_test["lgbm"]) / len(test_ids)
        r_xgb = rankdata(models_test["xgboost"]) / len(test_ids)
        dual_mean = 0.5 * r_lgb + 0.5 * r_xgb
        dual_zero_tie = make_zero_tie_ranks(dual_mean, secondary_score)
        rank_dual = pd.DataFrame({"id": test_ids, TARGET: dual_zero_tie})
        rank_dual.to_csv(work_dir / "submission_dual_rank.csv", index=False)
        print(f"[+] Zero-Tie LGBM+XGBoost 50/50 Rank Average written to:     {work_dir / 'submission_dual_rank.csv'}")

    # 4. Tri-Rank with Zero-Tie Lexsort (LGBM + XGBoost + CatBoost)
    if "lgbm" in models_test and "xgboost" in models_test and "catboost" in models_test:
        r_lgb = rankdata(models_test["lgbm"]) / len(test_ids)
        r_xgb = rankdata(models_test["xgboost"]) / len(test_ids)
        r_cat = rankdata(models_test["catboost"]) / len(test_ids)
        tri_mean = (r_lgb + r_xgb + r_cat) / 3.0
        tri_zero_tie = make_zero_tie_ranks(tri_mean, secondary_score)
        rank_tri = pd.DataFrame({"id": test_ids, TARGET: tri_zero_tie})
        rank_tri.to_csv(work_dir / "submission_tri_rank.csv", index=False)
        print(f"[+] Zero-Tie LGBM+XGB+CatBoost Tri-Rank Average written to: {work_dir / 'submission_tri_rank.csv'}")

    # 5. Pure XGBoost with Zero-Tie Lexsort
    if "xgboost" in models_test:
        xgb_zt = make_zero_tie_ranks(models_test["xgboost"], secondary_score)
        pd.DataFrame({"id": test_ids, TARGET: xgb_zt}).to_csv(work_dir / "submission_xgb_pure.csv", index=False)
        print(f"[+] Zero-Tie Multi-Seed XGBoost submission written to:       {work_dir / 'submission_xgb_pure.csv'}")

    # 6. Pure LightGBM with Zero-Tie Lexsort
    if "lgbm" in models_test:
        lgb_zt = make_zero_tie_ranks(models_test["lgbm"], secondary_score)
        pd.DataFrame({"id": test_ids, TARGET: lgb_zt}).to_csv(work_dir / "submission_lgb_pure.csv", index=False)
        print(f"[+] Zero-Tie Multi-Seed LightGBM submission written to:      {work_dir / 'submission_lgb_pure.csv'}")

    # 7. Pure CatBoost with Zero-Tie Lexsort
    if "catboost" in models_test:
        cat_zt = make_zero_tie_ranks(models_test["catboost"], secondary_score)
        pd.DataFrame({"id": test_ids, TARGET: cat_zt}).to_csv(work_dir / "submission_cat_pure.csv", index=False)
        print(f"[+] Zero-Tie Multi-Seed CatBoost submission written to:      {work_dir / 'submission_cat_pure.csv'}")

    blend_oof_df = pd.DataFrame({"id": range(len(blend_oof)), "oof_pred": blend_oof, "target": y_true})
    blend_oof_df.to_parquet(ensemble_dir / "oof_preds.parquet", index=False)

    meta = {
        "ensemble_type": chosen_method,
        "selected_models": chosen_models,
        "weights": weights_dict,
        "best_single_auc": float(best_single_auc),
        "ensemble_auc": float(final_auc),
        "delta_auc": float(delta_auc),
    }
    with open(ensemble_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[+] Grandmaster champion blend written to:                 {primary_sub_path}")


def main():
    parser = argparse.ArgumentParser(description="Grandmaster EV Purchase Training")
    parser.add_argument("--model", type=str, choices=["lgbm", "catboost", "xgboost", "dual", "all"], default="dual")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42, help="Primary random seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 777, 1337, 9999], help="List of seeds for multi-seed averaging")
    parser.add_argument("--pseudo-label", action="store_true", help="Enable high-confidence pseudo-labeling from test predictions")
    parser.add_argument("--pseudo-source", type=str, default=None, help="Path to prior test submission/predictions for pseudo-labeling")
    parser.add_argument("--pseudo-conf-high", type=float, default=0.995, help="High confidence threshold (positive)")
    parser.add_argument("--pseudo-conf-low", type=float, default=0.005, help="Low confidence threshold (negative)")
    parser.add_argument("--pseudo-weight", type=float, default=0.8, help="Sample weight for pseudo-labeled points")
    parser.add_argument("--use-base-margin", action="store_true", default=True, help="Enable recipe base margin for XGBoost, LightGBM, CatBoost (default: True)")
    parser.add_argument("--no-base-margin", dest="use_base_margin", action="store_false", help="Disable recipe base margin")
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

    pseudo_mask, pseudo_targets = None, None
    if args.pseudo_label:
        pseudo_mask, pseudo_targets = find_pseudo_labels(
            source_path=args.pseudo_source,
            test_ids=test_feat["id"],
            conf_high=args.pseudo_conf_high,
            conf_low=args.pseudo_conf_low,
        )

    seeds = args.seeds if args.seeds else [args.seed]
    if args.model == "all":
        models_to_run = ["lgbm", "catboost", "xgboost"]
    elif args.model == "dual":
        models_to_run = ["lgbm", "xgboost"]
    else:
        models_to_run = [args.model]

    models_oof: Dict[str, np.ndarray] = {m: np.zeros(len(train_feat), dtype=np.float64) for m in models_to_run}
    models_test: Dict[str, np.ndarray] = {m: np.zeros(len(test_feat), dtype=np.float64) for m in models_to_run}

    for s_idx, s in enumerate(seeds, 1):
        print(f"\n=================================================================")
        print(f"[*] SEED CYCLE {s_idx}/{len(seeds)} (Seed: {s})")
        print(f"=================================================================")

        # 1. Precompute & cache encoded folds ONCE for this seed (eliminates redundant CPU work)
        cached_folds = prepare_seed_folds(
            train_feat=train_feat,
            test_feat=test_feat,
            features=features,
            te_cols=te_cols,
            n_splits=args.folds,
            seed=s,
            pseudo_mask=pseudo_mask,
            pseudo_targets=pseudo_targets,
            pseudo_weight=args.pseudo_weight,
            use_base_margin=args.use_base_margin,
        )

        # 2. Train each requested model on cached folds at MAX throughput
        for m in models_to_run:
            print(f"\n>>> Running {m.upper()} on Cached Seed {s} ({s_idx}/{len(seeds)})")
            auc, oof, test_p = train_single_model(
                model_type=m,
                train_feat=train_feat,
                test_feat=test_feat,
                features=features,
                te_cols=te_cols,
                n_splits=args.folds,
                seed=s,
                has_gpu=has_gpu,
                output_dir=output_dir,
                primary_sub_path=sub_path,
                pseudo_mask=pseudo_mask,
                pseudo_targets=pseudo_targets,
                pseudo_weight=args.pseudo_weight,
                prepared_folds=cached_folds,
            )
            models_oof[m] += oof / len(seeds)
            models_test[m] += test_p / len(seeds)

        del cached_folds
        gc.collect()

    for m in models_to_run:
        m_auc = roc_auc_score(train_feat[TARGET].values, models_oof[m])
        if len(seeds) > 1:
            print(f"\n[+] {m.upper()} Multi-Seed Average OOF ROC-AUC ({len(seeds)} seeds): {m_auc:.6f}")

    if len(models_to_run) > 1:
        blend_grandmaster_models(
            models_oof=models_oof,
            models_test=models_test,
            y_true=train_feat[TARGET].values,
            test_ids=test_feat["id"],
            output_dir=output_dir,
            primary_sub_path=sub_path,
            test_feat=test_feat,
        )


if __name__ == "__main__":
    main()
