"""
Kaggle Autonomous Multi-Agent System (KAMAS) - Top-1 Execution Engine
Targeting Global Rank 1 (0.94672+) on Kaggle Playground Series s6e9

Synthesizes the unanimous consensus of 11 frontier LLM deep research reports:
1. Ground-Truth Reverse-Engineered Latent Coordinates (0.93769 baseline signal)
2. Simpson's Paradox Inversion via Within-Cohort Z-Scores & Explicit Native City_Type Multiplicative Interactions
3. Procedural Hard Saturation Flags & Discrete Generator Modulo Artifacts
4. Dual-Stream Inductive Bias Stacking:
   - Stream A: Free-Tree GBDTs (LGBM, XGBoost, CatBoost)
   - Stream B: Base-Margin Residual GBDTs (LGBM, XGBoost, CatBoost with base_margin = 2.17464 * (recipe - 5.61235))
5. Stream C: Boundary Specialist Model (trained with Gaussian boundary sample weighting w_i = 1 + 7*exp(-(z/0.30)^2))
6. Hierarchical Two-Tier Gating:
   - Tier 1: Logit-space fusion of Stream A + Stream B
   - Tier 2: Gaussian Gated Refinement with Stream C: p_final = (1 - g(x))*p_backbone + g(x)*p_boundary
7. Calibrated Micro-Jitter Zero-Tie Tie-Breaking:
   - Preserves 100% of the true probability distribution (mean ~ 0.1748)
   - Continuous 1e-9 secondary score lexicographical tie breaking (0 ties)
"""

import argparse
import concurrent.futures
import gc
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit, logit, ndtr
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.grandmaster_features import build_grandmaster_features, TARGET


def get_gpu_count() -> int:
    """Returns number of available CUDA GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    try:
        res = os.popen("nvidia-smi -L").read()
        lines = [l for l in res.strip().split("\n") if "GPU " in l]
        if lines:
            return len(lines)
    except Exception:
        pass
    return 0


def detect_gpu() -> bool:
    """Detects if CUDA GPU is available and logs device details."""
    count = get_gpu_count()
    if count > 0:
        try:
            import torch
            names = [torch.cuda.get_device_name(i) for i in range(count)]
            print(f"[+] {count} GPU(s) detected via PyTorch: {', '.join(names)}")
        except Exception:
            print(f"[+] {count} GPU(s) detected via nvidia-smi")
        return True
    print("[-] No GPU detected. Running on high-performance multi-threaded CPU.")
    return False


_LGBM_DEVICE_CACHE: Optional[str] = None
_XGB_DEVICE_CACHE: Optional[str] = None
_CB_DEVICE_CACHE: Optional[str] = None


def probe_lgbm_device(has_gpu: bool) -> str:
    """
    Safely probes whether LightGBM has CUDA GPU tree learner enabled in its binary.
    Gracefully falls back to high-performance multi-threaded CPU if not compiled in.
    """
    global _LGBM_DEVICE_CACHE
    if _LGBM_DEVICE_CACHE is not None:
        return _LGBM_DEVICE_CACHE

    if not has_gpu:
        _LGBM_DEVICE_CACHE = "cpu"
        return "cpu"

    try:
        import lightgbm as lgb
        X_micro = np.random.randn(20, 2).astype(np.float32)
        y_micro = np.array([0, 1] * 10, dtype=np.float32)
        ds_micro = lgb.Dataset(X_micro, label=y_micro, free_raw_data=False)
        try:
            bst = lgb.train({"device": "cuda", "verbose": -1}, ds_micro, num_boost_round=1)
            del bst, ds_micro
            print("[+] LightGBM native CUDA acceleration ENABLED (device='cuda')")
            _LGBM_DEVICE_CACHE = "cuda"
            return "cuda"
        except Exception:
            pass
    except Exception:
        pass

    print("[*] LightGBM CUDA Tree Learner not enabled in binary. Safely utilizing high-performance CPU (n_jobs=-1).")
    _LGBM_DEVICE_CACHE = "cpu"
    return "cpu"


def probe_xgb_device(has_gpu: bool) -> str:
    """
    Safely probes whether XGBoost has CUDA device acceleration available.
    Supports XGBoost 2.0+ (device='cuda') and legacy (tree_method='gpu_hist').
    """
    global _XGB_DEVICE_CACHE
    if _XGB_DEVICE_CACHE is not None:
        return _XGB_DEVICE_CACHE

    if not has_gpu:
        _XGB_DEVICE_CACHE = "cpu"
        return "cpu"

    try:
        import xgboost as xgb
        dmat = xgb.DMatrix(np.zeros((10, 2), dtype=np.float32), label=np.array([0, 1] * 5, dtype=np.float32))
        try:
            bst = xgb.train({"tree_method": "hist", "device": "cuda"}, dmat, num_boost_round=1)
            del bst
            print("[+] XGBoost CUDA acceleration ENABLED (device='cuda')")
            _XGB_DEVICE_CACHE = "cuda"
            return "cuda"
        except Exception:
            pass
        try:
            bst = xgb.train({"tree_method": "gpu_hist"}, dmat, num_boost_round=1)
            del bst
            print("[+] XGBoost GPU acceleration ENABLED (tree_method='gpu_hist')")
            _XGB_DEVICE_CACHE = "gpu_hist"
            return "gpu_hist"
        except Exception:
            pass
    except Exception:
        pass

    print("[*] XGBoost running on CPU (n_jobs=-1).")
    _XGB_DEVICE_CACHE = "cpu"
    return "cpu"


def probe_cb_device(has_gpu: bool) -> str:
    """
    Safely probes whether CatBoost has GPU acceleration available.
    """
    global _CB_DEVICE_CACHE
    if _CB_DEVICE_CACHE is not None:
        return _CB_DEVICE_CACHE

    if not has_gpu:
        _CB_DEVICE_CACHE = "CPU"
        return "CPU"

    try:
        from catboost import CatBoostClassifier, Pool
        p = Pool(np.zeros((10, 2), dtype=np.float32), np.array([0, 1] * 5, dtype=np.float32))
        cb = CatBoostClassifier(iterations=1, task_type="GPU", verbose=False)
        cb.fit(p)
        del cb, p
        print("[+] CatBoost GPU acceleration ENABLED (task_type='GPU')")
        _CB_DEVICE_CACHE = "GPU"
        return "GPU"
    except Exception as e:
        print(f"[*] CatBoost GPU unavailable ({e}). Safely running on CPU.")

    _CB_DEVICE_CACHE = "CPU"
    return "CPU"


def locate_data_dir(custom_path: Optional[str] = None) -> Path:
    """Locates the raw or processed dataset directory."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    env_dir = os.environ.get("KAGGLE_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.extend([
        Path("/kaggle/input/competitions/playground-series-s6e9"),
        Path("/kaggle/input/playground-series-s6e9"),
        PROJECT_ROOT / "data" / "processed" / "playground-series-s6e9",
        PROJECT_ROOT / "data" / "raw" / "playground-series-s6e9",
    ])
    for p in candidates:
        if p.exists() and ((p / "train.parquet").exists() or (p / "train.csv").exists()):
            print(f"[+] Dataset located at: {p}")
            return p
    raise FileNotFoundError(f"Could not locate playground-series-s6e9 dataset in candidates: {candidates}")


def load_dataset(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Loads train, test, and optional original seed datasets."""
    print("[*] Ingesting dataset files...")
    if (data_dir / "train.parquet").exists():
        train_df = pl.read_parquet(data_dir / "train.parquet").to_pandas()
    else:
        train_df = pd.read_csv(data_dir / "train.csv")

    if (data_dir / "test.parquet").exists():
        test_df = pl.read_parquet(data_dir / "test.parquet").to_pandas()
    else:
        test_df = pd.read_csv(data_dir / "test.csv")

    orig_df = None
    orig_candidates = [
        Path("/kaggle/working/electric-vehicle/data/original/EV_Adoption_and_Range_Anxiety_Dataset.csv"),
        Path("/kaggle/input/ev-adoption-and-range-anxiety-dataset/EV_Adoption_and_Range_Anxiety_Dataset.csv"),
        PROJECT_ROOT / "data" / "original" / "EV_Adoption_and_Range_Anxiety_Dataset.csv",
    ]
    for cand in orig_candidates:
        if cand.exists():
            orig_df = pd.read_csv(cand)
            print(f"[+] Ground-Truth Original Dataset loaded from: {cand} ({len(orig_df)} samples)")
            break

    print(f"[+] Ingested train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df, orig_df


def prepare_seed_folds(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    features: List[str],
    te_cols: List[str],
    n_splits: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Precomputes & caches Target-Encoded fold matrices once per random seed.
    Eliminates redundant CPU encoding across models, downcasting to float32 (<1.5 GB RAM total).
    """
    print(f"\n[*] Pre-computing & Caching {n_splits}-Fold Encodings for Seed {seed}...")
    start_t = time.time()

    X = train_feat[features].copy()
    y = train_feat[TARGET].values
    X_test = test_feat[features].copy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds_data = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        f_t0 = time.time()
        X_tr = X.iloc[train_idx].copy()
        y_tr = y[train_idx].copy()
        X_va = X.iloc[val_idx].copy()
        y_va = y[val_idx].copy()
        X_te = X_test.copy()

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

        # Downcast float64 to float32
        f64_tr = X_tr.select_dtypes(include=["float64"]).columns
        if len(f64_tr) > 0:
            X_tr[f64_tr] = X_tr[f64_tr].astype("float32")
            X_va[f64_tr] = X_va[f64_tr].astype("float32")
            X_te[f64_tr] = X_te[f64_tr].astype("float32")

        # Base Margins & Boundary Weights
        margin_tr = X_tr["feat_recipe_base_margin"].values.astype(np.float32) if "feat_recipe_base_margin" in X_tr.columns else None
        margin_va = X_va["feat_recipe_base_margin"].values.astype(np.float32) if "feat_recipe_base_margin" in X_va.columns else None
        margin_te = X_te["feat_recipe_base_margin"].values.astype(np.float32) if "feat_recipe_base_margin" in X_te.columns else None

        # Gaussian boundary sample weights: w_i = 1.0 + 7.0 * exp(-(z / 0.30)^2)
        if "feat_recipe_dist_to_boundary" in X_tr.columns:
            z_tr = X_tr["feat_recipe_dist_to_boundary"].values
            sw_boundary_tr = (1.0 + 7.0 * np.exp(-((z_tr / 0.30) ** 2))).astype(np.float32)
        else:
            sw_boundary_tr = np.ones(len(y_tr), dtype=np.float32)

        folds_data.append({
            "fold": fold,
            "X_tr": X_tr,
            "y_tr": y_tr,
            "margin_tr": margin_tr,
            "sw_boundary_tr": sw_boundary_tr,
            "X_va": X_va,
            "y_va": y_va,
            "margin_va": margin_va,
            "X_te": X_te,
            "margin_te": margin_te,
            "val_idx": val_idx,
        })
        print(f"  [Fold {fold}/{n_splits} Encoded] ({time.time() - f_t0:.1f}s) | Fold RAM: {X_tr.memory_usage().sum() / 1e6:.1f} MB")

    print(f"[+] All {n_splits} folds pre-encoded in {time.time() - start_t:.1f}s.")
    gc.collect()
    return folds_data


def train_lgbm(
    folds_data: List[Dict[str, Any]],
    n_splits: int,
    seed: int,
    has_gpu: bool,
    use_base_margin: bool = False,
    is_boundary_specialist: bool = False,
    n_jobs: Optional[int] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Trains a LightGBM model across all folds."""
    import lightgbm as lgb

    n_samples = sum(len(f["val_idx"]) for f in folds_data)
    n_test = len(folds_data[0]["X_te"])
    oof_preds = np.zeros(n_samples, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)

    tag = "BOUNDARY_SPECIALIST" if is_boundary_specialist else ("BASE_MARGIN" if use_base_margin else "FREE_TREE")
    print(f"\n--- Training LightGBM [{tag}] (Seed: {seed}) ---")

    if is_boundary_specialist:
        lr, leaves, depth, alpha, reg_l, min_child = 0.025, 63, 5, 0.30, 8.0, 50
    elif use_base_margin:
        # Constrained shallow trees force fitting strictly the residual error without re-learning the recipe
        lr, leaves, depth, alpha, reg_l, min_child = 0.025, 31, 5, 0.20, 6.0, 100
    else:
        # Free-Tree high capacity exploratory splits
        lr, leaves, depth, alpha, reg_l, min_child = 0.035, 127, 8, 0.10, 0.5, 40

    cpu_cores = max(1, (os.cpu_count() or 4) - 1) if n_jobs is None else n_jobs

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": lr,
        "num_leaves": leaves,
        "max_depth": depth,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": min_child,
        "reg_alpha": alpha,
        "reg_lambda": reg_l,
        "random_state": seed,
        "verbose": -1,
        "n_jobs": cpu_cores,
    }
    lgb_dev = probe_lgbm_device(has_gpu)
    if lgb_dev != "cpu":
        params["device"] = lgb_dev

    fold_aucs = []
    for fold_info in folds_data:
        fold = fold_info["fold"]
        X_tr = fold_info["X_tr"]
        y_tr = fold_info["y_tr"]
        X_va = fold_info["X_va"]
        y_va = fold_info["y_va"]
        X_te = fold_info["X_te"]
        val_idx = fold_info["val_idx"]

        sw = fold_info["sw_boundary_tr"] if is_boundary_specialist else None
        init_tr = fold_info["margin_tr"] if use_base_margin else None
        init_va = fold_info["margin_va"] if use_base_margin else None
        init_te = fold_info["margin_te"] if use_base_margin else None

        trn_data = lgb.Dataset(X_tr, label=y_tr, weight=sw, init_score=init_tr, free_raw_data=False)
        val_data = lgb.Dataset(X_va, label=y_va, init_score=init_va, reference=trn_data, free_raw_data=False)

        callbacks = [lgb.early_stopping(stopping_rounds=80, verbose=False)]
        num_rounds = 2000 if is_boundary_specialist else (2500 if use_base_margin else 3000)

        model = lgb.train(
            params,
            trn_data,
            num_boost_round=num_rounds,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        val_raw = model.predict(X_va, raw_score=use_base_margin)
        if use_base_margin:
            val_p = expit(init_va + val_raw)
            te_p = expit(init_te + model.predict(X_te, raw_score=True))
        else:
            val_p = val_raw
            te_p = model.predict(X_te)

        oof_preds[val_idx] = val_p.astype(np.float32)
        test_preds += (te_p / n_splits).astype(np.float32)

        f_auc = roc_auc_score(y_va, val_p)
        fold_aucs.append(f_auc)
        print(f"  Fold {fold}/{n_splits} AUC: {f_auc:.6f} (Best Iter: {model.best_iteration})")
        del trn_data, val_data, model
        gc.collect()

    y_true = np.zeros(n_samples, dtype=int)
    for f in folds_data:
        y_true[f["val_idx"]] = f["y_va"]
    overall_auc = roc_auc_score(y_true, oof_preds)
    print(f"[+] LightGBM [{tag}] Mean Fold AUC: {np.mean(fold_aucs):.6f} | Overall OOF AUC: {overall_auc:.6f}")
    return overall_auc, oof_preds, test_preds


def _train_single_xgb_fold(
    fold_info: Dict[str, Any],
    base_params: Dict[str, Any],
    device_str: Optional[str],
    use_base_margin: bool,
    is_boundary_specialist: bool,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Trains a single XGBoost fold with explicit device targeting."""
    import xgboost as xgb

    fold = fold_info["fold"]
    X_tr = fold_info["X_tr"]
    y_tr = fold_info["y_tr"]
    X_va = fold_info["X_va"]
    y_va = fold_info["y_va"]
    X_te = fold_info["X_te"]
    val_idx = fold_info["val_idx"]

    sw = fold_info["sw_boundary_tr"] if is_boundary_specialist else None
    m_tr = fold_info["margin_tr"] if use_base_margin else None
    m_va = fold_info["margin_va"] if use_base_margin else None
    m_te = fold_info["margin_te"] if use_base_margin else None

    # Deep copy params for thread safety
    p = dict(base_params)
    if device_str is not None:
        p["device"] = device_str

    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sw, base_margin=m_tr)
    dval = xgb.DMatrix(X_va, label=y_va, base_margin=m_va)
    dtest = xgb.DMatrix(X_te, base_margin=m_te)

    num_rounds = 2000 if is_boundary_specialist else (2500 if use_base_margin else 3000)
    evallist = [(dval, "eval")]

    model = xgb.train(
        p,
        dtrain,
        num_boost_round=num_rounds,
        evals=evallist,
        early_stopping_rounds=80,
        verbose_eval=False,
    )

    val_p = model.predict(dval).astype(np.float32)
    te_p = model.predict(dtest).astype(np.float32)
    best_it = model.best_iteration
    del dtrain, dval, dtest, model
    gc.collect()

    f_auc = roc_auc_score(y_va, val_p)
    return fold, val_idx, val_p, te_p, f_auc, best_it


def train_xgboost(
    folds_data: List[Dict[str, Any]],
    n_splits: int,
    seed: int,
    has_gpu: bool,
    use_base_margin: bool = False,
    is_boundary_specialist: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Trains an XGBoost model across all folds with dual-GPU distribution."""
    n_samples = sum(len(f["val_idx"]) for f in folds_data)
    n_test = len(folds_data[0]["X_te"])
    oof_preds = np.zeros(n_samples, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)

    tag = "BOUNDARY_SPECIALIST" if is_boundary_specialist else ("BASE_MARGIN" if use_base_margin else "FREE_TREE")
    print(f"\n--- Training XGBoost [{tag}] (Seed: {seed}) ---")

    if is_boundary_specialist:
        lr, depth, min_child, alpha, reg_l, sub, col = 0.020, 4, 20, 0.30, 15.0, 0.90, 0.65
    elif use_base_margin:
        # Constrained shallow trees force residual learning
        lr, depth, min_child, alpha, reg_l, sub, col = 0.025, 5, 20, 0.20, 8.0, 0.85, 0.75
    else:
        # Free-Tree high capacity
        lr, depth, min_child, alpha, reg_l, sub, col = 0.035, 7, 6, 0.10, 3.0, 0.85, 0.75

    xgb_dev = probe_xgb_device(has_gpu)
    num_gpus = get_gpu_count() if has_gpu else 0

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": lr,
        "max_depth": depth,
        "min_child_weight": min_child,
        "subsample": sub,
        "colsample_bytree": col,
        "reg_alpha": alpha,
        "reg_lambda": reg_l,
        "random_state": seed,
        "n_jobs": 1 if (num_gpus >= 2 and xgb_dev == "cuda") else -1,
    }
    if xgb_dev == "cuda":
        params["tree_method"] = "hist"
        params["device"] = "cuda:0"
    elif xgb_dev == "gpu_hist":
        params["tree_method"] = "gpu_hist"
    else:
        params["tree_method"] = "hist"
        params["device"] = "cpu"

    fold_aucs = []

    # If 2+ GPUs detected, parallelize fold execution across cuda:0 and cuda:1
    if num_gpus >= 2 and xgb_dev == "cuda":
        print(f"[+] Dual GPUs active: Distributing XGBoost folds across cuda:0 and cuda:1 in parallel...")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        _train_single_xgb_fold,
                        fold_info,
                        params,
                        f"cuda:{idx % 2}",
                        use_base_margin,
                        is_boundary_specialist,
                    )
                    for idx, fold_info in enumerate(folds_data)
                ]
                for fut in concurrent.futures.as_completed(futures):
                    fold, val_idx, val_p, te_p, f_auc, best_it = fut.result()
                    oof_preds[val_idx] = val_p
                    test_preds += (te_p / n_splits)
                    fold_aucs.append(f_auc)
                    dev_name = f"cuda:{(fold - 1) % 2}"
                    print(f"  Fold {fold}/{n_splits} AUC: {f_auc:.6f} (Best Iter: {best_it}) [{dev_name}]")
        except Exception as e:
            print(f"[!] Note: Parallel multi-GPU encountered: {e}. Executing sequentially on cuda:0.")
            oof_preds.fill(0)
            test_preds.fill(0)
            fold_aucs.clear()
            for fold_info in folds_data:
                fold, val_idx, val_p, te_p, f_auc, best_it = _train_single_xgb_fold(
                    fold_info, params, "cuda:0", use_base_margin, is_boundary_specialist
                )
                oof_preds[val_idx] = val_p
                test_preds += (te_p / n_splits)
                fold_aucs.append(f_auc)
                print(f"  Fold {fold}/{n_splits} AUC: {f_auc:.6f} (Best Iter: {best_it})")
    else:
        dev_str = "cuda:0" if xgb_dev == "cuda" else None
        for fold_info in folds_data:
            fold, val_idx, val_p, te_p, f_auc, best_it = _train_single_xgb_fold(
                fold_info, params, dev_str, use_base_margin, is_boundary_specialist
            )
            oof_preds[val_idx] = val_p
            test_preds += (te_p / n_splits)
            fold_aucs.append(f_auc)
            print(f"  Fold {fold}/{n_splits} AUC: {f_auc:.6f} (Best Iter: {best_it})")

    y_true = np.zeros(n_samples, dtype=int)
    for f in folds_data:
        y_true[f["val_idx"]] = f["y_va"]
    overall_auc = roc_auc_score(y_true, oof_preds)
    print(f"[+] XGBoost [{tag}] Mean Fold AUC: {np.mean(fold_aucs):.6f} | Overall OOF AUC: {overall_auc:.6f}")
    return overall_auc, oof_preds, test_preds


def train_catboost(
    folds_data: List[Dict[str, Any]],
    n_splits: int,
    seed: int,
    has_gpu: bool,
    use_base_margin: bool = False,
    is_boundary_specialist: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Trains a CatBoost model across all folds with multi-GPU and CPU acceleration."""
    from catboost import CatBoostClassifier, Pool

    n_samples = sum(len(f["val_idx"]) for f in folds_data)
    n_test = len(folds_data[0]["X_te"])
    oof_preds = np.zeros(n_samples, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)

    tag = "BOUNDARY_SPECIALIST" if is_boundary_specialist else ("BASE_MARGIN" if use_base_margin else "FREE_TREE")
    print(f"\n--- Training CatBoost [{tag}] (Seed: {seed}) ---")

    if is_boundary_specialist:
        lr, depth, l2_reg = 0.025, 5, 10.0
        n_iters = 2000
    elif use_base_margin:
        lr, depth, l2_reg = 0.025, 5, 8.0
        n_iters = 2000
    else:
        # Free-Tree: depth 6 symmetric tree = 64 leaves; 2x faster than depth 7 with superior regularization
        lr, depth, l2_reg = 0.035, 6, 4.0
        n_iters = 2500

    cb_dev = probe_cb_device(has_gpu)
    num_gpus = get_gpu_count() if has_gpu else 0

    params = {
        "iterations": n_iters,
        "learning_rate": lr,
        "depth": depth,
        "l2_leaf_reg": l2_reg,
        "eval_metric": "AUC",
        "random_seed": seed,
        "verbose": False,
        "task_type": cb_dev,
        "early_stopping_rounds": 80,
        "thread_count": -1,
    }
    if cb_dev == "GPU":
        params["border_count"] = 128
        if num_gpus >= 2:
            params["devices"] = "0:1"
            print("[+] CatBoost multi-GPU enabled across Dual Tesla T4s (devices='0:1')")
        else:
            params["devices"] = "0"

    fold_aucs = []
    for fold_info in folds_data:
        fold = fold_info["fold"]
        X_tr = fold_info["X_tr"]
        y_tr = fold_info["y_tr"]
        X_va = fold_info["X_va"]
        y_va = fold_info["y_va"]
        X_te = fold_info["X_te"]
        val_idx = fold_info["val_idx"]

        sw = fold_info["sw_boundary_tr"] if is_boundary_specialist else None
        m_tr = fold_info["margin_tr"] if use_base_margin else None
        m_va = fold_info["margin_va"] if use_base_margin else None
        m_te = fold_info["margin_te"] if use_base_margin else None

        pool_tr = Pool(X_tr, y_tr, weight=sw, baseline=m_tr)
        pool_va = Pool(X_va, y_va, baseline=m_va)
        pool_te = Pool(X_te, baseline=m_te)

        cb = CatBoostClassifier(**params)
        cb.fit(pool_tr, eval_set=pool_va)

        val_p = cb.predict_proba(pool_va)[:, 1]
        te_p = cb.predict_proba(pool_te)[:, 1]

        oof_preds[val_idx] = val_p.astype(np.float32)
        test_preds += (te_p / n_splits).astype(np.float32)

        f_auc = roc_auc_score(y_va, val_p)
        fold_aucs.append(f_auc)
        print(f"  Fold {fold}/{n_splits} AUC: {f_auc:.6f} (Best Iter: {cb.get_best_iteration()})")
        del pool_tr, pool_va, pool_te, cb
        gc.collect()

    y_true = np.zeros(n_samples, dtype=int)
    for f in folds_data:
        y_true[f["val_idx"]] = f["y_va"]
    overall_auc = roc_auc_score(y_true, oof_preds)
    print(f"[+] CatBoost [{tag}] Mean Fold AUC: {np.mean(fold_aucs):.6f} | Overall OOF AUC: {overall_auc:.6f}")
    return overall_auc, oof_preds, test_preds


def hierarchical_gated_blend(
    stream_a_oof: np.ndarray,
    stream_a_test: np.ndarray,
    stream_b_oof: np.ndarray,
    stream_b_test: np.ndarray,
    stream_boundary_oof: np.ndarray,
    stream_boundary_test: np.ndarray,
    y_true: np.ndarray,
    recipe_diff_tr: np.ndarray,
    recipe_diff_te: np.ndarray,
    stream_nn_oof: Optional[np.ndarray] = None,
    stream_nn_test: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Two-Tier Hierarchical Stacking Architecture:
    Tier 1: Global Backbone in Logit Space (Stream A Free-Trees + Stream B Base-Margin Residuals + optional Stream D Neural).
    Tier 2: Gaussian Gated Refinement with Stream C (Boundary Specialist).
    """
    print("\n=================================================================")
    print("[*] TWO-TIER HIERARCHICAL GATED STACKING OPTIMIZATION")
    print("=================================================================")

    # Clip to prevent logit explosion
    eps = 1e-7
    a_oof_c = np.clip(stream_a_oof, eps, 1.0 - eps)
    b_oof_c = np.clip(stream_b_oof, eps, 1.0 - eps)
    a_te_c = np.clip(stream_a_test, eps, 1.0 - eps)
    b_te_c = np.clip(stream_b_test, eps, 1.0 - eps)
    bound_oof_c = np.clip(stream_boundary_oof, eps, 1.0 - eps)
    bound_te_c = np.clip(stream_boundary_test, eps, 1.0 - eps)

    l_a_oof = logit(a_oof_c)
    l_b_oof = logit(b_oof_c)
    l_a_te = logit(a_te_c)
    l_b_te = logit(b_te_c)

    has_nn = stream_nn_oof is not None and stream_nn_test is not None
    if has_nn:
        nn_oof_c = np.clip(stream_nn_oof, eps, 1.0 - eps)
        nn_te_c = np.clip(stream_nn_test, eps, 1.0 - eps)
        l_nn_oof = logit(nn_oof_c)
        l_nn_te = logit(nn_te_c)

        def loss_tier1_nn(w):
            w_a, w_b, w_nn = w
            l_back = w_a * l_a_oof + w_b * l_b_oof + w_nn * l_nn_oof
            p_back = expit(l_back)
            return -roc_auc_score(y_true, p_back)

        res_tier1 = minimize(
            loss_tier1_nn,
            [0.48, 0.48, 0.04],
            method="SLSQP",
            bounds=[(0.05, 0.90), (0.05, 0.90), (0.0, 0.10)],
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        )
        best_w_a, best_w_b, best_w_nn = res_tier1.x
        oof_backbone = expit(best_w_a * l_a_oof + best_w_b * l_b_oof + best_w_nn * l_nn_oof)
        test_backbone = expit(best_w_a * l_a_te + best_w_b * l_b_te + best_w_nn * l_nn_te)
        print(f"[Tier 1 Backbone] Weights: Stream A={best_w_a:.4f} | Stream B={best_w_b:.4f} | Stream D (Neural)={best_w_nn:.4f}")
    else:
        # 1. Optimize Tier 1 Backbone (weight of A vs B)
        def loss_tier1(w):
            w_a = w[0]
            w_b = 1.0 - w_a
            l_back = w_a * l_a_oof + w_b * l_b_oof
            p_back = expit(l_back)
            return -roc_auc_score(y_true, p_back)

        res_tier1 = minimize(loss_tier1, [0.5], method="Nelder-Mead")
        best_w_a = float(np.clip(res_tier1.x[0], 0.05, 0.95))
        best_w_b = 1.0 - best_w_a
        oof_backbone = expit(best_w_a * l_a_oof + best_w_b * l_b_oof)
        test_backbone = expit(best_w_a * l_a_te + best_w_b * l_b_te)
        print(f"[Tier 1 Backbone] Stream A Weight: {best_w_a:.4f} | Stream B Weight: {best_w_b:.4f}")

    backbone_auc = roc_auc_score(y_true, oof_backbone)
    print(f"[Tier 1 Backbone] OOF ROC-AUC:    {backbone_auc:.6f}")

    # 2. Optimize Tier 2 Gaussian Gate: p_final = (1 - g(x)) * p_backbone + g(x) * p_boundary
    # where g(x) = alpha * exp(-(z / sigma)^2)
    def loss_gate(params):
        alpha, sigma = params
        g_tr = alpha * np.exp(-((recipe_diff_tr / sigma) ** 2))
        p_final = (1.0 - g_tr) * oof_backbone + g_tr * bound_oof_c
        return -roc_auc_score(y_true, p_final)

    init_params = [0.40, 0.25]
    bounds = [(0.0, 1.0), (0.10, 0.60)]
    res_gate = minimize(loss_gate, init_params, method="L-BFGS-B", bounds=bounds)
    best_alpha, best_sigma = res_gate.x

    g_tr = best_alpha * np.exp(-((recipe_diff_tr / best_sigma) ** 2))
    g_te = best_alpha * np.exp(-((recipe_diff_te / best_sigma) ** 2))

    oof_final = (1.0 - g_tr) * oof_backbone + g_tr * bound_oof_c
    test_final = (1.0 - g_te) * test_backbone + g_te * bound_te_c
    final_auc = roc_auc_score(y_true, oof_final)

    if final_auc < backbone_auc:
        print(f"[-] Gated refinement ({final_auc:.6f}) <= backbone ({backbone_auc:.6f}). Preserving pristine global backbone.")
        oof_final = oof_backbone
        test_final = test_backbone
        final_auc = backbone_auc
        best_alpha = 0.0

    print(f"[Tier 2 Gate]     Optimal Alpha: {best_alpha:.4f} | Sigma: {best_sigma:.4f}")
    print(f"[Tier 2 Gated]    Final OOF ROC-AUC:  {final_auc:.6f} (+{final_auc - backbone_auc:+.6f} over backbone)")

    # 3. Zone-Specific Forensic Metrics
    z_abs = np.abs(recipe_diff_tr)
    auc_b10 = roc_auc_score(y_true[z_abs < 0.10], oof_final[z_abs < 0.10])
    auc_b20 = roc_auc_score(y_true[z_abs < 0.20], oof_final[z_abs < 0.20])
    auc_b30 = roc_auc_score(y_true[z_abs < 0.30], oof_final[z_abs < 0.30])
    auc_tail = roc_auc_score(y_true[z_abs >= 0.50], oof_final[z_abs >= 0.50])

    print("\n[Forensic Sub-Zone Performance]:")
    print(f"  Boundary |z| < 0.10 AUC: {auc_b10:.6f} (Samples: {(z_abs < 0.10).sum()})")
    print(f"  Boundary |z| < 0.20 AUC: {auc_b20:.6f} (Samples: {(z_abs < 0.20).sum()})")
    print(f"  Boundary |z| < 0.30 AUC: {auc_b30:.6f} (Samples: {(z_abs < 0.30).sum()})")
    print(f"  Clean Tail |z| >= 0.50 AUC: {auc_tail:.6f} (Samples: {(z_abs >= 0.50).sum()})")

    meta = {
        "backbone_w_a": best_w_a,
        "backbone_w_b": best_w_b,
        "gate_alpha": best_alpha,
        "gate_sigma": best_sigma,
        "backbone_auc": float(backbone_auc),
        "final_auc": float(final_auc),
        "delta_auc": float(final_auc - backbone_auc),
        "auc_boundary_10": float(auc_b10),
        "auc_boundary_20": float(auc_b20),
        "auc_boundary_30": float(auc_b30),
        "auc_tail": float(auc_tail),
    }
    return final_auc, oof_final, test_final, meta


def make_zero_tie_ranks(primary_scores: np.ndarray, secondary_scores: Optional[np.ndarray] = None) -> np.ndarray:
    """Continuous Lexicographical Zero-Tie Ranking (np.lexsort)."""
    n = len(primary_scores)
    if secondary_scores is not None and len(secondary_scores) == n:
        sort_order = np.lexsort((secondary_scores, primary_scores))
    else:
        sort_order = np.argsort(primary_scores, kind="mergesort")

    ranks = np.empty(n, dtype=np.float64)
    ranks[sort_order] = (np.arange(n, dtype=np.float64) + 0.5) / n
    return ranks


def extract_recipe_diff(df: pd.DataFrame) -> np.ndarray:
    """Extracts or computes (Buy_Score - 5.61235) with guaranteed fallback."""
    if "feat_recipe_dist_to_boundary" in df.columns:
        return df["feat_recipe_dist_to_boundary"].values
    if "feat_buy_recipe_score" in df.columns:
        return df["feat_buy_recipe_score"].values - 5.61235
    inc = df["Annual_Income_USD"].astype(float).values
    env = df["Environmental_Concern_Level"].astype(float).values if "Environmental_Concern_Level" in df.columns else 3.0
    sub = (df["Subsidy_Available"].astype(str) == "Yes").astype(float).values if "Subsidy_Available" in df.columns else 0.0
    anx = df["Range_Anxiety_Level"].astype(str).values if "Range_Anxiety_Level" in df.columns else "Low"
    score = (
        1.2 * (inc / 100000.0)
        + 0.6 * env
        + 2.0 * sub
        - 1.0 * (anx == "Medium").astype(float)
        - 3.0 * (anx == "High").astype(float)
    )
    return score - 5.61235


def extract_recipe_score(df: pd.DataFrame) -> np.ndarray:
    """Extracts or computes the raw linear Buy_Score for zero-tie ranking."""
    if "feat_buy_recipe_score" in df.columns:
        return df["feat_buy_recipe_score"].values
    return extract_recipe_diff(df) + 5.61235


def main():
    parser = argparse.ArgumentParser(description="KAMAS Top-1 Execution Engine")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of Stratified K-Fold splits")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds to train")
    parser.add_argument("--models", type=str, nargs="+", default=["lgbm", "xgboost", "catboost"], help="Base models")
    parser.add_argument("--data-dir", type=str, default=None, help="Custom dataset directory")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/models_top1", help="Output directory")
    args = parser.parse_args()

    start_total = time.time()
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    has_gpu = detect_gpu()

    # 1. Ingestion
    data_dir = locate_data_dir(args.data_dir)
    train_raw, test_raw, orig_raw = load_dataset(data_dir)
    test_ids = test_raw["id"]

    # 2. Build 109 Grandmaster Features
    print("\n[*] Generating 109 Grandmaster Features (Simpson Inversion, Boundary Geometry, Modulo Artifacts)...")
    train_feat, test_feat, features, te_cols = build_grandmaster_features(train_raw, test_raw, orig_raw)
    print(f"[+] Features constructed: {len(features)} total | Target-encoded: {len(te_cols)}")

    global train_feat_y
    train_feat_y = train_feat[TARGET].values

    recipe_diff_tr = extract_recipe_diff(train_feat)
    recipe_diff_te = extract_recipe_diff(test_feat)
    sec_score_te = extract_recipe_score(test_feat)

    # Accumulators across seeds
    all_stream_a_oof = np.zeros(len(train_feat), dtype=np.float32)
    all_stream_a_test = np.zeros(len(test_feat), dtype=np.float32)
    all_stream_b_oof = np.zeros(len(train_feat), dtype=np.float32)
    all_stream_b_test = np.zeros(len(test_feat), dtype=np.float32)
    all_boundary_oof = np.zeros(len(train_feat), dtype=np.float32)
    all_boundary_test = np.zeros(len(test_feat), dtype=np.float32)

    total_seeds = len(args.seeds)
    for s_idx, seed in enumerate(args.seeds, 1):
        print(f"\n=================================================================")
        print(f"[*] SEED CYCLE {s_idx}/{total_seeds} (Seed: {seed})")
        print(f"=================================================================")

        folds_data = prepare_seed_folds(train_feat, test_feat, features, te_cols, args.n_splits, seed)

        # -------------------------------------------------------------
        # STREAM A: Free-Tree GBDTs
        # -------------------------------------------------------------
        seed_a_oof = np.zeros(len(train_feat), dtype=np.float32)
        seed_a_test = np.zeros(len(test_feat), dtype=np.float32)
        n_a = len(args.models)

        # Execute LightGBM (CPU) and XGBoost (Dual GPU) concurrently to maximize hardware saturation
        if "lgbm" in args.models and "xgboost" in args.models and has_gpu:
            print("\n[+] CONCURRENT PIPELINE: Launching LightGBM on CPU (3 vCPUs) and XGBoost on Dual Tesla T4s simultaneously!")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pe:
                fut_lgb = pe.submit(train_lgbm, folds_data, args.n_splits, seed, has_gpu, False, False)
                fut_xgb = pe.submit(train_xgboost, folds_data, args.n_splits, seed, has_gpu, False, False)

                _, oof_lgb, te_lgb = fut_lgb.result()
                _, oof_xgb, te_xgb = fut_xgb.result()

            seed_a_oof += (oof_lgb / n_a) + (oof_xgb / n_a)
            seed_a_test += (te_lgb / n_a) + (te_xgb / n_a)

            if "catboost" in args.models:
                _, oof_cb, te_cb = train_catboost(folds_data, args.n_splits, seed, has_gpu, use_base_margin=False)
                seed_a_oof += oof_cb / n_a
                seed_a_test += te_cb / n_a
        else:
            for m in args.models:
                if m == "lgbm":
                    _, oof_m, te_m = train_lgbm(folds_data, args.n_splits, seed, has_gpu, use_base_margin=False)
                elif m == "xgboost":
                    _, oof_m, te_m = train_xgboost(folds_data, args.n_splits, seed, has_gpu, use_base_margin=False)
                elif m == "catboost":
                    _, oof_m, te_m = train_catboost(folds_data, args.n_splits, seed, has_gpu, use_base_margin=False)
                seed_a_oof += oof_m / n_a
                seed_a_test += te_m / n_a

        all_stream_a_oof += seed_a_oof / total_seeds
        all_stream_a_test += seed_a_test / total_seeds

        # -------------------------------------------------------------
        # STREAM B: Base-Margin Residual GBDTs
        # -------------------------------------------------------------
        seed_b_oof = np.zeros(len(train_feat), dtype=np.float32)
        seed_b_test = np.zeros(len(test_feat), dtype=np.float32)
        n_b = len(args.models)

        if "lgbm" in args.models and "xgboost" in args.models and has_gpu:
            print("\n[+] CONCURRENT PIPELINE: Launching Base-Margin LightGBM (CPU) and XGBoost (Dual GPU) simultaneously!")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pe:
                fut_lgb = pe.submit(train_lgbm, folds_data, args.n_splits, seed, has_gpu, True, False)
                fut_xgb = pe.submit(train_xgboost, folds_data, args.n_splits, seed, has_gpu, True, False)

                _, oof_lgb, te_lgb = fut_lgb.result()
                _, oof_xgb, te_xgb = fut_xgb.result()

            seed_b_oof += (oof_lgb / n_b) + (oof_xgb / n_b)
            seed_b_test += (te_lgb / n_b) + (te_xgb / n_b)

            if "catboost" in args.models:
                _, oof_cb, te_cb = train_catboost(folds_data, args.n_splits, seed, has_gpu, use_base_margin=True)
                seed_b_oof += oof_cb / n_b
                seed_b_test += te_cb / n_b
        else:
            for m in args.models:
                if m == "lgbm":
                    _, oof_m, te_m = train_lgbm(folds_data, args.n_splits, seed, has_gpu, use_base_margin=True)
                elif m == "xgboost":
                    _, oof_m, te_m = train_xgboost(folds_data, args.n_splits, seed, has_gpu, use_base_margin=True)
                elif m == "catboost":
                    _, oof_m, te_m = train_catboost(folds_data, args.n_splits, seed, has_gpu, use_base_margin=True)
                seed_b_oof += oof_m / n_b
                seed_b_test += te_m / n_b

        all_stream_b_oof += seed_b_oof / total_seeds
        all_stream_b_test += seed_b_test / total_seeds

        # -------------------------------------------------------------
        # STREAM C: Boundary Specialist GBDT (XGBoost)
        # -------------------------------------------------------------
        _, seed_c_oof, seed_c_test = train_xgboost(
            folds_data, args.n_splits, seed, has_gpu, use_base_margin=True, is_boundary_specialist=True
        )
        all_boundary_oof += seed_c_oof / total_seeds
        all_boundary_test += seed_c_test / total_seeds

    # -----------------------------------------------------------------
    # OPTIONAL STREAM D: Neural Manifold Auto-Discovery
    # -----------------------------------------------------------------
    nn_oof = None
    nn_test = None
    nn_candidates = [
        PROJECT_ROOT / "models" / "nn_tabular",
        Path("/kaggle/working/electric-vehicle/models/nn_tabular"),
        Path("/kaggle/working/models/nn_tabular"),
    ]
    for nnc in nn_candidates:
        if (nnc / "oof_preds.parquet").exists() and (nnc / "test_preds.parquet").exists():
            try:
                df_nn_oof = pl.read_parquet(nnc / "oof_preds.parquet").to_pandas()
                df_nn_test = pl.read_parquet(nnc / "test_preds.parquet").to_pandas()
                col_oof = "pred" if "pred" in df_nn_oof.columns else ("oof_pred" if "oof_pred" in df_nn_oof.columns else df_nn_oof.columns[-1])
                col_test = "pred" if "pred" in df_nn_test.columns else df_nn_test.columns[-1]
                if len(df_nn_oof) == len(train_feat) and len(df_nn_test) == len(test_feat):
                    nn_oof = df_nn_oof[col_oof].values.astype(np.float32)
                    nn_test = df_nn_test[col_test].values.astype(np.float32)
                    print(f"[+] Discovered & Integrated Stream D: Tabular Neural Network from {nnc}")
                    break
            except Exception as e:
                print(f"[!] Note on neural tabular auto-discovery: {e}")

    # -----------------------------------------------------------------
    # HIERARCHICAL GATED ENSEMBLE
    # -----------------------------------------------------------------
    final_auc, oof_final, test_final, meta = hierarchical_gated_blend(
        all_stream_a_oof,
        all_stream_a_test,
        all_stream_b_oof,
        all_stream_b_test,
        all_boundary_oof,
        all_boundary_test,
        train_feat_y,
        recipe_diff_tr,
        recipe_diff_te,
        stream_nn_oof=nn_oof,
        stream_nn_test=nn_test,
    )

    # -----------------------------------------------------------------
    # SAVE ARTIFACTS & GENERATE SUBMISSIONS
    # -----------------------------------------------------------------
    work_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else PROJECT_ROOT
    ensemble_dir = out_path / "ensemble_top1"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    # 1. Champion: Calibrated Probability with Micro-Jitter Zero-Tie Resolution
    if sec_score_te is not None:
        sec_norm = (sec_score_te - np.nanmean(sec_score_te)) / (np.nanstd(sec_score_te) + 1e-7)
        micro_zero_tie = test_final + 1e-9 * sec_norm
    else:
        micro_zero_tie = test_final

    sub_micro = pd.DataFrame({"id": test_ids, TARGET: micro_zero_tie})
    sub_micro.to_csv(work_dir / "submission_micro_zero_tie.csv", index=False)
    sub_micro.to_csv(work_dir / "submission_top1_champion.csv", index=False)
    sub_micro.to_csv(work_dir / "submission.csv", index=False)
    sub_micro.to_parquet(work_dir / "submission_top1_champion.parquet", index=False)
    sub_micro.to_parquet(work_dir / "submission.parquet", index=False)

    print(f"\n[+] CHAMPION Micro-Jitter Zero-Tie Submission written to: {work_dir / 'submission.csv'}")
    print(f"    Distribution Mean: {micro_zero_tie.mean():.6f} (Ground Truth: 0.17485) | Std: {micro_zero_tie.std():.6f}")

    # 2. Pure Probability Submission (for ablation checking)
    pd.DataFrame({"id": test_ids, TARGET: test_final}).to_csv(work_dir / "submission_pure_prob.csv", index=False)

    # 3. Uniform Rank Submission (np.lexsort)
    zero_tie_uniform = make_zero_tie_ranks(test_final, sec_score_te)
    pd.DataFrame({"id": test_ids, TARGET: zero_tie_uniform}).to_csv(work_dir / "submission_uniform_rank.csv", index=False)

    # Save OOF and Metadata
    oof_df = pd.DataFrame({
        "id": range(len(oof_final)),
        "oof_pred": oof_final,
        "stream_a_oof": all_stream_a_oof,
        "stream_b_oof": all_stream_b_oof,
        "stream_c_boundary_oof": all_boundary_oof,
        "target": train_feat_y,
    })
    oof_df.to_parquet(ensemble_dir / "oof_preds.parquet", index=False)
    oof_df.to_parquet(work_dir / "oof_preds_top1.parquet", index=False)

    meta["total_runtime_minutes"] = (time.time() - start_total) / 60.0
    with open(ensemble_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(work_dir / "metrics_top1.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n=================================================================")
    print(f"[+] KAMAS Top-1 Training & Stacking Complete in {meta['total_runtime_minutes']:.1f} minutes!")
    print(f"    Final Gated Ensemble OOF ROC-AUC: {final_auc:.6f}")
    print(f"    Ready for Submission: {work_dir / 'submission.csv'}")
    print(f"=================================================================")


if __name__ == "__main__":
    main()
