#!/usr/bin/env python3
"""
PyTorch Tabular Residual Neural Network (Tabular ResNet with Generative Prior Head)
Stream D for Kaggle Playground Series s6e9 (Predicting Electric Vehicle Purchases).

Key Architectural Innovations:
1. Entity Embeddings for Discrete Artifacts:
   - 'Age' mapped to 45-dim categorical embedding (breaks the 45-sided die uniform artifact).
   - Domain categoricals (City_Type, Range_Anxiety, Subsidy, etc.) with optimal embedding dimensions.
2. Residual Skip Blocks:
   - Multi-stage ResNet blocks (Linear -> BN -> GELU -> Dropout -> Linear -> BN -> Skip Add)
     for smooth, non-axis-aligned decision boundaries.
3. Generative Recipe Prior Head:
   - Injects the reverse-engineered linear recipe score (Buy_Score - 5.61235) as a learnable additive prior.
   - Forces the neural network to optimize exclusively on the subtle non-linear residual curvature.
4. 10-Fold Stratified Cross-Validation:
   - Perfectly aligns with certified 10-fold splits for seamless Tier 1 auto-discovery in kaggle_train_top1.py.
5. High-Throughput Hardware Acceleration:
   - Optimized for Kaggle Tesla T4 GPU (Batch Size: 4096, pin_memory, fast CUDA execution in < 10 mins).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:  # type: ignore
        pass
    class _MockTorch:  # type: ignore
        class Tensor:
            pass
        class device:
            pass
    torch = _MockTorch()  # type: ignore
    class _MockNN:  # type: ignore
        class Module:
            pass
        class ModuleList(list):
            pass
        class Identity:
            pass
        class Sequential:
            def __init__(self, *args): pass
        class BatchNorm1d:
            def __init__(self, *args, **kwargs): pass
        class Linear:
            def __init__(self, *args, **kwargs): pass
        class GELU:
            def __init__(self, *args, **kwargs): pass
        class Dropout:
            def __init__(self, *args, **kwargs): pass
        class Embedding:
            def __init__(self, *args, **kwargs): pass
        def Parameter(self, *args, **kwargs): return None
    nn = _MockNN()

from src.features import build_grandmaster_features, TARGET
from kaggle.paths import resolve_data_dir, resolve_output_dir


def detect_device() -> "torch.device":
    """Detects available PyTorch accelerator."""
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for kaggle_train_nn.py. Install torch or run in Kaggle GPU environment.")
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        dev_name = torch.cuda.get_device_name(0)
        print(f"[+] PyTorch CUDA acceleration ENABLED: {dev_name} ({num_devices} device(s) detected)")
        return torch.device("cuda:0")
    print("[*] PyTorch CUDA not detected. Utilizing CPU.")
    return torch.device("cpu")


def extract_recipe_diff(df: pd.DataFrame) -> np.ndarray:
    """Extracts or computes (Buy_Score - 5.61235) with guaranteed fallback."""
    if "feat_recipe_dist_to_boundary" in df.columns:
        return df["feat_recipe_dist_to_boundary"].values.astype(np.float32)
    if "feat_buy_recipe_score" in df.columns:
        return (df["feat_buy_recipe_score"].values - 5.61235).astype(np.float32)
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
    return (score - 5.61235).astype(np.float32)


class TabularDataset(Dataset):
    """PyTorch Dataset supporting Categorical, Numerical, Recipe Baseline, and Target tensors."""
    def __init__(
        self,
        cats: np.ndarray,
        nums: np.ndarray,
        recipe_diff: np.ndarray,
        targets: Optional[np.ndarray] = None,
    ):
        self.cats = torch.tensor(cats, dtype=torch.long)
        self.nums = torch.tensor(nums, dtype=torch.float32)
        self.recipe_diff = torch.tensor(recipe_diff, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None

    def __len__(self) -> int:
        return len(self.nums)

    def __getitem__(self, idx: int):
        if self.targets is not None:
            return self.cats[idx], self.nums[idx], self.recipe_diff[idx], self.targets[idx]
        return self.cats[idx], self.nums[idx], self.recipe_diff[idx]


class ResNetBlock(nn.Module):
    """Residual Block with LayerNorm, GELU, and Dropout for Tabular Deep Learning."""
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.bn1(x)
        out = self.fc1(out)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.fc2(out)
        out = self.dropout(self.gelu(out))
        return residual + out


class TabularResNet(nn.Module):
    """
    State-of-the-art Tabular ResNet with Learned Entity Embeddings
    and Generative Prior Head for binary classification.
    """
    def __init__(
        self,
        emb_sizes: List[Tuple[int, int]],
        n_numerical: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.15,
        use_recipe_prior: bool = True,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=n_cats, embedding_dim=emb_dim)
            for n_cats, emb_dim in emb_sizes
        ])
        total_emb_dim = sum(e for _, e in emb_sizes)
        self.num_bn = nn.BatchNorm1d(n_numerical) if n_numerical > 0 else nn.Identity()

        in_dim = total_emb_dim + n_numerical
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])

        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

        self.use_recipe_prior = use_recipe_prior
        if use_recipe_prior:
            # Learnable scalar scaling for generative prior
            self.w_recipe = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor, recipe_diff: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(emb_outs, dim=1) if emb_outs else torch.empty((x_cat.size(0), 0), device=x_cat.device)
        x_num = self.num_bn(x_num)
        x = torch.cat([x_emb, x_num], dim=1)

        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)

        logits = self.head(h).squeeze(-1)
        if self.use_recipe_prior and recipe_diff is not None:
            logits = logits + self.w_recipe * recipe_diff
        return logits


def locate_data_dir(custom_path: Optional[str] = None) -> Path:
    """Locates dataset with dynamic environment resolution."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    candidates.extend([
        Path("/kaggle/input/competitions/playground-series-s6e9"),
        Path("/kaggle/input/playground-series-s6e9"),
        PROJECT_ROOT / "data" / "processed" / "playground-series-s6e9",
        PROJECT_ROOT / "data",
    ])
    for c in candidates:
        if c.exists() and ((c / "train.parquet").exists() or (c / "train.csv").exists()):
            return c
    raise FileNotFoundError(f"Could not locate dataset in candidates: {candidates}")


def load_dataset(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Loads train, test, and original dataset with Parquet prioritized."""
    print(f"[*] Ingesting dataset files from: {data_dir}")
    if (data_dir / "train.parquet").exists():
        tr = pl.read_parquet(data_dir / "train.parquet").to_pandas()
    else:
        tr = pd.read_csv(data_dir / "train.csv")

    if (data_dir / "test.parquet").exists():
        te = pl.read_parquet(data_dir / "test.parquet").to_pandas()
    else:
        te = pd.read_csv(data_dir / "test.csv")

    orig_path = PROJECT_ROOT / "data" / "original" / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
    orig = pd.read_csv(orig_path) if orig_path.exists() else None
    if orig is not None:
        print(f"[+] Ground-Truth Original Dataset loaded from: {orig_path} ({len(orig)} samples)")

    print(f"[+] Ingested train: {tr.shape}, test: {te.shape}")
    return tr, te, orig


def train_nn_model(
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    n_splits: int = 10,
    epochs: int = 15,
    batch_size: int = 4096,
    lr: float = 1.5e-3,
    hidden_dim: int = 256,
    num_blocks: int = 3,
    dropout: float = 0.15,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Executes full 10-fold Stratified Cross-Validation for Stream D Tabular ResNet.
    Saves certified OOF and test prediction artifacts for immediate top-tier stacking.
    """
    start_time = time.time()
    device = detect_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    base_out = Path(output_dir) if output_dir else PROJECT_ROOT
    model_dir = base_out / "models" / "nn_tabular"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data & Generate 109 Grandmaster features
    d_dir = locate_data_dir(data_dir)
    train_raw, test_raw, orig_raw = load_dataset(d_dir)
    test_ids = test_raw["id"].values

    print("\n[*] Constructing Grandmaster Tabular Feature Matrix...")
    train_feat, test_feat, features, te_cols = build_grandmaster_features(train_raw, test_raw, orig_raw)
    print(f"[+] Feature matrix constructed: {len(features)} total columns")

    y_train = train_feat[TARGET].values.astype(np.float32)
    recipe_diff_tr = extract_recipe_diff(train_feat)
    recipe_diff_te = extract_recipe_diff(test_feat)

    # 2. Extract and Categorize Explicit Entities vs Numericals
    # Explicit discrete columns that benefit from Entity Embeddings
    explicit_cats = [
        "Age",
        "City_Type",
        "Range_Anxiety_Level",
        "Subsidy_Available",
        "Vehicle_Type_Preference",
        "Payment_Method",
        "Marital_Status",
        "Gender",
        "Employment_Status",
        "Home_Ownership",
    ]
    # Identify active categoricals in features
    cat_cols = [c for c in explicit_cats if c in train_feat.columns]
    # Any additional object/string columns
    for c in features:
        if c not in cat_cols and train_feat[c].dtype == object:
            cat_cols.append(c)

    # Numericals = remaining non-target features
    num_cols = [c for c in features if c not in cat_cols and c != TARGET and c != "id"]

    print(f"[*] Feature Partition: {len(cat_cols)} Categoricals (with Entity Embeddings) | {len(num_cols)} Numericals")

    # Encode Categoricals
    emb_sizes = []
    cat_tr_encoded = np.zeros((len(train_feat), len(cat_cols)), dtype=np.int64)
    cat_te_encoded = np.zeros((len(test_feat), len(cat_cols)), dtype=np.int64)

    for idx, col in enumerate(cat_cols):
        tr_s = train_feat[col].astype(str).values
        te_s = test_feat[col].astype(str).values
        unique_vals = sorted(list(set(tr_s).union(set(te_s))))
        val_map = {v: i for i, v in enumerate(unique_vals)}

        cat_tr_encoded[:, idx] = np.array([val_map.get(v, 0) for v in tr_s], dtype=np.int64)
        cat_te_encoded[:, idx] = np.array([val_map.get(v, 0) for v in te_s], dtype=np.int64)

        n_unique = len(unique_vals)
        # Allocate up to 16 embedding dims for Age (45 categories) and 4-8 for smaller sets
        emb_dim = min(16, max(4, (n_unique + 1) // 2))
        emb_sizes.append((n_unique, emb_dim))

    # 3. Partition Folds
    folds_path = d_dir / "folds.parquet"
    if folds_path.exists():
        folds = pl.read_parquet(folds_path)["fold"].to_numpy()
        print(f"[+] Using certified folds from: {folds_path}")
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = np.zeros(len(train_feat), dtype=int)
        for f, (_, val_idx) in enumerate(skf.split(train_feat, y_train)):
            folds[val_idx] = f
        print(f"[+] Generated {n_splits}-fold Stratified CV partition.")

    oof_preds = np.zeros(len(train_feat), dtype=np.float32)
    test_preds = np.zeros(len(test_feat), dtype=np.float32)
    fold_aucs = []

    print("\n=================================================================")
    print(f"[*] TRAINING {n_splits}-FOLD TABULAR RESNET (STREAM D)")
    print(f"    Batch Size: {batch_size} | Epochs: {epochs} | LR: {lr} | Device: {device}")
    print(f"    Residual Prior Head: ENABLED | Hidden Dim: {hidden_dim} | Blocks: {num_blocks}")
    print("=================================================================")

    # 4. Training Loop Across Folds
    for fold in range(n_splits):
        f_start = time.time()
        tr_mask = folds != fold
        va_mask = folds == fold

        # Robust Scaling strictly fitted on train fold to avoid data leakage
        scaler = RobustScaler()
        tr_num = scaler.fit_transform(train_feat.loc[tr_mask, num_cols].values.astype(np.float32))
        va_num = scaler.transform(train_feat.loc[va_mask, num_cols].values.astype(np.float32))
        te_num = scaler.transform(test_feat[num_cols].values.astype(np.float32))

        tr_cat = cat_tr_encoded[tr_mask]
        va_cat = cat_tr_encoded[va_mask]
        te_cat = cat_te_encoded

        tr_rec = recipe_diff_tr[tr_mask]
        va_rec = recipe_diff_tr[va_mask]
        te_rec = recipe_diff_te

        train_ds = TabularDataset(tr_cat, tr_num, tr_rec, y_train[tr_mask])
        val_ds = TabularDataset(va_cat, va_num, va_rec, y_train[va_mask])
        test_ds = TabularDataset(te_cat, te_num, te_rec)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

        model = TabularResNet(
            emb_sizes=emb_sizes,
            n_numerical=len(num_cols),
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            use_recipe_prior=True,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        best_auc = 0.0
        best_val_preds = None
        best_state = None

        for ep in range(1, epochs + 1):
            model.train()
            for x_c, x_n, x_r, y_b in train_loader:
                x_c, x_n, x_r, y_b = x_c.to(device), x_n.to(device), x_r.to(device), y_b.to(device)
                optimizer.zero_grad()
                logits = model(x_c, x_n, x_r)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Fast evaluation on validation fold
            model.eval()
            val_probs = []
            with torch.no_grad():
                for x_c, x_n, x_r, _ in val_loader:
                    x_c, x_n, x_r = x_c.to(device), x_n.to(device), x_r.to(device)
                    probs = torch.sigmoid(model(x_c, x_n, x_r)).cpu().numpy()
                    val_probs.extend(probs)
            val_probs = np.array(val_probs, dtype=np.float32)
            ep_auc = roc_auc_score(y_train[va_mask], val_probs)

            if ep_auc > best_auc:
                best_auc = ep_auc
                best_val_preds = val_probs
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        oof_preds[va_mask] = best_val_preds
        fold_aucs.append(best_auc)
        f_time = time.time() - f_start
        print(f"  [Fold {fold+1}/{n_splits}] Best Val ROC-AUC: {best_auc:.6f} ({f_time:.1f}s)")

        # Restore best checkpoint and evaluate test fold
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        fold_te_probs = []
        with torch.no_grad():
            for x_c, x_n, x_r in test_loader:
                x_c, x_n, x_r = x_c.to(device), x_n.to(device), x_r.to(device)
                probs = torch.sigmoid(model(x_c, x_n, x_r)).cpu().numpy()
                fold_te_probs.extend(probs)
        test_preds += (np.array(fold_te_probs, dtype=np.float32) / n_splits)

    overall_auc = roc_auc_score(y_train, oof_preds)
    mean_fold_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    total_time = time.time() - start_time

    print("\n=================================================================")
    print(f"[+] TABULAR RESNET (STREAM D) 10-FOLD TRAINING COMPLETE!")
    print(f"    Overall OOF ROC-AUC: {overall_auc:.6f}")
    print(f"    Mean Fold ROC-AUC:   {mean_fold_auc:.6f} (+/- {std_auc:.6f})")
    print(f"    Total Runtime:       {total_time / 60:.2f} minutes")
    print("=================================================================")

    # 5. Persist Certified Artifacts Matching top1 Pipeline Contract
    # Target directory locations
    save_dirs = [model_dir]
    kaggle_working_models = Path("/kaggle/working/models/nn_tabular")
    if Path("/kaggle/working").exists():
        save_dirs.append(kaggle_working_models)

    for s_dir in save_dirs:
        s_dir.mkdir(parents=True, exist_ok=True)
        # OOF Predictions
        oof_df = pl.DataFrame({
            "id": range(len(oof_preds)),
            "oof_pred": oof_preds,
            "pred": oof_preds,
            "target": y_train,
            "Will_Buy_EV": oof_preds,
            "fold": folds,
        })
        oof_df.write_parquet(s_dir / "oof_preds.parquet", compression="zstd")

        # Test Predictions
        test_df_out = pl.DataFrame({
            "id": test_ids,
            "pred": test_preds,
            "Will_Buy_EV": test_preds,
        })
        test_df_out.write_parquet(s_dir / "test_preds.parquet", compression="zstd")

        # Metrics summary
        metrics_summary = {
            "model_name": "nn_tabular_resnet",
            "metric": "roc_auc",
            "overall_cv": float(overall_auc),
            "mean_fold_cv": float(mean_fold_auc),
            "std_cv": float(std_auc),
            "fold_scores": [float(s) for s in fold_aucs],
            "runtime_minutes": total_time / 60.0,
        }
        with open(s_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)

        # CSV Submission
        sub_csv = pd.DataFrame({"id": test_ids, "Will_Buy_EV": test_preds})
        sub_csv.to_csv(s_dir / "submission.csv", index=False)

    if Path("/kaggle/working").exists():
        pd.DataFrame({"id": test_ids, "Will_Buy_EV": test_preds}).to_csv(
            "/kaggle/working/submission_nn.csv", index=False
        )

    print(f"[+] Certified Neural Artifacts successfully persisted to: {model_dir}")
    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Stream D Tabular ResNet Runner")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to competition dataset")
    parser.add_argument("--output-dir", type=str, default=None, help="Artifact output directory")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of cross-validation folds")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=4096, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Peak learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="ResNet hidden dimension")
    parser.add_argument("--num-blocks", type=int, default=3, help="Number of residual blocks")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout probability")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_nn_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
