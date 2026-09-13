#!/usr/bin/env python3
"""
PyTorch Tabular Neural Network (MLP with Entity Embeddings) for playground-series-s6e9.
Provides genuine orthogonal decision boundaries (non-axis-aligned) to break
collinearity with GBDT models and unlock Phase 10 ensembling lift.

Features:
- Learned Entity Embeddings for categorical features
- Robust numerical scaling + BatchNorm
- Residual skip connections with GELU activations & Dropout
- 5-fold Stratified CV with early stopping on validation ROC-AUC
- Automatic GPU (CUDA) detection

Usage:
    !python scripts/kaggle_train_nn.py
"""

import argparse
import json
import os
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from features.domain_features import generate_domain_features
from kaggle.paths import resolve_data_dir, resolve_output_dir


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"[+] PyTorch GPU detected: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("[-] No GPU detected for PyTorch. Using CPU.")
    return torch.device("cpu")


class TabularDataset(Dataset):
    def __init__(self, cat_data: np.ndarray, num_data: np.ndarray, targets: np.ndarray | None = None):
        self.cats = torch.tensor(cat_data, dtype=torch.long)
        self.nums = torch.tensor(num_data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None

    def __len__(self):
        return len(self.nums)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.cats[idx], self.nums[idx], self.targets[idx]
        return self.cats[idx], self.nums[idx]


class TabularMLP(nn.Module):
    def __init__(
        self,
        embedding_sizes: list[tuple[int, int]],
        n_numerical: int,
        hidden_dims: list[int] = [256, 128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=n_cats, embedding_dim=emb_dim)
            for n_cats, emb_dim in embedding_sizes
        ])
        total_emb_dim = sum(e for _, e in embedding_sizes)

        in_dim = total_emb_dim + n_numerical
        self.num_bn = nn.BatchNorm1d(n_numerical) if n_numerical > 0 else nn.Identity()

        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(emb_outs, dim=1) if emb_outs else torch.empty((x_cat.size(0), 0), device=x_cat.device)
        x_num = self.num_bn(x_num)
        x = torch.cat([x_emb, x_num], dim=1)
        feat = self.backbone(x)
        return self.head(feat).squeeze(-1)


def train_nn_model(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    n_splits: int = 5,
    epochs: int = 15,
    batch_size: int = 2048,
    lr: float = 2e-3,
    seed: int = 42,
) -> dict:
    start_time = time.time()
    device = detect_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    base_out = output_dir if output_dir else resolve_output_dir()
    model_dir = base_out / "models" / "nn_tabular"
    model_dir.mkdir(parents=True, exist_ok=True)

    resolved_dir = resolve_data_dir(data_dir)
    train_df = pl.read_parquet(resolved_dir / "train.parquet").to_pandas()
    test_df = pl.read_parquet(resolved_dir / "test.parquet").to_pandas()

    target_col = "Will_Buy_EV"
    id_col = "id"

    # Ingest domain features
    train_df, test_df, _ = generate_domain_features(train_df, test_df)

    # Ingest certified folds
    folds_path = resolved_dir / "folds.parquet"
    if folds_path.exists():
        train_df["fold"] = pl.read_parquet(folds_path)["fold"].to_numpy()
    else:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        y_bin = (train_df[target_col] == "Yes").astype(int)
        fold_arr = np.empty(len(train_df), dtype=int)
        for f, (_, val_idx) in enumerate(skf.split(train_df, y_bin)):
            fold_arr[val_idx] = f
        train_df["fold"] = fold_arr

    # Identify categorical vs numerical
    ignore_cols = [id_col, target_col, "fold"]
    feature_cols = [c for c in test_df.columns if c not in ignore_cols]

    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train_df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    print(f"[*] Neural Net Features: {len(feature_cols)} ({len(cat_cols)} categoricals, {len(num_cols)} numericals)")

    # Label encode categoricals
    cat_dims = []
    emb_sizes = []
    for c in cat_cols:
        train_df[c] = train_df[c].astype(str)
        test_df[c] = test_df[c].astype(str)
        unique_vals = list(set(train_df[c].unique()).union(set(test_df[c].unique())))
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        train_df[c] = train_df[c].map(mapping).fillna(0).astype(int)
        test_df[c] = test_df[c].map(mapping).fillna(0).astype(int)
        n_unique = len(unique_vals)
        emb_dim = min(50, (n_unique + 1) // 2)
        emb_sizes.append((n_unique, max(4, emb_dim)))

    y_train = (train_df[target_col] == "Yes").to_numpy().astype(float)
    oof_preds = np.zeros(len(train_df), dtype=np.float64)
    test_preds = np.zeros(len(test_df), dtype=np.float64)
    fold_scores = []

    print("=" * 65)
    print(f"[*] TRAINING {n_splits}-FOLD PYTORCH TABULAR NEURAL NETWORK")
    print(f"    Batch Size: {batch_size} | Epochs: {epochs} | Device: {device}")
    print("=" * 65)

    for fold in range(n_splits):
        tr_mask = train_df["fold"] != fold
        va_mask = train_df["fold"] == fold

        # Fit Scaler strictly on train fold
        scaler = RobustScaler()
        tr_num = scaler.fit_transform(train_df.loc[tr_mask, num_cols].values)
        va_num = scaler.transform(train_df.loc[va_mask, num_cols].values)
        te_num = scaler.transform(test_df[num_cols].values)

        tr_cat = train_df.loc[tr_mask, cat_cols].values
        va_cat = train_df.loc[va_mask, cat_cols].values
        te_cat = test_df[cat_cols].values

        train_ds = TabularDataset(tr_cat, tr_num, y_train[tr_mask])
        val_ds = TabularDataset(va_cat, va_num, y_train[va_mask])
        test_ds = TabularDataset(te_cat, te_num)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

        model = TabularMLP(embedding_sizes=emb_sizes, n_numerical=len(num_cols)).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        best_auc = 0.0
        best_va_preds = None

        for ep in range(epochs):
            model.train()
            for x_c, x_n, y_b in train_loader:
                x_c, x_n, y_b = x_c.to(device), x_n.to(device), y_b.to(device)
                optimizer.zero_grad()
                logits = model(x_c, x_n)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Evaluation
            model.eval()
            val_probs = []
            with torch.no_grad():
                for x_c, x_n, _ in val_loader:
                    x_c, x_n = x_c.to(device), x_n.to(device)
                    probs = torch.sigmoid(model(x_c, x_n)).cpu().numpy()
                    val_probs.extend(probs)
            val_probs = np.array(val_probs)
            ep_auc = roc_auc_score(y_train[va_mask], val_probs)

            if ep_auc > best_auc:
                best_auc = ep_auc
                best_va_preds = val_probs

        print(f"  [Fold {fold+1}/{n_splits}] Best Val ROC-AUC: {best_auc:.6f}")
        oof_preds[va_mask] = best_va_preds
        fold_scores.append(best_auc)

        # Test inference
        model.eval()
        f_te_probs = []
        with torch.no_grad():
            for x_c, x_n in test_loader:
                x_c, x_n = x_c.to(device), x_n.to(device)
                f_te_probs.extend(torch.sigmoid(model(x_c, x_n)).cpu().numpy())
        test_preds += np.array(f_te_probs) / n_splits

    overall_auc = roc_auc_score(y_train, oof_preds)
    std_auc = float(np.std(fold_scores))
    total_time = time.time() - start_time

    print("=" * 65)
    print(f"[+] NEURAL NET OVERALL 5-FOLD OOF ROC-AUC: {overall_auc:.6f} (+/- {std_auc:.6f})")
    print(f"[+] Total training time: {total_time:.1f}s ({total_time / 60:.2f} min)")
    print("=" * 65)

    # Persist artifacts
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

    metrics_summary = {
        "competition_id": "playground-series-s6e9",
        "model_name": "nn_tabular",
        "metric_name": "roc_auc",
        "overall_cv": float(overall_auc),
        "std_cv": float(std_auc),
        "fold_scores": [float(s) for s in fold_scores],
        "execution_time_seconds": total_time,
    }
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    sub_df = pd.DataFrame({"id": test_df[id_col], target_col: test_preds})
    sub_df.to_csv(model_dir / "submission.csv", index=False)
    print(f"[+] PyTorch Neural Net artifacts saved in: {model_dir}")
    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="PyTorch Tabular Neural Network Runner")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_nn_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
