"""
KAMAS Baseline Tabular GBDT Training Template
Executes leak-free K-Fold training with out-of-fold and test predictions.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, root_mean_squared_error


def run_baseline(
    data_dir: str,
    output_dir: str,
    target_col: str,
    id_col: str,
    metric_name: str = "roc_auc",
    is_classification: bool = True,
    n_splits: int = 5,
    seed: int = 42,
) -> None:
    data_p = Path(data_dir)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    train_df = pl.read_parquet(data_p / "train.parquet").to_pandas()
    test_df = pl.read_parquet(data_p / "test.parquet").to_pandas()
    folds_df = pl.read_parquet(data_p / "folds.parquet").to_pandas()

    train_df = train_df.merge(folds_df, on=id_col, how="inner")

    features = [c for c in test_df.columns if c != id_col]

    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    fold_scores = []

    for fold in range(n_splits):
        tr = train_df[train_df["fold"] != fold]
        va = train_df[train_df["fold"] == fold]

        X_tr, y_tr = tr[features], tr[target_col]
        X_va, y_va = va[features], va[target_col]

        model_cls = lgb.LGBMClassifier if is_classification else lgb.LGBMRegressor
        model = model_cls(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
            n_jobs=-1,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        if is_classification:
            val_p = model.predict_proba(X_va)[:, 1] if len(model.classes_) == 2 else model.predict_proba(X_va)
            test_p = model.predict_proba(test_df[features])[:, 1] if len(model.classes_) == 2 else model.predict_proba(test_df[features])
        else:
            val_p = model.predict(X_va)
            test_p = model.predict(test_df[features])

        oof_preds[va.index] = val_p
        test_preds += test_p / n_splits

        score = roc_auc_score(y_va, val_p) if metric_name == "roc_auc" else root_mean_squared_error(y_va, val_p)
        fold_scores.append(float(score))

    overall_score = (
        roc_auc_score(train_df[target_col], oof_preds)
        if metric_name == "roc_auc"
        else root_mean_squared_error(train_df[target_col], oof_preds)
    )

    metrics = {
        "metric_name": metric_name,
        "overall_cv": float(overall_score),
        "fold_scores": fold_scores,
        "std_cv": float(np.std(fold_scores)),
    }

    with open(out_p / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    oof_out = pd.DataFrame({id_col: train_df[id_col], target_col: train_df[target_col], "pred": oof_preds})
    pl.from_pandas(oof_out).write_parquet(out_p / "oof_preds.parquet")

    test_out = pd.DataFrame({id_col: test_df[id_col], "pred": test_preds})
    pl.from_pandas(test_out).write_parquet(out_p / "test_preds.parquet")

    print(f"Training complete. Overall CV: {overall_score:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="experiments/artifacts/baseline")
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--metric", default="roc_auc")
    args = parser.parse_args()

    run_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_col=args.target_col,
        id_col=args.id_col,
        metric_name=args.metric,
    )
