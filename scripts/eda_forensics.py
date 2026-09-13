#!/usr/bin/env python3
"""
Data Forensics & EDA Script for playground-series-s6e9
Performs schema standardization into Parquet and deep forensic analysis.
"""

import json
from pathlib import Path
import numpy as np
import polars as pl
from scipy import stats

RAW_DIR = Path("data/raw/playground-series-s6e9")
PROCESSED_DIR = Path("data/processed/playground-series-s6e9")
ARTIFACTS_DIR = Path("experiments/artifacts")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("[1] Loading raw CSVs with Polars...")
    train_raw = pl.read_csv(RAW_DIR / "train.csv")
    test_raw = pl.read_csv(RAW_DIR / "test.csv")
    sample_sub_raw = pl.read_csv(RAW_DIR / "sample_submission.csv")

    print(f"Train raw shape: {train_raw.shape}")
    print(f"Test raw shape: {test_raw.shape}")
    print(f"Sample sub raw shape: {sample_sub_raw.shape}")

    # Standardize types and write to Parquet
    print("[2] Standardizing to Parquet...")
    train_parquet_path = PROCESSED_DIR / "train.parquet"
    test_parquet_path = PROCESSED_DIR / "test.parquet"
    sample_parquet_path = PROCESSED_DIR / "sample_submission.parquet"
    sample_csv_path = PROCESSED_DIR / "sample_submission.csv"

    train_raw.write_parquet(train_parquet_path, compression="zstd")
    test_raw.write_parquet(test_parquet_path, compression="zstd")
    sample_sub_raw.write_parquet(sample_parquet_path, compression="zstd")
    sample_sub_raw.write_csv(sample_csv_path)

    print(f"[+] Saved train to {train_parquet_path} ({train_parquet_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"[+] Saved test to {test_parquet_path} ({test_parquet_path.stat().st_size / (1024*1024):.2f} MB)")

    # Forensic Analysis
    print("[3] Conducting Forensic Analysis...")
    feature_cols = [c for c in test_raw.columns if c != "id"]
    id_col = "id"
    target_col = "Will_Buy_EV"

    # 1. Missingness Audit
    missing_report = {}
    for col in train_raw.columns:
        tr_null = train_raw[col].null_count()
        te_null = test_raw[col].null_count() if col in test_raw.columns else None
        missing_report[col] = {
            "train_missing_count": tr_null,
            "train_missing_pct": float(tr_null / train_raw.height * 100),
            "test_missing_count": te_null,
            "test_missing_pct": float(te_null / test_raw.height * 100) if te_null is not None else None,
        }

    # 2. Categorical Profiling
    cat_cols = [c for c in feature_cols if train_raw[c].dtype == pl.String or train_raw[c].dtype == pl.Categorical]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    print(f"Identified {len(cat_cols)} categorical features: {cat_cols}")
    print(f"Identified {len(num_cols)} numerical features: {num_cols}")

    categorical_report = {}
    for col in cat_cols:
        tr_vc = train_raw[col].value_counts()
        te_vc = test_raw[col].value_counts()

        tr_dict = {row[col]: row["count"] for row in tr_vc.iter_rows(named=True)}
        te_dict = {row[col]: row["count"] for row in te_vc.iter_rows(named=True)}

        tr_cats = set(tr_dict.keys())
        te_cats = set(te_dict.keys())
        unseen_in_test = list(te_cats - tr_cats)
        missing_in_test = list(tr_cats - te_cats)

        # Compute category distribution percentages in train
        tr_dist = {k: round(v / train_raw.height, 5) for k, v in tr_dict.items()}
        te_dist = {k: round(v / test_raw.height, 5) for k, v in te_dict.items()}

        categorical_report[col] = {
            "cardinality_train": len(tr_cats),
            "cardinality_test": len(te_cats),
            "unseen_in_test": unseen_in_test,
            "missing_in_test": missing_in_test,
            "train_frequencies": tr_dict,
            "train_distribution": tr_dist,
            "test_distribution": te_dist,
            "mode": max(tr_dict, key=tr_dict.get) if tr_dict else None,
        }

    # 3. Numerical Profiling
    numerical_report = {}
    for col in num_cols:
        tr_series = train_raw[col].drop_nulls().to_numpy()
        te_series = test_raw[col].drop_nulls().to_numpy()

        numerical_report[col] = {
            "train": {
                "mean": float(np.mean(tr_series)),
                "std": float(np.std(tr_series)),
                "min": float(np.min(tr_series)),
                "p25": float(np.percentile(tr_series, 25)),
                "median": float(np.median(tr_series)),
                "p75": float(np.percentile(tr_series, 75)),
                "max": float(np.max(tr_series)),
                "skewness": float(stats.skew(tr_series)),
                "kurtosis": float(stats.kurtosis(tr_series)),
            },
            "test": {
                "mean": float(np.mean(te_series)),
                "std": float(np.std(te_series)),
                "min": float(np.min(te_series)),
                "p25": float(np.percentile(te_series, 25)),
                "median": float(np.median(te_series)),
                "p75": float(np.percentile(te_series, 75)),
                "max": float(np.max(te_series)),
                "skewness": float(stats.skew(te_series)),
                "kurtosis": float(stats.kurtosis(te_series)),
            },
            "drift_mean_diff_pct": float(abs(np.mean(tr_series) - np.mean(te_series)) / (np.std(tr_series) + 1e-9) * 100),
        }

    # 4. Target Variable Distribution
    y_raw = train_raw[target_col]
    target_counts = {row[target_col]: row["count"] for row in y_raw.value_counts().iter_rows(named=True)}
    n_pos = target_counts.get("Yes", 0)
    n_neg = target_counts.get("No", 0)
    pos_rate = n_pos / train_raw.height
    neg_rate = n_neg / train_raw.height
    imbalance_ratio = n_neg / (n_pos if n_pos > 0 else 1)

    # Numerical target for correlation check
    y_num = (train_raw[target_col] == "Yes").to_numpy().astype(int)
    target_report = {
        "column_name": target_col,
        "counts": target_counts,
        "positive_rate": float(pos_rate),
        "negative_rate": float(neg_rate),
        "imbalance_ratio": float(imbalance_ratio),
        "skewness": float(stats.skew(y_num)),
        "entropy": float(stats.entropy([pos_rate, neg_rate])),
    }

    # 5. Identifier & Ordering Analysis
    train_ids = train_raw[id_col].to_numpy()
    test_ids = test_raw[id_col].to_numpy()

    is_tr_id_monotonic = bool(np.all(np.diff(train_ids) > 0))
    is_te_id_monotonic = bool(np.all(np.diff(test_ids) > 0))
    id_overlap = len(set(train_ids).intersection(set(test_ids)))

    # Correlation between id and target
    id_corr = float(np.corrcoef(train_ids, y_num)[0, 1])

    id_analysis = {
        "id_column": id_col,
        "train_id_min": int(train_ids[0]),
        "train_id_max": int(train_ids[-1]),
        "test_id_min": int(test_ids[0]),
        "test_id_max": int(test_ids[-1]),
        "is_train_id_strictly_monotonic": is_tr_id_monotonic,
        "is_test_id_strictly_monotonic": is_te_id_monotonic,
        "id_overlap_count": id_overlap,
        "train_test_split_boundary": {
            "last_train_id": int(train_ids[-1]),
            "first_test_id": int(test_ids[0]),
            "contiguous_sequence": bool(test_ids[0] == train_ids[-1] + 1),
        },
        "id_target_linear_correlation": id_corr,
    }

    # 6. Target Correlations with numerical features
    feature_correlations = {}
    for col in num_cols:
        col_vals = train_raw[col].to_numpy()
        corr = float(np.corrcoef(col_vals, y_num)[0, 1])
        feature_correlations[col] = corr

    # Categorical association with target (cramers_v / empirical risk)
    cat_target_associations = {}
    for col in cat_cols:
        agg = train_raw.group_by(col).agg([
            pl.len().alias("count"),
            (pl.col(target_col) == "Yes").mean().alias("target_mean")
        ])
        cat_target_associations[col] = {
            row[col]: {
                "count": row["count"],
                "positive_rate": round(float(row["target_mean"]), 4)
            }
            for row in agg.iter_rows(named=True)
        }

    # Synthesize Complete Report
    eda_report = {
        "competition_id": "playground-series-s6e9",
        "forensics_timestamp": "2026-09-13T04:59:00Z",
        "dataset_dimensions": {
            "train": {"rows": train_raw.height, "cols": train_raw.width},
            "test": {"rows": test_raw.height, "cols": test_raw.width},
            "sample_submission": {"rows": sample_sub_raw.height, "cols": sample_sub_raw.width},
            "feature_count": len(feature_cols),
        },
        "missingness": missing_report,
        "categorical_profiles": categorical_report,
        "numerical_profiles": numerical_report,
        "target_profile": target_report,
        "identifier_analysis": id_analysis,
        "linear_correlations_with_target": feature_correlations,
        "categorical_target_associations": cat_target_associations,
    }

    out_path = ARTIFACTS_DIR / "eda_report.json"
    with open(out_path, "w") as f:
        json.dump(eda_report, f, indent=2)

    print(f"[+] Forensic EDA report written successfully to {out_path}")

if __name__ == "__main__":
    main()
