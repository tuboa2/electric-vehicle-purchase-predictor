"""
Kaggle Comprehensive Diagnostics & Audit Suite
Audits all models, OOF predictions, submissions, correlations, distributions,
residual errors, and environment metrics into a single forensic dossier.
"""

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import kurtosis, skew
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def format_bytes(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def run_diagnostics():
    start_time = time.time()
    print("=" * 80)
    print("       KAMAS COMPREHENSIVE KAGGLE PROGRESS & FORENSICS AUDITOR")
    print("=" * 80)

    working_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else PROJECT_ROOT
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "environment": {},
        "submissions": {},
        "models_oof": {},
        "correlations": {},
        "hard_sample_analysis": {},
    }

    # 1. Environment & Hardware
    import platform
    report["environment"]["os"] = platform.platform()
    report["environment"]["python_version"] = sys.version.split()[0]
    try:
        import torch
        report["environment"]["torch_version"] = torch.__version__
        report["environment"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["environment"]["gpu_count"] = torch.cuda.device_count()
            report["environment"]["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as e:
        report["environment"]["torch_error"] = str(e)

    for pkg in ["lightgbm", "xgboost", "catboost", "polars", "sklearn"]:
        try:
            mod = __import__(pkg)
            report["environment"][f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:
            report["environment"][f"{pkg}_version"] = "not_installed"

    print("\n--- 1. SYSTEM & ENVIRONMENT ---")
    print(f"Python: {report['environment']['python_version']} | OS: {report['environment']['os']}")
    print(f"CUDA Available: {report['environment'].get('cuda_available', False)}")
    if report["environment"].get("gpu_names"):
        print(f"GPUs Detected: {', '.join(report['environment']['gpu_names'])}")
    for k, v in report["environment"].items():
        if k.endswith("_version") and k != "python_version":
            print(f"  - {k:<20s}: {v}")

    # 2. Inventory of Submission Files
    print("\n--- 2. SUBMISSION AUDIT & DISTRIBUTION HEALTH ---")
    candidate_sub_files = []

    search_dirs = [working_dir, working_dir / "ensemble_grandmaster", working_dir / "ensemble_sota"]
    models_dir = working_dir / "models"
    if models_dir.exists():
        for d in models_dir.iterdir():
            if d.is_dir():
                search_dirs.append(d)

    for s_dir in search_dirs:
        if s_dir.exists():
            for f in s_dir.iterdir():
                if f.is_file() and (f.suffix == ".csv" or f.suffix == ".parquet"):
                    if any(kw in f.name.lower() for kw in ["sub", "test_pred"]):
                        candidate_sub_files.append(f)

    # De-duplicate
    seen_paths = set()
    sub_files = []
    for f in candidate_sub_files:
        res = f.resolve()
        if res not in seen_paths:
            seen_paths.add(res)
            sub_files.append(f)

    sub_data = {}
    print(f"{'File Name':<35s} {'Rows':<8s} {'Nulls':<6s} {'Mean':<8s} {'Std':<8s} {'Ties':<8s} {'Min':<8s} {'Max':<8s}")
    print("-" * 95)

    for f in sorted(sub_files, key=lambda x: x.name):
        try:
            df = pl.read_parquet(f).to_pandas() if f.suffix == ".parquet" else pd.read_csv(f)
            if len(df) != 286571:
                continue

            target_col = [c for c in df.columns if c.lower() != "id"][-1]
            vals = df[target_col].values.astype(np.float64)
            sub_data[f.name] = vals

            n_null = int(np.isnan(vals).sum())
            n_unique = int(len(np.unique(vals)))
            n_ties = len(vals) - n_unique
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))

            print(f"{f.name:<35s} {len(vals):<8d} {n_null:<6d} {mean_v:<8.4f} {std_v:<8.4f} {n_ties:<8d} {min_v:<8.2e} {max_v:<8.4f}")

            report["submissions"][f.name] = {
                "path": str(f),
                "size_bytes": f.stat().st_size,
                "size_formatted": format_bytes(f.stat().st_size),
                "rows": len(vals),
                "null_count": n_null,
                "unique_predictions": n_unique,
                "ties_count": n_ties,
                "zero_ties_guaranteed": (n_ties == 0),
                "mean": mean_v,
                "std": std_v,
                "min": min_v,
                "max": max_v,
                "quantiles": {
                    "p01": float(np.percentile(vals, 1)),
                    "p05": float(np.percentile(vals, 5)),
                    "p25": float(np.percentile(vals, 25)),
                    "p50": float(np.percentile(vals, 50)),
                    "p75": float(np.percentile(vals, 75)),
                    "p95": float(np.percentile(vals, 95)),
                    "p99": float(np.percentile(vals, 99)),
                },
                "skewness": float(skew(vals)),
                "kurtosis": float(kurtosis(vals)),
            }
        except Exception as e:
            report["submissions"][f.name] = {"error": str(e)}

    # 3. Correlation Matrix across submissions
    print("\n--- 3. PAIRWISE CORRELATION MATRIX (PEARSON & SPEARMAN) ---")
    if len(sub_data) >= 2:
        df_all_subs = pd.DataFrame(sub_data)
        pearson_corr = df_all_subs.corr(method="pearson").round(6).to_dict()
        spearman_corr = df_all_subs.corr(method="spearman").round(6).to_dict()
        report["correlations"]["pearson"] = pearson_corr
        report["correlations"]["spearman"] = spearman_corr

        top_pairs = []
        names = list(sub_data.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                p_c = pearson_corr[n1][n2]
                s_c = spearman_corr[n1][n2]
                top_pairs.append((n1, n2, p_c, s_c))

        # Sort by lowest correlation (highest diversity)
        top_pairs.sort(key=lambda x: x[2])
        print("Most Diverse (Lowest Correlation) Pairs:")
        for n1, n2, pc, sc in top_pairs[:8]:
            print(f"  - {n1} vs {n2}: Pearson={pc:.6f}, Spearman={sc:.6f}")

    # 4. Model OOF Performance & Metrics Check
    print("\n--- 4. CROSS-VALIDATION GROUND TRUTH METRICS ---")
    oof_files = []
    if models_dir.exists():
        for d in models_dir.iterdir():
            if d.is_dir():
                oof_f = d / "oof_preds.parquet"
                if not oof_f.exists():
                    oof_f = d / "oof_preds.csv"
                if oof_f.exists():
                    oof_files.append((d.name, oof_f))

    ens_dir = working_dir / "ensemble_grandmaster"
    if (ens_dir / "oof_preds.parquet").exists():
        oof_files.append(("ensemble_grandmaster", ens_dir / "oof_preds.parquet"))

    y_true = None
    y_raw_df = None

    for m_name, o_file in oof_files:
        try:
            df_o = pl.read_parquet(o_file).to_pandas() if o_file.suffix == ".parquet" else pd.read_csv(o_file)
            pred_col = next((c for c in ["oof_pred", "pred", "probability"] if c in df_o.columns), df_o.columns[1])
            y_col = next((c for c in ["target", "Will_Buy_EV", "label"] if c in df_o.columns), None)

            if y_true is None and y_col is not None:
                raw_y = df_o[y_col]
                if not pd.api.types.is_numeric_dtype(raw_y):
                    y_true = (raw_y.astype(str).str.strip().str.lower() == "yes").astype(float).values
                else:
                    y_true = raw_y.astype(float).values
                y_raw_df = df_o

            preds = df_o[pred_col].values.astype(np.float64)

            auc_score = float(roc_auc_score(y_true, preds)) if y_true is not None else 0.0
            brier = float(brier_score_loss(y_true, np.clip(preds, 0.0, 1.0))) if y_true is not None else 0.0
            logloss = float(log_loss(y_true, np.clip(preds, 1e-7, 1 - 1e-7))) if y_true is not None else 0.0

            print(f"Model: {m_name:<25s} | OOF AUC: {auc_score:.6f} | LogLoss: {logloss:.6f} | Brier: {brier:.6f} | Mean Pred: {np.mean(preds):.4f}")

            report["models_oof"][m_name] = {
                "file": str(o_file),
                "samples": len(preds),
                "oof_auc": auc_score,
                "log_loss": logloss,
                "brier_score": brier,
                "mean": float(np.mean(preds)),
                "std": float(np.std(preds)),
            }
        except Exception as e:
            report["models_oof"][m_name] = {"error": str(e)}

    # 5. Error & Residual Forensics
    print("\n--- 5. ERROR RESIDUAL & FAILURE MODE FORENSICS ---")
    if y_true is not None and y_raw_df is not None:
        try:
            # Let's inspect worst false positives and false negatives on the highest AUC model
            best_model_name = max(
                (k for k in report["models_oof"] if "oof_auc" in report["models_oof"][k]),
                key=lambda k: report["models_oof"][k]["oof_auc"],
            )
            df_best = pl.read_parquet(report["models_oof"][best_model_name]["file"]).to_pandas()
            best_preds = df_best[next(c for c in ["oof_pred", "pred"] if c in df_best.columns)].values

            residuals = best_preds - y_true
            abs_errors = np.abs(residuals)

            report["hard_sample_analysis"]["best_model_used"] = best_model_name
            report["hard_sample_analysis"]["mean_absolute_error"] = float(np.mean(abs_errors))
            report["hard_sample_analysis"]["worst_fp_threshold"] = float(np.percentile(best_preds[y_true == 0], 99.9))
            report["hard_sample_analysis"]["worst_fn_threshold"] = float(np.percentile(best_preds[y_true == 1], 0.1))

            print(f"Selected Benchmark Model for Error Analysis: {best_model_name}")
            print(f"  Mean Absolute Error: {np.mean(abs_errors):.6f}")
            print(f"  Worst False Positives (Label=0, Pred > {report['hard_sample_analysis']['worst_fp_threshold']:.4f}): {np.sum((y_true == 0) & (best_preds > 0.80))} samples")
            print(f"  Worst False Negatives (Label=1, Pred < {report['hard_sample_analysis']['worst_fn_threshold']:.4f}): {np.sum((y_true == 1) & (best_preds < 0.20))} samples")
        except Exception as e:
            report["hard_sample_analysis"]["error"] = str(e)

    # 6. Save JSON Report and Package Zip Archive
    report_json_path = working_dir / "kaggle_diagnostic_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Diagnostic report written to: {report_json_path}")

    # Create downloadable zip
    pack_dir = working_dir / "diagnostic_pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(report_json_path, pack_dir / "kaggle_diagnostic_report.json")

    # Copy top 4 submissions to pack
    for sub_name, s_info in list(report["submissions"].items())[:6]:
        if "path" in s_info:
            shutil.copy(s_info["path"], pack_dir / sub_name)

    zip_out = shutil.make_archive(str(working_dir / "kaggle_diagnostics_pack"), "zip", pack_dir)
    print(f"[+] Full diagnostics pack zipped and ready for download: {zip_out}")

    print("\n" + "=" * 80)
    print(f"[+] AUDIT COMPLETE in {time.time() - start_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_diagnostics()
