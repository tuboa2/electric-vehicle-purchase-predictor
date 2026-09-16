"""
Kaggle Dual-Stream Champion Blender
Fuses Free-Tree (LB 0.94634) and Base-Margin Residual (CV 0.946312) in Logit Space
with Calibrated Micro-Jitter Zero-Tie Resolution.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import polars as pl
from scipy.special import expit, logit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compute_recipe_score(df: pd.DataFrame) -> np.ndarray:
    """Computes the ground-truth linear utility score for continuous tie-breaking."""
    inc = df["Annual_Income_USD"].astype(float).values
    env = df["Environmental_Concern_Level"].astype(float).values if "Environmental_Concern_Level" in df.columns else 3.0
    sub = (df["Subsidy_Available"].astype(str) == "Yes").astype(float).values if "Subsidy_Available" in df.columns else 0.0
    anx = df["Range_Anxiety_Level"].astype(str).values if "Range_Anxiety_Level" in df.columns else "Low"
    anx_med = (anx == "Medium").astype(float)
    anx_high = (anx == "High").astype(float)

    score = (
        1.2 * (inc / 100000.0)
        + 0.6 * env
        + 2.0 * sub
        - 1.0 * anx_med
        - 3.0 * anx_high
    )
    return score


def main():
    parser = argparse.ArgumentParser(description="Dual-Stream Logit Blender")
    parser.add_argument("--sub-a", type=str, default="submission (3).csv", help="Stream A submission (Free-Tree, LB 0.94634)")
    parser.add_argument("--sub-b", type=str, default="submission_micro_zero_tie.csv", help="Stream B submission (Base-Margin)")
    parser.add_argument("--weight-a", type=float, default=0.50, help="Weight for Stream A in logit space")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate files
    cand_a = [Path(args.sub_a), PROJECT_ROOT / args.sub_a, Path(f"/kaggle/working/{args.sub_a}")]
    cand_b = [Path(args.sub_b), PROJECT_ROOT / args.sub_b, Path(f"/kaggle/working/{args.sub_b}")]

    path_a = next((p for p in cand_a if p.exists()), None)
    path_b = next((p for p in cand_b if p.exists()), None)

    if not path_a or not path_b:
        print(f"[!] Error: Could not locate submissions:\n  A: {path_a} (from {cand_a})\n  B: {path_b} (from {cand_b})")
        sys.exit(1)

    print(f"[+] Loading Stream A (Free-Tree):   {path_a}")
    df_a = pd.read_csv(path_a) if str(path_a).endswith(".csv") else pl.read_parquet(path_a).to_pandas()
    print(f"[+] Loading Stream B (Base-Margin): {path_b}")
    df_b = pd.read_csv(path_b) if str(path_b).endswith(".csv") else pl.read_parquet(path_b).to_pandas()

    target_col = "Will_Buy_EV"
    pred_col_a = target_col if target_col in df_a.columns else [c for c in df_a.columns if c != "id"][0]
    pred_col_b = target_col if target_col in df_b.columns else [c for c in df_b.columns if c != "id"][0]

    p_a = df_a[pred_col_a].values
    p_b = df_b[pred_col_b].values
    test_ids = df_a["id"]

    corr = np.corrcoef(p_a, p_b)[0, 1]
    print(f"[+] Pairwise Correlation between Stream A and Stream B: {corr:.6f}")

    # Clip to prevent numerical divergence
    eps = 1e-7
    p_a_c = np.clip(p_a, eps, 1.0 - eps)
    p_b_c = np.clip(p_b, eps, 1.0 - eps)

    w_a = args.weight_a
    w_b = 1.0 - w_a
    print(f"[*] Blending in Logit Space: {w_a:.2f} * logit(A) + {w_b:.2f} * logit(B)...")
    logit_blend = w_a * logit(p_a_c) + w_b * logit(p_b_c)
    p_blend = expit(logit_blend)

    # Secondary score for continuous tie breaking
    secondary_score = None
    data_cand = [
        Path("/kaggle/input/competitions/playground-series-s6e9/test.parquet"),
        Path("/kaggle/input/competitions/playground-series-s6e9/test.csv"),
        PROJECT_ROOT / "data" / "processed" / "playground-series-s6e9" / "test.parquet",
    ]
    for dc in data_cand:
        if dc.exists():
            test_raw = pl.read_parquet(dc).to_pandas() if str(dc).endswith(".parquet") else pd.read_csv(dc)
            secondary_score = compute_recipe_score(test_raw)
            print(f"[+] Secondary recipe score computed from {dc} for continuous tie breaking.")
            break

    if secondary_score is not None:
        sec_norm = (secondary_score - np.nanmean(secondary_score)) / (np.nanstd(secondary_score) + 1e-7)
        micro_zero_tie = p_blend + 1e-9 * sec_norm
    else:
        micro_zero_tie = p_blend

    sub_df = pd.DataFrame({"id": test_ids, "Will_Buy_EV": micro_zero_tie})

    out_csv = out_dir / "submission_dual_stream_champion.csv"
    out_parquet = out_dir / "submission_dual_stream_champion.parquet"
    sub_df.to_csv(out_csv, index=False)
    sub_df.to_parquet(out_parquet, index=False)

    if Path("/kaggle/working").exists():
        sub_df.to_csv("/kaggle/working/submission.csv", index=False)
        print("[+] Also updated /kaggle/working/submission.csv for direct submit!")

    print(f"\n=================================================================")
    print(f"[+] DUAL-STREAM CHAMPION BLEND READY!")
    print(f"    CSV:     {out_csv}")
    print(f"    Parquet: {out_parquet}")
    print(f"    Shape:   {sub_df.shape}")
    print(f"    Mean:    {sub_df['Will_Buy_EV'].mean():.6f} (Ground-truth target prevalence: ~0.1748)")
    print(f"    Std:     {sub_df['Will_Buy_EV'].std():.6f}")
    print(f"    Min/Max: {sub_df['Will_Buy_EV'].min():.6f} / {sub_df['Will_Buy_EV'].max():.6f}")
    print(f"=================================================================")


if __name__ == "__main__":
    main()
