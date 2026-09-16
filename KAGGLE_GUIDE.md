# Kaggle Execution Guide: Strategic Breakthrough to 0.946+

**Competition:** Playground Series - Season 6, Episode 9 (`playground-series-s6e9`)  
**Target:** `Will_Buy_EV` (Binary Classification)  
**Evaluation Metric:** `ROC-AUC`  
**Current Baseline Score:** 0.94168  
**Target Score:** **0.9467+**

---

## The 3 Breakthrough Techniques Included in this Update

1. **Ground-Truth Physical Dataset Ingestion:**  
   The underlying real-world dataset (`itzzomkar/ev-adoption-behavior-and-range-anxiety`) with 10,000 samples is bundled directly in `data/original/` and injected strictly into each training fold.
2. **Bayesian Target Encoding with M-Estimate Smoothing ($m=25.0$):**  
   Encodes high-order consumer compound tuples (`City x Car`, `Car x Subsidy`, `City x Subsidy`, `City x Car x Subsidy`) inside the CV loop.
3. **Orthogonal Model Diversity:**  
   Ensembles **LightGBM + CatBoost + XGBoost + PyTorch Tabular Neural Network** (with learned Entity Embeddings and residual GELU layers).

---

## Kaggle Notebook Execution (Dual T4 GPU Recommended)

### Step 1: Pull Latest Repository & Enable GPU Acceleration
In your Kaggle notebook, run:
```python
%cd /kaggle/working/electric-vehicle
!git pull origin main

# Optional: Upgrade LightGBM for native CUDA GPU acceleration
!pip install -q lightgbm --upgrade
```

### Step 2: Immediate Recovery Submission (Zero-Wait)
If you want to submit right away, you have `submission_grandmaster_meta_blend.csv` and `submission_dual_rank.csv` already generated with zero ties and uncorrupted ranks:
```python
import pandas as pd
# Load the uncorrupted 100-model meta-blend directly from tracked parquet
df = pd.read_parquet("/kaggle/working/electric-vehicle/submission_grandmaster_meta_blend.parquet")
print(f"Shape: {df.shape} | Unique ranks: {df['Will_Buy_EV'].nunique()} | Ties: {len(df) - df['Will_Buy_EV'].nunique()}")
df.to_csv("/kaggle/working/submission.csv", index=False)
print("Ready to submit /kaggle/working/submission.csv!")
```

### Step 3: Launch Grandmaster Top-1 Pipeline (LGBM GPU + CatBoost GPU + XGBoost GPU)
To train the full multi-view orthogonal ensemble with 10 folds and 3 seeds across all three GPU architectures:
```bash
!python scripts/kaggle_train_grandmaster.py --model all --folds 10 --seeds 42 2024 777
```
*Key features:*
- **LightGBM:** Auto-probes `device='cuda'` -> `device='gpu'` -> multi-threaded CPU.
- **CatBoost:** Native GPU acceleration (`task_type="GPU"`).
- **XGBoost:** Native CUDA acceleration (`tree_method="hist"`, `device="cuda"`).
- **Blending:** Automatically runs Nelder-Mead on Probability, Rank, and Logit spaces across all single, dual, and tri-model combinations.
- **Submissions emitted:**
  1. `submission.csv` / `submission_zero_tie_champion.csv`: Best OOF Nelder-Mead blend with continuous zero-tie ranking.
  2. `submission_tri_rank.csv`: 33% LGBM + 33% XGB + 33% CatBoost zero-tie rank average.
  3. `submission_dual_rank.csv`: 50% LGBM + 50% XGBoost zero-tie rank average.
  4. `submission_pure_prob.csv`: Unperturbed Nelder-Mead probability blend.
  5. `submission_cat_pure.csv`, `submission_xgb_pure.csv`, `submission_lgb_pure.csv`: Single-model multi-seed baselines.

### Step 4: Package Artifacts (Optional)
To download all new models, OOFs, and test predictions:
```bash
!python scripts/package_kaggle_artifacts.py
```
