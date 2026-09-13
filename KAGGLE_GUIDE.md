# Kaggle Execution Guide: Phase 8 Feature Engineering & Phase 9 Model Exploration

**Competition:** Playground Series - Season 6, Episode 9 (`playground-series-s6e9`)  
**Target:** `Will_Buy_EV` (Binary Classification)  
**Evaluation Metric:** `ROC-AUC`  
**Current Baseline Score:** 0.94168 (LightGBM Raw Features)  
**Target Score:** 0.9467+ (Ensemble of LightGBM + CatBoost + XGBoost with Domain Features)

---

## 1. Setup in Kaggle Notebook (Dual T4 GPU Recommended)

In the Kaggle Notebook right sidebar:
1. **Accelerator:** Set to **GPU T4 x2** (or GPU P100).
2. **Internet:** Turn **ON** (needed for `git pull` from GitHub).
3. **Data:** Ensure `playground-series-s6e9` is attached under **Input**.

In the first cell, pull the latest code:

```python
%cd /kaggle/working
!git clone https://github.com/tuboa2/electric-vehicle-purchase-predictor.git electric-vehicle 2>/dev/null || (cd electric-vehicle && git pull origin main)
%cd /kaggle/working/electric-vehicle
```

---

## 2. Phase 8 & 9: Train Diverse Models with Domain Features

Each command trains a 5-fold cross-validated model, automatically extracts the domain feature interactions (charging density, commute ratios, economic capacity, and subsidy gating), detects GPU acceleration, and saves predictions to `/kaggle/working/models/<model_name>/`.

### Model 1: LightGBM (Leaf-wise Tree Growth)
```bash
!python scripts/kaggle_train.py --model lgbm --features domain
```

### Model 2: CatBoost (Symmetric Oblivious Trees + GPU + Native Categoricals)
```bash
!python scripts/kaggle_train.py --model catboost --features domain
```

### Model 3: XGBoost (Histogram-based Depth-wise GBDT + CUDA)
```bash
!python scripts/kaggle_train.py --model xgboost --features domain
```

---

## 3. Phase 10: Run the Blending & Ensembling Engine

Once you have trained at least 2 models, run the automated blender:

```bash
!python scripts/kaggle_blend.py
```

### What the Blender Does:
1. Scans all candidate models in `/kaggle/working/models/`.
2. Computes the out-of-fold prediction correlation matrix.
3. Evaluates **Strategy A** (Percentile Rank Averaging) and **Strategy B** (Nelder-Mead Metric Optimization on ROC-AUC).
4. Selects the highest CV ensemble and validates Gate 5 ($\Delta \text{CV} > 0.0005$).
5. Automatically writes the final ensemble predictions directly to `/kaggle/working/submission.csv`!

---

## 4. Submitting to Kaggle

On the right panel under **Data $\to$ Output**:
- The file `/kaggle/working/submission.csv` is generated and certified (286,571 rows, 0 nulls, correct headers).
- Click the **Submit** button next to `submission.csv` to submit directly to the leaderboard!
