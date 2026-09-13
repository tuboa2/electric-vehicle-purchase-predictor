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

### Step 1: Pull the Latest Repository
```python
%cd /kaggle/working/electric-vehicle
!git pull origin main
```

### Step 2: Train Model 1 (LightGBM + Original Dataset + Target Encoding)
```bash
!python scripts/kaggle_train.py --model lgbm --features domain --use-original
```

### Step 3: Train Model 2 (CatBoost GPU + Original Dataset)
```bash
!python scripts/kaggle_train.py --model catboost --features domain --use-original
```

### Step 4: Train Model 3 (XGBoost CUDA + Original Dataset)
```bash
!python scripts/kaggle_train.py --model xgboost --features domain --use-original
```

### Step 5: Train Model 4 (PyTorch Tabular Neural Network on GPU)
```bash
!python scripts/kaggle_train_nn.py --epochs 12 --batch-size 2048
```

### Step 6: Execute the Multi-Model Blender
```bash
!python scripts/kaggle_blend.py
```

### Step 7: Submit to Kaggle
- The ensemble blender automatically writes the verified final predictions to `/kaggle/working/submission.csv`.
- On the right panel under **Data $\to$ Output**, click **Submit** next to `submission.csv`.
