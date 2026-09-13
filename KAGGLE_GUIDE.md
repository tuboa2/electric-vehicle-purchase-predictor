# Kaggle Notebook & GitHub Deployment Guide

**Competition:** Playground Series - Season 6, Episode 9 (`playground-series-s6e9`)  
**Target:** `Will_Buy_EV` (Binary Classification)  
**Evaluation Metric:** `ROC-AUC`  

This guide provides step-by-step instructions for pushing the repository to GitHub and flawlessly pulling/running it in a Kaggle Notebook.

---

## 1. Step 1: Push Repository to GitHub

Ensure all files are committed to the `main` branch locally, then push to your GitHub remote repository:

```bash
# 1. Add your GitHub repository as remote origin (replace with your repo URL)
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/electric-vehicle.git

# 2. Verify git status
git status

# 3. Push to GitHub
git branch -M main
git push -u origin main
```

---

## 2. Step 2: Running Inside a Kaggle Notebook

Create a new Kaggle notebook on the competition page: [https://www.kaggle.com/competitions/playground-series-s6e9](https://www.kaggle.com/competitions/playground-series-s6e9).

### Method A: One-Liner Execution (Recommended)

In the first cell of your Kaggle notebook, run:

```bash
# 1. Clone your GitHub repository
!git clone https://github.com/<YOUR_GITHUB_USERNAME>/electric-vehicle.git

# 2. Change directory and install dependencies
%cd /kaggle/working/electric-vehicle
!pip install -r requirements.txt --quiet

# 3. Run the automated Kaggle training script
!python scripts/kaggle_train.py \
    --data-dir /kaggle/input/playground-series-s6e9 \
    --output-dir /kaggle/working
```

This single command will:
1. Auto-detect raw competition data from `/kaggle/input/playground-series-s6e9`.
2. Generate deterministic 5-fold `StratifiedKFold` splits (`seed=42`).
3. Train 5-fold LightGBM with early stopping.
4. Output out-of-fold and test predictions.
5. Generate and strictly validate `/kaggle/working/submission.csv` (286,571 rows, zero NaNs, correct header `id,Will_Buy_EV`).

---

### Method B: Interactive Jupyter Notebook Execution

You can also directly import and run [kaggle_pipeline.ipynb](file:///home/kazuha/Documents/competitions/electric-vehicle/kaggle_pipeline.ipynb):
1. On Kaggle, click **File -> Import Notebook** and upload `kaggle_pipeline.ipynb`.
2. Ensure the competition dataset `playground-series-s6e9` is attached under **Input**.
3. Run all cells.

---

## 3. Environment & Path Resolution

The repository includes [kaggle/paths.py](file:///home/kazuha/Documents/competitions/electric-vehicle/kaggle/paths.py) which automatically detects whether code is executing on Kaggle or locally:

| Resource | Kaggle Notebook Environment | Local Machine Environment |
| :--- | :--- | :--- |
| **Input Data** | `/kaggle/input/playground-series-s6e9` | `data/processed/playground-series-s6e9` or `data/raw/...` |
| **Output Submission** | `/kaggle/working/submission.csv` | `experiments/submissions/submission.csv` |
| **Artifacts** | `/kaggle/working/` | `experiments/artifacts/` |

---

## 4. Submission Verification Checklist (Gate 6)

Before submitting, the runner automatically asserts:
- [x] File exists at `/kaggle/working/submission.csv`
- [x] Exact row count matches test set ($286,571$ rows)
- [x] Header matches exact contract: `id,Will_Buy_EV`
- [x] Zero null or NaN values
- [x] Predictions strictly bounded in $[0.0, 1.0]$
- [x] ID ordering is strictly preserved ($668665 \to 955235$)
