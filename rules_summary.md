# Kaggle Competition Rules & Constraints Summary

**Competition ID:** `playground-series-s6e9`  
**Title:** Playground Series - Season 6, Episode 9  
**Domain:** EV Adoption Behavior & Range Anxiety  
**Evaluator:** Scout Agent (`scout`)  
**Certification Date:** 2026-09-13  

---

## 1. Submission & Account Rules
- **Account Policy:** Strictly one account per participant. No multiple accounts.
- **Team Size:** Maximum 3 members. Team mergers permitted before deadline.
- **Daily Submission Limit:** 10 submissions per calendar day (UTC reset).
- **Final Submissions:** Exactly 2 final submissions selectable for Private Leaderboard ranking.
- **Timeline:**
  - Start Date: September 1, 2026
  - Final Submission Deadline: September 30, 2026 at 23:59 UTC

---

## 2. Submission Format & File Contract
- **Format:** CSV file (`submission.csv` or zipped CSV).
- **Header:** `id,Will_Buy_EV`
- **Rows:** Exactly 286,571 test prediction rows (+1 header row).
- **Prediction Values:** Float probability values in range $[0.0, 1.0]$. No NaNs, nulls, or infinite values permitted.
- **Competition Execution Mode:** Standard CSV submission (not a mandated code competition).

---

## 3. Official Evaluation Metric
- **Metric:** **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve).
- **Optimization Direction:** Maximize ($\uparrow$).
- **Implementation:** [evaluation/metrics.py](file:///home/kazuha/Documents/competitions/electric-vehicle/evaluation/metrics.py) (`MetricRegistry.roc_auc`) and [evaluation/metric.py](file:///home/kazuha/Documents/competitions/electric-vehicle/evaluation/metric.py).
- **Edge-Case Protections:** Probability bounds clipping $[10^{-15}, 1 - 10^{-15}]$, NaN/Inf fallback replacement, automatic 2D/1D tensor shaping, and string label decoding (`"Yes"` $\to 1$, `"No"` $\to 0$).

---

## 4. Hardware & Quota Guardrails (KAMAS Operational Constraints)
- **Weekly GPU Quota:** 28.0h internal safety cap (reserving 2.0h for emergency final runs).
- **Session Timeout:** 10.5h hard cutoff to prevent unpersisted 12h Kaggle kernel kills.
- **Inference Isolation:** All model artifacts, feature encoders, and weights must run locally or offline with `internet_allowed=False`.
