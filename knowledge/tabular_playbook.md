# Tabular Competitive Machine Learning Playbook

> **Strategic Memory (Layer 3)**  
> Curated heuristics, feature engineering patterns, and model configurations derived from Kaggle Grandmaster benchmarks.

---

## 1. Feature Engineering Core Heuristics

1. **Aggregation & Groupby Statistics:**
   - Group numerical features by high-importance categoricals: `mean`, `std`, `min`, `max`, `skew`.
   - Calculate relative differences: $x - \text{mean}(x_{\text{group}})$ and ratios: $x / (\text{mean}(x_{\text{group}}) + \epsilon)$.
2. **Target Encoding:**
   - Always apply out-of-fold target encoding within the CV loop.
   - Smooth with m-estimate smoothing: $S = \frac{n \cdot \bar{y}_c + m \cdot \bar{y}}{n + m}$.
   - Never compute target encoding globally before train/val split.
3. **Frequency Encoding:**
   - Replace categoricals with their normalized value counts. Extremely effective for tree-based models on anonymized columns.
4. **Interaction Features:**
   - Arithmetic combinations ($A \times B$, $A / (B + 1)$) of top-ranked features identified by permutation importance.

---

## 2. Model Zoo & Ensembling

1. **Diverse Model Families:**
   - LightGBM (fast, leaf-wise split, excellent baseline).
   - CatBoost (symmetric trees, superior handling of raw categorical features and text).
   - XGBoost (exact/approximate greedy depth-wise splits, distinct regularization mechanics).
2. **Hill-Climbing Ensemble Selection:**
   - Forward selection with replacement on out-of-fold predictions.
   - Stop when metric delta $< 0.0001$.
