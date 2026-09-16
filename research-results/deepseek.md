# Top-1 Execution Blueprint: Breaking 0.94672

## 0. Diagnosis: Where the +0.00038 Lives

Your dossier already reverse-engineered the generating recipe. The gap is not in hyperparameter tuning—it is in **modeling the residual non-linearities around the 5.61235 boundary** and **exploiting the test set’s 286k unlabeled rows**. The current GBDT fleet is axis-aligned and collinear ($r = 0.999$). To capture +0.00038, you must inject **smooth continuous manifold learners** and **boundary-local specialists** into the stack, then fuse everything in logit space with a meta-learner that can learn segment-specific weights.

---

## 1. Core Strategy: Five-Layer Stacking Architecture

The final ensemble is a **two-level stack**:

```
Level 0 (Base Models):
  ├─ Stream A: Free-Tree GBDT (LGBM + XGB + CatBoost × 5 seeds × 10 folds)
  ├─ Stream B: Base-Margin GBDT (same models, recipe as init_score)
  ├─ Stream C: FT-Transformer (5 seeds × 10 folds)
  ├─ Stream D: TabNet (5 seeds × 10 folds)
  └─ Stream E: Boundary Specialist XGB (trained on |recipe − 5.61235| < 0.5)

Level 1 (Meta-Learner):
  └─ Logit-space Ridge with segment features
      (Urban/Suburban/Rural one-hot as meta-features)
```

The critical insight: **the meta-learner must see City_Type** so it can learn _different_ blend weights for Urban, Suburban, and Rural segments—directly resolving the Simpson’s Paradox at the fusion stage.

---

## 2. Stream A & B: Dual-Stream GBDT (Your Existing Strength)

You already have this running. The exact configuration:

**Stream A (Free-Tree):**

| Parameter                                | LGBM | XGBoost | CatBoost |
| ---------------------------------------- | ---- | ------- | -------- |
| `n_estimators`                           | 3000 | 3000    | 3000     |
| `learning_rate`                          | 0.02 | 0.02    | 0.02     |
| `num_leaves` / `max_depth`               | 127  | 7       | 7        |
| `min_child_samples` / `min_child_weight` | 40   | 5       | 5        |
| `subsample`                              | 0.85 | 0.85    | 0.85     |
| `colsample_bytree`                       | 0.75 | 0.75    | 0.75     |
| `reg_alpha`                              | 0.1  | 0.1     | —        |
| `reg_lambda`                             | 0.3  | 0.3     | 3.0      |
| `early_stopping_rounds`                  | 200  | 200     | 200      |

**Stream B (Base-Margin):** Identical hyperparameters, but set `init_score = 2.17464 × (recipe_score − 5.61235)` before training. The trees then learn **only the residual wobble** on top of the linear utility function. This is the exact configuration that produced your 0.946312 OOF champion.

**Fusion of A and B:** In logit space, weight $w_A = 0.55$, $w_B = 0.45$. Their $r = 0.9990$ means the variance reduction from averaging is small but non-zero—this alone contributes roughly +0.00002 to +0.00003.

---

## 3. Stream C: FT-Transformer (The Non-Collinear Workhorse)

This is the single most impactful addition. FT-Transformer learns **token-wise embeddings** for each feature and applies self-attention, producing a smooth continuous decision manifold fundamentally different from axis-aligned trees.

**Exact architecture (targeting 0.9430+ individual OOF):**

```python
# Feature Tokenizer
d_token = 64          # token embedding dimension
n_blocks = 2          # number of Transformer blocks
n_heads = 4           # attention heads per block
attention_dropout = 0.1
ffn_dropout = 0.1

# Training
optimizer = AdamW(lr=1e-4, weight_decay=1e-5)
scheduler = CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
batch_size = 512
epochs = 200
early_stopping_patience = 25

# Regularization
feature_tokenizer_dropout = 0.05   # drop tokens randomly
```

**Why these values:** The FT-Transformer reference implementation demonstrates that gains from more heads and depth **diminish past 4 heads / 2 blocks**. With 13 features (plus your engineered interactions), 64-dim tokens give the attention mechanism enough bandwidth to model pairwise interactions without overfitting. The dropout rates are calibrated for ~670k training rows—aggressive enough to prevent memorization, light enough to retain signal.

**Feature preprocessing for FT-Transformer:**

- Numerical features: Quantile-transform to Gaussian, then standardize.
- Categorical features: Learned embeddings (dimension = min(16, cardinality/2)).
- **Critical:** Include `recipe_score`, `dist_to_boundary`, and `abs_dist_to_boundary` as numerical tokens. The Transformer can learn smooth, non-monotonic functions of these that trees cannot represent.

**Expected contribution:** Individual OOF ~0.9420–0.9440. Correlation with GBDT stream: $r \approx 0.965$–0.970. This orthogonality is where the real gain lives.

---

## 4. Stream D: TabNet (Sequential Attention Diversity)

TabNet is the second non-collinear source. Its **sequential attention mechanism** performs soft feature selection at each decision step—a fundamentally different inductive bias from both trees and Transformers.

**Exact configuration:**

```python
# pytorch_tabnet
n_steps = 5                    # decision steps (sweet spot: 3–10)
n_d = 8                        # feature transformer dimension
n_a = 8                        # attention transformer dimension
gamma = 1.5                    # relaxation parameter
lambda_sparse = 1e-4           # sparsity regularization
momentum = 0.02
clip_value = 1.0
batch_size = 256
virtual_batch_size = 128       # ghost batch normalization
lr = 2e-2                      # higher LR works well with StepLR
scheduler = StepLR(step_size=10, gamma=0.9)
epochs = 200
patience = 20
```

**Why these values:** The Nature Scientific Reports TabNet study found that 5 decision steps is the optimal compromise between accuracy and computational time, and lr = 0.02 with StepLR is stable for tabular binary classification. Virtual batch size 128 is critical—it provides batch normalization statistics without the noise of full-batch BN. `lambda_sparse = 1e-4` is aggressive enough to force interpretable feature selection but not so strong that it collapses to a single feature.

**Expected contribution:** Individual OOF ~0.9400–0.9420. Correlation with GBDT: $r \approx 0.955$–0.965. Lower than FT-Transformer, meaning higher ensemble value.

---

## 5. Stream E: Boundary Specialist XGBoost

This is the **surgical strike**. Train a dedicated XGBoost on **only** the samples where $|recipe\_score − 5.61235| < 0.5$. In this region, the linear formula is maximally uncertain, and subtle non-linear cues (Age modulations, commute-time interactions, charging density residuals) become decisive.

**Training protocol:**

```python
# Filter training data
boundary_mask = np.abs(train[‘recipe_score’] - 5.61235) < 0.5
boundary_train = train[boundary_mask]  # ~22% of data, ~147k rows
boundary_test = test[np.abs(test[‘recipe_score’] - 5.61235) < 0.5]

# XGBoost params (tuned for local classification)
params = {
    ‘n_estimators’: 2000,
    ‘max_depth’: 5,              # shallower — less overfitting in local region
    ‘learning_rate’: 0.03,
    ‘subsample’: 0.80,
    ‘colsample_bytree’: 0.70,
    ‘min_child_weight’: 10,      # higher — boundary region is noisy
    ‘reg_alpha’: 0.5,
    ‘reg_lambda’: 1.0,
    ‘eval_metric’: ‘auc’,
    ‘early_stopping_rounds’: 150
}
```

**Key features for the specialist model (beyond the global feature set):**

```python
# Signed distance features
df[‘dist_to_boundary’] = df[‘recipe_score’] - 5.61235
df[‘abs_dist’] = np.abs(df[‘dist_to_boundary’])
df[‘is_uncertain’] = (df[‘abs_dist’] < 0.35).astype(int)

# Age modulation in boundary zone
df[‘age_x_dist’] = df[‘Age’] * df[‘dist_to_boundary’]
df[‘age_squared_dist’] = (df[‘Age’] ** 2) * df[‘dist_to_boundary’]

# Charging interaction (Simpson’s Paradox explicit)
df[‘charging_per_km’] = df[‘Charging_Stations_Near_Home’] / (df[‘Daily_Commute_km’] + 1.0)
df[‘urban_charging’] = (df[‘City_Type’] == ‘Urban’) * df[‘charging_per_km’]
df[‘suburban_charging’] = (df[‘City_Type’] == ‘Suburban’) * df[‘charging_per_km’]
df[‘rural_charging’] = (df[‘City_Type’] == ‘Rural’) * df[‘charging_per_km’]

# Income-residual interaction
df[‘income_resid’] = df[‘Annual_Income_USD’] - 30000  # spike threshold
df[‘income_resid_x_subsidy’] = df[‘income_resid’] * df[‘Subsidy_Available’]
```

**How to integrate into the stack:** The boundary specialist’s predictions replace the global ensemble’s predictions **only for test samples within $|recipe − 5.61235| < 0.5$**. Outside this region, the global ensemble dominates. This is a **gated ensemble**—gating is hard (binary mask), not soft, because the specialist is trained exclusively on boundary data.

**Expected contribution:** +0.00008 to +0.00012 in the boundary region’s AUC.

---

## 6. Stream F: High-Confidence Pseudo-Labeling (The 286k Lever)

The test set is 286,571 rows—42.8% the size of training. High-confidence pseudo-labels from your best ensemble can inject this unlabeled signal back into training.

**Protocol:**

1. **Generate pseudo-labels** from the Level-1 meta-ensemble (Streams A–E fused).
2. **Select only extreme-confidence samples:**
   ```python
   high_conf_pos = test[ensemble_pred > 0.990]
   high_conf_neg = test[ensemble_pred < 0.008]
   ```
   These thresholds yield ~zero label noise because the Bayes error in these regions is negligible. With $p > 0.990$, the expected false positive rate is < 1.5%; with $p < 0.008$, the expected false negative rate is < 0.3%.
3. **Weight the pseudo-labeled samples** at $w = 0.60$ during retraining. This ensures true labels dominate but pseudo-labels still tighten leaf estimates.
4. **Retrain all streams** (A–E) on the augmented dataset: $668k + \text{high\_conf\_rows}$.
5. **Do NOT include pseudo-labels in validation folds**—only in training folds. This prevents leakage and gives honest CV estimates.

**Why this works:** Chris Deotte used pseudo-labeling in Playground Series competitions, noting that high-confidence labels make the feature-space partitioning more stable. The mechanism is variance reduction: pseudo-labeled points act as anchors in regions where the training set is sparse, preventing trees from over-fitting to local noise.

**Expected contribution:** +0.00004 to +0.00008.

---

## 7. Meta-Learner: Segment-Aware Logit Ridge

The fusion step is where Simpson’s Paradox is finally resolved. Do **not** use a global blend weight. Use a **segment-conditional meta-learner**.

**Features for the meta-learner (per test sample):**

| Feature        | Source                                                |
| -------------- | ----------------------------------------------------- |
| `logit_A`      | Stream A (Free-Tree) ensemble logit                   |
| `logit_B`      | Stream B (Base-Margin) ensemble logit                 |
| `logit_C`      | FT-Transformer ensemble logit                         |
| `logit_D`      | TabNet ensemble logit                                 |
| `logit_E`      | Boundary Specialist logit (only for boundary samples) |
| `is_urban`     | 1 if City_Type == Urban                               |
| `is_suburban`  | 1 if City_Type == Suburban                            |
| `is_rural`     | 1 if City_Type == Rural                               |
| `recipe_score` | Raw linear utility                                    |
| `abs_dist`     | $                                                     | recipe − 5.61235 | $   |

**Meta-learner:** Ridge regression with interaction terms between segment dummies and model logits:

```python
from sklearn.linear_model import Ridge

# Design matrix: [logits_A..E, is_urban, is_suburban, is_rural,
#                 is_urban*logit_A..E, is_suburban*logit_A..E, is_rural*logit_A..E]
meta_X = pd.concat([
    logit_df,
    segment_dummies,
    segment_dummies * logit_df,  # interactions
], axis=1)

meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X_train, y_train)
```

**Why Ridge and not Nelder-Mead:** Ridge with segment interactions allows the model to learn **different blend weights for Urban, Suburban, and Rural**. The Simpson’s Paradox in `Charging_Stations_Near_Home` means the optimal weighting of Stream A vs. Stream B differs by city type. A global blend cannot capture this. Ridge with interactions can.

**Post-processing:** Apply a final **logit-space isotonic calibration**? No—isotonic regression can overfit the 57k public LB. Instead, use **Platt scaling** with 3-fold internal CV on OOF predictions. This is conservative and avoids LB overfitting.

---

## 8. Expected Score Trajectory

| Component                           | Incremental Gain | Cumulative AUC    |
| ----------------------------------- | ---------------- | ----------------- |
| Current best (`submission (3).csv`) | —                | 0.94634           |
| Dual-stream A+B fusion              | +0.00002         | 0.94636           |
| FT-Transformer (Stream C)           | +0.00007         | 0.94643           |
| TabNet (Stream D)                   | +0.00004         | 0.94647           |
| Boundary Specialist (Stream E)      | +0.00010         | 0.94657           |
| Pseudo-labeling (Stream F)          | +0.00006         | 0.94663           |
| Segment-aware meta-learner          | +0.00005         | **0.94668**       |
| LB noise (±0.00014 SE)              | —                | **0.9465–0.9468** |

This trajectory conservatively estimates 0.94668 on private LB. The gap to Chris Deotte’s 0.94672 is within $0.3\sigma$ of the binomial standard error—meaning **any remaining gap is likely unclosable by modeling alone** and is within the noise floor of the 57k public / 229k private split.

---

## 9. Critical Implementation Notes

**GPU acceleration:** Use RAPIDS `cuml` for LightGBM and XGBoost. The full pipeline (5 streams × 5 seeds × 10 folds) will take ~4–6 hours on a single T4/P100 with GPU-accelerated GBDT. FT-Transformer and TabNet training adds ~3–4 hours on GPU.

**Seed diversity:** For FT-Transformer and TabNet, use 5 seeds per fold. The variance across seeds is larger for neural models than GBDT; seed averaging is non-negotiable.

**Adversarial validation check:** Before finalizing, train a quick LGBM to distinguish train from test. If adversarial AUC > 0.58, there is distribution shift and the pseudo-labeling step must be more conservative (raise thresholds to 0.995 / 0.005). The dossier does not report adversarial validation results—run it first.

**Do not over-optimize on public LB:** The public LB is only 20% of test (57k samples). Each LB evaluation has ±0.00014 standard error. Any single submission’s public score is noisy. Use OOF CV as the primary selection criterion, and treat LB as a sanity check only. The 150-model Base-Margin run had the best OOF ever recorded (0.946312) but scored lower on LB than `submission (3).csv` purely due to test-subsample noise—your dossier already correctly diagnoses this.

**Final submission selection:** Submit the Level-1 meta-learner output (Platt-scaled). Do not rank-normalize. Do not power-transform. Raw calibrated probabilities in logit space preserve the asymmetric distribution needed for optimal ROC-AUC.

---

## 10. The Brutal Truth

The gap to Rank 1 is +0.00038. With the full five-stream stack described above, you will realistically reach **0.9466–0.9467 OOF**, which corresponds to **0.9465–0.9468 private LB** given the noise floor. Whether you surpass Chris Deotte on the private leaderboard depends on which side of the 0.00014 noise band you land on.

**The only guaranteed path to Rank 1 is to maximize OOF CV.** If your OOF CV exceeds 0.9464, you have done everything modelable. The rest is the Bernoulli coin flip of the private split. Do not chase public LB fluctuations—they are signal noise. Chase OOF.
