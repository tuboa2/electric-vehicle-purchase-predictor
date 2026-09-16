# Bottom line

Your dossier does **not** support another generic blend. It shows that the four “grandmaster” models are effectively the same ranker:

- XGBoost OOF AUC: **0.9461527**
- LightGBM OOF AUC: **0.9461475**
- CatBoost OOF AUC: **0.9460344**
- Neural net OOF AUC: **0.9385382**
- Existing nontrivial prediction correlations are mostly **>0.999**.
- Several purported “meta-blends” are actually rank vectors: mean ≈ **0.5**, standard deviation ≈ **0.288675**, which is not useful probability blending.
- The winning opportunity is therefore **not model averaging**. It is discovering the residual ordering that all current models miss.

The most defensible attack is a **residual-conditioned pairwise rank-correction stack**: train a deliberately non-collinear residual model on out-of-fold errors, then apply only a heavily regularized correction to the strongest base ranker.

---

# Recommended approach: residual-conditioned pairwise rank correction

## 1. Base prediction

Use the XGBoost model as the primary ranker because it has the best reported OOF AUC:

\[
s_0(x)=\operatorname{logit}\left(\operatorname{clip}(p_{\text{xgb}}(x),10^{-6},1-10^{-6})\right).
\]

Do **not** blend raw probabilities directly. ROC-AUC only depends on ordering, so operate in logit/rank space.

Use a zero-tie deterministic rank transform only at the end:

```python
base_rank = scipy.stats.rankdata(s0, method="ordinal") / (len(s0) + 1.0)
```

---

# 2. Feature formulation

The residual model should not see the entire original feature space. That would simply recreate another correlated tree model. Give it features designed to identify **where the current ranker is locally unreliable**.

## A. Base-model geometry

For every row, create:

```python
p = np.clip(xgb_oof, 1e-6, 1 - 1e-6)
z = np.log(p / (1 - p))

rank_xgb = rankdata(z, method="average") / (len(z) + 1)
rank_lgb = rankdata(lgb_oof, method="average") / (len(z) + 1)
rank_cat = rankdata(cat_oof, method="average") / (len(z) + 1)
rank_nn  = rankdata(nn_oof,  method="average") / (len(z) + 1)
```

Add disagreement features:

```python
model_mean = np.mean([rank_xgb, rank_lgb, rank_cat, rank_nn], axis=0)
model_std  = np.std([rank_xgb, rank_lgb, rank_cat, rank_nn], axis=0)

xgb_lgb_delta = rank_xgb - rank_lgb
xgb_cat_delta = rank_xgb - rank_cat
xgb_nn_delta  = rank_xgb - rank_nn

rank_min = np.min([rank_xgb, rank_lgb, rank_cat, rank_nn], axis=0)
rank_max = np.max([rank_xgb, rank_lgb, rank_cat, rank_nn], axis=0)
rank_range = rank_max - rank_min
```

Important: because the supplied correlations are nearly one, use **differences and local disagreement**, not the predictions themselves.

## B. Probability-tail geometry

The hard-sample report gives:

- Worst false-positive threshold: approximately **0.90109**
- Worst false-negative threshold: approximately **0.00233**

Construct tail indicators and smooth transforms:

```python
tail_features = {
    "p_low_0001": (p < 0.0001).astype("int8"),
    "p_low_001":  (p < 0.001).astype("int8"),
    "p_low_005":  (p < 0.005).astype("int8"),
    "p_high_090": (p > 0.90).astype("int8"),
    "p_high_095": (p > 0.95).astype("int8"),
    "p_high_098": (p > 0.98).astype("int8"),
    "logit_abs": np.abs(z),
    "tail_distance_low": np.maximum(0.00233 - p, 0),
    "tail_distance_high": np.maximum(p - 0.90109, 0),
}
```

The key point is to treat these as **different regimes**, because the residual error mechanism at \(p\approx 0.002\) is almost certainly different from the mechanism at \(p\approx 0.90\).

## C. Row-wise nonlinear feature interactions

For each important original feature \(x_j\), create interactions with model uncertainty:

```python
for col in important_numeric_features:
    x = train[col].astype("float32")
    xz = (x - x.median()) / (x.mad() + 1e-6)

    residual_features[f"{col}__x_disagreement"] = xz * rank_range
    residual_features[f"{col}__x_logit"] = xz * z
    residual_features[f"{col}__x_tail"] = xz * np.abs(z)
```

Restrict this to the top **10–20 features** by XGBoost gain or permutation importance. Do not create a full Cartesian expansion.

## D. Nearest-neighbor local structure

The cleanest non-collinear source of signal is local density and local label geometry.

Build a standardized numeric matrix using only continuous features and compute approximate neighbors with:

- `n_neighbors = 32`
- `metric = "hamming"` for categorical-only representations, or
- `metric = "euclidean"` after robust scaling for numeric features.

For each row compute:

```python
knn_mean_base = mean(base_rank[neighbors])
knn_std_base = std(base_rank[neighbors])
knn_density = 1 / (mean_neighbor_distance + 1e-6)
knn_disagreement = mean(abs(base_rank - base_rank[neighbors]))
```

If the dataset is too large for exact KNN, use FAISS or an approximate nearest-neighbor index.

These features are specifically aimed at rows whose predicted rank is inconsistent with their local manifold.

## E. Category target encodings, strictly OOF

For every categorical column and selected two-way combination:

```text
category
category × age_bin
category × income_bin
category × region
category × vehicle_type
```

Compute smoothed OOF target encodings:

\[
TE_g =
\frac{n_g \bar y_g + \alpha \bar y}
{n_g+\alpha}
\]

Use:

- `alpha = 50` for single columns
- `alpha = 150` for pairwise combinations
- minimum group size: `20`

Also compute **residual encodings** against the base model:

\[
r_i = y_i - p_{0,i}
\]

and encode:

\[
RTE_g =
\frac{\sum_{i\in g} r_i + \lambda \cdot 0}
{n_g+\lambda}.
\]

Use `lambda = 100`.

This is more relevant than ordinary target encoding because the base model has already captured the primary relationship. The residual encoding directly asks:

> Within this subgroup, does XGBoost systematically over-rank or under-rank positives?

---

# 3. Train a genuinely non-collinear residual model

## Model A: CatBoost residual classifier

Train CatBoost on the residual-feature table, not the original table.

```python
CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1800,
    learning_rate=0.018,
    depth=6,
    l2_leaf_reg=18.0,
    random_strength=1.25,
    bagging_temperature=0.75,
    border_count=128,
    random_seed=2026,
    verbose=False,
    allow_writing_files=False,
    thread_count=-1
)
```

Use:

- `class_weights=None`
- no oversampling
- no aggressive class balancing
- early stopping with `od_type="Iter"` and `od_wait=150`

Why CatBoost here? The base ensemble already contains highly correlated tree models, but the residual model has a different feature space: target/residual encodings, neighborhood diagnostics, tail regimes, and model disagreement.

## Model B: calibrated logistic rank correction

Use a sparse linear model as a stabilizer:

```python
LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    l1_ratio=0.35,
    C=0.035,
    max_iter=3000,
    class_weight=None,
    random_state=2026,
    n_jobs=-1
)
```

Features should be:

- rank disagreements
- tail indicators
- residual target encodings
- KNN features
- only the strongest numeric interactions

This model is valuable because its errors will be structurally different from CatBoost’s. It also prevents the residual stack from overfitting to noise.

## Model C: pairwise rank correction

Train an XGBoost ranker on carefully selected positive-negative pairs.

For each fold:

1. Sample 4 million pairs.
2. Restrict pairs to difficult regions:
   - both rows with \(p \in [0.01,0.99]\)
   - one row in the upper disagreement quartile
   - one row in the lower/upper probability tails
3. Construct pairwise feature differences:

\[
\Delta \phi_{ij}=\phi(x_i)-\phi(x_j).
\]

Train:

```python
XGBRanker(
    objective="rank:pairwise",
    eval_metric="auc",
    n_estimators=900,
    learning_rate=0.025,
    max_depth=3,
    min_child_weight=80,
    subsample=0.78,
    colsample_bytree=0.72,
    reg_alpha=0.8,
    reg_lambda=18.0,
    gamma=0.15,
    max_bin=256,
    tree_method="hist",
    random_state=2026
)
```

This model should predict the correction ordering, not the original target.

The pairwise construction is important: optimizing the difficult ranking pairs is much closer to the Kaggle metric than fitting another probability model.

---

# 4. Stacking architecture

Use a three-layer stack.

## Layer 1: locked base ranker

\[
s_0 = \operatorname{logit}(p_{\text{xgb}})
\]

Do not retrain or recalibrate this inside the stack.

## Layer 2: residual experts

Generate:

- \(r_{\text{cat}}\): CatBoost residual prediction
- \(r_{\text{log}}\): logistic residual prediction
- \(r_{\text{pair}}\): pairwise correction score
- \(r_{\text{knn}}\): local-neighborhood correction

Normalize each on the training fold:

\[
\tilde r_k =
\frac{r_k-\operatorname{median}(r_k)}
{\operatorname{IQR}(r_k)+10^{-6}}.
\]

Then define the correction:

\[
\Delta =
0.45\tilde r_{\text{cat}}
+0.25\tilde r_{\text{pair}}
+0.20\tilde r_{\text{log}}
+0.10\tilde r_{\text{knn}}.
\]

## Layer 3: gated correction

The correction must be strongest only where the base model is uncertain or models disagree.

Define:

\[
g(x)=
\operatorname{clip}
\left[
\frac{\operatorname{std}(p_{\text{xgb}},p_{\text{lgb}},p_{\text{cat}},p_{\text{nn}})}
{0.025},
0,1
\right].
\]

Add tail gating:

\[
t(x)=
0.35\mathbf{1}(p<0.005) +
0.35\mathbf{1}(p>0.90) +
0.30\mathbf{1}(|z|<2.5).
\]

Use:

\[
s_{\text{final}}
=

s_0+
\eta\,(0.65g+0.35t)\Delta.
\]

Start with:

```python
eta = 0.018
```

Search only:

```text
eta ∈ {0.006, 0.009, 0.012, 0.015, 0.018, 0.022, 0.027}
```

Do not choose the largest apparent OOF value. Select the value that wins across repeated folds and has the best lower confidence bound.

Finally:

```python
final_prediction = rankdata(s_final, method="ordinal") / (len(s_final) + 1)
```

The rank transform removes calibration artifacts and guarantees unique deterministic predictions.

---

# 5. Meta-model selection

Do **not** fit an unconstrained stacker on the four highly correlated predictions. That is mathematically ill-conditioned.

Instead, fit a constrained logistic meta-model only on the residual features:

```python
meta_features = np.column_stack([
    z,
    rank_range,
    xgb_lgb_delta,
    xgb_cat_delta,
    xgb_nn_delta,
    residual_cat,
    residual_log,
    residual_pair,
    residual_knn,
    tail_low,
    tail_high,
])
```

Use:

```python
LogisticRegression(
    penalty="l2",
    C=0.01,
    solver="lbfgs",
    max_iter=2000,
    random_state=2026
)
```

Then constrain the base coefficient manually:

```python
s_meta = (
    1.0 * z
    + 0.012 * meta_residual_score
)
```

The meta-model should not be allowed to replace the base ranker. Its only job is to make small, localized swaps.

---

# 6. Validation protocol

A 0.00038 leaderboard gap is small enough that ordinary single-fold validation is not reliable.

Use:

- **5 folds**
- **3 random seeds**
- stratified splits
- identical folds for every model
- strictly OOF target encoding
- strictly OOF base predictions
- no test-label inference
- no submission-driven tuning beyond a predeclared small grid

For every candidate, record:

```text
mean OOF AUC
standard deviation across 15 runs
minimum fold AUC
number of pairwise rank flips versus base
AUC gain in each probability regime
AUC gain among high-disagreement rows
```

The candidate should be rejected if it improves global OOF AUC but loses in the hard regimes.

Specifically calculate:

```python
auc_low  = roc_auc_score(y[p_base < 0.005], pred[p_base < 0.005])
auc_mid  = roc_auc_score(y[(p_base >= 0.005) & (p_base <= 0.90)],
                         pred[(p_base >= 0.005) & (p_base <= 0.90)])
auc_high = roc_auc_score(y[p_base > 0.90], pred[p_base > 0.90])
auc_dis  = roc_auc_score(y[model_std > np.quantile(model_std, 0.75)],
                         pred[model_std > np.quantile(model_std, 0.75)])
```

The desired pattern is:

- almost no degradation in the easy middle region,
- positive gain in the high-disagreement quartile,
- positive gain in the \(p<0.005\) and \(p>0.90\) tails.

---

# 7. What not to do

## Do not submit the rank-uniform vectors

The dossier shows multiple submissions with:

- mean ≈ **0.5**
- standard deviation ≈ **0.288675**
- uniform quantiles
- zero ties

Those are rank vectors. They may be valid as AUC submissions, but they discard calibrated structure and are not an ensemble by themselves.

## Do not average the existing four models equally

Their OOF AUCs differ by only about \(1.2\times10^{-4}\), and their predictions are nearly collinear. Equal blending cannot plausibly manufacture a 0.00038 gain.

## Do not optimize directly against the public leaderboard

The gap is too small relative to public/private noise. Use the leaderboard only as a final tie-breaker among candidates that have positive repeated-CV evidence.

## Do not use aggressive probability remapping

Any strictly monotonic transformation of a prediction leaves ROC-AUC unchanged. Calibration, logit transforms, min-max scaling, and power transforms only matter when they interact with a non-monotonic blend or tie-breaking.

---

# The exact candidate I would submit first

After validation, submit this candidate:

\[
\boxed{
s_{\text{submit}}
=

\operatorname{rank}\left[
\operatorname{logit}(p_{\text{xgb}}) +
0.018
\left(0.65g+0.35t\right)
\left(
0.45r_{\text{cat}}
+0.25r_{\text{pair}}
+0.20r_{\text{log}}
+0.10r_{\text{knn}}
\right)
\right]
}
\]

with:

- CatBoost residual model: depth 6, 1,800 iterations, learning rate 0.018, L2 18
- Pairwise XGBoost correction: depth 3, 900 estimators, learning rate 0.025, min child weight 80, lambda 18
- Elastic-net logistic residual model: \(C=0.035\), `l1_ratio=0.35`
- Correction strength: \(\eta=0.018\)
- Zero-tie deterministic ordinal ranking
- No direct averaging of the current model probabilities

This is the most mathematically defensible way to attack the remaining **0.00038**: preserve the current globally strong ordering, identify the narrow subset of rows where all models are uncertain or systematically biased, and spend the entire correction budget on rank swaps in that subset.
