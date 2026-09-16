## Executive read: what the +0.00038 actually is

First, the honest math, because it dictates the strategy. Using the Hanley–McNeil AUC variance formula on the **private** split (40,073 positives / 189,183 negatives): **SE ≈ 0.00079** (conservative bound; empirically it's ~0.0003–0.0004). Deotte's +0.00038 is therefore **~1σ on private** — meaningfully beatable only if you (a) add genuine orthogonal signal he hasn't extracted, and (b) shrink variance harder than he does. Dossier §4.1 already proved your 150-seed base-margin run _was_ the better model; the public LB just couldn't see it. The play is not "a better single model." It is a **gated, oracle-informed, variance-shrunk stack** built around the one thing you have that pure GBDT competitors don't: the exact generating recipe.

Here is the single unified approach — call it the **Oracle-Gated Residual Stack (OGRS)**.

---

## 1. The Architecture (one pipeline, four stages)

```
RAW FEATURES ──► FEATURE MATRIX (§2) ──► L0 ZOO (§3) ──► GATED L1 (§4) ──► L2 LOGIT BLEND (§5) ──► PSEUDO-LABEL ROUND 2 (§6)
                                              │                                     │
                                    10-fold × 5 seeds OOF               constrained weights,
                                    (20 folds for NN)                   optimized on OOF AUC
```

The single most brutal, non-collinear idea in this design: **hard-gate inference by distance to the known decision boundary.** The recipe score is a _known deterministic function_ of the features — not a prediction. So you can partition the test set at inference time using ground-truth-generating information:

- **|BuyScore − 5.61235| ≥ 0.30** → recipe is decisive; blend L0 zoo normally.
- **|BuyScore − 5.61235| < 0.30** → the recipe is _non-identifiable_ here; this is where ~95% of residual error lives (your diagnostics: MAE concentrates at FP threshold 0.901 / FN threshold 0.0023). Route these rows to a **Boundary Specialist** trained _only_ on the boundary band, plus a **Second-Order Interaction model** that captures what the linear recipe provably cannot.

This gating converts a uniform +0.00038 hunt into a targeted attack on the exact region containing the error mass.

---

## 2. Exact Feature Matrix (the "X")

All features computed once, cached as parquet. `B` = BuyScore from the §2.1 recipe; `τ = 5.61235`; `m = 2.17464·(B − τ)` = base margin.

| #     | Feature                      | Formula / construction                                                                                                                                                                                                                                             |
| ----- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1–13  | Raw features                 | passthrough                                                                                                                                                                                                                                                        |
| 14    | `recipe_score`               | B                                                                                                                                                                                                                                                                  |
| 15    | `base_margin`                | m                                                                                                                                                                                                                                                                  |
| 16    | `dist_to_boundary`           | B − τ                                                                                                                                                                                                                                                              |
| 17    | `abs_dist`                   | \|B − τ\|                                                                                                                                                                                                                                                          |
| 18    | `is_boundary`                | 1[abs_dist < 0.30]                                                                                                                                                                                                                                                 |
| 19    | `boundary_side`              | sign(B − τ) · abs_dist (signed magnitude — one feature, preserves both)                                                                                                                                                                                            |
| 20–22 | Cohort one-hots              | Urban / Suburban / Rural from City_Type                                                                                                                                                                                                                            |
| 23    | `charging_per_km`            | Charging_Stations_Near_Home / (Daily_Commute_km + 1.0)                                                                                                                                                                                                             |
| 24–26 | Simpson interactions         | `is_urban × charging_per_km`, `is_suburban × charging_per_km`, `is_rural × charging_per_km` — **these three features alone resolve the paradox**; the sign-flip becomes a learnable linear term instead of a tree-splits-everywhere problem                        |
| 27    | `income_commute`             | Annual_Income_USD / 100000 × Daily_Commute_km                                                                                                                                                                                                                      |
| 28    | `income_env`                 | (Annual_Income/1e5) × Environmental_Concern                                                                                                                                                                                                                        |
| 29    | `subsidy_env`                | Subsidy_Available × Environmental_Concern                                                                                                                                                                                                                          |
| 30    | `income_floor_flag`          | 1[Annual_Income == 30000] (the synthetic floor spike — a pure artifact, trees waste splits rediscovering it)                                                                                                                                                       |
| 31–38 | **In-fold target encodings** | For each categorical (City_Type, Range_Anxiety, Education_Level, Gender, etc.): TE computed on the 9 training folds only, smoothed: `te = (Σy + 0.3·μ·C) / (n + 0.3·C)`, C=20, μ=0.1748. **Never** fit on the held-out fold — this is the #1 silent leakage source |
| 39–44 | Recipe-conditional residuals | For each of the 4 recipe terms (income, env, subsidy, range_anxiety_penalty): `term_value × is_boundary` — lets the specialist re-weight recipe terms locally                                                                                                      |
| 45    | `age_bucket`                 | floor(Age/5) — captures the discrete 45-sided-die artifact                                                                                                                                                                                                         |

**Drop** `Will_Buy_EV`, IDs. Keep `Age` raw _and_ bucketed — the die artifact is discrete; trees with `min_data_in_leaf` ≥ 50 cannot isolate individual ages.

---

## 3. L0 Model Zoo — exact hyperparameters

Three inductive biases, deliberately decorrelated. All trained **10-fold stratified × 5 seeds (seeds 2021–2025)** with early stopping on fold-AUC.

**Stream A — Free Trees** (your 0.94634 recipe, untouched):

```python
LGBM: lr=0.02, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.80,
      bagging_fraction=0.80, bagging_freq=1, lambda_l2=3.0, max_depth=-1,
      12000 rounds, early_stop=400, objective='binary', metric='auc'
XGB:  max_depth=7, eta=0.02, min_child_weight=30, subsample=0.80,
      colsample_bytree=0.70, reg_lambda=3.0, reg_alpha=0.1,
      tree_method='hist', 12000 rounds, early_stop=400
CAT:  depth=7, lr=0.03, l2_leaf_reg=3.0, random_strength=1.0,
      bagging_temperature=1.0, 12000 rounds, early_stop=400, loss='Logloss'
```

**Stream B — Residual Trees** (base margin; learns only the wobble):

```python
Same three configs, but: max_depth=5, num_leaves=31, lr=0.015,
min_data_in_leaf=200, lambda_l2=8.0  ← heavily regularized: the residual
signal is weak and the trees must NOT be allowed to relearn the recipe
(which would re-collinearize Stream A and B).
base_margin = 2.17464 * (B - 5.61235)  on every train/predict call.
```

**Stream C — Non-Tree Orthogonals** (the anti-collinearization layer; tree↔tree r = 0.999 is your enemy):

```python
# C1. Penalized logistic on recipe + interactions (GLM — totally different function class)
logit(y) = m + β·[features 23–29, 31–38, 44]
sklearn LogisticRegression(C=0.05, penalty='l2', solver='lbfgs', max_iter=2000)

# C2. Tabular NN with entity embeddings (extend kaggle_train_nn.py)
- embeddings: dim = min(50, (cardinality+1)//2) per categorical
- numeric branch: Linear(64) → GELU → Linear(32)
- embedding concat + numeric → 512 → 256 → 128 (GELU, BatchNorm1d, Dropout 0.20)
- output: single logit ADDED to frozen base_margin m (residual head!)
- AdamW lr=1e-3, weight_decay=1e-4, batch=4096, cosine annealing, 60 epochs,
  SWA last 10 epochs, 5 seeds, 20-fold OOF (NNs are high-variance; more folds = tighter OOF)
- Target: push NN OOF 0.9385 → 0.942+ (residual head + cohort features make this realistic)

# C3. Boundary Specialist — the gated weapon
Train ONLY on rows with abs_dist < 0.45 (expect ~8–12% of data; expand 0.30→0.45
at training so the gate margin 0.30 is interpolated, not extrapolated).
LGBM: num_leaves=15, min_data_in_leaf=500, lr=0.01, lambda_l2=20,
      feature_fraction=0.60, 8000 rounds, early_stop=300.
Features: ALL of §2, including recipe-conditional residual terms 39–44.
Crucially, ALSO append each row's abs_dist as a sample weight = exp(-abs_dist/0.15)
so the specialist is sharpest exactly at the boundary, and blend predictions
across the 0.30–0.45 overlap band (§4).
```

The residual head on the NN is the subtlest non-collinearization move: forcing every model to predict `logit(p) − m` means Stream A, B, C1, C2 share the _calibration_ but compete only on the wobble geometry — ensemble diversity where it matters, agreement where the recipe already decides.

---

## 4. Gated L1 — how the gate works at inference

```
for each test row:
    if abs_dist < 0.30:  p = 0.55·boundary_specialist + 0.45·L2_blend      (in logit space)
    if 0.30 ≤ abs_dist < 0.45:  p = w(d)·specialist + (1-w(d))·L2_blend,
        w(d) = 0.55 · exp(-(d-0.30)/0.15)   # smooth handoff, no discontinuity
    else:  p = L2_blend
```

AUC is rank-based, so the gate weights (0.55/0.45) must be **tuned on OOF within the band only** — optimize band-restricted AUC via 1-D search on `w ∈ [0.3, 0.8]`, not global AUC. Tun­ing on global AUC will wrongly shrink `w` because the band is only ~10% of rows.

**Validation rule (this is the whole game):** OOF must be computed through the _same_ gate logic. Never evaluate the specialist on non-band rows in OOF — that inflates its apparent contribution and you'll mis-set `w`.

---

## 5. L2 — the logit blender

L2 input vector per row (all in logit space): `[lgbm_A, xgb_A, cat_A, lgbm_B, xgb_B, cat_B, glm, nn]`.

- **Do not** use an unconstrained logistic meta-learner — with 8 correlated columns it overfits OOF by ~0.0001–0.0002 (fatal at this margin).
- Use your existing `kaggle_master_blend.py` Nelder–Mead, but with the **simplex constrained to the unit simplex** (weights ≥ 0, Σw = 1). This is ridge-like shrinkage for free and is exactly the variance-reduction the 150-seed run proved works.
- Strong prior: `w_nn ≈ 0.04–0.06`, `w_glm ≈ 0.03–0.05`, remainder split roughly evenly across A and B streams with Stream A slightly favored (Free Trees still carry the sharpest global geometry — your 0.94634).
- Optimize on OOF AUC of **668,665 rows**, then verify stability: refit weights on 9 of 10 folds, score the 10th; require |ΔAUC| < 0.00003 across folds. If unstable, halve the NN/GLM weights.

---

## 6. Pseudo-labeling Round 2 (the Deotte-class move)

Round 1 = full §1–§5 pipeline, trained on the 668,665 original rows only. Then:

1. Score test (286,571 rows) with the Round-1 gate stack.
2. Take positives with **p̂ > 0.995** and negatives with **p̂ < 0.003** (tighter than the dossier's 0.990/0.008 — at this confidence the injected noise is negligible; expect ~120–160k rows retained).
3. **Balance**: downsample the negative pseudo-labeled pool so the pseudo-set is ~15% positive (slightly below the true 17.48% rate to counter the mild positive bias of confidence thresholds). Cap pseudo-set size at 60% of original training size.
4. Concatenate with sample_weight = 0.6 for pseudo rows, 1.0 for real rows.
5. Retrain the **entire pipeline** (Streams A, B, C1, C2, specialist, gate, blender) identically.
6. **Final submission = 50/50 logit-average of Round-1 and Round-2 outputs.** Never go 100% Round-2 — if pseudo-labeling has any subtle drift, the average caps the downside while keeping most of the gain. This is your private-LB insurance policy.

---

## 7. Gain budget and execution order

| Component                            | Expected OOF ΔAUC       | Evidence anchor                                                  |
| ------------------------------------ | ----------------------- | ---------------------------------------------------------------- |
| Simpson interaction features (24–26) | +0.00006–0.00010        | dossier Component 1; turns a sign-flip into a linear term        |
| In-fold target encodings (31–38)     | +0.00004–0.00008        | standard GM-tier gain on high-cardinality synthetic categoricals |
| Gated boundary specialist (§4)       | +0.00008–0.00015        | 95% of error mass in the band; diagnostics MAE 0.137             |
| NN residual head + GLM diversity     | +0.00003–0.00005        | dossier §4.3 (NN blend already bought +0.00001 on LB at r=0.967) |
| Dual-stream A/B stacking             | +0.00002–0.00004        | dossier §4.1 — your two best models are complementary            |
| Pseudo-label Round 2                 | +0.00004–0.00008        | 286k unlabeled rows = 43% of train                               |
| **Total**                            | **+0.00027 – +0.00050** | ceiling ≈ 0.94661–0.94684 private expectation                    |

Realistic landing zone: **0.94655–0.94670**, with a genuine (not coin-flip) shot at 0.94672+, because unlike pure reseeding, features #24–26 and the gated specialist add _signal orthogonal to the recipe_, not just variance reduction.

**Execution order** (each step is independently verifiable on OOF — stop anywhere):

1. Feature matrix + TEs → rerun Stream A (sanity: OOF ≥ 0.94634)
2. Add Simpson features + specialist gate → OOF checkpoint (target ≥ 0.94645)
3. Add Streams B, C1, C2 → L2 blend (target ≥ 0.94650)
4. Pseudo-label Round 2 + 50/50 average → final
5. Submit Round-2-average only after step 2's OOF gain ≥ +0.00008; otherwise bank the step-2 model and spend remaining submissions on weight sensitivity probes (±0.05 on gate weight, ±0.02 on NN weight — pick the config with the best **worst-fold** OOF, not the best mean).

Two final warnings, because at this margin discipline beats cleverness: **(1)** every OOF number above must come from the gated pipeline evaluated identically at train and inference time — an ungated OOF will lie to you by ~0.0001; **(2)** select your final submission by _worst-fold_ OOF across the 10 folds, never by public LB — with SE ≈ 0.0003–0.0008 on private, the leaderboard is a noisy witness, and the worst-fold criterion is your only defense against a config that got lucky on fold 7.
