I dug through the dossier and checked the live Kaggle competition page as well. One important correction first: the live public leaderboard I can currently see is already showing a different trajectory than the dossier—its displayed #1 is **Rapha at 0.94644**, and Kaggle explicitly notes that the public board is only ~20% of test data and the final 80% can reorder the standings. ([Kaggle][1]) So I would treat `0.94672` as the benchmark you supplied, not assume it is still the live #1 score.

More importantly, I would **not** attack the remaining 0.00038 with another giant LGBM/XGB/CatBoost ensemble. Your forensic results strongly suggest that is the wrong direction: XGB/LGBM/CatBoost are already essentially collinear, with the NN being the only demonstrated source of meaningful orthogonality. The data itself also gives us something much more valuable: an approximate latent generating function.

## The single approach I would build

### **Recipe-Constrained Residual Mixture-of-Experts Ranker**

The idea is:

> **Do not ask one model to rediscover the entire probability function.**
>
> Give it the known synthetic utility score as the dominant coordinate, then train a specialist to learn only the residual ordering errors around that coordinate, with a separate regime model for the Simpson's-paradox infrastructure variables.

Mathematically:

$$
S(x)=
1.2\frac{Income}{100000}
+0.6\,EnvironmentalConcern
+2\,Subsidy
-1\,I(RangeAnxiety=Medium)
-3\,I(RangeAnxiety=High)
$$

and define

$$
z=S(x)-5.61235.
$$

Your dossier says the direct recipe only achieves AUC 0.93769, while the tree ensemble reaches ~0.9463. That is exactly what we want: **the recipe contains the global ordering, while the remaining ~0.009 AUC is primarily residual structure.**

The final score should therefore be:

$$
\boxed{
F(x)=
z
+\alpha(z)\,R_{\text{residual}}(x)
+\beta(z)\,R_{\text{regime}}(x)
}
$$

where the coefficients are learned from OOF data rather than arbitrarily chosen.

The critical point is that the residual models should **not** be allowed to relearn the entire recipe.

---

# 1. Feature space: construct the latent geometry first

I would create exactly these groups.

### A. Ground-truth coordinates

```python
income100 = Annual_Income_USD / 100_000.0

range_med  = (Range_Anxiety == "Medium").astype(float)
range_high = (Range_Anxiety == "High").astype(float)

recipe = (
    1.2 * income100
    + 0.6 * Environmental_Concern
    + 2.0 * Subsidy_Available
    - 1.0 * range_med
    - 3.0 * range_high
)

z = recipe - 5.61235
```

Then:

```text
z
abs_z
z²
z³
sign(z)
sigmoid(2.17464*z)
```

Do **not** throw away `z` after calculating these.

It is the principal latent coordinate.

The dossier's calibrated function is

$$
p_0=\sigma(2.17464z),
$$

which gives a mean probability extremely close to the empirical target prevalence.

---

# 2. Boundary geometry

The biggest opportunity is not the whole dataset.

Your dossier reports that >95% of errors occur in approximately

$$
5.2<S<5.8
$$

or equivalently

$$
|z|<0.3.
$$

So construct:

```python
boundary_10 = (abs_z < 0.10).astype(np.int8)
boundary_20 = (abs_z < 0.20).astype(np.int8)
boundary_30 = (abs_z < 0.30).astype(np.int8)
boundary_40 = (abs_z < 0.40).astype(np.int8)
boundary_50 = (abs_z < 0.50).astype(np.int8)

boundary_weight = np.exp(-(z / 0.30)**2)
```

The Gaussian weight is important.

Instead of throwing away 99% of the data and training on the boundary, train on everything but give the boundary region much greater influence.

For example:

$$
w_i = 1+7\exp[-(z_i/0.30)^2].
$$

This gives:

- far from boundary → weight ≈ 1
- z = ±0.30 → weight ≈ 3.57
- z = 0 → weight = 8

That is statistically cleaner than hard filtering.

---

# 3. Attack the Simpson's paradox explicitly

This is the second major component.

The dossier specifically identifies a sign reversal involving:

- `Charging_Stations_Near_Home`
- `City_Type`
- `Daily_Commute_km`

and says the unconditional relationship reverses after conditioning.

So don't simply feed the raw variables to another GBDT.

Create the conditional coordinates.

```python
commute = Daily_Commute_km
charging = Charging_Stations_Near_Home

charging_per_km = charging / (commute + 1.0)

commute_x_charging = commute * charging

log_commute = np.log1p(commute)
log_charging = np.log1p(charging)

urban = (City_Type == "Urban").astype(float)
suburban = (City_Type == "Suburban").astype(float)
rural = (City_Type == "Rural").astype(float)
```

Then:

```python
urban_charge_density = urban * charging_per_km
suburban_charge_density = suburban * charging_per_km
rural_charge_density = rural * charging_per_km

urban_commute = urban * commute
suburban_commute = suburban * commute
rural_commute = rural * commute

urban_charge = urban * charging
suburban_charge = suburban * charging
rural_charge = rural * charging
```

These are already recommended by the forensic dossier.

I would go one step further and create **boundary × regime interactions**:

```python
boundary_charge = boundary_30 * charging_per_km
boundary_commute = boundary_30 * commute

urban_boundary_charge = urban * boundary_30 * charging_per_km
suburban_boundary_charge = suburban * boundary_30 * charging_per_km
rural_boundary_charge = rural * boundary_30 * charging_per_km
```

This is much more targeted than generic polynomial expansion.

---

# 4. Synthetic-artifact features

The dossier identifies two obvious generator artifacts:

- `Age` is discrete uniform 25–69.
- Income has a $30,000 floor/spike.

Exploit them explicitly:

```python
age = Age
income = Annual_Income_USD

age_mod_2 = age % 2
age_mod_3 = age % 3
age_mod_5 = age % 5
age_mod_7 = age % 7

age_bin_5 = age // 5
age_bin_10 = age // 10

income_floor_distance = income - 30_000
income_above_floor = (income >= 30_000).astype(np.int8)

income_log = np.log1p(income)
income_sqrt = np.sqrt(income)

income_recipe_interaction = income100 * Environmental_Concern
income_subsidy = income100 * Subsidy_Available
```

The modulo variables are intentionally weird.

That is precisely why I would test them.

In a synthetic generator, a seemingly meaningless discrete artifact can encode generator structure that a smooth model otherwise averages away.

---

# 5. Model architecture

I would use **three models only**, but make their roles fundamentally different.

### Model A — Recipe-residual XGBoost

This is the main weapon.

Use XGBoost 3.x with the recipe as `base_margin`.

Your installed environment already reports XGBoost 3.4.1. XGBoost explicitly supports using a margin as the starting prediction, which is exactly the mechanism needed here. ([XGBoost Documentation][2])

Base margin:

$$
m_0=2.17464z.
$$

Exact starting configuration I'd use:

```python
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",

    "tree_method": "hist",

    "max_depth": 7,
    "min_child_weight": 12,

    "learning_rate": 0.025,
    "n_estimators": 4000,

    "subsample": 0.82,
    "colsample_bytree": 0.78,

    "gamma": 0.05,

    "reg_alpha": 0.15,
    "reg_lambda": 8.0,

    "max_bin": 256,

    "seed": seed,
}
```

But the important change is not the hyperparameters.

It's this:

$$
\boxed{
\text{XGB learns } y-\sigma(m_0)
\text{ rather than rediscovering }m_0.
}
$$

Use:

```python
base_margin = 2.17464 * z
```

for both train and validation/test predictions.

XGBoost's documented mechanism allows the base margin to serve as the initial prediction margin. ([XGBoost Documentation][2])

---

# 6. Model B — Boundary residual specialist

This should be **structurally different**, not another copy of Model A.

Use XGBoost:

```python
boundary_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",

    "tree_method": "hist",

    "max_depth": 4,
    "min_child_weight": 20,

    "learning_rate": 0.015,
    "n_estimators": 6000,

    "subsample": 0.90,
    "colsample_bytree": 0.65,

    "gamma": 0.10,

    "reg_alpha": 0.30,
    "reg_lambda": 15.0,

    "max_bin": 128,

    "seed": seed,
}
```

But train it with **boundary weights**:

$$
w_i =
1+15e^{-(z_i/0.25)^2}.
$$

So you're effectively asking:

> "Given that I already know approximately where this person sits on the generator's main decision axis, what explains why the actual target ordering differs?"

This model gets:

```text
Age
age modulo features
income residual features
Environmental_Concern
Daily_Commute_km
Charging_Stations_Near_Home
City_Type
charging/commute ratios
all boundary interactions
recipe residual coordinates
```

but **do not give it the raw recipe as a dominant unrestricted feature**.

Otherwise it simply reconstructs Model A.

---

# 7. Model C — smooth non-tree residual model

This is where I would deviate from your previous NN experiment.

Don't use a generic FT-Transformer and hope it discovers something.

Use a **small residual MLP** whose input is deliberately different:

```text
z
charging_per_km
log_commute
log_charging
Age
Age mod 2/3/5/7
income residual
Environmental_Concern
commute × charging
City-Type one-hot
boundary features
```

Architecture:

```text
Input
  ↓
Linear(64)
LayerNorm
GELU
  ↓
Linear(64)
GELU
  ↓
Residual skip
  ↓
Linear(32)
GELU
  ↓
Linear(1)
```

Training:

```text
optimizer = AdamW
lr = 8e-4
weight_decay = 2e-3
batch_size = 4096
epochs = 80
warmup = 5 epochs
cosine decay
dropout = 0.05
```

Early stop on OOF AUC.

Why bother?

Because your existing NN achieved only 0.938538, but its correlation with the tree ensemble was materially lower (~0.967 versus ~0.999 among GBDTs).

That means its **absolute performance is weak but its error direction is valuable**.

We're going to force the NN to model the residual geometry rather than compete against the GBDT for the entire problem.

---

# 8. The actual stack

This is the part I'd consider the "brutal" component.

Do **not** simply average:

$$
0.5A+0.5B.
$$

Build OOF predictions from all three models.

For each training sample obtain:

$$
p_A,\quad p_B,\quad p_C.
$$

Convert them to logits:

$$
l_A=\operatorname{logit}(p_A)
$$

etc.

Then construct:

$$
\boxed{
L =
l_A
+
w_1(z)(l_B-l_A)
+
w_2(z)(l_C-l_A)
}
$$

where:

$$
w_1(z)=\sigma(a_1+b_1|z|+c_1z^2)
$$

and

$$
w_2(z)=\sigma(a_2+b_2|z|+c_2z^2).
$$

This is much more mathematically sensible than global blending.

Why?

Because the models should not contribute equally everywhere.

Far from the decision boundary:

$$
w_1,w_2\rightarrow0.
$$

The recipe-residual model dominates.

Near the boundary:

$$
w_1,w_2
$$

increase.

The specialists get control exactly where the base model is least certain.

---

# 9. But don't let the meta-model overfit

This is critical.

Your public leaderboard contains only ~57k observations, while training has 668k. Your dossier already demonstrates that tiny CV improvements can reverse on the public board.

Therefore I would **not** train a flexible LightGBM meta-model.

Use a tiny logistic/ridge meta-model:

$$
L =
\theta_0
+\theta_1l_A
+\theta_2l_B
+\theta_3l_C
+\theta_4z
+\theta_5z^2
+\theta_6|z|
+\theta_7l_Bz
+\theta_8l_Cz.
$$

Optimize:

$$
\theta^*
=
\arg\max_{\theta}
AUC_{\text{OOF}}.
$$

But regularize heavily.

I would use:

```python
LogisticRegression(
    C=0.03,
    penalty="l2",
    solver="lbfgs",
    max_iter=2000
)
```

on:

```text
logit_A
logit_B
logit_C
z
abs(z)
z²
logit_B * exp(-(z/0.30)²)
logit_C * exp(-(z/0.30)²)
```

The Gaussian terms are the important part.

---

# 10. The final prediction

I would actually simplify the final deployment equation to:

$$
\boxed{
L(x)
=
l_A
+
\alpha G(z)(l_B-l_A)
+
\beta G(z)(l_C-l_A)
}
$$

where

$$
G(z)=e^{-(z/0.30)^2}.
$$

Start with:

$$
\alpha=0.30
$$

and

$$
\beta=0.03.
$$

Then optimize these against **OOF predictions**, not the leaderboard.

The reason for keeping the NN at only a few percent is directly supported by your experiments: the NN's lower raw AUC but lower correlation makes it useful as an error-corrector rather than a primary predictor.

---

# 11. I would NOT use the proposed pseudo-labeling immediately

This is one place where I'd modify the dossier.

The dossier recommends pseudo-labeling predictions above 0.990 or below 0.008 with weight 0.60.

I don't consider the statement that those test labels contain "zero target noise" sufficiently established from the evidence shown.

More importantly, **pseudo-labeling is unlikely to be where the decisive 0.00038 comes from**.

The test distribution is already generated from the same synthetic mechanism. If the generator's latent structure is what we're exploiting, adding millions of effectively redundant high-confidence observations can mostly tighten already-correct regions while doing little for the ranking errors that actually affect AUC.

I would only run pseudo-labeling as a secondary experiment after the residual architecture works.

---

# 12. The really important validation experiment

Don't ask:

> "Did CV increase?"

Ask:

> **"Did the model improve ranking specifically inside the residual frontier?"**

Create four AUCs:

```text
AUC(all)
AUC(|z| < 0.10)
AUC(|z| < 0.20)
AUC(|z| < 0.30)
```

Then calculate:

$$
\Delta AUC_{\text{boundary}}
=
AUC_{\text{new,boundary}}
-
AUC_{\text{champion,boundary}}.
$$

This is the metric I would use to decide whether the approach is actually attacking the gap.

If your full OOF rises by 0.00003 but boundary AUC rises by 0.001+, that's much more interesting than a generic +0.00005 improvement.

---

# 13. One more trick: optimize the ranking, not probability calibration

Your previous experiments strongly suggest this.

The dossier reports that rank normalization hurt performance, while probability/logit blending performed better.

That's important because ROC-AUC only cares about ordering:

$$
AUC=P(F(X^+)>F(X^-)).
$$

Therefore the optimization target is:

$$
\max_F
P[
F(X^+)>F(X^-)
].
$$

Not:

$$
\min \text{log-loss}.
$$

Not:

$$
\min \text{Brier}.
$$

Not:

$$
\text{match target prevalence}.
$$

So after generating the final OOF stack, perform a **monotonic score search** over the residual correction:

$$
F=z+\lambda R.
$$

Test:

$$
\lambda\in
\{0.00,0.02,0.04,\ldots,0.50\}
$$

then locally optimize around the best region.

This is cheap and directly aligned with the metric.

---

# 14. Training layout

I would use:

```text
668,665 train
│
├── 10-fold stratified OOF
│
├── Model A
│     Recipe base-margin XGB
│     5 seeds
│
├── Model B
│     Boundary-weighted residual XGB
│     5 seeds
│
└── Model C
      Residual MLP
      5 seeds
```

But the **stacker must only see OOF predictions**.

Never train the stacker on in-sample base predictions.

Final training:

```text
10 folds × 5 seeds × 3 models
= 150 fitted models
```

That sounds excessive, but your existing 150-model experiment already demonstrated the computational pattern; its failure was primarily inductive bias, not insufficient ensemble size. The dossier reports that its OOF AUC was 0.946312 while its public score was 0.94628.

---

# 15. What I expect the contribution hierarchy to look like

I would **not** assume the dossier's claimed `+0.00012 +0.00010 +...` gains are additive. Those are hypotheses, not independently measured causal contributions.

My expected hierarchy is:

```text
Recipe-constrained residual XGB
        ↓
largest contribution

Boundary-specialist residual
        ↓
second largest

Explicit City × Charging × Commute geometry
        ↓
third

Residual MLP
        ↓
small but potentially decisive

Pseudo-labeling
        ↓
only if experiments prove it
```

The key is that these improvements attack **different error subspaces** rather than adding more highly correlated trees.

---

# 16. Why I think this is materially different from what you've already done

Your current situation is essentially:

$$
\underbrace{\text{GBDT ensemble}}_{\text{very strong}}
+
\underbrace{\text{more GBDTs}}_{\rho\approx0.999}
$$

The proposed system is:

$$
\boxed{
\text{Known latent utility}
+
\text{conditional residual learner}
+
\text{boundary specialist}
+
\text{smooth residual learner}
}
$$

That changes the geometry of the problem.

Your existing GBDTs have OOF AUCs of approximately:

$$
XGB=0.946153
$$

$$
LGBM=0.946147
$$

$$
CatBoost=0.946034
$$

while the NN is only 0.938538.

So trying to find a slightly better generic GBDT is attacking a nearly exhausted direction.

The forensic evidence instead says:

$$
\rho_{\text{GBDT,GBDT}}\approx0.999
$$

while

$$
\rho_{\text{GBDT,NN}}\approx0.967.
$$

That's the mathematical justification for forcing the models to specialize rather than merely ensembling more of the same.

---

## My exact "Top-1 attempt" configuration

If I had one serious run left, I'd use:

| Component           | Configuration                                                                                 |     |        |
| ------------------- | --------------------------------------------------------------------------------------------- | --- | ------ |
| Base latent         | `2.17464 × (recipe - 5.61235)`                                                                |     |        |
| Model A             | XGB residual, depth 7, `min_child_weight=12`, `eta=.025`, 4000 trees                          |     |        |
| Model B             | XGB boundary specialist, depth 4, `min_child_weight=20`, `eta=.015`, 6000 trees               |     |        |
| Boundary weighting  | `1 + 15 exp(-(z/0.25)^2)`                                                                     |     |        |
| Model C             | 64→64→32 residual MLP, GELU, LayerNorm, AdamW                                                 |     |        |
| MLP LR              | `8e-4`                                                                                        |     |        |
| MLP WD              | `2e-3`                                                                                        |     |        |
| MLP dropout         | `.05`                                                                                         |     |        |
| Main boundary       | `                                                                                             | z   | < .30` |
| Specialist features | charging/km, commute×charging, City×charging, City×commute, age modulo, income-floor features |     |        |
| Fusing              | **logit residual stacking**                                                                   |     |        |
| Initial weights     | A=1, B≈0.30, C≈0.03                                                                           |     |        |
| Meta model          | L2 logistic, `C=.03`                                                                          |     |        |
| Seeds               | 5                                                                                             |     |        |
| Folds               | 10 stratified                                                                                 |     |        |
| Final tie handling  | deterministic `rank + tiny epsilon`, only after prediction                                    |     |        |

And I would **not** submit the raw probability from the meta-model until checking its OOF ranking against:

1. your `submission (3)`,
2. your best current ensemble,
3. pure recipe,
4. recipe + Model A,
5. recipe + A+B,
6. recipe + A+B+C.

The winning OOF ranking should then be tested against the public LB—but remember that Kaggle itself warns that the public 20% is not the final 80%, so don't overreact to a single 0.0000x movement. ([Kaggle][1])

One implementation detail is especially safe: both XGBoost's `base_margin` mechanism and LightGBM's `init_score` are officially supported mechanisms for supplying an initial prediction, so this isn't relying on an ad-hoc "feature trick." ([XGBoost Documentation][2])

**Bottom line:** I would stop spending compute on “more powerful GBDT.” The most defensible attack on the remaining gap is to turn the reverse-engineered `Buy Score` into an explicit latent coordinate, make XGBoost learn only its unexplained residual, give a second model almost exclusive responsibility for the `|z| < 0.3` frontier, and use a heavily regularized **conditional logit stack** to let those specialists modify the base ordering only where they have evidence to do so.

That is the architecture I'd bet the remaining compute budget on—not because it guarantees `0.94672` (nothing in the dossier can establish that), but because it is the approach most directly aligned with the demonstrated error structure rather than another iteration of the already-saturated GBDT direction.

[1]: https://www.kaggle.com/competitions/playground-series-s6e9/leaderboard?utm_source=chatgpt.com "Predicting Electric Vehicle Purchases | Kaggle"
[2]: https://xgboost.readthedocs.io/en/stable/tutorials/intercept.html?utm_source=chatgpt.com "Intercept — xgboost 3.3.0 documentation"
