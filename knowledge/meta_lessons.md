# Longitudinal Meta Lessons & System Self-Reflection

> **Meta Memory (Layer 4)**  
> Longitudinal insights on agent efficiency, failure patterns, and search trajectory optimizations.

---

## Systematic Invariants

1. **The Public LB Trap:**
   - Optimizing directly for Public LB without local CV alignment causes catastrophic private shake-ups. CV must remain the primary truth sensor.
2. **Tabular Deep Learning vs GBDT:**
   - Across 95% of standard tabular tasks, GBDT ensembles (LightGBM + CatBoost + XGBoost) outperform TabNet and MLPs while training $10\times$ faster.
   - Allocate GPU budget to tree variants and target encoding before attempting neural nets on tabular data.
3. **Hardware Traps:**
   - Pascal architecture (Tesla P100) lacks modern CUDA sm_60 kernels in recent Kaggle base images with PyTorch cu128. Always verify accelerator and fallback to Dual T4 or L4.
