# Empirical Benchmark Citations and Platform Evidence Dossier

This document compiles the authoritative citations, platform telemetry, and empirical studies extracted across the 10 research models.

---

## 1. Multi-Agent Systems & Coordination Theory

1. **Google Research & Openlayer Multi-Agent Synthesis (2026):**
   - *Key finding:* Multi-agent supervisor patterns boost throughput by 80% on parallelizable tasks, but degrade sequential reasoning performance by 39–70% due to quadratic coordination tax and context fragmentation.
   - *Implication:* Reject flat swarms; enforce sequential phase transitions with bounded supervisor authority.
2. **ExecuGraph: Execution-Grounded Validation (arXiv:2607.20499):**
   - *Key finding:* LLM-as-a-judge textual review displays semantic flip rates up to 61.3% and fails to catch physical runtime boundary errors.
   - *Implication:* Gating code commits and submissions must be grounded strictly in physical sandbox execution.
3. **Wink: Recovering from Misbehaviors in Coding Agents (arXiv:2602.17037):**
   - *Key finding:* Phase-gated debugging protocols (isolating root cause before permitting code edits) eliminate "guess-and-check" hallucination loops.
4. **TeamBench & MLE-bench (2026):**
   - *Key finding:* LLM verifiers can act as bottlenecks and suffer from false acceptance if not armed with quantitative metrics.

---

## 2. Competitive Machine Learning & Kaggle Grandmaster Heuristics

1. **NVIDIA Grandmasters' Playbook (2025–2026):**
   - Multi-seed XGBoost, LightGBM, and CatBoost ensembles beat single-seed models by reducing variance.
   - Out-of-Fold (OOF) prediction stacking with regularized meta-models (Ridge, LightGBM) or greedy Hill Climbing consistently outperforms manual weighting.
2. **Adversarial Validation (Fastai / GM lineage):**
   - Train a binary classifier to distinguish training observations from test observations.
   - If AUC $\approx 0.50$, train and test distributions are identical.
   - If AUC $> 0.58$, significant covariate shift exists, requiring adversarial sample re-weighting or feature elimination.
3. **Automated Feature Engineering (OpenFE, 2024–2026):**
   - Two-stage gradient-boosted feature generation yields +1.9% average accuracy improvement across 49 benchmark datasets, demonstrating that feature engineering has a far higher ROI than hyperparameter sweeps.
4. **Pseudo-Labeling Guardrails:**
   - Multi-round soft pseudo-labeling using strict confidence thresholds ($p \ge 0.98$ or $p \le 0.02$) with down-weighting (0.5) to prevent label noise feedback loops.

---

## 3. Platform Capabilities & Telemetry

1. **Antigravity CLI (agy) Architecture:**
   - Subagents run asynchronously in the background without blocking terminal workflows.
   - Workspace skills live under `.agents/skills/`; global skills under `~/.gemini/antigravity-cli/skills/`.
   - Custom agents are defined via YAML frontmatter in `.agents/agents/*.md`.
   - Headless mode (`agy -p`) supports `--json-schema` and `--output-format json|stream-json`.
2. **Gemini 3.8 Flash High Specifications:**
   - Context window: 1,048,576 tokens.
   - Output ceiling: 65,536 tokens (practical output chunk size: ~4,000 tokens).
   - High effort level: Engages iterative reasoning loops and extra tool-calling turns.
   - Cache-Augmented Generation: 90% discount on cached input tokens ($0.075/M cached vs. $0.75/M fresh).
3. **Kaggle Environment Constraints:**
   - Quotas: ~30 GPU hours per week; 12-hour session limit.
   - Code Competitions: Internet disabled during evaluation; all dependencies and pretrained models must be self-contained or mounted via Kaggle Datasets.
   - Hardware: Dual NVIDIA Tesla T4 (15GB each) and NVIDIA L4 are reliable; NVIDIA Tesla P100 fails in recent PyTorch images due to Pascal sm_60 incompatibility.
