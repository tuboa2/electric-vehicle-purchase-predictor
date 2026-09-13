# Research Extraction: gemini.md

```yaml
source:
  file: research/01-raw/gemini.md
  model: Gemini (DeepMind-aligned system research)
  date_if_available: 2026-09-13

claims:
  - claim: Gemini 3.8 Flash High features 1M input window, 65k output limit, 90.8% on Terminal-Bench 2.1, 73.7% on DeepSWE v1.1; utilizes GQA and FlashAttention-3.
    category: model_capability
    evidence: Google DeepMind model card / cloud docs (Tier T1).
    confidence: high
    implementation_relevance: high
  - claim: Cache-Augmented Generation (CAG) offers a 90% discount on cached tokens ($0.075/M vs $0.75/M standard input), making large static context blocks cost-effective.
    category: cost_efficiency
    evidence: Gemini API pricing structure (Tier T1).
    confidence: high
    implementation_relevance: high
  - claim: Text-only LLM-as-a-judge reviews exhibit false-negative flip rates up to 61.3% on semantic rephrasings and fail to detect runtime boundary flaws.
    category: validation_risk
    evidence: ExecuGraph paper (arXiv:2607.20499) (Tier T3).
    confidence: high
    implementation_relevance: critical (validation must be execution-grounded in physical sandbox, not textual debate).
  - claim: Tabular Neural Networks (RealMLP, TabM, FT-Transformer) combined with GBDTs (XGBoost, LightGBM, CatBoost) provide essential error diversity for ensembling.
    category: kaggle_technique
    evidence: Kaggle winning writeups (Playground S6E4/S6E5/S6E7) (Tier T5).
    confidence: high
    implementation_relevance: high
  - claim: Command execution policy should prefer sandboxing rather than `--dangerously-skip-permissions` where supported to isolate execution.
    category: platform_security
    evidence: Antigravity CLI security docs (Tier T1).
    confidence: high
    implementation_relevance: medium_high

architectural_recommendations:
  - recommendation: Hybrid Hierarchical Supervisor with Execution-Grounded Validation (ExecuGraph). Acceptance of any code or submission is gated strictly by physical execution outcomes (CV, runtime, memory).
    rationale: LLMs are unreliable judges of code semantics in the abstract; ground truth comes from sandbox execution.
    dependencies: Execution environment / subprocess sandbox.
    risks: Requires robust sandbox error trapping and isolation.
  - recommendation: 19 specialized agents organized into 7 cells (Strategic Commander, Competition Intelligence, Data Forensics & Validation, Modeling & Experimentation, Ensemble & Post-Processing, Adversarial & Submissions, Evolution).
    rationale: Minimizes shared mutable state and prevents context pollution.
    dependencies: Custom agent frontmatter.
    risks: 19 agents may introduce excessive coordination overhead if not strictly gated.
  - recommendation: Use Hill Climbing greedy search for OOF blend weights and stacking meta-models (Ridge / LightGBM) over N x C probability matrices.
    rationale: Standard GM technique to extract maximum ensemble gain from diverse OOFs.
    dependencies: Clean OOF predictions saved as Parquet/CSV.
    risks: Hill climbing can overfit if fold counts are small.

kaggle_recommendations:
  - recommendation: Local CV is primary ground truth; ignore public LB if it contradicts local CV (protects against private LB shakeout).
    expected_value: critical (prevents shakeout).
    evidence: Kaggle competitive history.
  - recommendation: Implement phase-gated debugging (isolate root cause before editing code; avoid guess-and-check loops).
    expected_value: high (prevents token churn and hallucination spirals).
    evidence: Wink paper (arXiv:2602.17037) & catalog skill.
  - recommendation: Polars with lazy execution (`scan_csv`, predicate pushdown) for feature engineering to avoid 16GB RAM crashes.
    expected_value: high (memory efficiency on Kaggle notebook).
    evidence: Polars vs Pandas benchmarks.

agent_recommendations:
  - role: Commander
    responsibility: Global state machine, resource allocation, stopping decisions.
    justification: Centralized strategic coordinator.
  - role: Competition Researcher & Rule Compliance Agent
    responsibility: Parse rules, metrics, submission contracts, external data licensing.
    justification: Guarantees legal submission validity.
  - role: Data Forensics & EDA Agent
    responsibility: Polars profiling, anomaly detection, distribution shift analysis.
    justification: Identifies data quality and modality.
  - role: Validation Architect
    responsibility: Design CV splits (GroupKFold, TimeSeriesSplit), ensure strict OOF alignment; holds absolute VETO power.
    justification: CV integrity dictates competition outcome.
  - role: Model Researcher, Feature Engineer, HPO Agent, Experiment Manager
    responsibility: Modality-specific model exploration, hypothesis-driven feature generation, Optuna tuning, trial registry.
    justification: Core iterative modeling cycle.
  - role: Ensemble Agent & Performance Engineer
    responsibility: OOF stacking, Hill Climbing weights, vectorization, memory optimization.
    justification: Maximizes predictive power and stays within limits.
  - role: Adversarial Reviewer & Leakage Hunter
    responsibility: Red-team attacks, leakage probing, inference feature availability check.
    justification: Catches hidden fatal flaws before execution.
  - role: Kaggle Executor & Artifact Analyst
    responsibility: Environment packaging via uv, CLI kernel push, artifact download, schema verification.
    justification: Manages platform interaction.
  - role: Error Analyst, Strategy Evolution Agent, Memory Curator
    responsibility: Residual analysis, translating failures/successes into generalizable heuristics, knowledge graph indexing.
    justification: Powers recursive self-improvement.

memory_recommendations:
  - Meta-Kaggle Knowledge Graph mapping causal relationships between dataset characteristics, strategies, and empirical outcomes.
  - Layered memory: Working (in-memory), Project (`knowledge/`), Strategic (`knowledge/strategies/`), Meta (`knowledge/agent-performance/`).
  - Cache-Augmented Generation (CAG) for massive historical reference blocks.

self_evolution_recommendations:
  - Recursive Self-Improvement (RSI): Execution outcomes converted into reusable evidence.
  - Skill evolution via subprocess execution tests and negative controls.
  - Phase-gated debugging to prevent trial-and-error code churn.

compute_recommendations:
  - Expected value prioritization formula: `(Expected Score Gain * Confidence * Information Gain) / (Compute Cost * Risk)`.
  - Partitioning: CPU-only for Polars feature engineering and Hill Climbing; GPU exclusively for Neural Networks and GPU-boosted trees.
  - Fast dependency installation via `uv`.

automation_recommendations:
  - CLI execution via `kaggle kernels push` and `kaggle kernels output`.
  - Git worktrees via `using-git-worktrees` for parallel branch isolation.
  - Headless execution with sandboxed terminal commands.

concerns:
  - 19 agents may exceed the minimum effective agent roster and increase token costs unless cells are activated selectively.
  - High reliance on prompt caching requires consistent prefix construction.

contradictions:
  - Recommends 19 agents (brief asks for minimal, effective architecture without dumb swarms; Claude recommended 13).
  - Proposes deep tabular NNs (RealMLP/TabM) as mandatory baseline diversity, whereas Claude focused primarily on GBDT baselines first.
```
