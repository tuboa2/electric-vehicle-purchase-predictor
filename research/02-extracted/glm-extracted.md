# Research Extraction: glm.md

```yaml
source:
  file: research/01-raw/glm.md
  model: GLM (Super Z / Principal Kaggle Competition Strategist)
  date_if_available: 2026-09-13

claims:
  - claim: Antigravity CLI (agy) is a terminal-native, filesystem-aware coding assistant supporting custom agents, async subagents, workspace/global skills, hooks, MCP, and headless execution (`--headless` / `-p`).
    category: platform_capability
    evidence: Official Antigravity docs / release notes (Tier T1/T4).
    confidence: high
    implementation_relevance: critical
  - claim: Gemini 3.8 Flash High is the high-reasoning-effort variant of Gemini 3.8 Flash; effective on bounded coding/tool-use tasks but degrades on unguided multi-step planning without intermediate verification or adversarial review.
    category: model_capability
    evidence: Google DeepMind model card / empirical long-context benchmarks (Tier T1/T3).
    confidence: high
    implementation_relevance: critical
  - claim: Pure LLM multi-agent debate on every turn is cost-ineffective and causes conversational ping-pong; communication must be mediated via a filesystem blackboard of structured YAML/JSON/Parquet artifacts.
    category: multi_agent_topology
    evidence: Peer-reviewed literature on multi-agent debate scaling laws (Tier T3).
    confidence: high
    implementation_relevance: critical
  - claim: Kaggle code competitions mandate offline execution (Internet=False), 20GB disk, 16GB RAM, ~30h/week GPU quota, 9-12h session wall-clock timeout, and strict reproducible submission contracts.
    category: kaggle_constraints
    evidence: Official Kaggle documentation and competition rules (Tier T1/T2).
    confidence: high
    implementation_relevance: critical
  - claim: Over 90% of Kaggle private leaderboard shake-ups are caused by public-LB chasing, target leakage in preprocessing, or cross-validation failure where train/val splits fail to mirror test distribution.
    category: validation_risk
    evidence: Historical Kaggle Grandmaster writeups 2020-2026 (Tier T2/T5).
    confidence: high
    implementation_relevance: critical
  - claim: The 2,121-skill catalog contains zero Kaggle-specific competitive ML skills; core tabular ML skills (validation design, adversarial validation, GBDT suites, hill-climbing ensembling) must be created locally.
    category: skill_catalog
    evidence: Empirical catalog grep audit (Tier T4).
    confidence: high
    implementation_relevance: critical

architectural_recommendations:
  - recommendation: Hybrid Hierarchical Supervisor + Blackboard + 3 Critic-Generator Veto Gates (Candidate B: ~12 agents).
    rationale: Provides complete workflow coverage without the 2x token overhead of debating every step, while guarding high-stakes decision points (validation design, ensembling, submission).
    dependencies: Local filesystem structure, schema-validated YAML/JSON, Git worktrees.
    risks: Requires strict file lock or optimistic concurrency control on blackboard state files.
  - recommendation: Decoupled filesystem blackboard store (`state/` or `experiments/blackboard/`) replacing direct message-passing pipes.
    rationale: Protects against non-interactive CLI stdout truncation and prevents context window degradation across long agent runs.
    dependencies: Pydantic schemas and atomic write-then-rename filesystem operations.
    risks: None; significantly improves operational robustness.
  - recommendation: Sandboxed experiment execution using ephemeral Git worktrees (`git worktree add`).
    rationale: Enables parallel training experiments and zero corruption of the main Git repository branch.
    dependencies: Local Git repository.
    risks: Disk usage if worktrees are not cleanly pruned upon completion.

kaggle_recommendations:
  - recommendation: Adversarial validation (LightGBM train vs test classifier ROC-AUC) as an immutable requirement before modeling proceeds.
    expected_value: high (identifies covariate shift and leaking temporal/group IDs early).
    evidence: Kaggle winning solutions (Tier T2/T5).
  - recommendation: Validation Architect must hold unconditional VETO power over the research plane.
    expected_value: critical (prevents fatal data leakage and invalid CV schemes).
    evidence: Standard Grandmaster postmortem finding.
  - recommendation: Caruana Forward Ensemble Selection (Hill Climbing with replacement) on out-of-fold predictions with diversity correlation checks.
    expected_value: high (+0.5% to +2.0% metric improvement on held-out private LB).
    evidence: Caruana et al. (2004) and competitive ML consensus (Tier T3/T5).
  - recommendation: Offline self-contained submission bundling with pre-packaged dependency wheels and Kaggle Datasets.
    expected_value: critical (mandatory for code competitions).
    evidence: Kaggle rules (Tier T1/T2).

agent_recommendations:
  - role: Commander
    responsibility: Global strategic direction, phase transitions, budget allocation, stopping decisions; never touches code directly.
  - role: Competition Researcher (Scout)
    responsibility: Extract competition metadata, rules, metric mathematical formulation, and submission schema.
  - role: Data Forensics Agent
    responsibility: Schema analysis, data profiling, missingness, target distribution, anomaly detection.
  - role: EDA Specialist
    responsibility: Visualizations, correlation analysis, and multimodal feature profiling.
  - role: Validation Architect
    responsibility: Design leak-free CV scheme (Group, Stratified, TimeSeries); holds supreme VETO power.
  - role: Feature Engineer
    responsibility: Formulate hypothesis-driven feature transformations with pre-registered rationales.
  - role: Model Researcher
    responsibility: Propose diverse model families (GBDT, Neural Tabular, Linear baselines).
  - role: HPO Agent
    responsibility: Efficient hyperparameter optimization via Optuna TPE.
  - role: Trainer (Runner)
    responsibility: Physical training execution in Git worktrees, emitting metrics, OOF predictions, and artifacts.
  - role: Ensembler (Blender)
    responsibility: Hill-climbing ensemble selection, model diversity checks, OOF weight optimization.
  - role: Adversarial Reviewer
    responsibility: Unconditional VETO authority at Gate 1 (Validation Design), Gate 2 (Ensemble Selection), and Gate 3 (Final Submission).
  - role: Kaggle Executor
    responsibility: Kaggle CLI automation, kernel pushing, status polling, and LB score tracking.
  - role: Memory Curator
    responsibility: Maintain 4-layer memory hierarchy, synthesize postmortems, and promote heuristics.

memory_recommendations:
  - recommendation: Four-layer memory hierarchy: Working (ephemeral), Project (SQLite/YAML), Strategic (domain playbooks in markdown), and Meta (system self-reflection).
    rationale: Preserves provenance and separates verified empirical facts from weak speculation.
    dependencies: Local filesystem and SQLite.
    risks: Speculation creep if promotion thresholds are not strictly enforced.

hardware_compute_recommendations:
  - recommendation: Enforce 28.0-hour weekly GPU ceiling (saving 2h buffer) and 10.5-hour session watchdog before Kaggle's 12.0-hour hard kill.
  - recommendation: Avoid Pascal P100 GPUs due to missing modern CUDA sm_60 kernels in recent Kaggle base images; route to Dual T4 or L4.
```
