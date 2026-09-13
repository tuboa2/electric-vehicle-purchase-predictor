# Research Extraction: deepseek.md

```yaml
source:
  file: research/01-raw/deepseek.md
  model: DeepSeek (Reasoning / R1-derived analysis)
  date_if_available: 2026-09-13

claims:
  - claim: Multi-agent systems outperform single agents on parallel tasks (+80%) but degrade significantly (39-70%) on sequential reasoning due to error propagation and coordination overhead.
    category: multi_agent_theory
    evidence: Openlayer / Multi-Agent Orchestration research (Tier T2/T3).
    confidence: high
    implementation_relevance: high
  - claim: Gemini 3.8 Flash High ranks #10 in coding (80.8) and #17 in agent tasks (66.6); exhibits diligence and high tool-calling iteration, but weaker multi-step unguided planning.
    category: model_capability
    evidence: ChatBench benchmark scores / DeepMind model release data (Tier T1).
    confidence: high
    implementation_relevance: critical (necessitates aggressive task decomposition and independent verification).
  - claim: Kaggle GPU quota is ~30h/week (T4x2 or P100) with 12h session limits; CPU is effectively unmetered; GPU offers zero acceleration for scikit-learn / standard pandas pipelines.
    category: kaggle_constraints
    evidence: Kaggle platform limits documentation (Tier T1).
    confidence: high
    implementation_relevance: high
  - claim: Polars zero-copy columnar execution and Parquet predicate pushdown dramatically reduce memory footprint (4-6 GB vs 16 GB pandas OOM).
    category: compute_optimization
    evidence: Benchmark comparisons Polars vs Pandas (Tier T4).
    confidence: high
    implementation_relevance: medium_high

architectural_recommendations:
  - recommendation: Hierarchical supervisor with graph-based state management and checkpointing.
    rationale: Provides explicit control flow, prevents coordination chaos of flat swarms, handles execution resumption.
    dependencies: State machine / graph orchestrator.
    risks: Complexity of graph transitions if state definitions become too tangled.
  - recommendation: Structured YAML communication artifacts with explicit confidence, findings, evidence citations, and risk fields.
    rationale: Eliminates free-form chatter and token waste; enables programmatic parsing.
    dependencies: Defined JSON/YAML schemas.
    risks: Schema validation errors if an agent outputs malformed fields.
  - recommendation: Separate framework core (`kaggle-agent-core/`) from competition project directory (`competition-project/`).
    rationale: Enables clean reusability across multiple competitions without code pollution.
    dependencies: Modular directory organization.
    risks: None; standard good software architecture.

kaggle_recommendations:
  - recommendation: Mandatory adversarial validation AUC check (threshold AUC < 0.60; investigate if >= 0.58).
    expected_value: high (catches train/test distribution shift before submission).
    evidence: Standard Kaggle Grandmaster best practice.
  - recommendation: Never retry a failed experiment without changing at least one variable (hypothesis, parameters, data version, validation).
    expected_value: high (prevents infinite error loops).
    evidence: Systematic debugging principle.
  - recommendation: Automated Kaggle kernel push/status/pull loop via CLI with explicit wait/status polling.
    expected_value: high (unattended execution).
    evidence: Kaggle CLI quick reference.

agent_recommendations:
  - role: Commander (Supervisor)
    responsibility: Strategic director, resource allocation, priority formula execution, stopping decisions.
    justification: Required to orchestrate sub-cells and maintain global objective.
  - role: Competition Researcher
    responsibility: Rules, metrics, submission contract, external data rules, past solutions.
    justification: Prevents disqualification and guides problem framing.
  - role: Data Forensics
    responsibility: Profiling, schema, missingness, distribution shift, leakage detection.
    justification: Early detection of data issues.
  - role: Validation Architect
    responsibility: CV strategy, fold design, leakage tests; holds VETO power.
    justification: CV must correlate with private leaderboard.
  - role: Feature Engineer
    responsibility: Hypothesis-driven feature creation (not blind enumeration).
    justification: Unfocused feature generation overfits.
  - role: Model Researcher
    responsibility: Model family selection based on problem type and data scale.
    justification: Matches right model class to data.
  - role: HPO Agent
    responsibility: Bounded Optuna hyperparameter optimization.
    justification: Efficient parameter search.
  - role: Ensemble Optimizer
    responsibility: OOF correlation analysis, hill-climbing weights, stacking.
    justification: Blending diverse models drives podium finishes.
  - role: Error Analyst
    responsibility: Residual structure, subgroup failure analysis, calibration.
    justification: Pinpoints where features or models fail.
  - role: Leakage Hunter
    responsibility: Target, temporal, and group contamination detection.
    justification: Protects against artificial CV inflation.
  - role: Kaggle Executor
    responsibility: Kernel preparation, push, status check, output retrieval.
    justification: Manages external execution.
  - role: Artifact Analyst
    responsibility: Metrics parsing, registry updates, OOF validation.
    justification: Guarantees standardized ingestion.
  - role: Memory Curator
    responsibility: Knowledge base maintenance, deduplication, stale detection.
    justification: Manages multi-competition memory.
  - role: Skill Evolver
    responsibility: Failure analysis -> skill candidate -> evaluation -> promotion.
    justification: Implements EvoSkill paradigm.
  - role: Final Auditor
    responsibility: Pre-submission verification; holds VETO power.
    justification: Final safety gate.

memory_recommendations:
  - 4 layers: Working (in-memory state), Project (`knowledge/competitions/{name}/`), Strategic (`knowledge/strategies/`), Meta (`knowledge/agent-performance/`).
  - Knowledge record schema with ID, statement, evidence count, confidence, counterexamples, provenance.
  - Promotion rules: Speculation (1 exp) -> Hypothesis (3 exp across >=2 comps) -> Validated Principle (5+ exp, <10% counterexamples).
  - Context assembly using top-k retrieval rather than full transcript dumping.

self_evolution_recommendations:
  - EvoSkill-inspired closed-loop pipeline: Failure -> Classification -> Root Cause -> Generalizable Pattern -> Skill Draft -> Adversarial Review -> Versioned Skill (v0.1) -> A/B Test -> Statistical Review -> Promotion or Rejection.
  - Agent performance tracking (task success rate, experiment quality %, useful discoveries, false positive rate, wasted compute, reproducibility).
  - Explicit rule: no agent self-modifies core orchestration without candidate benchmarking and human approval.

compute_recommendations:
  - Priority formula: `priority = (expected_score_gain * confidence * information_gain) / (compute_cost * risk)`.
  - Accelerator decision tree: CPU for data processing & GBDTs on <1M rows; GPU strictly for Neural Networks.
  - Polars + Parquet + Arrow for zero-copy streaming data processing.
  - Maximum 30 GPU-hours/week quota tracked rigidly.

automation_recommendations:
  - Full CLI integration: `kaggle kernels push -p ./notebooks -t 43200 --accelerator nvidiaTeslaT4`.
  - Git branching policy (main vs experiments/*) with milestone commits.

concerns:
  - Multi-agent coordination overhead can degrade sequential reasoning by 39-70%.
  - High tool-calling iteration of Gemini 3.8 Flash High can burn tokens if tasks are underspecified.
  - Public LB overfitting if CV is not strictly insulated.

contradictions:
  - Proposes 15 agents (slightly higher than Claude's 13, but categorizes them into clear tiers).
  - Mentions LangGraph as dominant 2026 pattern, but recommends Antigravity native graph / custom agents rather than external framework dependency.
```
