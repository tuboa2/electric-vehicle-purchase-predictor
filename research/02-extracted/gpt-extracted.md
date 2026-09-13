# Research Extraction: gpt.md

```yaml
source:
  file: research/01-raw/gpt.md
  model: GPT-4o / o-series architecture synthesis
  date_if_available: 2026-09-13

claims:
  - claim: Hierarchical Blackboard + Planner/Executor/Verifier + Dynamic Experiment Workers decisively beats unstructured swarms.
    category: multi_agent_theory
    evidence: Multi-agent coordination overhead analysis; empirical failure modes of conversational swarms.
    confidence: high
    implementation_relevance: critical
  - claim: Agents must not own truth. LLMs propose hypotheses/code; deterministic Python code computes metrics/CV/hashes; Kaggle provides execution truth; Git guarantees lineage.
    category: system_integrity
    evidence: Foundation of scientific computing; prevents LLM self-delusion and hallucinated metrics.
    confidence: high
    implementation_relevance: critical
  - claim: Antigravity native subagents have independent contexts, concurrent execution, isolated workspace branching (Git worktrees), but enforce strict nesting depth limits.
    category: platform_capability
    evidence: Official Antigravity documentation.
    confidence: high
    implementation_relevance: high
  - claim: 12-agent roster across 5 planes (Control, Evidence, Research, Experiment, Execution) achieves complete coverage without redundancy.
    category: agent_roster
    evidence: Structural role decomposition matching the competition lifecycle.
    confidence: high
    implementation_relevance: high
  - claim: Experiment classes must be tiered: Tier A (Structural/CV), Tier B (High-value model families), Tier C (Optimization/HPO), Tier D (Ensembles), Tier E (Micro-tuning).
    category: experiment_design
    evidence: Kaggle competitive strategy; 80/20 rule of ML gains.
    confidence: high
    implementation_relevance: high

architectural_recommendations:
  - recommendation: Blackboard architecture with strict typed schemas (`blackboard/state.json`, `blackboard/board.yaml`).
    rationale: Decouples agents from fragile conversational history and handles asynchronous execution.
    dependencies: Filesystem storage, JSON/YAML schemas.
    risks: File lock contention if concurrency is uncoordinated.
  - recommendation: Deterministic separation of concerns: Python decides metrics/CV/hashes, Kaggle decides runtime/leaderboard, Git decides lineage, Gemini decides strategy/hypotheses.
    rationale: Prevents LLM-as-a-judge hallucinations.
    dependencies: Local Python scripts and verification test harnesses.
    risks: None; core principle of scientific computing.
  - recommendation: Shallow hierarchy (Commander -> Domain Leads / Workers) to respect Antigravity nesting limits.
    rationale: Avoids deep recursion and context bloat.
    dependencies: Orchestration state machine.
    risks: None.

kaggle_recommendations:
  - recommendation: Tiered experiment execution policy: prioritize Tier A (CV setup) and Tier B (model families) before burning compute on Tier C (HPO).
    expected_value: high (maximizes information gained per GPU-hour).
    evidence: Diminishing returns of hyperparameter tuning on suboptimal models.
  - recommendation: Separate hardware abstraction into CPU workloads (Polars feature pipelines, data forensics) and GPU workloads (T4/P100 for NNs and large GBDTs).
    expected_value: high (preserves 30h weekly GPU quota).
    evidence: Kaggle hardware allocation guidelines.
  - recommendation: Mandatory pre-submission validation and compliance veto gate (`final_audit_pass.json`).
    expected_value: critical (prevents invalid submissions and wasted daily submission counts).
    evidence: Standard Kaggle competition rules.

agent_recommendations:
  - role: Commander
    responsibility: Strategic control plane, goal definition, budget management, stopping criteria.
    justification: Keeps system aligned on global competition objectives.
  - role: Competition Intelligence
    responsibility: Rules extraction, evaluation metrics, submission contracts, external data constraints.
    justification: Grounding and compliance.
  - role: Data Forensics
    responsibility: Data auditing, schema parsing, missingness, drift analysis.
    justification: First line of defense against data anomalies.
  - role: Validation Architect
    responsibility: CV split strategy, leakage detection, adversarial validation; holds VETO power.
    justification: Guarantees CV aligns with private LB.
  - role: Leakage Compliance
    responsibility: Probes feature generation fold-safety, external data leakage, temporal leaks; holds VETO power.
    justification: Prevents disqualification and catastrophic shakeout.
  - role: Model Strategist
    responsibility: Selects diverse model families (GBDTs, Tabular NNs, Transformers).
    justification: Prevents premature convergence on a single model family.
  - role: Feature Strategist
    responsibility: Hypothesis-driven feature creation and transformation pipelines.
    justification: Focuses feature engineering on domain hypotheses.
  - role: Experiment Manager
    responsibility: Maintains atomic experiment records, evaluates outcomes, enforces metadata schemas.
    justification: Core ledger of empirical results.
  - role: Ensemble Strategist
    responsibility: OOF correlation analysis, Hill Climbing blend weights, stacking.
    justification: Extracts maximum ensemble diversity.
  - role: Error Analyst
    responsibility: Inspects residuals, misclassified samples, and subgroup failures to generate new hypotheses.
    justification: Directs future feature engineering based on error patterns.
  - role: Kaggle Executor
    responsibility: Packages code, pushes notebooks via Kaggle CLI, monitors remote execution, pulls outputs.
    justification: Isolates remote execution mechanics.
  - role: Artifact Memory
    responsibility: Validates artifacts against contract, updates memory layers, extracts generalizable knowledge.
    justification: Ingests physical artifacts into persistent storage.

memory_recommendations:
  - 4 layers: Working (in-memory state), Project (`knowledge/competitions/`), Strategic (`knowledge/strategies/`), Meta (`knowledge/meta/`).
  - Strict evidence-weighting: Observed Fact, Strong Empirical Pattern, Weak Hypothesis, Speculation.
  - Meta-Kaggle knowledge graph mapping dataset signatures to successful modeling strategies.

self_evolution_recommendations:
  - Skill and agent evolution driven strictly by empirical failure and regression tests.
  - Never allow agents to rewrite their own core prompts or orchestration logic without benchmark verification.
  - A/B testing of skill modifications against synthetic benchmark tasks.

compute_recommendations:
  - Strict compute ledger allocating GPU-hours per experiment tier.
  - Immediate termination of experiments that exceed resource limits (RAM/time).
  - Use Polars and Parquet for all intermediate feature caches.

automation_recommendations:
  - Automated Kaggle CLI execution (`kaggle kernels push`, `kaggle kernels output`).
  - Git worktrees for parallel experiment isolation.
  - Automated retry with exponential backoff on transient network/Kaggle CLI errors.

concerns:
  - Excessive agents can dilute focus unless invoked strictly on demand.
  - Model hallucinations regarding benchmark scores if metrics are not parsed from disk artifacts.

contradictions:
  - Proposes 12 agents (slight variation from Claude's 13 and Gemini's 19, but cleanly maps into 5 functional planes).
  - Explicitly rejects text-based debate and LLM-as-judge as authoritative verification.
```
