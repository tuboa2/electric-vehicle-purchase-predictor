# Research Extraction: julius.md

```yaml
source:
  file: research/01-raw/julius.md
  model: Julius (Empirical Data Science Specialist)
  date_if_available: 2026-09-13

claims:
  - claim: Smallest effective production roster is 9 core roles (Commander, Competition Intelligence, Data Forensics, Validation Architect, Experiment Lead, Model/Feature Specialist, Artifact Analyst, Adversarial Auditor, Memory Curator) with short-lived dynamic workers.
    category: agent_roster
    evidence: Empirical multi-agent overhead analysis; diminishing returns of specialized roles.
    confidence: high
    implementation_relevance: critical
  - claim: Strict bifurcation of Control Plane (Antigravity agents, prompts, policies, routing) vs Data Plane (deterministic Python, CV, models, artifacts, Kaggle CLI).
    category: system_architecture
    evidence: Scientific software engineering principles; LLMs cannot be trusted to compute metrics.
    confidence: high
    implementation_relevance: critical
  - claim: Complex infrastructure (Vector databases, Kubernetes, distributed message brokers, large graph databases) is unnecessary and harmful overhead for single-engineer competition loops; SQLite / JSONL with content hashing is superior.
    category: architectural_simplicity
    evidence: Operational maintenance cost and failure rates of heavy infra in ML workflows.
    confidence: high
    implementation_relevance: critical
  - claim: Unconstrained multi-agent debate is ineffective when participants share the same underlying model and reasoning biases; diversity and intrinsic reasoning quality drive debate gains, not group size.
    category: multi_agent_theory
    evidence: Empirical peer-reviewed literature on multi-agent debate (Tier T3).
    confidence: high
    implementation_relevance: high

architectural_recommendations:
  - recommendation: Hierarchical Planner-Executor-Verifier with durable filesystem blackboard and SQLite/JSONL evidence store.
    rationale: Provides clear ownership, prevents state loss, decouples agent lifecycle from execution lifespan.
    dependencies: Filesystem, SQLite/JSONL, schema validators.
    risks: Filesystem access speed if artifact directories become massive (mitigated by Parquet and partition indexing).
  - recommendation: Task Envelope and Result Envelope schemas for all agent transitions.
    rationale: Enforces pre-conditions, input files, resource budgets, and explicit stop conditions before any worker executes.
    dependencies: Schemas for tasks and results.
    risks: Overhead of creating task files (negligible compared to compute).
  - recommendation: Defer speculative components (Vector DB, Kubernetes, graph databases, microservices) until empirical evidence proves simpler tools are limiting.
    rationale: Prevents premature overengineering.
    dependencies: None.
    risks: None; reduces failure surface.

kaggle_recommendations:
  - recommendation: Validation strategy must pass explicit leakage tests (temporal embargo, group isolation, fold-specific preprocessing) before modeling begins; Validation Architect holds hard veto.
    expected_value: critical (insulates against private LB shakeout).
    evidence: Kaggle competitive track record and scikit-learn leakage documentation.
  - recommendation: Modality router defines default model candidate families: Tabular (CatBoost, LightGBM, XGBoost, HistGradientBoosting, TabM); Vision (DINOv2, CNN/ViT); NLP (DeBERTa-v3, fine-tuned LLMs); Time-series (lag/rolling features, purged splits).
    expected_value: high (matches architecture to problem modality).
    evidence: Kaggle Grandmaster winning solutions.
  - recommendation: Model library retaining all valid OOF predictions from all trials to fuel post-processing and diversity-weighted ensembles.
    expected_value: high (harvests value from past failed/suboptimal runs in final stacking).
    evidence: Automated ensemble selection literature.

agent_recommendations:
  - role: Commander
    responsibility: Strategy, phase transitions, resource allocation, final submission authorization.
    justification: Core strategic authority.
  - role: Competition Intelligence
    responsibility: Rules, metrics, submission contracts, external data constraints, deadlines.
    justification: Grounding and compliance.
  - role: Data Forensics
    responsibility: Schema, missingness, duplicates, cardinality, drift, modality identification.
    justification: Data sanity before modeling.
  - role: Validation Architect
    responsibility: Design validation splits, audit fold boundaries, test leakage; holds hard VETO power.
    justification: Guards the primary truth mechanism (CV).
  - role: Experiment Lead
    responsibility: Converts hypotheses into immutable task specifications, runs trials, tracks budgets.
    justification: Experimentation execution manager.
  - role: Model/Feature Specialist
    responsibility: Generates hypothesis-driven feature views and selects model candidate families.
    justification: Specialized modeling expertise.
  - role: Artifact Analyst
    responsibility: Validates outputs against artifact contract, computes CV/OOF metrics, performs error analysis.
    justification: Standardized ingestion into memory.
  - role: Adversarial Auditor
    responsibility: Attacks leakage, rule compliance, reproducibility, and private-LB robustness; holds VETO power.
    justification: Independent failure-seeker.
  - role: Memory Curator
    responsibility: Extracts compact evidence records, manages SQLite/JSONL memory, detects contradictions and stale facts.
    justification: Cross-competition learning curator.

memory_recommendations:
  - Layered memory: Working (current run), Project (competition facts & experiments), Strategic (cross-competition patterns), Meta (agent and scheduler performance).
  - Storage: SQLite and JSONL with content hashing (no vector DB initially).
  - Structured queries based on modality, metric, cardinality, and evidence level before semantic search.
  - 5-tier confidence tags: observed_fact, strong_empirical_pattern, engineering_practice, weak_hypothesis, speculation.

self_evolution_recommendations:
  - Self-reflection without external execution grounding degenerates into repeated error loops. All critique must be tied to physical artifacts.
  - Skill evolution: failure cluster -> root cause -> skill candidate -> draft + tests -> sandbox benchmark -> adversarial review -> A/B test -> promote/reject.
  - Core orchestration, permission policies, and routing logic are immutable to autonomous agents; changes require human review.

compute_recommendations:
  - Priority formula: `priority = (expected_gain * confidence * information_gain * reversibility) / (compute_cost * risk * coordination_cost)`.
  - Phase-based compute budget shares: 10% Forensics/Validation, 10% Baselines, 35% Features/Models, 20% HPO, 15% Ensembles, 10% Final Audit.
  - Profile CPU/RAM/GPU before execution; avoid GPU allocation for pandas/scikit-learn workflows.

automation_recommendations:
  - Kaggle CLI kernel push/status/output pipeline.
  - Git worktrees for experiment isolation.
  - Headless Antigravity execution (`agy -p`) with structured JSON output and scoped permissions.

concerns:
  - Premature infrastructure bloat (vector DBs, message brokers) causing operational paralysis.
  - Public LB overfitting if teams treat public scores as ground truth.
  - Multi-agent debate wasting tokens with no diversity gain.

contradictions:
  - Recommends 9 core roles (leaner than Gemini's 19 or Claude's 13), handling other tasks via short-lived ephemeral workers.
  - Rejects vector databases in Phase 1 in favor of SQLite/JSONL + metadata filtering, contradicting papers suggesting mandatory RAG vector stores.
```
