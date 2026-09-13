# Research Extraction: grok.md

```yaml
source:
  file: research/01-raw/grok.md
  model: Grok (xAI architecture review)
  date_if_available: 2026-09-13

claims:
  - claim: Marginal intelligence gain per additional agent drops sharply once core roles are covered; extra agents mainly increase coordination tax, context pollution, and failure surface.
    category: multi_agent_theory
    evidence: Analysis of multi-agent dynamics under single mid-tier reasoning model (Tier T3).
    confidence: high
    implementation_relevance: critical
  - claim: Antigravity CLI natively supports asynchronous subagents, effort levels, workspace/global skills, plugins, and headless `-p` execution.
    category: platform_capability
    evidence: Primary Antigravity CLI documentation (Tier T1).
    confidence: high
    implementation_relevance: high
  - claim: Generic AutoML rarely wins Kaggle competitions alone; competitive advantage stems from validation design, combinatorial feature engineering, diverse ensembling, and leakage elimination.
    category: kaggle_technique
    evidence: 2025-2026 Kaggle Grandmaster writeups (Tier T5).
    confidence: high
    implementation_relevance: high
  - claim: Quotas and hardware limits are volatile and competition-dependent; system must query live capabilities rather than hardcode limits.
    category: kaggle_constraints
    evidence: Kaggle platform policy updates.
    confidence: high
    implementation_relevance: high

architectural_recommendations:
  - recommendation: Hybrid Hierarchical Planner-Executor-Verifier (PEV) with asynchronous subagents, adversarial verification, and artifact memory.
    rationale: Directly matches Antigravity primitives, minimizes coordination overhead, enforces verification before promotion.
    dependencies: Antigravity subagent dispatch, local filesystem.
    risks: Complexity if verification gates are not automated.
  - recommendation: Dynamic capability probing (`refresh_capabilities.py` writing `capabilities/kaggle_current.yaml`).
    rationale: Prevents stale assumptions about quotas, GPU shapes, and CLI paths.
    dependencies: Automated probing scripts.
    risks: None; standard defensive engineering.
  - recommendation: Modular split: `kaggle-agent-core/` (reusable engine) vs `competition-<slug>/` (per-competition runtime).
    rationale: Prevents cross-competition code entanglement.
    dependencies: Clean directory design.
    risks: None.

kaggle_recommendations:
  - recommendation: Validation strategy must answer "Why should this CV estimate correlate with private LB?" before any modeling proceeds.
    expected_value: critical (prevents private leaderboard shakeout).
    evidence: Kaggle competitive history.
  - recommendation: Combinatorial feature engineering and feature interaction prioritization over exhaustive HPO sweeps.
    expected_value: high (feature engineering yields higher ROI than tuning suboptimal models).
    evidence: Kaggle GM winning solution patterns.
  - recommendation: Hill-climbing on Out-of-Fold (OOF) predictions to determine blend weights; reject correlated models.
    expected_value: high (maximizes ensemble gain).
    evidence: Standard post-processing technique.

agent_recommendations:
  - role: Commander
    responsibility: Strategic director, EV scheduling, resource allocation, stopping rules, final synthesis.
    justification: Core orchestrator.
  - role: Competition Researcher
    responsibility: Rules, metrics, data constraints, timelines, baseline discovery.
    justification: Grounding and compliance.
  - role: Data Forensics + EDA
    responsibility: Schema, quality, leakage signals, distributions, train/test shift.
    justification: Early data sanity.
  - role: Validation Architect
    responsibility: CV strategy, fold design, leakage tests; holds VETO power.
    justification: Decisive role for private LB generalization.
  - role: Feature / Model Researcher
    responsibility: Hypothesis-driven feature creation and model family selection.
    justification: Guided exploration.
  - role: Experiment Manager
    responsibility: Registry, prioritization, trial isolation, artifact validation.
    justification: Central ledger of empirical work.
  - role: Trainer / HPO
    responsibility: Parallel isolated model runs in git worktrees.
    justification: Model execution engine.
  - role: Ensemble Agent
    responsibility: Correlation analysis, diversity check, hill-climbing, stacking.
    justification: Post-processing and ensembling.
  - role: Adversarial Reviewer / Leakage Hunter / Error Analyst
    responsibility: Red-team attacks, residual structure analysis; holds VETO power.
    justification: Verification before submission.
  - role: Kaggle Executor
    responsibility: Kernel packaging, CLI push, status polling, output ingestion.
    justification: Kaggle interaction layer.
  - role: Artifact Analyst + Knowledge Curator
    responsibility: Ingest metrics, update memory, promote validated findings.
    justification: Translates runs into persistent knowledge.
  - role: Compliance / Final Auditor
    responsibility: Rules audit, pre-submission checklist; holds VETO power.
    justification: Final safety check.
  - role: Performance Engineer (on-demand)
    responsibility: Polars optimization, memory profiling, serialization bottlenecks.
    justification: Performance tuning when needed.

memory_recommendations:
  - 4 layers: Working, Project, Strategic, Meta.
  - Structured YAML/JSON with explicit evidence counts, confidence, counterexamples, and last_validated timestamps.
  - Retrieval-augmented context assembly (top-k semantic/keyword) rather than transcript dumping.

self_evolution_recommendations:
  - Failure-driven skill candidate pipeline: failure -> root cause -> generalizable pattern -> draft skill -> test cases -> adversarial review -> versioned candidate -> A/B test -> promote or reject.
  - Controlled evolution: no agent modifies core orchestration without benchmark verification.
  - Agent performance tracking (success rate, wasted compute, false positive rate).

compute_recommendations:
  - Expected Value formula: `priority = (expected_gain * confidence * information_gain) / (compute_cost * risk)`.
  - Explicit stopping rules: plateau, low remaining EV, compute budget exhaustion.
  - Polars/Arrow/Parquet data stack; reserve GPU strictly for deep learning and heavy boosting.

automation_recommendations:
  - Automated Kaggle CLI push/pull loop.
  - Git worktrees for parallel experiment isolation.
  - Scripted headless invocation via `agy -p`.

concerns:
  - Overfitting to public LB through repeated submissions.
  - Risk of unvalidated skill proliferation if evolution is uncontrolled.
  - Kaggle quota exhaustion from runaway parallel tasks.

contradictions:
  - Proposes 14 roles (combines several functions to keep roster under 15).
  - Explicitly rejects pure blackboard complexity as "overkill" if treated as a dynamic shared message bus, but endorses structured artifact memory on the filesystem.
```
