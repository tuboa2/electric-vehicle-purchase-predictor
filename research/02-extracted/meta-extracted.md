# Research Extraction: meta.md

```yaml
source:
  file: research/01-raw/meta.md
  model: Meta (Llama 3 / Meta AI research perspective)
  date_if_available: 2026-09-13

claims:
  - claim: More agents != better; coordination overhead scales quadratically O(n^2); best expected LB performance under Gemini 3.8 Flash High is achieved by a 6+1 architecture (Meta-Supervisor + 5 specialists + 1 Critic + 1 Memory Curator).
    category: multi_agent_theory
    evidence: Empirical multi-agent overhead research (Tier T3).
    confidence: high
    implementation_relevance: high
  - claim: Stacking and greedy ensemble selection over OOF arrays (Hill Climbing / Ridge-weighted) consistently dominate recent competitive Kaggle solutions.
    category: kaggle_technique
    evidence: April 2025 - 2026 Kaggle Playground 1st/2nd place writeups (Tier T5).
    confidence: high
    implementation_relevance: high
  - claim: Automated feature generation with OpenFE provides +1.9% average accuracy over base across 49 tabular datasets.
    category: feature_engineering
    evidence: OpenFE benchmark literature (Tier T3).
    confidence: medium_high
    implementation_relevance: medium
  - claim: P100 GPU fails with default Kaggle image PyTorch (cu128) due to missing sm_60 kernels; system should automatically prioritize T4 or L4.
    category: kaggle_hardware_pitfall
    evidence: Kaggle platform issue logs (Tier T4).
    confidence: high
    implementation_relevance: high

architectural_recommendations:
  - recommendation: 6+1 architecture (Meta-Supervisor + 5 specialists + 1 Critic + 1 Memory Curator) communicating through a numbered file blackboard (`.agents/blackboard/00_competition.json`, `01_data_profile.json`, etc.).
    rationale: Prevents coordination tax and context bloat while maintaining structured pipeline transitions.
    dependencies: Filesystem blackboard directory.
    risks: If specialists are overly broad, individual tasks may suffer from lack of specialization.
  - recommendation: Critic-Verifier gate performing independent verification and voting to accept/reject experiments before promotion.
    rationale: Prevents premature promotion of overfit or leaked solutions.
    dependencies: Verification scripts.
    risks: False rejections if critic criteria are overly rigid.
  - recommendation: Explicit refreshable capabilities layer (`capabilities/kaggle_compute.json`, `capabilities/gemini_model.json`).
    rationale: Isolates volatile platform quotas and pricing from core code.
    dependencies: Scripted probes.
    risks: None.

kaggle_recommendations:
  - recommendation: Day-1 Adversarial validation (train classifier on train vs test; AUC > 0.58 indicates significant distribution shift).
    expected_value: high (early warning of covariate shift).
    evidence: Fastai / Kaggle Grandmaster standard methodology.
  - recommendation: Pseudo-labeling with conservative thresholds (e.g. p >= 0.98 or p <= 0.02) and 0.5 sample weighting to prevent label corruption.
    expected_value: medium_high (effective when unlabeled test set is large).
    evidence: ISU x OSF 2nd place solution.
  - recommendation: Greedy Hill Climbing over OOF prediction arrays with max cap of 15 models and Ridge regularization.
    expected_value: high (stable multi-model blending).
    evidence: Kaggle ensemble winning writeups.

agent_recommendations:
  - role: Meta-Supervisor
    responsibility: Control unit, task selection from blackboard, 30-min heartbeat, stopping conditions.
    justification: Centralized coordinator.
  - role: Competition-Intel
    responsibility: Parse overview, rules, evaluation metric, external data policy via web.
    justification: Grounding and compliance.
  - role: Data-Detective
    responsibility: Adversarial validation, leakage scanning, temporal analysis, distribution profiling.
    justification: Data sanity before modeling.
  - role: Feature-Architect
    responsibility: Feature engineering (OpenFE/FeatureTools), fold design locking.
    justification: Feature generation.
  - role: Model-Strategist
    responsibility: GBDT/NN selection, HPO (Optuna), AutoGluon stacking, Hill Climbing.
    justification: Model exploration and ensembling.
  - role: Execution-Engineer
    responsibility: Kaggle CLI automation, kernel push, DVC/MLflow tracking, quota management.
    justification: Execution interface.
  - role: Critic-Verifier
    responsibility: Adversarial verification, OOF-LB correlation audit, shake-up avoidance.
    justification: Independent verification gate.
  - role: Memory-Curator
    responsibility: Background knowledge extraction, cross-competition transfer, temporal knowledge graph.
    justification: Persistent learning.

memory_recommendations:
  - 3-tier memory: Working (`core.md`), Episodic (project-scoped SQLite/DVC), Strategic (cross-competition temporal KG).
  - Context compression via LLMLingua / LongLLMLingua (20x token reduction) for historical logs.
  - Causal chains in memory: ProblemType -> DatasetChar -> Validation -> FeatureStrategy -> ModelFamily -> Outcome.

self_evolution_recommendations:
  - Skill discovery: Scan experiment ledger after competitions for repeated winning configurations, generate new SKILL.md.
  - A/B benchmark evaluation of skill changes before promotion to global directory.
  - 9-code failure taxonomy for root-cause diagnosis.

compute_recommendations:
  - Expected Information Gain (EIG) scheduler formula: `EIG = (expected_gain) / (compute_cost * failure_risk)`.
  - Multi-fidelity experimentation: 10% data / 1 fold first, promote to full training only if promising.
  - Reserve 20% of weekly GPU quota for final submissions.

automation_recommendations:
  - Kaggle CLI kernel push with internet disabled for code competitions.
  - Git worktrees for parallel experiment isolation.
  - Scripted headless execution via `agy --headless` or `agy -p`.

concerns:
  - Proposing heavy dependencies (Graphiti, Neo4j, FalkorDB) adds operational friction.
  - Token consumption of Gemini 3.8 Flash High at high reasoning effort.
  - Kaggle P100 PyTorch CUDA failure.

contradictions:
  - Proposes a 7-agent (6+1) roster, the leanest among all research documents.
  - Recommends heavy graph databases (Neo4j/Graphiti) for memory, which contradicts Julius's strong recommendation to stick to simple SQLite/JSONL.
```
