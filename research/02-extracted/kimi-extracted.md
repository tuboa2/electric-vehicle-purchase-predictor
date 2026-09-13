# Research Extraction: kimi.md

```yaml
source:
  file: research/01-raw/kimi.md
  model: Kimi (Moonshot AI architecture synthesis)
  date_if_available: 2026-09-13

claims:
  - claim: Single agents match or beat multi-agent systems on ~64% of general tasks; multi-agent coordination degrades sequential reasoning by 39-70%; single-model debate causes sycophancy cascades rather than truth.
    category: multi_agent_theory
    evidence: Google / Openlayer multi-agent synthesis papers (Tier T3).
    confidence: high
    implementation_relevance: critical (limits multi-agent sprawl; restricts debate).
  - claim: Parallelism is proven to help (+80%) only on genuinely independent parallel subtasks (such as concurrent ML experiments in isolated worktrees).
    category: multi_agent_theory
    evidence: Supervisor fan-out benchmarks (Tier T3).
    confidence: high
    implementation_relevance: critical
  - claim: Antigravity CLI supports headless `--output-format text|json|stream-json`, `--json-schema` enforcement, and fails loudly with non-zero exit on invalid model slugs.
    category: platform_capability
    evidence: Primary Antigravity CLI documentation (Tier T1).
    confidence: high
    implementation_relevance: high
  - claim: Kaggle P100 GPU image has a known pitfall: default PyTorch (cu128) lacks Pascal sm_60 kernels (`torch.cuda.is_available()` is True but first CUDA op fails). Avoid P100 unless rebuilt; prefer T4 or L4.
    category: kaggle_hardware_pitfall
    evidence: Kaggle user discussions & technical writeups (Tier T4).
    confidence: medium_high
    implementation_relevance: high

architectural_recommendations:
  - recommendation: Two-tier hierarchical supervisor (Commander -> 9 specialized workers) with file-based blackboard (`state/`) and git worktree isolation.
    rationale: Minimizes coordination overhead, eliminates shared mutable state collisions, matches verified empirical multi-agent strengths.
    dependencies: Filesystem storage, atomic writes (write to tmp + rename).
    risks: Filesystem race conditions if write-rename pattern is omitted.
  - recommendation: Gate G0-G7 phase-gated execution pipeline with hard veto points at G2 (Validation design) and G6 (Adversarial audit).
    rationale: Pre-conditions and post-conditions must pass before expensive modeling or submission proceeds.
    dependencies: Phase-gate state machine.
    risks: Pipeline stalls if validation fails repeatedly (mitigated by explicit remediation instructions).
  - recommendation: Refreshable `capabilities/` layer (`capabilities/kaggle_quotas.md`, `capabilities/antigravity_cli.md`) refreshed at initialization.
    rationale: Quotas, accelerator IDs, and CLI flags drift over time.
    dependencies: Probing scripts.
    risks: None.

kaggle_recommendations:
  - recommendation: Mandatory adversarial validation AUC check (target AUC <= 0.55 clean, 0.55-0.70 investigate, > 0.70 redesign CV or match distribution).
    expected_value: high (prevents private leaderboard collapse).
    evidence: Standard competitive ML practice (fastai lineage).
  - recommendation: Diverse ensembles (blending GBDTs, Tabular NNs, and linear baselines) with Hill-Climbing on OOF predictions; multi-seed ensembling.
    expected_value: high (uncorrelated error distributions boost ensemble metric).
    evidence: NVIDIA Grandmasters' Playbook / winning solutions.
  - recommendation: Multi-round pseudo-labeling with soft labels and k-fold split isolation to avoid leakage.
    expected_value: medium_high (effective when test set is large).
    evidence: BirdCLEF winning solutions.

agent_recommendations:
  - role: Commander
    responsibility: Strategic supervisor, phase gate enforcement, EV scheduling, budget accounting, stopping decisions.
    justification: Top-tier orchestrator.
  - role: Scout (Competition Intelligence)
    responsibility: Parse rules, metrics, submission contracts, external data licensing, past art.
    justification: Grounding and compliance.
  - role: Forensic (Data Forensics & EDA)
    responsibility: Modality detection, schema, missingness, duplicates, target distribution, drift report.
    justification: Early data sanity.
  - role: Validator (Validation Architect)
    responsibility: Split strategy design, fold integrity checks, adversarial validation; holds hard VETO power.
    justification: Guarantees CV correlates with private LB.
  - role: Engineer (Feature & Model Strategist)
    responsibility: Hypothesis-driven feature creation, model family selection, HPO parameter bounds.
    justification: Structured ML exploration.
  - role: Runner
    responsibility: Executes trials in isolated worktrees, enforces deterministic seeds, emits standardized artifacts.
    justification: Deterministic execution agent.
  - role: Blender (Ensemble Optimizer)
    responsibility: OOF correlation analysis, Hill Climbing blend weights, stacking, pseudo-labeling guardrails.
    justification: Post-processing and ensembling.
  - role: Adversary (Adversarial Reviewer)
    responsibility: Leakage audits, error analysis, license check, pre-submission checklist; holds hard VETO power.
    justification: Independent failure-seeking gate.
  - role: Executor (Kaggle Executor)
    responsibility: Prepares kernels, executes `kaggle kernels push/output`, monitors status, submits.
    justification: Cloud execution interface.
  - role: Chronicler (Artifact & Knowledge Curator)
    responsibility: Ingests artifacts, validates schemas, updates registry, compresses lessons into strategic memory.
    justification: Memory management and self-evolution.

memory_recommendations:
  - 4 layers: Working (`state/session.yaml`), Project (`knowledge/`), Strategic (`~/.gemini/antigravity-cli/skills/kaggle-os-memory/knowledge/`), Meta (`knowledge/meta-learning/`).
  - Strict evidence ranking: Fact (verified in current comp), Pattern (replicated >=2 comps), Hypothesis (unreplicated), Speculation.
  - File-based blackboard with atomic write (tmp + rename) and lock discipline.

self_evolution_recommendations:
  - Controlled evolution without self-modifying orchestration: Chronicler proposes prompt/skill updates, Commander evaluates on benchmark case suite (e.g. 20 leakage traps), human approves high-risk updates.
  - Cross-competition prior transfer: retrieve historical competitions by modality and signature to initialize priors for the EV scheduler.

compute_recommendations:
  - Expected Value formula: `priority = (expected_gain * confidence * information_gain) / (compute_cost * (1 + risk))`.
  - Polars lazy frames + Parquet/Arrow caches for feature generation.
  - Strict 24h/week GPU cap (reserving 20% of the 30h quota for final submission).

automation_recommendations:
  - Headless script wrapper (`agy_call.py`) utilizing `--json-schema` and `--output-format stream-json`.
  - Automated Kaggle CLI execution (`kaggle kernels push`, `kaggle kernels output`).
  - Git worktree isolation for parallel experiments.

concerns:
  - Custom subagent model-pinning may not be supported in some CLI versions.
  - Sycophancy in LLM-to-LLM debates when models share weights.
  - Hidden test set scoring failures in Kaggle code competitions.

contradictions:
  - Recommends exactly 10 agents (fits between Julius's 9 and Claude's 13).
  - Explicitly rejects debate loops (replaces with one-shot critic pass).
```
