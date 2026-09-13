# Research Extraction: claude.md

```yaml
source:
  file: research/01-raw/claude.md
  model: Claude (Sonnet/Opus 4.6 Thinking tier analysis)
  date_if_available: 2026-09-13

claims:
  - claim: Antigravity CLI (agy) replaced Gemini CLI in mid-2026; supports Skills, Hooks, Subagents, and Plugins.
    category: platform_capability
    evidence: Official docs / product release notes (Tier T1/T4). Corroborated by live binary inspection.
    confidence: high
    implementation_relevance: critical
  - claim: Gemini 3.8 Flash High is an effort-level variant of Gemini 3.8 Flash designed for agentic tool use; strong at SE/tool calls but below frontier-reasoning tier (Gemini 3 Pro).
    category: model_capability
    evidence: Google DeepMind model card / cloud docs (Tier T1).
    confidence: high
    implementation_relevance: critical
  - claim: Headless non-interactive execution (`agy -p` / `--print`) has known bug reports regarding silent hangs or dropped stdout in non-TTY/subprocess execution.
    category: platform_risk
    evidence: GitHub issues on google-antigravity/antigravity-cli (Tier T4).
    confidence: medium_high
    implementation_relevance: critical (requires filesystem blackboard / sentinel files rather than relying solely on captured stdout).
  - claim: Supplied 2,121-skill catalog contains zero Kaggle-specific or tabular-ML competition skills (no XGBoost, LightGBM, CatBoost, AutoGluon).
    category: skill_catalog
    evidence: Grep audit of catalog.
    confidence: high
    implementation_relevance: critical (core ML competition skills must be implemented locally).
  - claim: Kaggle code competitions enforce ~30h/week GPU quota, session wall-clock limits (9-12h), internet disabled during scoring, and strict notebook re-run contracts.
    category: kaggle_constraints
    evidence: Kaggle platform documentation and rules (Tier T2).
    confidence: high
    implementation_relevance: high

architectural_recommendations:
  - recommendation: Hierarchical Commander + 5 bounded Cells + filesystem blackboard + high-authority Adversarial verification gate + serialized GPU execution.
    rationale: Prevents context overload on workhorse model, provides resilient state persistence against headless CLI failures, enforces budget discipline.
    dependencies: Local filesystem structure, schema-validated JSON/YAML.
    risks: Filesystem contention if locks are not handled; latency if polling intervals are poorly tuned.
  - recommendation: Blackboard communication architecture (state.json, run sentinels, artifact directories) rather than chat-message piping.
    rationale: Protects against subprocess stdout truncation and context loss across subagent boundaries.
    dependencies: blackboard/state.json schema.
    risks: Requires strict file schema validation.
  - recommendation: Categorical rejection of `*-delegate` skills and third-party model gateways.
    rationale: Prevents silent violation of the Antigravity + Gemini 3.8 Flash High platform constraint.
    dependencies: Rule enforcement in agent configuration.
    risks: None; strengthens compliance.

kaggle_recommendations:
  - recommendation: Pre-register competition format (Code Competition vs CSV submission) and internet constraints before writing code.
    expected_value: high (prevents submission failure in hidden evaluation).
    evidence: Kaggle competition rules standard practice.
  - recommendation: Adversarial validation (train vs test classifier AUC check) on every tabular problem before CV sign-off.
    expected_value: high (detects covariate shift and train/test leakage early).
    evidence: Standard Kaggle Grandmaster competitive practice.
  - recommendation: Pre-compute and store out-of-fold (OOF) predictions in parquet format for diversity and ensembling.
    expected_value: high (enables stacking and multi-model blend verification).
    evidence: Standard competitive ML requirement.

agent_recommendations:
  - role: Commander
    responsibility: High-level strategy, resource allocation, experiment priority, stop/go decisions; never touches code or raw submission files directly.
    justification: Prevents context pollution and maintains strategic focus.
  - role: Competition Researcher
    responsibility: Extract competition brief, evaluation metric, submission rules, timeline, external data allowances.
    justification: Grounding phase is mandatory for legal and technical compliance.
  - role: Data/EDA Agent
    responsibility: Schema analysis, missingness, cardinality, distributions, target relationship, dataset signature generation.
    justification: Combined Data Forensics and EDA avoids redundant data passes.
  - role: Validation Architect
    responsibility: Splitting strategy (group/stratified/temporal), target leakage detection, adversarial validation; possesses VETO power.
    justification: CV failure is the #1 cause of private leaderboard shakeout.
  - role: Feature Engineer
    responsibility: Hypothesis-driven feature creation, fold-safe encoding, interaction generation.
    justification: Features must be tied to pre-registered hypotheses, not random generation.
  - role: Model Researcher / HPO
    responsibility: Model family selection (XGB/LGBM/CatBoost/NN) and bounded hyperparameter tuning.
    justification: Combined role avoids tuning models that should be discarded.
  - role: Experiment Manager
    responsibility: Ledger maintenance, metadata enforcement, tracking CV/OOF/resources.
    justification: Atomic record of all trials prevents duplicate or unlearned experiments.
  - role: Ensemble Agent
    responsibility: Out-of-fold blend optimization, correlation matrix audit, diversity validation.
    justification: Ensembling must verify true diversity rather than blending correlated models.
  - role: Leakage Hunter + CV/Error Auditor
    responsibility: Adversarial cell probing feature leakage, temporal leakage, group leakage, systematic error subgroups; holds VETO power.
    justification: Independent failure-seeking agent catches issues builder agents ignore.
  - role: Kaggle Executor
    responsibility: Kaggle CLI automation, kernel push/pull, headless execution with sentinel monitoring.
    justification: Isolates environment and cloud execution mechanics.
  - role: Artifact Analyst
    responsibility: Parse outputs, verify parquet/csv artifacts, compute metrics, update ledger.
    justification: Guarantees standardized ingestion into memory.
  - role: Knowledge Curator
    responsibility: Cross-competition pattern extraction, meta-learning hypothesis maintenance, failure indexing.
    justification: Ensures learning across experiments and competitions without self-modifying orchestration.
  - role: Final Auditor
    responsibility: Mandatory pre-submission sanity check (diff against prior submission, schema check, compliance check); holds VETO power.
    justification: Prevents fatal submission errors.

memory_recommendations:
  - Layered directory structure: knowledge/ with subfolders for competitions, strategies, validation, features, models, ensembles, failures, successes, meta-learning, playbooks.
  - Atomic experiment files in YAML/JSON with schema validation.
  - Promotion discipline: weak_hypothesis -> strong_pattern (requires replication across >=2 seeds or >=2 competitions).
  - Explicit evidence_for and evidence_against tracking for all meta-learning hypotheses.

self_evolution_recommendations:
  - Failure-driven skill candidate creation: failure -> root cause -> pattern across >=3 occurrences -> draft skill -> adversarial review -> human sign-off -> A/B test -> promote/reject.
  - Agent trust levels (high, medium, low) tracked via performance metrics (success rate, false positives, wasted compute, private-LB delta); trust modifies secondary verification depth.
  - Orchestration self-rewriting is strictly prohibited; human approval required for structural changes.

compute_recommendations:
  - Explicit compute budget ledger (config/compute_budget.yaml).
  - Zero GPU allocation for CPU-bound stages (Research, EDA, feature hypothesis, leakage scan).
  - Serialized GPU training runs; reserve wall-clock GPU hours strictly for final submission notebook.
  - 25% overspend halts stage for Commander re-approval.

automation_recommendations:
  - Filesystem blackboard state (`blackboard/state.json`) with polling sentinels for async tasks.
  - Git worktrees for parallel experiment tracks.
  - Pre-submission sign-off file (`final_audit_pass.json`) required before Kaggle submission packager executes.

concerns:
  - Headless execution hangs/drops output in some agy versions.
  - Flash-tier model may fail at deep mathematical or architectural synthesis if prompt context is bloated.
  - Overfitting public leaderboard through excessive submissions.

contradictions:
  - Contradicts swarms/open-ended multi-agent debates (argues debate on identical models wastes compute).
  - Contradicts within-competition evolutionary agent populations (argues GPU limits make this prohibitive).
  - Contradicts automatic prompt/orchestration self-rewriting (requires human-in-the-loop promotion).
```
