# Research Extraction: mistral.md

```yaml
source:
  file: research/01-raw/mistral.md
  model: Mistral (Large / Le Chat enterprise research perspective)
  date_if_available: 2026-09-13

claims:
  - claim: Hierarchical supervisor-worker architectures dominate production agent systems (70% of enterprise deployments); supervisor patterns boost parallel tasks by 80% but degrade sequential reasoning by 70%.
    category: multi_agent_theory
    evidence: Enterprise multi-agent deployment survey (Tier T2/T3).
    confidence: high
    implementation_relevance: critical
  - claim: "More agents = better" is empirically false; marginal intelligence gains diminish rapidly past core roles while coordination overhead and failure surface increase.
    category: multi_agent_theory
    evidence: Agent architecture taxonomy and benchmark studies (Tier T3).
    confidence: high
    implementation_relevance: critical
  - claim: Feature engineering is the #1 performance lever in competitive ML; GBDT families (XGBoost, LightGBM, CatBoost) win essentially every tabular competition; stacking diverse models provides the final podium edge.
    category: kaggle_technique
    evidence: Tabular SOTA and Kaggle Grandmaster winning writeups (Tier T5).
    confidence: high
    implementation_relevance: high
  - claim: Context engineering (retrieval, compression, structured formatting) is required for long-horizon agent stability; compression achieves 10:1 to 100:1 token reduction.
    category: context_engineering
    evidence: Context engineering literature (Tier T3).
    confidence: high
    implementation_relevance: high

architectural_recommendations:
  - recommendation: Hierarchical Supervisor-Worker architecture with bounded functional cells and an independent Adversarial review layer.
    rationale: Commander maintains strategic goal alignment and compute budget; cells isolate domain complexity; adversarial gate catches leaks.
    dependencies: Antigravity custom agents, structured YAML contracts.
    risks: Potential bottleneck at Commander if delegation requires excessive round-trips.
  - recommendation: Machine-readable agent communication protocol with status, confidence, findings, evidence citations, actions, risks, and next tasks.
    rationale: Prevents ungrounded conversational drift and allows automated validation.
    dependencies: Schema validation.
    risks: None.
  - recommendation: Four-layer memory system: Working Memory (in-context), Project Memory (`knowledge/competitions/`), Strategic Memory (`knowledge/strategies/`), Meta Memory (`knowledge/meta/`).
    rationale: Provides clean separation between transient task state and durable cross-competition principles.
    dependencies: Structured filesystem directories.
    risks: Stale strategic memory if promotion criteria are lax.

kaggle_recommendations:
  - recommendation: Validation strategy is a first-class citizen; CV must be theoretically and empirically justified to correlate with private LB before modeling begins.
    expected_value: critical (insulates against private LB shakeout).
    evidence: Kaggle competitive history.
  - recommendation: GBDT baseline (XGBoost, LightGBM, CatBoost) established early before attempting deep neural networks or extensive HPO.
    expected_value: high (fast, robust benchmark).
    evidence: Kaggle Grandmaster tabular playbook.
  - recommendation: Strict compute budgeting allocating GPU hours based on expected information gain; reserve wall-clock hours for submission notebook.
    expected_value: high (prevents mid-competition quota exhaustion).
    evidence: Kaggle 30h weekly GPU quota constraint.

agent_recommendations:
  - role: Kaggle Commander
    responsibility: Global strategy, resource allocation, stopping decisions, phase transitions.
    justification: Central supervisory authority.
  - role: Competition Researcher
    responsibility: Extract rules, evaluation metrics, submission contracts, external data constraints.
    justification: Grounding and compliance.
  - role: Data Forensics
    responsibility: Profiling, schema parsing, missingness, leakage detection.
    justification: Early data quality verification.
  - role: Validation Architect
    responsibility: Designs robust CV splits; holds hard VETO power over invalid validation designs.
    justification: Decisive role in competition success.
  - role: Feature Engineer
    responsibility: Hypothesis-driven feature creation and transformation pipelines.
    justification: Highest-leverage modeling activity.
  - role: Model Researcher
    responsibility: Evaluates and selects appropriate model families based on modality.
    justification: Guides model search.
  - role: HPO Agent
    responsibility: Bounded hyperparameter tuning via Optuna.
    justification: Fine-tunes promising models.
  - role: Experiment Manager
    responsibility: Maintains experiment registry and trial metadata.
    justification: Central ledger of trials.
  - role: Ensemble Agent
    responsibility: Searches for complementary models, performs OOF blending and stacking.
    justification: Maximizes ensemble diversity.
  - role: Adversarial Reviewer & Leakage Hunter
    responsibility: Probes leakage, tests inference feature availability, attacks solution robustness; holds VETO power.
    justification: Independent failure-seeking cell.
  - role: Error Analyst
    responsibility: Analyzes residuals and subgroup errors to inform future feature hypotheses.
    justification: Generates error-driven insights.
  - role: Kaggle Executor
    responsibility: Notebook preparation, Kaggle CLI execution, status monitoring.
    justification: Cloud execution interface.
  - role: Artifact Analyst
    responsibility: Ingests and parses outputs, validates artifact contracts.
    justification: Ingestion engine.
  - role: Strategy Evolution & Memory Curator
    responsibility: Translates evidence into reusable strategies, maintains knowledge base.
    justification: Powers self-evolution and cross-competition transfer.

memory_recommendations:
  - Layered memory: Working, Project, Strategic, Meta.
  - Strict evidence citation linking claims to experiment IDs and artifact files.
  - Context compression techniques to keep active prompt size within optimal limits.

self_evolution_recommendations:
  - Closed-loop skill evolution: Failure observation -> Classification -> Root cause -> Generalizable pattern -> Skill draft -> Adversarial review -> Versioned candidate -> A/B test -> Promotion/Rejection.
  - Controlled evolution without self-modifying orchestration.

compute_recommendations:
  - Priority formula balancing expected score gain, confidence, information gain against compute cost and risk.
  - Polars and Parquet for high-performance memory-efficient data processing.
  - Workload-based hardware routing (CPU for data prep/EDA; GPU for deep learning and heavy boosting).

automation_recommendations:
  - Automated Kaggle CLI execution (`kaggle kernels push`, `kaggle kernels output`).
  - Git worktrees for parallel experiment isolation.

concerns:
  - 15 agents may add coordination latency unless inactive agents remain unspawned.
  - Risk of context bloat on long-horizon sessions without aggressive compaction.

contradictions:
  - Recommends 15 agents (matches DeepSeek's count, higher than Julius's 9 and Meta's 7).
  - Explicitly confirms supervisor-worker architecture over pure blackboard or swarm.
```
