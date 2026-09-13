# Research Extraction: qwen.md

```yaml
source:
  file: research/01-raw/qwen.md
  model: Qwen (Alibaba Cloud / Scientific validation research)
  date_if_available: 2026-09-13

claims:
  - claim: Hierarchical supervisor architecture is essential for Gemini 3.8 Flash High to manage practical token output constraints (~4k effective per pass) and balance high throughput with token expenditure controls.
    category: model_architecture_fit
    evidence: Empirical API behavior observations and DeepSWE / MLE-bench studies (Tier T3).
    confidence: high
    implementation_relevance: critical
  - claim: Knowledge-base size is a poor proxy for intelligence; self-evolution must be proven via functional competence in cross-competition transfer experiments comparing Fresh State vs Evolved State on unseen competitions.
    category: evaluation_methodology
    evidence: HASTE and MLEvolve evaluation methodologies (Tier T3).
    confidence: high
    implementation_relevance: critical
  - claim: Controlled ablation framework is mandatory to justify every cell (Full System vs Ablated Adversarial Cell, Ablated Memory, Ablated Scheduler, Ablated Hierarchy).
    category: scientific_validation
    evidence: Standard multi-agent empirical ablation methodology (Tier T3).
    confidence: high
    implementation_relevance: critical
  - claim: Modality Router analyzing dataset file signatures decouples high-level strategic coordination from modality-specific pipelines (Tabular vs CV vs NLP vs Time Series).
    category: modular_design
    evidence: Multi-modal agent orchestrators (UniDataBench, AutoKaggle) (Tier T3).
    confidence: high
    implementation_relevance: high

architectural_recommendations:
  - recommendation: Hierarchical Supervisor with modular cells (Modality Router, Experiment Scheduler, Persistent Memory, Adversarial Review Cell, Modality-specific Execution Cells).
    rationale: Decouples meta-planning from granular implementation; enforces verification before code commits.
    dependencies: Modular component interfaces (ACP/ECP protocols).
    risks: Interface complexity if protocols are over-engineered.
  - recommendation: Standardized three-level evaluation framework: (1) Knowledge accumulation rate, (2) Decision & trajectory quality (Plan Quality, Tool Correctness, Step Efficiency), (3) Competitive outcomes (Private LB score, Medal Success Rate).
    rationale: Isolates failure points across the multi-step trajectory instead of relying solely on final leaderboard score.
    dependencies: Execution trace logging.
    risks: None; strengthens system observability.
  - recommendation: Controlled ablation framework built into the system to measure Cell-level contributions.
    rationale: Proves whether each component (Adversarial Cell, Memory, Scheduler) actually improves performance or merely adds overhead.
    dependencies: Feature flags / configuration toggles for each cell.
    risks: Requires running multiple ablation trials.

kaggle_recommendations:
  - recommendation: Cross-competition transfer protocol: Train/evolve on Cohort A -> Freeze evolved memory -> Benchmark Fresh Agent vs Evolved Agent on unseen Cohort B -> Measure delta in time-to-baseline, compute efficiency, and private LB score.
    expected_value: critical (definitive scientific proof of self-evolution).
    evidence: Standard empirical transfer learning methodology.
  - recommendation: Modality Router automatically selects pipeline based on dataset files before initiating task decomposition.
    expected_value: high (prevents inappropriate modeling choices).
    evidence: AutoKaggle / UniDataBench frameworks.

agent_recommendations:
  - role: Supervisor (Commander)
    responsibility: Strategic planning, task decomposition, resource scheduling, meta-reasoning, state coordination.
    justification: Centralized supervisory intelligence.
  - role: Modality Router
    responsibility: Inspects dataset files, classifies problem domain, activates appropriate pipeline orchestrator.
    justification: Domain-specific routing.
  - role: Experiment Scheduler
    responsibility: Concurrency management, compute budget control, token spend optimization.
    justification: Resource guardrail.
  - role: Model Cell
    responsibility: Develops modality-specific architectures and training pipelines.
    justification: Specialized code generation.
  - role: Adversarial Cell
    responsibility: Independent verification, syntax/linter checks, logic flaw detection, leakage auditing.
    justification: Prevents cascading errors from model hallucinations.
  - role: Persistent Memory Layer
    responsibility: Stores rich semantic embeddings of validated artifacts, strategies, and lessons for cross-competition transfer.
    justification: Powers self-evolution.

memory_recommendations:
  - Persistent memory module storing validated artifacts with contextual metadata (dataset schema, validation strategy, empirical outcome).
  - Cross-competition transfer evaluation comparing fresh agent memory vs populated memory on held-out competitions.
  - Vector/semantic indexing for cross-competition retrieval.

self_evolution_recommendations:
  - True self-evolution is demonstrated by performance differential on transfer tasks, not lines of saved memory text.
  - Controlled ablation of memory module to test whether accumulated knowledge provides statistically significant lift.

compute_recommendations:
  - Centralized Experiment Scheduler controlling token usage and wall-clock time.
  - Bounded task generation to respect practical ~4k output token limits.

automation_recommendations:
  - Automated trace logging (DeepEval-style trajectory evaluation) capturing all intermediate tool calls and decisions.
  - Standardized evaluation against MLE-bench Lite style held-out competition sets.

concerns:
  - Verifier/adversarial agents can act as bottlenecks or have high false-acceptance rates if prompts are not calibrated.
  - High thinking effort of Gemini 3.8 Flash burning excessive tokens if loops are unbounded.

contradictions:
  - Strongly advocates vector database for persistent memory, whereas Julius demonstrated that SQLite/JSONL is sufficient and less prone to operational failures in Phase 1.
  - Focuses heavily on academic benchmarks (MLE-bench, SWE-bench) over Kaggle-specific CLI execution quirks.
```
