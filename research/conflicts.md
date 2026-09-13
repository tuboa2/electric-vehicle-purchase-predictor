# Research Conflicts and Resolutions: Architectural Trade-Off Analysis

This document details the major technical disagreements, competing paradigms, and conflicting claims identified across the 10 independent research documents (`claude.md`, `deepseek.md`, `gemini.md`, `gpt.md`, `grok.md`, `julius.md`, `kimi.md`, `meta.md`, `mistral.md`, `qwen.md`), along with the evidence-weighted resolution for KAGGLE-OS.

---

## 1. Conflict: Agent Roster Size (Minimalism vs. Granular Specialization)

### The Positions
- **Granular Swarm Camp (15–19 Agents):**
  - **Gemini (19 agents):** Decomposes every task into separate agents: Commander, Competition Researcher, Rule Compliance, Data Forensics, EDA, Validation Architect, Model Researcher, Feature Engineer, HPO, Experiment Manager, Ensemble, Performance Engineer, Adversarial Reviewer, Leakage Hunter, Kaggle Executor, Artifact Analyst, Strategy Evolution, Error Analyst, Memory Curator.
  - **DeepSeek (15 agents) & Mistral (15 agents):** Propose 15 specialized agents partitioned across 3 to 7 cells.
  - *Argument:* Complete functional isolation minimizes individual agent prompt complexity and eliminates role ambiguity.
- **Lean Functional Camp (7–10 Agents):**
  - **Meta (7 agents - 6+1):** Meta-Supervisor + 5 specialists (Intel, Data, Feature, Model, Execution) + 1 Critic + 1 Memory Curator.
  - **Julius (9 core roles + on-demand workers):** Commander, Competition Intelligence, Data Forensics, Validation Architect, Experiment Lead, Model/Feature Specialist, Artifact Analyst, Adversarial Auditor, Memory Curator.
  - **Kimi (10 agents):** Commander, Scout, Forensic, Validator, Engineer, Runner, Blender, Adversary, Executor, Chronicler.
  - **Claude (13 agents):** Argues that marginal intelligence drops sharply past 13 roles and merges EDA with Data Forensics, HPO with Model Research, and folds Performance Engineering into a reusable skill.
  - *Argument:* Coordination overhead scales quadratically $O(n^2)$; on a single mid-tier reasoning model (Gemini 3.8 Flash High), each additional agent adds token overhead, context handoff latency, and error propagation risk.

### The Conflict
Does fine-grained agent specialization improve Kaggle solution quality, or does the resulting coordination tax degrade overall performance?

### The Evidence
- Multi-agent orchestration synthesis research (Google / Openlayer) demonstrates that multi-agent systems excel at parallelizable subtasks (+80%) but degrade sequential reasoning by 39–70% due to coordination tax.
- In Kaggle competitions, stages are strictly sequential: Rules → Data Forensics → Validation Lock → Baseline → Feature/Model Iteration → Ensembling → Submission.
- Having separate agents for "Data Forensics" and "EDA", or "Model Researcher" and "HPO", requires passing intermediate artifacts between agents with no independent verification gain.

### Resolution: 11 Core Roles Across 5 Functional Planes
Adopt an **11-agent core architecture**:
1. **Control Plane:** `Commander` (Supervisor), `Scout` (Competition Intelligence)
2. **Evidence Plane:** `Data/EDA Specialist`, `Validation Architect` (VETO), `Leakage/Compliance Auditor` (VETO)
3. **Research & Experiment Plane:** `Feature/Model Strategist`, `Experiment Manager`, `Runner` (Execution in Git worktrees), `Blender` (Ensemble Optimizer)
4. **Verification & Execution Plane:** `Kaggle Executor`, `Artifact Analyst`
5. **Evolution Plane:** `Memory Curator / Chronicler`

*Cross-cutting concerns (Performance Engineering, Data Profiling, Package Management) are implemented as Antigravity Skills, not separate agents.*

---

## 2. Conflict: Memory Architecture Backend (Graph/Vector DB vs. Structured Filesystem/SQLite)

### The Positions
- **Heavy Infrastructure Camp (Vector DB & Temporal Graph):**
  - **Meta:** Recommends a Graphiti / Neo4j / FalkorDB temporal knowledge graph to track bi-temporal evolution of Kaggle strategies, combined with FAISS for episodic memory.
  - **Qwen:** Mandates general-purpose vector database embeddings for all code artifacts and contextual metadata.
  - *Argument:* Semantic similarity and graph-based causal chaining enable flexible cross-competition transfer.
- **Lightweight Portable Camp (Filesystem, SQLite, JSONL, Parquet):**
  - **Julius, Claude, Kimi, GPT:** Explicitly reject vector databases and graph databases for initial implementation. Recommends SQLite + JSONL + Parquet with content hashing and structured metadata filtering.
  - *Argument:* Running local Neo4j or vector services introduces brittle daemon dependencies, credential complexity, serialization friction, and zero demonstrable private-LB gain for a single engineer.

### The Conflict
Is semantic vector/graph search necessary for cross-competition retrieval, or is structured relational/file-based indexing superior?

### The Evidence
- Kaggle problem spaces are naturally categorized by discrete metadata: `modality` (tabular, vision, nlp, audio, time-series), `metric` (auc, logloss, rmse, f1), `dataset_signature` (row count, feature count, high-cardinality flags, missingness ratio), and `validation_scheme` (group, temporal, stratified).
- Exact structured SQL / metadata queries (`WHERE modality='tabular' AND has_high_cardinality=true`) retrieve relevant past strategies with 100% precision and zero embedding distortion.
- Content hashing (SHA-256) guarantees immutable provenance and reproducibility.

### Resolution: File-First SQLite + JSONL + Parquet
Adopt **SQLite + JSONL + Parquet with metadata filtering**:
- Working Memory: In-memory session state + `blackboard/state.json`.
- Project Memory: `knowledge/competitions/<slug>/` and `artifacts/<exp_id>/`.
- Strategic Memory: `knowledge/strategies/` (versioned YAML records with evidence citations and promotion status).
- Meta Memory: SQLite database (`knowledge/meta/experiments.db`) tracking tabular experiment runs, resource utilization, and agent performance scorecards.
- *Vector database and graph database backends are deferred until empirical ablation demonstrates that structured retrieval is limiting.*

---

## 3. Conflict: Multi-Agent Debate as a Verification Mechanism

### The Positions
- **Debate Advocates:**
  - **Gemini & Meta:** Propose multi-agent debate (Critic vs. Generator, or multi-model voting) to evaluate modeling choices and code quality before execution.
  - *Argument:* Debate reduces individual hallucination and catches edge cases through dialectical reasoning.
- **Debate Skeptics:**
  - **Claude, Kimi, Julius, GPT:** Argue that running debate between multiple instances of the *same* underlying model (Gemini 3.8 Flash High) produces sycophancy cascades and token churn without increasing cognitive diversity.
  - *Argument:* True diversity comes from diverse models or physical execution feedback. Debate on a single model merely wastes tokens and latency.

### The Conflict
Does LLM-to-LLM text debate improve code and validation rigor when all participants use the same reasoning model?

### The Evidence
- Recent empirical multi-agent literature demonstrates that peer debate among instances of the same model results in sycophancy cascades, where agents converge on flawed consensus rather than uncovering subtle bugs.
- ExecuGraph (arXiv:2607.20499) proved that LLM-as-a-judge exhibits up to 61.3% semantic flip rates and fails to detect physical runtime boundary flaws.
- In competitive ML, ground truth comes from deterministic execution (out-of-fold cross-validation scores, adversarial validation AUC, memory profiling), not persuasive rhetoric.

### Resolution: Execution-Grounded Adversarial Verification with Veto Power
Reject open-ended conversational debate. Replace with **Execution-Grounded Adversarial Verification**:
- The Adversarial Reviewer / Leakage Hunter operates as a single-pass failure-seeking auditor.
- Its critique must be backed by **deterministic test execution**:
  1. Adversarial validation classifier execution (AUC check).
  2. Leakage assertion tests (fold boundary intersection check, future leakage tests).
  3. Physical code execution in a sandbox.
  4. Submission format and row-count verification.
- Holds binding **VETO power** that halts the pipeline if physical checks fail.

---

## 4. Conflict: Autonomous Self-Evolution Scope

### The Positions
- **Autonomous Open-Loop Camp:**
  - Proposes autonomous agents that can rewrite their own orchestration loops, mutate prompts, generate and install new skills dynamically, and schedule arbitrary background cron jobs.
- **Controlled Scientific Closed-Loop Camp:**
  - **Claude, DeepSeek, GPT, Grok, Julius, Kimi:** Strongly mandate that no agent may ever self-modify core orchestration, permission policies, or stopping rules.
  - Skill and prompt evolution must follow strict scientific gating: Failure -> Root Cause -> Skill Draft -> Adversarial Review -> A/B Benchmark -> Human-in-the-Loop Sign-Off.

### The Conflict
How much autonomy should the agent system have in modifying its own operational code and prompts?

### The Evidence
- Autonomous self-modifying code without regression test suites rapidly suffers from catastrophic forgetting, security boundary escape, and circular debugging loops.
- The user directive explicitly mandates: "Never let the system blindly rewrite its own architecture. Self-modification must be evaluated."

### Resolution: Controlled, Gated Self-Evolution
- **Core Orchestration & Permissions:** 100% IMMUTABLE to autonomous agents.
- **Evolutionary Surfaces:**
  1. **Knowledge Graph & Strategy Priors:** Automatically updated upon completion of validated experiments ($evidence\_count \ge 2$).
  2. **Skill Candidates:** When a failure pattern occurs $\ge 3$ times, the Chronicler drafts a candidate `SKILL.md` with explicit evaluation test cases.
  3. **A/B Testing:** Candidate skills are benchmarked against baseline on historical tasks.
  4. **Human Gate:** Permanent promotion of new skills or agent trust level modifications requires explicit human sign-off.

---

## 5. Conflict: Tabular Modeling Defaults (GBDT First vs. Deep Tabular NNs Day-1)

### The Positions
- **Deep Tabular Day-1 (Gemini):** Mandates including modern Tabular Neural Networks (RealMLP, TabM, FT-Transformer) as foundational Day-1 baselines alongside GBDTs to guarantee error diversity.
- **GBDT-First Pragmatism (Claude, Kimi, Julius, Meta, Mistral):** Prioritizes Gradient Boosted Decision Trees (LightGBM, CatBoost, XGBoost) as the immediate default. Neural tabular architectures are introduced strictly in the ensembling phase if compute permits.

### The Conflict
Should deep tabular architectures be trained concurrently from Phase 1, or should GBDTs establish the benchmark first?

### The Evidence
- GBDTs consistently deliver the highest score-per-compute-second on tabular competitions, training in seconds to minutes on CPU/GPU.
- Tabular NNs require significant GPU training time, sensitive learning rate schedules, and extensive normalization.
- In Kaggle competitions, 80% of modeling gains come from feature engineering evaluated rapidly through fast GBDT iterations.
- Furthermore, certain tabular models (e.g. RealTabPFN weights) have restrictive non-commercial licenses that violate competition rules.

### Resolution: GBDT Fast Baseline -> Tabular NN Diversity Layer
- **Phase 1 (Baseline & Feature Iteration):** Exclusively deploy fast GBDT models (LightGBM, CatBoost, XGBoost) to quickly map the validation noise band and iterate on feature hypotheses.
- **Phase 2 (Ensemble Diversification):** If dataset size and compute budget permit, train TabM / RealMLP on identical cross-validation folds.
- **Phase 3 (Blending):** Ingest OOF predictions from both GBDT and NN models into the Hill Climbing blender to exploit uncorrelated errors.

---

## 6. Conflict: Kaggle GPU Hardware Abstraction (P100 vs. T4 vs. L4)

### The Positions
- Several general reports suggest using P100 (16GB VRAM) as a standard Kaggle accelerator.
- **Kimi, Meta, Claude** identify a specific, verified platform pitfall: The current Kaggle Docker environment with PyTorch cu128 lacks Pascal `sm_60` CUDA kernels. `torch.cuda.is_available()` returns `True`, but the first tensor operation throws a runtime CUDA exception.

### Resolution: Automated Hardware Prioritization
The hardware abstraction layer (`capabilities/hardware.py`) automatically discovers available accelerators and implements an explicit safety filter:
1. Prioritize `NvidiaTeslaT4` (or `NvidiaTeslaT4-x2`) and `NvidiaL4`.
2. Reject `NvidiaTeslaP100` for PyTorch $\ge 2.2$ workloads unless fallback CPU emulation or rebuilt kernels are explicitly present.
3. Automatically route CPU-suitable workloads (Polars feature pipelines, EDA, scikit-learn, small GBDTs) to CPU instances to conserve the 30h/week GPU quota.
