# Architecture Decision Register (ADR): KAGGLE-OS

This register documents the foundational architectural decisions for KAGGLE-OS, grounded in empirical evidence from 10 independent research documents and verified Antigravity CLI + Kaggle platform capabilities.

---

### ADR-01: Orchestration Topology and Supervisor Model
- **decision:** Hierarchical Supervisor with Bounded Functional Cells
- **question:** Should the multi-agent system use a flat swarm, decentralized debate, a single monolithic supervisor, or a hierarchical supervisor?
- **options:**
  1. Flat peer-to-peer swarm without a central manager
  2. Multi-agent debate / conversational loop
  3. Single monolithic supervisor doing all routing and evaluation
  4. Hierarchical supervisor (Commander) with bounded specialized cells
- **evidence:** Multi-agent benchmarks (Openlayer, Google Research) show multi-agent systems improve parallel subtasks (+80%) but degrade sequential reasoning by 39–70%. Debate on identical models causes sycophancy cascades without genuine cognitive diversity.
- **supporting_sources:** Claude, DeepSeek, Gemini, GPT, Grok, Julius, Kimi, Meta, Mistral, Qwen (All 10).
- **contradicting_sources:** None for production ML workflows.
- **verification:** Verified on software engineering benchmarks (ExecuGraph arXiv:2607.20499; DeepSWE v1.1).
- **tradeoffs:** Hierarchical supervisor introduces a single coordination point, but completely prevents quadratic coordination overhead $O(n^2)$ and infinite debate loops.
- **chosen_option:** Option 4 (Hierarchical supervisor with bounded cells).
- **why:** Kaggle workflows have strict sequential phases (Rules → Forensics → Validation → Modeling → Ensembling) interspersed with parallelizable experiment execution. A two-tier hierarchy matches this problem structure.
- **rejected_options:** Options 1 and 2 (chaotic, high token waste); Option 3 (context overload on a single prompt).
- **confidence:** High (A-tier evidence).
- **reversible:** Yes, cell configurations can be re-grouped in configuration.

---

### ADR-02: Epistemic Authority and Truth Separation
- **decision:** Strict Separation of Control Plane (LLM Reasoning) and Data Plane (Deterministic Execution)
- **question:** Who owns truth? Should LLMs evaluate code correctness and metrics, or should deterministic code be the sole authority?
- **options:**
  1. LLM-as-a-judge (agents read code and debate whether it worked)
  2. Fully deterministic script evaluation (Python scripts compute scores, hashes, and validation checks)
  3. Hybrid with physical execution as the sole decision predicate
- **evidence:** ExecuGraph (arXiv:2607.20499) demonstrated that LLM-as-a-judge exhibits up to 61.3% semantic flip rates on rephrased prompts and cannot detect runtime boundary bugs or memory OOMs.
- **supporting_sources:** GPT, Julius, Gemini, Claude, Kimi, Mistral.
- **contradicting_sources:** None.
- **verification:** Standard automated software test harnesses and scikit-learn cross-validation.
- **tradeoffs:** Requires writing deterministic verification scripts, but guarantees 100% mathematical integrity of logged scores.
- **chosen_option:** Option 3 (Hybrid: LLM proposes hypotheses and code; deterministic Python scripts compute scores, splits, hashes, and Kaggle leaderboard telemetry).
- **why:** Prevents model hallucinations from polluting the experiment ledger or claiming false improvements.
- **rejected_options:** Option 1 (unreliable and hallucination-prone); Option 2 alone (lacks creative hypothesis generation).
- **confidence:** High (A-tier evidence).
- **reversible:** No; non-negotiable core engineering law.

---

### ADR-03: Inter-Agent Communication Medium
- **decision:** Durable Filesystem Blackboard (`blackboard/state.json`) with Atomic Writes and Sentinels
- **question:** Should agents communicate via chat history, CLI stdout pipes, an in-memory message broker, or a filesystem blackboard?
- **options:**
  1. Shared chat context window
  2. Captured stdout/stdin pipes from `agy -p`
  3. In-memory message bus (RabbitMQ / Redis)
  4. Durable filesystem blackboard with atomic file writes (write tmp + rename) and completion sentinels
- **evidence:** Subprocess execution of `agy -p` in non-TTY environments has known failure modes (silent output drops, terminal hangs). Passing state via chat history rapidly exceeds context boundaries.
- **supporting_sources:** Claude, GPT, Julius, Kimi, Meta.
- **contradicting_sources:** Grok (cautioned against complex blackboard systems, but endorsed file state).
- **verification:** Direct inspection of Antigravity CLI non-interactive execution contracts.
- **tradeoffs:** Filesystem polling has millisecond latency overhead, but provides crash-resilient persistence, complete observability, and clean isolation across subagents.
- **chosen_option:** Option 4 (Durable filesystem blackboard).
- **why:** Eliminates stdout capture fragility, survives agent crashes, and allows human inspection at any time.
- **rejected_options:** Options 1 and 2 (fragile, subject to truncation); Option 3 (unnecessary daemon dependency).
- **confidence:** High (A-tier evidence).
- **reversible:** Yes, storage format can be adjusted.

---

### ADR-04: Core Agent Roster Sizing
- **decision:** 11 Core Roles Across 5 Functional Planes
- **question:** How many agents should exist in the default system?
- **options:**
  1. Minimalist (6–7 agents, e.g. Meta)
  2. Lean Functional (9–11 agents, e.g. Julius, Kimi, GPT)
  3. Bounded Specialization (12–14 agents, e.g. Claude, Grok)
  4. Highly Granular (15–19 agents, e.g. DeepSeek, Mistral, Gemini)
- **evidence:** Empirical cognitive load research shows that agent performance peaks between 8–12 specialized roles. Beyond 15 roles, marginal intelligence diminishes while coordination overhead scales quadratically.
- **supporting_sources:** Julius, Kimi, GPT, Claude, Meta.
- **contradicting_sources:** Gemini, DeepSeek, Mistral (advocated 15–19 agents).
- **verification:** Multi-agent benchmark studies (ChatBench, TeamBench).
- **tradeoffs:** 11 agents requires combining closely related roles (e.g. Data Forensics + EDA; Model Selection + HPO), but eliminates unnecessary intermediate handoffs.
- **chosen_option:** Option 2 / 3 blend: **11 Core Roles**:
  - `Commander` (Supervisor)
  - `Scout` (Competition Intelligence)
  - `Data/EDA Specialist` (Data Forensics & Exploratory Analysis)
  - `Validation Architect` (CV Design & Leakage Testing — VETO)
  - `Feature/Model Strategist` (Hypothesis Generation & Model Selection)
  - `Experiment Manager` (Registry & Ledger)
  - `Runner` (Worktree Execution & Artifact Emission)
  - `Blender` (Ensemble Optimizer & Stacking)
  - `Adversarial Auditor` (Failure-Seeking & Compliance — VETO)
  - `Kaggle Executor` (Kernel Packaging & CLI Push/Pull)
  - `Memory Curator / Chronicler` (Artifact Ingestion & Knowledge Evolution)
- **why:** Covers every required phase of competitive ML without redundant communication layers.
- **rejected_options:** Option 1 (overloads single roles); Option 4 (excessive handoffs and token spend).
- **confidence:** High (B-tier evidence).
- **reversible:** Yes, agents are defined as modular `.agents/agents/*.md` files.

---

### ADR-05: Validation Veto Authority
- **decision:** Absolute Binding VETO Power for Validation Architect and Adversarial Auditor
- **question:** Should validation warnings be advisory or blocking?
- **options:**
  1. Advisory warnings (Commander may proceed if CV is high)
  2. Voting consensus (majority vote among modeling agents)
  3. Absolute binding VETO (hard pipeline stop on failed validation or leakage)
- **evidence:** Kaggle competition history proves that high local CV with leakage or improper split schemes reliably leads to catastrophic private leaderboard drops (shakeouts). Overfitting to public LB is a classic trap.
- **supporting_sources:** All 10 research documents unanimously support validation veto power.
- **contradicting_sources:** None.
- **verification:** Scikit-learn validation guidelines; fastai adversarial validation methodology.
- **tradeoffs:** May temporarily halt modeling until splits are fixed, but prevents wasting days of compute and submission slots on corrupted models.
- **chosen_option:** Option 3 (Absolute binding VETO).
- **why:** Scientific integrity must override modeling eagerness. A model trained on leaky folds is worse than useless.
- **rejected_options:** Options 1 and 2 (disastrous for private LB generalization).
- **confidence:** Very High (A/B-tier consensus).
- **reversible:** No.

---

### ADR-06: Memory Architecture Backend
- **decision:** Four-Layer Memory Backed by Filesystem, SQLite, JSONL, and Parquet
- **question:** Should memory be a heavy Vector DB / Graph DB (Neo4j, Mem0, Graphiti) or structured local files and SQLite?
- **options:**
  1. In-memory transcript context only
  2. Vector database (Chroma / Qdrant / FAISS) + Graph DB (Neo4j / FalkorDB)
  3. Four-layer architecture using SQLite, JSONL, Parquet, and Markdown files with metadata filtering
- **evidence:** Kaggle strategies are naturally indexed by categorical metadata (modality, metric, dataset scale, cardinality). Structured SQL/file queries provide exact retrieval without semantic hallucination or daemon failures.
- **supporting_sources:** Julius, Claude, Kimi, GPT.
- **contradicting_sources:** Meta (proposed Graphiti/Neo4j), Qwen (proposed Vector DB).
- **verification:** Local file I/O benchmarks; operational reliability in headless environments.
- **tradeoffs:** Lacks fuzzy semantic embedding search initially, but eliminates external service dependencies, authentication issues, and heavy RAM overhead.
- **chosen_option:** Option 3 (Four-layer SQLite + JSONL + Parquet architecture):
  - Layer 1 (Working): In-memory + `blackboard/state.json`
  - Layer 2 (Project): `knowledge/competitions/<slug>/` + `artifacts/<exp_id>/`
  - Layer 3 (Strategic): `knowledge/strategies/*.yaml` (cross-competition evidence)
  - Layer 4 (Meta): `knowledge/meta/experiments.db` (agent scorecards & run metrics)
- **why:** 100% portable, zero-maintenance, reproducible, and transparent.
- **rejected_options:** Option 1 (context loss); Option 2 (premature overengineering for Phase 1).
- **confidence:** High (B-tier evidence).
- **reversible:** Yes; vector indexing can be added as a plugin later if retrieval quality proves limiting.

---

### ADR-07: Experiment Scheduling and Resource Allocation
- **decision:** Expected Value (EV) Priority Formula with Resource Ledger
- **question:** How should candidate experiments be selected and scheduled?
- **options:**
  1. First-In-First-Out (FIFO) queue
  2. Greedy maximization of last observed CV score
  3. Random stochastic exploration
  4. Expected Value formula: $\text{Priority} = \frac{\text{Expected\_Gain} \times \text{Confidence} \times \text{Info\_Gain}}{\text{Compute\_Cost} \times (1 + \text{Risk})}$
- **evidence:** Diminishing returns of exhaustive HPO sweeps on suboptimal feature sets; strict Kaggle 30h/week GPU quota.
- **supporting_sources:** DeepSeek, Gemini, GPT, Grok, Julius, Kimi, Meta, Mistral (8 models).
- **contradicting_sources:** None.
- **verification:** Multi-armed bandit literature and Bayesian optimization theory.
- **tradeoffs:** Requires agents to explicitly estimate expected gain, confidence, and risk before running code.
- **chosen_option:** Option 4 (Expected Value prioritization formula).
- **why:** Optimizes the primary target: `Expected Private LB Improvement / Kaggle Compute`.
- **rejected_options:** Options 1, 2, and 3 (wasteful, vulnerable to local minima).
- **confidence:** High (B-tier evidence).
- **reversible:** Yes; weight parameters in `config/compute_budget.yaml` can be tuned.

---

### ADR-08: Parallelism and Experiment Isolation
- **decision:** Git Worktrees for Independent ML Experiments; Strict Sequentiality for Phase Transitions
- **question:** Should agents run concurrently, and how should parallel experiments be isolated?
- **options:**
  1. Completely sequential execution on a single branch
  2. Concurrent agent execution with shared directory access
  3. Git worktree isolation for independent experiment branches
- **evidence:** Antigravity natively supports background subagents; Git worktrees allow multiple directories sharing a single `.git` repository, enabling concurrent model training without file overwrite collisions.
- **supporting_sources:** Claude, DeepSeek, Gemini, GPT, Julius, Kimi, Meta, Mistral.
- **contradicting_sources:** None.
- **verification:** Native Git worktree CLI verification (`git worktree add`).
- **tradeoffs:** Minor disk space usage per active worktree; requires clean worktree cleanup upon experiment completion.
- **chosen_option:** Option 3 (Git worktrees for independent trials; sequential phase gating for strategy transitions).
- **why:** Prevents file race conditions while exploiting Antigravity's concurrent subagents.
- **rejected_options:** Option 1 (underutilizes CPU/time); Option 2 (catastrophic file race conditions).
- **confidence:** High (A-tier verified).
- **reversible:** Yes.

---

### ADR-09: Self-Evolution Scope and Boundaries
- **decision:** Closed-Loop Empirical Evolution; Immutable Core Orchestration; Human Approval Gate
- **question:** How should self-evolution function without causing system collapse?
- **options:**
  1. Open-loop autonomous code self-rewriting (agents modify their own python scripts and prompts)
  2. Static system with zero learning
  3. Closed-loop failure-driven skill and prior evolution with benchmark verification and human-in-the-loop sign-off
- **evidence:** Autonomous self-rewriting systems without formal verification collapse into prompt degradation, circular bug-fixing loops, and security boundary violations.
- **supporting_sources:** Claude, DeepSeek, GPT, Grok, Julius, Kimi, Meta.
- **contradicting_sources:** None.
- **verification:** EvoSkill (2026) and scientific ablation methodology.
- **tradeoffs:** Requires human review for permanent skill promotions, but guarantees system stability and prevents security regressions.
- **chosen_option:** Option 3 (Closed-loop failure-driven evolution with human sign-off).
- **why:** Matches the core prompt rule: "Never let the system blindly rewrite its own architecture. Self-modification must be evaluated."
- **rejected_options:** Option 1 (dangerous and unstable); Option 2 (fails the self-evolution directive).
- **confidence:** High (B-tier evidence).
- **reversible:** No.

---

### ADR-10: Kaggle Hardware Allocation and P100 Safety Policy
- **decision:** Workload-Driven Hardware Allocation with Automated P100 Deprioritization
- **question:** How should Kaggle compute resources (CPU vs. GPU) be managed?
- **options:**
  1. Always request GPU for all notebook runs
  2. Manual hardware selection per task
  3. Automated workload-based selection: CPU for Polars/EDA/tabular GBDTs; GPU (T4/L4) for NNs and heavy models; explicitly avoid P100
- **evidence:** Kaggle hardware docs state GPUs offer zero acceleration for pandas/scikit-learn. Kaggle Docker PyTorch cu128 images crash on Pascal sm_60 (P100).
- **supporting_sources:** Kimi, Meta, Claude, Kaggle platform guidelines.
- **contradicting_sources:** None.
- **verification:** PyTorch CUDA compute capability compatibility matrices.
- **tradeoffs:** Requires kernel metadata generator to dynamically set accelerator tags.
- **chosen_option:** Option 3 (Automated workload-based selection).
- **why:** Preserves the scarce 30h/week GPU quota and prevents unexpected CUDA runtime failures.
- **rejected_options:** Option 1 (quota wasted in days); Option 2 (breaks unattended automation).
- **confidence:** High (A-tier verified).
- **reversible:** Yes; configured in `config/compute_budget.yaml`.

---

### ADR-11: Public vs. Private Leaderboard Strategy
- **decision:** Private-LB Robustness First; Public LB Treated as a Noisy Sensor
- **question:** How should public leaderboard feedback guide model selection?
- **options:**
  1. Always select the highest public-LB submission
  2. Ignore public LB completely
  3. Trust local CV as ground truth; use public LB strictly to detect catastrophic distribution shift; select final submissions based on CV + diversity
- **evidence:** In competitive ML history, public LB evaluates only a small fraction (often 10–30%) of test data. Teams optimizing for public LB consistently experience severe shakeouts on the private leaderboard reveal.
- **supporting_sources:** All 10 research documents.
- **contradicting_sources:** None.
- **verification:** Historic Kaggle competition shakeup distributions.
- **tradeoffs:** May occasionally yield a slightly lower public LB rank during the competition, but maximizes private LB placement at close.
- **chosen_option:** Option 3 (CV-first, public LB as noisy diagnostic sensor).
- **why:** Matches the fundamental optimization target: *Private Leaderboard Performance*.
- **rejected_options:** Option 1 (causes catastrophic shakeout); Option 2 (ignores valuable sensor data for pipeline bugs).
- **confidence:** Very High (A-tier consensus).
- **reversible:** No.

---

### ADR-12: Skill Catalog Governance
- **decision:** Categorical Rejection of External Delegation Skills and Third-Party Gateways
- **question:** How should candidate skills from the 2,121-skill catalog be governed?
- **options:**
  1. Install all skills that mention ML or agent capabilities
  2. Permit delegation to external models (Claude, GPT, Codex) when tasks get hard
  3. Enforce strict governance: Reject all delegation skills (`*-delegate`) and model gateways; adopt only general engineering skills (`uv`, `worktrees`, `polars`); hand-build Kaggle core ML skills
- **evidence:** Standing user directive strictly mandates that the system operates using Antigravity CLI + Gemini 3.8 Flash High as the sole reasoning model.
- **supporting_sources:** Claude, Julius, Kimi, GPT.
- **contradicting_sources:** None.
- **verification:** Standing workspace directives and skill audit results.
- **tradeoffs:** Requires building domain-specific Kaggle skills from scratch, but ensures 100% compliance and eliminates security risks.
- **chosen_option:** Option 3 (Strict governance).
- **why:** Enforces system boundaries and prevents silent model substitution.
- **rejected_options:** Options 1 and 2 (violate core platform constraints).
- **confidence:** Very High (A-tier constraint).
- **reversible:** No.
