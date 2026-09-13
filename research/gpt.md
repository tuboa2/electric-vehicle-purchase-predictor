# Complete architecture blueprint: Autonomous Kaggle OS for Antigravity + Gemini 3.8 Flash High

I completed the research against the supplied specification, the 2,121-skill catalog, current Antigravity documentation, current Kaggle documentation/CLI, recent 2026 Kaggle solution writeups, and relevant agent/automation research.

The central conclusion is:

> **Do not build a swarm. Build a hierarchical, evidence-gated experimentation operating system in which Gemini agents generate and execute hypotheses, while deterministic software, structured artifacts, validation gates, and empirical results control what the system is allowed to believe.**

That directly matches the supplied requirement to optimize **expected private-leaderboard performance per unit of human effort and Kaggle compute**, rather than maximizing agent count. 

---

# 1. Executive architecture

The recommended system is a **Hierarchical Blackboard + Planner/Executor/Verifier + Dynamic Experiment Workers** architecture.

```text
                         ┌──────────────────────────┐
                         │     KAGGLE COMMANDER     │
                         │ Strategic control plane  │
                         └────────────┬─────────────┘
                                      │
                     ┌────────────────┼────────────────┐
                     │                │                │
                     ▼                ▼                ▼
              INTELLIGENCE       DATA / VALIDATION   STRATEGY
                DOMAIN              DOMAIN            DOMAIN
                     │                │                │
                     └────────────────┼────────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ EXPERIMENT SCHEDULER     │
                         │ EV × cost × information  │
                         └────────────┬─────────────┘
                                      │
                       ┌──────────────┼──────────────┐
                       │              │              │
                       ▼              ▼              ▼
                  WORKER A       WORKER B       WORKER C
                 hypothesis     hypothesis     hypothesis
                  isolated       isolated       isolated
                       │              │              │
                       └──────────────┼──────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ DETERMINISTIC EXECUTION  │
                         │ Python / Kaggle / tests  │
                         └────────────┬─────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ ARTIFACT INGESTION       │
                         │ metrics / OOF / logs     │
                         └────────────┬─────────────┘
                                      ▼
               ┌──────────────────────┼──────────────────────┐
               │                      │                      │
               ▼                      ▼                      ▼
          ERROR ANALYST         ENSEMBLE ENGINE       ADVERSARIAL
                                                         REVIEW
               │                      │                      │
               └──────────────────────┼──────────────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ VALIDATION / COMPLIANCE  │
                         │      VETO GATE            │
                         └────────────┬─────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ KAGGLE SUBMISSION        │
                         └────────────┬─────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ KNOWLEDGE EVOLUTION      │
                         │ strategy / memory / skill│
                         └────────────┬─────────────┘
                                      │
                                      └──────► next cycle
```

This is deliberately **not** the naive "20 agents all talk to each other" design.

Antigravity's native architecture makes the hierarchy particularly appropriate: subagents have independent contexts, can run concurrently, can use isolated Git worktrees, and can communicate with their parent/peers. There is also a hard nesting-depth limit, which is another reason to keep the hierarchy shallow and intentional. ([Google Antigravity][1]) 

---

# 2. What the research establishes

## 2.1 Antigravity is capable enough to host the control plane

Current Antigravity CLI supports:

- asynchronous subagents
- custom agents
- workspace/global skills
- plugins
- MCP
- hooks
- headless execution
- model selection
- configurable reasoning effort
- background tasks
- scheduled tasks
- Git worktrees
- agent monitoring

The CLI explicitly exposes `gemini-3.8-flash-high`, making the user's model constraint technically viable. ([Google Antigravity][2])

Gemini 3.8 Flash currently has a **1,048,576-token input limit and 65,536-token maximum output**, structured outputs, function calling, code execution, search grounding, URL context and high thinking mode. Google describes it as designed for long-horizon software engineering and autonomous agents. ([Google AI for Developers][3])

But the architecture should **not exploit the million-token context by dumping the repository into every agent**.

The prompt's layered-context requirement is correct: global rules → role context → competition brief → relevant memory → task → artifacts → recent experiments. 

---

# 3. Why a hierarchy beats a swarm

I evaluated the requested architecture families:

| Architecture              | Recommendation      | Reason                                                    |
| ------------------------- | ------------------- | --------------------------------------------------------- |
| Flat swarm                | Reject              | Coordination/noise/state explosion                        |
| Pure supervisor           | Good                | Simple but supervisor becomes bottleneck                  |
| Manager-worker            | Good                | Strong for bounded experiments                            |
| Hierarchical supervisor   | **Core**            | Best fit for competition decomposition                    |
| Blackboard                | **Core**            | Structured shared experimental state                      |
| Parallel specialists      | **Use selectively** | Excellent for independent hypotheses                      |
| Debate                    | **Gate only**       | Useful for adversarial review, not continuous operation   |
| Critic-generator          | **Use**             | Strong for validation/code review                         |
| Planner-executor-verifier | **Core**            | Separates reasoning from evidence                         |
| Evolutionary population   | **Limited use**     | Useful for experiment/model populations, not whole agents |
| Hybrid                    | **Winner**          | Combines the useful properties without swarm chaos        |

The research does not justify assuming that simply adding agents improves performance. Multi-agent debate research, for example, finds improvements can be marginal and highly dependent on the setting. ([arXiv][4])

So the architecture uses **multiple agents where parallelism produces independent information**, not because "multi-agent" sounds powerful.

---

# 4. Exact agent roster

I recommend **12 persistent roles + dynamically created experiment workers**.

## Control plane

### 1. `commander`

Owns:

- global objective
- resource allocation
- experiment portfolio
- stopping
- agent routing
- strategic synthesis
- final candidate selection

It **does not write most ML code**.

Its primary job is deciding:

> What should we learn next?

---

### 2. `competition-intelligence`

Creates:

```text
competition_intelligence.md
```

Extracts:

- metric
- objective
- deadline
- data
- submission mechanism
- rules
- external-data policy
- internet policy
- compute constraints
- pretrained model restrictions
- team rules
- evaluation protocol
- known baselines
- historical solution patterns

This directly implements the prompt's requested research phase. 

---

## Evidence plane

### 3. `data-forensics`

Responsible for:

- schema
- missingness
- duplicates
- cardinality
- target
- distribution
- train/test differences
- group structure
- temporal structure
- identifiers
- leakage candidates
- suspicious columns

Produces machine-readable dataset fingerprints.

---

### 4. `validation-architect`

**Highest-authority agent before modeling.**

It determines:

- KFold
- StratifiedKFold
- GroupKFold
- StratifiedGroupKFold
- time-series splits
- blocked temporal CV
- nested CV where justified
- holdouts
- adversarial validation

It has a **hard veto**.

No expensive experiment may enter the scheduler unless validation is approved.

The prompt explicitly requires this veto model. 

---

### 5. `leakage-compliance`

This combines two things that should be independently checked but share a veto boundary:

**Leakage**

- target leakage
- temporal leakage
- duplicated entities
- train/test contamination
- fold contamination
- feature-generation leakage
- pseudo-label contamination

**Competition compliance**

- external data
- internet
- pretrained models
- licenses
- APIs
- submission rules
- compute
- team restrictions

The system must never treat "higher CV" as evidence that an experiment is legitimate.

---

## Research plane

### 6. `model-strategist`

Chooses candidate model families based on:

```text
problem type
+
dataset size
+
feature types
+
validation
+
compute
+
historical evidence
```

It does **not** automatically select one favorite model.

### Tabular

Candidates:

- LightGBM
- XGBoost
- CatBoost
- HistGradientBoosting
- ExtraTrees
- linear models
- RealMLP
- TabM
- other appropriate modern tabular models

Recent 2026 Kaggle results show why diversity matters: high-ranking solutions have combined CatBoost/XGBoost/LightGBM, neural tabular models, target encoding and multiple feature representations. ([Kaggle][5])

### Vision

Candidates:

- pretrained CNN
- ViT
- modern vision backbone
- augmentation
- TTA
- multi-view architectures

A 2026 first-place vision solution used DINOv3 ViT-Huge+, LLRD, EMA and ensemble diversity; another first-place biomass solution used a dual-stream DINO-like model and online/test-time training. ([Kaggle][6])

### NLP

Candidates:

- TF-IDF
- linear models
- pretrained transformer
- embeddings
- task-specific fine-tuning
- probability/representation ensembles

A 2026 first-place medical NLP solution won with a comparatively straightforward pretrained BERT + class weighting + focal loss + stratified CV + fold ensemble, while several more complicated alternatives failed to improve LB. ([Kaggle][7])

### Time series

Candidates:

- naive/statistical baselines
- lag/rolling features
- GBDT
- sequence models
- temporal transformers where justified
- seasonal/global-local models

The system should distinguish forecasting structure from ordinary random CV. A successful time-series solution explicitly exploited local autoregression plus global seasonal structure. ([Kaggle][8])

---

### 7. `feature-strategist`

This is deliberately a **hypothesis generator**, not a feature spammer.

It proposes:

```text
feature hypothesis
→ expected mechanism
→ leakage analysis
→ expected value
→ compute cost
→ test
```

That matters because current research finds that automated feature-engineering systems are often difficult to control and frequently lack explicit time/memory constraints. ([arXiv][9])

---

## Experiment plane

### 8. `experiment-manager`

Maintains:

```text
experiment registry
experiment DAG
resource budget
candidate queue
experiment status
result provenance
```

It creates **dynamic worker agents**.

A worker might receive:

```yaml
experiment_id: exp_0047
hypothesis: "Ordered target encoding improves minority-class discrimination"
dataset_version: d17
feature_version: f23
validation_version: v8
model: catboost
budget:
  gpu_minutes: 45
  cpu_minutes: 90
success_criterion: "OOF macro-F1 improvement >= 0.001"
```

The worker is disposable.

This is critical.

**The architecture should not create permanent agents for every possible experiment.**

---

### 9. `ensemble-strategist`

Consumes OOF predictions.

Measures:

- OOF score
- prediction correlation
- error correlation
- subgroup complementarity
- calibration
- diversity
- robustness

Then evaluates:

- weighted blending
- rank averaging
- stacking
- hill climbing
- constrained optimization

Recent Kaggle results strongly support this as a major performance layer. The 2026 3rd-place solution used RealMLP, CatBoost and XGBoost variants plus OOF rank/hill-climbing ensemble optimization; another 12th-place solution used a 41-model stack with ordered target encoding. ([Kaggle][10])

But **ensemble size is not the objective**.

The objective is:

> marginal private-LB robustness per unit compute.

---

### 10. `error-analyst`

Studies:

- worst examples
- residuals
- subgroup performance
- calibration
- model disagreement
- distribution shift
- systematic failure

It generates the next hypotheses.

---

## Execution plane

### 11. `kaggle-executor`

Owns:

- notebook generation
- metadata
- Kaggle CLI
- execution
- artifact download
- status polling
- reproducibility

Kaggle's official CLI currently supports competition operations, datasets, models and kernels/notebooks. ([GitHub][11])

The kernel API currently supports:

```bash
kaggle kernels init
kaggle kernels push
kaggle kernels pull
kaggle kernels output
kaggle kernels status
```

and accelerator selection on `kernels push`. ([GitHub][12])

---

### 12. `artifact-memory`

This is the system's institutional-memory officer.

Responsibilities:

- ingest artifacts
- update experiment registry
- update knowledge
- detect contradictions
- score evidence
- update strategy priors
- update agent performance
- identify reusable skills
- archive failed hypotheses

The prompt's desired four-layer memory architecture is retained. 

---

# 5. The most important design: agents do not own truth

The system has three different authorities.

```text
Gemini agents
     ↓
hypotheses / plans / code
     ↓
deterministic execution
     ↓
artifacts / metrics
     ↓
validation + evidence
     ↓
knowledge
```

Never:

```text
agent says X worked
→ memory records X worked
```

Instead:

```text
agent hypothesis
→ experiment
→ artifact
→ independently verified metric
→ evidence record
→ confidence update
```

That implements the supplied requirement to distinguish:

- observed fact
- strong empirical pattern
- weak hypothesis
- speculation. 

---

# 6. Blackboard architecture

The shared state should be a **filesystem-backed structured blackboard**, not agent conversation.

```text
.blackboard/
├── state.yaml
├── objectives.yaml
├── constraints.yaml
├── active_tasks.yaml
├── experiment_queue.yaml
├── experiment_results.jsonl
├── candidate_models.yaml
├── validation_status.yaml
├── compliance_status.yaml
├── submission_state.yaml
└── decisions.jsonl
```

Agents communicate primarily through artifacts.

The prompt explicitly calls for machine-readable agent communication rather than free-form chatter. 

---

# 7. Agent communication protocol

Every agent returns:

```yaml
schema_version: "1.0"

task:
  id: "task-0047"
  type: "experiment_analysis"

status: "completed"

confidence:
  overall: 0.82
  epistemic_class: "empirical"

findings:
  - claim: "Target encoding improved OOF macro-F1"
    value: 0.0021
    evidence_refs:
      - "artifact://exp-0047/metrics.json"

actions:
  - "promote_candidate"

artifacts:
  - "artifacts/exp-0047/metrics.json"
  - "artifacts/exp-0047/oof_predictions.parquet"

recommendations:
  - action: "test_on_second_split"
    expected_information_gain: 0.71

risks:
  - "Potential group leakage"

next_tasks:
  - "validate-target-encoding"

provenance:
  agent_version: "experiment-manager@1.3"
  skill_versions:
    - "validation@2.1"
```

This gives the system deterministic downstream parsing.

---

# 8. Memory architecture

## Layer 1 — Working memory

Only current competition state.

```text
.blackboard/
```

Short-lived.

---

## Layer 2 — Project memory

Competition-specific.

```text
knowledge/project/
├── competition.md
├── dataset.md
├── validation.md
├── discoveries/
├── failures/
├── successful_models/
└── final_strategy.md
```

---

## Layer 3 — Strategic memory

Cross-competition.

```text
knowledge/strategic/
├── tabular/
├── vision/
├── nlp/
├── timeseries/
├── multimodal/
├── validation/
├── ensembles/
├── leakage/
├── compute/
└── submission_strategy/
```

---

## Layer 4 — Meta memory

How the **agent system itself** performs.

```text
knowledge/meta/
├── agents/
├── prompts/
├── skills/
├── routing/
├── scheduler/
├── failure_modes/
└── architecture/
```

---

# 9. Evidence-weighted knowledge

Every reusable claim should look like:

```yaml
claim_id: strat-cb-high-cardinality-001

claim: "CatBoost is frequently a strong first baseline for
  medium-sized high-cardinality categorical tabular data."

evidence:
  experiments: 17
  competitions: 9
  positive: 12
  neutral: 3
  negative: 2

confidence: 0.78

scope:
  problem_type: tabular
  categorical_cardinality: high
  dataset_size: medium

counterexamples:
  - competition_x
  - competition_y

last_validated: 2026-09-10

status: strong_empirical_pattern
```

Never store:

```yaml
CatBoost: best model
```

because it is too context-free.

---

# 10. Meta-Kaggle knowledge graph

Use relationships such as:

```text
Competition
    │
    ├── Problem Type
    │
    ├── Dataset Fingerprint
    │
    ├── Validation Strategy
    │
    ├── Feature Families
    │
    ├── Model Families
    │
    ├── Ensemble Strategy
    │
    ├── Compute Profile
    │
    └── Outcome
```

The graph is not primarily for fancy graph infrastructure.

Initially, use **SQLite + structured JSON/YAML records + indexes**.

Only introduce a vector DB if retrieval quality demonstrably requires it.

That is an important anti-overengineering decision.

---

# 11. Experiment scheduler

The scheduler is the heart of the system.

The prompt proposes:

$$
priority =
\frac{
expected\_score\_gain \times confidence \times information\_gain
}{
compute\_cost \times risk
}
$$

I recommend extending it to:

$$
EV_i =
\frac{
P(success_i)
\times E[\Delta Score_i]
+
\lambda I_i
+
\mu D_i
}{
C_i
\times R_i
}
$$

Where:

- `P(success)` = estimated probability of working
- `ΔScore` = expected score improvement
- `I` = information gain
- `D` = ensemble diversity contribution
- `C` = compute cost
- `R` = risk
- `λ`, `μ` = configurable weights

This prevents a common Kaggle failure:

> spending 80% of compute trying to squeeze 0.0001 from the current model while ignoring a cheap structural hypothesis.

---

# 12. Experiment classes

The scheduler should divide experiments into:

### Tier A — Structural

Highest priority:

- validation
- leakage
- target construction
- data interpretation
- train/test shift
- grouping
- temporal structure

### Tier B — High-value modeling

- new model family
- strong feature hypothesis
- pretrained model
- major representation change

### Tier C — Optimization

- hyperparameters
- augmentation
- regularization
- target encoding variations

### Tier D — Ensemble

- blend
- stack
- rank average
- hill climb
- calibration

### Tier E — micro-optimization

- tiny feature variations
- small parameter changes
- low-information experiments

The system should refuse to spend most of its budget on Tier E while Tier A uncertainty remains high.

---

# 13. Experiment lifecycle

```text
IDEA
 ↓
HYPOTHESIS
 ↓
DUPLICATE CHECK
 ↓
LEAKAGE CHECK
 ↓
VALUE ESTIMATION
 ↓
SCHEDULE
 ↓
ISOLATED WORKTREE
 ↓
IMPLEMENT
 ↓
UNIT TEST
 ↓
LOCAL RUN
 ↓
KAGGLE RUN
 ↓
ARTIFACT VALIDATION
 ↓
OOF/CV ANALYSIS
 ↓
ERROR ANALYSIS
 ↓
ADVERSARIAL REVIEW
 ↓
PROMOTE / REJECT / REVISE
 ↓
MEMORY UPDATE
```

---

# 14. Validation system

Validation is not a parameter.

It is a **versioned scientific artifact**.

```text
validation/
├── strategy.md
├── validation_config.yaml
├── folds/
│   ├── fold_00.parquet
│   ├── fold_01.parquet
│   └── ...
├── leakage_tests/
├── distribution_tests/
└── validation_report.json
```

Every model references:

```yaml
validation_version: "v17"
```

This makes comparisons meaningful.

---

# 15. Private leaderboard strategy

The system must explicitly model:

```text
TRAIN
  ↓
CV
  ↓
OOF
  ↓
PUBLIC LB
  ↓
PRIVATE LB
```

as different signals.

The public LB should have a low default weight.

Recent Kaggle writeups provide concrete evidence for this: a 2026 solution describes surviving a major public-LB shake-up through more robust stacking, while another first-place solution deliberately selected based on the underlying competition structure rather than blindly optimizing a public signal. ([Kaggle][13])

Therefore:

```yaml
leaderboard_signal_policy:
  cv: 0.60
  oof: 0.25
  public_lb: 0.10
  domain_prior: 0.05
```

These are **initial priors**, not universal constants.

The system learns them from competition history.

---

# 16. Kaggle execution architecture

Kaggle becomes the remote execution layer.

```text
Antigravity
    │
    ▼
local experiment
    │
    ▼
Git commit
    │
    ▼
Kaggle kernel metadata
    │
    ▼
kaggle kernels push
    │
    ▼
Kaggle execution
    │
    ├── metrics
    ├── OOF
    ├── predictions
    ├── logs
    └── submission
    │
    ▼
kaggle kernels output
    │
    ▼
local artifact store
    │
    ▼
artifact analyst
```

The official CLI explicitly supports this workflow. ([GitHub][12])

---

# 17. Verified Kaggle commands

These are commands I would allow the executor to use.

### Initialize

```bash
kaggle kernels init -p kaggle/kernel
```

### Execute

```bash
kaggle kernels push -p kaggle/kernel
```

### Execute with timeout

```bash
kaggle kernels push -p kaggle/kernel --timeout 43200
```

### Check status

```bash
kaggle kernels status USER/KERNEL
```

### Retrieve artifacts

```bash
kaggle kernels output USER/KERNEL -p artifacts/current
```

### Pull notebook

```bash
kaggle kernels pull USER/KERNEL -p kaggle/kernel -m
```

These commands are documented by the official Kaggle CLI repository. ([GitHub][12])

The CLI's current changelog also shows continuing changes to accelerators, submission behavior, quotas and kernel execution, reinforcing the need for the requested **capabilities refresh layer** rather than hardcoding infrastructure assumptions. ([GitHub][14])

---

# 18. Kaggle hardware abstraction

This is a major architectural decision.

Do **not** hardcode:

```yaml
gpu: P100
ram: 29GB
runtime: 12h
```

as framework constants.

Kaggle's notebook documentation currently lists 12-hour CPU/GPU sessions, 9-hour TPU sessions, 20GB working storage and example CPU/GPU/TPU configurations, while the current CLI documents a changing accelerator inventory and warns that some accelerator types become retired/replaced. ([Kaggle][15])

Therefore:

```text
capabilities/
├── kaggle_capabilities.json
├── accelerator_inventory.json
├── competition_constraints.json
├── cli_capabilities.json
└── refresh_timestamp.json
```

At the beginning of every competition:

```text
refresh capabilities
       ↓
inspect competition-specific restrictions
       ↓
choose execution profile
```

This directly implements the prompt's requirement not to hardcode volatile Kaggle information. 

---

# 19. Accelerator selection policy

The executor should classify workloads.

### CPU preferred

- pandas/Polars
- LightGBM
- XGBoost
- CatBoost
- feature engineering
- statistical models

### GPU preferred

- CNN
- ViT
- transformer
- large embedding models
- deep tabular models
- GPU ensemble optimization where beneficial

### TPU candidate

Only when:

- framework compatibility is proven
- workload is large enough
- data pipeline can feed it
- TPU-specific implementation cost is justified

The system must benchmark rather than assume.

Kaggle itself notes that accelerator choice should be tied to workload characteristics; notebook documentation gives explicit examples of GPU benefits for neural-network workloads. ([Kaggle][15])

---

# 20. Artifact contract

Every run produces:

```text
artifacts/
└── EXP-0047/
    ├── experiment.json
    ├── metrics.json
    ├── resource_usage.json
    ├── predictions.parquet
    ├── oof_predictions.parquet
    ├── submission.csv
    ├── model_metadata.json
    ├── feature_metadata.json
    ├── logs/
    ├── plots/
    └── reports/
```

The prompt's minimum artifact contract is retained. 

---

# 21. Artifact integrity

Before an artifact enters memory:

```text
schema validation
        ↓
checksum
        ↓
experiment ID verification
        ↓
dataset version verification
        ↓
feature version verification
        ↓
validation version verification
        ↓
metric recomputation where possible
        ↓
accept
```

No artifact = no knowledge update.

---

# 22. Self-healing

The requested failure taxonomy should become executable.

```text
DATA_FAILURE
DEPENDENCY_FAILURE
CODE_FAILURE
MEMORY_FAILURE
TIMEOUT
KAGGLE_FAILURE
VALIDATION_FAILURE
MODEL_FAILURE
AGENT_FAILURE
ORCHESTRATION_FAILURE
ARTIFACT_FAILURE
```

The prompt explicitly requires this classification and prohibits blindly repeating the same failed action. 

Recovery policy:

```text
failure
 ↓
classify
 ↓
determine whether retry is safe
 ↓
change one relevant variable
 ↓
retry within budget
 ↓
if repeated → quarantine hypothesis
```

Example:

```text
OOM
→ reduce batch size
→ memory-map data
→ reduce precision
→ reduce model
→ retry

NOT:

OOM
→ run same experiment again
→ OOM
→ run again
```

---

# 23. Skill architecture

Use three levels.

## Global Antigravity skills

Competition-independent:

```text
~/.gemini/antigravity-cli/skills/
```

Antigravity officially supports global skills and workspace-local skills. ([Google Antigravity][16])

Use global skills for:

- experiment protocol
- structured communication
- context engineering
- artifact validation
- Git discipline
- Kaggle CLI
- failure recovery
- experiment tracking
- evidence grading

---

## Project skills

```text
competition/.agents/skills/
```

Contain:

- competition-specific metric
- dataset peculiarities
- validation rules
- model recipes
- domain-specific features
- submission constraints

---

## Dynamic skills

Generated only after repeated evidence.

```text
skill candidate
→ evaluation
→ adversarial review
→ A/B test
→ promotion
```

The prompt explicitly requires this promotion pipeline. 

---

# 24. Recommended catalog skills

The catalog contains 2,121 skills and explicitly says they should be treated as a capability inventory rather than blindly installed. 

My recommended initial subset is small.

## Tier 1 — directly useful

### Orchestration

- `agent-orchestrator`
- `multi-agent-task-orchestrator`
- `agent-orchestration-multi-agent-optimize`
- `parallel-agents`
- `dispatching-parallel-agents`

The catalog confirms these capabilities exist. 

### Memory

- `agent-memory`
- `agent-memory-mcp`
- `tree-ring-memory`

The latter is particularly interesting because its stated purpose includes local-first evidence, audit and consolidation without transcript dumping. 

### Context

- `context-engineering`
- `context-optimization`
- `context-degradation`
- `context-guardian`

These map directly to the required layered-context architecture. 

### Evaluation

- `agent-evaluation`
- `evaluation`
- `advanced-evaluation`

### ML

- `data-scientist`
- `ml-engineer`
- `mlops-engineer`
- `scikit-learn`
- `polars`

The catalog explicitly contains these. 

### Reproducibility

- `using-git-worktrees`
- `uv-package-manager`
- `testing-qa`
- `lint-and-validate`

The Git worktree and `uv` skills are explicitly present. 

---

# 25. Skills I would NOT install

Do not install:

- generic agent swarms
- alternative LLM gateways
- OpenRouter orchestration
- Claude-specific orchestration
- local-LLM infrastructure
- unrelated SaaS integrations
- multiple competing memory databases
- multiple vector databases
- generic MLOps platforms

unless an experiment proves they solve an actual bottleneck.

This is particularly important because the user's model constraint is explicitly:

> Antigravity CLI + Gemini 3.8 Flash High

and not Claude/GPT/Pro/external LLMs. 

---

# 26. Additional capabilities not taken directly from the catalog

Rather than adding another huge external framework, create these as **local Antigravity skills**:

```text
kaggle-capability-refresh
kaggle-rule-auditor
kaggle-submission-auditor
dataset-fingerprint
validation-veto
leakage-auditor
experiment-scheduler
experiment-replay
artifact-contract-validator
private-lb-robustness
ensemble-diversity
resource-budgeter
skill-evolution
agent-performance-evaluator
knowledge-promoter
```

These are more valuable than installing another generic agent framework because they encode the actual competition-specific control logic.

---

# 27. Directory structure

The final repository should look like this:

```text
kaggle-autonomous-os/
│
├── .agents/
│   ├── agents/
│   │   ├── commander.md
│   │   ├── competition-intelligence.md
│   │   ├── data-forensics.md
│   │   ├── validation-architect.md
│   │   ├── leakage-compliance.md
│   │   ├── model-strategist.md
│   │   ├── feature-strategist.md
│   │   ├── experiment-manager.md
│   │   ├── ensemble-strategist.md
│   │   ├── error-analyst.md
│   │   ├── kaggle-executor.md
│   │   └── artifact-memory.md
│   │
│   ├── skills/
│   │   ├── kaggle-capability-refresh.md
│   │   ├── validation-veto.md
│   │   ├── leakage-auditor.md
│   │   ├── experiment-protocol.md
│   │   ├── artifact-contract.md
│   │   ├── private-lb-robustness.md
│   │   ├── compute-optimization.md
│   │   └── evidence-promotion.md
│   │
│   └── rules.md
│
├── agents/
├── orchestration/
│   ├── commander.py
│   ├── scheduler.py
│   ├── router.py
│   ├── state_machine.py
│   └── recovery.py
│
├── schemas/
│   ├── experiment.schema.json
│   ├── artifact.schema.json
│   ├── agent_result.schema.json
│   ├── knowledge.schema.json
│   └── submission.schema.json
│
├── capabilities/
│   ├── kaggle.json
│   ├── accelerators.json
│   ├── cli.json
│   └── rules.json
│
├── memory/
│   ├── working/
│   ├── project/
│   ├── strategic/
│   └── meta/
│
├── evaluation/
│   ├── agent/
│   ├── scheduler/
│   ├── validation/
│   ├── experiments/
│   └── system/
│
├── templates/
│   ├── competition/
│   ├── notebook/
│   ├── experiment/
│   └── reports/
│
├── scripts/
│   ├── init_competition.py
│   ├── refresh_capabilities.py
│   ├── ingest_artifacts.py
│   ├── validate_artifact.py
│   ├── build_submission.py
│   └── benchmark_system.py
│
├── policies/
│   ├── security.yaml
│   ├── compliance.yaml
│   ├── compute.yaml
│   ├── stopping.yaml
│   └── promotion.yaml
│
└── competitions/
    └── <competition-slug>/
        ├── competition.yaml
        ├── data/
        ├── src/
        ├── notebooks/
        ├── validation/
        ├── experiments/
        ├── artifacts/
        ├── reports/
        ├── knowledge/
        ├── ensemble/
        ├── .blackboard/
        └── README.md
```

This separates the reusable operating system from individual competitions, as required by the specification. 

---

# 28. Configuration

Example:

```yaml
system:
  name: kaggle-autonomous-os
  version: "0.1.0"

model:
  provider: antigravity
  model: gemini-3.8-flash-high
  effort: high

objective:
  primary: expected_private_lb
  secondary:
    - human_effort
    - compute_efficiency
    - reproducibility

autonomy:
  default: high
  max_parallel_workers: 6
  max_nested_agents: 3

experiments:
  max_active: 6
  max_total_compute_hours: 100
  minimum_expected_value: 0.05

validation:
  require_approval: true
  require_leakage_audit: true

submission:
  public_lb_weight: 0.10
  require_final_audit: true

memory:
  promotion_threshold: 0.75
  stale_after_days: 180

evolution:
  auto_promote_skills: false
  auto_modify_orchestrator: false
```

---

# 29. Initialization workflow

User supplies:

```text
COMPETITION=<slug>
PROJECT_DIRECTORY=<path>
```

Then:

```text
0. inspect machine
1. verify agy
2. verify Gemini model
3. verify Kaggle CLI
4. verify authentication
5. refresh Kaggle capabilities
6. create competition project
7. download data
8. inspect rules
9. fingerprint dataset
10. construct validation
11. establish baseline
```

The system should only ask the human when an actual human-only operation is needed, consistent with the supplied workflow. 

---

# 30. Example Antigravity bootstrap

The verified CLI supports headless execution and model selection. ([Google Antigravity][2])

Conceptually:

```bash
agy -p "
Initialize the autonomous Kaggle competition system.

Competition: march-machine-learning-mania-2026
Project: ./competitions/march-machine-learning-mania-2026

Do not begin expensive modeling.

First:
1. inspect environment
2. refresh Kaggle capabilities
3. inspect competition rules
4. inspect data
5. construct validation strategy
6. produce baseline plan
7. report blockers
" \
--model gemini-3.8-flash-high \
--effort high
```

The exact CLI model slug is verified against current Antigravity documentation. ([Google Antigravity][2])

---

# 31. Competition execution loop

The complete autonomous loop becomes:

```text
BOOT
 ↓
CAPABILITY REFRESH
 ↓
COMPETITION INTELLIGENCE
 ↓
DATA FORENSICS
 ↓
VALIDATION VETO
 ↓
BASELINE
 ↓
HYPOTHESIS GENERATION
 ↓
EXPERIMENT VALUE RANKING
 ↓
PARALLEL WORKERS
 ↓
LOCAL TEST
 ↓
KAGGLE EXECUTION
 ↓
ARTIFACT INGESTION
 ↓
RESULT ANALYSIS
 ↓
ERROR ANALYSIS
 ↓
ENSEMBLE SEARCH
 ↓
ADVERSARIAL REVIEW
 ↓
STRATEGY UPDATE
 ↓
STOPPING CHECK
 ├── continue → experiment scheduler
 └── finish → final audit
 ↓
SUBMISSION
 ↓
POST-SUBMISSION ANALYSIS
 ↓
KNOWLEDGE PROMOTION
 ↓
SKILL/AGENT EVALUATION
```

This is essentially the prompt's requested:

`OBSERVE → HYPOTHESIZE → PLAN → PARALLELIZE → IMPLEMENT → EXECUTE → MEASURE → CRITIQUE → SELECT → MEMORIZE → GENERALIZE → EVOLVE`

with deterministic gates added between each phase. 

---

# 32. Model selection strategy

The system should maintain a **model portfolio**, not a single "best model."

For tabular:

```text
baseline:
  logistic / linear
  HistGB
  CatBoost
  LightGBM
  XGBoost

then:
  RealMLP
  TabM
  specialized models
```

Recent 2026 competitions demonstrate that:

- classical GBDTs remain highly competitive
- modern tabular neural networks can contribute
- different model families often improve an ensemble
- target encoding and representation diversity can matter
- a complex model isn't automatically better

Examples include 3rd-, 9th-, 12th- and 25th-place solutions using combinations of GBDTs, neural tabular models, target encoding and ensemble strategies. ([Kaggle][10])

---

# 33. Ensemble policy

Do not:

```text
train 50 models
average everything
```

Instead:

```text
candidate models
      ↓
OOF quality
      ↓
prediction correlation
      ↓
error correlation
      ↓
diversity contribution
      ↓
blend candidate
      ↓
robustness test
```

A model is accepted into the ensemble only if it provides at least one of:

1. superior standalone performance
2. meaningful prediction diversity
3. subgroup improvement
4. robustness improvement
5. computationally cheap complementary signal

---

# 34. Domain-specific strategy

## Tabular

Primary attack surface:

```text
data understanding
→ leakage
→ grouping
→ target encoding
→ categorical treatment
→ feature interactions
→ GBDT diversity
→ neural tabular diversity
→ ensemble
```

---

## Time series

Primary attack surface:

```text
forecast horizon
→ temporal CV
→ seasonality
→ leakage
→ lag structure
→ global/local structure
→ hierarchical relationships
→ model ensemble
```

Never random-shuffle temporal data unless the competition structure demonstrably permits it.

---

## Vision

Primary attack surface:

```text
label quality
→ split strategy
→ pretrained backbone
→ crop/view strategy
→ augmentation
→ fine-tuning
→ TTA
→ ensemble
```

The 2026 DINOv3 first-place example demonstrates that backbone choice, crop strategy, LLRD and ensemble diversity can all matter. ([Kaggle][6])

---

## NLP

Primary attack surface:

```text
data quality
→ label consistency
→ pretrained model
→ tokenizer
→ class imbalance
→ CV
→ fold ensemble
→ post-processing
```

The 2026 mammography winner is an excellent warning against unnecessary complexity: several more elaborate attempts did not beat a disciplined BERT-based approach. ([Kaggle][7])

---

## Multimodal

Use:

```text
modality specialists
      ↓
representation validation
      ↓
late fusion
      ↓
intermediate fusion only if justified
      ↓
ensemble
```

---

# 35. AutoML policy

Do **not** make an external AutoML framework the brain of the system.

Instead:

```text
Antigravity
  = strategy / hypothesis / orchestration

Python
  = deterministic experimentation

Kaggle
  = compute

AutoML
  = bounded experiment generator
```

This preserves interpretability and lets the scheduler terminate low-value searches.

---

# 36. HPO strategy

Use staged optimization.

### Stage 1

Coarse:

```text
10–30 configurations
```

### Stage 2

Promising region:

```text
20–50 configurations
```

### Stage 3

Only if expected value remains high:

```text
fine search
```

Never run giant grid searches by default.

The scheduler must compare:

```text
HPO expected gain
vs
new model family
vs
feature hypothesis
vs
ensemble improvement
```

---

# 37. Pseudo-labeling

Pseudo-labeling is **conditional**, not a default.

Require:

```text
high model confidence
+
distribution similarity
+
validation evidence
+
stability across models
```

Then compare:

```text
baseline
vs
pseudo-label candidate
```

using untouched validation.

---

# 38. External data

The system treats external data as a special high-risk experiment.

Pipeline:

```text
discover
 ↓
license
 ↓
competition permission
 ↓
provenance
 ↓
train/test accessibility
 ↓
leakage analysis
 ↓
value estimate
 ↓
experiment
```

Kaggle explicitly states that external data is only permitted where competition rules allow it. ([Kaggle][17])

---

# 39. Final submission gate

No direct:

```text
agent says submit
→ submit
```

Instead:

```text
COMMANDER
    ↓
FINAL AUDITOR
    ↓
VALIDATION AUDITOR
    ↓
LEAKAGE AUDITOR
    ↓
COMPLIANCE AUDITOR
    ↓
REPRODUCIBILITY AUDITOR
    ↓
SUBMISSION FILE CHECK
    ↓
SUBMIT
```

The adversarial reviewer has veto authority, exactly as required by the source specification. 

---

# 40. Final auditor checklist

```text
[ ] correct competition
[ ] correct metric
[ ] correct submission columns
[ ] correct row count
[ ] no NaNs where forbidden
[ ] no duplicated IDs
[ ] inference preprocessing matches training
[ ] features available at inference
[ ] no leakage
[ ] validation is defensible
[ ] external data compliant
[ ] internet policy compliant
[ ] model license compliant
[ ] runtime acceptable
[ ] memory acceptable
[ ] reproducible
[ ] artifacts stored
[ ] experiment registered
[ ] final candidate justified
[ ] no public-LB overfitting evidence
```

---

# 41. Self-evolution architecture

The system evolves at **three different speeds**.

## Fast: competition strategy

Minutes/hours.

```text
experiment → strategy
```

## Medium: skills

Days/weeks.

```text
repeated failure
→ generalized procedure
→ skill candidate
→ A/B test
```

## Slow: architecture

Across competitions.

```text
system version A
→ benchmark
→ system version B
→ statistically/evidentially better?
→ promote
```

This separation prevents the system from rewriting its own brain every time an experiment fails.

---

# 42. Agent evolution

For every agent:

```yaml
agent_id: ensemble-strategist

tasks: 43

success_rate: 0.88
useful_discovery_rate: 0.67
false_positive_rate: 0.11
compute_waste_rate: 0.08
reproducibility: 0.96

downstream_effect:
  positive: 19
  neutral: 21
  negative: 3
```

Then routing can adapt.

For example:

```text
high-performing ensemble strategist
→ more ensemble tasks

poor feature strategist on time-series
→ reduced authority
→ stronger verifier
```

But the orchestrator itself does not automatically rewrite.

---

# 43. Skill evolution

The promotion rule should require:

```text
candidate skill
+
>= N evidence cases
+
measurable improvement
+
no major regression
+
adversarial review
+
A/B comparison
```

Example:

```text
3 competitions:
target encoding frequently causes leakage
        ↓
repeated failure pattern
        ↓
"fold-safe-target-encoding" skill
        ↓
benchmark
        ↓
promote
```

This is much safer than autonomous skill installation.

---

# 44. Context engineering

Each worker receives:

```text
SYSTEM RULES
+
ROLE
+
COMPETITION BRIEF
+
DATA FINGERPRINT
+
VALIDATION CONTRACT
+
RELEVANT MEMORY
+
TASK SPEC
+
RELEVANT ARTIFACTS
```

It does **not** receive:

```text
all historical experiments
all conversations
all repository files
all Kaggle knowledge
```

Retrieval score should combine:

$$
relevance \times evidence\_strength \times recency \times scope\_match
$$

This is a better use of Gemini's large context than indiscriminate context stuffing.

---

# 45. Security model

There are four boundaries.

## Agent boundary

Agents get only required tools.

Antigravity supports explicit subagent tool configuration and inherited permissions. ([Google Antigravity][1])

## File boundary

Experiments use isolated worktrees.

## Kaggle boundary

Executor alone gets Kaggle credentials.

## Knowledge boundary

Agents cannot directly promote knowledge.

Only the memory curator can.

---

# 46. Credential policy

Never place:

```text
KAGGLE_API_TOKEN
GEMINI_API_KEY
```

inside:

- Git
- prompts
- experiment artifacts
- logs
- memory
- notebooks unless required by Kaggle runtime mechanism

Antigravity supports headless execution, while its authentication documentation also supports API-key-based operation for non-interactive environments. ([Google Antigravity][18])

---

# 47. Git strategy

Use:

```text
main
│
├── experiment/exp-001
├── experiment/exp-002
├── experiment/exp-003
└── release/submission-v1
```

Workers get isolated worktrees.

Successful experiments merge into main.

Failed experiments remain historical evidence.

Never delete failures merely because they were unsuccessful.

Failures are part of the knowledge base.

---

# 48. Benchmark methodology

This is essential.

You cannot claim the autonomous architecture works merely because it successfully writes code.

Create three benchmark classes.

## A. Agent benchmark

Tasks:

- identify leakage
- choose CV
- interpret dataset
- design experiment
- diagnose failure
- select ensemble

Metrics:

```text
accuracy
false-positive rate
hallucination rate
time
token usage
tool efficiency
```

---

## B. Kaggle replay benchmark

Use historical competitions where final solutions are known.

Give the system only information available at the appropriate historical point.

Measure:

```text
baseline score
system score
top-10 distance
top-1 distance
compute
human interventions
time-to-good-solution
```

This is the most meaningful architecture test.

---

## C. Architecture A/B

Compare:

```text
A = single Gemini agent
B = supervisor + workers
C = hierarchical blackboard system
```

Same:

- competition
- compute
- data
- model
- time budget

Measure private/public/CV performance where historical data permits.

---

# 49. System-level KPIs

The main metric should be:

$$
K =
\frac{
\text{expected private-LB improvement}
}{
\text{human hours} + \alpha\text{compute cost}
}
$$

Secondary:

- time to baseline
- time to medal-quality solution
- experiments/hour
- useful discoveries/hour
- failed experiment percentage
- recovery rate
- leakage detection recall
- reproducibility rate
- memory retrieval usefulness
- skill improvement rate

---

# 50. Implementation phases

## Phase 1 — Core

Build:

- project structure
- commander
- experiment schema
- blackboard
- Git integration
- Gemini configuration
- Kaggle CLI wrapper

---

## Phase 2 — Evidence

Build:

- data forensics
- validation architecture
- leakage checks
- artifact contract
- experiment registry

---

## Phase 3 — Autonomous experiments

Build:

- scheduler
- dynamic workers
- worktree isolation
- resource budgets
- result ingestion

---

## Phase 4 — Kaggle automation

Build:

- notebook generation
- kernel metadata
- push
- execution
- artifact retrieval
- submission workflow

---

## Phase 5 — Ensemble

Build:

- OOF store
- correlations
- blending
- stacking
- hill climbing
- robustness selection

---

## Phase 6 — Memory

Build:

- project memory
- strategic memory
- meta memory
- evidence promotion
- contradiction detection

---

## Phase 7 — Evolution

Build:

- agent metrics
- skill candidates
- A/B testing
- architecture benchmarking

---

## Phase 8 — Production hardening

Build:

- compliance
- credential isolation
- recovery
- capability refresh
- stopping rules
- final auditor

---

# 51. What should remain deterministic

This distinction is extremely important.

### Gemini should decide

- hypotheses
- strategies
- experiment design
- interpretation
- routing
- prioritization

### Python should decide

- metrics
- CV splits
- checksums
- artifact validation
- resource measurements
- ensemble calculations
- statistical tests

### Kaggle should decide

- actual remote execution
- competition scoring
- official submission results

### Git should decide

- source version
- provenance
- experiment lineage

The model should never be the sole authority for facts that can be computed.

---

# 52. What is genuinely evidence-backed vs inferred

## Strong evidence

**Antigravity capabilities**

- asynchronous subagents
- custom agents
- workspace/global skills
- plugins
- MCP
- hooks
- headless operation
- Gemini 3.8 Flash High availability

These are directly documented. ([Google Antigravity][1])

**Kaggle automation**

- kernel push/pull/output/status
- competition CLI
- notebook-based execution
- competition-specific restrictions

These are directly documented. ([GitHub][12])

**Kaggle performance patterns**

Recent competition writeups demonstrate:

- GBDT diversity
- target encoding
- OOF blending
- stacking
- hill climbing
- pretrained vision models
- TTA
- domain-specific feature engineering
- careful validation
- occasional simple-model wins

These are Kaggle-proven examples, not universal laws. ([Kaggle][10])

---

## Strong engineering inference

These are architectural recommendations derived from the evidence:

- hierarchical orchestration
- blackboard state
- dynamic experiment workers
- validation veto
- deterministic artifact contracts
- evidence-weighted memory
- capability refresh
- experiment EV scheduling
- controlled skill evolution

They are **design conclusions**, not claims that one published paper has proven this exact architecture wins Kaggle competitions.

---

## Promising but unproven

- automatically learning the optimal agent topology
- dynamically changing agent authority based on downstream LB impact
- cross-competition meta-learning that reliably improves future medals
- automatically generating new ML skills
- learning public-LB reliability weights
- using an internal knowledge graph to materially improve leaderboard performance

These should be benchmarked rather than assumed.

---

# 53. Known limitations

### 1. Gemini is still one reasoning model

Multiple agents are **multiple contexts**, not independent intelligence.

This architecture mitigates correlated errors but does not eliminate them.

---

### 2. No architecture guarantees #1

The prompt correctly says top-1 should be an optimization target, not a promise. 

---

### 3. Domain insight remains difficult

A generic autonomous system may miss a deep domain-specific insight that an expert competitor recognizes immediately.

---

### 4. Kaggle rules change

That is why the capability layer is mandatory.

---

### 5. Compute remains the ultimate bottleneck

Agent reasoning is comparatively cheap relative to large-scale model training.

The scheduler therefore needs to optimize **Kaggle compute**, not just LLM tokens.

---

### 6. Public LB can corrupt the system

The system must actively resist submission-as-search.

---

### 7. Self-evolution can become self-deception

This is why promotion must require empirical evidence and controlled A/B tests.

---

# 54. The final operating philosophy

The system should **not** behave like:

> "I have 30 agents, therefore I am intelligent."

It should behave like:

```text
I have an uncertain hypothesis.
        ↓
I quantify its value.
        ↓
I test it cheaply.
        ↓
I validate the test.
        ↓
I record exactly what happened.
        ↓
I compare it against alternatives.
        ↓
I update my beliefs.
        ↓
I decide what experiment is now most valuable.
```

That is the core competitive advantage.

The supplied specification describes this as accumulating **validated decision-making ability**, rather than merely accumulating information. 

---

# 55. Final recommended architecture

If I had to reduce the entire research to one concrete implementation decision, it would be:

```text
                    GEMINI 3.8 FLASH HIGH
                             │
                             ▼
                     KAGGLE COMMANDER
                             │
                ┌────────────┴────────────┐
                │                         │
          STRATEGY STATE             BLACKBOARD
                │                         │
                └────────────┬────────────┘
                             ▼
                   EXPECTED-VALUE SCHEDULER
                             │
                  ┌──────────┼──────────┐
                  ▼          ▼          ▼
              WORKER A   WORKER B   WORKER C
                  │          │          │
                  └──────────┼──────────┘
                             ▼
                     DETERMINISTIC ML
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
                 OOF/CV          KAGGLE EXEC
                    │                 │
                    └────────┬────────┘
                             ▼
                      ARTIFACT STORE
                             │
                 ┌───────────┼───────────┐
                 ▼           ▼           ▼
             ERROR       ENSEMBLE    ADVERSARY
             ANALYST      ENGINE       REVIEW
                 └───────────┼───────────┘
                             ▼
                    VALIDATION VETO
                             │
                             ▼
                       FINAL AUDIT
                             │
                             ▼
                        SUBMISSION
                             │
                             ▼
                  EVIDENCE-WEIGHTED MEMORY
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
               STRATEGY UPDATE    SKILL UPDATE
                    │                 │
                    └────────┬────────┘
                             ▼
                         NEXT CYCLE
```

**This is the architecture I would actually build.**

It is Antigravity-native because it uses the platform's custom agents, asynchronous subagents, worktrees, skills, plugins, hooks, MCP and headless execution rather than recreating those mechanisms externally. ([Google Antigravity][1])

It is Kaggle-native because its control loop revolves around **competition rules → validation → experiments → Kaggle execution → artifacts → OOF/LB evidence → ensemble → submission**, rather than being a generic coding-agent framework. That directly satisfies the central requirement of the supplied specification. 

And, most importantly, **the system's unit of intelligence is not the agent—it is the verified experiment and the validated knowledge extracted from it.**

### Key current references

- [Google Antigravity CLI — Headless mode](https://www.antigravity.google/docs/cli/headless/)
- [Google Antigravity — Subagents](https://www.antigravity.google/docs/subagents)
- [Google Antigravity — Plugins & Skills](https://www.antigravity.google/docs/cli/plugins)
- [Gemini 3.8 Flash documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)
- [Official Kaggle CLI](https://github.com/Kaggle/kaggle-cli)
- [Kaggle Notebook documentation](https://www.kaggle.com/docs/notebooks)
- [Kaggle Competition documentation](https://www.kaggle.com/docs/competitions)

[1]: https://antigravity.google/docs/subagents "https://antigravity.google/docs/subagents"
[2]: https://www.antigravity.google/docs/cli/headless/ "https://www.antigravity.google/docs/cli/headless/"
[3]: https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash?authuser=0 "https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash?authuser=0"
[4]: https://arxiv.org/abs/2401.05998 "https://arxiv.org/abs/2401.05998"
[5]: https://www.kaggle.com/competitions/playground-series-s6e3/writeups/9th-place-solution "https://www.kaggle.com/competitions/playground-series-s6e3/writeups/9th-place-solution"
[6]: https://www.kaggle.com/competitions/multi-view-pig-posture-recognition/writeups/public-1st-place-solution-dinov3-vit-huge-llrd "https://www.kaggle.com/competitions/multi-view-pig-posture-recognition/writeups/public-1st-place-solution-dinov3-vit-huge-llrd"
[7]: https://www.kaggle.com/competitions/spr-2026-mammography-report-classification/writeups/1st-place-solution "https://www.kaggle.com/competitions/spr-2026-mammography-report-classification/writeups/1st-place-solution"
[8]: https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/writeups/arthur-suilin-1st-place-solution "https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/writeups/arthur-suilin-1st-place-solution"
[9]: https://arxiv.org/abs/2508.13932 "https://arxiv.org/abs/2508.13932"
[10]: https://www.kaggle.com/competitions/playground-series-s6e2/writeups/3rd-place-solution "https://www.kaggle.com/competitions/playground-series-s6e2/writeups/3rd-place-solution"
[11]: https://github.com/Kaggle/kaggle-cli "https://github.com/Kaggle/kaggle-cli"
[12]: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md "https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md"
[13]: https://www.kaggle.com/competitions/playground-series-s6e6/writeups/top-63-solution-surviving-the-shake-up-with-p "https://www.kaggle.com/competitions/playground-series-s6e6/writeups/top-63-solution-surviving-the-shake-up-with-p"
[14]: https://github.com/Kaggle/kaggle-cli/blob/main/CHANGELOG.md "https://github.com/Kaggle/kaggle-cli/blob/main/CHANGELOG.md"
[15]: https://www.kaggle.com/docs/notebooks "https://www.kaggle.com/docs/notebooks"
[16]: https://www.antigravity.google/docs/cli/plugins "https://www.antigravity.google/docs/cli/plugins"
[17]: https://www.kaggle.com/docs/competitions "https://www.kaggle.com/docs/competitions"
[18]: https://www.antigravity.google/docs/cli/install/ "https://www.antigravity.google/docs/cli/install/"
