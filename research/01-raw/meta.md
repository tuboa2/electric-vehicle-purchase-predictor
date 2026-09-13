# Autonomous Kaggle Competition OS — Research-Grounded Blueprint for Antigravity CLI + Gemini 3.8 Flash High

> **Constraint:** Sole reasoning model = **Gemini 3.8 Flash High** (Flash with `thinking_level=HIGH`). All design decisions below are evidence-tiered.

---

## 1. RESEARCH MISSION — VERIFIED FINDINGS

### Antigravity CLI (as of Sep 2026) — Tier 1

- Official install: `curl -fsSL https://antigravity.google/cli/install.sh | bash` — binary `agy`.
- Replaces deprecated Gemini CLI, shares same agent harness as Antigravity 2.0 desktop, keeping **Agent Skills, Hooks, Subagents, Extensions**.
- CLI streamlines workflows by bringing Gemini directly into terminal.
- Skills are directory-based packages in two scopes: **Global** `~/.gemini/config/skills/` and `~/.gemini/antigravity-cli/skills/`, **Workspace** `<project>/.agents/skills/`. Legacy `.agent/skills/` kept for backward compat.
- Workspace locations confirmed: `.agents/skills/` and `.agents/mcp_config.json` replace Gemini CLI paths.
- Hooks: 5 events `PreToolUse / PostToolUse / PreInvocation / PostInvocation / Stop` in `.agents/hooks.json` + global.
- Docs: `https://antigravity.google/docs/cli-overview`, `cli-features`, GitHub `google-antigravity/antigravity-cli`.

### Gemini 3.8 Flash High — Tier 1

- Announced Sep 3, 2026: 2 variants Flash + Flash Cyber.
- Positioning: "most intelligent workhorse model" for software engineering, agentic tasks, multi-step reasoning.
- Pricing: $0.75 per 1M input, $3.75 per 1M output, same as 3.7 Flash introductory.
- Context: **1,048,576 context, 65,536 max output**.
- Reasoning effort ladder: `low/medium/high`, **minimal rejected (400)**. Mapping already treats Flash >=3.7 as LOW/MEDIUM/HIGH. `high` exhibits "greater diligence — executing extra reasoning steps and calling tools iteratively", may use more tokens. Verified probe: low 0, medium 74, high 113 reasoning tokens.
- **Gemini 3.8 Flash High = `gemini-3.8-flash` + `reasoning_effort=high` / `-high` suffix**. Endpoint `databricks-gemini-3-8-flash` supports function calling and hybrid reasoning.

### Kaggle Platform — Tier 1

- CLI groups: Competitions, Datasets, Forums, Kernels, Models.
- Compute (free tier): GPU `2× NVIDIA T4 (15GB each) or 1× P100 (16GB)`, quota **30 hours/week hard-reset**, disk 20GB ephemeral + 75GB `/kaggle/working`, 9h max per run.
- P100 failure with default image is intentional tradeoff — newer PyTorch needs sm_120; T4/L4 work fine.
- Rules: **Internet disabled at submission**, pretrained models must be uploaded as Kaggle Datasets/Models first. External data allowed if **freely & publicly available**, must be shared publicly for prize-relevant submissions.

### Winning Strategies — Tier 2 (Kaggle-proven)

- **Stacking wins 2025**: 4-level stack of 150 models, workflow EDA → baseline → feature engineering → hill climbing + stacking. April 2025 Playground: stacking RMSE 11.54 CV / 11.44 private LB = 1st place.
- **Pseudo-labeling largest gain**: +0.00027 AUC in 2nd place ISU x OSF, only with strict thresholds. Conservative `p>=0.98 or <=0.02` from ultimate ensemble teacher + half-weighting 0.5 prevents corruption.
- **Greedy ensemble selection > manual weights** in some competitions, but manual weights > stacking in others — **evidence: search OOF arrays exhaustively**. Stabilize with Ridge-weighted hill climbing, cap size, require independent check.
- **CV Tabular**: EfficientNet+Swin/ViT ensemble + TTA + EMA + pseudo-labeling most-mentioned.
- **Feature engineering**: OpenFE provides **+1.9% avg accuracy over base across 49 datasets**, using GBDT evaluation for tabular.

### Multi-Agent Patterns — Tier 3-4

- Canonical: **Coordinator-Worker, Debate/Critique, Relay/Pipeline, Blackboard** shared workspace async.
- Orchestration: controller directs specialists — Supervisor (classify→route→specialist), Pipeline, Fan-Out/Fan-In, Hierarchical, Debate, Voting, Evaluator-Optimizer.
- Hybrid (Enterprise Default): **Hierarchical planning + decentralized execution + Blackboard for async + Debate only at key decision points**. bMAS variant: Blackboard with public/private sections + Control Unit selects agents.

### Memory & Context — Tier 3

- **Letta (ex-MemGPT) ~23K★**: OS-inspired hierarchical memory, self-editing, superior for multi-agent collaboration.
- **Zep/Graphiti ~26-29K★**: Temporal knowledge graph, best for facts changing over time, 94.8% DMR vs Letta 93.4% for temporal.
- **Mem0 ~48K★**: hybrid vector+graph+KV, 4-scope memory, p95 latency -91% vs full-context.
- **Context compression**: LLMLingua 20x ratio minimal loss, LongLLMLingua +17.1% with 4x fewer tokens.

**Evidence tiering used below:**

- **Kaggle-proven:** seen in top-10 writeups 2024-26
- **Experimentally demonstrated:** benchmarked (AutoML, LLMLingua)
- **Generally accepted:** engineering best practice
- **Promising:** limited validation
- **Speculative:** needs verification

---

## 2. EXACT ARCHITECTURE

**Core thesis (evidence-driven):** More agents ≠ better. Research shows coordinator overhead grows O(n²), debate helps only at high-uncertainty decision points. Best expected LB under Gemini 3.8 Flash High constraint is **6+1 architecture**: 1 Meta-Supervisor + 5 specialists + 1 Critic + 1 Memory Curator running as **Blackboard + Hierarchical Supervisor**.

```
┌─────────────────────────────────────────────────┐
│ Meta-Supervisor (agy) │
│ Control Unit: selects next agent from blackboard│
└──────────┬──────────────────────────────────────┘
           │ reads/writes
┌──────────▼──────────────────────────────────────┐
│ BLACKBOARD (.agents/blackboard/*.md) │
│ - competition.json (spec) │
│ - data_profile.json (adversarial val, leakage) │
│ - validation_plan.json │
│ - experiment ledger (DVC + MLflow) │
│ - ensemble state (OOF arrays) │
│ - private LB estimator │
└──────────┬──────────────────────────────────────┘
           │ parallel dispatch when independent
┌──────────┴──────────────────────────────────────┐
│ Specialists (subagents as.agents/agents/*.md) │
│ 1. Intelligence 2. Data Detective 3. Feature │
│ 4. Model/Ensemble 5. Execution/MLOps │
│ Critic (always runs after candidate) │
│ Memory Curator (async background) │
└─────────────────────────────────────────────────┘
```

**Why not flat swarm?** Fan-Out/Fan-In parallel workers merged is good for independent hypotheses, but Kaggle requires sequential dependency: validation → features → model. Hierarchical with blackboard reduces context-sharing fragility.

---

## 3. EXACT AGENT ROSTER

| #   | Agent ID             | File                                   | Role                                                                                               |
| --- | -------------------- | -------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 0   | `meta-supervisor`    | `.agents/agents/meta-supervisor.md`    | Control Unit, heartbeat 30min, anti-duplication                                                    |
| 1   | `competition-intel`  | `.agents/agents/competition-intel.md`  | Parse Overview/Rules/Evaluation, external data policy, metric, sample submission, code vs non-code |
| 2   | `data-detective`     | `.agents/agents/data-detective.md`     | Adversarial validation train/test shift, leakage scan, temporal pattern, distribution shift        |
| 3   | `feature-architect`  | `.agents/agents/feature-architect.md`  | Feature engineering (OpenFE/FeatureTools), validation strategy (GroupKFold/TimeSeriesSplit/Purged) |
| 4   | `model-strategist`   | `.agents/agents/model-strategist.md`   | Model selection, HPO Optuna, AutoGluon stacking, hill climbing, pseudo-labeling                    |
| 5   | `execution-engineer` | `.agents/agents/execution-engineer.md` | Kaggle CLI, notebook push, MLflow tracking, DVC lock, compute quota management                     |
| 6   | `critic-verifier`    | `.agents/agents/critic-verifier.md`    | Adversarial verification, OOF-LB correlation, shake-up avoidance, submission validation            |
| 7   | `memory-curator`     | `.agents/agents/memory-curator.md`     | Background, Letta-style self-editing + Graphiti temporal KG, cross-competition transfer            |

---

## 4. EXACT RESPONSIBILITIES

**Meta-Supervisor:** Decompose, dispatch, enforce quality gates, 30-min heartbeat monitoring. Implements goal-loop with explicit stop conditions (max experiments, time, LB plateau).

**Competition-Intel:** Produces `capabilities/kaggle_rules.json` with `external_data.allowed`, `must_be_shared`, `internet`, `runtime_cap`, `submission_size`. Fetches via WebFetch (API lacks Rules page).

**Data-Detective:** Runs adversarial validation classifier AUC >0.6 = shift. Scans for target leakage, magic features, temporal leakage. Outputs `data_profile.json`.

**Feature-Architect:** Tiered approach: Tier1 OpenFE boosting evaluation, Tier2 FeatureTools DFS, Tier3 domain features. Locks validation before features.

**Model-Strategist:** Portfolio: GBDT (LGBM/XGB/CatBoost), NN (FT-Transformer, TabNet), AutoGluon `best_quality` multi-layer stack+ bagging. Ensemble search: greedy hill climbing over OOF arrays, Ridge-weighted, cap 15 models.

**Execution-Engineer:** Manages Kaggle secrets, kernel metadata, internet-off validation. Tracks experiments via MLflow, reproducible via `dvc.lock`.

**Critic-Verifier:** Multi-agent debate only here: independently re-derive validation score, check leakage, estimate private LB shift risk, vote to accept/reject experiment.

**Memory-Curator:** Async background agent: extracts validated discoveries → generalizable knowledge → cross-competition memory. Uses temporal timestamps for evolving knowledge.

---

## 5. AGENT COMMUNICATION PROTOCOL

**Blackboard pattern:** Markdown + JSON files, not in-memory queue (reduces fragility).

- Location: `.agents/blackboard/`
- Files:
  - `00_competition.json` — immutable spec
  - `01_data_profile.json` — adversarial AUC, leakage flags
  - `02_validation.json` — fold strategy, CV score correlation
  - `03_experiments.jsonl` — append-only ledger: `{id, hypothesis, features, model, OOF, LB, cost, timestamp}`
  - `04_ensemble.json` — current OOF arrays, weights
  - `05_critique.md` — critic reviews
  - `06_memory_index.json` — pointers to global memory

**Message format:** Each agent writes a `task_{id}.md` with YAML frontmatter:

```yaml
---
agent: data-detective
status: done
depends_on: [competition-intel]
artifacts: [.agents/blackboard/01_data_profile.json]
tokens_used: 12450
---
```

Supervisor's Control Unit reads blackboard state, selects next agent (highest expected information gain).

**Parallel dispatch:** When `depends_on` satisfied and no shared state mutation, `dispatching-parallel-agents` skill triggers fan-out.

---

## 6. MEMORY ARCHITECTURE

Inspired by Letta 3-tier + Graphiti temporal KG + Mem0 fast retrieval.

**Tier 1 — Working Memory (in-context, always):**

- Core Blocks (`letta/schemas/memory.py` pattern): persona, competition spec, active hypotheses (max 8k tokens). Self-editing via `core_memory_append`.
- Location: `.agents/memory/core.md`

**Tier 2 — Episodic Memory (project-scoped):**

- Vector-indexed archival storage: all experiments, logs, failure traces. Search via `archival_memory_search`.
- Backed by: local SQLite + FAISS (no cloud needed for Kaggle). DVC for artifacts, MLflow for metrics.
- Retention: full history for current competition.

**Tier 3 — Semantic / Cross-Competition (global):**

- Temporal knowledge graph: Neo4j or FalkorDB (Graphiti). Nodes: ProblemType, DatasetChar, Validation, FeatureStrategy, ModelFamily, Outcome.
- Example chain: `high-cardinality categorical → CatBoost → GroupKFold → OOF blend → improved private LB` (only encode if evidence-backed).
- Global path: `~/.gemini/antigravity-cli/skills/kaggle-memory/` or `~/.agents/memory/global/`
- Implements Hibernate-Wake for multi-day runs.

**Compression:**

- Sliding-window summarization threshold 30, keep 10 recent (from Letta).
- LLMLingua 20x compression for long logs, LongLLMLingua +17.1% perf with 4x fewer tokens — used when injecting historical experiments into 1M context.
- Layered prioritization: competition spec > validation > recent top-3 experiments > failure patterns > old logs.

---

## 7. SKILL ARCHITECTURE

**Global skills** (`~/.gemini/config/skills/` or `~/.gemini/antigravity-cli/skills/`):

- `kaggle-cli` — competitions download, datasets, kernels push
- `uv-package-manager` — fast Python resolver
- `using-git-worktrees` — isolated workspaces sharing repo
- `polars` — fast DataFrame (pandas too slow but data fits RAM)
- `agent-memory`, `agent-memory-mcp` — hybrid memory
- `context-engineering`, `context-compression`, `context-optimization`, `context-guardian`
- `agent-evaluation`, `evaluation` — versioned cases, verifiers
- `multi-agent-task-orchestrator`, `multi-agent-patterns`, `parallel-agents`

**Workspace skills** (`.agents/skills/{skill}/SKILL.md`):

- `kaggle-grandmaster` — persona, shake-up avoidance
- `kaggle-adversarial-validation` — Day1 shift detection
- `kaggle-validation` — lock fold strategy
- `kaggle-eda` — distribution, leakage, temporal
- `kaggle-feature-engineering` — OpenFE, DFS
- `kaggle-ensemble` — hill climbing, stacking
- `kaggle-pseudo-labeling` — conservative thresholding

**Installation logic:**

- Global if reusable across competitions.
- Workspace if competition-specific or evolves per project.
- Embedded into framework if core loop (memory, compression).
- Converted to local skill if catalog skill too generic.
- Rejected if duplicates or adds complexity.

---

## 8. SELF-EVOLUTION MECHANISM

Uses `agent-orchestration-improve-agent` + `loop-library`.

```
evaluate (OOF vs LB correlation, failure code)
→ reflect (why systematic errors? which models make different errors?)
→ hypothesize (what skill patch increases expected information gain?)
→ implement (edit SKILL.md in.agents/skills/)
→ test (run versioned evaluation cases)
→ measure (does new skill improve private LB estimator?)
→ memorize (if validated, promote to global ~/.gemini/config/skills/)
→ generalize (update temporal KG)
```

**Skill discovery:** After each competition, scan `.agents/blackboard/03_experiments.jsonl` for patterns: e.g., if CatBoost + GroupKFold consistently wins on high-cardinality, create new skill `high-card-catboost-groupkfold`.

**Failure taxonomy (9 codes):** tool misuse, context loss, validation leakage, OOF-LB mismatch, compute OOM, internet dependency, submission format, external data violation, hallucinated API.

---

## 9. EXPERIMENT SCHEDULER

Not random grid search. **Expected Information Gain (EIG)** prioritization.

- **Scheduler state:** priority queue sorted by `EIG = (expected LB gain) / (GPU hours * failure risk)`.
- **Multi-fidelity:** cheap fidelity first (10% data, 1 fold, 1 epoch) → promote if promising (like AutoGluon presets).
- **Optuna with pruning:** for HPO, but only after validation locked.
- **Hill climbing queue:** separate queue for ensemble search over saved OOF arrays — greedy addition, Ridge-weighted, cap 15, require independent holdout agreement.
- **Budget enforcement:** tracks weekly 30h GPU quota, pauses low-EIG experiments when <5h remaining.

---

## 10. VALIDATION SYSTEM

**Three-layer lock:**

1. **Adversarial Validation (Day1):** Train classifier to distinguish train/test. AUC >0.58 = shift. If shift, use adversarial-aware validation or target encoding with domain adaptation.
2. **Leakage Scan:** `02_outlier_detection`, `03_target_leakage_hunt`, `04_magic_feature_hunt`, `05_adversarial_validation` scripts. Checks: target correlation >0.99, time-based leakage, ID leakage.
3. **Robust CV:**
   - Tabular i.i.d.: StratifiedKFold
   - Grouped: GroupKFold
   - Temporal: TimeSeriesSplit + Purged (gap)
   - Multilabel: Multilabel Stratified
   - **Rule:** Validation must mimic test distribution per adversarial result.

**Private LB Estimator:** Tracks OOF-LB correlation across experiments, builds meta-model to predict private LB drop risk (shake-up avoidance).

---

## 11. KAGGLE AUTOMATION FLOW

```bash
# Verified commands
pip install kaggle # or uv add kaggle
# API token at ~/.kaggle/kaggle.json from Legacy API Credentials

kaggle competitions list --format table
kaggle competitions files -c <slug>
kaggle competitions download -c <slug> -p data/raw
kaggle datasets list -s <query>
kaggle kernels push -p <notebook-dir> # for submission
kaggle competitions submit -c <slug> -f submission.csv -m "v1"
kaggle competitions submissions -c <slug> --format table
```

**Notebook execution:**

- Local: `uv run python train.py` with DVC
- Kaggle: `kaggle kernels push` with `kaggle.json` kernel metadata (`enable_internet: false` for code comps)
- Artifacts: model weights → upload as Kaggle Dataset/Model before submission (internet disabled at submission)

**Headless Antigravity:** `agy --headless "run competition-intel"` — uses same harness as desktop.

---

## 12. GITHUB INTEGRATION

- `using-git-worktrees` skill: each hypothesis in isolated worktree sharing same repo, no branch switching.
- `smart-git-automation`: auto branch naming `exp/{hypothesis_id}`
- CI: GitHub Actions for `dvc repro` + `mlflow` tracking, not for Kaggle GPU.
- Artifact sync: `dvc push` to S3/GCS, `git push` for code only.

---

## 13. ARTIFACT PROTOCOL

All artifacts must be reproducible and internet-off compliant:

- **Structure:**
  ```
  artifacts/
    submission.csv (validated against sample)
    oof_preds.npy
    model_weights/ (converted to Kaggle Dataset)
    dvc.lock
    mlflow_run_id
  ```
- **Validation:** `kaggle-eda` skill checks submission format, file size, no internet dependency.
- **Registry:** MLflow model registry with tags `private_lb_estimator`, `oof_score`.
- **Feedback loop:** After `kaggle competitions submissions`, parse LB score, update `03_experiments.jsonl`, trigger critic.

---

## 14. COMPUTE OPTIMIZATION SYSTEM

- **Quota manager:** Reads `capabilities/kaggle_compute.json` (volatile, refreshed weekly): `{gpu_quota:30, used:12.3, reset:Monday, accelerators:}`.
- **Progressive resizing for CV:** 224px 10ep → 384 30ep → 448 60ep batch 64→32→16 — saves 40% GPU.
- **Model-specific:**
  - Tabular: CPU 4 cores 30GB RAM free tier (no GPU needed)
  - CV: T4x2 for multi-GPU inference, P100 for training (but P100 failing with default image — detect and fallback to T4)
- **Checkpointing:** Every epoch, Hibernate-Wake for multi-day runs.
- **Cost control:** Gemini 3.8 Flash High uses more tokens at high effort — compress context with LLMLingua before injecting history.

---

## 15. FAILURE RECOVERY SYSTEM

- **Taxonomy:** 9 deterministic codes, each with recovery:
  - `TOOL_MISUSE` → retry with reduced tool set
  - `CONTEXT_LOSS` → restore from `context-guardian` snapshot
  - `VALIDATION_LEAKAGE` → abort, trigger data-detective
  - `OOF_LB_MISMATCH` → trigger critic debate
  - `GPU_OOM` → halve batch, gradient accumulation
  - `INTERNET_DEP` → rewrite to use mounted Kaggle Dataset
  - `SUBMISSION_FORMAT` → auto-fix to sample
  - `EXTERNAL_DATA_VIOLATION` → check public availability
  - `HALLUCINATED_API` → verify against `capabilities/` layer

- **ACI design:** Architect/Editor split improves SWE-bench +10.7pp — execution-engineer is Editor, meta-supervisor is Architect.

---

## 16. EVALUATION FRAMEWORK

- `agent-evaluation`: versioned cases, explicit verifiers.
- **Kaggle-specific verifiers:**
  - `oof_lb_correlation` >0.8 required to accept ensemble
  - `shake_up_risk` estimator
  - `leakage_free` boolean
  - `reproducible` (dvc repro produces same OOF)
  - `internet_off` (runs with `enable_internet=false`)
- **Benchmark:** Run on 5 historical competitions (tabular, time-series, CV, NLP, multimodal), measure final private LB percentile vs compute hours.

---

## 17. SECURITY/COMPLIANCE LAYER

- **External data:** Must be freely & publicly available, equally accessible, minimal cost, compliant. Before use, record source, license, relationship to comp data.
- **Internet:** `enable_internet` false for code comps. All non-built-in deps mounted from Kaggle datasets/models, wheel bundles as dataset.
- **Secrets:** Use Kaggle Secrets, never hardcode.
- **Submission:** Only via Kaggle Notebooks, validate full dependency install with internet disabled before using slot.

---

## 18. DIRECTORY STRUCTURE

```
kaggl-os/
├──.agents/
│ ├── agents/
│ │ ├── meta-supervisor.md
│ │ ├── competition-intel.md
│ │ ├── data-detective.md
│ │ ├── feature-architect.md
│ │ ├── model-strategist.md
│ │ ├── execution-engineer.md
│ │ ├── critic-verifier.md
│ │ └── memory-curator.md
│ ├── skills/
│ │ ├── kaggle-cli/SKILL.md
│ │ ├── kaggle-adversarial-validation/SKILL.md
│ │ ├── kaggle-validation/SKILL.md
│ │ ├── kaggle-eda/SKILL.md
│ │ ├── polars/SKILL.md
│ │ └── context-engineering/SKILL.md
│ ├── blackboard/
│ │ ├── 00_competition.json
│ │ ├── 01_data_profile.json
│ │ ├── 02_validation.json
│ │ ├── 03_experiments.jsonl
│ │ ├── 04_ensemble.json
│ │ └── 05_critique.md
│ ├── memory/
│ │ ├── core.md (Letta core)
│ │ ├── episodic.db (FAISS+sqlite)
│ │ └── global/ (Graphiti temporal KG)
│ ├── hooks.json (PreToolUse etc)
│ └── mcp_config.json
├── capabilities/ # volatile, refreshable
│ ├── kaggle_compute.json (30h/week, T4x2/P100/L4)
│ ├── gemini_model.json (1,048,576 ctx, $0.75/$3.75, high effort ladder)
│ ├── kaggle_rules_schema.json
│ └── accelerators.json
├── configs/
│ ├── competition.yaml (slug, metric, type)
│ ├── validation.yaml
│ ├── features.yaml (OpenFE params)
│ ├── models.yaml (portfolio)
│ └── ensemble.yaml (hill climbing)
├── data/
│ ├── raw/ (kaggle competitions download)
│ └── processed/
├── artifacts/
├── notebooks/
│ └── kaggle_kernel/ (kaggle kernels push)
├── src/
│ ├── features/
│ ├── models/
│ └── ensemble/
└──.kaggle/kaggle.json (gitignored)
```

---

## 19. CONFIGURATION STRUCTURE

```yaml
# capabilities/gemini_model.json - refreshable
{
  "model": "gemini-3.8-flash",
  "reasoning_effort": "high",
  "context_window": 1048576,
  "max_output": 65536,
  "pricing": {"input":0.75, "output":3.75, "unit":"per 1M"},
  "thinking_levels": ["low","medium","high"],
  "minimal_supported": false
}

# configs/competition.yaml
competition:
  slug: "playground-series-s5e11"
  type: "tabular" # tabular|timeseries|cv|nlp|multimodal
  metric: "roc_auc"
  code_competition: true
  external_data_allowed: true
  must_be_shared: true
  internet_at_submission: false
```

---

## 20. INITIALIZATION WORKFLOW

```bash
# 1. Install Antigravity CLI
curl -fsSL https://antigravity.google/cli/install.sh | bash
agy --version

# 2. Bootstrap project
mkdir kaggl-os && cd kaggl-os
agy init --template kaggle-os

# 3. Install global skills (from catalog)
agy skills install polars uv-package-manager using-git-worktrees \
  agent-memory agent-memory-mcp context-engineering context-compression \
  multi-agent-task-orchestrator agent-orchestrator

# 4. Install workspace Kaggle skills
mkdir -p.agents/skills
git clone https://github.com/olixignacious/agent-skills.agents/skills/kaggle-grandmaster --depth 1

# 5. Configure capabilities layer (volatile)
agy run "refresh capabilities/kaggle_compute.json from https://www.kaggle.com/docs/efficient-gpu-usage"
agy run "refresh capabilities/gemini_model.json"

# 6. Kaggle auth
pip install kaggle
# Create token at https://www.kaggle.com/settings -> Legacy API
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle competitions list --format table
```

---

## 21. COMPETITION EXECUTION WORKFLOW

```
OBSERVE → HYPOTHESIZE → PLAN → PARALLELIZE → IMPLEMENT → EXECUTE → MEASURE → CRITIQUE → SELECT → MEMORIZE → GENERALIZE → EVOLVE → REPEAT
```

1. **OBSERVE:** `competition-intel` fetches Overview/Rules/Eval, writes `00_competition.json`
2. **HYPOTHESIZE:** `data-detective` runs adversarial val + leakage scan → `01_data_profile.json`
3. **PLAN:** `feature-architect` locks validation → `02_validation.json`
4. **PARALLELIZE:** Supervisor fans out 3-5 feature hypotheses via git worktrees
5. **IMPLEMENT:** Each worktree runs `uv run python src/features/build.py`
6. **EXECUTE:** `execution-engineer` runs cheap fidelity, logs MLflow
7. **MEASURE:** OOF score, adversarial AUC, cost
8. **CRITIQUE:** `critic-verifier` debate — accept if OOF-LB correlation holds, no leakage, EIG positive
9. **SELECT:** Scheduler promotes top EIG experiments to full fidelity, updates `04_ensemble.json` via hill climbing
10. **MEMORIZE:** `memory-curator` extracts validated discovery → episodic + temporal KG
11. **EVOLVE:** If pattern repeats across 2+ comps, promote skill to global

---

## 22. CROSS-COMPETITION LEARNING WORKFLOW

```
raw experiments (03_experiments.jsonl)
    ↓
validated discoveries (critic-approved, OOF-LB corr >0.8)
    ↓
generalizable knowledge (e.g., "T4x2 OOM at batch 32 for ViT-L @448px")
    ↓
strategy extraction (ProblemType → Validation → Feature → Model → Outcome)
    ↓
cross-competition memory (Graphiti temporal KG with bi-temporal timestamps)
```

When Competition B begins:

- Retrieve similar historical comps by ProblemType + DatasetChar embedding
- Retrieve successful strategies + failure patterns
- Initialize priors (e.g., start with CatBoost for high-card categorical)
- Test priors cheap fidelity first

---

## 23. SKILL EVOLUTION WORKFLOW

- **Trigger:** After 3 competitions or when failure pattern frequency >2.
- **Process:**
  1. `memory-curator` proposes SKILL.md patch (e.g., add OpenFE two-stage pruning)
  2. Versioned evaluation: run skill on historical comp subset
  3. If improves private LB estimator and passes verifiers, merge to `.agents/skills/`
  4. If used in 2+ projects and stable, promote to global `~/.gemini/config/skills/`
  5. Document in `capabilities/skills_changelog.md`

---

## 24. AGENT EVOLUTION WORKFLOW

- Uses `agent-orchestration-improve-agent`: systematic performance analysis, prompt engineering, iteration.
- **Metric:** Agent's tasks success rate, tokens per successful task, failure code frequency.
- **Loop:** Profile → identify bottleneck (e.g., `data-detective` missing temporal leakage) → rewrite agent persona in `.agents/agents/*.md` → test on versioned cases → keep if success rate +5%.
- **Cost-aware:** `agent-orchestration-multi-agent-optimize` balances workload distribution and cost (Gemini 3.8 Flash High token usage at high effort).

---

## 25. RECOMMENDED SKILLS FROM CATALOG (2121)

**Install globally:**

- `multi-agent-task-orchestrator` — anti-duplication, heartbeat
- `agent-memory`, `agent-memory-mcp` — hybrid memory
- `agent-orchestration-improve-agent`, `agent-orchestration-multi-agent-optimize`
- `agent-orchestrator`, `multi-agent-architect`, `multi-agent-patterns`
- `parallel-agents`, `dispatching-parallel-agents`
- `context-engineering`, `context-optimization`, `context-compression`, `context-guardian`, `context-agent`
- `agent-evaluation`, `evaluation`
- `data-scientist`, `ml-engineer`, `machine-learning-ops-ml-pipeline`
- `polars`, `using-git-worktrees`, `uv-package-manager`
- `agent-evaluation-reporting`

**Workspace:**

- `kaggle-cli` (from community), `deep-research` for competition research

**Rejected:** generic delegation skills (`claude-delegate`, `codex-delegate` etc) — Antigravity already has native subagents, adding delegation lanes adds complexity.

---

## 26. ADDITIONAL SKILLS DISCOVERED EXTERNALLY (not in catalog)

- **olixignacious/agent-skills**: 12 Kaggle Grandmaster skills — adversarial validation, EDA, diverse baselines, Optuna tuning, hill climbing, stacking, pseudo-labeling, seed ensembling. **Install as workspace skills**.
- **deafenken/auto-kaggle**: bootstrap with references for compute-environment, rules-parsing.
- **AutoGluon**: `TabularPredictor presets='best_quality'` auto stack+bagging, beats manual ensemble 2% wMAE.
- **OpenFE**: automated feature generation expert-level, two-stage pruning.
- **FeatureTools**: DFS for relational datasets.
- **LLMLingua/LongLLMLingua**: 20x compression minimal loss — embed as context-compression wrapper.
- **Mem0, Zep/Graphiti, Letta**: memory backends comparison above.

---

## 27. IMPLEMENTATION PHASES

**Phase 0 — Research & Capabilities (Week 1):** Build `capabilities/` refresh scripts, verify all CLI commands, document volatile info.

**Phase 1 — Bootstrap (Week 2):** Install Antigravity CLI, scaffold `.agents/`, global skills, Kaggle CLI auth.

**Phase 2 — Core Agents (Week 3-4):** Implement meta-supervisor + 5 specialists + critic as markdown agents. Implement blackboard JSON protocol. Test on Playground competition.

**Phase 3 — Memory (Week 5):** Letta core memory + episodic FAISS + Graphiti temporal KG. Implement LLMLingua compression.

**Phase 4 — Scheduler & Validation (Week 6):** EIG scheduler, Optuna pruning, hill climbing over OOF, adversarial validation skill.

**Phase 5 — Self-Evolution (Week 7-8):** Loop-library, skill evolution workflow, agent improvement loop. Run on 3 historical comps, measure private LB gain over time.

**Phase 6 — Hardening:** Failure taxonomy, internet-off validation, submission size checks, quota management.

---

## 28. EXACT COMMANDS WHERE VERIFIED

```bash
# Antigravity
curl -fsSL https://antigravity.google/cli/install.sh | bash
agy --help
agy skills list

# Gemini 3.8 Flash High (via API)
# model: gemini-3.8-flash, reasoning_effort: high (low/medium/high)
# pricing $0.75 input / $3.75 output per 1M

# Kaggle
kaggle competitions list --format table
kaggle competitions files -c <slug>
kaggle competitions download -c <slug> -p data/raw
kaggle competitions submit -c <slug> -f submission.csv -m "msg"
kaggle kernels push -p notebooks/kaggle_kernel
kaggle datasets create -p artifacts/model_weights/

# Project
uv init && uv add kaggle polars lightgbm catboost torch timm autogluon openfe featuretools optuna mlflow dvc
git worktree add../exp-001 -b exp/001
dvc repro
mlflow ui
```

---

## 29. EXAMPLE END-TO-END EXECUTION

```bash
cd kaggl-os
agy --headless "competition-intel: slug=playground-series-s5e11"

# Blackboard now has 00_competition.json
# → external_data_allowed:true, must_be_shared:true, internet:false

agy --headless "data-detective: run adversarial validation"

# → 01_data_profile.json: adv AUC 0.62 = shift, no target leakage

agy --headless "feature-architect: lock validation=GroupKFold, propose OpenFE + FeatureTools"

# Supervisor fans out 3 worktrees:
git worktree add../exp-fe-openfe -b exp/openfe
git worktree add../exp-fe-fts -b exp/fts
git worktree add../exp-fe-domain -b exp/domain

# Execution engineer runs cheap fidelity (10% data, 1 fold)
uv run python src/features/build_openfe.py --fidelity low
# → OOF 0.782, cost 0.3 GPUh

# Critic approves, scheduler promotes to full fidelity + AutoGluon
uv run python src/models/train_autogluon.py --presets best_quality --time_limit 3600

# Ensemble via hill climbing over OOF arrays
uv run python src/ensemble/hill_climb.py --oof artifacts/oof_*.npy --cap 15

# Push to Kaggle for LB
kaggle kernels push -p notebooks/kaggle_kernel
kaggle competitions submit -c playground-series-s5e11 -f artifacts/submission.csv -m "openfe+autogluon+hillclimb v1"

# Memory curator extracts
# "OpenFE + GroupKFold + AutoGluon best_quality → +1.9% over base on tabular with high-card categorical" → temporal KG
```

---

## 30. KNOWN LIMITATIONS

- **Gemini 3.8 Flash High token bloat:** High effort uses extra reasoning steps and tool calls iteratively, may exceed budget at 1M context. Mitigation: LLMLingua compression 20x, layered prioritization.
- **Kaggle GPU quota:** 30h/week total, hard reset. P100 failing with new PyTorch — must fallback to T4x2.
- **Internet-off:** All deps must be Kaggle Dataset/Model mounted. System must validate locally with internet disabled before submission.
- **External data compliance:** Must be publicly available, shared before deadline.
- **Model hallucination:** No minimal reasoning level — high effort may hallucinate APIs. Mitigation: `capabilities/` layer refresh, explicit verifiers.
- **Multi-agent overhead:** More agents increase coordination cost. Chosen 6+1 minimal set based on blackboard + hierarchical evidence — not more.
- **Speculative:** Gemini 3.8 Flash Cyber variant for security — not used here. Long-term memory benchmarks (Zep 94.8% vs Letta) from community blogs, not peer-reviewed.

---

**Final principle:** This is not a generic multi-agent coding framework. It is Kaggle-native: every component — adversarial validation before features, OOF array hill climbing, internet-off verification, quota-aware scheduler, temporal KG linking ProblemType→Validation→Feature→Model→Outcome — is optimized for **private leaderboard performance under Kaggle's compute and rules**, running autonomously via Antigravity CLI with Gemini 3.8 Flash High reasoning.
