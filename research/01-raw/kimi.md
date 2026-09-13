# KAGGLE-OS
## A Self-Evolving, Reusable Autonomous Kaggle Competition System for Antigravity CLI + Gemini 3.8 Flash High

**Version:** 1.0 · **Date:** 2026-09-13 · **Status:** Research-validated blueprint (Phase 0–2 ready to implement)

---

# PART A — RESEARCH REPORT

## 1. Executive Summary

KAGGLE-OS is a competition-agnostic, self-evolving multi-agent framework that runs on the Antigravity CLI with Gemini 3.8 Flash High as its sole reasoning model. The design is deliberately **not** a swarm. The evidence (Section 6) shows single agents match or beat multi-agent systems on ~64% of tasks, multi-agent coordination degrades sequential reasoning by 39–70%, and a single model debating itself produces sycophancy cascades, not truth. The winning architecture is therefore a **two-tier hierarchical supervisor with a file-based blackboard**: one Commander owning strategy and stopping decisions, ~9 specialized workers, a shared structured state directory, and adversarial verification only on the critical path (validation design and pre-submission). Parallelism is applied *only* to genuinely independent ML experiments, which is exactly where supervisor fan-out is proven to help (+80% on parallel tasks).

The system's edge comes not from agent count but from: (a) a rigorous validation-first gate, (b) an experiment registry with expected-value scheduling, (c) machine-readable Kaggle artifact loops via `kaggle kernels push/output`, (d) a four-layer memory that accumulates validated cross-competition knowledge, and (e) controlled skill/agent evolution with A/B benchmarks. All volatile facts (Kaggle quotas, accelerator lists, CLI syntax) live in a refreshable `capabilities/` layer, never hardcoded in the core.

## 2. Current Antigravity Capabilities (verified, Tier 1)

- **Headless execution:** `agy -p "<prompt>"` with `--output-format text|json|stream-json`; `--json-schema` enforces structured output; `--model`, `--effort low|medium|high`, `--agent`, `--sandbox`, `--print-timeout` (default 5m); headless runs are stateless unless `--continue`/`--conversation` is used. Unknown model slugs **fail loudly with a non-zero exit** (verified behavior — safe for pinned pipelines).
- **Streaming NDJSON control:** `stream-json` emits `init` / `step_update` / `result` events; `step_update` carries `tool_info` (name, parameters, output, error) and `subagent_info` (type_name, role, conversation_id, log_uri). An external Python orchestrator can drive multi-turn sessions by holding stdin open and dispatching on the `event` field.
- **Async subagents:** the shared Antigravity harness supports asynchronous subagents for parallel background work; the Agent Manager (Mission Control) delegates to specialized subagents (browser, terminal) without blocking the main conversation.
- **Plugins:** namespaced bundles staged at `~/.gemini/antigravity-cli/plugins/<name>/` containing `plugin.json` (required), `mcp_config.json`, `hooks.json`, `skills/`, `agents/`, `rules/`. Hooks intercept pre/post tool execution (ideal for artifact validation gates).
- **Skills:** workspace skills under `.agents/skills/`; global skills under `~/.gemini/antigravity-cli/skills/` (auto-imported as slash commands in every workspace).
- **Rules/context:** `GEMINI.md` per project for persistent role/rules; settings in `~/.gemini/antigravity-cli/settings.json` (includes default `model`).
- **Surfaces:** CLI (terminal-first), Antigravity 2.0 desktop (visual orchestration, cron sidecars, worktrees), and a Python SDK. KAGGLE-OS targets the CLI surface first, desktop second.
- **Known limitation (verify at install time):** per the vendor forum, custom subagents have shared the main session's model and per-subagent model pinning was still a roadmap item as of mid-2026. KAGGLE-OS therefore treats subagents as *parallel context-isolated workers on the same model* and compensates via role prompts, `--json-schema` contracts, and bounded task envelopes rather than model-tier routing.

## 3. Gemini 3.8 Flash High Profile

- Third Flash release in six weeks (after 3.6/3.7); positioned as the daily workhorse of Antigravity; "often approaching the performance of higher-cost frontier models." Adjustable thinking levels per task; cost $0.75/1M input, $3.75/1M output tokens.
- **Design consequences:** cheap enough for many short, scoped calls — favor *many small verified calls* over few giant prompts; `--effort high` reserved for Commander planning and adversarial review; `--effort low/medium` for extraction, summarization, and artifact parsing. Context engineering (retrieval, compression, structured state) is the compensation mechanism for reasoning depth, per the mission constraint.

## 4. Kaggle Infrastructure Constraints (Tier 1 + verified secondary)

| Resource | Current value (2026) | Volatility |
|---|---|---|
| Session runtime | 12h CPU/GPU, 9h TPU | Medium — verify per competition |
| Weekly GPU quota | ~30h guaranteed, floats higher (40h reported) | High |
| Weekly TPU quota | ~20h | High |
| Accelerators (CLI IDs) | NvidiaTeslaT4(+Highmem), NvidiaL4(X1), NvidiaTeslaA100, NvidiaH100, NvidiaRtxPro6000, TpuV5E8, TpuV6E8, TpuV3(8/1VmV3)8, NvidiaTeslaP100 | High — some restricted per competition |
| P100 caveat | Default image's PyTorch (cu128) lacks Pascal sm_60 kernels: `torch.cuda.is_available()` is True but first CUDA op fails | Engineering pitfall — avoid P100 unless torch rebuilt |
| Notebook output dir | up to 20GB persisted (`/kaggle/working`) | Low-Medium |
| Competition restriction axes | CPU/GPU runtime, internet access, external data (rules-gated) | Per competition |
| Secrets | Kaggle Secrets + `UserSecretsClient` (web UI only, no CLI) | Low |
| CLI verbs | `kaggle competitions {list,files,download,submit,submissions,leaderboard}`, `datasets {list,download,create,version}`, `kernels {init,push,pull,output,status,files}` | Low-Medium |

GPU acceleration does **not** benefit most pandas/sklearn workflows — accelerator selection must be workload-driven (GBDT on GPU where supported, CV/transformers on GPU/TPU, EDA/feature-gen on CPU with Polars).

## 5. Kaggle Winning-Strategy Evidence Synthesis

**Experimentally demonstrated / Kaggle-proven (Tier 1–2):**
- **Ensembling + seed diversity + retrain-on-100%:** NVIDIA Grandmasters' Playbook: multi-seed XGBoost ensembles clearly beat single-seed; final retrain on full data adds LB gains.
- **Multi-round pseudo-labeling with soft labels:** strongest models as teachers; k sets of pseudo-labels to avoid leakage when using k-fold; filter low-confidence; fine-tune on original data last. Proven across BirdCLEF and fertilizer challenges.
- **Adversarial validation:** train/test separability AUC ≈ 0.5 ⇒ same distribution; AUC ≫ 0.5 ⇒ covariate shift; feature importances of the adversarial model identify drifting features; can also prune train rows predicted as "train-like" (Tier 2, fastai lineage).
- **Diverse ensembles beat big singles:** recent top solutions (e.g., BirdCLEF 2026 2nd place: diverse ensemble + pseudo-labeling; LLM comps: fine-tune + multi-dataset pseudo-labeling + TTA).
- **Tabular foundation models are now first-class:** TabPFN-2.5 reports 100% win rate vs default XGBoost on ≤10k-sample classification; TabICL v2 (fully open source, regression support), TabFM lead 19-dataset benchmarks; **TFM errors are uncorrelated with GBDT errors — ideal ensemble diversity**. Caveat: RealTabPFN weights are non-commercial — license check is mandatory per competition.
- **Gradient-boosted trees remain the scale king** for large tabular data; AutoGluon-class stacking still strong when compute allows.

**Generally accepted engineering practice:** robust CV (group/temporal/stratified chosen by structure), OOF discipline, TTA for vision, leakage audits, reproducible seeds, Parquet/Arrow caches, Polars lazy frames for in-RAM work.

**Promising but weakly validated:** tabular foundation models at >100k rows in competition settings; LLM-generated feature hypotheses ranked by adversarial validation.

**Speculative (do not auto-promote):** fully autonomous HPO with LLM priors beating Optuna on budget; self-evolving prompts improving LB without benchmark evidence.

## 6. Multi-Agent Architecture Comparison (10 architectures)

| # | Architecture | Evidence | Verdict for KAGGLE-OS |
|---|---|---|---|
| 1 | Supervisor | +80% on parallel tasks, −70% on sequential reasoning (Google/Openlayer synthesis); predictable, linear coordination overhead | **Adopt (top tier)** |
| 2 | Hierarchical supervisor | Fits decomposition boundaries; medium overhead scaling with depth | **Adopt (2 tiers only — deeper = complexity without evidence)** |
| 3 | Manager-worker | Same as supervisor in single-model setting | Adopt (merged into #1/#2) |
| 4 | Blackboard/shared-memory | LbMAS: token-economical, competitive with SOTA static/dynamic MAS; risk: read-write conflicts | **Adopt (file-based blackboard with lock discipline, not in-memory)** |
| 5 | Parallel specialist swarm | Helps only for genuinely independent tasks; quadratic coordination otherwise | **Adopt narrowly (independent experiment fan-out via Agent Manager + git worktrees)** |
| 6 | Debate | Reduces hallucinations vs single pass, but sycophancy cascading; limit ≤3 agents | **Reject as a loop; use one-shot critic pass instead** |
| 7 | Critic-generator | Maker-checker at 40–60% lower cost when roles are asymmetric; here same model, so value = verification prompts, not model asymmetry | **Adopt on critical path only (validation + pre-submission)** |
| 8 | Planner-executor-verifier | Strong fit for phase-gated workflows | **Adopt (maps onto competition playbook phases)** |
| 9 | Evolutionary agent population | Search-based MAS (ADAS/AFlow/MaAS) needs supervised training steps and heavy tokens; unjustified for one-model CLI | **Reject** |
| 10 | Hybrid | Evidence supports combining supervisor + blackboard + bounded critic | **SELECTED** |

**Decision:** Hybrid = two-tier hierarchical supervisor (Commander → cells) + file-based blackboard (`state/`) + planner-executor-verifier phase gates + one-shot adversarial critic on critical paths + narrow parallel fan-out. This is the smallest architecture consistent with the evidence.

## 7. Risks and Expected Bottlenecks

1. **Gemini 3.8 Flash reasoning depth** on long-horizon orchestration → mitigated by decomposition, structured state, `--effort high` only where warranted.
2. **Antigravity custom-subagent model pinning** may be unavailable → verify at install; design degrades gracefully to sequential headless calls.
3. **Kaggle quota variance** → capability-refresh layer + budget-aware scheduler.
4. **CV/LB divergence** → validation gate with adversarial validation; submission-selection policy limiting public-LB overfitting.
5. **Agent hallucination in artifact interpretation** → machine-readable artifacts + schema-validated ingestion + numeric verification in hooks.
6. **Runaway experimentation** → explicit stopping policy + per-run token/$ guardrails.
7. **Code-comp runtime limits** (often 2× runtime on resubmission rules) → runtime budget tracked per experiment.

---

# PART B — IMPLEMENTATION BLUEPRINT (30 deliverables)

## 1. Exact Architecture

```
                         ┌──────────────────────────────┐
                         │  1. COMMANDER (supervisor)   │
                         │  strategy · budget · stop    │
                         └──────────────┬───────────────┘
                                        │  routes via state/blackboard
        ┌───────────────┬───────────────┼───────────────┬────────────────┐
        ▼               ▼               ▼               ▼                ▼
  2. SCOUT        3. FORENSIC      4. VALIDATOR    5. ENGINEER      6. RUNNER
 competition     data forensics   validation      features+models train/eval
 intelligence    EDA reports      architect (VETO) HPO policies   artifact emit
        │               │               │               │                │
        └───────────────┴───────┬───────┴───────────────┴────────────────┘
                                ▼
                    state/  (blackboard: registry, brief,
                                knowledge, budgets, decisions)
                                │
        ┌───────────────────────┼────────────────────────┐
        ▼                       ▼                        ▼
   7. BLENDER             8. ADVERSARY             9. EXECUTOR
   ensemble/hillclimb     leakage·error·rules     kaggle kernels
   stack/blend/OOF        compliance (VETO)       push/status/output
        │                       │                        │
        └───────────────────────┴────────────────────────┘
                                ▼
                    10. CHRONICLER (artifact ingestion,
                    memory curation, strategy evolution)
                                │
                                ▼
                     ┌────────────────────┐
                     │  HUMAN CHECKPOINT  │  (credentials, acceptance,
                     │  only when needed  │   irreversible submits)
                     └────────────────────┘
```

- **Tier 1:** Commander. **Tier 2:** all other agents. No third tier.
- **Blackboard:** `state/` directory (YAML/JSON/Parquet) — single-writer-per-file, atomic writes (write tmp + rename), append-only event log. No shared mutable memory between concurrent agents; parallel work uses git worktrees.
- **Critical path (planner-executor-verifier):** PHASE gates G0–G6 (Section 20) enforced by Commander; Validator and Adversary hold **veto** power.

## 2–4. Exact Agent Roster, Responsibilities, Contracts

All agents are Antigravity **custom agents** (plugin `agents/` dir, Markdown frontmatter: role, tools, skills, output schema). All return the YAML envelope (Section 5). Effort defaults: Commander high; Validator/Adversary high; Scout/Forensic/Engineer/Runner medium; Blender/Chronicler/Executor low-medium.

| Agent | Core responsibilities | Explicit non-goals |
|---|---|---|
| **commander** | Own objective; phase gate decisions; experiment prioritization via EV scheduler; budget accounting (tokens, Kaggle GPU-hours, submission slots); stopping policy; synthesis into `decisions.md` | Never writes model code; never submits |
| **scout** | Parse competition page/rules via web; produce `competition_intelligence.md`; extract metric formula, submission format, external-data/internet/compute restrictions, timeline, prior similar competitions | No modeling advice without evidence labels |
| **forensic** | Schema/type detection (tabular/TS/text/image/audio/graph/multimodal); dataset inventory; missingness/duplicates/cardinality; target analysis; train/test drift report; leakage report; EDA report | May not propose models |
| **validator** | **VETO power.** Design validation strategy (KFold/GroupKFold/TimeSeriesSplit/stratified + purging); adversarial validation harness; fold integrity tests; CV↔LB correlation tracking; must answer "why will CV track private LB?" | Blocks modeling until gate passes |
| **engineer** | Feature hypotheses (targeted, ranked by evidence); model-family selection policy (GBDT / TFM / linear / NN-CV / transformer per modality); preprocessing contract (fit-on-train-only enforcement); HPO configs (Optuna, pruned, budget-capped) | No blind feature explosions (>200 features needs hypothesis log) |
| **runner** | Execute experiments in isolated worktrees; deterministic seeds; emit full artifact contract; runtime/memory logging; failure classification before retry | Never changes validation design |
| **blender** | OOF management; correlation/diversity analysis; greedy/hill-climb ensemble; stacking with nested CV; blend weight robustness (perturbation test); pseudo-labeling loop (k-set, soft) with leakage guard | Cannot add a member without OOF evidence |
| **adversary** | **VETO power.** Leakage hunts (target/temporal/group/contamination); error analysis (worst slices, calibration, residual structure); rules compliance (external data, internet, licenses incl. TFM licenses); pre-submission audit checklist | Attacks the solution, never improves it directly |
| **executor** | Build Kaggle-compatible kernel + `kernel-metadata.json`; `kaggle kernels push/status/output`; `kaggle competitions submit`; submission slot ledger; secrets handling via Kaggle Secrets | No code changes outside packaging |
| **chronicler** | Artifact ingestion & schema validation; experiment registry updates; knowledge compression (fact/pattern/hypothesis/speculation tiers with evidence counts); strategy evolution proposals; agent-performance metrics | Cannot edit running code |

## 5. Agent Communication Protocol

File-first, YAML-envelope second, free text never:

```yaml
# every agent response (enforced via agy --json-schema where headless)
status: success|partial|failed|blocked
confidence: 0.0-1.0
findings:            # max 10 bullets, each with evidence pointer
  - claim: ...
    evidence: path/to/artifact-or-log#line
actions: [...]       # concrete, idempotent commands/files
artifacts: [...]     # machine-readable outputs produced
recommendations: [...]
risks: [...]
next_tasks:          # proposals for Commander's queue only
  - task: ...
    expected_gain: ...
    compute_cost: ...
veto: false          # validator/adversary only
```

Rules: agents communicate through `state/` artifacts; cross-agent chatter prohibited; every claim carries a provenance pointer; Commander is the only agent that may write `state/queue.yaml`.

## 6. Memory Architecture (4 layers)

| Layer | Location | Contents | TTL/compression |
|---|---|---|---|
| **Working** | session context + `state/session.yaml` | current phase, active task, last 3 experiment summaries | reset per session; context-compression discipline (keep <30% of window as history) |
| **Project** | `<project>/knowledge/` + `state/` | competition brief, EDA/validation/leakage reports, experiment registry, OOF store | full fidelity during comp; distilled at close |
| **Strategic** | `~/.gemini/antigravity-cli/skills/kaggle-os-memory/knowledge/` | cross-competition: strategies/, validation/, feature_engineering/, models/, ensembles/, leakage/, kaggle/, compute/, failures/, successes/, playbooks/ | structured YAML records; dedup; stale-mark after 180d; contradiction resolution by evidence rank |
| **Meta** | `knowledge/meta-learning/` + `knowledge/agent-performance/` | agent success rates, false-positive rates, token cost/task, prompt versions, skill versions, A/B results | updated by Chronicler only |

Provenance is mandatory: every strategic record carries `evidence_count`, `competition_types[]`, `confidence`, `counterexamples[]`, `last_validated`. **Speculation never auto-promotes**: hypothesis → experiment → evidence → knowledge. Candidate catalog skills: `agent-memory` (hybrid persistent search), `tree-ring-memory` (recall/evidence/audit lifecycle), `mesh-memory` (semantic MCP recall); embedded as the Chronicler's tooling, chosen ONE, default `agent-memory` pattern (file-based, no external DB).

## 7. Skill Architecture

- **Plugin `kaggle-os`** (global): `agents/` (10 agents), `skills/` (core playbooks), `rules/` (coding + ML rules), `hooks.json` (artifact gate, secret-scan, unslop-commit), optional `mcp_config.json`.
- **Global skills** (competition-agnostic, `~/.gemini/antigravity-cli/skills/`): validation-design, leakage-detection, experiment-management, ensemble-optimization, artifact-analysis, kaggle-compute-optimization, kaggle-cli-automation, adversarial-validation, pseudo-labeling-guardrails.
- **Workspace skills** (`.agents/skills/`, per competition): competition-rules, domain-knowledge, dataset-specific-analysis, competition-playbook.
- **Skill evolution pipeline** (Section 22) never installs unreviewed skills; every skill has version/purpose/trigger/dependencies/evidence/expected_benefit/failure_modes/evaluation_cases.

## 8. Self-Evolution Mechanism

Core loop: `OBSERVE → HYPOTHESIZE → PLAN → PARALLELIZE → IMPLEMENT → EXECUTE → MEASURE → CRITIQUE → SELECT → MEMORIZE → GENERALIZE → EVOLVE → REPEAT`, implemented as the phase machine in Section 20 plus:

- Every finished experiment writes a structured record (Section 14).
- Nightly (or on-idle) **distillation**: raw experiments → validated discoveries (`evidence_count ≥ 2` across contexts) → strategy entries (with priors for next competition initialization).
- **Prior transfer:** at competition start, Scout+Commander retrieve top-k similar historical competitions by problem type × dataset signature × modality → initialize model-family priors, validation priors, feature priors, and known-failure checks.
- **Meta-knowledge graph** (lightweight, file-based): `Competition → ProblemType → DatasetChars → Validation → Features → ModelFamily → Ensemble → Outcome` edges only created when supported by ≥1 validated record; used for retrieval, not inference.

## 9. Experiment Scheduler

Each queued experiment:

```yaml
priority: float           # computed
expected_gain: float      # predicted ΔCV (meta-learning prior + bandit update)
confidence: 0-1
information_gain: 0-1     # reduction in model/validation uncertainty
compute_cost: gpu_hours|cpu_hours|tokens
risk: 0-1                 # leakage/instability/deadline risk
dependencies: [exp_ids]
family: feature|model|hpo|validation|ensemble|postprocess|external
```

`priority = (expected_gain × confidence × information_gain) / (compute_cost × (1 + risk))`

- Priors per family come from meta-memory (e.g., "GBDT→TFM add-on diversity gain" prior from past comps); posteriors updated by results (Thompson-style: sample from posterior, don't argmax — preserves exploration).
- Hard constraints: weekly GPU-hour budget (default 24h of the ~30h quota, reserve 20%), submission-slot budget (default: max 2/day, 10/week unless rules say otherwise), deadline-aware risk ramp.
- Commander re-ranks the queue after every ingestion event; starvation guard: no family waits >3 cycles.

## 10. Validation System

- Artifacts: `validation/strategy.md`, `validation/validation_config.yaml`, `validation/folds/`, `validation/leakage_tests/`.
- **Gate G2 (blocking):** Validator must produce a convincing causal argument CV→private-LB (temporal structure? groups? repeated entities? submission-time distribution?), plus:
  - adversarial validation AUC (target: ≤ 0.55 warn / ≤ 0.70 investigate hard / > 0.70 redesign or distribution-match),
  - fold integrity tests (no entity overlap across folds, no future data in past folds, preprocessing fit inside folds),
  - duplication/entity-resolution audit.
- CV/LB tracking table maintained from day one; if |CV−LB| grows, trigger distribution-shift investigation before further tuning.
- Public LB treated as a noisy sensor: used for detection of pipeline errors, not for gradient descent on submissions.

## 11. Kaggle Automation Flow

```
local project (git worktree) → kernel/ dir (notebook/script + kernel-metadata.json)
  → kaggle kernels push -p kernel/ [--accelerator NvidiaTeslaT4] [--timeout s]
  → poll: kaggle kernels status <kernel>  (until complete/error)
  → kaggle kernels output <kernel> -p artifacts/ --file-pattern ".*\.(csv|json|log)$"
  → Chronicler ingestion → registry update → Commander re-plan
  → (optional) kaggle competitions submit -c <slug> -f submission.csv -m "..."
```

- Accelerators chosen by workload policy (CPU for GBDT-small/Polars prep; T4/L4 default GPU; A100/H100 only if competition grants and workload justifies; avoid P100 per cu128/sm_60 pitfall; TPU v5e/v6e for transformer/vision at scale).
- Runtime caps enforced (`--timeout` + in-notebook watchdog writing partial artifacts every N minutes so 12h kills still yield OOF/submission).
- Secrets only via Kaggle Secrets (`UserSecretsClient`), never in code/git; `hooks.json` secret-scan blocks commits matching credential patterns.

## 12. GitHub Integration

- Branch model: `main` (stable) + `experiments/<exp-id>` (worktree-isolated) → merge via Commander review after artifact verification.
- Auto-maintained: `README.md`, `COMPETITION.md` (brief), `EXPERIMENTS.md` (registry view), `CHANGELOG.md` (unslop-commit style, human-tone).
- `.gitignore`: data/, artifacts/models/, credentials, `*.parquet` caches (unless small and intentional).
- Hooks: pre-commit secret scan; post-merge experiment-registry consistency check.

## 13. Artifact Protocol (stable across competitions)

Every run (local or Kaggle) must emit:

```
artifacts/<exp_id>/
├── experiment.json        # full spec: hypothesis, change, seeds, params, validation ref
├── metrics.json           # cv scores per fold, oof metric, runtime, memory
├── oof_predictions.csv    # (or .parquet)
├── predictions.csv        # test predictions
├── submission.csv         # only when submission-ready
├── resource_usage.json    # wall time, CPU%, peak RAM, GPU-hours
├── logs/                  # training logs, warnings
└── reports/               # importance, curves, error slices, prediction distribution plots
```

Schema enforced by `hooks.json` gate + Chronicler validation; missing artifact ⇒ experiment marked invalid, excluded from blending.

## 14. Experiment Registry Schema

```yaml
experiment:
  id, competition, problem_type, dataset_signature(hash of schema+stats)
  hypothesis, intervention, family
  feature_version, dataset_version, code_commit
  validation: {strategy, n_folds, adversarial_auc}
  model: {family, params, seed}
  results: {cv_mean, cv_std, oof_metric, public_lb, private_lb}
  resources: {runtime_s, peak_ram_mb, gpu_hours, tokens}
  conclusion: {outcome, effect_size, confidence, reproducibility_check}
  promotion: {fact|pattern|hypothesis|speculation}
```

## 15. Compute Optimization System

- **ML compute:** Polars lazy + Parquet/Arrow caches (catalog `polars` skill); categorical/entity indexes; `joblib` parallelism bounded to core count; GBDT GPU backends where net-positive (measure!); feature-store via versioned Parquet; no repeated CSV parsing; incremental dataset versioning via `kaggle datasets version` for >output-limit artifacts.
- **AI compute:** task envelopes ≤ ~4k tokens context; retrieval-not-dump; `--json-schema` outputs (no re-parsing prose); `stream-json` to kill hung runs fast; per-day token budget with `runaway-guard`-style wallet cap; `.geminiignore`-finops discipline on data/model dirs (catalog `geminiignore-finops` skill).
- **Bottleneck benchmarking:** Runner profiles each experiment (cProfile/py-spark-style stage timers on Kaggle); auto-generated `resource_usage.json` feeds the scheduler's compute_cost estimates.

## 16. Failure Recovery (self-healing)

| Class | Signal | Route |
|---|---|---|
| DATA_FAILURE | schema/shape/NaN gates | Forensic re-profile; pin dataset version |
| DEPENDENCY_FAILURE | import/pip errors | uv lockfile pinning; fallback image packages |
| CODE_FAILURE | exception, nonzero exit | 3-attempt budget w/ different hypotheses, then rollback (catalog `break-ai-fix-loops` discipline) |
| MEMORY_FAILURE | OOM, swap thrash | downcast dtypes, Polars streaming, chunking, feature pruning |
| TIMEOUT | watchdog/12h kill | partial-artifact recovery; re-plan with smaller budget |
| KAGGLE_FAILURE | API/quota/verification errors | exponential backoff, queue re-schedule, capability refresh |
| VALIDATION_FAILURE | fold integrity/adversarial AUC regression | Validator veto → back to design |
| MODEL_FAILURE | diverging loss/Nan | seed/config mutation; fallback family |
| AGENT_FAILURE | schema-invalid envelope, hallucinated artifact | output rejected; agent reliability score −; task re-routed |
| ORCHESTRATION_FAILURE | state corruption/lock | event-log replay; atomic-write restore |
| ARTIFACT_FAILURE | missing/invalid contract files | ingestion blocked; run marked invalid |

Never retry identically: every retry must change at least one hypothesis variable and log the delta.

## 17. Evaluation Framework (benchmarking the system itself)

Metrics tracked per competition and in aggregate: time-to-first-baseline, time-to-medal-pace, CV↔LB correlation achieved, final percentile (target bands: top-10%/5%/1%/podium/#1 as attainable), compute efficiency (LB-gain per GPU-hour), token cost per LB-gain, failure-recovery rate, agent false-positive rate (Adversary/Validator vetoes upheld), memory retrieval hit-rate, unnecessary-work ratio (queued experiments never run), reproducibility rate (re-run Δ within noise).

Internal agent benchmark (catalog `agent-evaluation`, `evaluation`, `agent-evaluation-reporting` skills): versioned case suites per agent role (e.g., 20 seeded leakage traps for Adversary; 12 fold-design traps for Validator), run on every prompt/skill change; regression ⇒ change rejected.

## 18. Security/Compliance Layer

- Compliance checklist (Adversary, pre-submission + on external-data use): external-data permission, internet restriction, pretrained-model/TFM license (RealTabPFN = non-commercial), team/merger rules, submission frequency, code-comp runtime rules, reproducibility requirement, credential hygiene.
- Credentials: `~/.kaggle/kaggle.json` local only (gitignored, hook-verified); Kaggle-side secrets via Secrets UI only.
- Sandbox: `--sandbox` for untrusted fetched code (forked public notebooks); no arbitrary network during local runs except allowlisted domains.
- Never probe the private leaderboard beyond allowed submission patterns.

## 19. Reusability / Directory Structure

```
kaggle-os-core/                       # the plugin + templates (global)
├── plugin.json
├── agents/                           # 10 agent definitions
├── skills/                           # core playbooks (validation-design, ...)
├── rules/                            # ml-rules.md, coding-rules.md
├── hooks.json
├── templates/
│   ├── competition-project/          # scaffold copied per competition
│   │   ├── GEMINI.md
│   │   ├── competition.yaml
│   │   ├── kernel/kernel-metadata.json
│   │   ├── src/  notebooks/  data/
│   │   ├── state/  knowledge/  artifacts/  reports/  validation/
│   │   └── experiments/
│   └── capabilities/                 # REFRESHABLE volatile facts
│       ├── kaggle_quotas.md          # quotas, session limits, accelerator IDs
│       ├── antigravity_cli.md        # verified flags/syntax + version
│       └── model_profile.md          # gemini-3.8-flash-high notes
├── policies/                         # scheduler, stopping, budgets (YAML)
└── scripts/
    ├── agy_call.py                   # headless wrapper (json-schema, retry, stream-json)
    ├── kaggle_loop.sh                # push→status→output→ingest
    └── capability_refresh.md         # procedure + sources

competition-<slug>/                   # per-competition workspace
└── (from template; .agents/skills/ for local skills)
```

`capabilities/` is refreshed by Scout at every competition init (never trust hardcoded numbers).

## 20. Configuration Structure + Phase Machine

`competition.yaml`: slug, project_dir, metric, direction, deadline, budgets {gpu_h/week, submissions/week, tokens/day}, risk_profile, modality, rules_digest_ref.

Gates: **G0 init → G1 intelligence → G2 validation design (VETO) → G3 baseline → G4 experimentation loop → G5 ensemble/pseudo-label → G6 adversarial audit (VETO) → submit → ingest → iterate; G7 final audit → final submission → close & distill.** Dynamic skips: modality routing disables irrelevant stages (no CV pipelines for pure tabular, no HPO if baseline within noise of prior-strong, no pseudo-labeling without OOF evidence).

## 21. Initialization Workflow (hands-off)

Input: `COMPETITION`, `PROJECT_DIRECTORY`.
1. Verify `agy` ≥ required version; `agy models` confirms `gemini-3.8-flash-high`; verify custom-agent support; install plugin `kaggle-os`.
2. Env probe: OS, Python, uv, git, kaggle CLI + `~/.kaggle/kaggle.json`, CPU/RAM/GPU.
3. Scaffold project from template; write `competition.yaml`; refresh `capabilities/`.
4. Download competition data via `kaggle competitions download`.
5. Proceed to G1. **Stop only for human-only actions:** accepting rules, credentials, paid decisions.

## 22. Skill Evolution Workflow

`Failure → classify → root cause → generalizable? → draft skill → evaluate (case suite + A/B) → adversarial review → versioned skill → A/B on live tasks → promote/reject`. Skills live in plugin (global) or workspace (local); Chronicler proposes, Commander approves, hook blocks promotion without evaluation record.

## 23. Agent Evolution Workflow

Controlled, never self-modifying core: Chronicler logs per-agent metrics (task success, discovery yield, false-positive rate, wasted compute, tokens/task). Commander adjusts: routing probability, context allowance, verification requirements, parallelism eligibility. Prompt/role changes require A/B vs previous version on the case suite; promote only if superior, else rollback. Versions tracked: `system_version, agent_version, skill_version, prompt_version, experiment_version`.

## 24. Cross-Competition Learning Workflow

End of competition: raw experiments → validated discoveries → strategy extraction → priors updated (with evidence counts) → failure patterns appended → playbook diffs summarized for human review. Start of next: retrieve similar comps → initialize priors → test priors as first experiments (cheap confirmations) → adjust.

## 25. Recommended Skills from the 2,121-Skill Catalog

**Install globally / embed in plugin:**

| Skill | Use in KAGGLE-OS | Disposition |
|---|---|---|
| `multi-agent-task-orchestrator` | Queue discipline, anti-duplication, quality gates for Commander | Embed (inspiration for queue protocol) |
| `agent-memory` | Strategic/meta memory implementation | Embed |
| `agent-memory-mcp` | Optional MCP-backed memory | Evaluate; default file-based |
| `agent-evaluation` + `agent-evaluation-reporting` | Agent case suites + regression reporting | Install global |
| `evaluation` | System-level eval framework | Install global |
| `context-engineering`, `context-compression`, `context-optimization`, `context-guardian`, `context-degradation` | Context architecture + safe compaction | Embed (context policy) |
| `data-scientist`, `ml-engineer`, `scikit-learn` | Role grounding for Engineer/Runner | Embed as role skills |
| `polars` | Compute optimization playbook | Install global |
| `uv-package-manager` | Reproducible envs | Install global |
| `using-git-worktrees` | Parallel experiment isolation | Install global |
| `parallel-agents`, `dispatching-parallel-agents` | Fan-out policy | Embed (narrow) |
| `multi-agent-patterns`, `multi-agent-architect` | Architecture reference | Inspiration only |
| `agent-orchestration-improve-agent`, `agent-orchestration-multi-agent-optimize` | Agent evolution loop | Inspiration only |
| `machine-learning-ops-ml-pipeline`, `data-quality-frameworks` | Pipeline + data validation patterns | Inspiration/embed |
| `optim-agent` | HPO agent discipline | Inspiration for HPO policies |
| `deep-research`, `search-specialist` | Scout research procedure | Embed (Scout) |
| `loop-library` | Bounded loops w/ stop rules | Inspiration (stopping policy) |
| `systematic-debugging`, `break-ai-fix-loops`, `bug-hunter` | Failure recovery | Install global |
| `llm-structured-output` | Envelope/schema contracts | Embed |
| `runaway-guard` | Token/$ budgets | Embed (budget policy) |
| `geminiignore-finops` | Context cost control | Install global |
| `tree-ring-memory` / `mesh-memory` | Memory alternatives | Evaluate one |
| `skill-check`, `effective-agent-skills`, `manage-skills` | Skill lifecycle quality | Install global |
| `unslop-commit` | Commit hygiene | Install global |
| `prompt-engineering` (+ `-patterns`) | Prompt discipline | Inspiration |
| `plotly` | Artifact reports | Optional |

**Reject (with reason):** `langgraph`/`langchain-architecture`/`pydantic-ai`/`autonomous-agents` frameworks (external orchestration duplicates Antigravity harness; violates "Antigravity-native"); `llm-council`/`routerbase-model-gateway`/`unified-ai-gateway`/`sandbase-mcp` (route to non-Gemini models — violates core constraint); `local-llm-expert`/`huggingface-local-models`/`unsloth-finetuning` (local LLMs — violates constraint); `agent-manager-skill`/`tmux` CLI multiplexing (redundant vs Agent Manager); `m365-*`, `azure-*`, `aws-*`, `n8n-*`, Rube/Composio automation (irrelevant SaaS surface); `odw`/`open-dynamic-workflows` (duplicates harness orchestration); `aider/claude/codex/grok/...-delegate` + `dispatch` (delegate to other-model CLIs — violates constraint); persona/debate theater skills (`ilya-sutskever`, `yann-lecun-debate`); health/legal/finance verticals; `design-it/*` UI styles; `kimi-delegate` (other vendor CLI).

*Note: the catalog snapshot provided for this design was truncated mid-way through the `media` category; the disposition table above covers all visible skills plus the high-value skills named in the mission brief. A full catalog pass is scheduled as a Phase-1 task for the Engineer/Chronicler before installation.*

## 26. Additional External Skills/Resources Discovered (not in catalog)

- `kaggle-adversarial-validation` skill (skills.rest) — align with internal validator harness; adopt pattern, not dependency.
- **AutoGluon** (tabular AutoML baseline policy), **Optuna** (pruned HPO), **MAPIE/conformal** (uncertainty for ensemble selection), **polars**, **duckdb** (mid-scale analytics), **kaggle-cli** docs as canonical automation reference. All are ML libraries (allowed — the model constraint applies to *reasoning*, not numeric tooling).

## 27. Implementation Phases

| Phase | Scope | Exit criterion |
|---|---|---|
| **P0 (wk 1)** | Core scaffold: plugin, state/blackboard, artifact schemas, `agy_call.py`, kaggle_loop, capabilities layer, Git template | One `agy -p` headless call with JSON-schema envelope end-to-end |
| **P1 (wk 2–3)** | Agents 1–5 + gates G0–G3; competition intelligence + forensics + validation harness on 2 historical tabular comps (e.g., a Kaggle playground regression + a binary classification) | Baseline + validation gate produces trustworthy CV; artifacts validate |
| **P2 (wk 4–6)** | Agents 6–10, scheduler, memory layers, ensemble engine, failure recovery; run full loop on one live code competition (CPU/tabular) | ≥5-experiment loop with ingestion-driven replanning; zero human artifact handling |
| **P3 (wk 7–10)** | Skill/agent evolution, A/B harness, meta-learning graph; run one vision or NLP competition (GPU budget) | Cross-competition prior retrieval demonstrably biases first experiments correctly |
| **P4 (ongoing)** | Hard: adversarial case suites ≥80% trap detection; benchmark report vs single-agent baseline | Go/no-go for autonomous operation |

## 28. Benchmark Methodology

Baseline comparison (mandatory, per the evidence): **single-agent KAGGLE-OS** (Commander does everything, no fan-out) vs **full multi-agent** on identical competition sets and budgets. Metrics from Section 17. Pre-register: multi-agent wins only if LB-gain/compute and failure-recovery improve without accuracy regression. Benchmark comps: 1 tabular playground, 1 image code-comp, 1 time-series. Also micro-benchmarks: Validator trap suite, Adversary leakage suite, envelope schema compliance rate, ingestion latency.

## 29. Exact Verified Commands

```bash
agy models                                    # confirm gemini-3.8-flash-high slug
agy -p "..." --model gemini-3.8-flash-high --effort high \
    --output-format json --json-schema schemas/envelope.json
agy -p "..." --output-format stream-json    # NDJSON progress/events
kaggle competitions download -c <slug> -p data/
kaggle kernels push -p kernel/ --accelerator NvidiaTeslaT4 --timeout 39600
kaggle kernels status <user>/<kernel>
kaggle kernels output <user>/<kernel> -p artifacts/ --file-pattern ".*\.(csv|json)$"
kaggle competitions submit -c <slug> -f submission.csv -m "exp-042 blend v3"
git worktree add ../exp-042 -b experiments/exp-042
```

## 30. Example End-to-End Execution

```bash
$ agy --model gemini-3.8-flash-high
> COMPETITION: birdclef-2027  PROJECT_DIRECTORY: ~/comps/birdclef-2027
```
Commander: G0 env probe → capabilities refresh → data download → G1 Scout produces `competition_intelligence.md` (audio classification, code comp, internet off, external data banned, runtime cap) → G2 Forensic (multimodal: audio+spectrogram stats, train/test drift, adversarial AUC 0.52 ✓) + Validator (GroupKFold by species-recording site; VETO cleared) → G3 Runner: GBDT-on-spectral-features baseline + pretrained audio transformer baseline (T4, capped) → artifacts ingested; LB probe #1 (sanity). G4 loop: Engineer proposes augmentation + TTA + pseudo-labeling (soft, k-set); scheduler ranks TTA-highest EV; 3 parallel worktree experiments (Agent Manager fan-out) → Blender: OOF diversity check, greedy ensemble +0.8% CV → G6 Adversary: leakage hunt (fold overlap ✓, rules compliance ✓, license audit ✓) → Executor: kernel push → output → submit → ingest → Chronicler: distills 3 patterns, updates priors (audio comps: pretrained-CNN + TTA + pseudo-label ranked high-EV). Iterate until stopping policy triggers (3 cycles without CV gain + submission budget 80% used) → final audit → final submission → close & distill. Human touched the loop twice: rule acceptance, one credential check.

## Known Limitations

1. No architecture can guarantee top-1; the system maximizes *probability* of top-1%/5%/medal under budget.
2. Gemini 3.8 Flash long-horizon reasoning < frontier Pro-class models; compensated by decomposition and verification, at higher token cost.
3. Custom-subagent model pinning unconfirmed in Antigravity as of research date; if unavailable, parallelism is same-model only (design already assumes this).
4. Kaggle quotas/accelerator availability drift; `capabilities/` must be refreshed per competition (numbers here are Sept-2026 snapshots).
5. Public-LB signal is adversarially noisy; overfitting it is a policy risk, not fully solvable technically.
6. Cross-competition meta-learning needs ≥3–5 completed competitions before priors are strong.
7. Fully hands-off operation assumes stable Antigravity API; harness changes may require Commander prompt/skill updates (versioned, A/B'd).
8. TFM licensing (non-commercial weights) can disqualify otherwise-best models; compliance layer mitigates but cannot eliminate rule-change risk.
