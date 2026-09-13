**Research Report & Architecture Blueprint: Autonomous Kaggle Competition Operating System (Antigravity CLI + Gemini 3.8 Flash High)**

_Evidence cutoff: September 2026. All claims grounded in primary sources (Antigravity docs, Gemini model cards, Kaggle CLI/docs, recent writeups). Speculation is explicitly labeled._

### 1. Executive Summary

The strongest realistically achievable system under the sole constraint **Antigravity CLI + `gemini-3.8-flash-high`** is a **lean hierarchical Planner-Executor-Verifier architecture with asynchronous subagents, structured artifact memory, EV-prioritized experiment scheduling, and controlled self-evolution**.

It is **not** a naive 50-agent swarm. Marginal intelligence gain per additional agent drops sharply once core roles (Commander, Validation Architect, Experiment Manager, Adversarial Reviewer, Artifact/Knowledge Curator) are covered; extra agents mainly increase coordination tax, context pollution, and failure surface under a single mid-tier reasoning model.

Optimization target: maximize expected private-LB performance per unit of human effort and Kaggle compute, subject to rules, quotas, reproducibility, and model limitations. Top-1 is an aspirational optimization target, never a guarantee.

Core loop (evidence-backed from Kaggle Grandmaster practice + Antigravity capabilities):

```
OBSERVE (data + rules + history) → HYPOTHESIZE → PLAN (EV ranking)
→ PARALLELIZE (isolated subagents/worktrees) → IMPLEMENT → EXECUTE (Kaggle CLI)
→ MEASURE (artifacts) → CRITIQUE (adversarial) → SELECT → MEMORIZE (structured)
→ GENERALIZE → EVOLVE (skills/agents under A/B) → REPEAT
```

### 2. Current Antigravity Capabilities (Primary Sources)

- **CLI**: `agy` (TUI + headless `-p`). Shared agent harness with Antigravity 2.0. Workspace-scoped conversations.
- **Models**: Explicit selection via `--model gemini-3.8-flash-high` (or `/model`). Effort levels (high/medium/low). List via `agy models`.
- **Skills**: Markdown blueprints (`SKILL.md`) under `.agents/skills/` (workspace) or global. Auto-discovered and injected on match.
- **Plugins**: Bundles of skills + agents + hooks + MCP (`plugin.json` + directories). Install via `agy plugin install <url|path>`. Staged under `~/.gemini/.../plugins/`.
- **Subagents**: Asynchronous, parallel, full tool access (code, shell, web, MCP). Spawned by main agent; managed via `/agents`. Background tasks do not block.
- **Hooks**: Pre/post tool, model call, session events. Local scripts.
- **MCP**: Global (`~/.gemini/config/mcp_config.json`) + workspace (`.agents/mcp_config.json`). Interactive `/mcp`.
- **Headless**: `agy -p "prompt" --model gemini-3.8-flash-high --effort high`. JSON-structured output possible; suitable for automation loops.
- **Permissions/Sandbox**: Fine-grained allow/ask/deny; terminal sandbox.
- **Other**: `/fork`, conversation export, shared settings, AGENTS.md / .agents/ rules.

**Implication**: Native support for hierarchical + parallel specialist execution without external frameworks. Prefer native subagents + skills over custom multi-agent runtimes.

### 3. Gemini 3.8 Flash High Capabilities Relevant to Architecture

- Strong long-horizon software engineering (DeepSWE v1.1 ~73.7%, competitive with larger models at fraction of cost).
- Designed for autonomous agents, multi-step planning, tool orchestration, reduced failed loops.
- 1M context, 64k output, thinking levels (use `high` for strategic decisions), structured outputs, function calling, code execution, search, computer-use (preview).
- Default for Antigravity agents. Knowledge cutoff ~March 2026 (refresh via tools).
- Limitations to compensate for: residual hallucination risk on subtle leakage/CV design → mandatory verification + adversarial cell + empirical feedback; weaker pure creative insight than frontier models → heavy reliance on structured memory + retrieval of past evidence + parallel hypothesis generation.

Compensation strategy (evidence-based): decomposition into specialized roles, structured YAML/JSON contracts, retrieval-augmented memory (not full transcript dump), parallel independent experiments, adversarial critique with veto, execution feedback loops, A/B on prompts/skills.

### 4. Kaggle Infrastructure Constraints (Must Be Queried Live)

- **CLI**: `kaggle kernels init/push/pull/output/status/logs`. `push` uploads + runs; `--accelerator` / `machine_shape` (NvidiaTeslaT4, L4, A100, H100, TPU variants as of Sep 2026; availability competition-dependent).
- **Typical free-tier limits** (approximate; refresh every competition): ~30 GPU hours/week, 12 h session (GPU/CPU), ~9 h TPU, ~20 GB working disk + temp, RAM ~13–32 GB depending on accelerator. CPU unlimited but slower.
- **Rules**: External data allowed if publicly equally accessible at no/minimal cost. Internet often disabled in competition notebooks. Submission frequency limits. Private LB is official ranking. Team rules, no private sharing outside teams.
- **Artifacts**: Output downloadable via CLI; design stable schema so ingestion is automatic.

**System policy**: Never hard-code quotas. Maintain `capabilities/kaggle_current.yaml` refreshed by a lightweight agent at competition start.

### 5. Kaggle-Winning Strategy Analysis (Recent Evidence)

Top solutions (2025–2026 writeups) consistently show:

- Robust validation design before modeling (temporal/group/adversarial validation; CV must predict private LB).
- Domain-driven + combinatorial feature engineering > pure HPO.
- Diverse ensembles (hill-climbing on OOF, stacking, residual modeling) with correlation/error analysis; correlated models hurt.
- Leakage hunting and distribution-shift detection (adversarial validation, train/test diffs).
- Fast credible baseline → EV-prioritized experiments.
- Post-processing for shift; pseudo-labeling/TTA only with evidence.
- For tabular: LightGBM/XGBoost/CatBoost + modern NNs (TabM etc.); heavy interaction/encoding features.
- Non-obvious edge often comes from structure exploitation, external public data (when allowed), or clever post-processing rather than “better model family”.

Generic AutoML rarely wins alone; elite performance requires the above + compute discipline.

### 6–7. Multi-Agent Architecture Comparison & Recommendation

Compared (evidence from multi-agent literature + Antigravity native primitives + Kaggle practice):

| Architecture              | Strengths             | Weaknesses under Flash High + Kaggle | Verdict             |
| ------------------------- | --------------------- | ------------------------------------ | ------------------- |
| Pure Supervisor           | Simple                | Bottleneck, limited parallel         | Insufficient        |
| Hierarchical Supervisor   | Good control          | Still sequential bias                | Useful base         |
| Manager-Worker            | Parallel              | Coordination tax                     | Partial             |
| Blackboard                | Shared state          | Complexity, consistency              | Overkill            |
| Parallel Specialist Swarm | Throughput            | Chaos, duplication, no synthesis     | Reject pure         |
| Debate                    | Critique              | Token cost, slow                     | Use for final audit |
| Critic-Generator          | Quality               | Limited exploration                  | Use inside cells    |
| Planner-Executor-Verifier | Evidence alignment    | Needs good planner                   | **Core**            |
| Evolutionary Population   | Long-term improvement | Expensive, needs careful selection   | Controlled only     |
| Hybrid                    | Best of above         | Design complexity                    | **Selected**        |

**Recommended: Hybrid Hierarchical Planner-Executor-Verifier with Async Subagents + Adversarial Cell + Memory Curator**

```
KAGGLE COMMANDER (Strategic Director / Planner)
├── RESEARCH CELL (Competition Intelligence)
├── DATA CELL (Forensics + EDA + Validation Architect — veto power)
├── MODEL CELL (Model Research + HPO + Feature Hypotheses)
├── EXPERIMENT CELL (Manager + parallel isolated Trainers)
├── ENSEMBLE / ANALYST CELL
├── ADVERSARIAL CELL (Leakage Hunter + CV Audit + Error Analyst — veto)
├── SUBMISSION / KAGGLE EXECUTOR CELL
├── ARTIFACT INGESTION + KNOWLEDGE EVOLUTION
└── MEMORY CURATOR + STRATEGY EVOLUTION (meta)
```

Parallelism only for independent experiments (git worktrees or isolated dirs). Shared mutable state forbidden. Commander owns prioritization, stopping, and final synthesis. Validation Architect and Adversarial Reviewer have hard veto on invalid CV or leakage risk.

This maps cleanly onto Antigravity native subagents + skills.

### 8. Agent Roster (Minimum Effective Set)

1. **Commander** — global objective, EV scheduling, resource allocation, stopping rules, synthesis, final submission decision.
2. **Competition Researcher** — rules, metric, data restrictions, timelines, baselines, similar past competitions.
3. **Data Forensics + EDA** — schema, quality, leakage signals, distributions, train/test shift, target analysis, modality detection.
4. **Validation Architect** (veto) — CV strategy, fold design, leakage tests, justification “why this correlates with private LB”.
5. **Feature / Model Researcher** — hypothesis generation (not blind generation), model family selection by evidence + constraints.
6. **Experiment Manager** — registry, prioritization, isolation, logging.
7. **Trainer / HPO** (parallel instances) — reproducible runs producing standard artifacts.
8. **Ensemble Agent** — correlation, diversity, hill-climb, stacking evaluation on OOF.
9. **Adversarial Reviewer / Leakage Hunter / Error Analyst** (veto) — attack current solution.
10. **Kaggle Executor** — notebook/script prep, `kaggle kernels push/output`, status polling.
11. **Artifact Analyst + Knowledge Curator** — parse metrics, update registry/memory, promote knowledge.
12. **Strategy Evolution + Memory Curator** — meta patterns with evidence counts/confidence.
13. **Compliance / Final Auditor** — rules check before any submission.
14. **Performance Engineer** (on-demand) — I/O, Polars, memory, caching.

Fewer than 15 specialized roles; many implemented as skills or lightweight subagent prompts rather than persistent processes. Track per-agent metrics (success, wasted compute, false positives, downstream LB impact) for controlled evolution.

### 9–11. Communication, Memory, Self-Evolution

**Protocol** (strict, machine-readable):

```yaml
status: success|partial|fail
confidence: 0-1
findings: [...]
evidence: [paths, metrics, citations]
actions_taken: [...]
artifacts: [list]
recommendations: [...]
risks: [...]
next_tasks: [...]
veto: false|true (reason)
```

**Memory layers**:

- Working: current task state (ephemeral).
- Project: competition-specific (reports/, experiments/, knowledge/competitions/<slug>/).
- Strategic: cross-competition patterns (with evidence_count, confidence, counterexamples, last_validated).
- Meta: agent/skill performance, system version history.

Use structured YAML/JSON + light retrieval (keyword + semantic if MCP vector available). Compress aggressively; never dump full transcripts. Provenance and contradiction detection mandatory. Stale knowledge detection by date + re-validation trigger.

**Self-evolution**:

- Experiment → structured knowledge record (hypothesis, intervention, result, confidence, reproducibility).
- Skill evolution pipeline: failure classification → root cause → generalizable? → draft skill → evaluation cases → adversarial review → versioned → A/B → promote/reject. Never auto-install unvalidated skills.
- Agent evolution: performance metrics → candidate prompt/role change → benchmark against baseline → promote only if superior.
- Hypothesis → Experiment → Evidence → Knowledge → Strategy → New Hypothesis loop is the core.

### 12–30. Remaining Architecture Components (Summary)

- **Experiment Scheduler**: priority ≈ (expected_gain × confidence × information_gain) / (compute_cost × risk). Explicit stopping rules (plateau, low EV remaining, deadline, unstable validation, diversity exhausted).
- **Validation**: first-class artifact. Must answer correlation-with-private-LB question before modeling proceeds.
- **Kaggle Automation**: local project → git → kernel-metadata.json + code → `kaggle kernels push` → poll status → `output` → local artifacts → analysis. Isolated worktrees for parallel experiments.
- **Artifacts (stable schema)**: metrics.json, oof_predictions.csv, submission.csv, experiment.json, resource_usage.json, logs/, reports/.
- **Compute Optimization**: Polars/Arrow/Parquet preferred; decide GPU only if workload benefits; cache aggressively; vectorize; avoid repeated CSV; uv for envs.
- **Failure Recovery**: classify (DATA/DEPENDENCY/CODE/MEMORY/TIMEOUT/KAGGLE/VALIDATION/MODEL/AGENT/ORCHESTRATION/ARTIFACT) → targeted recovery; never blind retry.
- **Compliance**: dedicated check for external data, internet, submission limits, licensing, reproducibility.
- **Context Engineering**: layered (global rules + role + competition brief + retrieved memory + task + recent artifacts). Use catalog skills for compression/optimization/guardian.
- **Reusability**: `kaggle-agent-core/` (agents, skills, schemas, orchestration, policies, templates) separate from `competition-project/` (data, src, experiments, artifacts, knowledge, competition.yaml).

### 22. Skill Selection from Catalog (2,121 skills)

**Install/use (global or core)**:

- multi-agent-task-orchestrator, parallel-agents, dispatching-parallel-agents, multi-agent-patterns, agent-orchestrator, agent-orchestration-multi-agent-optimize, agent-orchestration-improve-agent
- agent-memory, agent-memory-mcp, context-engineering, context-compression, context-optimization, context-guardian, context-agent
- agent-evaluation, evaluation
- data-scientist, ml-engineer, machine-learning-ops-ml-pipeline, polars, uv-package-manager
- using-git-worktrees
- Relevant orchestration and verification skills (orchestrate, verification-before-completion, etc.)

**Project-local**: competition-specific rules, domain playbooks, dataset signatures.

**Reject or inspiration-only**: heavy external frameworks (full LangGraph unless proven necessary), unrelated domains (voice, web UI, security offensive tools), pure delegation to other CLIs (codex-delegate etc. — stay inside Antigravity), unvalidated evolutionary swarms.

**External discoveries**: Kaggle’s own skills repo (if present), official Kaggle CLI patterns, Polars-centric pipelines, modern tabular libs (TabM etc. when evidence supports). Prefer native Antigravity skills over new MCP servers unless clear ROI.

### 41–51. Implementation Blueprint (Exact Deliverables)

**Directory (core)**:

```
kaggle-agent-core/
├── AGENTS.md                 # Commander + role prompts + contracts
├── agents/                   # Subagent definitions (.md or plugin)
├── skills/                   # Global reusable skills
├── hooks/                    # Pre/post tool, artifact validation
├── memory/                   # Schemas + retrieval helpers
├── schemas/                  # experiment.yaml, artifact.json, knowledge.yaml
├── orchestration/            # Scheduler, routing, stopping
├── evaluation/               # Agent/system benchmarks
├── scripts/                  # kaggle_push_loop.py, artifact_ingest.py, refresh_capabilities.py
├── templates/                # competition.yaml, kernel-metadata, baseline notebooks
├── policies/                 # EV formula, veto rules, compliance
└── config/
```

**Competition project**:

```
competition-<slug>/
├── competition.yaml
├── data/  reports/  validation/  experiments/  artifacts/  knowledge/  src/  notebooks/
└── .agents/skills/  (project overrides)
```

**Initialization workflow** (`agy --model gemini-3.8-flash-high`):

1. Detect env (OS, Python, uv/pip, git, kaggle CLI, credentials, hardware).
2. `kaggle competitions accept` / download if needed (human only for credentials/accept).
3. Create structure + competition.yaml.
4. Spawn Research + Data Forensics cells in parallel.
5. Produce competition_intelligence.md + data reports.
6. Validation Architect designs + justifies CV.
7. Baseline → experiment loop under Commander.

**Default Playbook** (dynamic skip):
INIT → COMPETITION RESEARCH → DATA FORENSICS → EDA → VALIDATION DESIGN → BASELINE → FAST EXPERIMENTS (EV) → FEATURE/MODEL SEARCH → HPO → ERROR ANALYSIS → ENSEMBLING → ADVERSARIAL REVIEW → KAGGLE RUN → ARTIFACT INGESTION → KNOWLEDGE UPDATE → STRATEGY EVOLUTION → FINAL AUDIT → SUBMISSION.

**Example end-to-end (headless-capable)**:

```bash
cd competition-project
agy -p "Initialize and run full autonomous pipeline for this competition. Use only gemini-3.8-flash-high. Maximize expected private LB. Follow core policies." --model gemini-3.8-flash-high --effort high
```

**Known Limitations**:

- Single-model reasoning ceiling (compensated by structure/verification).
- Kaggle quota volatility and competition-specific rules require live refresh.
- Skill evolution still needs human gate for high-risk changes.
- Pure creative domain insight remains partly human-dependent for true top-1 edge cases.
- No guarantee of #1; system maximizes probability of top tiers under constraints.

**Implementation Phases**:

1. Core schemas + Commander + Validation + Artifact loop + Kaggle CLI automation.
2. Parallel experiment isolation + EV scheduler.
3. Memory + knowledge promotion.
4. Adversarial + compliance.
5. Skill/agent evolution under A/B.
6. Cross-competition meta-learning + benchmarks.

**Benchmark Plan**: Internal agent evals (success rate, wasted compute, leakage detection accuracy) + replay of past competitions with known private scores + ablation on architecture variants.

This design is Kaggle-native, Antigravity-native, evidence-driven, compute-efficient, and self-improving within the hard model constraint. It prioritizes private-LB robustness, leakage resistance, and minimal human intervention while remaining realistic for a single engineer to maintain and evolve.
