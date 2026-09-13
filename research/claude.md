# Kaggle Competition OS
### A self-evolving, competition-agnostic multi-agent system for Antigravity CLI + Gemini 3.8 Flash High

*Prepared September 13, 2026. Grounded in live web research conducted the same day;
volatile facts (CLI paths/flags, Kaggle quotas) are flagged for re-verification —
see §17 Known Limitations and `capabilities/probe.py` in the scaffold.*

---

## 0. Executive summary

- **Antigravity CLI is real and matches most of the brief.** Google sunset Gemini
  CLI in favor of Antigravity CLI in mid-2026, carrying forward Agent Skills,
  Hooks, Subagents, and Extensions-as-Plugins. Dynamic/on-the-fly subagents,
  async background tasks, and scheduled (cron-style) agents are documented
  platform features, not aspirational.
- **Gemini 3.8 Flash High is real**, released ~September 2, 2026, as the
  "high" effort-level variant of Gemini 3.8 Flash — Google's current agentic
  workhorse model, positioned between Flash-Lite and Pro. It is good at
  long-horizon tool-calling and software-engineering-style tasks for its cost
  tier, but it is explicitly *not* the frontier-reasoning tier (that's Gemini
  3 Pro). The architecture below is designed around that reality: compensate
  through decomposition and verification, not through assuming frontier
  reasoning depth per call.
- **Headless automation is the system's biggest real operational risk, not a
  hypothetical one.** Multiple independent, dated reports (including issues
  filed against the official `google-antigravity/antigravity-cli` repo)
  describe `agy -p`/`--print` hanging or silently dropping output in
  non-TTY/subprocess contexts, and `permissions.allow` not being honored in
  headless mode at all as of some releases. A "minimal human intervention"
  system built on top of a shaky headless contract will fail silently unless
  it is designed around that fact from day one. This drove the single biggest
  architectural decision below: **agents communicate through files on disk
  (a blackboard), never through captured stdout.**
- **The 2,121-skill catalog you supplied contains zero Kaggle-specific skills
  and essentially zero tabular-ML-competition skills** (no XGBoost/LightGBM/
  CatBoost/AutoGluon/adversarial-validation skill exists in it at all — grep-
  verified). It is overwhelmingly general-purpose coding-agent, web/mobile
  dev, security, and infra tooling. It is genuinely useful for the
  *orchestration/agent-infrastructure* layer (memory, context management,
  evaluation, multi-agent patterns) and for domain-specific tooling *if* a
  competition is CV/NLP (Hugging Face skills), but the Kaggle- and tabular-
  modeling-specific core of this system has to be hand-built. That is
  reflected honestly in §15-16 below rather than papered over.
- **A meaningful fraction of the catalog is actively disqualified by your own
  hard constraint.** Any skill whose job is "delegate this task to Codex /
  Claude / Cursor / Aider / a model gateway" is a silent-substitution risk
  and is rejected categorically, not case-by-case (§16).
- The recommended architecture is a **hierarchical Commander + bounded
  parallel Cells + filesystem blackboard + a narrow, high-authority
  Adversarial verification gate** — not a debate swarm, not a large
  evolutionary population. Rationale in §3.
- This document does not, and cannot honestly, guarantee a #1 finish. It
  defines what the system optimizes for, how it measures itself, and where
  its ceiling is (§17).

## Evidence tiers used throughout this document

| Tier | Meaning | Used for |
|---|---|---|
| **T1** | Official docs / official product blog / official model card | Antigravity CLI existence & feature set, Gemini 3.8 Flash High existence |
| **T2** | Official competition/platform behavior as documented by Kaggle | GPU quota order-of-magnitude, session limits, code-competition mechanics |
| **T3** | Peer-reviewed / widely-replicated ML research | CV design principles, adversarial validation, stacking theory |
| **T4** | Credible third-party technical writeups, GitHub issues on the official repo | Antigravity headless-mode bugs, exact directory-path candidates |
| **T5** | Experienced-practitioner consensus (Kaggle grandmaster writeups, general ML engineering practice) | Feature engineering heuristics, ensembling-when-diverse heuristic |
| **T6** | Community discussion / single blog post / SEO content | Flagged explicitly wherever used; not used for any load-bearing claim |

Every non-obvious claim below is tagged with its tier in parentheses, e.g. (T1), (T4).
Untagged statements are standard, slow-changing ML/software-engineering practice (T3/T5)
that this document treats as settled rather than re-deriving from first principles.

---

## Table of contents

1. Research findings (Antigravity, Gemini 3.8 Flash High, Kaggle constraints)
2. Skill catalog audit
3. Architecture decision (and why it deviates from the brief's diagram)
4. Agent roster & responsibilities
5. Agent communication protocol
6. Memory architecture
7. Skill architecture (global vs. per-project vs. embedded vs. rejected)
8. Self-evolution mechanism (skills + agents)
9. Experiment scheduler
10. Validation system
11. Kaggle automation flow
12. GitHub integration
13. Artifact protocol
14. Compute optimization system
15. Failure recovery system
16. Evaluation framework
17. Security / compliance layer
18. Directory & configuration structure
19. Workflows (init, competition run, cross-competition learning, skill/agent evolution)
20. Recommended catalog skills / externally discovered tools
21. Implementation phases
22. Benchmark methodology
23. Exact commands (verified vs. best-known)
24. Worked example: one full competition pass
25. Known limitations

---

## 1. Research findings

### 1.1 Antigravity CLI (T1/T4)

Google announced Antigravity (an "agent-first" development platform, distinct
from a chat-in-sidebar tool) in November 2025 (T1). In May–June 2026, Google
sunset the original Gemini CLI and replaced it with **Antigravity CLI**,
binary name `agy`, explicitly carrying forward "Agent Skills, Hooks,
Subagents, and Extensions" from Gemini CLI, the last now rebranded as
**plugins** (T1). Confirmed capabilities relevant to this project (T1/T4):

- **Skills**: Markdown files with YAML frontmatter, discovered at the
  workspace level from `.agents/skills/<name>/SKILL.md` — this is the one
  location every source agrees on. Global skill locations are *not*
  consistently reported across sources (candidates seen: `~/.gemini/config/skills/`,
  `~/.gemini/antigravity-cli/skills/`, `~/.gemini/antigravity/builtin/skills/`)
  (T4) — treat this as unresolved and detect it empirically (see
  `capabilities/probe.py` in the scaffold) rather than hardcoding one.
- **Subagents**: the main agent can define and spawn specialized subagents
  dynamically at runtime to work on focused subtasks without polluting its
  own context window (T1/T4) — this is the platform primitive the
  Cell/Commander design below is built on.
- **Hooks**: JSON-defined lifecycle interceptors (`SessionStart`, before/after
  tool calls, after file edits) (T1/T4). Whether a formal "workflow" primitive
  exists as a first-class scheduled/triggered object, vs. `.agents/workflows/`
  simply being a documentation convention some integrators use, is
  inconsistently described across sources (T4/T6) — this document treats
  workflows as **agent-followed documentation reinforced by hooks**, not as a
  guaranteed automatic trigger, until you've confirmed otherwise against your
  installed version.
- **Async/background execution & scheduling**: long-running tasks run in the
  background without blocking the UI; cron-style scheduled agent runs are a
  documented Antigravity 2.0 feature (T1/T4).
- **MCP support**: first-class (T1).
- **Headless / non-interactive mode — the important caveat**: `agy -p "<prompt>"`
  (aliases `--print`/`--prompt`) runs once and exits, intended for scripting
  (T4). However, multiple dated reports against the official
  `google-antigravity/antigravity-cli` GitHub repo describe real, specific
  failure modes as of mid-2026 (T4):
  - `-p`/`--print` hanging indefinitely or emitting no stdout at all when
    invoked from a non-TTY context (piped, redirected, spawned as a
    subprocess) — exactly the invocation pattern any orchestrator wrapping
    `agy` needs.
  - `permissions.allow` in `settings.json` not being consulted at all in
    headless mode in at least one reported version, making
    `--dangerously-skip-permissions` the only way to get unattended execution
    to proceed — which then has its own separate reported silent-failure
    issues.
  - The "persist to settings.json" interactive approval option not actually
    persisting anything.
  - There is, as of the sources found, **no `--headless` flag and no
    output-guarantee flag** (`--no-tty`, `--json` for print mode) — unlike
    some peer CLIs (T4).

  **Design consequence**: this system cannot assume "call `agy -p` and read
  its stdout" is reliable. It must (a) run the invocation with a bounded
  timeout, (b) always have the invoked agent write its result to a file
  (the blackboard `run_sentinel`), and (c) have the orchestrating layer poll
  that file rather than trust captured output. See §5, §15.

### 1.2 Gemini 3.8 Flash High (T1)

Gemini 3.8 Flash is Google's current Flash-tier "workhorse" model, released
~September 2, 2026, positioned as materially better than 3.7 Flash at
software engineering, agentic tool use, and multi-step reasoning, while
remaining below Pro-tier frontier reasoning in raw capability (T1, Google
DeepMind model card and Google Cloud docs). "High" is a documented **effort
level**, not a separate model file — effort levels trade token spend/latency
for quality, and Google's own docs note the model "works harder" at higher
effort levels (more tool calls, more reasoning steps) (T1). Two consequences
for this design:

1. Treat `gemini-3.8-flash-high` as **capability-constrained relative to
   frontier reasoning**, not merely "a cheaper Pro." Compensate via
   decomposition + verification (per your own §3 constraint), not by writing
   longer prompts.
2. Because higher effort levels consume more tokens/turns per task (T1), the
   architecture should default *lower*-context, *narrower*-scope agents
   (small Cells with a tight remit) over a small number of do-everything
   agents — this cuts both cost and the chance of the model losing the thread
   inside an overloaded context window.

### 1.3 Kaggle compute & rules constraints (T2, with T4/T6 caveats on exact numbers)

- Kaggle Notebooks provide free GPU access with a **weekly quota historically
  around 30 hours**, occasionally raised via a "floating quota" experiment;
  the exact number has changed more than once historically and should be
  re-checked in-app rather than trusted from any document, including this one
  (T2/T4 — Kaggle's own docs describe the quota as demand-dependent).
- Individual **notebook sessions have a hard wall-clock limit** (commonly
  cited around 9–12 hours) (T2/T4) — long training runs must checkpoint.
- **"Code Competitions"** (the now-standard format for most active
  competitions) require the *submission* to be a notebook that re-runs
  end-to-end against a hidden test set, typically **with internet access
  disabled during that scored run** — this is a per-competition rule that
  must be re-confirmed for every competition, not assumed globally (T2). CSV-
  upload-only competitions still exist but are a minority.
- External data permissibility, team-size limits, and merger rules are
  **set per-competition** in the rules page and must be extracted by the
  Competition Researcher agent before any external data is used (T2).

### 1.4 ML technique currency (T3/T5 — general knowledge, not re-derived from fresh search)

Gradient-boosted trees (XGBoost/LightGBM/CatBoost) remain the strong default
for medium-sized tabular data; adversarial validation, group/temporal-aware
cross-validation, out-of-fold stacking, and correlation-aware ensembling
remain the standard toolkit for competitive tabular ML (T3/T5). For CV/NLP/
multimodal competitions, fine-tuning pretrained backbones (vision
transformers, modern LLM/embedding backbones) plus task-specific heads and
test-time augmentation remains standard practice (T5). None of this is
claimed to be a September-2026 discovery — it is treated as settled
engineering practice the system should encode as local skills, verified
against current library APIs at build time rather than assumed stable
forever (library APIs move faster than the underlying technique).

---

## 2. Skill catalog audit (2,121 skills, grep-verified against the supplied file)

**Headline finding**: searching the catalog for `kaggle`, `xgboost`,
`lightgbm`, `catboost`, `automl`, `autogluon` returns **zero matches**.
`optuna`/hyperparameter-optimization is represented by exactly one generic,
non-Kaggle skill (`optim-agent`). One tabular-adjacent skill exists
(`polars`) and one general ML skill (`scikit-learn`). This is a catalog built
for general coding-agent orchestration and software/web engineering, not for
Kaggle-style competitive ML — a fact the architecture must not paper over
(see §20 for the full disposition table, and §17 Known Limitations).

**Category-level read of the 2,121 skills** (by count, largest first):
`development` (187), `security` (86), `business` (67), `content` (67),
`automation` (55), `marketing` (57), `cloud` (146), `uncategorized` (332),
`ai-ml` (129) — the ai-ml bucket is the only one with meaningful density of
agent-orchestration/memory/context-engineering skills relevant here, and even
it is dominated by generic LLM-app-building, RAG, and persona-roleplay
skills (e.g. simulated "Yann LeCun" / "Sam Altman" / "Ilya Sutskever" persona
skills) that are irrelevant to this project and are rejected outright.

A specific, important compliance finding: the catalog contains an entire
`agent-orchestration` sub-family of **`*-delegate` skills** (`aider-delegate`,
`claude-delegate`, `codex-delegate`, `cursor-delegate`, `grok-delegate`,
`kimi-delegate`, `agy-delegate`, etc.) whose explicit purpose is routing
coding tasks to a *different* CLI/model. Every one of these is disqualified
categorically by your own hard constraint (§3 of your brief: no silent
substitution of Claude/GPT/other Gemini variants/OpenRouter/etc.) — not
because the skills are poorly made, but because installing any of them
creates exactly the failure mode the constraint exists to prevent. Same
verdict for third-party model-gateway skills (`routerbase-model-gateway`,
`sandbase-mcp`, `unified-ai-gateway`). This is treated as a **standing
governance rule** for all future skill additions, not a one-time catalog
decision (§7, §16).

---

## 3. Architecture decision

### Comparison of the ten candidate architectures

| Architecture | Fit for Flash-tier model | Fit for Kaggle compute limits | Fit for headless-mode risk | Verdict |
|---|---|---|---|---|
| Pure supervisor (1 agent does everything, others are tools) | Poor — overloads context on a mid-tier model | N/A | N/A | Rejected |
| Hierarchical supervisor | Good — bounded scope per level | Good | Neutral | **Core of the pick** |
| Manager-worker | Good, similar to hierarchical | Good | Neutral | Folded into hierarchical |
| Blackboard / shared-memory | Excellent — decouples agents from fragile session-to-session handoff | Neutral | **Directly mitigates the headless stdout-capture bug (T4)** | **Adopted as the comms layer** |
| Parallel specialist swarm (unbounded) | Risky — context/coordination overhead scales with agent count on a workhorse model | Bad — Kaggle GPU is a shared, quota-limited resource; unbounded parallel GPU jobs just queue or blow the weekly budget | Bad — more headless invocations = more exposure to the stdout bug | Rejected as a default; used only for CPU-bound, non-GPU stages (research, EDA, hypothesis generation) |
| Debate architecture (same model debating itself) | Weak evidence of benefit — debate gains come mainly from *diverse* reasoning styles/models; N identical Flash-High instances debating add cost with limited diversity payoff (T5, general finding, not Kaggle-specific) | Costly | Costly | Rejected as a general mechanism |
| Critic-generator | Good, narrow | Good | Good | **Adopted, scoped to the Adversarial Cell** |
| Planner-executor-verifier | Good | Good | Good | **Adopted as the per-stage pattern inside each Cell** |
| Evolutionary agent population | Interesting for skill *versions* over long timescales, wrong for within-competition experimentation (too much redundant compute for a 30h/week GPU budget) | Bad within a single competition | Neutral | Rejected within-competition; **repurposed** for skill evolution only (§8), where "population" = versioned skill candidates evaluated over time, not simultaneous live agents |
| Hybrid | — | — | — | **This is the pick**, composed from the rows above |

### The pick, stated plainly

**Hierarchical Commander → 5 bounded Cells → filesystem blackboard →
narrow Adversarial verification gate with veto power → serialized
GPU-bound execution against a compute ledger.**

### How and why this deviates from the brief's suggested diagram

1. **The comms backbone is explicitly a filesystem blackboard, not implied
   agent-to-agent messaging.** This is not a stylistic choice — it is a
   direct response to the headless-mode output-capture bugs found in §1.1
   (T4). If two agent invocations pass information only through captured
   stdout, and that channel is documented as unreliable in exactly the
   non-interactive context this system needs, the system will fail silently.
   Every load-bearing fact is a file (`blackboard/state.json`,
   `knowledge/experiments/*.yaml`), and every long-running/headless call
   writes a completion sentinel that the orchestrator polls for, rather than
   trusting whatever came back on stdout.
2. **Parallelism is capped by resource type, not by agent-design taste.**
   CPU-bound, non-GPU stages (competition research, EDA, feature
   *hypothesis* generation, leakage scanning) can and should run several
   subagents in parallel — Antigravity's dynamic-subagent + async-background
   features (T1) support this natively and it costs nothing against the
   Kaggle GPU ledger. GPU-bound stages (model training, HPO) are
   **serialized against `config/compute_budget.yaml`** — running five
   training jobs "in parallel" against a shared 30-hour/week quota (T2) does
   not produce five times the information; it produces the same information
   in a more confusing order and a higher chance of quota exhaustion before
   the useful experiment runs.
3. **Debate is removed as a general mechanism and replaced by a narrow,
   high-authority verification gate** (Validation Architect + Leakage
   Hunter + CV/Error Auditor, collectively the "Adversarial Cell").
   Kaggle placements are lost far more often to leakage, CV/LB mismatch, and
   distribution shift than to a lack of creative brainstorming (T5,
   practitioner consensus) — so the one place this design spends extra
   verification budget is exactly there, with real veto power that halts the
   pipeline, rather than diffusing that budget into open-ended multi-agent
   debate about modeling choices in general.
4. **"Ensembler" and "Analyst" from the brief's list are merged into a
   2-agent Experiment Cell**, and "EDA Agent" is merged into "Data Forensics"
   as one agent — see §4 for the full merge rationale. More agents than
   necessary is not free on a Flash-tier model: every hand-off is a place
   where instructions and nuance can be lost, and the brief's own §3
   correctly warns against paying for depth with prompt length instead of
   structure — the same logic argues against paying for rigor with agent
   *count* instead of structure.

---

## 4. Agent roster & responsibilities (13 agents / 5 Cells + Command + Knowledge)

Trimmed from the brief's ~19-role brainstorm. Every merge/cut is justified,
not just asserted:

| Original role(s) | Final agent | Why |
|---|---|---|
| Commander | **Commander** | Unchanged — owns strategy, budget, stop/go, never touches submission directly |
| Competition Researcher | **Competition Researcher** | Unchanged |
| Data Forensics Agent + EDA Agent | **Data/EDA Agent** | Both consume the same inputs (schema, distributions, target) and produce complementary halves of one report — splitting them just adds a hand-off with no independent-verification benefit |
| Validation Architect | **Validation Architect** | Unchanged, **keeps veto power** exactly as specified |
| Feature Engineer | **Feature Engineer** | Unchanged, explicitly hypothesis-driven per the brief |
| Model Researcher + HPO Agent | **Model Researcher/HPO** | Model family selection and tuning that family are one continuous decision in practice; splitting them risks the HPO agent tuning a family the Researcher would have already deprioritized |
| Experiment Manager | **Experiment Manager** | Unchanged — owns the ledger schema enforcement (§9) |
| Ensemble Agent | **Ensemble Agent** | Unchanged |
| Adversarial Reviewer + Leakage Hunter + Error Analyst | **Leakage Hunter + CV/Error Auditor** (2 agents, not 3) | Leakage detection is a distinct, high-precision skill worth its own agent per the brief's emphasis; general adversarial "try to break it" review and error-pattern analysis overlap enough (both are "look at where predictions fail and why") to combine without losing rigor |
| Performance Engineer | **Folded into a shared skill**, not a standalone agent | Its concerns (I/O, vectorization, caching) are cross-cutting checklist items best enforced via a skill invoked by whichever agent is training/processing, not a separate agent that has to be looped in and out of every stage |
| Kaggle Executor | **Kaggle Executor** | Unchanged, but see §11 for the headless-risk-aware execution contract |
| Artifact Analyst | **Artifact Analyst** | Unchanged |
| Memory Curator + Strategy Evolution Agent | **Knowledge Curator** | Both consume experiment evidence and write to `knowledge/`; splitting them invites disagreement about which one "owns" a given piece of knowledge |
| Final Auditor | **Final Auditor** | Unchanged, **keeps veto power** |

Full per-agent responsibilities, inputs, outputs, and stopping/escalation
rules are implemented as actual `.agents/skills/*/SKILL.md` files in the
scaffold for the five highest-leverage agents (Commander, Validation
Architect, Leakage Hunter, plus the cross-cutting Experiment Ledger and
Submission Packager skills). The remaining eight follow the identical
template — see `kaggle-os/README.md` in the scaffold for the checklist to
replicate it.

---

## 5. Agent communication protocol

**Primary channel: the filesystem, not chat history.** Concretely:

1. `blackboard/state.json` — single current-state document (schema in the
   scaffold). Whoever acts writes it before finishing their turn. Whoever
   starts a turn reads it before doing anything else, including before
   trusting anything from its own (possibly truncated/lost) prior context.
2. `knowledge/experiments/<competition>/<exp_id>.yaml` — the experiment
   ledger. This is the *only* legitimate way a result becomes visible to
   another agent; no agent may act on a training result it only "remembers"
   from conversation, because a fresh subagent invocation may not share that
   conversation at all.
3. `blocking_issues[]` inside the blackboard is a hard interrupt: any agent
   with veto power (Validation Architect, Leakage Hunter at `confirmed`
   severity, Final Auditor) writes here, and **every other agent must check
   this list and refuse to proceed past a `veto` severity item.**
4. **Headless/subagent invocations always write a `run_sentinel`** (task_id,
   status, artifact_paths) to a well-known path before the invocation is
   considered complete, specifically because stdout capture from `agy -p` in
   non-TTY/subprocess contexts is documented as unreliable (T4, §1.1). The
   orchestrating agent polls for the sentinel file with a timeout, and treats
   "sentinel never appeared" as a failure to retry/escalate (§15), not as
   silent success.
5. **Human-in-the-loop points are also files**: `blackboard/final_audit_pass.json`
   (required before submission) and skill-promotion approval records (§8) are
   deliberately outside the agents' own write authority — a human (or a
   separate, higher-trust review pass) has to produce them.

---

## 6. Memory architecture

Directory layout matches the brief's §7 almost exactly, because it was
already well-designed — the change is making each folder's *contract*
explicit (full schemas are in the scaffold under `knowledge/`):

```
knowledge/
├── competitions/<slug>/brief.md, timeline.md
├── strategies/                # promoted, cross-competition, evidence-cited
├── validation/<slug>/audit_log.md, noise_estimate.md
├── feature_engineering/       # hypothesis-tagged, links to exp_ids
├── models/                    # per-problem_type digest
├── ensembles/                 # what combined well / looked-diverse-but-didn't
├── leakage/<slug>/findings.md + cross-competition patterns.md
├── kaggle/<slug>/rules_extract.md, submission_format.md, external_data.md
├── compute/                   # spend post-mortems vs. budget
├── failures/                  # same schema as successes — not a graveyard
├── successes/                 # promoted experiments (confidence >= strong_pattern)
├── experiments/<slug>/<exp_id>.yaml   # the atomic unit of truth (schema.yaml)
├── skills/                    # provenance/version log for LOCAL skills
├── agent-performance/<agent>/<slug>.yaml + rolling_summary.yaml
├── meta-learning/<hypothesis-slug>.yaml   # evidence_for / evidence_against
└── playbooks/                 # fully worked recipes, the end product
```

**Promotion discipline** (this is the load-bearing rule that keeps the system
honest, per the brief's explicit warning against auto-promoting speculation):
an experiment starts at `confidence: weak_hypothesis`. It only becomes
`strong_pattern` after replication across ≥2 seeds or ≥2 structurally
different competitions, **and that promotion is performed exclusively by the
Knowledge Curator**, never by the agent that ran the original experiment.
Every `meta-learning/` hypothesis file is required to carry an
`evidence_against` list — a hypothesis with zero recorded counterexamples is
treated as under-tested, not confirmed.

## 7. Skill architecture: global vs. per-project vs. embedded vs. rejected

Antigravity's own workspace-vs-global skill split (T1/T4) maps directly onto
a simple rule: **anything Kaggle-competition-specific is a workspace skill
under `.agents/skills/`; anything that is a genuinely generic engineering
habit (package management, git hygiene, cost discipline) is installed
globally once and inherited by every future competition project.** Full
disposition table for catalog skills is in §20; the split rule itself:

- **Global** (install once, reused by every future competition workspace):
  `uv-package-manager`, `using-git-worktrees`, `polars`, cost/reliability
  guardrail patterns (adapted from `runaway-guard`, `tool-use-guardian`).
- **Per-project / workspace** (`.agents/skills/`): everything Kaggle- and
  competition-specific — the five implemented in the scaffold plus the eight
  templated ones in §4, and any conditionally-relevant domain skill (e.g.
  Hugging Face fine-tuning skills, installed *only* when the active
  competition is CV/NLP/multimodal, not by default).
- **Embedded / rewritten as a local skill** (the catalog entry's *pattern* is
  worth keeping, but its content is too generic to use as-is): agent-memory
  concepts → rewritten as the `experiments/`+`meta-learning/` schema;
  `context-engineering`/`context-guardian` concepts → rewritten as the
  blackboard-write discipline in §5; `evaluation`/`agent-evaluation-reporting`
  concepts → rewritten as §16.
- **Inspiration only, do not install**: `multi-agent-task-orchestrator`
  (anti-duplication + quality-gates concept), `polis-protocol` (a "learning
  router assigns work by track record" concept — this is essentially the
  Agent Evolution trust-level mechanism in §8, independently arrived at),
  `odw`/open-dynamic-workflows (adversarial verification of parallel agents
  concept, already covered by the Adversarial Cell).
- **Rejected**: all `*-delegate` skills and third-party model-gateway skills
  (§2, standing rule), all off-domain persona/roleplay skills, and anything
  whose only relevance is superficial keyword overlap (e.g. general web/app
  builder skills).

## 8. Self-evolution mechanism

### Skill evolution (per the brief's pipeline, made concrete)

```
Failure logged (knowledge/failures/) → Knowledge Curator classifies root cause
→ Is it a one-off, or a pattern across ≥3 similar failures? → if pattern:
draft a skill candidate (SKILL.md + evaluation cases) → Adversarial Cell
reviews the draft the same way it reviews a model → human approves/rejects
(this step is deliberately NOT automated — the brief itself requires this)
→ versioned skill lands in .agents/skills/, logged in knowledge/skills/
→ A/B: run the next 2-3 comparable situations with and without the skill
→ promote (keep) or roll back, evidence recorded either way.
```

### Agent evolution

Each agent's `knowledge/agent-performance/<agent>/<slug>.yaml` (schema in
scaffold) tracks task success rate, false-positive rate, wasted compute
attributable to its calls, reproducibility, and — once available — downstream
private-LB impact. A **rolling summary** feeds the `trust_level` field in
`config/agent_roster.yaml`. Trust level changes what happens downstream, not
what the agent is allowed to *attempt*:

- `high` trust: downstream agents can act on this agent's output with a
  lighter secondary check.
- `medium` trust: a specific other agent re-checks before anything material
  happens (see `verified_by` column in the scaffold's roster file).
- `low` trust (e.g. Kaggle Executor, because headless execution is the
  least-reliable link in the whole chain, §1.1): output is **never** trusted
  without the run-sentinel file confirming completion.

Consistent with the brief's explicit instruction: **agents never
self-modify core orchestration.** Trust-level changes are proposed by the
Knowledge Curator from the performance data and require the same
human-approval step as a new skill.

## 9. Experiment scheduler

The Commander ranks candidate experiments by **expected information gain per
compute-hour**, not novelty, using three inputs: (a) open questions on the
blackboard, (b) `meta-learning/` priors for this `problem_type` +
`dataset_signature`, (c) remaining budget in `config/compute_budget.yaml`.
Concretely, an experiment is prioritized higher when it would *falsify* a
current assumption (e.g. "does this feature family actually help, or is the
CV gain within the noise band?") over one that just adds more of an already-
validated feature family. The full stage list and skip conditions are
implemented as `.agents/workflows/competition-pipeline.md` in the scaffold —
skip conditions are stated as concrete, checkable rules (e.g. "skip HPO if
baseline CV is already within the noise band of the best comparable
historical result for this `problem_type`"), not vague judgment calls.

## 10. Validation system

The Validation Architect (full checklist implemented in the scaffold's
`SKILL.md`) runs, in order: (1) grouping/temporal correctness against the
*real* prediction task, (2) target-leakage scan, (3) duplicate-entity check,
(4) **adversarial validation** (train a classifier to distinguish train vs.
test rows; a held-out AUC well above 0.5 means the split is distinguishable
and the CV scheme needs adjusting — T3, standard technique) (5) stratification
correctness, (6) CV/public-LB gap tracking across submissions. A **veto**
from this checklist is a hard stop for every downstream Cell — not a
recommendation. The Leakage Hunter runs a complementary, narrower scan
focused specifically on feature construction and external-data
contamination, and can independently raise a `confirmed`-severity blocking
issue with the same authority as a Validation Architect veto.

## 11. Kaggle automation flow

Given the headless-mode risk in §1.1, the Kaggle Executor's contract is
deliberately paranoid:

1. Confirm competition format (**Code Competition** vs. CSV upload) from
   `knowledge/kaggle/<slug>/submission_format.md` — these have materially
   different automation paths.
2. Prepare the notebook/script and — for Code Competitions — **smoke-test it
   in a clean session**, not just the interactive session it was written in,
   including the internet-off constraint if the competition enforces one (T2).
3. Any Kaggle CLI/API call that could be long-running or is invoked
   non-interactively writes a `run_sentinel` (see §5) before/while running;
   the orchestrator polls for it with a timeout rather than trusting captured
   output, exactly as it does for `agy` subagent calls.
4. On error, log the raw error to `knowledge/failures/<slug>/submission_errors.md`
   verbatim — submission-format failures are cheap to prevent the second time
   and expensive to keep re-discovering.
5. **No submission call is made without `blackboard/final_audit_pass.json`**
   existing and matching the `exp_id` being submitted (§17 compliance layer).

---

## 12. GitHub integration

**Git worktrees, one per active experiment line** (`using-git-worktrees` from
the catalog is a legitimate, direct fit here — §20): this lets the Model Cell
try 2-3 structurally different approaches without branch-switching churn, and
each worktree's result lands in the shared `knowledge/experiments/` ledger
regardless of which worktree produced it. Recommended pattern: one PR per
*promoted* experiment (not per raw run — that would be noise), with the PR
description auto-populated from the experiment YAML (hypothesis, intervention,
CV score, conclusion), so the repo history doubles as a human-readable
experiment log. Failure/negative-result experiments are still committed to
`knowledge/failures/` on the main line even when their worktree is discarded —
the knowledge is the point, not the code.

## 13. Artifact protocol

Every stage that produces output writes to a predictable, typed location so
the Artifact Analyst doesn't have to guess:

```
artifacts/<competition>/<exp_id>/
├── oof_predictions.parquet
├── test_predictions.parquet
├── model/                     # serialized model file(s)
├── metrics.yaml               # mirrors the experiment ledger's score fields
├── logs/
├── plots/                     # error analysis, calibration, feature importance
└── submission.csv             # only present once this exp_id is submission-ready
```

The Artifact Analyst's job is narrow and mechanical on purpose: parse this
fixed structure into the experiment ledger entry, flag anything missing
(e.g. an `exp_id` with a `cv_score` but no `oof_predictions.parquet` cannot be
used for ensembling — OOF predictions are required for correlation/diversity
checks) rather than silently proceeding with partial data.

## 14. Compute optimization system

`config/compute_budget.yaml` (scaffold) is a ledger, not a suggestion: every
stage has a pre-allocated GPU-hour budget, CPU-only stages get zero GPU
allocation by design (research, EDA, feature-hypothesis generation, leakage
scanning all run without touching the Kaggle GPU quota at all), and a hard
`reserve_hours_for_final_submission` amount is walled off from speculative
experimentation. Overspend by more than 25% on any stage halts that stage
and requires Commander re-approval — this is the mechanical enforcement of
the brief's "do not perform expensive HPO if the baseline is already strong"
instruction, turned into a number instead of a vibe.

## 15. Failure recovery system

Three failure classes, three different recoveries:

1. **Headless-invocation failure** (agy call hangs / drops output / a
   permission silently blocks a tool call, §1.1): bounded timeout →
   check for the run-sentinel file → if absent, retry once with
   `--dangerously-skip-permissions` inside a sandboxed/disposable environment
   → if still absent, escalate to blackboard `blocking_issues` and stop
   rather than guessing whether it worked.
2. **Experiment failure** (training errors, data errors): logged to
   `knowledge/failures/` with the *same* schema rigor as a success (§6) —
   this is knowledge, not noise, and directly prunes future search space for
   the same `problem_type`.
3. **Validation/leakage failure** (a veto): halts every downstream Cell until
   a human or the responsible agent resolves the specific `blocking_issue`;
   nothing "works around" a veto by proceeding on a different branch while
   it's open.

## 16. Evaluation framework

Two separate things are evaluated, and conflating them is a common mistake
this design avoids:

- **Experiment quality** (was the *hypothesis* well-formed, was the CV sound,
  did the conclusion match the evidence) — audited via the experiment ledger
  schema itself; an experiment missing a pre-registered hypothesis or a
  validation-strategy reference is treated as not well-formed enough to count,
  regardless of its score.
- **Agent quality** (§8's agent-performance tracking) — success rate,
  false-positive rate, wasted compute, reproducibility, downstream LB impact.

Benchmarking the *system* (as opposed to a single competition run) is
addressed separately in §22, because "did this competition go well" and "is
this architecture actually good" are different questions with different time
horizons.

---

## 17. Security / compliance layer

- **Model-routing compliance** (hard rule, enforced as an Antigravity rule
  file in the scaffold, `.agents/rules/kaggle-compliance.md`): no agent may
  invoke another coding CLI or reasoning model as a silent substitute. This
  is checked at the skill-catalog level (§2, §20) and restated as a runtime
  rule so it survives even if a future skill addition would otherwise violate
  it.
- **Competition-rule compliance**: internet-access-at-inference and external-
  data permissibility are **extracted and logged per-competition**
  (`knowledge/kaggle/<slug>/rules_extract.md`) before any experiment touches
  data — never assumed from a prior competition's rules, because these
  reliably vary competition-to-competition (T2).
- **Data-integrity compliance**: every join/aggregation is checked against
  fold boundaries; leakage is a veto condition, not a warning (§10).
- **Submission gate**: no submission without an experiment-ledger entry, a
  Final Auditor sign-off file, and a sanity diff against the previous
  submission (catches accidental identical/degenerate outputs).
- **Secrets handling**: Kaggle API credentials and any Google API keys used
  for headless auth are kept out of the knowledge base and out of git
  history entirely — treat this the same as any other credential, not as a
  Kaggle-specific concern warranting a bespoke exception.

---

## 18. Directory & configuration structure

Fully implemented in the scaffold; summarized here:

```
project-root/
├── AGENTS.md                       # root charter, read every session
├── KAGGLE-OS-BLUEPRINT.md          # this document
├── capabilities/
│   ├── probe.py                    # detects volatile facts, writes:
│   └── detected.json               # ← generated, never hand-edited
├── .agents/
│   ├── skills/<agent-name>/SKILL.md            (13 total, 5 implemented)
│   ├── rules/kaggle-compliance.md
│   └── workflows/competition-pipeline.md
├── blackboard/
│   ├── state.json                  # ← generated at runtime, schema below
│   ├── state_schema.json
│   └── final_audit_pass.json       # ← only the Final Auditor writes this
├── config/
│   ├── agent_roster.yaml           # trust levels, cell membership
│   ├── compute_budget.yaml         # the GPU/CPU ledger
│   └── mcp_servers.yaml            # explicit allowlist
├── knowledge/                      # full tree in §6
├── artifacts/<competition>/<exp_id>/   # full layout in §13
└── (your actual competition code, notebooks, worktrees)
```

Configuration is deliberately split into **volatile** (`capabilities/detected.json`,
regenerated by a script) vs. **stable-but-editable** (`config/*.yaml`, human-
tunable policy) vs. **runtime state** (`blackboard/*.json`, machine-written) —
never mix these three, or the system starts trusting stale assumptions as if
they were live facts, which is exactly the failure mode §1.1 and §25 warn
about.

---

## 19. Workflows

### Initialization
`capabilities/probe.py` → confirm Antigravity + Kaggle CLI are found and note
which global-skill path is actually live → classify the competition →
Competition Researcher extracts rules → Data/EDA Agent profiles the data →
Validation Architect's first checklist pass (must APPROVE before any model
training).

### Competition execution
The full staged pipeline with skip logic is `.agents/workflows/competition-pipeline.md`
in the scaffold (§9 summarizes the ranking logic that drives it).

### Cross-competition learning
```
raw experiments → validated discoveries (Knowledge Curator promotion, §6)
→ generalizable knowledge (meta-learning/ hypothesis update, evidence_for/against)
→ strategy extraction (strategies/, cited to source exp_ids)
→ cross-competition memory (playbooks/, once evidence is dense enough)
```
When a new competition starts, the Commander queries `meta-learning/` and
`playbooks/` for matching `dataset_signature` patterns *before* proposing a
baseline — this is what makes the system actually get better at Kaggle over
time rather than just accumulating logs.

### Skill evolution / Agent evolution
Both detailed in §8, both require a human-approval step by design.

---

## 20. Recommended skills (from the supplied catalog) & externally discovered tools

### From the supplied 2,121-skill catalog — disposition table

| Skill | Disposition | Why |
|---|---|---|
| `uv-package-manager` | **Install globally** | Generic, high-value, zero downside for any future Python project |
| `using-git-worktrees` | **Install globally** | Directly matches §12's parallel-experiment-line pattern |
| `polars` | **Install per-project** | Real fit for large tabular data that outgrows pandas; not needed for small/CV/NLP competitions |
| `scikit-learn` | **Install per-project** | Standard tabular baseline tooling |
| `matplotlib` / `seaborn` / `plotly` | **Install per-project** | EDA & error-analysis plotting; `plotly` specifically for interactive artifact review |
| `statsmodels` | **Install per-project (tabular/time-series)** | ADF-type distribution-shift and time-series diagnostics |
| `context-compression`, `context-optimization` | **Embed / rewrite locally** | Concepts are right, generic implementation isn't Kaggle-schema-aware — folded into blackboard discipline (§5) |
| `agent-memory`, `agent-memory-mcp`, `memory-systems` | **Embed / rewrite locally** | Same reasoning — rewritten as the `knowledge/` schema (§6), which is more structured than a generic hybrid-memory store needs to be |
| `evaluation`, `advanced-evaluation`, `agent-evaluation-reporting` | **Embed / rewrite locally** | Rewritten as §16's two-track (experiment quality vs. agent quality) framework |
| `data-quality-frameworks` | **Use as inspiration** | Great Expectations/dbt-test concepts are relevant to Data Forensics but the skill itself targets production data pipelines, not one-shot competition datasets |
| `optim-agent` | **Use as inspiration** | Generic HPO-agent pattern; the actual HPO logic needs to be Kaggle/tabular-specific, built locally |
| `multi-agent-task-orchestrator` | **Use as inspiration** | Anti-duplication + quality-gate concept, already reflected in the Commander/blackboard design |
| `polis-protocol` | **Use as inspiration** | "Learning router assigns work by track record" is independently arrived at as the Agent Evolution trust-level mechanism (§8) |
| `odw` (open-dynamic-workflows) | **Use as inspiration** | Adversarial-verification-of-parallel-agents concept, already covered by the Adversarial Cell |
| `tool-use-guardian`, `runaway-guard` | **Install globally (adapted)** | Cost/reliability guardrail habits worth having on every project, including outside Kaggle |
| `moyu` (anti-over-engineering guardrail) | **Install globally (adapted)** | Good philosophical fit for "don't ensemble correlated models without justification" / "what can be eliminated rather than added" |
| Hugging Face family (`hugging-face-model-trainer`, `-jobs`, `-vision-trainer`, `-trackio`, `trl-training`, `unsloth-finetuning`, `train-sentence-transformers`) | **Install per-project, conditionally** | Only for CV/NLP/multimodal competitions requiring fine-tuning; irrelevant overhead for tabular competitions |
| `*-delegate` family (`aider-delegate`, `claude-delegate`, `codex-delegate`, `cursor-delegate`, `grok-delegate`, `kimi-delegate`, `agy-delegate`, etc.) | **Rejected (standing rule)** | Silent model/CLI substitution — violates the hard constraint categorically |
| `routerbase-model-gateway`, `sandbase-mcp`, `unified-ai-gateway` | **Rejected (standing rule)** | Same reasoning — third-party model routing |
| Persona/roleplay skills (`yann-lecun*`, `sam-altman`, `ilya-sutskever`) | **Rejected** | Off-domain, no Kaggle relevance, invites unearned authority on technical claims |
| `agentmail`, `agentphone`, `aomi-transact`, `fal-*`, `molykit`, blockchain/crypto/voice-agent skills | **Rejected** | Off-domain |
| The remaining ~2,000 skills (security, mobile, frontend, marketing, database-migration, devops categories, etc.) | **Rejected (not relevant)** | No plausible path to improving Kaggle leaderboard performance |

### Externally discovered (not in the catalog) — recommended for the local skill layer

These are general, slow-changing ML-engineering tools/techniques (T3/T5 —
existing knowledge, not verified via fresh search this session; re-confirm
current library APIs before building against them, since APIs move faster
than the underlying techniques):

- **Gradient boosting**: XGBoost, LightGBM, CatBoost — the catalog has
  *none* of these; this is the single biggest gap identified in §2 and
  should be the first local skill built (`.agents/skills/gradient-boosting-suite/`).
- **AutoML-as-a-baseline-generator**: AutoGluon (tabular) as a fast, strong
  baseline to calibrate the noise band against, not as the final model.
- **HPO**: Optuna, for the Model Researcher/HPO agent's actual tuning logic
  (the catalog's `optim-agent` is a generic pattern, not an implementation).
- **Feature engineering utilities**: `category_encoders` (target/leave-one-out
  encoding done fold-safely), `feature-engine`.
- **Experiment tracking (lightweight, local-first)**: a simple local
  MLflow-style run log is a reasonable implementation backend for the
  `knowledge/experiments/` schema — the schema is the contract, MLflow (or a
  flat-file store, which the scaffold uses to avoid an extra service
  dependency) is an implementation detail.
- **Data versioning**: DVC or plain content-hashing of input datasets, so an
  experiment's `dataset_signature` is reproducible, not just descriptive.

---

## 21. Implementation phases

| Phase | Scope | Exit criteria |
|---|---|---|
| **0 — Grounding** | Run `capabilities/probe.py`; confirm actual Antigravity CLI version, global-skill path, and Kaggle CLI on the real machine; re-verify current GPU quota in-app | `capabilities/detected.json` populated and manually sanity-checked |
| **1 — Scaffold** | Land the directory structure, the 5 implemented skills, the compliance rule, the blackboard schema (this document's scaffold) | Antigravity loads the workspace skills without error |
| **2 — Single-competition dry run** | Run the full pipeline manually-supervised on one *completed* (so ground truth exists) past tabular competition, human approving every stage transition | Every stage produces a well-formed artifact per §13; no blocking issue is silently bypassed |
| **3 — Remaining agents** | Build the other 8 agent skills using the same template, add the gradient-boosting/HPO local skills from §20 | Full agent roster present; local skill gap from §2/§20 closed for tabular problem_type |
| **4 — Reduced-supervision run** | Real, currently-running competition; human approves only at Cell boundaries and the submission gate, not every micro-decision | System completes a full cycle (baseline → submission) without a human writing an experiment entry by hand |
| **5 — Cross-competition learning validation** | Second competition of a similar `problem_type`; confirm `meta-learning/` priors actually change the Commander's first-experiment choice vs. Phase 2's cold start | Measurable reduction in "wasted" early experiments vs. Phase 2 baseline |
| **6 — CV/NLP/multimodal extension** | Conditionally install the Hugging Face skill family (§20); build the missing domain-specific local skills for those problem types | A non-tabular competition run reaches the same rigor bar as Phase 4 |

Do not skip Phase 2. Running the untested pipeline unsupervised on a live,
scored competition before it has been exercised against a known-outcome past
competition is how a headless-mode bug (§1.1) turns into a wasted submission,
not just a caught error.

## 22. Benchmark methodology (for the system itself, not one competition)

Backtest against **completed** Kaggle competitions with known public and
private leaderboards, across at least 3 `problem_type`s (tabular, CV or NLP,
time-series), measuring:

- percentile the system's best CV-selected submission would have landed at
  on the *private* leaderboard (the number that actually matters — public LB
  is the metric most prone to overfitting via repeated submission),
- number of human interventions required per run (the "per unit of human
  effort" half of the brief's own optimization target),
- compute-hours spent vs. quota (the "per unit of Kaggle compute" half),
- CV/private-LB gap, as a direct measure of whether the Validation Architect's
  checklist is actually working, not just running.

Re-run this backtest after every Phase in §21 and after any skill/agent
promotion (§8) — a promotion that doesn't measurably move these numbers on
the backtest set is a candidate for rollback, not a permanent addition.

## 23. Exact commands (verified vs. best-known — treat accordingly)

| Command | Status | Notes |
|---|---|---|
| `agy -p "<prompt>" --dangerously-skip-permissions` | **Best-known (T4)**, re-verify | The unattended-invocation shape reported by third-party guides; official flag reference not directly confirmed in this research pass |
| `agy plugin install <url>` | **Best-known (T4)** | Reported plugin-install syntax |
| `agy agents run <subagent-name>` | **Best-known (T4)** | Reported subagent-invocation syntax |
| `.agents/skills/<name>/SKILL.md` (YAML frontmatter: `name`, `description`) | **Corroborated across multiple independent sources (T1/T4)** | The one path every source agrees on |
| `kaggle competitions submit -c <slug> -f <file> -m "<message>"` | **Standard, long-stable Kaggle CLI syntax (T2)**, re-verify flags | Confirm against `kaggle --help` — do not hand-copy blindly, per §25 |
| Global skill path (`~/.gemini/config/skills/` or similar) | **Unresolved — probe, don't assume (T4)** | See §1.1, §18 |

## 24. Worked example: one full competition pass (tabular, illustrative)

1. **Init** — Competition Researcher extracts: metric = RMSE, Code Competition,
   internet disabled at scoring, no external data restriction stated. Written
   to `knowledge/kaggle/<slug>/`.
2. **Data/EDA** — 50 columns, one high-cardinality categorical (200k levels),
   mild target skew, no obvious duplicate rows. `dataset_signature` computed
   and matched against `meta-learning/` → one existing hypothesis found:
   *"high-cardinality categorical + medium tabular size favors CatBoost as a
   baseline"* (`confidence: strong_pattern`, 4 prior competitions for, 0
   against).
3. **Validation Architect** — GroupKFold on a natural entity grouping found
   in the data (not a naive random split); adversarial-validation AUC = 0.51
   (no meaningful shift) → **APPROVE**.
4. **Baseline** — CatBoost, default params, per the retrieved prior. CV
   logged as `exp_id: rmse-comp-0001`, `confidence: weak_hypothesis` until it
   replicates.
5. **Fast experiments** — 3 cheap feature hypotheses tested (target encoding
   of the categorical done fold-safely, two interaction features); one beats
   noise band, two don't — all three logged, including the two failures.
6. **Model search** — CatBoost (already strong) + LightGBM as a diversity
   candidate. HPO **skipped** — Commander logs: "baseline within 0.4% of the
   best comparable historical RMSE for this `dataset_signature`; expected HPO
   gain does not justify the 6-hour GPU allocation."
7. **Ensembling** — OOF correlation between CatBoost and LightGBM = 0.89
   (diverse enough) → weighted blend tested, beats either alone.
8. **Adversarial Cell** — Leakage Hunter flags the target-encoded feature for
   a fold-safety re-check (confirms it's fine, closes the flag); CV/Error
   Auditor finds no systematic subgroup failure.
9. **Kaggle run** — notebook smoke-tested in a clean session with internet
   disabled, matching the competition's real scoring environment; run
   sentinel confirms completion; artifacts ingested.
10. **Result analysis** — public LB score within the recorded noise band of
    CV → no red flag.
11. **Knowledge update** — the CatBoost-baseline hypothesis's `evidence_for`
    count increments to 5; the two failed feature hypotheses are logged to
    `knowledge/failures/` so the next similar competition doesn't re-try them
    from scratch.
12. **Final Auditor** — full checklist passes, writes
    `blackboard/final_audit_pass.json`.
13. **Submission** — Kaggle Submission Packager submits only now, with the
    sign-off file present.

---

## 25. Known limitations

- **Headless-execution reliability is a real, documented open risk** (T4,
  §1.1), not a hypothetical edge case — this is the single biggest reason
  "minimal human intervention" cannot mean "zero supervision" yet. The
  filesystem-blackboard design mitigates it but does not eliminate the
  underlying CLI bug; if Google ships a fix, re-evaluate whether the
  sentinel-polling overhead can be relaxed.
- **Documentation on Antigravity's global-skill path is inconsistent across
  sources as of this research pass** (§1.1) — resolved by probing, not by
  picking one source to trust.
- **The supplied skill catalog has no Kaggle- or tabular-competition-specific
  content** (§2) — the domain core of this system is hand-built, not
  assembled from existing skills, and its quality is bounded by the
  engineering effort actually put into the local skills, not by the catalog's
  size.
- **Gemini 3.8 Flash High is a workhorse-tier model, not the frontier-reasoning
  tier** (T1, §1.2). This system is designed to reliably capture the gains
  that come from rigor, thoroughness, and leak/overfit avoidance — it is not
  designed to assume superhuman novel-insight generation, and competitions
  that are won primarily on a single brilliant, non-obvious modeling idea are
  the hardest case for this architecture, same as they would be for most
  human teams without deep domain expertise in that specific problem.
- **Kaggle's GPU quota (~30h/week) and session limits are a hard ceiling**
  (T2/T4) that no amount of orchestration quality can compute around — the
  system optimizes what to spend that budget on, it cannot create more of it.
- **Full autonomous self-evolution is deliberately not implemented** — skill
  and agent-trust promotion require a human-approval step, per the brief's
  own explicit instruction not to auto-promote speculation or auto-install
  skills. "Minimal human intervention" has an intentional floor, not an
  oversight.
- **This document does not, and should not, claim a guaranteed #1 finish.**
  It defines the optimization target (top-1%/top-5%/top-10%/medal
  probability, per §4 of the brief) and the measurement framework (§22) for
  tracking progress toward it honestly.
- **Antigravity CLI is young and still changing rapidly** (public preview →
  November 2025; CLI rebrand → May/June 2026; multiple corroborating sources
  describe breaking changes within months of each other). Treat every exact
  command in §23 as best-known-as-of-September-2026, and re-verify before
  depending on it in an unattended run.

---

*End of blueprint. The accompanying `kaggle-os-scaffold.zip` implements the
directory structure, 5 of the 13 agent skills, the blackboard schema, the
capabilities probe, and the compute/roster/knowledge configs described above
— follow the same pattern to complete the remaining 8 agents (§4, §21 Phase 3).*
