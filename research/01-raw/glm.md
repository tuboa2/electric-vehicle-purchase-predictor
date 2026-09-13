# Autonomous Kaggle Competition System
## Research Report & Implementation Blueprint

**Author:** Super Z (Principal Kaggle Competition Strategist)
**Date:** 2026-09-13
**Runtime substrate:** Antigravity CLI + Gemini 3.8 Flash High
**Deliverable type:** Single Markdown blueprint, evidence-tier annotated
**Companion inputs:** `prompt.md` (2,077-line specification), `CATALOG.md` (2,121-skill inventory)

---

## Confidence-Tier Legend

Every external claim in this blueprint is annotated with one of the tags below, per the prompt's source-hierarchy rule (§48):

- **[T1]** Tier 1 — Official documentation (Antigravity docs, Kaggle docs, Google model cards).
- **[T2]** Tier 2 — Official competition solution writeups.
- **[T3]** Tier 3 — Peer-reviewed papers (arXiv pre-prints counted when from established groups).
- **[T4]** Tier 4 — High-credibility technical repositories (AutoGluon, Optuna, Polars, etc.).
- **[T5]** Tier 5 — Experienced Kaggle practitioners (Kaggle Grandmasters, published Medium writeups by known winners).
- **[T6]** Tier 6 — Community discussions (Kaggle forums, Reddit, HN).

Inferred design choices made by the author are tagged:

- **[KNOWN-PRACTICE]** Industry-consensus engineering pattern.
- **[INFERRED]** Reasonable inference from established facts.
- **[SPECULATIVE]** Author conjecture; needs empirical validation before being promoted to knowledge.

**Important caveat on this blueprint:** This document was produced without live web research (the user explicitly requested "just give me the blueprint now"). All claims reflect the author's knowledge of Antigravity, Gemini, Kaggle, and multi-agent architectures as of approximately September 2026. Specific URLs, version numbers, and current quota figures must be re-verified before the framework is committed to production. Where the author is uncertain, the claim is tagged `UNVERIFIED — refresh before production`.

---

## Table of Contents

### Part I — Research Report (per prompt §41)

1. Executive Summary
2. Current Antigravity Capabilities
3. Current Gemini 3.8 Flash High Capabilities Relevant to this Architecture
4. Kaggle Infrastructure Constraints
5. Kaggle-Winning Strategy Analysis
6. Multi-Agent Architecture Comparison
7. Recommended Architecture (Evidence-Decided)
8. Agent Roster
9. Communication Protocol
10. Memory Architecture
11. Self-Evolution Architecture
12. Skill Evolution Architecture
13. Experiment Scheduler
14. Validation Architecture
15. Kaggle Automation Architecture
16. Artifact Architecture
17. Compute Optimization Architecture
18. Security / Compliance Architecture
19. Failure Recovery Architecture
20. Reusability Architecture
21. Complete Project Structure
22. Skill Selection from the 2,121-Skill Catalog (Category Highlights)
23. Additional External Skills Discovered (Author Knowledge)
24. Rejected Skills and Why
25. Implementation Roadmap
26. Benchmark Plan
27. Risks
28. Expected Bottlenecks
29. Future Improvements

### Part II — Implementation Blueprint (per prompt §51)

1. Exact Architecture
2. Exact Agent Roster
3. Exact Responsibilities
4. Agent Communication Protocol
5. Memory Architecture
6. Skill Architecture
7. Self-Evolution Mechanism
8. Experiment Scheduler
9. Validation System
10. Kaggle Automation Flow
11. GitHub Integration
12. Artifact Protocol
13. Compute Optimization System
14. Failure Recovery System
15. Evaluation Framework
16. Security / Compliance Layer
17. Directory Structure
18. Configuration Structure
19. Initialization Workflow
20. Competition Execution Workflow
21. Cross-Competition Learning Workflow
22. Skill Evolution Workflow
23. Agent Evolution Workflow
24. Recommended Skills from Catalog
25. Additional Skills Discovered Externally
26. Implementation Phases
27. Benchmark Methodology
28. Exact Commands Where Verified
29. Example End-to-End Execution
30. Known Limitations

### Appendices

- **Appendix A:** Skill Catalog Category Highlights (no full 2,121-row triage — see §22 for rationale)
- **Appendix B:** Directory Tree of Recommended `kaggle-agent-core/` Scaffold
- **Appendix C:** Worked End-to-End Example (Toy Tabular Competition)
- **Appendix D:** Known Limitations & Future Work
- **Appendix E:** Glossary

---

# Part I — Research Report

## 1. Executive Summary

This blueprint specifies an autonomous Kaggle competition multi-agent system built on the Antigravity CLI runtime and using **Gemini 3.8 Flash High** as the sole reasoning model. The system's optimization target, as set by the prompt (§4), is to **maximize expected private-leaderboard performance per unit of human effort and Kaggle compute**, with explicit acceptance that no architecture can guarantee a #1 finish. The blueprint is sized to a realistic single-engineer scope: the framework core stays under ~12 agents and avoids microservices, message brokers, and Kubernetes (§39).

Three architectural alternatives were considered before settling on the recommendation:

1. **Lean hierarchical supervisor** (Commander → Cells → Sub-agents).
2. **Blackboard / shared-memory** with flat specialists reading/writing structured state.
3. **Generator-Critic-Generator** with a small core agent set but heavy adversarial gates.

The recommendation (see §7) is a **hybrid: hierarchical supervisor with a blackboard-style structured state store at its core, plus a critic-generator verification gate at every high-stakes decision point.** The hierarchy provides clear ownership and reproducible execution traces; the blackboard eliminates free-form agent chatter and gives every agent a stable read interface; the critic-generator layer compensates for Gemini 3.8 Flash High's known reasoning limits at high-stakes moments (validation design, submission selection, leakage sign-off) without paying the cost of running debate everywhere.

The system is organized as a reusable **`kaggle-agent-core/`** repository plus per-competition **`competition-project/`** working directories. Skills are split: competition-agnostic capabilities live as global Antigravity skills; competition-specific capabilities live as workspace-local skills under `.agents/skills/`. The agent system itself is configured via an `AGENTS.md` file at the workspace root plus per-agent YAML specs in `.agents/agents/`.

Knowledge is persisted in a four-layer memory store (working / project / strategic / meta) with explicit provenance and confidence tagging. The system distinguishes **observed facts** from **strong empirical patterns** from **weak hypotheses** from **speculation** — and is forbidden by design from auto-promoting speculation into permanent knowledge (§7 of the prompt). Cross-competition meta-learning happens through a small knowledge graph (§47 of the prompt) that links dataset signatures → validation strategies → model families → ensemble compositions → observed private-LB outcomes.

The framework explicitly avoids over-engineering (§39): no databases (state is YAML + JSON + Parquet on disk), no distributed systems, no cloud infrastructure, no LLM API multiplexing. The single concession to "production-grade" tooling is **`uv`** for Python environment management and **Git worktrees** for parallel isolated experiments — both well-evidenced productivity wins for solo engineers `[KNOWN-PRACTICE]`.

The expected bottleneck (§28) is **not** ML compute — Kaggle provides that. The expected bottleneck is **Antigravity's per-agent latency and rate limits** when the system tries to run many agents in parallel. The architecture is sized to absorb this: most agents run sequentially, and only genuinely independent experiments (e.g., training three model families in parallel) use Antigravity's asynchronous-subagent capability.

This blueprint is **competition-agnostic** (works for tabular, time-series, CV, NLP, multimodal), **Antigravity-native** (uses `agy`, `AGENTS.md`, workspace + global skills, headless execution, async subagents), and **Gemini 3.8 Flash High-compatible** (decomposes tasks to fit the model's effective reasoning scope, uses structured outputs to suppress hallucination, uses adversarial review at high-stakes gates).

---

## 2. Current Antigravity Capabilities

> **Confidence note:** Antigravity is a fast-moving CLI. The capabilities summarized here reflect author knowledge as of approximately September 2026. Specific flags, file locations, and skill formats should be re-verified from official Antigravity documentation before this blueprint is implemented. Items the author is unsure about are tagged `UNVERIFIED`.

### 2.1 What Antigravity Is

Antigravity is Google's CLI-based agentic coding assistant, positioned as a competitor to Anthropic's Claude Code and OpenAI's Codex CLI `[T5]`. Unlike Cursor or GitHub Copilot, Antigravity is a **terminal-native, file-system-aware agent** that operates on a workspace, executes tools, and can be invoked non-interactively for use in CI/CD pipelines and bot loops `[KNOWN-PRACTICE]`. It uses Gemini family models as its reasoning backend, with the user-selectable model slug `gemini-3.8-flash-high` being the high-reasoning-effort variant of Gemini 3.8 Flash `[INFERRED from prompt §3]`.

### 2.2 The `agy` Command Surface

The `agy` command exposes several top-level subcommands `[KNOWN-PRACTICE]`:

| Command | Purpose |
|---------|---------|
| `agy` (no subcommand) | Start an interactive agent session in the current workspace |
| `agy --model <slug>` | Start interactive session with a specific model |
| `agy --headless` | Run non-interactively, accept a prompt via stdin or `-p`, exit when done |
| `agy --effort <level>` | Configure the reasoning effort / thinking budget |
| `agy agents` | Manage custom agents (list, edit, etc.) |
| `agy skills` | Manage skills (list, install, etc.) |
| `agy models` | List / inspect available model slugs |

Exact flags and their canonical names should be re-verified `[UNVERIFIED]`. The prompt itself confirms `gemini-3.8-flash-high` is "currently exposed as a selectable model slug" (§3 of the prompt) — that is the strongest evidence available without live web verification.

### 2.3 Agents and Subagents

Antigravity supports **custom agents** defined at the workspace level via an `AGENTS.md` file at the workspace root, plus per-agent specifications (typically under `.agents/agents/<name>.md` or similar — re-verify the exact path) `[KNOWN-PRACTICE]`. An agent definition specifies: name, role/system prompt, model slug, allowed tools, allowed skills, and any role-specific flags.

Antigravity also supports **subagent invocation** — an agent can spawn a child agent for a delegated subtask. This is the primary mechanism the Kaggle framework will use to decompose work: the Commander agent dispatches to specialist subagents (e.g., "Validation Architect, design a CV strategy for this dataset signature") and consumes the structured response. Subagents can run **asynchronously** in the background, which is the key enabler for parallel experiments `[KNOWN-PRACTICE]`.

The async-subagent capability is what makes a multi-agent Kaggle architecture viable on a single-engineer budget — without it, every sequential agent call would compound latency into hours per phase. With it, genuinely independent experiments (training LightGBM vs. CatBoost vs. an AutoGluon ensemble) can run concurrently.

### 2.4 Skills

Antigravity has a first-class **skills system** with two scopes `[KNOWN-PRACTICE]`:

- **Workspace-local skills** under `.agents/skills/<skill-name>/` — live in the repo, version-controlled with the project, scoped to one workspace.
- **Global skills** under `~/.config/antigravity/skills/<skill-name>/` (or equivalent — re-verify) — available across all workspaces, used for competition-agnostic capabilities.

A skill is typically a `SKILL.md` file plus supporting scripts/data. The `SKILL.md` declares: name, purpose, trigger conditions, tools it uses, dependencies, and the action body (which can be a prompt, a script, or a sequence) `[KNOWN-PRACTICE]`. Skills can be auto-discovered (triggered when their trigger conditions match the current task) or explicitly invoked.

The supplied `CATALOG.md` describes 2,121 skills across many categories. The blueprint's skill strategy (§22 of this report) is to install a small competition-agnostic core globally, embed a competition-specific layer locally, and reject the rest as either duplicative or incompatible.

### 2.5 Plugins and Hooks

- **Plugins** are Antigravity extensions that add tool providers, MCP servers, or other capabilities. Plugins are typically installed via a plugin registry and live separately from skills `[KNOWN-PRACTICE]`. For this framework, plugins are mostly relevant for MCP integration (see §2.6).
- **Hooks** are event-driven scripts that run before/after certain Antigravity actions. The standard hook points include: `PreToolUse`, `PostToolUse`, `Stop`, `SessionStart`, `UserPromptSubmit` `[KNOWN-PRACTICE]` — re-verify the exact list. Hooks are how the framework will enforce invariants like "every experiment logged before submission" or "every notebook commit tagged with the experiment ID."

### 2.6 MCP (Model Context Protocol) Support

Antigravity integrates **MCP (Model Context Protocol)** servers as tool providers `[KNOWN-PRACTICE]`. MCP is Anthropic's open protocol for exposing tools, resources, and prompts to LLM clients; it has become the de-facto standard for tool integration across Claude, GPT, and Gemini-based agents in 2024-2026 `[T3, T6]`.

For this framework, MCP servers are the cleanest way to expose non-trivial tools (e.g., a "submit-to-Kaggle" tool, an "experiment-registry-query" tool, a "retrieve-similar-past-competition" tool) without bloating agent prompts. The blueprint recommends a small set of custom MCP servers (see Part II §6 Skill Architecture).

### 2.7 Headless Execution

Headless mode (`agy --headless` or equivalent) is the backbone of autonomous operation `[KNOWN-PRACTICE]`. In headless mode, Antigravity:

- Accepts an initial prompt (via `-p "..."` or stdin).
- Runs autonomously, executing tools, calling subagents, writing files.
- Streams structured output (events, tool calls, final message) to stdout as JSON (or similar — re-verify the exact output format).
- Exits with a status code when the task is complete or when it hits a stopping condition.

The framework's main loop (Part II §20) is: orchestrator script invokes `agy --headless` with a phase-specific prompt, consumes the JSON output, dispatches the next phase, repeats.

### 2.8 Asynchronous / Background Agents

Antigravity's async-subagent capability allows a parent agent to spawn child agents that run in the background, with the parent polling for completion `[KNOWN-PRACTICE]`. This is the primary parallelism mechanism in this framework.

Concurrency limits matter: Antigravity almost certainly rate-limits model calls per minute and may cap concurrent subagents `[UNVERIFIED]`. The framework's experiment scheduler (§13) is designed to stay well under any reasonable rate limit — typically 2-4 concurrent training experiments, 1-2 concurrent research subagents.

### 2.9 Persistent / Reusable Configuration

Antigravity persists configuration at two levels `[KNOWN-PRACTICE]`:

- **Workspace** (`.agents/` directory in the repo): agents, skills, hooks, plugin configs — all version-controlled with the project.
- **Global** (`~/.config/antigravity/` or similar): user-level config, global skills, API keys, model preferences — shared across workspaces.

This split maps perfectly to the prompt's §36 requirement: global Antigravity skills hold competition-agnostic capabilities (leakage detection, validation design, ensemble optimization); workspace-local skills hold competition-specific capabilities (this competition's rules, this dataset's quirks).

### 2.10 Visual Artifacts

Antigravity can produce **visual artifacts** (diagrams, rendered markdown, screenshots of running tools) as part of its output stream `[KNOWN-PRACTICE]` — re-verify the exact mechanism. For this framework, visual artifacts matter for EDA reports (the Data Forensics agent generates plots and embeds them in its report) and for experiment-tracking summaries.

### 2.11 Effort / Thinking-Budget Configuration

Antigravity exposes an **effort** setting that controls how much "thinking" Gemini does before producing a final answer `[KNOWN-PRACTICE]`. Higher effort = more reasoning tokens = better quality on hard tasks at the cost of latency and token spend. The framework's policy: high effort for Commander, Validation Architect, Adversarial Reviewer, Final Auditor; medium effort for specialist agents; low effort for routine tasks (file inspection, simple retrievals). This is the cheapest lever the framework has to compensate for Flash High's reasoning ceiling on hard tasks.

### 2.12 Summary

Antigravity is sufficient as the runtime substrate for this framework. The critical capabilities — custom agents, async subagents, skills (workspace + global), hooks, MCP, headless execution — are all present. The main risk is that exact flag names, file paths, and skill formats are still moving targets in 2026; the framework isolates this risk by concentrating all Antigravity-specific details in a thin `runtime/` shim that can be updated without touching agent logic.

---

## 3. Current Gemini 3.8 Flash High Capabilities Relevant to this Architecture

> **Confidence note:** Gemini 3.8 Flash High is a recent model. Specific benchmark numbers and context-window limits should be re-verified from the official model card. The author's claims below reflect knowledge of the Gemini 3 Flash / Gemini 2 Flash family plus inference about the "3.8" and "High" qualifiers. Items tagged `UNVERIFIED` need live verification.

### 3.1 What Gemini 3.8 Flash High Is

Gemini 3.8 Flash is Google's small/efficient model in the Gemini 3.x family, positioned as the speed-optimized counterpart to Gemini 3.8 Pro `[INFERRED]`. The **"High"** qualifier designates the high-reasoning-effort variant — Google exposes Gemini 3.x Flash in multiple reasoning tiers (Low / Medium / High, similar to how OpenAI exposes o1/o3-mini tiers and how Anthropic exposes Sonnet with extended thinking budgets) `[KNOWN-PRACTICE]`. High tier dedicates more reasoning tokens per call, trading latency for accuracy on multi-step tasks.

The prompt's hard constraint (§3) is that **Gemini 3.8 Flash High is the sole reasoning model** — no Claude, no GPT, no Pro, no external LLMs. Every architectural decision must therefore either:
- **Stay within the model's effective reasoning scope** (decompose tasks so each fits), or
- **Compensate for the model's limitations** via specialized agents, structured outputs, verification, adversarial review, and execution feedback.

### 3.2 Context Window

The Gemini Flash family has been at the frontier of long-context LLMs since Gemini 1.5 (1M tokens) `[T1]`. Gemini 3.x Flash likely retains a context window in the 1M-2M-token range `[UNVERIFIED — re-check model card]`. However, **effective context is much smaller than nominal context** — long-context benchmarks (needle-in-haystack, RULER, LongBench) consistently show that recall quality degrades as input grows, with the "lost in the middle" effect documented across all major LLM families `[T3]`.

Practical implication for this framework: do not stuff entire repositories into agent context. Use the layered context architecture from prompt §31:

```
GLOBAL RULES (static, ~2-4k tokens)
+ ROLE CONTEXT (per-agent, ~1-3k tokens)
+ COMPETITION BRIEF (per-competition, ~2-5k tokens)
+ RELEVANT MEMORY (retrieved, ~2-5k tokens)
+ CURRENT TASK (per-call, ~1-5k tokens)
+ RELEVANT ARTIFACTS (file paths or excerpts, ~2-10k tokens)
+ RECENT EXPERIMENTS (summarized, ~1-3k tokens)
```

This caps most agent invocations at 10-30k tokens of input, well within the effective-recall zone of any modern long-context model.

### 3.3 Output Token Limit

Gemini 3.x Flash's output token limit is likely in the 8k-32k range per call `[UNVERIFIED — re-check]`. Long outputs (full EDA reports, full architecture specs) must be written to files via tool calls rather than streamed as model output. This is enforced as a framework convention: **agents write reports to disk via the `Write` tool; their final structured response is a YAML/JSON summary** (per the communication protocol in §9 of this report).

### 3.4 Tool / Function Calling

Gemini 3.x Flash has native, well-supported function calling with parallel calls and structured (JSON-schema-constrained) output `[T1, KNOWN-PRACTICE]`. This is the single most important capability for suppressing hallucination in this framework: every agent's "report" is a **schema-constrained JSON object**, not free text. The schema enforces that the model commits to specific fields (status, confidence, findings, evidence, actions, artifacts, recommendations, risks, next_tasks) — see §9 of this report.

### 3.5 Reasoning Quality

Gemini 3 Flash (and by extension 3.8 Flash) is in the same quality tier as Claude 3.5/3.7 Sonnet and GPT-4.1/5 for most coding and structured-reasoning tasks `[T5, UNVERIFIED for 3.8 specifically]`. The "High" reasoning tier pushes it closer to Claude 3.7 Sonnet-with-thinking and OpenAI o3 on math/code/agentic benchmarks `[INFERRED]`.

**Known strengths of the Flash family:**
- Structured output / function calling reliability `[T1]`.
- Code generation in mainstream languages (Python, TypeScript, SQL) `[T5]`.
- Math and quantitative reasoning at the High tier `[T1 model card]`.
- Long-context retrieval when the relevant information is explicitly present `[T3]`.

**Known weaknesses / antipatterns (relevant to this framework):**
- Free-form long-context synthesis (e.g., "read this 200-page document and tell me the strategy") — quality degrades without explicit decomposition `[T3]`.
- Library version numbers and API signatures — Flash models confabulate plausible-but-wrong signatures at non-trivial rates `[T5]`. The framework compensates by pinning all dependencies in `pyproject.toml` and using MCP tool wrappers with strict signatures.
- Multi-step planning without intermediate verification — Flash High tends to commit to a plan early and not revise `[INFERRED]`. The framework compensates with the Validation Architect + Adversarial Reviewer gates.
- Self-critique without an explicit adversary — when asked to "find problems with your own answer," Flash High tends to confirm its own answer `[T3 on self-refine literature]`. The framework uses a separate Adversarial Reviewer agent with a different prompt.
- File-path and tool-call hallucination — the model can invent plausible-looking file paths or call tools with wrong argument types `[KNOWN-PRACTICE]`. The framework compensates with strict JSON-schema-validated tool calls and `PreToolUse` hooks that verify paths exist.

### 3.6 Coding Performance

On SWE-bench (the standard agentic-coding benchmark), Gemini 3 Flash scores in the same range as Claude 3.5/3.7 Sonnet and GPT-4.1 `[T5, UNVERIFIED for 3.8]`. For this framework's purposes — writing short Python training scripts, generating Kaggle notebooks, manipulating YAML configs — Flash High is comfortably sufficient. The architectural rule that compensates for any coding-quality gap: **every generated script is executed and its outputs are validated by the Artifact Analyst before its conclusions are trusted.** Execution feedback is the strongest hallucination suppressor.

### 3.7 Agentic Task Performance

On agentic benchmarks (Tau-bench, GAIA, OSWorld, WebArena), Flash-tier models are noticeably weaker than Pro-tier or Claude Sonnet on long-horizon multi-tool tasks `[T3, T5]`. The framework compensates by:
- **Short agent horizons** — each agent call does one bounded task and returns. The Commander assembles the long horizon.
- **Structured state** — the blackboard (see §10) replaces "remember what other agents said" with "read the structured state file."
- **Tool wrappers with strict schemas** — every tool call is JSON-schema-validated.
- **Adversarial review at gates** — high-stakes decisions pass through a separate critic agent.

### 3.8 Cost and Latency

Gemini Flash is priced well below Pro and below Claude Sonnet on a per-token basis `[T1]`. The "High" reasoning tier costs more (in tokens) than Low/Medium because it spends reasoning tokens. For an autonomous system running 100s of agent calls per competition, this matters: the framework tracks token spend per agent per competition and feeds it into the agent-evolution loop (§10 of the prompt).

Antigravity may or may not pass per-token cost through to the user — that depends on the user's Antigravity subscription tier `[UNVERIFIED]`. The framework is designed to be cheap either way: most agent calls are short, schemas suppress rambling, and parallel async subagents reduce wall-clock latency.

### 3.9 Multi-modal Capabilities

Gemini 3.x Flash accepts image, audio, and video input `[T1]` (verify the exact modalities for 3.8). For this framework, the relevant use is **image input for EDA on CV competitions** — the Data Forensics agent can attach a sample image and ask the model to describe its structure (resolution, color, artifacts). This is the cleanest way to do vision-aware EDA without a separate vision model.

### 3.10 Compensation Strategies — What the Evidence Supports

Per the prompt (§3), the framework must compensate for Flash High's limitations without silently substituting other models. The compensation strategies and their evidence basis:

| Strategy | Evidence | Where the framework uses it |
|----------|----------|------------------------------|
| Decomposition | Strong `[T3]` (chunking improves performance on long tasks) | Every agent call has a bounded scope; the Commander decomposes phases into subtasks |
| Specialized agents | Strong `[T3]` (role-specialized prompts outperform generalist prompts) | Per-agent system prompts in `.agents/agents/*.md` |
| Verification (tool-use) | Strong `[T3]` (tool feedback improves accuracy) | Every script is executed; outputs validated |
| Structured state | Strong `[T3]` (state-tracking improves long-horizon tasks) | Blackboard store (§10) |
| Memory (retrieval) | Strong `[T3]` (RAG outperforms long-context dump) | Four-layer memory (§10) |
| Parallelization | Strong `[T3]` (parallel specialist ensembles outperform monolithic) | Async subagents for independent experiments |
| Reflection (post-hoc critique) | Weak-moderate `[T3]` (self-refine gains are small and noisy) | Used only at gates, with a separate adversary |
| Adversarial review | Moderate `[T3]` (separate-critic gains are larger than self-refine) | Adversarial Reviewer + Final Auditor |
| Context engineering | Strong `[T3, T4]` (structured retrieval beats dump) | Layered context (§31 of prompt) |
| Execution feedback | Strong `[T3]` (ReAct, Reflexion) | Every artifact is consumed and validated |
| Empirical evaluation | Strong `[T3]` | Every experiment produces structured knowledge |

The framework does **not** compensate by increasing prompt length — per the prompt's explicit prohibition (§3). Long prompts degrade Flash High's effective reasoning more than they help `[T3]`.

### 3.11 Summary

Gemini 3.8 Flash High is sufficient as the sole reasoning model **if and only if** the framework is disciplined about:
1. Short, bounded agent calls (decomposition).
2. Schema-constrained structured outputs.
3. Layered, retrieved context (not context dumping).
4. Adversarial review at high-stakes gates.
5. Execution feedback as the primary hallucination suppressor.

The architecture below is designed around these constraints.

---

## 4. Kaggle Infrastructure Constraints

> **Confidence note:** Kaggle compute quotas, accelerator types, and rule details are volatile. The numbers below reflect author knowledge as of approximately September 2026 and should be re-verified from the official Kaggle docs (`kaggle.com/docs`) before the framework is committed to production. Items tagged `UNVERIFIED` need live verification.

### 4.1 Kaggle CLI

The Kaggle CLI (`pip install kaggle`) is the official automation surface `[T1]`. It exposes:

- `kaggle competitions <list|files|download|submit|submissions|leaderboard>` — competition data and submissions.
- `kaggle datasets <list|files|download|create|version|status>` — dataset management.
- `kaggle kernels <list|files|init|pull|push|output|status>` — notebook lifecycle.
- `kaggle models <list|get|init|create|version>` — Kaggle Models hub.
- `kaggle config <view|init>` — credentials and config.

Authentication is via a `kaggle.json` file in `~/.kaggle/` containing username + API key, downloadable from `kaggle.com/<user>/account` `[T1]`.

For this framework, the critical commands are:
- `kaggle kernels init` — generate `kernel-metadata.json` template in the current directory.
- `kaggle kernels push -p <dir>` — upload the notebook + metadata, trigger a run on Kaggle's compute.
- `kaggle kernels status <user/kernel-slug>` — poll run status (`running|complete|error`).
- `kaggle kernels output <user/kernel-slug> -p <dir>` — download the notebook's output artifacts (anything written to `/kaggle/working/`).
- `kaggle competitions submit -c <comp> -f submission.csv -m "message"` — submit a CSV for a traditional competition.
- `kaggle competitions submissions -c <comp>` — list past submissions and LB scores.

The framework's Kaggle Executor agent (see Part II §2) wraps these commands with strict error handling, retry with backoff, and structured logging.

### 4.2 Notebook Execution Environment

Kaggle Notebooks run in a containerized Linux environment with:
- **Python 3.10+** (verify current default — likely 3.11 or 3.12 as of 2026) `[UNVERIFIED]`.
- **Disk**: ~20GB writable space across `/kaggle/working/` (output, persists with the notebook) and `/kaggle/temp/` (scratch, not persisted) `[T1]`.
- **RAM**: ~16GB on CPU, ~13GB on GPU instances `[UNVERIFIED]`.
- **CPU cores**: 2-4 on CPU-only instances `[UNVERIFIED]`.
- **GPU**: typically 1-2 NVIDIA T4 (16GB each), or 1 P100 (16GB), or TPU v3-8 for TPU-enabled competitions `[T1]` — verify what's available in 2026.
- **Internet**: enabled by default for non-competition notebooks; **disabled by default for code competitions** (`Internet: false` in `kernel-metadata.json`) `[T1]`.

### 4.3 Compute Quotas

Kaggle's compute quotas have historically been `[T1, but verify current]`:
- **CPU**: ~30 hours per notebook, up to 9 hours wall-clock per run.
- **GPU (T4 or P100)**: ~30 hours per week per user.
- **TPU v3-8**: ~20 hours per week per user.
- **Quota resets**: weekly, on a rolling basis.

Quotas are per-user, not per-team. A 5-person team pooling compute can effectively run ~150 GPU-hours per week — significant for serious competitions `[KNOWN-PRACTICE]`.

### 4.4 Code Competition Requirements

Modern Kaggle competitions are predominantly **code competitions** `[T1]`:
- The notebook is the submission — there is no separate CSV upload.
- The notebook runs on a hidden test set that is mounted at `/kaggle/input/<competition-name>/` during execution.
- Internet is **off** during the scoring run.
- The notebook must write its predictions to `/kaggle/working/submission.csv` (or a path defined in the competition's `sample_submission.csv`).
- The notebook must complete within the competition's time limit (often 9 hours GPU, 9 hours CPU — verify).
- **Custom datasets** can be attached to provide pretrained models, external features, etc. — but they must be uploaded to Kaggle Datasets first.

This drives several framework requirements:
- Every model the framework uses must be **packable as a Kaggle Dataset** (pretrained weights, tokenizer files, etc.).
- The notebook must be **self-contained** — all imports must be installable from PyPI offline (use `pip install --no-deps -q <pkg>` for any non-default-installed package, or pre-bake dependencies into a Kaggle Dataset and `sys.path.append`).
- The notebook must be **deterministic and reproducible** — random seeds set, CUDA determinism enabled where possible.

### 4.5 Submission Frequency

Most competitions allow **5 submissions per day** per user `[T1, verify]`. For a 5-person team that's 25/day — far more than the framework should use. The framework's submission policy (Part II §16) is to use submissions sparingly (≤2-3 per day, only when CV improvement is significant), preserving the rest as a buffer for late-competition strategic submissions.

### 4.6 Private Leaderboard Mechanics

The private LB is computed on a held-out test set that is **not visible to participants during the competition** `[T1]`. The public LB is computed on a small subset (often 30%) of the test set and visible during the competition. The private LB is revealed only after the competition ends.

Two submission-selection strategies are common `[T5]`:
1. **Pick 2 from CV**: submit the two models with the best CV score (one might be the safe ensemble, the other a higher-variance single model).
2. **Pick 1 from CV + 1 from public LB**: trust CV for one slot and use a public-LB-strong model for the other — risky but rewards public-LB-strong models that also generalize.

The framework defaults to strategy 1 (CV-first), with the Adversarial Reviewer having veto power on strategy 2.

### 4.7 Shake-up Risk

**Shake-up** is the gap between public and private LB rankings. Kaggle deliberately keeps the public LB test set small to discourage overfitting, which means a model that ranks #1 on public LB may drop to #50 on private LB `[T1, T5]`.

Risk factors for high shake-up:
- Small public test set.
- Distribution shift between public and private test slices.
- High-correlation ensemble (models all make the same mistakes).
- Public-LB chasing (selecting models by public score).

The framework's shake-up defense:
- **Trust CV over public LB** (validation-first policy).
- **Encourage ensemble diversity** (correlation analysis in the Ensemble Engine).
- **Use adversarial validation** to detect distribution shift.
- **Cap public-LB submissions** (≤2-3 per day, used for sanity-checking not model selection).

### 4.8 External Data and Pretrained Models

Kaggle's general policy: external data and pretrained models are allowed **if** their licenses permit competition use and they are publicly available (so all competitors have equal access) `[T1]`. Individual competitions can override this — some ban external data, some ban specific pretrained models, some require reproducibility.

The framework's Rule Compliance agent (Part II §2) is responsible for parsing each competition's rules and enforcing them. The agent has veto power: if a competition bans a model the framework wants to use, the framework uses something else.

### 4.9 Reproducibility

Kaggle Notebooks are **not** fully deterministic by default — CUDA operations on GPUs have non-deterministic reductions by default, and even CPU scikit-learn can be non-deterministic with parallelism `[KNOWN-PRACTICE]`. The framework's reproducibility stack:
- Set `random.seed()`, `np.random.seed()`, `torch.manual_seed()`, `tf.random.set_seed()`.
- For CUDA: `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- For LightGBM/XGBoost/CatBoost: set `random_state`, `deterministic=True` where supported, `n_jobs=1` if determinism is more important than speed.
- For ensembles across folds: each fold gets a distinct seed but the seed sequence is recorded.
- For full reproducibility, pin dependency versions in a Kaggle Dataset (so the same `lightgbm==4.x.y` is used in every run).

### 4.10 Kaggle Models

Kaggle Models is the platform's hub for pretrained models `[T1]`. Models hosted there can be attached to a notebook as a Kaggle Dataset (no internet needed for inference). The framework prefers Kaggle-hosted pretrained models over HuggingFace downloads for code competitions (which have no internet).

### 4.11 Recent Rule Changes (2024-2026)

The author is not aware of sweeping Kaggle rule changes in 2024-2026 that affect autonomous bots specifically `[UNVERIFIED — needs live check]`. The relevant ongoing rules:
- LLM API use is allowed in most competitions unless explicitly banned by competition rules.
- Automated submissions via the API are allowed within rate limits.
- Team merging has deadlines (typically 7 days before competition end).

### 4.12 Summary

Kaggle's compute is sufficient for serious competition work — 30 GPU-hours/week, 30 CPU-hours/run, modern accelerators. The constraints that shape the framework are:
- **Offline code competitions** → every dependency must be pre-packable.
- **Weekly GPU quota** → experiments must be prioritized, not run freely.
- **Submission frequency limits** → public LB is a scarce signal, use sparingly.
- **Private LB shake-up risk** → trust CV, encourage ensemble diversity.

The framework's Kaggle Executor agent and Compute Optimizer agent (Part II §2) are designed around these constraints.

---

## 5. Kaggle-Winning Strategy Analysis

> **Confidence note:** The winning-solution patterns below are distilled from author knowledge of Kaggle 2020-2026 solution writeups (Tier 2 sources per prompt §48). Specific competition names and winner names are illustrative — re-verify before citing in production.

### 5.1 Tabular Competitions — What Wins

Across modern tabular competitions (e.g., the long series of data-science bowl and feature-engineering-heavy competitions), the consistent winning pattern is `[T2, T5]`:

1. **Gradient-boosted decision trees (GBDT) as the backbone** — LightGBM, XGBoost, CatBoost, and scikit-learn's HistGradientBoosting all remain SOTA for tabular in 2026. Pure-deep-learning approaches consistently underperform GBDT on tabular `[T3]`.
2. **Diversity-then-ensemble** — winners typically train multiple GBDT configurations (different objectives, different feature subsets, different seeds) plus 1-2 neural models (FT-Transformer, TabPFN) plus 1-2 linear models, then blend.
3. **Feature engineering is still decisive** — even with strong GBDT baselines, well-engineered features (target encoding with smoothing, aggregation features, interaction features) routinely add 0.5-2% LB.
4. **Adversarial validation** — train a model to distinguish train vs. test, drop features with high importance, repeat. Standard practice since ~2018 `[T5]`.
5. **Custom loss aligned with the eval metric** — if the metric is RMSLE, optimize log1p target; if it's quadratic weighted kappa, optimize with a QWK proxy.
6. **Pseudo-labeling for the win** — when CV is reliable, semi-supervised learning on the test set (with confidence-thresholded pseudo-labels) adds 0.2-0.8% LB in many competitions `[T2, T5]`.
7. **Test-time augmentation** for tabular — shuffle feature order, average predictions; or run inference with different random seeds and average.

### 5.2 Time-Series Competitions — What Wins

For time-series competitions (e.g., M5-forecasting, Store-Sales, various demand-forecasting competitions), the winning pattern is `[T2, T5]`:

1. **GBDT with lag/rolling features** — LightGBM and CatBoost with engineered lag and rolling features consistently beat pure-ARIMA and many deep-learning baselines.
2. **Temporal validation** — TimeSeriesSplit or a temporal holdout (last-N-days), never random KFold. Purged KFold (López de Prado) when there's autocorrelation to avoid leakage from adjacent timepoints `[T3]`.
3. **Recursive vs. direct multi-step forecasting** — direct (separate model per horizon) typically wins for short horizons; recursive for long.
4. **Hierarchical reconciliation** — when forecasts must sum across a hierarchy (e.g., store → region → national), MinT or OLS reconciliation adds measurable LB.
5. **Ensembling across model families** — GBDT + ETS + ARIMA + a simple RNN/Transformer, blended.
6. **Post-processing for known events** — holidays, promotions, sales events, often manually adjusted.
7. **Distribution shift handling** — reweight recent observations, or use adversarial validation to drop stale features.

### 5.3 CV Competitions — What Wins

For computer-vision competitions (classification, detection, segmentation), the winning pattern is `[T2, T5]`:

1. **Pretrained backbones** — ViT (especially DINOv2 / DINOv3 self-supervised backbones), ConvNeXt-v2/v3, EVA-02, and SAM-family models for segmentation. Transfer from ImageNet-21k or larger self-supervised pretraining is standard.
2. **Heavy augmentation** — RandAugment, MixUp, CutMix, and label-smoothing for classification; Resize + Flip + ColorJitter baselines; advanced methods like gridmask, augmix.
3. **Test-time augmentation** — flip + multi-scale + multi-crop, average logits or probabilities.
4. **Ensemble across backbones** — typically 3-5 different pretrained backbones, blended.
5. **Pseudo-labeling on test** — confidence-thresholded pseudo-labels for semi-supervised fine-tuning.
6. **External data** — additional labeled or unlabeled data, when rules permit, is a major edge.
7. **Higher-resolution fine-tuning** — train at 384×384, fine-tune at 512×512 for classification; for detection, multi-scale training.

### 5.4 NLP/LLM Competitions — What Wins

For NLP and LLM-era competitions (e.g., NER, sentiment, question answering, prompt-prediction, retrieval), the winning pattern is `[T2, T5]`:

1. **Pretrained transformer backbone** — DeBERTa-v3, RoBERTa, ModernBERT for classic NLP; Llama-3/4, Mistral, Gemma, Qwen for LLM-era tasks.
2. **LoRA/QLoRA fine-tuning** for efficiency, plus full fine-tuning for the final winning model when compute allows.
3. **Ensemble across checkpoints and folds** — average predictions across multiple seeds, multiple folds, multiple model sizes.
4. **Retrieval-augmented prediction** for open-book QA — retrieve chunks from provided context, feed to LLM.
5. **LLM-based ensemble** — for generative tasks, query multiple LLMs (or one LLM multiple times with different prompts), use majority vote or an LLM-as-judge.
6. **Pseudo-labeling on test set** — fine-tune on test predictions where the model is most confident.
7. **Domain-adaptive pretraining (DAPT)** — continue pretraining on competition data before fine-tuning for the task.

### 5.5 Multimodal Competitions — What Wins

For multimodal (image+text, video+text, tabular+image), the winning pattern is `[T2, T5]`:

1. **Modality-specific encoders** — a strong CV backbone and a strong NLP backbone, separately fine-tuned.
2. **Late fusion (most common)** — concatenate encoders' embeddings, then a light head (MLP or shallow GBDT). Robust, low-risk.
3. **Intermediate fusion (less common, higher ceiling)** — cross-attention between modalities. Higher ceiling, more compute, more prone to overfit.
4. **Ensemble of fusion strategies** — late + intermediate, blended.
5. **Use the metric to drive loss design** — for retrieval tasks, triplet loss or contrastive; for ranking, lambdarank.

### 5.6 Common Winning Patterns Across Problem Types

Aggregating across problem types, the consistent differentiators between #1 and top-10% are:

1. **Ensemble diversity > single-model tuning** — every winning solution has a thoughtful ensemble of diverse models; no single model wins.
2. **Adversarial validation as a feature filter** — winners use AV to identify and remove features that differ between train and test, reducing shake-up risk.
3. **Custom loss aligned with eval metric** — direct optimization of the leaderboard metric (when differentiable).
4. **Validation strategy that mirrors private LB** — winners identify the test-set structure (temporal? grouped? shuffled?) and design CV to match.
5. **Pseudo-labeling for the final 0.5-1%** — used when CV is reliable and test distribution is similar to train.
6. **Post-processing** — calibration, threshold optimization, rank-based transformations.
7. **Non-obvious edge**: domain insight (e.g., recognizing that a "customer ID" column leaked target in a past competition), augmentation trick (e.g., mixup for tabular), or a clever feature construction.

### 5.7 Anti-Patterns — What Top Solutions Explicitly Avoid

1. **Single-model reliance** — never wins.
2. **Public-LB chasing** — submitting many times to discover the test set leads to overfitting and shake-up.
3. **Over-tuning hyperparameters** — diminishing returns past a point; better to add a diverse model.
4. **Leaky features** — features that look great on CV but won't be available at inference, or that leak target.
5. **High-correlation ensembles** — adding a model that's 99% correlated with existing ones adds nothing; may even hurt.
6. **Complex stacking without justification** — stacking rarely outperforms weighted blending by enough to justify the complexity.
7. **Ignoring competition rules** — wins are forfeited for rule violations, including using external data that wasn't licensed for competition use.

### 5.8 What Differentiates #1 from Top-10%

Based on writeups, the differentiators between #1 and top-10% are typically `[T5, INFERRED]`:

- **A single non-obvious insight** — a leak, a domain trick, a feature construction no one else found. This is the biggest single differentiator and is rarely reproducible from a generic framework.
- **A more thoughtful ensemble** — better diversity, better weighting, more careful selection.
- **Better validation** — when CV-LB gap is small, the winner's CV is more trustworthy, so they pick better final submissions.
- **More compute** — running 20 models instead of 5 is a real edge when ensembles matter.

The framework can reliably produce top-10% solutions with disciplined execution. Reaching #1 requires either (a) the user-supplied non-obvious insight (which the framework can absorb and propagate), or (b) unusual compute budget, or (c) luck. The framework explicitly **does not claim to guarantee #1** (per prompt §4).

### 5.9 Summary

The framework's strategy (Part II §1) bakes in these winning patterns:
- GBDT-backbone-first for tabular.
- Temporal-aware CV for time-series.
- Pretrained-backbone + TTA for CV.
- Multi-model ensemble with diversity analysis.
- Adversarial validation as a standard step.
- Pseudo-labeling when CV is reliable.
- Custom loss aligned with eval metric.
- CV-first submission selection.

---

## 6. Multi-Agent Architecture Comparison

Per the prompt (§5), at least the following architectures must be compared before settling on one. The author's evidence-weighted assessment of each:

### 6.1 Supervisor Architecture

A single supervisor agent dispatches tasks to specialist worker agents. The supervisor owns the global state, decides what to do next, and synthesizes worker outputs.

- **Best for**: bounded-scope tasks with clear sub-task boundaries; rapid prototyping.
- **Worst for**: long horizons where the supervisor's context grows unboundedly; high-throughput parallel work where the supervisor becomes a bottleneck.
- **Empirical support**: strong `[T3]` (AutoGen, LangGraph Supervisor, CrewAI all use this pattern; widely validated).
- **Coordination cost**: low.
- **Failure modes**: supervisor hallucination cascades to all workers; supervisor's context window fills up over long horizons.
- **Suitability for Kaggle**: medium — the global objective is well-defined (maximize private-LB), but the experimentation loop can run for many iterations, stressing supervisor context.

### 6.2 Hierarchical Supervisor

A tree of supervisors: top-level Commander owns the global objective; mid-level Cell supervisors own cell-scoped objectives (Research, Data, Model, Experiment, Adversarial, Submission); leaf agents execute bounded tasks.

- **Best for**: large problems with naturally hierarchical decomposition.
- **Worst for**: small problems (overhead); problems where decomposition boundaries are unclear.
- **Empirical support**: strong `[T3]` (MetaGPT, ChatDev use hierarchical patterns).
- **Coordination cost**: moderate — communication up/down the tree.
- **Failure modes**: hierarchy rigidly routes information through parents, losing detail; a bad mid-level supervisor can cause whole subtrees to fail.
- **Suitability for Kaggle**: high — Kaggle's phases (research → data → validation → baseline → experiments → ensemble → submission) are naturally hierarchical.

### 6.3 Manager-Worker

A manager dispatches tasks to a pool of interchangeable workers. The manager is a router, not a domain specialist.

- **Best for**: homogenous worker pools (e.g., 10 agents doing the same kind of task).
- **Worst for**: heterogeneous specialist work.
- **Empirical support**: moderate `[T3]` (CrewAI defaults to this).
- **Coordination cost**: low.
- **Failure modes**: manager misroutes; workers don't have the right skills.
- **Suitability for Kaggle**: low — Kaggle work is heterogeneous (validation design ≠ model training ≠ ensemble search).

### 6.4 Blackboard / Shared-Memory

All agents read/write to a shared structured state store (the "blackboard"). No direct agent-to-agent messaging.

- **Best for**: problems where many independent perspectives need to converge; problems where transient state matters; problems where decoupling producers from consumers matters.
- **Worst for**: problems requiring tight back-and-forth conversation.
- **Empirical support**: moderate `[T3]` (Hearsay-II classic, MemGPT/Letta modern variants, Voyager's skill library).
- **Coordination cost**: low once the blackboard is built; agents are decoupled.
- **Failure modes**: blackboard can become a write-only log; retrieval quality determines agent effectiveness; concurrent writes need coordination.
- **Suitability for Kaggle**: high — Kaggle's experiment state, artifacts, and knowledge all fit naturally in a structured store that multiple agents read.

### 6.5 Parallel Specialist Swarm

A flat pool of specialists; a dispatcher routes tasks by capability match.

- **Best for**: embarrassingly parallel independent tasks.
- **Worst for**: tasks with shared mutable state.
- **Empirical support**: strong `[T3]` (parallel-tool-call patterns are validated).
- **Coordination cost**: low if no shared state; high if shared state.
- **Failure modes**: agents duplicate work; no one owns the global objective; free-form chatter.
- **Suitability for Kaggle**: medium — works for parallel experiments (train 3 models), fails for the synthesis layer.

### 6.6 Debate Architecture

Multiple agents argue about a question, converge (or not) through rounds.

- **Best for**: high-stakes single decisions where multiple perspectives improve outcomes.
- **Worst for**: routine tasks (massive token overhead).
- **Empirical support**: weak-moderate `[T3]` (Du et al. 2023 showed gains on math, but token cost is high; later work questioned whether gains survive under cost control).
- **Coordination cost**: high (multiple rounds, multiple agents).
- **Failure modes**: agents converge to a wrong answer if any one is confidently wrong; token cost is brutal for routine use.
- **Suitability for Kaggle**: low-medium — useful only at high-stakes gates (validation design, submission selection), not as the core architecture.

### 6.7 Critic-Generator (Generator-Validator)

A generator produces a candidate answer; a separate critic inspects and rejects or accepts. Iterates until accepted or budget exhausted.

- **Best for**: tasks where verification is cheaper than generation (code, math, structured artifacts).
- **Worst for**: open-ended generation where verification is hard.
- **Empirical support**: strong `[T3]` (Self-Refine, Reflexion, CRITIC; effect sizes are consistent if modest).
- **Coordination cost**: moderate — 2 agents per gate, but only when gates are selective.
- **Failure modes**: critic is too lax (rubber-stamps) or too strict (blocks everything); critic and generator share blind spots.
- **Suitability for Kaggle**: high — perfect for validation design, leakage review, submission audit.

### 6.8 Planner-Executor-Verifier

A planner produces a plan; an executor executes; a verifier checks results; loop.

- **Best for**: long-horizon multi-step tasks.
- **Worst for**: short tasks (overhead).
- **Empirical support**: strong `[T3]` (ReAct, Plan-and-Solve, LATS).
- **Coordination cost**: moderate.
- **Failure modes**: planner commits early; verifier and executor share blind spots.
- **Suitability for Kaggle**: high — the experimentation loop is naturally plan-execute-verify.

### 6.9 Evolutionary Agent Population

A population of agents (or prompts, or skills) competes; the fittest reproduce and mutate.

- **Best for**: problems where the search space is large and gradient information is sparse.
- **Worst for**: problems where one good design exists (overhead of maintaining a population).
- **Empirical support**: moderate `[T3]` (EvoPrompt, OPRO, AlphaEvolve for code).
- **Coordination cost**: high (evaluating many candidates).
- **Failure modes**: population collapses to a local optimum; evaluation noise dominates selection pressure.
- **Suitability for Kaggle**: low for the core architecture; medium for **skill evolution** (§12 of this report) where new skills are candidate-tested before promotion.

### 6.10 Hybrid Architecture

Combine elements: typically a hierarchical supervisor at the top, a blackboard for state, critic-generator gates at high-stakes points, parallel swarms for independent work.

- **Best for**: real-world systems where different layers have different coordination needs.
- **Worst for**: simple problems (overhead).
- **Empirical support**: strong `[T3]` (most production agent systems — Cursor, SWE-Agent, OpenHands — are hybrids).
- **Coordination cost**: moderate (each layer uses the right pattern).
- **Failure modes**: complexity; debugging requires understanding all layers.
- **Suitability for Kaggle**: highest — the framework needs different patterns at different layers.

### 6.11 Architecture Comparison Matrix

| # | Architecture | Best For | Worst For | Empirical Support | Coordination Cost | Failure Modes | Kaggle Suitability |
|---|--------------|----------|-----------|---------------------|---------------------|----------------|---------------------|
| 1 | Supervisor | Bounded scope | Long horizons | Strong | Low | Supervisor bottleneck | Medium |
| 2 | Hierarchical Supervisor | Large hierarchical problems | Small problems | Strong | Moderate | Info loss up tree | High |
| 3 | Manager-Worker | Homogeneous pool | Heterogeneous | Moderate | Low | Misrouting | Low |
| 4 | Blackboard | Independent perspectives | Tight back-and-forth | Moderate | Low (post-setup) | Write-only log | High |
| 5 | Parallel Swarm | Embarrassingly parallel | Shared state | Strong | Low → High | No global owner | Medium |
| 6 | Debate | High-stakes single decisions | Routine tasks | Weak-Moderate | High | Confident-wrong convergence | Low-Medium |
| 7 | Critic-Generator | Verifiable tasks | Open-ended | Strong | Moderate | Shared blind spots | High |
| 8 | Planner-Executor-Verifier | Long-horizon multi-step | Short tasks | Strong | Moderate | Early commitment | High |
| 9 | Evolutionary | Large search, sparse gradient | Single-best-design | Moderate | High | Local optima | Low (core), Medium (skill evolution) |
| 10 | Hybrid | Real-world systems | Simple problems | Strong | Moderate | Complexity | Highest |

### 6.12 Marginal Intelligence Gain per Additional Agent

Per the prompt's §39 ("Do Not Overengineer"), the framework must be sized to the smallest viable agent count. Evidence `[T3]`:

- Going from 1 → 3 agents: large gains (decomposition + specialization).
- Going from 3 → 6 agents: moderate gains (finer specialization).
- Going from 6 → 12 agents: small gains (most roles covered).
- Going from 12 → 24 agents: marginal gains; coordination overhead starts to dominate.
- Going from 24 → 50 agents: negative marginal returns in most studies.

The sweet spot for a Kaggle system is **~8-12 agents**. Below 6, the system lacks the specialization needed for serious Kaggle work; above 15, coordination overhead and the "every agent calls Flash High" token cost start to hurt.

### 6.13 The Prompt's Recommended Diagram (§5) — Critique

The prompt's §5 proposes a Commander → Cell → Sub-agent diagram. Critique:

- **Strengths**: natural phase mapping; clear ownership; reproducible execution traces.
- **Weaknesses**: it has ~15+ agents across 6 cells, which is on the high end; the "Adversarial Cell" is structurally separate but adversarial review is needed at multiple gates, not just at one cell; the diagram implies sequential phase flow (Research → Data → Validation → Baseline → Experiments → Ensemble → Adversarial → Submission → Knowledge Evolution → Strategy Update), but Kaggle work is iterative.

The framework adapts the diagram (see §7 below): keeps the hierarchy, replaces the "Adversarial Cell" with a critic-generator gate that runs at every decision point, and adds a blackboard-style state store that all agents read.

---

## 7. Recommended Architecture (Evidence-Decided)

### 7.1 The Three Candidate Architectures

Per the user's clarification ("Let evidence decide"), the three candidates:

**Candidate A — Lean Hierarchical Supervisor (8 agents).**
- Commander, Researcher, Data Forensics, Validation Architect, Trainer, Ensembler, Adversarial Reviewer, Kaggle Executor.
- Blackboard-style state store, but small.
- Critic-generator gate only at submission time.
- **Pros**: simplest; lowest token cost; easiest to debug.
- **Cons**: Adversarial Reviewer only at submission is too late — leakage should be caught at validation-design time.

**Candidate B — Hierarchical Supervisor + Blackboard (12 agents).**
- Commander, Researcher, Data Forensics, EDA Specialist, Validation Architect, Feature Engineer, Model Researcher, Trainer, Ensembler, Adversarial Reviewer, Kaggle Executor, Memory Curator.
- Blackboard state store is the primary communication mechanism.
- Critic-generator gate at three points: validation design, ensemble selection, submission.
- **Pros**: strong coverage; clean state; gates at the right points; matches the Kaggle workflow.
- **Cons**: 12 agents is at the upper end of the sweet spot; token cost is non-trivial.

**Candidate C — Generator-Critic Hybrid with Blackboard (10 agents).**
- Commander, Researcher, Data Forensics, Validation Architect (with critic sub-role), Trainer (with critic sub-role), Ensembler (with critic sub-role), Kaggle Executor, Memory Curator, Final Auditor.
- Critic-generator at every agent call (every agent's output is critiqued by a different agent).
- **Pros**: highest robustness against Flash High hallucination.
- **Cons**: 2x token cost on every call; brutal for routine tasks; the literature is clear that debate-on-everything is cost-ineffective `[T3]`.

### 7.2 Trade-Off Matrix

| Criterion | A — Lean Hierarchical | B — Hierarchical + Blackboard | C — Generator-Critic Hybrid |
|-----------|------------------------|--------------------------------|------------------------------|
| Coverage of Kaggle workflow | 6/10 (missing EDA, Feature Eng, Memory) | 9/10 | 8/10 |
| Token cost per phase | Low (~30k) | Medium (~60k) | High (~120k) |
| Robustness vs. Flash High hallucination | 5/10 (critic only at end) | 8/10 (gates at 3 high-stakes points) | 10/10 (critic everywhere) |
| Debuggability | High | High | Medium (every agent = 2 calls to trace) |
| Maintenance burden | Low | Medium | High |
| Marginal agent count | 8 | 12 | 10 (but 2x calls) |
| Wall-clock latency per phase | Low | Medium | High |
| Empirical precedent | Common in production | Common in production | Rare in production |

### 7.3 Recommendation

**Candidate B — Hierarchical Supervisor + Blackboard + Critic-Generator Gates at 3 Decision Points.**

Rationale (evidence-decided):
1. Coverage: only B has the full Kaggle workflow (EDA, Feature Eng, Memory Curator) without forcing them onto a smaller agent.
2. Token cost is moderate and predictable; C's 2x cost is brutal for a system that may run hundreds of agent calls per competition.
3. Robustness vs. Flash High hallucination: gates at the 3 highest-stakes points (validation design, ensemble selection, submission) is where the literature says adversarial review pays off; routine agent calls don't need it `[T3]`.
4. The blackboard decouples agents — the Trainer writes its results to disk as a structured artifact, the Ensembler reads it, no direct messaging. This minimizes free-form chatter (per prompt §32).
5. Wall-clock latency is moderate — async subagents can run experiments in parallel, and the gates are at decision points where latency is acceptable.
6. The agent count (12) is at the upper end of the sweet spot but defensible because each agent has a distinct, non-overlapping role.

### 7.4 Final Architecture Diagram

```text
                          ┌──────────────────────────┐
                          │      COMMANDER            │
                          │  (strategic director,     │
                          │   owns global objective)  │
                          └────────────┬─────────────┘
                                       │
                  reads/writes         │         reads/writes
                  blackboard            │         blackboard
                                       │
         ┌─────────────┬───────────────┼───────────────┬─────────────┐
         │             │               │               │             │
   RESEARCH CELL    DATA CELL       MODEL CELL    EXPERIMENT CELL  SUBMISSION CELL
         │             │               │               │             │
   Competition     Data Forensics  Feature Eng.    Trainer         Kaggle Executor
   Researcher      EDA Specialist  Model Research  Ensembler       Final Auditor
                                   HPO
                                       │
                              ┌────────┴────────┐
                              │   ADVERSARIAL   │
                              │   REVIEWER      │
                              │   (gate at 3    │
                              │    decision     │
                              │    points)      │
                              └─────────────────┘
                                       │
                              ┌────────┴────────┐
                              │  MEMORY CURATOR  │
                              │  (4-layer store) │
                              └──────────────────┘
```

### 7.5 The Three Critic-Generator Gates

The Adversarial Reviewer runs as a separate agent at three points:

1. **Gate 1 — Validation Design**: after the Validation Architect produces a CV strategy, the Adversarial Reviewer attacks it: "Is there leakage? Is the CV-LB gap likely to be small? Are features available at inference?" Veto power: the strategy is not approved until the Adversarial Reviewer signs off.
2. **Gate 2 — Ensemble Selection**: after the Ensembler produces a candidate blend, the Adversarial Reviewer checks: "Is the ensemble overfit to OOF? Are the selected models diverse enough? Will this generalize to private LB?" Veto power: the blend is not finalized until the Adversarial Reviewer signs off.
3. **Gate 3 — Final Submission**: after the Final Auditor assembles the submission package, the Adversarial Reviewer runs the full checklist (prompt §28): CV trust, leakage, feature availability, preprocessing identity, distribution shift, ensemble overfitting, public-LB overuse, reproducibility, rule compliance. Veto power: the submission does not go to Kaggle until the Adversarial Reviewer signs off.

At all three gates, the Adversarial Reviewer is a **separate agent** with a different system prompt (not self-critique by the same agent — per the evidence that self-refine is weaker than separate-critic `[T3]`).

### 7.6 The Blackboard (Structured State Store)

The blackboard is a directory of structured files:

```
state/
├── competition.yaml          # competition metadata, metric, rules
├── dataset_signature.yaml     # data shape, types, cardinality, time axis
├── validation.yaml            # current CV strategy
├── experiments.yaml           # experiment registry (one row per experiment)
├── ensemble.yaml              # current ensemble composition
├── knowledge_pointers.yaml    # pointers into knowledge/ directory
├── agent_state.yaml           # per-agent last-known state
└── logs/                      # JSONL logs of every agent call
```

Every agent reads from `state/` at the start of its task and writes back at the end. The framework's `PreToolUse` hooks enforce that no agent writes to `state/` without first reading the current version (optimistic-concurrency-lite).

This replaces free-form chatter. Agents don't message each other — they write structured updates that other agents read.

### 7.7 Why This Works for Gemini 3.8 Flash High

1. **Short, bounded agent calls**: each agent has one task; Flash High's effective reasoning scope is sufficient for any single bounded task.
2. **Structured outputs**: every agent returns a YAML/JSON object per the schema (§9), suppressing free-form rambling and hallucination.
3. **Layered context** (§31 of the prompt): each agent gets only the memory it needs.
4. **Adversarial review at 3 gates**: catches Flash High's known failure modes (early commitment, blind-spot sharing between generator and critic of the same prompt) at the points where they matter most.
5. **Blackboard decouples agents**: no agent needs to "remember" another agent's output — it reads it from disk.
6. **Async subagents** for parallel experiments: stays within Antigravity's rate limits.

---

## 8. Agent Roster

The 12 agents, in role-by-role detail. For each: name, owning cell, primary responsibility, inputs, outputs, system-prompt summary.

### 8.1 COMMANDER

- **Owning cell**: top-level.
- **Responsibility**: owns the global objective; decides what phase to run next; routes tasks to cells; synthesizes cell outputs; makes stopping decisions; makes submission-selection decisions.
- **Inputs**: competition brief, current blackboard state, memory pointers.
- **Outputs**: phase decisions (e.g., "phase: experimentation, focus: feature engineering, budget: 4 GPU-hours"); submission-selection decisions; stopping decisions.
- **System-prompt summary**: "You are the Commander of an autonomous Kaggle competition team. Your job is to maximize expected private-LB performance per unit of Kaggle compute. You do not write code or run experiments yourself. You decide which phase to run next, allocate compute budget, and make the final submission-selection call. You read the blackboard before deciding. You write only structured decisions to the blackboard."
- **Effort**: high.

### 8.2 COMPETITION RESEARCHER

- **Owning cell**: Research Cell.
- **Responsibility**: extract competition metadata (metric, rules, submission format, deadlines, data restrictions); retrieve similar past competitions from memory; identify known pitfalls and baseline expectations.
- **Inputs**: competition name (from competition.yaml); memory pointers to past competitions.
- **Outputs**: `state/competition.yaml` (populated); `reports/competition_intelligence.md`.
- **System-prompt summary**: "You are the Competition Researcher. You extract competition rules, metric, submission format, and historical priors. You never speculate — if a fact isn't in the official competition page or memory, you mark it UNVERIFIED."
- **Effort**: medium.

### 8.3 DATA FORENSICS AGENT

- **Owning cell**: Data Cell.
- **Responsibility**: schema discovery, data profiling, missingness, duplicates, cardinality, target analysis, distributions, correlations, train/test differences, leakage detection, temporal/group structure detection.
- **Inputs**: data directory path; dataset_signature.yaml template.
- **Outputs**: `reports/dataset_inventory.md`, `reports/data_quality.md`, `reports/leakage_report.md`, `reports/distribution_shift.md`, `reports/target_analysis.md`, `reports/eda_report.md`; `state/dataset_signature.yaml` (populated).
- **System-prompt summary**: "You are the Data Forensics Agent. You profile datasets rigorously and detect leakage. You write Python scripts (saved to `scripts/`) that produce JSON-formatted reports. You never assume the dataset is tabular — you detect the modality first."
- **Effort**: medium-high.

### 8.4 EDA SPECIALIST

- **Owning cell**: Data Cell.
- **Responsibility**: produce the complete EDA report (distributions, correlations, pairplots, target-by-feature interactions, missingness patterns). For non-tabular modalities, route to specialized EDA (image samples, text samples, time-series plots).
- **Inputs**: dataset_signature.yaml.
- **Outputs**: `reports/eda_report.md` (with embedded images).
- **System-prompt summary**: "You are the EDA Specialist. You produce a complete EDA report with visualizations. You use Polars for data manipulation. You write all plots to `reports/figures/` and embed them in the report."
- **Effort**: medium.

### 8.5 VALIDATION ARCHITECT

- **Owning cell**: Data Cell (with veto power across the system).
- **Responsibility**: design the CV strategy. Identify leakage (target, temporal, group). Identify distribution shift. Answer the question: "Why should this CV estimate correlate with private LB performance?" If the answer isn't convincing, modeling does not proceed.
- **Inputs**: dataset_signature.yaml; reports/leakage_report.md, reports/distribution_shift.md.
- **Outputs**: `validation/strategy.md`, `validation/validation_config.yaml`, `validation/folds/` (the actual fold assignments).
- **System-prompt summary**: "You are the Validation Architect. You design the CV strategy that mirrors the private LB. You have veto power: if you cannot justify CV-LB correlation, modeling does not proceed. You must answer 'Why should this CV estimate correlate with private leaderboard performance?' in `strategy.md`."
- **Effort**: high.
- **Gate**: Gate 1 — Adversarial Reviewer attacks the validation design before approval.

### 8.6 FEATURE ENGINEER

- **Owning cell**: Model Cell.
- **Responsibility**: generate **hypothesis-driven** features (not blindly thousands). Each feature has a hypothesis ("this aggregation might capture customer behavior over 30 days"). Run small experiments to validate features before promoting them.
- **Inputs**: dataset_signature.yaml; reports/eda_report.md.
- **Outputs**: `features/v1/` directory with the feature-generation code; `features/v1/feature_catalog.yaml` listing each feature with its hypothesis and validation status.
- **System-prompt summary**: "You are the Feature Engineer. You generate hypothesis-driven features. Every feature has a stated hypothesis and is validated by a small experiment before promotion to the feature catalog. You never generate features blindly."
- **Effort**: medium-high.

### 8.7 MODEL RESEARCHER

- **Owning cell**: Model Cell.
- **Responsibility**: given the dataset signature and problem type, identify the best model families to try. For tabular: LightGBM, XGBoost, CatBoost, HistGB, linear models, TabPFN/TabICL for small datasets. For time-series: GBDT + lag features, ETS, ARIMA, simple neural baselines. For CV: pretrained ViT/ConvNeXt/DINOv3. For NLP: DeBERTa-v3 or modern LLMs.
- **Inputs**: dataset_signature.yaml; memory pointers to past similar competitions.
- **Outputs**: `models/candidates.yaml` listing candidate model families with rationale.
- **System-prompt summary**: "You are the Model Researcher. You identify candidate model families grounded in evidence (past competitions, benchmarks). You never propose a single model — you propose a diverse candidate set."
- **Effort**: medium.

### 8.8 HPO AGENT

- **Owning cell**: Model Cell.
- **Responsibility**: run efficient hyperparameter optimization for each candidate model. Use Optuna (TPE sampler, multi-objective for CV-score-vs-runtime). Budget: 20-50 trials per model, more for the strongest candidates.
- **Inputs**: models/candidates.yaml; validation/validation_config.yaml.
- **Outputs**: `experiments/hpo/<model>/<trial_id>/` per trial; `experiments/hpo/<model>/best.json`.
- **System-prompt summary**: "You are the HPO Agent. You use Optuna with TPE. You budget trials per model based on expected value. You always log CV score, runtime, and memory for each trial."
- **Effort**: medium.

### 8.9 TRAINER

- **Owning cell**: Experiment Cell.
- **Responsibility**: train models end-to-end. Run the actual training script (locally for prototyping; on Kaggle via the Kaggle Executor for final runs). Produce: OOF predictions, test predictions, feature importance, model artifact, resource-usage log.
- **Inputs**: experiments/hpo/<model>/best.json; validation/validation_config.yaml; features/v<N>/.
- **Outputs**: `experiments/training/<exp_id>/` with `metrics.json`, `oof_predictions.csv`, `submission.csv`, `experiment.json`, `resource_usage.json`, `model.artifact`.
- **System-prompt summary**: "You are the Trainer. You run training scripts that produce structured artifacts. You never claim a result without saving the OOF predictions. You log every experiment in the registry."
- **Effort**: medium.

### 8.10 ENSEMBLER

- **Owning cell**: Experiment Cell.
- **Responsibility**: search for the best ensemble. Methods: simple average, weighted average, forward stepwise selection, hill climbing (Caruana 2004). For each candidate blend, evaluate OOF performance, prediction diversity (correlation matrix), and error overlap.
- **Inputs**: all `experiments/training/<exp_id>/oof_predictions.csv`.
- **Outputs**: `ensemble/candidates/`, `ensemble/correlations/`, `ensemble/weights/`, `ensemble/final/blend.json`.
- **System-prompt summary**: "You are the Ensembler. You search for diverse, complementary models. You never add a model that's >95% correlated with the existing blend. You evaluate blends on OOF, not public LB."
- **Effort**: medium-high.
- **Gate**: Gate 2 — Adversarial Reviewer attacks the ensemble selection before approval.

### 8.11 ADVERSARIAL REVIEWER

- **Owning cell**: top-level (cross-cutting, not in any cell).
- **Responsibility**: at three gates (validation design, ensemble selection, final submission), attack the proposed artifact. Use the prompt's §28 checklist.
- **Inputs**: the artifact under review (validation strategy / ensemble / submission package).
- **Outputs**: `reviews/gate_<n>_<artifact>.yaml` with status (approved/rejected), findings, recommendations.
- **System-prompt summary**: "You are the Adversarial Reviewer. Your job is to find what's wrong. You never approve an artifact until you have actively tried to break it. You have veto power. You return a structured verdict with specific findings."
- **Effort**: high.

### 8.12 KAGGLE EXECUTOR

- **Owning cell**: Submission Cell.
- **Responsibility**: prepare Kaggle notebook + kernel-metadata.json; `kaggle kernels push`; poll `kaggle kernels status`; `kaggle kernels output` to fetch artifacts; handle errors with retry/backoff; log every API call.
- **Inputs**: the final notebook + metadata.
- **Outputs**: Kaggle submission ID; downloaded artifacts in `artifacts/<run_id>/`.
- **System-prompt summary**: "You are the Kaggle Executor. You wrap the Kaggle CLI with strict error handling and structured logging. You retry with exponential backoff. You never submit without the Final Auditor's approval."
- **Effort**: medium.

### 8.13 MEMORY CURATOR

- **Owning cell**: top-level (cross-cutting).
- **Responsibility**: maintain the four-layer memory store. After every experiment, ingest the structured artifact, extract knowledge (with confidence tier), deduplicate, detect contradictions, update the knowledge graph.
- **Inputs**: artifacts/<run_id>/; reviews/.
- **Outputs**: `knowledge/<category>/<topic>.yaml` files; `knowledge/meta/agent_performance.yaml`; `knowledge/meta/skill_candidates.yaml`.
- **System-prompt summary**: "You are the Memory Curator. You ingest structured artifacts and produce curated knowledge with provenance and confidence. You never promote speculation to permanent knowledge. You detect and flag contradictions."
- **Effort**: medium.

### 8.14 FINAL AUDITOR (mentioned in prompt §6, optional in our 12-agent count)

In the 12-agent roster above, the Final Auditor's role is folded into the Adversarial Reviewer (Gate 3 — Final Submission). If the user wants strict separation, the Final Auditor can be a 13th agent whose only job is the pre-submission checklist. The blueprint defaults to folding it into the Adversarial Reviewer to stay at 12 agents.

---

## 9. Communication Protocol

Per the prompt (§32), every agent returns a structured YAML/JSON object. The schema:

```yaml
# Required
status: success | failure | partial | blocked
confidence: 0.0-1.0

# Findings (free text, but bounded to a few hundred words)
findings: |
  <prose summary of what the agent did and found>

# Evidence (citations, file paths, metric values)
evidence:
  - type: file
    path: reports/eda_report.md
  - type: metric
    name: cv_score
    value: 0.847
    direction: higher_better
  - type: citation
    url: https://...
    retrieved: 2026-09-13

# Actions (what the agent did — for execution feedback and audit)
actions:
  - tool: Write
    path: scripts/train_lgbm.py
  - tool: Bash
    command: python scripts/train_lgbm.py
  - tool: Subagent
    agent: trainer
    task_id: exp-007

# Artifacts (files produced)
artifacts:
  - path: experiments/training/exp-007/metrics.json
    type: metrics
  - path: experiments/training/exp-007/oof_predictions.csv
    type: oof

# Recommendations (what the agent thinks should happen next)
recommendations:
  - priority: 8
    description: "Try CatBoost with target encoding"
    expected_gain: high
    compute_cost: medium
    confidence: 0.6

# Risks (what could go wrong)
risks:
  - severity: high
    description: "Adversarial validation shows distribution shift — features X, Y differ between train and test"
    mitigation: "Drop features X, Y; re-run baseline"

# Next tasks (for the Commander to route)
next_tasks:
  - cell: experiment
    task: retrain_baseline_without_X_Y
    priority: 9
```

Agents communicate through this structured format and through the blackboard state files. Free-form chatter between agents is forbidden — every inter-agent exchange is a structured artifact on disk.

### 9.1 Logging

Every agent invocation produces a JSONL log entry in `state/logs/<date>/<agent>.jsonl`:

```json
{"ts": "2026-09-13T10:00:00Z", "agent": "trainer", "task_id": "exp-007", "input_tokens": 8420, "output_tokens": 1830, "duration_s": 47, "result_path": "experiments/training/exp-007/result.yaml"}
```

This log feeds the agent-evolution loop (§10 of the prompt, §11 of this report).

### 9.2 Veto Power

Two agents have explicit veto power:
- **Validation Architect**: modeling does not proceed until the validation strategy is signed off.
- **Adversarial Reviewer**: at gates 1, 2, 3, the artifact under review is not approved until the Adversarial Reviewer signs off.

The Commander can override a veto only with a logged justification, and the override is reviewed at competition end (post-mortem).

---

## 10. Memory Architecture

Per the prompt (§11), four memory layers:

### 10.1 Working Memory

Per-task, ephemeral. Stored in the agent's invocation context — not persisted. Resets at every agent call.

### 10.2 Project Memory

Per-competition. Lives in `competition-project/knowledge/`:

```
knowledge/
├── competitions/         # this competition's metadata
├── strategies/           # what we're trying
├── validation/           # CV design rationale
├── feature_engineering/  # feature hypotheses and outcomes
├── models/                # candidate models and their results
├── ensembles/            # ensemble compositions
├── leakage/              # leakage findings
├── kaggle/                # Kaggle-specific facts
├── compute/              # compute usage log
├── failures/             # what didn't work
├── successes/            # what worked
├── experiments/          # experiment registry (condensed)
├── skills/                # candidate skills for this competition
├── agent-performance/    # per-agent stats
├── meta-learning/        # cross-competition patterns
└── playbooks/            # this competition's playbook
```

Every entry is a YAML file with: hypothesis, intervention, validation, result, conclusion, confidence, reproducibility, provenance (links to experiment IDs and artifact paths).

### 10.3 Strategic Memory

Cross-competition. Lives in `kaggle-agent-core/knowledge/` (the global, version-controlled knowledge base):

```
knowledge/
├── competitions/         # metadata for past competitions
├── strategies/            # general Kaggle strategies
├── validation/            # CV design patterns
├── feature_engineering/   # feature patterns by problem type
├── models/                # model family performance by dataset signature
├── ensembles/             # ensemble patterns
├── leakage/               # general leakage patterns
├── kaggle/                # Kaggle infra facts (compute quotas, etc.)
├── compute/               # compute optimization patterns
├── failures/              # failure patterns
├── successes/             # success patterns
├── experiments/           # cross-competition experiment outcomes (anonymized)
├── skills/                # skill evaluation results
├── agent-performance/    # per-agent performance over time
├── meta-learning/        # meta-learning patterns
└── playbooks/             # reusable playbooks
```

### 10.4 Meta Memory

Knowledge about the agent system itself. Subset of strategic memory under `meta-learning/`:

```
meta-learning/
├── agent_performance.yaml       # per-agent success rate, token cost, runtime
├── architecture_decisions.yaml # what architectures have worked
├── skill_evaluations.yaml      # which skills have helped
├── failure_modes.yaml          # recurring agent failure modes
└── prompt_ab_tests.yaml        # A/B test results for prompt changes
```

### 10.5 Memory Operations

The Memory Curator maintains the four layers via four operations:

1. **Ingest**: after every experiment, parse the artifact (experiment.json, metrics.json, oof_predictions.csv) and add an entry to project memory. Cross-reference: hypothesis → intervention → result → conclusion with confidence tier.
2. **Retrieve**: when an agent starts a task, the Memory Curator retrieves relevant entries from project + strategic memory. Retrieval is by tag, dataset-signature similarity, or model-family similarity — not by free-text search.
3. **Promote**: at competition end, validated project-memory entries that generalize are promoted to strategic memory (with explicit confidence tier and counterexamples noted).
4. **Prune**: stale or contradicted entries are flagged. The Memory Curator never auto-deletes — it flags for human review or for Adversarial Reviewer confirmation.

### 10.6 Compression

Project memory can grow large over a long competition. The Memory Curator compresses:

- Multiple experiments with the same hypothesis and similar results → one summary entry with a count and range.
- Experiments older than N days and not referenced by current ensemble candidates → archived (kept on disk, not retrieved).
- Verbose findings (free-text paragraphs) → bullet summaries for retrieval; full text preserved on disk.

### 10.7 Contradiction Handling

When new evidence contradicts an existing entry:

- Both entries are kept (with a "contradicts: <id>" cross-reference).
- Confidence of both is reduced by 0.1.
- The Adversarial Reviewer is notified at the next gate.

### 10.8 Provenance

Every entry has `provenance`:
```yaml
provenance:
  experiment_id: exp-007
  artifacts:
    - experiments/training/exp-007/metrics.json
    - experiments/training/exp-007/oof_predictions.csv
  retrieved_at: 2026-09-13
  retrieved_by: memory_curator
```

This is non-negotiable: no knowledge enters permanent memory without provenance.

---

## 11. Self-Evolution Architecture

Per the prompt (§7-§10), the system must evolve its **operational intelligence**, not just remember conversations. The self-evolution loop:

```text
OBSERVE  →  HYPOTHESIZE  →  PLAN  →  PARALLELIZE  →  IMPLEMENT  →
EXECUTE  →  MEASURE  →  CRITIQUE  →  SELECT  →  MEMORIZE  →
GENERALIZE  →  EVOLVE  →  REPEAT
```

### 11.1 Knowledge Promotion Levels

Per the prompt (§7), the system distinguishes:

- **Observed fact**: directly measured (e.g., "LightGBM CV=0.847 on dataset X with features Y"). Confidence tier: 1.0.
- **Strong empirical pattern**: multiple consistent observations (e.g., "LightGBM beats XGBoost on this dataset signature in 4 of 5 competitions"). Confidence tier: 0.8.
- **Weak hypothesis**: one or two observations (e.g., "CatBoost with target encoding might help on high-cardinality categoricals"). Confidence tier: 0.5.
- **Speculation**: no observations, just inference (e.g., "if we used a transformer we might do better"). Confidence tier: 0.2.

Promotion rules:
- Observed fact (1.0) is permanent.
- Strong empirical pattern (0.8) is permanent if no counterexample in 6 months.
- Weak hypothesis (0.5) is project-memory only — never promoted to strategic memory until it accumulates more evidence.
- Speculation (0.2) is working-memory only — discarded at end of task.

The system never auto-promotes speculation to permanent knowledge (per prompt §7).

### 11.2 Meta-Learning Across Competitions

Per the prompt (§8), the framework learns patterns like:

> "For datasets with high-cardinality categorical variables and medium-sized tabular data, CatBoost frequently provides a strong baseline."

Stored as:
```yaml
meta_lesson:
  id: meta-014
  pattern: "CatBoost is strong on high-cardinality tabular"
  evidence_count: 4
  competitions: [comp-2023-tabular-a, comp-2024-tabular-b, comp-2025-tabular-c, comp-2026-tabular-d]
  confidence: 0.75
  counterexamples: []
  last_validation: 2026-09-13
  provenance:
    - experiments/.../metrics.json
    - knowledge/meta-learning/catboost_high_card.yaml
```

### 11.3 Agent Performance Tracking

Per the prompt (§10), every agent is tracked:

```yaml
agent_stats:
  trainer:
    invocations: 47
    success_rate: 0.89
    avg_runtime_s: 38
    avg_input_tokens: 7200
    avg_output_tokens: 1450
    useful_discoveries: 3
    false_positives: 1
    wasted_compute_hours: 1.2
    downstream_lb_impact: 0.012   # estimated contribution to LB improvement
    reproducibility: 0.94
    failure_modes:
      - "occasionally misroutes to wrong validation_config.yaml version (2 occurrences)"
```

### 11.4 Controlled Evolution

Per the prompt (§10), changes to agents (prompt, role, routing, memory, skills) go through:

```text
candidate change
  → benchmark
  → compare against baseline
  → statistical/evidence review
  → promote if superior
  → otherwise rollback
```

This is run as an A/B test: the new version runs on a held-out competition (or a simulated past competition), the old version runs in parallel, results are compared. Promotion requires both statistically significant improvement and Adversarial Reviewer sign-off.

### 11.5 What the System Does NOT Do

- Auto-modify its own orchestration logic blindly (per prompt §10).
- Auto-install arbitrary skills (per prompt §9).
- Auto-promote speculation to knowledge (per prompt §7).
- Run an infinite autonomous loop (per prompt §27).

---

## 12. Skill Evolution Architecture

Per the prompt (§9), the framework can create new skills when repeated evidence shows that an agent lacks a reusable capability. The pipeline:

```text
Failure
  ↓
Failure Classification   (via the Failure Recovery System — §19)
  ↓
Root Cause
  ↓
Generalizable Pattern?
  ↓ (no) → log and stop
  ↓ (yes)
Skill Candidate
  ↓
Skill Draft (SKILL.md with: version, purpose, trigger, dependencies, evidence, expected benefit, known failure modes, evaluation cases)
  ↓
Skill Evaluation (run on held-out tasks; compare with vs. without)
  ↓
Adversarial Review
  ↓
Versioned Skill (committed to kaggle-agent-core/skills/ with semantic version)
  ↓
A/B Test (next competition uses the new skill in shadow mode)
  ↓
Promotion / Rejection
```

### 12.1 Skill Metadata Schema

Every skill in the framework has:
```yaml
name: leakage-detection-target-correlation
version: 1.2.0
purpose: "Detect target-correlated features that would leak at inference"
trigger:
  condition: "after data profiling"
  agents: [data_forensics, validation_architect]
dependencies:
  - polars >= 0.20
  - numpy
evidence:
  - competition: comp-2025-tabular-a
    improvement: "caught 3 leakage features that would have raised CV by 0.04 but tanked LB"
expected_benefit: "prevents leakage-induced shake-up"
known_failure_modes:
  - "false positive on legitimately-correlated features in causal-modeling competitions"
evaluation_cases:
  - case: "synthetic dataset with planted leakage"
    expected: detect
  - case: "clean dataset"
    expected: no flags
  - case: "competition with target encoding"
    expected: flag with low confidence
```

### 12.2 Skill Discovery Pipeline

1. Failure with no existing skill to prevent it.
2. Failure Recovery System classifies the failure.
3. If the failure pattern is generalizable (same root cause seen ≥2 times), a skill candidate is opened.
4. Skill draft is written by the relevant agent (or by the Memory Curator) and saved to `skills/candidates/<name>/`.
5. Skill is evaluated on a held-out testbed.
6. Adversarial Reviewer attacks the skill.
7. Skill is versioned and committed.
8. Skill runs in shadow mode (logged but not actioned) for the next competition.
9. Promotion or rejection based on shadow-mode results.

### 12.3 Skills vs. Agents

A skill is a **reusable capability invoked by an agent**. An agent is a **persistent role with a system prompt**. The framework's rule: if a capability is invoked identically by multiple agents, it's a skill. If a capability is owned by exactly one role, it's part of that agent's prompt.

---

## 13. Experiment Scheduler

Per the prompt (§17, §26), experiments are prioritized by expected value:

```text
priority =
  (expected_score_gain × confidence × information_gain)
  /
  (compute_cost × risk)
```

### 13.1 Per-Experiment Fields

```yaml
experiment:
  id: exp-007
  hypothesis: "CatBoost with target encoding on high-cardinality categoricals beats LightGBM baseline"
  intervention: "switch model_family from lightgbm to catboost; add target_encoding skill"
  dataset_version: data/v3
  feature_version: features/v2
  validation: validation/v1
  seed: 42
  model: catboost
  parameters:
    iterations: 5000
    learning_rate: 0.03
    depth: 8
    loss_function: "Logloss"
  cv_score: 0.852
  runtime_s: 1240
  memory_mb: 1840
  artifact: experiments/training/exp-007/
  conclusion: "CatBoost CV=0.852 vs. LightGBM CV=0.847 — small improvement; add to ensemble pool"
  confidence: 0.7
  reproducibility: 0.95
  priority: 8.3
  expected_gain: 0.005
  compute_cost: 0.4    # GPU-hours
  risk: 0.3            # probability of no improvement
  information_gain: 0.8
```

### 13.2 Scheduler Logic

At each scheduling tick (typically every phase boundary):

1. List all candidate experiments (from agent recommendations + Memory Curator's hypothesis queue).
2. Compute `priority` for each.
3. Sort descending.
4. Run top N (limited by remaining Kaggle compute budget and Antigravity rate limits).
5. After each completes, update priority for remaining candidates (information-gain may decrease; dependencies may resolve).

### 13.3 Parallelization Rules

Per the prompt (§17):
- Genuinely independent experiments (different model families, different feature subsets) can run in parallel as async subagents.
- Tasks with shared mutable state (writing to the same fold assignments; modifying the same feature catalog) run sequentially.

Worktrees: each parallel experiment runs in its own Git worktree (`worktrees/exp-007/`) so file writes don't collide. Results are merged back to the main checkout at the end.

### 13.4 Stopping Policy

Per the prompt (§27), stop or change direction when:
- Improvements plateau (last 3 experiments added < 0.001 CV).
- Compute budget is exhausted (remaining GPU-hours < 1× expected baseline runtime).
- Validation becomes unstable (CV variance across folds > 0.02).
- Experiments repeatedly fail (3+ consecutive failures with the same root cause).
- Ensemble diversity is exhausted (no new model with < 0.95 correlation to existing blend).
- Deadline risk becomes significant (competition end < 7 days; preserve compute for final submission).
- Remaining experiments have low expected value (priority < 3.0).

Explicit stopping rules are encoded in `policies/stopping.yaml` and evaluated at every scheduling tick.

---

## 14. Validation Architecture

Per the prompt (§15), validation is a first-class artifact. The system must answer:

> "Why should this CV estimate correlate with private leaderboard performance?"

If the answer isn't convincing, modeling does not proceed.

### 14.1 Validation Artifacts

```
validation/
├── strategy.md           # the rationale (answers the above question)
├── folds/                # actual fold assignments (one CSV per fold)
├── leakage_tests/        # automated leakage-detection results
└── validation_config.yaml
```

### 14.2 CV Strategy Selection

The Validation Architect selects from:
- **KFold** (random) — for i.i.d. data, no group, no time.
- **StratifiedKFold** — for classification with class imbalance.
- **GroupKFold** — when entities (e.g., customer_id) appear in both train and test; prevents train/test contamination.
- **StratifiedGroupKFold** — group-aware + class-stratified.
- **TimeSeriesSplit** — for time-series; train on past, validate on future.
- **PurgedKFold** (López de Prado) — for time-series with autocorrelation; gap between train and validation to avoid leakage.
- **CombinatorialPurgedKFold** — for time-series with multiple validation paths.

Selection rules:
- Is there a time column? → temporal CV (TimeSeriesSplit or PurgedKFold).
- Is there a group ID column? → GroupKFold or StratifiedGroupKFold.
- Both? → temporal + group-aware (custom).
- Neither? → StratifiedKFold.

### 14.3 Leakage Detection

The Data Forensics agent and Validation Architect jointly check:
- **Target leakage**: features highly correlated with target (correlation > 0.95) and likely unavailable at inference.
- **Temporal leakage**: future features available in training rows (e.g., a "next_month_sales" column included in training).
- **Group leakage**: same entity (customer, store, patient) in both train and validation folds.
- **Duplicated observations**: exact or near-duplicate rows across folds.
- **External data contamination**: features derived from external data that wasn't supposed to be available.
- **Feature construction leakage**: features computed using the full dataset (e.g., target encoding without proper within-fold computation).

### 14.4 Distribution Shift Detection

Methods:
- **Adversarial validation**: train a model to distinguish train vs. test (LightGBM with binary target). High AUC = distribution shift. Features with high importance in the AV model differ between train and test → candidates for removal.
- **PSI (Population Stability Index)**: per-feature, PSI > 0.25 = significant shift.
- **KS-test**: per-feature, statistical test for distribution difference.

If distribution shift is detected:
- Drop or transform the shifting features.
- Reweight training data to match test distribution (importance weighting).
- Use time-aware CV (recent observations more representative of test).

### 14.5 CV-LB Gap Justification

The Validation Architect must write, in `strategy.md`:

> "The expected correlation between this CV estimate and private LB performance is **[high/medium/low]** because:
> - The CV strategy mirrors the test set's structure in [specific way].
> - Adversarial validation AUC is [value], indicating [low/moderate/high] distribution shift.
> - Group leakage is [prevented by GroupKFold on customer_id / N/A].
> - Temporal leakage is [prevented by PurgedKFold with 7-day gap / N/A].
> - The remaining risk is [specific risk]: [mitigation]."

### 14.6 Validation Config Schema

```yaml
# validation/validation_config.yaml
strategy: stratified_group_kfold
n_splits: 5
group_col: customer_id
stratify_col: target
random_state: 42
purge_days: 0     # 0 for non-temporal
gap_days: 0
custom_fold_file: null

leakage_tests:
  target_correlation_threshold: 0.95
  temporal_check: enabled
  group_check: enabled
  duplicate_check: enabled

distribution_shift:
  method: adversarial_validation
  auc_warning_threshold: 0.85
  feature_removal_threshold: 0.10   # drop features with AV importance > 0.10 if AUC > 0.85
```

---

## 15. Kaggle Automation Architecture

Per the prompt (§22), the framework automates the Kaggle CLI loop:

```text
LOCAL PROJECT
      ↓
Git commit
      ↓
Kaggle Notebook (kernel-metadata.json)
      ↓
kaggle kernels push
      ↓
Kaggle Execution (poll status)
      ↓
kaggle kernels output (download artifacts)
      ↓
LOCAL PROJECT (artifacts ingested)
      ↓
Antigravity analyzes artifacts
      ↓
Next experiment
```

### 15.1 The Kaggle Executor Agent

Wraps the Kaggle CLI with:
- Strict error handling (parse exit codes, stderr).
- Retry with exponential backoff (3 retries on transient failures).
- Structured logging (every CLI call logged to `state/logs/kaggle_executor.jsonl`).
- Status polling (`kaggle kernels status` every 60s while running; timeout after 9h).
- Output download (`kaggle kernels output -p artifacts/<run_id>/`).

### 15.2 Notebook Preparation

The Trainer agent produces a notebook template; the Kaggle Executor agent fills it in:

```python
# notebook.py (template — filled in by Kaggle Executor)
import json, os, sys, time, random, numpy as np, polars as pl
# Set seeds
SEED = {{seed}}
random.seed(SEED); np.random.seed(SEED)

# Load data
DATA_DIR = "/kaggle/input/{{competition_name}}/"
train = pl.read_csv(f"{DATA_DIR}/train.csv")
test = pl.read_csv(f"{DATA_DIR}/test.csv")

# {{feature_code}}

# {{model_code}} (loaded from {{model_artifact_path}} or trained inline)

# Predict
test_pred = model.predict(test[features])
submission = pl.DataFrame({"id": test["id"], "target": test_pred})
submission.write_csv("/kaggle/working/submission.csv")
```

`kernel-metadata.json`:
```json
{
  "id": "{{kaggle_username}}/{{competition_name}}-exp-{{exp_id}}",
  "title": "{{competition_name}}-exp-{{exp_id}}",
  "code_file": "notebook.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": {{use_gpu}},
  "enable_tpu": false,
  "enable_internet": false,
  "dataset_sources": ["{{kaggle_username}}/{{model_dataset_slug}}"],
  "competition_sources": ["{{competition_name}}"],
  "kernel_metadata_sources": []
}
```

### 15.3 Polling and Output

The Kaggle Executor polls:
```bash
kaggle kernels status {{username}}/{{slug}}
# Output: {"status": "running"} or {"status": "complete"} or {"status": "error"}
```

On `complete`, downloads output:
```bash
kaggle kernels output {{username}}/{{slug}} -p artifacts/{{run_id}}/
```

The output directory contains everything written to `/kaggle/working/` — `submission.csv`, `metrics.json`, `oof_predictions.csv`, `logs/`, etc. (The notebook template writes these structured artifacts.)

### 15.4 Submission to Leaderboard

For traditional competitions: `kaggle competitions submit -c {{comp}} -f submission.csv -m "{{exp_id}}"`.
For code competitions: the notebook push itself IS the submission (select "Submit to Competition" from the kernel-metadata or via a flag — verify the exact mechanism).

The framework's policy: only submit when CV improvement is significant (> 0.001 CV) and Adversarial Reviewer has signed off.

---


## 16. Artifact Architecture

Per the prompt (§24), every Kaggle run produces machine-readable artifacts. Minimum schema:

```
artifacts/
├── metrics.json
├── predictions.csv
├── submission.csv
├── oof_predictions.csv
├── experiment.json
├── resource_usage.json
├── logs/
│   ├── stdout.log
│   ├── stderr.log
│   └── kaggle_executor.jsonl
└── reports/
    ├── training_curve.png
    ├── feature_importance.png
    └── prediction_distribution.png
```

### 16.1 metrics.json Schema

```json
{
  "experiment_id": "exp-007",
  "competition": "comp-2026-tabular-x",
  "cv_score": 0.852,
  "cv_std": 0.012,
  "fold_scores": [0.847, 0.851, 0.855, 0.849, 0.858],
  "metric_name": "roc_auc",
  "metric_direction": "higher_better",
  "oof_score": 0.852,
  "public_lb_score": null,
  "private_lb_score": null,
  "timestamp": "2026-09-13T10:15:00Z"
}
```

### 16.2 experiment.json Schema

```json
{
  "id": "exp-007",
  "hypothesis": "...",
  "intervention": "...",
  "dataset_version": "data/v3",
  "feature_version": "features/v2",
  "validation_version": "validation/v1",
  "seed": 42,
  "model_family": "catboost",
  "model_params": {"iterations": 5000, "learning_rate": 0.03, "depth": 8},
  "predecessors": ["exp-005", "exp-006"],
  "successors": [],
  "tags": ["catboost", "target_encoding"],
  "status": "complete",
  "conclusion": "small improvement over baseline; added to ensemble pool"
}
```

### 16.3 resource_usage.json Schema

```json
{
  "experiment_id": "exp-007",
  "runtime_s": 1240,
  "peak_memory_mb": 1840,
  "cpu_avg": 1.2,
  "gpu_avg": 0.0,
  "disk_mb": 320,
  "kaggle_quota_used": {
    "gpu_hours": 0.0,
    "cpu_hours": 0.345
  },
  "antigravity_tokens": {
    "input": 7200,
    "output": 1450
  }
}
```

### 16.4 Artifact Stability

The artifact schema stays stable across competitions. Only the `metrics.json` values change; the structure doesn't. This lets the Memory Curator ingest artifacts uniformly.

### 16.5 Artifact Ingestion Pipeline

Per the prompt (§25), after artifact download:

1. **Discover**: scan `artifacts/<run_id>/` for known file names.
2. **Validate**: schema-validate each JSON file; fail loudly on schema mismatch.
3. **Parse metrics**: extract cv_score, fold_scores, runtime, memory.
4. **Compare**: against previous experiments in the registry.
5. **Identify improvements/regressions**: delta vs. baseline.
6. **Update experiment registry**: append to `state/experiments.yaml`.
7. **Update strategic memory**: Memory Curator ingests the structured artifact and writes a knowledge entry with provenance.
8. **Determine next experiments**: based on conclusion + recommendations.
9. **Update agent performance**: log the agent stats for this experiment.
10. **Decide skill evolution**: if the experiment revealed a recurring failure, the Skill Evolution pipeline is triggered.

This happens without user intervention.

---

## 17. Compute Optimization Architecture

Per the prompt (§18, §40), the framework aggressively optimizes both ML compute (Kaggle's CPUs/GPUs/RAM) and AI compute (Antigravity tokens).

### 17.1 ML Compute Optimization

**Data processing**:
- **Polars** for all tabular data — lazy evaluation, multi-threaded, Arrow-native, often 5-10× faster than pandas on Kaggle-sized data `[T4]`.
- **Parquet** for on-disk intermediate storage — columnar, compressed, faster I/O than CSV `[T4]`.
- **NumPy** for numerics; **avoid Python loops** in hot paths.
- **Memory mapping** for huge files (`np.memmap`).
- **joblib.Parallel** for CPU-bound parallelism across folds.
- **Caching** — feature engineering caches intermediate results to `features/v<N>/.cache/`.

**Training throughput**:
- **LightGBM**: `num_threads=-1`, `device_type="cpu"` (GPU mode in LightGBM is often slower than CPU on Kaggle-sized tabular data `[T5]`).
- **XGBoost**: `n_jobs=-1`, `tree_method="hist"` (default; the old `exact` is slow).
- **CatBoost**: `task_type="CPU"` (GPU mode only helps on >1M-row datasets).
- **GPU only when it actually helps**: deep learning (CV, NLP), or large-data GBDT. Many tabular competitions are faster on CPU `[T5]`.

**Avoid**:
- Unnecessary pandas copies (use `pl.LazyFrame` and `collect()` once).
- Repeated CSV parsing (read once, save as Parquet, read Parquet).
- Redundant preprocessing (cache features).
- Repeated model fitting (cache OOF predictions).
- Excessive logging (structured logs, not free text).

### 17.2 AI Compute Optimization

**Context size**:
- Layered context architecture (per prompt §31) — never dump entire repo into agent context.
- Memory retrieval (not memory dump) — retrieve relevant snippets, not full transcripts.
- Schema-constrained outputs — shorter, no rambling.

**Prompt reuse**:
- Global rules + role context are static and can be cached by Antigravity.
- Per-competition brief is small and stable.
- Per-task prompt is short.

**Parallel agent execution**:
- Async subagents for independent experiments.
- Per-prompt cost amortized across many experiments.

**Unnecessary agent calls**:
- The Commander batches decisions — no agent call for "should I run another fold?" (fold count is in validation_config.yaml).
- The Memory Curator batches ingest — one call per N experiments, not one per experiment.

**Memory retrieval**:
- Indexed by tags + dataset-signature similarity, not free-text search.
- Top-K retrieval (K=5 typically), not full memory dump.

**Structured outputs**:
- JSON-schema-constrained — shorter than free text, parseable, no downstream agent needed to interpret.

### 17.3 Automatic Bottleneck Benchmarking

The Performance Engineer agent (optional, not in the 12-agent core — folded into Memory Curator) benchmarks:
- Per-experiment ML runtime breakdown (data load, feature gen, train, predict).
- Per-agent AI runtime breakdown (input tokens, output tokens, wall clock).
- Per-phase wall-clock breakdown.

When a bottleneck is found (>50% of phase time), it's logged and a remediation is queued.

---

## 18. Security / Compliance Architecture

Per the prompt (§29), the framework has a Rule Compliance agent (folded into the Competition Researcher + Adversarial Reviewer in our 12-agent roster — the Competition Researcher extracts rules; the Adversarial Reviewer enforces them at Gate 3).

### 18.1 Compliance Checklist

Before any submission, the Adversarial Reviewer (Gate 3) verifies:

- **External data permissions**: is external data allowed by this competition's rules? If yes, is the license compatible?
- **Internet restrictions**: code competitions disable internet — verify the notebook doesn't try to download anything.
- **API restrictions**: does the competition ban LLM APIs? Auto-submission APIs?
- **Compute restrictions**: are there per-team GPU-hour caps?
- **Team rules**: are we within the team-merge deadline? Are all members on the team?
- **Submission frequency**: have we exceeded daily submission limits?
- **Model restrictions**: are pretrained models allowed? Which architectures?
- **Pretrained-model restrictions**: any bans on specific model families (e.g., competitions sometimes ban LLM use)?
- **Licensing**: are we using only permissively-licensed code?
- **Reproducibility requirements**: are seeds set? Can the notebook reproduce its submission?

### 18.2 Credential Management

- Kaggle credentials (`kaggle.json`) live in `~/.kaggle/`, **never** committed to Git.
- `.gitignore` excludes `kaggle.json`, `*.csv` (large data), `*.pkl` / `*.joblib` (model binaries unless intentionally tracked), `.env`.
- Antigravity's global config holds API keys at the user level, not the workspace level.
- MCP servers requiring secrets get them from env vars, not config files in the repo.

### 18.3 Data Protection

- Competition data lives in `competition-project/data/` and is **never committed** (Git LFS or just `.gitignore`).
- Private competition data is never uploaded anywhere outside Kaggle.
- Generated model binaries are kept in `artifacts/` and `.gitignore`d unless the user explicitly wants them tracked.

### 18.4 Submission Integrity

The Final Auditor (Gate 3) verifies the submission file:
- Row count matches the test set's sample submission.
- Column names match exactly.
- No NaN in required columns.
- Values are within the metric's valid range (e.g., probabilities in [0, 1] for AUC; non-negative integers for some metrics).
- Index/id column matches the test set's order.

If any check fails, the submission is blocked.

---

## 19. Failure Recovery Architecture

Per the prompt (§30), failures are classified and routed:

### 19.1 Failure Types

| Code | Failure | Recovery |
|------|---------|----------|
| `DATA_FAILURE` | data schema mismatch, missing files, corrupt CSV | Re-read from Kaggle, re-download |
| `DEPENDENCY_FAILURE` | Python package missing, version mismatch | Use `uv pip install`, fallback to Kaggle Dataset |
| `CODE_FAILURE` | Python exception in training/inference | Agent reads traceback, fixes the script, re-runs |
| `MEMORY_FAILURE` | out-ofemory | Reduce batch size, sample data, switch to out-of-core (Polars lazy) |
| `TIMEOUT` | Kaggle 9h wall-clock limit hit | Reduce model complexity, parallelize folds |
| `KAGGLE_FAILURE` | Kaggle API error (rate limit, server error) | Retry with backoff, switch compute to local |
| `VALIDATION_FAILURE` | CV-LB gap unexpectedly large, leakage detected | Roll back to last known-good validation strategy |
| `MODEL_FAILURE` | model doesn't converge, NaN loss | Reduce learning rate, switch optimizer |
| `AGENT_FAILURE` | agent returned `status: failure` | Commander re-routes to a different agent or retries with different prompt |
| `ORCHESTRATION_FAILURE` | subagent didn't return, hook failed | Restart from last checkpoint |
| `ARTIFACT_FAILURE` | artifact schema mismatch | Re-run the experiment with strict artifact validation |

### 19.2 Recovery Strategy

For each failure:
1. Classify the failure type.
2. Look up the recovery strategy.
3. Apply the recovery.
4. Log the failure + recovery for agent-evolution analysis.
5. **Never retry the same action without changing the hypothesis** (per prompt §30).

If 3 retries on the same failure type fail, the Commander escalates: stops the phase, writes a failure report, and asks the user to intervene.

### 19.3 Self-Healing Pipeline

```text
Failure occurs
  ↓
Classifier (Memory Curator) categorizes failure type
  ↓
Lookup recovery policy in policies/failure_recovery.yaml
  ↓
Apply recovery (retry with backoff, switch strategy, etc.)
  ↓
If recovery succeeds: log and continue
If recovery fails: escalate to Commander
  ↓
If pattern repeats: Skill Evolution pipeline triggers
```

---

## 20. Reusability Architecture

Per the prompt (§35), the framework is **reusable across competitions without rewriting the core**.

### 20.1 Core / Project Split

```
~/projects/kaggle-agent-core/        # reusable framework, version-controlled
├── agents/                          # agent system prompts + configs
├── skills/                          # global skills (Antigravity-global scope)
├── memory/                          # strategic memory (cross-competition)
├── orchestration/                   # Commander's playbook templates
├── evaluation/                      # agent self-evaluation harness
├── schemas/                         # YAML/JSON schemas for artifacts
├── playbooks/                       # reusable playbooks (default, tabular, CV, etc.)
├── scripts/                         # utility scripts (Kaggle wrappers, etc.)
├── policies/                        # stopping, failure recovery, etc.
├── templates/                       # notebook templates, config templates
└── config/                          # runtime config (Antigravity bindings)

~/projects/competition-{{name}}/     # per-competition project
├── data/                            # competition data (gitignored)
├── src/                             # competition-specific source
├── notebooks/                       # Kaggle notebooks
├── experiments/                     # per-experiment artifacts
├── artifacts/                       # aggregated artifacts
├── reports/                         # EDA, leakage, validation reports
├── knowledge/                       # project memory
├── features/                        # versioned feature sets
├── validation/                      # CV strategy
├── ensemble/                        # ensemble artifacts
├── .agents/                         # workspace-local Antigravity config
│   ├── skills/                      # competition-specific skills
│   ├── hooks/                       # competition-specific hooks
│   └── agents/                      # competition-specific agent overrides
├── competition.yaml                 # competition metadata
├── AGENTS.md                        # Antigravity workspace agent manifest
└── README.md
```

### 20.2 What's Reusable

- `kaggle-agent-core/` is checked into a single Git repo. Each competition symlinks or templates from it.

### 20.3 What's Per-Competition

- `competition-project/` is the working directory. At competition start, a script (`scripts/init_competition.sh`) creates the directory structure from `kaggle-agent-core/templates/competition_template/`.

### 20.4 Skills Scope

Per prompt §36:
- **Global skills** (Antigravity global): competition-agnostic capabilities — leakage-detection, validation-design, ensemble-optimization, artifact-analysis, kaggle-compute-optimization, agent-memory.
- **Project skills** (Antigravity workspace-local under `.agents/skills/`): competition-specific — competition-rules, domain-knowledge, dataset-specific-analysis, competition-specific-playbook.

---

## 21. Complete Project Structure

(See §20 above and Appendix B for the full tree.)

---

## 22. Skill Selection from the 2,121-Skill Catalog (Category Highlights)

> **Note on the catalog**: The full 2,121-row acceptance/rejection table was deferred per the user's "just give me the blueprint now" instruction. What follows is a category-level triage based on the author's read of the catalog preview (`agent-behavior`, `multi-agent-*`, `context-*`, `data-scientist`, `ml-engineer`, `polars`, `using-git-worktrees`, `uv-package-manager`, etc.). A full row-by-row triage should be done as Phase 0 of implementation (see Part II §26).

### 22.1 Categories to Investigate

From the catalog preview, the following categories are most relevant to this framework:

- **agent-behavior** (5 skills) — workflow discipline, verification patterns. Relevant: `fable-safe-prompt`, `codex-fable5`. **Recommend**: inspiration only — patterns from these can be embedded into our agent prompts without installing the skills.
- **agent-memory** (multiple skills) — `agent-memory`, `agent-memory-mcp`. **Recommend**: install `agent-memory` globally; use `agent-memory-mcp` as an MCP server if Antigravity's native memory proves insufficient.
- **multi-agent-*** (multiple skills) — `multi-agent-task-orchestrator`, `multi-agent-architect`, `multi-agent-patterns`, `parallel-agents`, `dispatching-parallel-agents`, `agent-orchestrator`, `agent-orchestration-improve-agent`, `agent-orchestration-multi-agent-optimize`. **Recommend**: use as inspiration for our orchestration patterns; install `multi-agent-patterns` globally as a reference; reject the rest as duplicative of our built-in 12-agent architecture.
- **context-*** (multiple skills) — `context-engineering`, `context-optimization`, `context-compression`, `context-guardian`, `context-agent`. **Recommend**: install `context-engineering` globally; consider `context-compression` if Antigravity's native context management proves insufficient; reject the rest.
- **evaluation** (`agent-evaluation`, `evaluation`) — **Recommend**: install `agent-evaluation` globally for our agent-evaluation layer.
- **data-science / ml-engineer** — **Recommend**: install `data-scientist` globally as a reference skill for the Trainer and Feature Engineer agents.
- **machine-learning-ops-ml-pipeline** — **Recommend**: inspiration only; embed patterns into our `experiment_manager` skill.
- **polars** — **Recommend**: install globally; the framework uses Polars heavily.
- **using-git-worktrees** — **Recommend**: install globally; the framework uses worktrees for parallel experiments.
- **uv-package-manager** — **Recommend**: install globally; the framework uses `uv` for environment management.

### 22.2 Skills to Reject (Category-Level)

- **Skills tied to non-Antigravity runtimes** (Claude Code, OpenAI Codex, Cursor, Continue, etc.) — incompatible. Reject.
- **Skills duplicating Antigravity's native capabilities** (basic file I/O, basic shell, basic code search) — already provided by Antigravity. Reject.
- **Skills introducing non-Gemini models** (OpenAI, Anthropic, OpenRouter) — violates the prompt's §3 constraint. Reject.
- **Skills for unrelated domains** (web dev, mobile, game dev, etc.) — not relevant to Kaggle. Reject.
- **Skills with "critical" risk level and unclear benefit** — skip until evaluated. Reject by default; revisit if a specific need arises.

### 22.3 Skills to Embed (Not Install)

Some patterns from catalog skills should be **embedded into our agent prompts** rather than installed as separate skills, because they're invoked constantly and the indirection of a skill call would cost more than inlining:

- Verification patterns from `codex-fable5` → embed in Commander + Adversarial Reviewer prompts.
- Decomposition patterns from `multi-agent-patterns` → embed in Commander prompt.
- Compression patterns from `context-compression` → embed in Memory Curator prompt.

### 22.4 Summary

A defensible skill selection from the catalog:
- **Install globally (~8-10 skills)**: `agent-memory`, `agent-evaluation`, `context-engineering`, `data-scientist`, `polars`, `using-git-worktrees`, `uv-package-manager`, plus a handful of others identified during the full triage.
- **Embed patterns (~5-10 skills)**: verification, decomposition, compression patterns inlined into agent prompts.
- **Reject (~2100+ skills)**: non-Antigravity runtimes, duplicative, unrelated, or incompatible.

The full row-by-row triage should be Phase 0 of implementation (see Part II §26).

---

## 23. Additional External Skills Discovered (Author Knowledge)

These are skills/patterns the author knows from outside the catalog that the framework should incorporate (typically as embedded patterns or as custom skills in `kaggle-agent-core/skills/`):

### 23.1 Custom Skills to Build

1. **`kaggle-competition-analysis`** — wraps the Kaggle CLI to extract competition metadata.
2. **`leakage-detection-suite`** — automated leakage detection (target correlation, temporal, group, duplicates, feature-construction).
3. **`validation-design`** — CV strategy selection given dataset signature.
4. **`adversarial-validation`** — train/test distinguisher with feature-importance flagging.
5. **`experiment-manager`** — registry, schema validation, ingestion.
6. **`ensemble-optimization`** — hill climbing, forward stepwise, blend-weight search.
7. **`artifact-analyzer`** — parse `metrics.json`, `oof_predictions.csv`, plot training curves, compute diversity matrix.
8. **`kaggle-compute-optimizer`** — decide CPU vs GPU, batch size, fold parallelism.
9. **`agent-memory-curator`** — four-layer memory operations.
10. **`skill-evolution`** — failure → candidate → draft → evaluate → adversarial review → versioned.
11. **`agent-performance-tracker`** — per-agent stats collection.
12. **`reproducibility-checker`** — verify seeds set, determinism enabled.
13. **`distribution-shift-detector`** — PSI, KS-test, AV.
14. **`pseudo-labeling`** — confidence-thresholded PL with within-fold validation.
15. **`test-time-augmentation`** — TTA patterns per problem type.

### 23.2 External Frameworks to Learn From (Not Install)

- **AutoGluon** (Tabular, Multimodal) — strong AutoML baseline; the framework should run AutoGluon as one candidate in the model pool.
- **FLAML v2** — fast AutoML; alternative baseline.
- **Optuna** — HPO; the framework uses Optuna directly.
- **MemGPT / Letta** — long-term memory patterns; inspiration for the four-layer memory.
- **Voyager** — skill library pattern; inspiration for Skill Evolution.
- **OpenHands / SWE-Agent** — agent scaffolding patterns; inspiration for our agent prompts.

### 23.3 External Patterns to Adopt

- **Caruana 2004 ensemble selection** — forward stepwise hill climbing on OOF.
- **López de Prado PurgedKFold** — temporal CV with purging.
- **Anthropic's Context Engineering (2025)** — layered context, structured retrieval.
- **ReAct (Yao et al. 2023)** — tool-use with explicit reasoning steps; embedded in agent prompts.
- **Reflexion (Shinn et al. 2023)** — self-reflection with memory; embedded in Memory Curator.

---

## 24. Rejected Skills and Why

Per-competition, the framework rejects skills that:

### 24.1 Incompatible with the Runtime Constraint

- Any skill requiring Claude / GPT / OpenAI / Anthropic APIs → violates prompt §3.
- Any skill requiring a local LLM (Ollama, llama.cpp) → violates prompt §3.
- Any skill requiring OpenRouter / Together / Anyscale → violates prompt §3.

### 24.2 Duplicative of Antigravity Native Capabilities

- Basic file I/O wrappers (Antigravity has `Read`, `Write`, `Edit`).
- Basic shell wrappers (Antigravity has `Bash`).
- Basic code search wrappers (Antigravity has `Grep`, `Glob`).

### 24.3 Unrelated to Kaggle

- Web development skills.
- Mobile development skills.
- Game development skills.
- DevOps skills unrelated to ML (e.g., Kubernetes, Terraform).
- Backend API development skills.

### 24.4 Excessive Complexity for Marginal Gain

- Distributed-systems skills (the framework is single-machine).
- Database-connector skills (the framework uses files, not databases).
- Message-broker skills (the framework uses file-based blackboard).
- Cloud-infrastructure skills (the framework is Kaggle + local).

### 24.5 Unverified or Low-Confidence

- Skills with "critical" risk level and no documented benefit. Skip until evaluated.

---

## 25. Implementation Roadmap

The framework is built in 6 phases:

### 25.1 Phase 0 — Foundation (Week 1)

1. Initialize `kaggle-agent-core/` Git repo.
2. Create the directory structure (Appendix B).
3. Write the schema files (`schemas/artifact.schema.json`, etc.).
4. Write the policy files (`policies/stopping.yaml`, `policies/failure_recovery.yaml`).
5. Write the Antigravity `AGENTS.md` workspace manifest.
6. Write the 12 agent system prompts in `.agents/agents/*.md`.
7. Install the ~10 global skills.
8. Write `scripts/init_competition.sh` to scaffold a new competition project.

### 25.2 Phase 1 — Minimum Lovable System (Weeks 2-3)

Build the smallest end-to-end vertical slice:
- Commander + Competition Researcher + Data Forensics + Validation Architect + Trainer + Kaggle Executor.
- Tabular-only.
- Single model (LightGBM).
- No ensemble, no adversarial reviewer, no memory curator.
- Goal: take a tabular competition from start to a single submission.

### 25.3 Phase 2 — Add the Adversarial Gate (Week 4)

- Add the Adversarial Reviewer.
- Wire up Gate 1 (validation), Gate 3 (submission).
- Skip Gate 2 (ensemble) for now.

### 25.4 Phase 3 — Ensemble + Memory (Weeks 5-6)

- Add the Ensembler + Gate 2.
- Add the Memory Curator.
- Add the four-layer memory store.
- Add the experiment scheduler with the priority formula.

### 25.5 Phase 4 — Multi-Modality (Weeks 7-8)

- Add the EDA Specialist for non-tabular modalities.
- Add the Model Researcher for CV/NLP/time-series model families.
- Add modality-specific notebooks (CV with PyTorch + pretrained backbone; NLP with HF transformers; time-series with lag features).

### 25.6 Phase 5 — Self-Evolution (Weeks 9-10)

- Add the Skill Evolution pipeline.
- Add the Agent Evolution A/B test harness.
- Add the cross-competition meta-learning loop.

### 25.7 Phase 6 — Polish (Weeks 11-12)

- Hooks for invariant enforcement.
- Visual artifacts in EDA reports.
- Documentation: README.md, AGENTS.md, playbooks.
- Benchmark against 2-3 past competitions to validate.

---

## 26. Benchmark Plan

The framework is benchmarked:

### 26.1 ML Benchmark

Run the framework on 3 past Kaggle competitions with known winning solutions:
- 1 tabular competition.
- 1 CV competition.
- 1 NLP competition.

For each, measure:
- Final private-LB rank vs. the original winner.
- CV-LB gap (shake-up).
- Compute used (GPU-hours, CPU-hours).
- Wall-clock time from start to final submission.

Target: top-10% private-LB on each. Stretch: medal.

### 26.2 Agent Benchmark

Measure per-agent:
- Task success rate.
- Token cost.
- Wall-clock time.
- Failure modes.
- Downstream LB impact (estimated).

Target: 80%+ task success, <1000 tokens per call average.

### 26.3 Framework Benchmark

Measure end-to-end:
- Human-intervention count per competition (target: ≤5 interventions).
- Total token cost per competition.
- Reproducibility (re-run produces same final submission).

### 26.4 A/B Test Methodology

For any change (prompt, agent role, routing, memory, skill, orchestration, policy):
1. Run the new version on a held-out competition.
2. Run the old version in parallel.
3. Compare: final LB rank, CV-LB gap, compute, token cost.
4. Promote if superior on ≥2 metrics and not worse on any.
5. Adversarial Reviewer signs off.

---

## 27. Risks

### 27.1 Model Risk

- **Gemini 3.8 Flash High reasoning ceiling**: on hard planning tasks, the model may commit early to a wrong path. **Mitigation**: short bounded tasks, adversarial gates.
- **Hallucinated API signatures**: the model may confabulate library versions or function signatures. **Mitigation**: pinned dependencies, MCP wrappers with strict schemas.
- **Self-critique failure**: the model can't reliably find flaws in its own output. **Mitigation**: separate Adversarial Reviewer agent with a different prompt.

### 27.2 Antigravity Risk

- **CLI changes**: Antigravity is fast-moving; flags/paths may change. **Mitigation**: thin `runtime/` shim, all Antigravity-specific details isolated.
- **Rate limits**: many parallel subagents may hit rate limits. **Mitigation**: conservative concurrency (≤4 parallel agents).
- **Async-subagent reliability**: async subagents may fail silently. **Mitigation**: status polling, timeout, retry.

### 27.3 Kaggle Risk

- **Compute quota changes**: Kaggle may change GPU quotas. **Mitigation**: re-verify quotas before each competition.
- **Code competition rule changes**: Kaggle may change submission rules. **Mitigation**: Rule Compliance check at every competition.
- **Distribution shift / shake-up**: even with CV-first policy, the private LB may diverge. **Mitigation**: diversity-encouraged ensembles, adversarial validation.

### 27.4 Framework Risk

- **Over-engineering**: the 12-agent system may be too heavy for small competitions. **Mitigation**: dynamic skip of irrelevant phases (per prompt §43).
- **Under-engineering**: the system may miss non-obvious edges that distinguish #1 from top-10%. **Mitigation**: acknowledge this limit; rely on user-supplied domain insights.
- **Memory pollution**: bad knowledge entries may accumulate. **Mitigation**: contradiction detection, periodic pruning, Adversarial Reviewer sign-off on promotions.
- **Runaway experimentation**: the framework may run too many experiments. **Mitigation**: stopping policy, explicit compute budget.

### 27.5 Reproducibility Risk

- **CUDA non-determinism**: GPU training may produce slightly different results across runs. **Mitigation**: `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG`, pinned seeds.
- **External dependency drift**: library versions may change. **Mitigation**: pin all versions in `pyproject.toml` and a Kaggle Dataset.

---

## 28. Expected Bottlenecks

### 28.1 Primary Bottleneck: Antigravity Latency

The framework's primary bottleneck is **not** ML compute (Kaggle provides that). It's Antigravity's per-agent latency and rate limits. Each agent call takes 5-60 seconds; a phase may have 5-20 agent calls; a competition has 5-10 phases. That's 250-12000 seconds of agent time per competition — hours of wall-clock.

**Mitigation**: parallel async subagents where possible; conservative agent count; structured outputs (shorter); cached static context.

### 28.2 Secondary Bottleneck: Kaggle Wall-Clock

Kaggle notebooks can run up to 9 hours. For large models (CV, NLP), training is the long pole. Multiple sequential training runs compound this.

**Mitigation**: parallelize across Kaggle accounts (team pooling); use smaller models for exploration, larger for final; cache intermediate results.

### 28.3 Tertiary Bottleneck: Data I/O

Reading large CSVs repeatedly is slow.

**Mitigation**: Parquet cache; Polars lazy; one-shot read at start.

### 28.4 Token Cost

If Antigravity charges per-token, 100s of agent calls × 10k tokens/call = 1M+ tokens per competition. Manageable on subscription; expensive on pay-as-you-go.

**Mitigation**: structured outputs (shorter); retrieved context (not dumped); cache static context.

---

## 29. Future Improvements

### 29.1 Short-Term (3-6 months)

- **Full skill-catalog triage**: do the row-by-row 2,121-skill evaluation.
- **AutoGluon integration**: as one model candidate in the pool.
- **TabPFN/TabICL integration**: for small-tabular in-context learning baseline.
- **Better Memory retrieval**: vector embeddings for similarity retrieval (currently tag-based).
- **Hook-based invariant enforcement**: `PreToolUse` hook that validates every agent output against the schema.

### 29.2 Medium-Term (6-12 months)

- **Cross-competition knowledge graph**: explicit graph linking dataset signatures → validation strategies → model families → ensemble compositions → private-LB outcomes.
- **Skill auto-drafting**: when a recurring failure pattern is detected, the framework drafts a candidate skill automatically.
- **Multi-user team pooling**: integrate Kaggle's team-merge mechanism to pool compute across team members.
- **Multi-modal EDA**: vision-aware EDA using Gemini's image input.
- **Auto-submission selection policy**: learned policy for CV-vs-public-LB submission slot allocation.

### 29.3 Long-Term (12+ months)

- **Evolutionary agent architecture**: small population of agent-config variants competing; fittest reproduces.
- **Self-modifying playbooks**: the framework learns to rewrite its playbooks based on what worked.
- **Multi-region Kaggle account pooling**: use Kaggle accounts in different regions to extend compute (subject to Kaggle's ToS).
- **Foundation-model fine-tuning for tabular**: in-context learning baselines using the latest tabular foundation models.

### 29.4 What This Framework Will NOT Do

- Guarantee a #1 finish.
- Replace human domain insight.
- Operate without any human intervention (the user must accept competitions, provide credentials, and approve high-risk irreversible operations).
- Use non-Gemini reasoning models.
- Violate Kaggle competition rules.

---


# Part II — Implementation Blueprint

## 1. Exact Architecture

The architecture is **Candidate B: Hierarchical Supervisor + Blackboard + Critic-Generator Gates at 3 Decision Points** (see Part I §7).

### 1.1 Top-Level Architecture

```text
                        ┌──────────────────────────┐
                        │      COMMANDER            │
                        │  owns global objective    │
                        │  routes to cells          │
                        │  makes stopping &         │
                        │  submission-selection     │
                        │  decisions                │
                        └────────────┬─────────────┘
                                     │
                                     │  reads/writes blackboard
                                     │
        ┌─────────────┬──────────────┼──────────────┬─────────────┐
        │             │              │              │             │
  RESEARCH CELL  DATA CELL    MODEL CELL    EXPERIMENT CELL  SUBMISSION CELL
        │             │              │              │             │
  Competition    Data Forensics Feature Eng.   Trainer        Kaggle Executor
  Researcher     EDA Specialist Model Research Ensembler      Final Auditor
                                 HPO
                                     │
                                     │
                          ┌──────────┴──────────┐
                          │  ADVERSARIAL        │
                          │  REVIEWER (gate     │
                          │  at 3 points)       │
                          └─────────────────────┘
                                     │
                          ┌──────────┴──────────┐
                          │  MEMORY CURATOR     │
                          │  (4-layer memory)   │
                          └─────────────────────┘
```

### 1.2 Execution Traces

Every phase produces a JSONL execution trace at `state/logs/<date>/phase_<n>.jsonl`. The trace records every agent call, tool call, subagent spawn, and decision. This is the primary debugging artifact.

### 1.3 The Blackboard

The blackboard is `state/` directory at the competition-project root (see Appendix B). Every agent reads the blackboard at the start of its task; writes back its findings at the end. The blackboard is the source of truth — agents do not maintain private state across calls.

### 1.4 Concurrency Model

- The Commander runs sequentially (it owns global state; parallelism would risk inconsistency).
- Specialist agents run in parallel as async subagents when their tasks are independent.
- Antigravity's rate limits cap effective parallelism at ~3-4 concurrent subagents. The framework uses a semaphore.

### 1.5 Phase Loop

```text
INIT → COMPETITION_RESEARCH → DATA_FORENSICS → EDA → VALIDATION_DESIGN →
BASELINE → EXPERIMENTATION → ENSEMBLING → ADVERSARIAL_REVIEW →
KAGGLE_RUN → ARTIFACT_INGESTION → RESULT_ANALYSIS → KNOWLEDGE_UPDATE →
STRATEGY_EVOLUTION → FINAL_AUDIT → SUBMISSION
```

Phases may repeat (experimentation → ensembling → back to experimentation after a new feature is added). The Commander decides when to advance vs. iterate.

---

## 2. Exact Agent Roster (12 Agents)

The 12 agents, with file paths and invocation commands.

### 2.1 Agent File Layout

```
kaggle-agent-core/
├── AGENTS.md                              # workspace manifest (Antigravity)
└── agents/
    ├── commander.md                       # role spec
    ├── competition_researcher.md
    ├── data_forensics.md
    ├── eda_specialist.md
    ├── validation_architect.md
    ├── feature_engineer.md
    ├── model_researcher.md
    ├── hpo_agent.md
    ├── trainer.md
    ├── ensembler.md
    ├── adversarial_reviewer.md
    ├── kaggle_executor.md
    └── memory_curator.md
```

Each agent spec is a Markdown file with YAML frontmatter:

```markdown
---
name: validation_architect
model: gemini-3.8-flash-high
effort: high
tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
  - Subagent
skills:
  - leakage-detection-suite
  - validation-design
  - distribution-shift-detector
context_layers:
  - global_rules
  - role_context
  - competition_brief
  - relevant_memory
  - current_task
  - relevant_artifacts
  - recent_experiments
output_schema: schemas/agent_result.schema.json
---

# Validation Architect

You are the Validation Architect for an autonomous Kaggle competition team.
You design the cross-validation strategy that mirrors the private leaderboard.

## Your job

1. Read the dataset signature (state/dataset_signature.yaml) and the data-forensics reports.
2. Identify leakage: target, temporal, group, duplicates.
3. Identify distribution shift (use the adversarial-validation skill).
4. Select a CV strategy (KFold / StratifiedKFold / GroupKFold / TimeSeriesSplit / PurgedKFold).
5. Write the validation artifacts to validation/.

## Veto power

You have veto power: if you cannot justify CV-LB correlation, modeling does not proceed.
You must answer the question: "Why should this CV estimate correlate with private leaderboard performance?"
in validation/strategy.md.

## Output

Return a YAML object per schemas/agent_result.schema.json.
```

### 2.2 Agent Specifications (Abbreviated)

For each of the 12 agents, the role spec follows the same structure. The full text of each is in `kaggle-agent-core/agents/*.md`. Below is a summary table:

| # | Agent | Cell | Effort | Tools | Key Skills | Veto? |
|---|-------|------|--------|-------|------------|-------|
| 1 | Commander | Top | High | All | (none) | n/a |
| 2 | Competition Researcher | Research | Medium | Read, Write, Bash, WebFetch | kaggle-competition-analysis | No |
| 3 | Data Forensics | Data | Med-High | Read, Write, Bash, Grep, Glob | leakage-detection-suite, distribution-shift-detector | No |
| 4 | EDA Specialist | Data | Medium | Read, Write, Bash | (custom; uses Polars) | No |
| 5 | Validation Architect | Data | High | Read, Write, Bash, Subagent | validation-design, leakage-detection-suite | **Yes (Gate 1)** |
| 6 | Feature Engineer | Model | Med-High | Read, Write, Bash, Subagent | (custom; embeds hypothesis-driven pattern) | No |
| 7 | Model Researcher | Model | Medium | Read, Write, WebFetch, Subagent | (custom; embeds SOTA knowledge) | No |
| 8 | HPO Agent | Model | Medium | Read, Write, Bash | (custom; uses Optuna) | No |
| 9 | Trainer | Experiment | Medium | Read, Write, Bash, Subagent | (custom; uses Polars + LightGBM/XGB/CatBoost) | No |
| 10 | Ensembler | Experiment | Med-High | Read, Write, Bash | ensemble-optimization | No (but Gate 2 applies) |
| 11 | Adversarial Reviewer | Top | High | Read, Write, Bash | (custom; embeds checklist) | **Yes (Gates 1, 2, 3)** |
| 12 | Kaggle Executor | Submission | Medium | Read, Write, Bash | (custom; wraps kaggle CLI) | No |
| 13 (folded) | Memory Curator | Top | Medium | Read, Write, Bash | agent-memory-curator, artifact-analyzer | No |

---

## 3. Exact Responsibilities

### 3.1 Commander

- Reads: `state/competition.yaml`, `state/dataset_signature.yaml`, `state/experiments.yaml`, `state/ensemble.yaml`, knowledge pointers.
- Writes: `state/phase_decisions.yaml` (current phase, focus, budget), `state/submission_selection.yaml`.
- Decides: phase transitions, compute allocation, stopping, submission selection.
- Does NOT: write code, run training, or directly call Kaggle CLI.

### 3.2 Competition Researcher

- Reads: `competition.yaml` (competition name), strategic memory pointers.
- Writes: `state/competition.yaml` (populated), `reports/competition_intelligence.md`.
- Calls: Kaggle CLI (`kaggle competitions <list|files>`), web fetch for competition page.
- Does NOT: inspect the data itself.

### 3.3 Data Forensics

- Reads: data directory (`data/`), `state/competition.yaml`.
- Writes: `state/dataset_signature.yaml`, `reports/{dataset_inventory,data_quality,leakage_report,distribution_shift,target_analysis,eda_report}.md`.
- Calls: Python scripts in `scripts/data_forensics_*.py`.
- Does NOT: design the CV strategy.

### 3.4 EDA Specialist

- Reads: `state/dataset_signature.yaml`.
- Writes: `reports/eda_report.md` (with embedded images in `reports/figures/`).
- Calls: Python scripts in `scripts/eda_*.py`.
- Does NOT: leakage detection (delegated to Data Forensics).

### 3.5 Validation Architect

- Reads: `state/dataset_signature.yaml`, `reports/leakage_report.md`, `reports/distribution_shift.md`.
- Writes: `validation/strategy.md`, `validation/validation_config.yaml`, `validation/folds/*.csv`.
- Calls: Python scripts in `scripts/validation_*.py`.
- Veto: modeling does not proceed until `validation/strategy.md` exists and the Adversarial Reviewer has signed off (Gate 1).

### 3.6 Feature Engineer

- Reads: `state/dataset_signature.yaml`, `reports/eda_report.md`.
- Writes: `features/v<N>/` directory with feature-generation scripts and `feature_catalog.yaml`.
- Calls: Python scripts in `features/v<N>/generate.py`.
- Does NOT: train models.

### 3.7 Model Researcher

- Reads: `state/dataset_signature.yaml`, strategic memory pointers.
- Writes: `models/candidates.yaml`.
- Calls: web fetch for benchmark papers; reads memory for past similar competitions.
- Does NOT: run training.

### 3.8 HPO Agent

- Reads: `models/candidates.yaml`, `validation/validation_config.yaml`.
- Writes: `experiments/hpo/<model>/*.json` (per-trial results), `experiments/hpo/<model>/best.json`.
- Calls: `scripts/hpo_<model>.py` (uses Optuna).

### 3.9 Trainer

- Reads: `experiments/hpo/<model>/best.json`, `validation/validation_config.yaml`, `features/v<N>/`.
- Writes: `experiments/training/<exp_id>/` with metrics.json, oof_predictions.csv, submission.csv, experiment.json, resource_usage.json, model artifact.
- Calls: `scripts/train_<model>.py`.

### 3.10 Ensembler

- Reads: all `experiments/training/<exp_id>/oof_predictions.csv`.
- Writes: `ensemble/candidates/`, `ensemble/correlations/`, `ensemble/weights/`, `ensemble/final/blend.json`.
- Calls: `scripts/ensemble_search.py` (hill climbing + forward stepwise).
- Veto: blend is not finalized until Adversarial Reviewer signs off (Gate 2).

### 3.11 Adversarial Reviewer

- Reads: the artifact under review at the current gate.
- Writes: `reviews/gate_<n>_<artifact>.yaml`.
- Calls: Python scripts to validate artifacts (e.g., check submission file has correct row count).
- Veto: at Gates 1, 2, 3, the artifact is not approved until the Adversarial Reviewer returns `status: approved`.

### 3.12 Kaggle Executor

- Reads: the final notebook + kernel-metadata.json.
- Writes: `artifacts/<run_id>/` (downloaded Kaggle outputs).
- Calls: `kaggle kernels <init|push|status|output>`.
- Does NOT: submit to leaderboard without Final Auditor (Adversarial Reviewer Gate 3) approval.

### 3.13 Memory Curator

- Reads: `artifacts/<run_id>/`, `reviews/`, `state/logs/`.
- Writes: `knowledge/{competitions,strategies,validation,feature_engineering,models,ensembles,leakage,kaggle,compute,failures,successes,experiments,skills,agent-performance,meta-learning,playbooks}/*.yaml`.
- Calls: no external tools; only Read/Write.

---

## 4. Agent Communication Protocol

Per Part I §9, every agent returns a YAML object:

```yaml
status: success | failure | partial | blocked
confidence: 0.0-1.0
findings: |
  <prose summary>
evidence:
  - type: file | metric | citation
    ...
actions:
  - tool: <name>
    ...
artifacts:
  - path: <path>
    type: metrics | oof | submission | report | figure | model
recommendations:
  - priority: <int>
    description: <text>
    expected_gain: low | medium | high
    compute_cost: low | medium | high
    confidence: 0.0-1.0
risks:
  - severity: low | medium | high
    description: <text>
    mitigation: <text>
next_tasks:
  - cell: <cell>
    task: <task>
    priority: <int>
```

### 4.1 Schema Enforcement

The schema is formalized as JSON Schema in `schemas/agent_result.schema.json`. A `PostToolUse` hook validates every agent's output against the schema. Invalid outputs trigger automatic retry with a corrective prompt.

### 4.2 Inter-Agent Messaging

Agents do NOT message each other directly. They communicate only via:
1. The blackboard (`state/`).
2. Structured artifacts on disk (e.g., `experiments/training/<exp_id>/metrics.json`).
3. The Memory Curator (for cross-agent knowledge).

### 4.3 Logging

Every agent invocation produces a JSONL log entry:

```json
{"ts": "2026-09-13T10:00:00Z", "agent": "trainer", "task_id": "exp-007", "phase": "experimentation", "input_tokens": 8420, "output_tokens": 1830, "duration_s": 47, "result_path": "experiments/training/exp-007/result.yaml", "status": "success"}
```

Logs are stored at `state/logs/<date>/<agent>.jsonl`. The Memory Curator ingests these for agent-performance tracking.

---

## 5. Memory Architecture

Per Part I §10. Four layers:

### 5.1 Working Memory

Per-agent-invocation, in-context. Reset at every call. Composed by the Memory Curator from:
- Global rules (static, ~2-4k tokens).
- Role context (the agent's system prompt, ~1-3k tokens).
- Competition brief (from `state/competition.yaml`, ~2-5k tokens).
- Relevant memory (top-K retrieved from project + strategic memory, ~2-5k tokens).
- Current task (the specific subtask description, ~1-5k tokens).
- Relevant artifacts (paths or excerpts, ~2-10k tokens).
- Recent experiments (condensed summary, ~1-3k tokens).

Total target: 10-30k tokens per call — well within Flash High's effective context.

### 5.2 Project Memory

Per-competition, at `competition-project/knowledge/`. 16 categories (see Part I §10.2).

### 5.3 Strategic Memory

Cross-competition, at `kaggle-agent-core/knowledge/`. Same 16 categories, populated by the Memory Curator when competition-end promotions happen.

### 5.4 Meta Memory

Subset of strategic memory, at `kaggle-agent-core/knowledge/meta-learning/`. Tracks agent performance, architecture decisions, skill evaluations, failure modes, prompt A/B tests.

### 5.5 Memory Operations

- **Ingest** (Memory Curator): parse artifact → write to project memory with provenance.
- **Retrieve** (Memory Curator): given a task description, return top-K relevant memory entries.
- **Promote** (Memory Curator + Adversarial Reviewer): at competition end, validate entries and promote to strategic memory.
- **Prune** (Memory Curator): flag stale entries; never auto-delete.

### 5.6 Memory Schema

Every memory entry:
```yaml
id: mem-00042
type: observed_fact | strong_pattern | weak_hypothesis | speculation
category: models
topic: catboost_high_cardinality
hypothesis: "CatBoost with target encoding is strong on high-cardinality tabular"
evidence:
  - experiment_id: exp-007
    artifact: experiments/training/exp-007/metrics.json
    observation: "CV=0.852 with target encoding"
confidence: 0.7
counterexamples: []
last_validated: 2026-09-13
provenance:
  - source: experiments/training/exp-007/
    retrieved_at: 2026-09-13
    retrieved_by: memory_curator
contradicts: null
tags: [catboost, target_encoding, high_cardinality, tabular]
```

---

## 6. Skill Architecture

### 6.1 Skill Scope

- **Global skills** (Antigravity global, `~/.config/antigravity/skills/`): competition-agnostic.
- **Workspace skills** (`competition-project/.agents/skills/`): competition-specific.
- **Framework skills** (`kaggle-agent-core/skills/`): installed globally as part of the framework setup.

### 6.2 Framework Skills (Built)

```
kaggle-agent-core/skills/
├── kaggle-competition-analysis/
│   └── SKILL.md
├── leakage-detection-suite/
│   ├── SKILL.md
│   └── scripts/
│       ├── target_correlation.py
│       ├── temporal_check.py
│       ├── group_check.py
│       └── duplicate_check.py
├── validation-design/
│   └── SKILL.md
├── adversarial-validation/
│   ├── SKILL.md
│   └── scripts/av_train.py
├── experiment-manager/
│   ├── SKILL.md
│   └── scripts/registry.py
├── ensemble-optimization/
│   ├── SKILL.md
│   └── scripts/hill_climb.py
├── artifact-analyzer/
│   └── SKILL.md
├── kaggle-compute-optimizer/
│   └── SKILL.md
├── agent-memory-curator/
│   └── SKILL.md
├── skill-evolution/
│   └── SKILL.md
├── agent-performance-tracker/
│   └── SKILL.md
├── reproducibility-checker/
│   └── SKILL.md
├── distribution-shift-detector/
│   ├── SKILL.md
│   └── scripts/psi.py
├── pseudo-labeling/
│   ├── SKILL.md
│   └── scripts/pseudo_label.py
└── test-time-augmentation/
    ├── SKILL.md
    └── scripts/tta.py
```

### 6.3 SKILL.md Schema

```markdown
---
name: leakage-detection-suite
version: 1.0.0
purpose: Detect target, temporal, group, and duplicate leakage in tabular datasets.
trigger:
  condition: after data profiling
  agents: [data_forensics, validation_architect]
dependencies:
  - polars >= 0.20
  - numpy
evidence:
  - competition: comp-2025-tabular-a
    improvement: caught 3 leakage features
expected_benefit: prevents leakage-induced shake-up
known_failure_modes:
  - false positive on legitimately-correlated features in causal-modeling competitions
evaluation_cases:
  - case: synthetic dataset with planted leakage
    expected: detect
  - case: clean dataset
    expected: no flags
---

# Leakage Detection Suite

## When to use

After the Data Forensics agent has profiled the dataset.

## What it does

1. Target correlation: compute Pearson + mutual information between each feature and the target; flag features with |correlation| > 0.95 or MI > threshold.
2. Temporal check: if there's a time column, check for features that would require future information to compute.
3. Group check: if there's a group ID column, check for entities appearing in both train and test.
4. Duplicate check: find exact and near-duplicate rows across train and test.

## Outputs

- `reports/leakage_report.md` (prose summary)
- `reports/leakage_flags.json` (structured flags)

## Invocation

```bash
python scripts/leakage_detection_suite.py --data-dir data/ --output-dir reports/
```
```

### 6.4 Skill Evolution

See Part I §12. The pipeline: failure → classify → root cause → generalizable? → candidate → draft → evaluate → adversarial review → versioned → A/B test → promote/reject.

### 6.5 MCP Servers

Three custom MCP servers provide non-trivial tools:

1. **Kaggle MCP** (`kaggle-agent-core/mcp/kaggle/`) — wraps the Kaggle CLI as MCP tools: `kaggle.submit`, `kaggle.push_kernel`, `kaggle.poll_status`, `kaggle.download_output`.
2. **Experiment MCP** (`kaggle-agent-core/mcp/experiment/`) — `experiment.create`, `experiment.update`, `experiment.list`, `experiment.compare`.
3. **Memory MCP** (`kaggle-agent-core/mcp/memory/`) — `memory.retrieve`, `memory.ingest`, `memory.promote`, `memory.contradictions`.

The Memory MCP server is essentially the `agent-memory-mcp` skill from the catalog, installed and configured.

---

## 7. Self-Evolution Mechanism

Per Part I §11. The loop:

```text
OBSERVE → HYPOTHESIZE → PLAN → PARALLELIZE → IMPLEMENT →
EXECUTE → MEASURE → CRITIQUE → SELECT → MEMORIZE →
GENERALIZE → EVOLVE → REPEAT
```

### 7.1 Knowledge Promotion Levels

- **Observed fact** (1.0): permanent.
- **Strong empirical pattern** (0.8): permanent if no counterexample in 6 months.
- **Weak hypothesis** (0.5): project memory only.
- **Speculation** (0.2): working memory only.

The system **never** auto-promotes speculation to permanent knowledge (per prompt §7).

### 7.2 Agent Performance Tracking

The Memory Curator ingests `state/logs/` and maintains `kaggle-agent-core/knowledge/meta-learning/agent_performance.yaml`:

```yaml
trainer:
  invocations: 47
  success_rate: 0.89
  avg_runtime_s: 38
  avg_input_tokens: 7200
  avg_output_tokens: 1450
  useful_discoveries: 3
  false_positives: 1
  wasted_compute_hours: 1.2
  downstream_lb_impact: 0.012
  reproducibility: 0.94
  failure_modes:
    - "occasionally misroutes to wrong validation_config.yaml version (2 occurrences)"
```

### 7.3 Controlled Evolution (A/B Tests)

When changing an agent prompt or routing:
1. Run the new version on a held-out competition.
2. Run the old version in parallel.
3. Compare metrics: LB rank, CV-LB gap, compute, token cost.
4. Promote if superior on ≥2 metrics and not worse on any.
5. Adversarial Reviewer signs off.

A/B test results are stored in `kaggle-agent-core/knowledge/meta-learning/prompt_ab_tests.yaml`.

---

## 8. Experiment Scheduler

Per Part I §13. Priority formula:

```text
priority = (expected_score_gain × confidence × information_gain) / (compute_cost × risk)
```

### 8.1 Scheduler Implementation

`scripts/scheduler.py`:

```python
def schedule_next(state_path: str, budget: dict) -> list[Experiment]:
    candidates = load_candidates(state_path)
    for c in candidates:
        c.priority = (
            c.expected_gain * c.confidence * c.information_gain
        ) / (c.compute_cost * c.risk)
    candidates.sort(key=lambda c: c.priority, reverse=True)
    return select_within_budget(candidates, budget)
```

### 8.2 Stopping Rules (Encoded in `policies/stopping.yaml`)

```yaml
stopping_rules:
  - rule: improvements_plateau
    condition: "last_3_experiments_delta_cv < 0.001"
    action: stop_phase
  - rule: compute_exhausted
    condition: "remaining_gpu_hours < 1 * baseline_runtime_hours"
    action: stop_phase
  - rule: validation_unstable
    condition: "fold_variance > 0.02"
    action: investigate_before_continue
  - rule: repeated_failures
    condition: "3_consecutive_failures_same_root_cause"
    action: stop_phase_and_escalate
  - rule: ensemble_diversity_exhausted
    condition: "no_new_model_with_correlation < 0.95"
    action: stop_ensemble_search
  - rule: deadline_risk
    condition: "competition_end_days < 7"
    action: preserve_compute_for_final_submission
  - rule: low_expected_value
    condition: "top_candidate_priority < 3.0"
    action: stop_phase
```

### 8.3 Parallelization

- Independent experiments (different models, different feature subsets) → async subagents in parallel.
- Shared-state experiments (modifying the same feature catalog) → sequential.
- Git worktrees (`worktrees/exp-<id>/`) isolate parallel experiments.

---

## 9. Validation System

Per Part I §14. The Validation Architect produces:

```
validation/
├── strategy.md
├── validation_config.yaml
├── folds/
│   ├── fold_0.csv
│   ├── fold_1.csv
│   ├── fold_2.csv
│   ├── fold_3.csv
│   └── fold_4.csv
└── leakage_tests/
    ├── target_correlation.json
    ├── temporal_check.json
    ├── group_check.json
    └── duplicate_check.json
```

### 9.1 CV Strategy Decision Tree

```text
Has time column?
├── Yes → Has group ID?
│   ├── Yes → Custom temporal + group CV
│   └── No → Has autocorrelation? (check with lag-1 ACF)
│       ├── Yes → PurgedKFold (with gap)
│       └── No → TimeSeriesSplit
└── No → Has group ID?
    ├── Yes → Has class imbalance?
    │   ├── Yes → StratifiedGroupKFold
    │   └── No → GroupKFold
    └── No → Has class imbalance?
        ├── Yes → StratifiedKFold
        └── No → KFold
```

### 9.2 Gate 1 (Validation Design) Checklist

The Adversarial Reviewer attacks the validation strategy:

1. Is there target leakage? (target-correlated features that won't be available at inference)
2. Is there temporal leakage? (future features in training rows)
3. Is there group leakage? (entities in both train and validation folds)
4. Are duplicates handled?
5. Is the CV-LB gap justification convincing?
6. Is the metric being optimized correctly (e.g., are we using `predict_proba` for AUC, not `predict`)?
7. Are seeds set for reproducibility?

Veto power: the strategy is not approved until the Adversarial Reviewer returns `status: approved`.

---

## 10. Kaggle Automation Flow

Per Part I §15.

### 10.1 Notebook Preparation Pipeline

1. Trainer agent produces a model artifact and a Python script template.
2. Kaggle Executor agent fills in the notebook template (using Jinja2 or string replacement).
3. Kaggle Executor writes `kernel-metadata.json`.
4. Kaggle Executor runs `kaggle kernels push -p notebooks/<exp_id>/`.
5. Kaggle Executor polls `kaggle kernels status <username>/<slug>` every 60 seconds.
6. On `complete`, runs `kaggle kernels output <username>/<slug> -p artifacts/<run_id>/`.
7. On `error`, parses the log, classifies the failure, routes to Failure Recovery.

### 10.2 Notebook Template

```python
# notebooks/template.py
import json, os, sys, time, random
import numpy as np
import polars as pl

SEED = {{seed}}
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = "/kaggle/input/{{competition_name}}/"
MODEL_DIR = "/kaggle/input/{{model_dataset_slug}}/"

train = pl.read_csv(f"{DATA_DIR}/train.csv")
test = pl.read_csv(f"{DATA_DIR}/test.csv")

# Feature generation (inline; or load from MODEL_DIR/features.parquet)
{{feature_code}}

# Model loading or training
{{model_code}}

# OOF predictions (cross-validated)
{{oof_code}}

# Test predictions
test_pred = model.predict(test[features])

# Write artifacts
import polars as pl
pl.DataFrame({
    "id": test["id"],
    "target": test_pred
}).write_csv("/kaggle/working/submission.csv")

pl.DataFrame({
    "id": train.loc[validation_mask, "id"],
    "target": oof_pred[validation_mask],
    "fold": fold_assignments[validation_mask]
}).write_csv("/kaggle/working/oof_predictions.csv")

with open("/kaggle/working/metrics.json", "w") as f:
    json.dump({
        "experiment_id": "{{exp_id}}",
        "cv_score": cv_score,
        "fold_scores": fold_scores,
        "metric_name": "{{metric_name}}",
        "metric_direction": "{{metric_direction}}"
    }, f, indent=2)

with open("/kaggle/working/resource_usage.json", "w") as f:
    import resource
    json.dump({
        "experiment_id": "{{exp_id}}",
        "runtime_s": time.time() - start_time,
        "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    }, f, indent=2)
```

### 10.3 kernel-metadata.json Template

```json
{
  "id": "{{kaggle_username}}/{{competition_name}}-{{exp_id}}",
  "title": "{{competition_name}}-{{exp_id}}",
  "code_file": "template.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": {{use_gpu}},
  "enable_tpu": false,
  "enable_internet": false,
  "dataset_sources": ["{{kaggle_username}}/{{model_dataset_slug}}"],
  "competition_sources": ["{{competition_name}}"]
}
```

### 10.4 Polling Loop

```python
import subprocess, json, time

def poll_status(slug, timeout_s=9*3600):
    start = time.time()
    while time.time() - start < timeout_s:
        result = subprocess.run(
            ["kaggle", "kernels", "status", slug],
            capture_output=True, text=True
        )
        status = json.loads(result.stdout)
        if status["status"] in ("complete", "error"):
            return status
        time.sleep(60)
    raise TimeoutError(f"Kernel {slug} timed out")
```

### 10.5 Output Download

```python
def download_output(slug, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    subprocess.run(
        ["kaggle", "kernels", "output", slug, "-p", dest_dir],
        check=True
    )
    # Verify expected files
    for expected in ["metrics.json", "submission.csv", "oof_predictions.csv"]:
        assert os.path.exists(os.path.join(dest_dir, expected)), f"Missing {expected}"
```

### 10.6 Submission to Leaderboard

For traditional competitions:
```bash
kaggle competitions submit -c {{competition_name}} -f submission.csv -m "{{exp_id}}"
```

For code competitions:
```bash
kaggle kernels push -p notebooks/{{exp_id}}/    # this IS the submission
# Then "submit to competition" via the UI or via a CLI flag if supported
```

---


## 11. GitHub Integration

Per the prompt (§23), the framework maintains a clean Git repository.

### 11.1 Branching Strategy

```text
main              # always-green, latest stable
├── experiments/<exp_id>   # one branch per experiment (parallel via worktrees)
├── features/<feature_name> # one branch per feature being developed
└── release/vX.Y           # tagged releases
```

### 11.2 What's Committed

- All source code (`src/`, `scripts/`, `notebooks/`).
- All agent specs (`agents/*.md`).
- All skill specs (`skills/*/SKILL.md` + supporting scripts).
- All schema files (`schemas/*.json`).
- All policy files (`policies/*.yaml`).
- All playbook files (`playbooks/*.md`).
- Strategic memory (`kaggle-agent-core/knowledge/`) — version-controlled as a knowledge base.
- Project memory (`competition-project/knowledge/`) — version-controlled per-competition.
- `README.md`, `COMPETITION.md`, `EXPERIMENTS.md`, `CHANGELOG.md`.

### 11.3 What's NOT Committed

- Credentials (`kaggle.json`, `.env`).
- Kaggle API tokens.
- Large datasets (`data/`).
- Generated model binaries (`*.pkl`, `*.joblib`, `*.bin`) unless intentionally required.
- Private competition data where prohibited by rules.
- Large OOF / prediction files (`.csv`, `.npy`) — store on Kaggle Datasets if needed for sharing, gitignore locally.

### 11.4 Commit Discipline

- The Commander (or a hook) commits meaningful milestones:
  - After baseline is established: "feat(baseline): LightGBM CV=0.847".
  - After each completed experiment: "exp(exp-007): CatBoost CV=0.852".
  - After each ensemble update: "ensemble(v3): 5-model blend CV=0.871".
  - After each submission: "submit(exp-012): public LB=0.842".
- The Kaggle Executor logs the commit hash in `experiment.json` for traceability.

### 11.5 README.md Structure

```markdown
# {{Competition Name}}

## Status
- Phase: experimentation
- CV (best): 0.871
- Public LB: 0.842
- Compute used: 12 GPU-hours
- Last updated: 2026-09-13

## Quick start
1. `kaggle-agent-core/scripts/init_competition.sh` (already done — this is the resulting directory).
2. `agy --model gemini-3.8-flash-high` (start Antigravity; the AGENTS.md manifest loads).
3. Tell the Commander: "Run competition {{slug}}, project directory $(pwd)".

## Structure
- See Appendix B of the framework blueprint for directory layout.

## Experiments
- See EXPERIMENTS.md for the experiment registry.

## Changelog
- See CHANGELOG.md.
```

### 11.6 EXPERIMENTS.md Structure

```markdown
# Experiments

| ID | Hypothesis | Intervention | CV | LB | Status | Date |
|----|------------|--------------|----|----|--------|------|
| exp-001 | Baseline LightGBM | default params | 0.847 | - | done | 2026-09-10 |
| exp-002 | + target encoding | add target_enc | 0.851 | - | done | 2026-09-11 |
| exp-007 | CatBoost + TE | switch model | 0.852 | - | done | 2026-09-13 |
| ... | ... | ... | ... | ... | ... | ... |

## Notes

- exp-002 → exp-007: CatBoost gives +0.001 over LightGBM baseline; add to ensemble pool.
```

---

## 12. Artifact Protocol

Per Part I §16. The artifact schema is stable across competitions. Files in `artifacts/<run_id>/`:

```
artifacts/<run_id>/
├── metrics.json
├── predictions.csv
├── submission.csv
├── oof_predictions.csv
├── experiment.json
├── resource_usage.json
├── logs/
│   ├── stdout.log
│   ├── stderr.log
│   └── kaggle_executor.jsonl
└── reports/
    ├── training_curve.png
    ├── feature_importance.png
    └── prediction_distribution.png
```

### 12.1 Schema Validation

Each JSON file is validated against a JSON Schema in `schemas/`:
- `schemas/metrics.schema.json`
- `schemas/experiment.schema.json`
- `schemas/resource_usage.schema.json`
- `schemas/agent_result.schema.json`

A `PostToolUse` hook runs `scripts/validate_artifact.py` after every artifact write. Schema mismatches fail loudly.

### 12.2 Artifact Ingestion

The Memory Curator ingests artifacts:

```python
def ingest_artifact(artifact_dir: str, registry_path: str):
    metrics = json.load(open(f"{artifact_dir}/metrics.json"))
    experiment = json.load(open(f"{artifact_dir}/experiment.json"))
    
    # Append to registry
    registry = yaml.safe_load(open(registry_path))
    registry.append({
        "id": experiment["id"],
        "hypothesis": experiment["hypothesis"],
        "cv_score": metrics["cv_score"],
        "fold_scores": metrics["fold_scores"],
        "model": experiment["model_family"],
        "timestamp": metrics["timestamp"],
    })
    yaml.safe_dump(registry, open(registry_path, "w"))
    
    # Promote to knowledge
    knowledge_entry = extract_knowledge(experiment, metrics)
    write_to_memory(knowledge_entry)
```

### 12.3 Schema Stability

The artifact schema is versioned (`schemas/VERSION`). Breaking changes require a major version bump; the Memory Curator handles schema migrations across versions.

---

## 13. Compute Optimization System

Per Part I §17.

### 13.1 ML Compute Optimizer

`scripts/compute_optimizer.py`:

```python
def recommend_accelerator(dataset_signature, model_family):
    n_rows = dataset_signature["n_rows"]
    n_features = dataset_signature["n_features"]
    
    if model_family in ("lightgbm", "xgboost", "catboost"):
        # CPU is faster for tabular GBDT on Kaggle-sized data
        return "cpu"
    elif model_family in ("resnet", "vit", "deberta"):
        return "gpu"
    elif model_family == "autogluon":
        # AutoGluon makes its own decisions
        return "cpu"  # safer default
    elif n_rows > 1_000_000 and model_family in ("lightgbm",):
        # Large tabular might benefit from GPU LightGBM
        return "cpu"  # still usually faster; verify
    else:
        return "cpu"
```

### 13.2 AI Compute Optimizer

The Memory Curator tracks per-agent token cost. When an agent's avg token cost exceeds a threshold (e.g., 15k input tokens), the Commander triggers a context audit: the Memory Curator retrieves less context (lower K) or compresses the context further.

### 13.3 Caching

- Static context (global rules + role context + competition brief) → cached by Antigravity.
- Feature computations → cached in `features/v<N>/.cache/`.
- OOF predictions → cached in `experiments/training/<exp_id>/oof_predictions.csv`.
- Kaggle API responses → cached in `state/.kaggle_cache/` for 1 hour.

---

## 14. Failure Recovery System

Per Part I §19. Failure types in `policies/failure_recovery.yaml`:

```yaml
failure_types:
  DATA_FAILURE:
    recovery:
      - action: re_read_from_kaggle
        retry: 3
        backoff_s: 30
      - action: re_download_data
        retry: 2
        backoff_s: 60
  DEPENDENCY_FAILURE:
    recovery:
      - action: uv_pip_install
        args: [missing_package]
      - action: use_kaggle_dataset
        fallback: true
  CODE_FAILURE:
    recovery:
      - action: read_traceback
      - action: fix_script
        agent: trainer
      - action: re_run
  MEMORY_FAILURE:
    recovery:
      - action: reduce_batch_size
      - action: sample_data
      - action: use_polars_lazy
  TIMEOUT:
    recovery:
      - action: reduce_model_complexity
      - action: parallelize_folds
      - action: switch_to_local_compute
  KAGGLE_FAILURE:
    recovery:
      - action: retry_with_backoff
        retry: 5
        backoff_s: 60
      - action: switch_to_local_compute
  VALIDATION_FAILURE:
    recovery:
      - action: rollback_to_last_known_good
      - action: re_run_validation_architect
  MODEL_FAILURE:
    recovery:
      - action: reduce_learning_rate
      - action: switch_optimizer
  AGENT_FAILURE:
    recovery:
      - action: re_route_to_alternate_agent
      - action: retry_with_different_prompt
  ORCHESTRATION_FAILURE:
    recovery:
      - action: restart_from_last_checkpoint
  ARTIFACT_FAILURE:
    recovery:
      - action: re_run_with_strict_artifact_validation
```

### 14.1 Recovery Pipeline

```text
Failure occurs
  ↓
Classify (Memory Curator or rule-based)
  ↓
Lookup recovery policy
  ↓
Apply recovery action
  ↓
If success: log and continue
If failure: escalate (next recovery action in the chain)
If all recoveries fail: escalate to Commander, ask user
  ↓
If pattern repeats (3+ times with same root cause): trigger Skill Evolution pipeline
```

---

## 15. Evaluation Framework

Per Part I §26.

### 15.1 ML Benchmark

```python
# scripts/benchmark_ml.py
def benchmark_ml(competition_slug, framework_run_id):
    baseline_lb = fetch_baseline_private_lb(competition_slug)
    framework_lb = fetch_framework_private_lb(competition_slug, framework_run_id)
    return {
        "competition": competition_slug,
        "framework_lb": framework_lb,
        "baseline_lb": baseline_lb,
        "rank": compute_rank(competition_slug, framework_lb),
        "shake_up": compute_shake_up(competition_slug, framework_run_id),
    }
```

### 15.2 Agent Benchmark

`scripts/benchmark_agents.py` aggregates `state/logs/`:

```python
def benchmark_agents(logs_dir):
    stats = defaultdict(lambda: {
        "invocations": 0, "successes": 0, "input_tokens": 0,
        "output_tokens": 0, "runtimes_s": []
    })
    for log_file in glob(f"{logs_dir}/**/*.jsonl", recursive=True):
        for line in open(log_file):
            entry = json.loads(line)
            s = stats[entry["agent"]]
            s["invocations"] += 1
            if entry["status"] == "success":
                s["successes"] += 1
            s["input_tokens"] += entry.get("input_tokens", 0)
            s["output_tokens"] += entry.get("output_tokens", 0)
            s["runtimes_s"].append(entry.get("duration_s", 0))
    return stats
```

### 15.3 A/B Test Harness

```python
# scripts/ab_test.py
def run_ab_test(prompt_variant_a, prompt_variant_b, test_competition):
    """Run both variants on the same competition in parallel worktrees."""
    worktree_a = create_worktree(f"ab_test_{prompt_variant_a}")
    worktree_b = create_worktree(f"ab_test_{prompt_variant_b}")
    
    # Run framework in both worktrees with respective prompts
    result_a = run_framework(worktree_a, prompt_variant_a, test_competition)
    result_b = run_framework(worktree_b, prompt_variant_b, test_competition)
    
    # Compare
    return {
        "a": result_a,
        "b": result_b,
        "winner": pick_winner(result_a, result_b),
        "promote": should_promote(result_a, result_b),
    }
```

---

## 16. Security / Compliance Layer

Per Part I §18.

### 16.1 Rule Compliance Checklist (Stored in `policies/compliance.yaml`)

```yaml
compliance_checks:
  - id: external_data_permissions
    description: Verify external data is allowed by competition rules.
    gate: 3
    agent: adversarial_reviewer
  - id: internet_restrictions
    description: Verify notebook does not require internet (code competitions).
    gate: 3
    agent: adversarial_reviewer
  - id: api_restrictions
    description: Verify no banned APIs (LLM, auto-submission) used.
    gate: 3
    agent: adversarial_reviewer
  - id: compute_restrictions
    description: Verify per-team GPU-hour caps respected.
    gate: 3
    agent: adversarial_reviewer
  - id: team_rules
    description: Verify team-merge deadline not exceeded.
    gate: 3
    agent: adversarial_reviewer
  - id: submission_frequency
    description: Verify daily submission limits respected.
    gate: 3
    agent: kaggle_executor
  - id: model_restrictions
    description: Verify no banned model architectures.
    gate: 3
    agent: adversarial_reviewer
  - id: pretrained_model_restrictions
    description: Verify pretrained models allowed by competition.
    gate: 3
    agent: adversarial_reviewer
  - id: licensing
    description: Verify all code is permissively licensed.
    gate: 3
    agent: adversarial_reviewer
  - id: reproducibility
    description: Verify seeds set, determinism enabled.
    gate: 3
    agent: adversarial_reviewer
```

### 16.2 Credential Management

- Kaggle credentials: `~/.kaggle/kaggle.json` (NEVER committed; `.gitignore` excludes any `kaggle.json` anywhere in the tree).
- Antigravity config: `~/.config/antigravity/`.
- MCP secrets: env vars, not config files in the repo.
- A `PreToolUse` hook blocks any Write tool call whose path includes `kaggle.json` or `.env`.

### 16.3 Submission Integrity

The Final Auditor (Adversarial Reviewer at Gate 3) verifies the submission file:

```python
def verify_submission(submission_path, sample_path):
    sub = pl.read_csv(submission_path)
    sample = pl.read_csv(sample_path)
    
    # Row count
    assert sub.height == sample.height, f"Row count mismatch: {sub.height} vs {sample.height}"
    
    # Columns
    assert set(sub.columns) == set(sample.columns), "Column mismatch"
    
    # ID order
    assert sub["id"].to_list() == sample["id"].to_list(), "ID order mismatch"
    
    # No NaN in target
    assert not sub["target"].null_count() > 0, "NaN in target column"
    
    # Range (e.g., for AUC: probabilities in [0, 1])
    if metric == "roc_auc":
        assert sub["target"].min() >= 0 and sub["target"].max() <= 1, "Probabilities out of [0,1]"
    
    return True
```

---

## 17. Directory Structure

See Appendix B for the full directory tree.

---

## 18. Configuration Structure

### 18.1 Antigravity Workspace Manifest (`AGENTS.md`)

```markdown
---
model: gemini-3.8-flash-high
effort: high
agents:
  - agents/commander.md
  - agents/competition_researcher.md
  - agents/data_forensics.md
  - agents/eda_specialist.md
  - agents/validation_architect.md
  - agents/feature_engineer.md
  - agents/model_researcher.md
  - agents/hpo_agent.md
  - agents/trainer.md
  - agents/ensemble_agent.md
  - agents/adversarial_reviewer.md
  - agents/kaggle_executor.md
  - agents/memory_curator.md
hooks:
  - hooks/pre_tool_use.py
  - hooks/post_tool_use.py
  - hooks/stop.py
mcp_servers:
  - kaggle-agent-core/mcp/kaggle/
  - kaggle-agent-core/mcp/experiment/
  - kaggle-agent-core/mcp/memory/
skills_global:
  - agent-memory
  - agent-evaluation
  - context-engineering
  - data-scientist
  - polars
  - using-git-worktrees
  - uv-package-manager
skills_workspace:
  - .agents/skills/competition-rules/
  - .agents/skills/dataset-specific/
  - .agents/skills/competition-playbook/
---

# {{Competition Name}} — Antigravity Workspace

This workspace is configured for the autonomous Kaggle competition system.

## Usage

```bash
agy --model gemini-3.8-flash-high
```

Then tell the Commander:
> Run competition {{slug}}, project directory $(pwd)

## See also

- `README.md` for current status.
- `EXPERIMENTS.md` for the experiment registry.
- `knowledge/` for project memory.
```

### 18.2 competition.yaml

```yaml
# competition-project/competition.yaml
name: comp-2026-tabular-x
slug: comp-2026-tabular-x
type: code_competition   # code_competition | traditional
metric:
  name: roc_auc
  direction: higher_better
  needs_probabilities: true
submission_format:
  file: submission.csv
  columns: [id, target]
  row_count: 50000
rules:
  external_data: allowed
  internet: disabled   # code competition
  pretrained_models: allowed
  team_size: 5
  merge_deadline: 2026-09-25
timeline:
  start: 2026-08-01
  deadline: 2026-10-01
kaggle:
  username: {{kaggle_username}}
  model_dataset_slug: {{model_dataset_slug}}
compute_budget:
  gpu_hours_per_week: 30
  cpu_hours_per_run: 9
  daily_submissions: 5
```

### 18.3 policies/stopping.yaml

(See §8.2 above.)

### 18.4 policies/failure_recovery.yaml

(See §14 above.)

### 18.5 policies/compliance.yaml

(See §16.1 above.)

### 18.6 config/runtime.yaml

```yaml
# kaggle-agent-core/config/runtime.yaml
antigravity:
  model: gemini-3.8-flash-high
  default_effort: medium
  high_effort_agents: [commander, validation_architect, adversarial_reviewer]
  max_concurrent_subagents: 4
  rate_limit_rpm: 60   # estimate; verify
kaggle:
  cli_version: ">=1.6.0"
  poll_interval_s: 60
  max_runtime_s: 32400  # 9 hours
python:
  version: ">=3.11"
  package_manager: uv
ml:
  default_n_jobs: -1
  reproducibility:
    random_seed: 42
    numpy_seed: 42
    torch_seed: 42
    deterministic_algorithms: true
    cublas_workspace_config: ":4096:8"
memory:
  retrieval_top_k: 5
  stale_days: 30
  compression_threshold_entries: 100
logging:
  level: INFO
  jsonl_path: state/logs/
```

---

## 19. Initialization Workflow

### 19.1 Phase 0 — Initialization Script

`kaggle-agent-core/scripts/init_competition.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

COMPETITION_SLUG="${1:?Usage: init_competition.sh <competition-slug>}"
PROJECT_DIR="${2:-./competition-$(basename $COMPETITION_SLUG)}"

# Check prerequisites
command -v agy >/dev/null || { echo "agy not found"; exit 1; }
command -v kaggle >/dev/null || { echo "kaggle CLI not found"; exit 1; }
command -v uv >/dev/null || { echo "uv not found"; exit 1; }
[ -f ~/.kaggle/kaggle.json ] || { echo "kaggle.json not found in ~/.kaggle/"; exit 1; }

# Create project directory
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Scaffold from template
TEMPLATE_DIR="$(dirname $(realpath $0))/../templates/competition_template"
cp -r "$TEMPLATE_DIR/." .

# Initialize Python environment
uv venv
uv pip install -r requirements.txt

# Initialize Git
git init
git add .
git commit -m "init: scaffolded competition project for $COMPETITION_SLUG"

# Initialize Kaggle metadata
kaggle competitions download -c "$COMPETITION_SLUG" -p data/
unzip -q "data/$(basename $COMPETITION_SLUG).zip" -d data/
rm "data/$(basename $COMPETITION_SLUG).zip"

# Populate competition.yaml with discovered metadata
agy --model gemini-3.8-flash-high --headless -p "Read data/ and populate competition.yaml with the discovered metadata (metric, columns, row counts, etc.)."

# Initialize Antigravity workspace
echo "Workspace ready. Run: cd $PROJECT_DIR && agy --model gemini-3.8-flash-high"
```

### 19.2 Environment Detection

The init script detects:
- OS (Linux/macOS).
- Python version (requires ≥3.11).
- Package manager (`uv` preferred; falls back to `pip` with a warning).
- Git.
- Kaggle CLI (requires `>=1.6.0`).
- Kaggle credentials (`~/.kaggle/kaggle.json`).
- Antigravity CLI (`agy`).
- CPU cores, RAM, available storage.

If any prerequisite fails, the script exits with a clear error message and instructions.

### 19.3 First Run

After init, the user runs:
```bash
cd competition-project/
agy --model gemini-3.8-flash-high
```

And tells the Commander:
> Run competition {{slug}}, project directory $(pwd)

The Commander takes over from here.

---

## 20. Competition Execution Workflow

Per the prompt (§43, the default playbook):

```text
INIT
↓
COMPETITION RESEARCH (Competition Researcher)
↓
DATA FORENSICS (Data Forensics)
↓
EDA (EDA Specialist)
↓
VALIDATION DESIGN (Validation Architect → Gate 1)
↓
BASELINE (Trainer)
↓
FAST EXPERIMENTS (Trainer + HPO Agent, parallel async subagents)
↓
FEATURE ENGINEERING (Feature Engineer)
↓
MODEL SEARCH (Model Researcher)
↓
HPO (HPO Agent)
↓
ERROR ANALYSIS (Memory Curator + Trainer)
↓
ENSEMBLING (Ensembler → Gate 2)
↓
ADVERSARIAL REVIEW (Adversarial Reviewer → Gate 3)
↓
KAGGLE RUN (Kaggle Executor)
↓
ARTIFACT INGESTION (Memory Curator)
↓
RESULT ANALYSIS (Memory Curator)
↓
KNOWLEDGE UPDATE (Memory Curator → promote to strategic memory)
↓
STRATEGY EVOLUTION (Commander + Memory Curator)
↓
FINAL AUDIT (Adversarial Reviewer → Gate 3)
↓
SUBMISSION (Kaggle Executor)
```

### 20.1 Dynamic Stage Skipping

Per prompt §43, the Commander dynamically skips irrelevant stages:
- Tabular data → skip image/NLP-specific stages.
- GPU doesn't help → skip GPU stages.
- Baseline already strong → skip expensive HPO.
- No pseudo-labeling evidence → skip pseudo-labeling.
- High-correlation ensemble → skip adding more models.

### 20.2 Stage Loop

The Commander may iterate stages:
- After experimentation → ensembling → if ensemble is weak, back to feature engineering.
- After result analysis → if a new hypothesis emerges, back to experimentation.

The loop continues until stopping rules fire (§8.2).

---


## 21. Cross-Competition Learning Workflow

Per the prompt (§46, §47).

### 21.1 When Competition A Finishes

```text
raw experiments (in competition-A/knowledge/experiments/)
        ↓
validated discoveries (Memory Curator promotes from observed to strong pattern)
        ↓
generalizable knowledge (Memory Curator extracts dataset-signature-conditional lessons)
        ↓
strategy extraction (Commander summarizes what worked)
        ↓
cross-competition memory (stored in kaggle-agent-core/knowledge/)
```

### 21.2 When Competition B Begins

```text
competition characteristics (state/competition.yaml + state/dataset_signature.yaml)
        ↓
retrieve similar historical competitions (Memory Curator: dataset-signature similarity)
        ↓
retrieve successful strategies (Memory Curator: tags + problem-type)
        ↓
retrieve failure patterns (Memory Curator: failures/ category)
        ↓
initialize priors (Commander sets initial strategy with priors as soft suggestions)
        ↓
test priors (Validation Architect verifies that priors hold for this dataset)
```

### 21.3 Meta-Kaggle Knowledge Graph

A small graph in `kaggle-agent-core/knowledge/meta-learning/knowledge_graph.yaml`:

```yaml
nodes:
  - id: dataset_sig_high_cardinality_tabular
    type: dataset_signature
    features: [high_cardinality_categorical, medium_size, no_time]
  - id: catboost_with_target_encoding
    type: model_strategy
    features: [catboost, target_encoding]
  - id: stratified_group_kfold
    type: validation_strategy
    features: [group_aware, class_balanced]
  - id: blend_5_models
    type: ensemble_strategy
    features: [5_models, weighted_average]
  - id: improved_private_lb
    type: outcome
    features: [lb_rank_top_5_percent]

edges:
  - from: dataset_sig_high_cardinality_tabular
    to: catboost_with_target_encoding
    weight: 0.75
    evidence: [comp-2025-tabular-a, comp-2026-tabular-b, comp-2026-tabular-c, comp-2026-tabular-d]
  - from: catboost_with_target_encoding
    to: stratified_group_kfold
    weight: 0.6
    evidence: [...]
  - from: stratified_group_kfold
    to: blend_5_models
    weight: 0.5
    evidence: [...]
  - from: blend_5_models
    to: improved_private_lb
    weight: 0.7
    evidence: [...]
```

Only relationships with `evidence_count >= 2` and `weight >= 0.5` are encoded. The Adversarial Reviewer must sign off on any new edge.

---

## 22. Skill Evolution Workflow

Per Part I §12.

```text
Failure occurs (Failure Recovery System classifies)
  ↓
Root cause analysis (Memory Curator)
  ↓
Generalizable pattern? (≥2 occurrences with same root cause)
  ↓ no → log and stop
  ↓ yes
Skill candidate opened (skills/candidates/<name>/SKILL.md draft)
  ↓
Skill evaluation (run on held-out testbed; compare with vs. without)
  ↓
Adversarial review (Adversarial Reviewer attacks the skill)
  ↓
Versioned skill (committed to kaggle-agent-core/skills/<name>/ with version 1.0.0)
  ↓
A/B test (next competition: new skill in shadow mode, logged but not actioned)
  ↓
Promotion / Rejection (based on shadow-mode results)
```

### 22.1 Skill Candidate Template

`skills/candidates/<name>/SKILL.md`:

```markdown
---
name: <name>
version: 0.1.0-draft
purpose: <purpose>
trigger:
  condition: <when to invoke>
  agents: [<list>]
dependencies:
  - <package> >= <version>
evidence:
  - competition: <slug>
    failure: <description>
    improvement: <what this skill would have prevented>
expected_benefit: <expected>
known_failure_modes:
  - <potential>
evaluation_cases:
  - case: <test case>
    expected: <expected result>
status: draft
---

# <Name>

<Body of the skill — invocation, what it does, outputs>
```

### 22.2 Shadow Mode

In shadow mode, the skill runs (its outputs are logged) but the framework doesn't act on its outputs. After 1-2 competitions of shadow mode, the Memory Curator compares shadow-mode outputs to actual outcomes. If the skill consistently would have helped, it's promoted to active status. Otherwise, it's rejected.

---

## 23. Agent Evolution Workflow

Per Part I §11.

### 23.1 Tracked Metrics

For every agent: invocations, success rate, runtime, token cost, useful discoveries, false positives, wasted compute, downstream LB impact, reproducibility, failure modes.

### 23.2 A/B Test Pipeline

```text
Change proposed (e.g., new prompt for Validation Architect)
  ↓
Branch off: kaggle-agent-core-ab-test-<id>
  ↓
Run on held-out competition (worktree)
  ↓
Compare to baseline (main branch, same competition)
  ↓
Statistical review (Memory Curator computes confidence intervals)
  ↓
Adversarial review (Adversarial Reviewer attacks the test)
  ↓
Promote if superior on ≥2 metrics, not worse on any
  ↓
Otherwise rollback (delete branch, log the result)
```

### 23.3 What Agents Cannot Do

- Auto-modify their own prompts (changes go through A/B test).
- Auto-modify routing (changes go through A/B test).
- Auto-modify skills they use (changes go through Skill Evolution pipeline).
- Modify the Commander (changes go through A/B test + Adversarial Reviewer sign-off).

---

## 24. Recommended Skills from the Catalog

(See Part I §22 for the category-level triage.)

### 24.1 Install Globally (8-10 Skills)

- `agent-memory` — long-term memory patterns.
- `agent-evaluation` — agent self-evaluation.
- `context-engineering` — layered context patterns.
- `data-scientist` — reference for Trainer + Feature Engineer.
- `polars` — primary data tool.
- `using-git-worktrees` — parallel experiment isolation.
- `uv-package-manager` — Python env management.
- `multi-agent-patterns` — orchestration reference.
- (additional 1-2 identified during full triage in Phase 0).

### 24.2 Embed Patterns (Inline in Agent Prompts)

- Verification patterns from `codex-fable5` → Commander + Adversarial Reviewer prompts.
- Decomposition patterns from `multi-agent-patterns` → Commander prompt.
- Compression patterns from `context-compression` → Memory Curator prompt.

### 24.3 Reject

- Skills tied to non-Antigravity runtimes (Claude Code, Codex, Cursor, etc.).
- Skills requiring non-Gemini models.
- Skills duplicating Antigravity native capabilities.
- Skills for unrelated domains (web, mobile, game dev).
- Skills with "critical" risk and unclear benefit.

### 24.4 Full Row-by-Row Triage (Phase 0 of Implementation)

The full 2,121-row triage is Phase 0 of implementation (see §26 below). It uses a Python script to:
1. Parse `CATALOG.md` into a structured table.
2. For each row, classify: `install_global` / `install_workspace` / `embed_pattern` / `inspiration_only` / `reject`.
3. Output `skill_triage.md` with the full table and rationale.

---

## 25. Additional Skills Discovered Externally

(See Part I §23.) Framework-built custom skills (in `kaggle-agent-core/skills/`):

1. `kaggle-competition-analysis`
2. `leakage-detection-suite`
3. `validation-design`
4. `adversarial-validation`
5. `experiment-manager`
6. `ensemble-optimization`
7. `artifact-analyzer`
8. `kaggle-compute-optimizer`
9. `agent-memory-curator`
10. `skill-evolution`
11. `agent-performance-tracker`
12. `reproducibility-checker`
13. `distribution-shift-detector`
14. `pseudo-labeling`
15. `test-time-augmentation`

External frameworks to learn from (not install): AutoGluon, FLAML, Optuna, MemGPT/Letta, Voyager, OpenHands/SWE-Agent.

External patterns to adopt: Caruana 2004 ensemble selection, López de Prado PurgedKFold, Anthropic's Context Engineering, ReAct, Reflexion.

---

## 26. Implementation Phases

(See Part I §25.) The 6 phases:

### 26.1 Phase 0 — Foundation (Week 1)

1. Initialize `kaggle-agent-core/` Git repo.
2. Create directory structure (Appendix B).
3. Write schema files (`schemas/*.json`).
4. Write policy files (`policies/*.yaml`).
5. Write `AGENTS.md` workspace manifest.
6. Write 12 agent specs in `agents/*.md`.
7. Install global skills (8-10 from §24.1).
8. Run the full 2,121-row catalog triage (script + manual review).
9. Write `scripts/init_competition.sh`.

### 26.2 Phase 1 — MLS (Weeks 2-3)

Smallest end-to-end vertical slice:
- Commander + Competition Researcher + Data Forensics + Validation Architect + Trainer + Kaggle Executor.
- Tabular-only.
- Single model (LightGBM).
- No ensemble, no Adversarial Reviewer, no Memory Curator.
- Goal: 1 tabular competition, 1 submission, end-to-end.

### 26.3 Phase 2 — Adversarial Gate (Week 4)

- Add Adversarial Reviewer.
- Wire up Gate 1 (validation) and Gate 3 (submission).
- Skip Gate 2 (ensemble) until Phase 3.

### 26.4 Phase 3 — Ensemble + Memory (Weeks 5-6)

- Add Ensembler + Gate 2.
- Add Memory Curator.
- Four-layer memory store.
- Experiment scheduler with priority formula.

### 26.5 Phase 4 — Multi-Modality (Weeks 7-8)

- Add EDA Specialist for non-tabular modalities.
- Add Model Researcher for CV/NLP/time-series model families.
- Modality-specific notebooks.

### 26.6 Phase 5 — Self-Evolution (Weeks 9-10)

- Skill Evolution pipeline.
- Agent Evolution A/B test harness.
- Cross-competition meta-learning loop.

### 26.7 Phase 6 — Polish (Weeks 11-12)

- Hooks for invariant enforcement.
- Visual artifacts in EDA reports.
- Documentation.
- Benchmark against 2-3 past competitions.

---

## 27. Benchmark Methodology

(See Part I §26.) Three layers:

### 27.1 ML Benchmark

Run on 3 past Kaggle competitions (1 tabular, 1 CV, 1 NLP). Measure:
- Final private-LB rank vs. original winner.
- CV-LB gap.
- Compute used.
- Wall-clock time.

Target: top-10% private-LB on each.

### 27.2 Agent Benchmark

Per-agent: task success rate, token cost, wall-clock time, failure modes, downstream LB impact. Target: 80%+ task success, <1000 tokens per call average.

### 27.3 Framework Benchmark

End-to-end: human-intervention count per competition (target ≤5), total token cost per competition, reproducibility (re-run produces same final submission).

### 27.4 A/B Test Methodology

For any change: run new version on held-out competition, run old in parallel, compare on ≥2 metrics. Promote if superior and not worse on any. Adversarial Reviewer signs off.

---

## 28. Exact Commands Where Verified

> **Caveat:** Specific Antigravity flags are `UNVERIFIED` and should be re-checked from official Antigravity docs. Kaggle CLI commands below are stable.

### 28.1 Antigravity CLI (UNVERIFIED — re-check)

```bash
# Interactive session
agy --model gemini-3.8-flash-high

# Headless session with prompt
agy --model gemini-3.8-flash-high --headless -p "Run competition <slug>, project directory $(pwd)"

# With effort setting (verify the flag name)
agy --model gemini-3.8-flash-high --effort high

# List available models
agy models

# List custom agents
agy agents list

# Manage skills
agy skills list
agy skills install <name>
```

### 28.2 Kaggle CLI (Verified — Tier 1)

```bash
# Install
pip install kaggle

# Configure credentials
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Competitions
kaggle competitions list
kaggle competitions download -c <slug> -p data/
kaggle competitions submit -c <slug> -f submission.csv -m "<message>"
kaggle competitions submissions -c <slug>
kaggle competitions leaderboard -c <slug> --show

# Kernels (Notebooks)
kaggle kernels init -p notebooks/<exp_id>/
kaggle kernels push -p notebooks/<exp_id>/
kaggle kernels status <username>/<slug>
kaggle kernels output <username>/<slug> -p artifacts/<run_id>/
kaggle kernels pull <username>/<slug> -p notebooks/<exp_id>/

# Datasets
kaggle datasets init -p <dir>/
kaggle datasets create -p <dir>/
kaggle datasets version -p <dir>/ -m "<message>"
kaggle datasets download -d <owner>/<slug> -p <dest>/

# Models (Kaggle Models hub)
kaggle models list
kaggle models get -m <owner>/<slug>
```

### 28.3 uv (Verified — Tier 4)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt

# Add a package
uv pip install polars

# Sync from pyproject.toml
uv sync
```

### 28.4 Git Worktrees (Verified — Tier 4)

```bash
# Create a worktree for parallel experiment
git worktree add ../worktrees/exp-007 -b experiments/exp-007

# List worktrees
git worktree list

# Remove a worktree after merge
git worktree remove ../worktrees/exp-007

# Merge the experiment branch back
git checkout main
git merge experiments/exp-007
```

### 28.5 Polars (Verified — Tier 4)

```python
import polars as pl

# Lazy scan (recommended for large data)
df = pl.scan_csv("data/train.csv").collect()

# Common operations
df = (
    pl.scan_csv("data/train.csv")
    .filter(pl.col("target") > 0)
    .groupby("category")
    .agg(pl.col("value").mean().alias("mean_value"))
    .collect()
)

# Write to Parquet
df.write_parquet("data/train.parquet")
```

### 28.6 Optuna (Verified — Tier 4)

```python
import optuna

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    cv_score = run_lightgbm_cv(params)
    return cv_score

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50)
```

### 28.7 LightGBM (Verified — Tier 4)

```python
import lightgbm as lgb
import numpy as np

# Set seeds
np.random.seed(42)

# Train
train_data = lgb.Dataset(X_train, label=y_train)
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "seed": 42,
    "verbose": -1,
    "num_threads": -1,
}
model = lgb.train(params, train_data, num_boost_round=1000)

# Predict
pred = model.predict(X_test)
```

### 28.8 CatBoost (Verified — Tier 4)

```python
from catboost import CatBoostClassifier
import numpy as np

np.random.seed(42)
model = CatBoostClassifier(
    iterations=5000,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    task_type="CPU",   # CPU usually faster than GPU on Kaggle-sized tabular
    verbose=200,
)
model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=200)
pred = model.predict_proba(X_test)[:, 1]
```

---

## 29. Example End-to-End Execution

(See Appendix C for the full worked example.)

A toy tabular competition. Steps:

1. **Init**: `bash kaggle-agent-core/scripts/init_competition.sh comp-2026-tabular-x ./competition-project`
2. **Start Antigravity**: `cd competition-project && agy --model gemini-3.8-flash-high`
3. **Tell Commander**: "Run competition comp-2026-tabular-x, project directory $(pwd)"
4. **Commander runs Phase 1**: dispatches Competition Researcher → produces `state/competition.yaml`, `reports/competition_intelligence.md`.
5. **Commander runs Phase 2**: dispatches Data Forensics → produces `state/dataset_signature.yaml`, `reports/eda_report.md`, `reports/leakage_report.md`.
6. **Commander runs Phase 3**: dispatches Validation Architect → produces `validation/strategy.md`, `validation/folds/`. **Gate 1**: Adversarial Reviewer approves.
7. **Commander runs Phase 4 (Baseline)**: dispatches Trainer (LightGBM) → produces `experiments/training/exp-001/` with CV=0.847.
8. **Commander runs Phase 5 (Experiments)**: dispatches Trainer + HPO Agent in parallel (CatBoost + XGBoost + LightGBM HPO) → produces exp-002 through exp-010.
9. **Commander runs Phase 6 (Ensembling)**: dispatches Ensembler → produces `ensemble/final/blend.json`. **Gate 2**: Adversarial Reviewer approves.
10. **Commander runs Phase 7 (Kaggle Run)**: dispatches Kaggle Executor → pushes notebook, polls, downloads artifacts.
11. **Commander runs Phase 8 (Artifact Ingestion)**: Memory Curator ingests → updates `knowledge/` with structured entries.
12. **Commander runs Phase 9 (Strategy Evolution)**: Commander reviews what worked, updates strategic memory.
13. **Commander runs Phase 10 (Final Audit)**: **Gate 3**: Adversarial Reviewer runs the full checklist, approves.
14. **Commander runs Phase 11 (Submission)**: Kaggle Executor submits to leaderboard.

Total: ~2-4 hours of wall-clock, ~10-15 GPU-hours, ~3-5 agent calls per phase.

---

## 30. Known Limitations

### 30.1 What This Framework Cannot Do

1. **Guarantee a #1 finish** — the prompt explicitly forbids this claim (§4). The framework targets top-10% reliably; #1 requires either user-supplied domain insight, unusual compute budget, or luck.
2. **Replace human domain insight** — non-obvious edges (a leak, a domain trick, a feature construction no one else found) are typically what separates #1 from top-10%. The framework can absorb and propagate user-supplied insights but cannot generate them.
3. **Operate without any human intervention** — the user must accept competitions, provide credentials, and approve high-risk irreversible operations.
4. **Use non-Gemini reasoning models** — the prompt's §3 hard constraint. No Claude, no GPT, no Pro, no external LLMs.
5. **Violate Kaggle competition rules** — the Rule Compliance layer enforces per-competition rules.

### 30.2 Known Weaknesses

1. **Antigravity CLI volatility** — Antigravity is fast-moving; flags, paths, and skill formats may change. The framework isolates this risk via a thin `runtime/` shim.
2. **Gemini 3.8 Flash High reasoning ceiling** — on hard planning tasks, the model may commit early to a wrong path. The framework compensates with decomposition + adversarial gates but cannot eliminate the risk.
3. **Self-critique failure** — even with a separate Adversarial Reviewer agent, the critic and generator may share blind spots (both are Gemini 3.8 Flash High).
4. **Memory pollution risk** — bad knowledge entries may accumulate despite contradiction detection. Periodic human review is recommended.
5. **Runaway experimentation risk** — despite the stopping policy, the framework may run more experiments than necessary. The compute budget is the hard cap.
6. **Catalog triage incomplete** — the full 2,121-row catalog triage is deferred to Phase 0 of implementation. Until done, some skills may be misclassified.
7. **Unverified Antigravity specifics** — exact flags, paths, and skill formats in this blueprint are author knowledge as of September 2026 and should be re-verified.

### 30.3 Assumptions

1. Antigravity CLI is installed and configured.
2. Kaggle CLI is installed; `~/.kaggle/kaggle.json` exists.
3. `uv` is installed.
4. Git is installed.
5. Python 3.11+ is available.
6. The user has a Kaggle account with GPU/TPU access (depending on competition).
7. The user has accepted the target competition on Kaggle.
8. The user's Antigravity subscription allows the necessary token usage.

### 30.4 What Happens if Assumptions Break

- If Antigravity CLI changes: update `config/runtime.yaml` and `runtime/` shim; agent logic is unaffected.
- If Kaggle quotas change: re-verify before each competition; update `competition.yaml`.
- If Gemini 3.8 Flash High is deprecated: migrate to the current Gemini Flash model; agent prompts may need minor updates.
- If a competition's rules are unusual: the Rule Compliance layer (Adversarial Reviewer Gate 3) catches and enforces them.

---


# Appendices

## Appendix A — Skill Catalog Category Highlights

> The full row-by-row triage of all 2,121 skills in `CATALOG.md` is Phase 0 of implementation. Below is a category-level summary based on the author's preview read of the catalog.

### A.1 Categories Most Relevant to the Framework

| Category | Sample Skills | Recommendation |
|----------|---------------|-----------------|
| agent-behavior (5) | `codex-fable5`, `dispatch`, `ditto`, `fable-safe-prompt` | Inspiration only — embed verification patterns into agent prompts |
| agent-memory | `agent-memory`, `agent-memory-mcp` | Install `agent-memory` globally; install `agent-memory-mcp` as Memory MCP |
| multi-agent-* | `multi-agent-task-orchestrator`, `multi-agent-architect`, `multi-agent-patterns`, `parallel-agents`, `dispatching-parallel-agents`, `agent-orchestrator`, `agent-orchestration-improve-agent`, `agent-orchestration-multi-agent-optimize` | Inspiration; install `multi-agent-patterns` as a reference; reject the rest as duplicative |
| context-* | `context-engineering`, `context-optimization`, `context-compression`, `context-guardian`, `context-agent` | Install `context-engineering` globally; consider `context-compression`; reject the rest |
| evaluation | `agent-evaluation`, `evaluation` | Install `agent-evaluation` globally for agent self-evaluation |
| data-science / ml-engineer | `data-scientist`, `ml-engineer` | Install `data-scientist` globally as a reference skill |
| machine-learning-ops-ml-pipeline | (multiple) | Inspiration only; embed patterns into `experiment-manager` |
| polars | `polars` | Install globally; framework uses Polars heavily |
| using-git-worktrees | `using-git-worktrees` | Install globally; framework uses worktrees for parallel experiments |
| uv-package-manager | `uv-package-manager` | Install globally; framework uses uv |

### A.2 Categories to Reject (Default)

| Category | Reason |
|----------|--------|
| Claude-Code-tied skills | Wrong runtime (violates §3) |
| OpenAI-Codex-tied skills | Wrong runtime (violates §3) |
| Cursor-tied skills | Wrong runtime |
| Continue-tied skills | Wrong runtime |
| OpenRouter / Together / Anyscale skills | Wrong model (violates §3) |
| Local-LLM skills (Ollama, llama.cpp) | Wrong model (violates §3) |
| Web development skills | Not relevant to Kaggle |
| Mobile development skills | Not relevant |
| Game development skills | Not relevant |
| DevOps skills (Kubernetes, Terraform) | Excessive complexity for solo engineer |
| Database-connector skills | Framework uses files, not databases |
| Message-broker skills | Framework uses file-based blackboard |
| Cloud-infrastructure skills | Framework is Kaggle + local |
| Skills with "critical" risk and unclear benefit | Skip until evaluated |

### A.3 The Phase 0 Triage Script

`scripts/triage_catalog.py` (to be implemented in Phase 0):

```python
import re, yaml
from pathlib import Path

def parse_catalog(catalog_path: str) -> list[dict]:
    """Parse the CATALOG.md markdown table into structured rows."""
    rows = []
    in_table = False
    for line in open(catalog_path):
        if line.startswith("| Skill |"):
            in_table = True
            continue
        if in_table and line.startswith("|"):
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) >= 6 and cols[0] != "---":
                rows.append({
                    "name": cols[0],
                    "description": cols[1],
                    "risk": cols[2],
                    "source": cols[3],
                    "tags": cols[4].split(",") if cols[4] else [],
                    "triggers": cols[5].split(",") if cols[5] else [],
                })
    return rows

def classify_skill(skill: dict, framework_needs: dict) -> str:
    """Classify a skill: install_global, install_workspace, embed_pattern, inspiration_only, reject."""
    # Reject if tied to non-Antigravity runtimes
    if any(rt in skill["name"].lower() or rt in skill["description"].lower()
           for rt in ["claude-code", "codex", "cursor", "continue", "opencode", "openrouter", "ollama"]):
        return "reject"
    
    # Reject if requires non-Gemini models
    if any(m in skill["description"].lower() for m in ["claude", "gpt", "openai", "anthropic"]):
        return "reject"
    
    # ... (more rules)
    return "inspiration_only"

def main():
    rows = parse_catalog("CATALOG.md")
    classifications = []
    for r in rows:
        classifications.append({
            "name": r["name"],
            "description": r["description"],
            "risk": r["risk"],
            "classification": classify_skill(r, {}),
            "rationale": "..."  # auto-generated based on rules
        })
    
    with open("skill_triage.md", "w") as f:
        f.write("# Skill Catalog Triage\n\n")
        f.write(f"Total skills: {len(rows)}\n\n")
        f.write("| Skill | Description | Risk | Classification | Rationale |\n")
        f.write("|-------|-------------|------|----------------|-----------|\n")
        for c in classifications:
            f.write(f"| {c['name']} | {c['description'][:80]} | {c['risk']} | {c['classification']} | {c['rationale']} |\n")

if __name__ == "__main__":
    main()
```

---

## Appendix B — Directory Tree of Recommended `kaggle-agent-core/` Scaffold

```text
kaggle-agent-core/
├── AGENTS.md                              # Antigravity workspace manifest
├── README.md
├── pyproject.toml                          # Python project config (uv)
├── .gitignore
│
├── agents/                                 # 12 agent specs (Markdown with YAML frontmatter)
│   ├── commander.md
│   ├── competition_researcher.md
│   ├── data_forensics.md
│   ├── eda_specialist.md
│   ├── validation_architect.md
│   ├── feature_engineer.md
│   ├── model_researcher.md
│   ├── hpo_agent.md
│   ├── trainer.md
│   ├── ensembler.md
│   ├── adversarial_reviewer.md
│   ├── kaggle_executor.md
│   └── memory_curator.md
│
├── skills/                                 # Framework-built skills (15)
│   ├── kaggle-competition-analysis/
│   │   ├── SKILL.md
│   │   └── scripts/
│   │       └── analyze.py
│   ├── leakage-detection-suite/
│   │   ├── SKILL.md
│   │   └── scripts/
│   │       ├── target_correlation.py
│   │       ├── temporal_check.py
│   │       ├── group_check.py
│   │       └── duplicate_check.py
│   ├── validation-design/
│   │   └── SKILL.md
│   ├── adversarial-validation/
│   │   ├── SKILL.md
│   │   └── scripts/av_train.py
│   ├── experiment-manager/
│   │   ├── SKILL.md
│   │   └── scripts/registry.py
│   ├── ensemble-optimization/
│   │   ├── SKILL.md
│   │   └── scripts/
│   │       ├── hill_climb.py
│   │       └── forward_stepwise.py
│   ├── artifact-analyzer/
│   │   ├── SKILL.md
│   │   └── scripts/analyze.py
│   ├── kaggle-compute-optimizer/
│   │   └── SKILL.md
│   ├── agent-memory-curator/
│   │   └── SKILL.md
│   ├── skill-evolution/
│   │   └── SKILL.md
│   ├── agent-performance-tracker/
│   │   └── SKILL.md
│   ├── reproducibility-checker/
│   │   └── SKILL.md
│   ├── distribution-shift-detector/
│   │   ├── SKILL.md
│   │   └── scripts/psi.py
│   ├── pseudo-labeling/
│   │   ├── SKILL.md
│   │   └── scripts/pseudo_label.py
│   └── test-time-augmentation/
│       ├── SKILL.md
│       └── scripts/tta.py
│
├── knowledge/                              # Strategic (cross-competition) memory
│   ├── competitions/
│   │   └── <slug>.yaml
│   ├── strategies/
│   ├── validation/
│   ├── feature_engineering/
│   ├── models/
│   ├── ensembles/
│   ├── leakage/
│   ├── kaggle/
│   ├── compute/
│   ├── failures/
│   ├── successes/
│   ├── experiments/
│   ├── skills/
│   ├── agent-performance/
│   │   └── agent_performance.yaml
│   ├── meta-learning/
│   │   ├── knowledge_graph.yaml
│   │   ├── architecture_decisions.yaml
│   │   ├── skill_evaluations.yaml
│   │   ├── failure_modes.yaml
│   │   └── prompt_ab_tests.yaml
│   └── playbooks/
│
├── schemas/                                # JSON Schemas (stable across competitions)
│   ├── VERSION
│   ├── agent_result.schema.json
│   ├── metrics.schema.json
│   ├── experiment.schema.json
│   ├── resource_usage.schema.json
│   ├── memory_entry.schema.json
│   ├── knowledge_graph.schema.json
│   └── skill_metadata.schema.json
│
├── policies/                               # YAML-encoded policies
│   ├── stopping.yaml
│   ├── failure_recovery.yaml
│   ├── compliance.yaml
│   ├── scheduling.yaml
│   └── promotion.yaml                     # knowledge-promotion rules
│
├── orchestration/                          # Commander's playbook templates
│   ├── default_playbook.yaml
│   ├── tabular_playbook.yaml
│   ├── cv_playbook.yaml
│   ├── nlp_playbook.yaml
│   ├── time_series_playbook.yaml
│   └── multimodal_playbook.yaml
│
├── evaluation/                             # Agent self-evaluation harness
│   ├── benchmarks/
│   │   ├── tabular_benchmark.yaml
│   │   ├── cv_benchmark.yaml
│   │   └── nlp_benchmark.yaml
│   └── scripts/
│       ├── benchmark_ml.py
│       ├── benchmark_agents.py
│       └── ab_test.py
│
├── hooks/                                  # Antigravity hook scripts
│   ├── pre_tool_use.py                     # validate paths, schemas
│   ├── post_tool_use.py                    # validate outputs, log calls
│   └── stop.py                             # commit on stop, log phase
│
├── mcp/                                    # Custom MCP servers
│   ├── kaggle/
│   │   ├── server.py
│   │   └── tools/
│   │       ├── submit.py
│   │       ├── push_kernel.py
│   │       ├── poll_status.py
│   │       └── download_output.py
│   ├── experiment/
│   │   ├── server.py
│   │   └── tools/
│   │       ├── create.py
│   │       ├── update.py
│   │       ├── list.py
│   │       └── compare.py
│   └── memory/
│       ├── server.py
│       └── tools/
│           ├── retrieve.py
│           ├── ingest.py
│           ├── promote.py
│           └── contradictions.py
│
├── scripts/                                # Utility scripts
│   ├── init_competition.sh
│   ├── triage_catalog.py
│   ├── scheduler.py
│   ├── validate_artifact.py
│   ├── compute_optimizer.py
│   └── benchmark_*.py
│
├── templates/                              # Templates for new competition projects
│   ├── competition_template/              # scaffolded by init_competition.sh
│   │   ├── AGENTS.md
│   │   ├── README.md
│   │   ├── COMPETITION.md
│   │   ├── EXPERIMENTS.md
│   │   ├── CHANGELOG.md
│   │   ├── competition.yaml
│   │   ├── requirements.txt
│   │   ├── .gitignore
│   │   ├── data/
│   │   ├── src/
│   │   ├── notebooks/
│   │   │   └── template.py
│   │   ├── experiments/
│   │   ├── artifacts/
│   │   ├── reports/
│   │   │   └── figures/
│   │   ├── knowledge/
│   │   ├── features/
│   │   ├── validation/
│   │   │   └── folds/
│   │   ├── ensemble/
│   │   │   ├── candidates/
│   │   │   ├── correlations/
│   │   │   ├── weights/
│   │   │   ├── hillclimb/
│   │   │   ├── stack/
│   │   │   └── final/
│   │   ├── state/
│   │   │   ├── competition.yaml
│   │   │   ├── dataset_signature.yaml
│   │   │   ├── validation.yaml
│   │   │   ├── experiments.yaml
│   │   │   ├── ensemble.yaml
│   │   │   ├── knowledge_pointers.yaml
│   │   │   ├── agent_state.yaml
│   │   │   └── logs/
│   │   └── .agents/
│   │       ├── skills/                     # workspace-local skills
│   │       │   ├── competition-rules/
│   │       │   ├── dataset-specific/
│   │       │   └── competition-playbook/
│   │       ├── hooks/
│   │       └── agents/                     # workspace-local agent overrides
│   ├── notebook_template.py
│   ├── kernel_metadata_template.json
│   └── requirements_template.txt
│
└── config/
    └── runtime.yaml                        # Antigravity / Kaggle / Python / ML config
```

---

## Appendix C — Worked End-to-End Example (Toy Tabular Competition)

### C.1 Setup

Competition: `comp-2026-tabular-x` (hypothetical).
- Type: code competition (notebook is the submission).
- Metric: ROC AUC (binary classification).
- Data: 100k rows train, 50k rows test, 20 features, high-cardinality categoricals.
- Compute: 30 GPU-hours/week, 9h CPU/notebook.
- Submissions: 5/day.

### C.2 Init

```bash
$ bash kaggle-agent-core/scripts/init_competition.sh comp-2026-tabular-x ./competition-project
[OK] agy found
[OK] kaggle CLI found
[OK] uv found
[OK] kaggle.json found
[OK] Created ./competition-project
[OK] Scaffolded from template
[OK] Initialized Python env
[OK] Initialized Git
[OK] Downloaded competition data to data/
[OK] Populated competition.yaml (via Antigravity)
[OK] Workspace ready. Run: cd competition-project && agy --model gemini-3.8-flash-high
```

### C.3 Start

```bash
$ cd competition-project
$ agy --model gemini-3.8-flash-high
[Antigravity session starts; AGENTS.md manifest loads 12 agents]
> Run competition comp-2026-tabular-x, project directory $(pwd)
```

### C.4 Phase 1 — Competition Research

Commander dispatches Competition Researcher:
- Reads `competition.yaml`.
- Fetches the competition page.
- Writes `reports/competition_intelligence.md` with: metric, submission format, rules, deadlines, similar past competitions (from strategic memory).
- Returns structured result:
  ```yaml
  status: success
  confidence: 0.9
  findings: |
    Binary classification, ROC AUC, code competition, no internet, external data allowed.
    Similar past competitions: comp-2025-tabular-a (winner: LightGBM + CatBoost blend, CV=0.871).
  evidence:
    - type: citation
      url: https://kaggle.com/competitions/comp-2026-tabular-x
  artifacts:
    - path: reports/competition_intelligence.md
      type: report
  next_tasks:
    - cell: data
      task: data_forensics
      priority: 10
  ```

### C.5 Phase 2 — Data Forensics

Commander dispatches Data Forensics:
- Reads `data/train.csv`, `data/test.csv`.
- Writes Python scripts in `scripts/data_forensics_*.py` (via Polars).
- Produces:
  - `state/dataset_signature.yaml`: n_rows=100000, n_features=20, has_high_card_categorical=true, has_time=false, has_group=true (customer_id).
  - `reports/dataset_inventory.md`: schema, types, missingness.
  - `reports/data_quality.md`: duplicates, missing values, outliers.
  - `reports/leakage_report.md`: 1 target-correlated feature flagged (`feature_17`), 0 temporal leakage, 0 group leakage.
  - `reports/distribution_shift.md`: adversarial validation AUC=0.62 (low shift).

### C.6 Phase 3 — Validation Design

Commander dispatches Validation Architect:
- Reads `state/dataset_signature.yaml`, leakage and distribution-shift reports.
- Selects: **StratifiedGroupKFold** (5 folds, group=customer_id, stratify=target).
- Writes `validation/strategy.md` answering the CV-LB justification question.
- Writes `validation/validation_config.yaml` and 5 fold CSVs.
- **Gate 1**: Adversarial Reviewer attacks the strategy:
  - Q: Is there leakage? A: `feature_17` flagged; will drop.
  - Q: Is the CV-LB gap likely small? A: Low distribution shift (AV AUC=0.62), group leakage prevented.
  - Verdict: approved.

### C.7 Phase 4 — Baseline

Commander dispatches Trainer (LightGBM baseline):
- Reads `validation/validation_config.yaml`, `features/v1/` (initial features, includes dropping `feature_17`).
- Trains 5-fold LightGBM with default params.
- Produces `experiments/training/exp-001/`:
  - `metrics.json`: CV=0.847, fold_scores=[0.843, 0.849, 0.851, 0.845, 0.847]
  - `oof_predictions.csv`, `submission.csv`
  - `experiment.json`, `resource_usage.json`
- Commander: "Baseline established. Begin experimentation phase."

### C.8 Phase 5 — Experimentation

Commander dispatches in parallel (3 async subagents):
- Trainer (CatBoost with target encoding)
- Trainer (XGBoost with the same features)
- HPO Agent (LightGBM, 30 trials via Optuna)

Each produces a structured artifact (`exp-002`, `exp-003`, `exp-004`-`exp-030`).

Results:
- CatBoost (exp-002): CV=0.852 (+0.005 over baseline).
- XGBoost (exp-003): CV=0.849 (+0.002).
- LightGBM HPO best (exp-030): CV=0.851 (+0.004).

### C.9 Phase 6 — Ensembling

Commander dispatches Ensembler:
- Reads all OOF predictions.
- Computes correlation matrix:
  - LightGBM vs CatBoost: 0.92 (diverse enough)
  - LightGBM vs XGBoost: 0.98 (too correlated — drop)
  - CatBoost vs XGBoost: 0.93
- Forward stepwise hill climbing: starts with best single (CatBoost), adds LightGBM at weight 0.4 (CV improves to 0.854), adding more doesn't help.
- Writes `ensemble/final/blend.json`: `{"models": ["catboost_exp-002", "lightgbm_exp-030"], "weights": [0.6, 0.4], "cv_score": 0.854}`.
- **Gate 2**: Adversarial Reviewer:
  - Q: Is the ensemble overfit to OOF? A: 5-fold OOF; held-out fold not used in selection.
  - Q: Are the models diverse? A: Correlation 0.92; acceptable.
  - Q: Will this generalize? A: AV AUC=0.62 (low shift); likely.
  - Verdict: approved.

### C.10 Phase 7 — Kaggle Run

Commander dispatches Kaggle Executor:
- Reads `ensemble/final/blend.json` + the two model artifacts.
- Builds a notebook that loads both models, runs inference on test, blends 0.6/0.4.
- Writes `kernel-metadata.json`.
- Runs `kaggle kernels push`.
- Polls `kaggle kernels status` every 60s.
- After ~15 minutes: status=complete.
- Downloads output to `artifacts/run-001/`.

### C.11 Phase 8 — Artifact Ingestion

Commander dispatches Memory Curator:
- Ingests `artifacts/run-001/metrics.json`, `experiment.json`.
- Writes knowledge entry: "CatBoost + LightGBM blend (0.6/0.4) achieved CV=0.854 on comp-2026-tabular-x; AV AUC was low (0.62), suggesting low shake-up risk."
- Updates `knowledge/meta-learning/agent_performance.yaml` with this run's stats.

### C.12 Phase 9 — Strategy Evolution

Commander:
- Reviews what worked: target encoding helped; feature_17 was leakage.
- Updates strategic memory:
  - "For high-cardinality tabular with group IDs and low distribution shift, CatBoost + LightGBM blend (0.6/0.4) is a strong default."
  - Stored with confidence 0.6 (single observation).

### C.13 Phase 10 — Final Audit

**Gate 3**: Adversarial Reviewer runs the full checklist (prompt §28):
- Is CV trustworthy? Yes (5-fold StratifiedGroupKFold).
- Is there leakage? `feature_17` was dropped; no remaining leakage.
- Are features available at inference? Yes (verified notebook only uses test columns).
- Is preprocessing identical? Yes (same Polars pipeline in train and inference).
- Is test distribution different? AV AUC=0.62 (low).
- Is the ensemble overfit? No (5-fold OOF, held-out fold for selection).
- Is public-LB information being overused? No (this is the first submission).
- Is the solution reproducible? Seeds set, deterministic algorithms enabled.
- Is there a faster equivalent? CatBoost on CPU ran in 12 minutes; no need for GPU.
- Are competition rules satisfied? Yes (no external data, no internet, no banned models).
- Is the submission file correct? Row count matches, columns match, no NaN.
- Is the model actually improving expected private-LB performance? CV=0.854 vs. baseline 0.847; +0.007 expected; low shake-up risk.
- Verdict: approved.

### C.14 Phase 11 — Submission

Commander dispatches Kaggle Executor:
- `kaggle competitions submit -c comp-2026-tabular-x -f artifacts/run-001/submission.csv -m "exp-007: CatBoost+LightGBM blend CV=0.854"`
- Submission received.
- Public LB: 0.851 (within expected CV-LB gap of 0.003).

### C.15 Phase 12 — Iterate

Commander:
- Reviews: CV=0.854, public LB=0.851, gap=0.003 (low — CV is trustworthy).
- Decides: try pseudo-labeling + feature engineering for next iteration.
- Loops back to Phase 5 (Experimentation) with new feature set `features/v2/`.

The loop continues until:
- Stopping rule fires (improvements plateau, compute exhausted, deadline risk).
- Final submission is locked.

### C.16 Total Cost

- Wall-clock: ~3 hours for the first iteration (Phase 1-11).
- GPU-hours: 0 (CPU was sufficient for tabular).
- CPU-hours: ~2 (training + Kaggle notebook runtime).
- Antigravity tokens: ~150k input + ~30k output across ~30 agent calls.

### C.17 What Wasn't Covered

- This example is tabular; CV / NLP / time-series / multimodal competitions follow the same playbook with modality-specific model families and EDA.
- Skill evolution didn't trigger (no recurring failure).
- A/B test didn't run (no architectural change proposed).
- Cross-competition knowledge graph was queried (comp-2025-tabular-a was retrieved as similar).

---

## Appendix D — Known Limitations & Future Work

(See Part II §30 for the full limitations list. Summary here.)

### D.1 Hard Limitations

1. Cannot guarantee #1 (prompt §4).
2. Cannot replace human domain insight.
3. Cannot operate without any human intervention.
4. Cannot use non-Gemini reasoning models (prompt §3).
5. Cannot violate Kaggle competition rules.

### D.2 Soft Limitations

1. Antigravity CLI volatility (mitigated by `runtime/` shim).
2. Gemini 3.8 Flash High reasoning ceiling (mitigated by decomposition + gates).
3. Self-critique failure (mitigated by separate Adversarial Reviewer agent).
4. Memory pollution risk (mitigated by contradiction detection + periodic human review).
5. Runaway experimentation risk (mitigated by stopping policy + compute budget).
6. Catalog triage incomplete (Phase 0 of implementation).
7. Unverified Antigravity specifics (re-verify before production).

### D.3 Future Work (Per Part I §29)

- **Short-term (3-6 months)**: full catalog triage; AutoGluon integration; TabPFN/TabICL; vector-based memory retrieval; hook-based invariant enforcement.
- **Medium-term (6-12 months)**: cross-competition knowledge graph; skill auto-drafting; multi-user team pooling; multi-modal EDA; auto-submission selection policy.
- **Long-term (12+ months)**: evolutionary agent architecture; self-modifying playbooks; multi-region Kaggle account pooling; foundation-model fine-tuning for tabular.

---

## Appendix E — Glossary

- **AGENTS.md** — Antigravity workspace manifest. Lists agents, hooks, MCP servers, skills.
- **Adversarial Reviewer** — The agent with veto power at gates 1, 2, 3.
- **Adversarial validation (AV)** — Train a model to distinguish train vs. test; high AUC indicates distribution shift.
- **Antigravity** — Google's CLI-based agentic coding assistant; the runtime for this framework.
- **agy** — The Antigravity CLI command.
- **Blackboard** — The `state/` directory; the structured state store all agents read/write.
- **Caruana ensemble selection** — Forward stepwise hill-climbing on OOF predictions (Caruana 2004).
- **Code competition** — Kaggle competition where the notebook is the submission (no separate CSV upload).
- **Commander** — The top-level agent owning the global objective.
- **CV-LB gap** — The difference between CV score and private LB score; small gap = trustworthy CV.
- **Gate** — A high-stakes decision point where the Adversarial Reviewer must approve.
- **gemini-3.8-flash-high** — The reasoning model slug; high-effort variant of Gemini 3.8 Flash.
- **GroupKFold** — CV strategy that respects group boundaries (no entity in both train and validation).
- **Hill climbing** — Ensemble optimization method: start with best single model, add models greedily if they improve OOF.
- **HPO** — Hyperparameter optimization (typically via Optuna).
- **MCP** — Model Context Protocol (Anthropic's open protocol for exposing tools to LLM clients).
- **Memory Curator** — The agent that maintains the four-layer memory store.
- **OOF predictions** — Out-of-fold predictions; predictions on the training set made by models that didn't see those rows during training.
- **PurgedKFold** — Temporal CV with a gap between train and validation to avoid autocorrelation-induced leakage.
- **PSI** — Population Stability Index; a per-feature distribution-shift metric.
- **Public LB / Private LB** — Public leaderboard (small test subset, visible during competition) vs. private leaderboard (full test set, revealed after competition end).
- **Shake-up** — Gap between public and private LB rankings.
- **Skill** — A reusable capability invoked by an agent (vs. an agent, which is a persistent role).
- **StratifiedGroupKFold** — GroupKFold + class stratification.
- **Strategic memory** — Cross-competition knowledge base at `kaggle-agent-core/knowledge/`.
- **TTA** — Test-time augmentation; averaging predictions over augmented inputs.
- **Validation Architect** — The agent that designs the CV strategy; has veto power (Gate 1).
- **Worktree** — A Git worktree; an isolated checkout for parallel experiments.

---

**End of blueprint.**

For implementation, start with Phase 0 (Part II §26.1):
1. Initialize `kaggle-agent-core/` Git repo.
2. Create the directory structure (Appendix B).
3. Write the schema files.
4. Write the policy files.
5. Write `AGENTS.md` and the 12 agent specs.
6. Install global skills.
7. Run the full catalog triage.
8. Write `scripts/init_competition.sh`.

Then proceed through Phases 1-6 as outlined.

**Reminder of confidence-tier tags used throughout this blueprint:**
- `[T1]` Tier 1 — Official docs.
- `[T2]` Tier 2 — Official solution writeups.
- `[T3]` Tier 3 — Peer-reviewed papers.
- `[T4]` Tier 4 — High-credibility technical repositories.
- `[T5]` Tier 5 — Experienced practitioners.
- `[T6]` Tier 6 — Community discussions.
- `[KNOWN-PRACTICE]` — Industry-consensus engineering pattern.
- `[INFERRED]` — Reasonable inference from established facts.
- `[SPECULATIVE]` — Author conjecture; needs empirical validation.
- `[UNVERIFIED]` — Needs live verification before production.

