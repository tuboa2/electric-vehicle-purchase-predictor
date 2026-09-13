# Cross-Document Research Synthesis: Autonomous Kaggle Multi-Agent System

**Synthesis Scope:** Analysis of 10 independent research documents (`claude.md`, `deepseek.md`, `gemini.md`, `gpt.md`, `grok.md`, `julius.md`, `kimi.md`, `meta.md`, `mistral.md`, `qwen.md`).  
**Operational Target:** Maximize expected private-leaderboard performance per unit of human effort and Kaggle compute using Antigravity CLI + Gemini 3.8 Flash High.

---

## 1. Executive Synthesis

Across 10 independent research perspectives, a remarkably coherent scientific consensus emerges regarding what makes an autonomous competitive ML system genuinely effective versus what introduces brittle, cosmetic complexity.

### Core Unifying Conclusions
1. **The Fallacy of Swarms:** Unstructured multi-agent swarms, peer-to-peer debates, and large evolutionary populations are catastrophic for competitive ML. Multi-agent coordination degrades sequential reasoning by 39–70% and introduces quadratic coordination overhead $O(n^2)$. High-performance systems require a **lean hierarchical supervisor** with strict phase gates and bounded specialized roles.
2. **Epistemic Separation of Concerns ("Agents Do Not Own Truth"):** The LLM must never be the sole judge of its own code, metrics, or validation validity. The LLM's role is hypothesis generation, strategy selection, code drafting, and qualitative synthesis. Deterministic Python code computes metrics, verifies cross-validation splits, hashes artifacts, and profiles memory. Kaggle computes leaderboard reality. Git worktrees manage physical lineage.
3. **Validation as the Ultimate Arbiter:** Flawed cross-validation is the #1 cause of catastrophic private-leaderboard shakeout. A dedicated Validation Architect and Adversarial Reviewer with binding **VETO power** must guard the pipeline before any modeling or submission occurs.
4. **Filesystem Blackboard as the Communication Backbone:** Non-interactive subprocess execution of coding CLIs (`agy -p`) can suffer from stdout capture truncation or terminal hangs. Communication must occur through a durable filesystem blackboard (`blackboard/state.json`, atomic task/result envelopes, and on-disk completion sentinels).
5. **Compute & Quota Discipline:** Kaggle's 30-hour weekly GPU quota and 12-hour session caps are hard ceilings. Workloads must be strictly segregated: CPU for Polars feature pipelines, data forensics, and linear baselines; GPU strictly for neural networks and heavy gradient boosting. Experiments must be ranked by an Expected Value (EV) priority formula.
6. **Controlled Self-Evolution:** Accumulating unverified text is not intelligence. The system evolves by:
   - Converting execution failures into generalizable lessons.
   - Drafting versioned candidate skills from recurring failure patterns ($\ge 3$ occurrences).
   - Testing candidate skills in an isolated sandbox with regression benchmarks.
   - Requiring human-in-the-loop sign-off before permanent promotion.
   - Demonstrating cross-competition transfer through measurable performance gains on unseen competitions.

---

## 2. The Unified Competition Control Loop

All 10 research documents converge on a closed-loop competition state machine that integrates scientific rigor, execution feedback, and persistent memory:

```text
       ┌─────────────────────────────────────────────────────────┐
       │                   01. RECONNAISSANCE                    │
       │  Capabilities Probe · Competition Intelligence · Rules  │
       └────────────────────────────┬────────────────────────────┘
                                    │
       ┌────────────────────────────▼────────────────────────────┐
       │                    02. DATA FORENSICS                   │
       │  Modality Detection · Schema Profiling · Drift Analysis │
       └────────────────────────────┬────────────────────────────┘
                                    │
       ┌────────────────────────────▼────────────────────────────┐
       │                 03. VALIDATION ARCHITECTURE             │
       │  Adversarial Validation · Fold Design · VETO GATE G2    │
       └────────────────────────────┬────────────────────────────┘
                                    │
       ┌────────────────────────────▼────────────────────────────┐
       │                  04. BASELINE BENCHMARK                 │
       │  Fast GBDT Baseline · Reproducible Artifact Contract    │
       └────────────────────────────┬────────────────────────────┘
                                    │
       ┌────────────────────────────▼────────────────────────────┐
       │                 05. HYPOTHESIS SCHEDULING               │
       │  Prior Retrieval · EV Prioritization · Budget Tracking  │
       └────────────────────────────┬────────────────────────────┘
                                    │
       ┌────────────────────────────▼────────────────────────────┐
       │                 06. PARALLEL IMPLEMENTATION             │
       │  Git Worktrees · Feature Engineering · Model Diversity  │
       └────────────────────────────┬────────────────────────────┘
                                    │
       ┌────────────────────────────▼────────────────────────────┐
       │                 07. DETERMINISTIC EXECUTION             │
       │  Local Subprocess / Kaggle CLI Push · Sentinel Tracking │
       └────────────────────────────┬────────────────────────────┘
                                    │
       ┌────────────────────────────▼────────────────────────────┐
       │                 08. ARTIFACT INGESTION                  │
       │  Parse Metrics · OOF Validation · Error Analysis        │
       └────────────────────────────┬────────────────────────────┘
                                    │
       ┌────────────────────────────▼────────────────────────────┐
       │                 09. ADVERSARIAL AUDIT                   │
       │  Leakage Hunt · Subgroup Analysis · VETO GATE G6        │
       └────────────────────────────┬────────────────────────────┘
                                    │
       ┌────────────────────────────▼────────────────────────────┐
       │                 10. ENSEMBLE OPTIMIZATION               │
       │  OOF Correlation · Hill Climbing · Stacking             │
       └────────────────────────────┬────────────────────────────┘
                                    │
       ┌────────────────────────────▼────────────────────────────┐
       │                 11. SUBMISSION & LEADERBOARD            │
       │  Final Sign-Off · Kaggle Submission · Delta Tracking    │
       └────────────────────────────┬────────────────────────────┘
                                    │
       ┌────────────────────────────▼────────────────────────────┐
       │                 12. KNOWLEDGE & SELF-EVOLUTION          │
       │  Memory Distillation · Prior Updates · Skill Evolution  │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    └───► Iterate / Next Competition
```

---

## 3. Platform Grounding: Antigravity CLI + Gemini 3.8 Flash High

### Antigravity CLI Verification
- **Binary & CLI Syntax:** `agy` supports `--model`, `--effort (low|medium|high)`, `--agent`, `--mode (plan|accept-edits)`, `--output-format (text|json|stream-json)`, and `--json-schema`.
- **Subagents:** Native support for asynchronous background subagents spawned dynamically.
- **Skills:** Workspace skills located at `.agents/skills/<name>/SKILL.md`; global skills located at `~/.gemini/antigravity-cli/skills/`.
- **Plugins:** Packages containing skills, agents, hooks, rules, and MCP configurations.
- **Execution Safeguard:** Non-interactive headless calls (`agy -p`) must pair with on-disk completion sentinels (`run_sentinel.json`) to prevent silent failures caused by unhandled TTY stdout pipes.

### Gemini 3.8 Flash High Behavior
- **Context & Output Limits:** 1,048,576 tokens input, 65,536 tokens theoretical max output (practically optimized when individual code blocks remain under 4,000 tokens).
- **Thinking Effort:** `high` effort triggers deep multi-step reasoning and iterative tool calls.
- **Economic Optimization:** Cache-Augmented Generation (CAG) provides a 90% discount on cached tokens ($0.075/M cached vs. $0.75/M uncompressed input), allowing massive static context blocks (competition rules, schemas, past playbooks) to be preserved without incurring excessive cost.
- **Decomposition Principle:** Gemini 3.8 Flash High performs exceptionally well on bounded, verifiable software engineering tasks (scoring 90.8% on Terminal-Bench 2.1), but degrades when given vague, open-ended multi-agent prompts. Work must be strictly decomposed into single-responsibility tasks with structured schemas.

---

## 4. Kaggle Infrastructure & Hardware Grounding

- **Weekly Quotas:** ~30 GPU hours per week (resets weekly). Session limit: 12 hours wall-clock.
- **Code Competition Constraint:** Internet access is strictly disabled during official evaluation runs. All model weights, utility packages, and preprocessing code must be self-contained or mounted as Kaggle Datasets.
- **Hardware Profile:**
  - `NvidiaTeslaT4` (dual 15GB VRAM) and `NvidiaL4` are fully functional and reliable.
  - `NvidiaTeslaP100` must be treated with caution due to CUDA Pascal `sm_60` kernel incompatibilities in recent PyTorch images.
  - Workload partitioning: Never use GPU for standard pandas/scikit-learn routines; reserve GPU compute exclusively for deep learning and GPU-accelerated gradient boosting.
- **CLI Workflow:** `kaggle kernels push -p <dir>` to upload and execute, `kaggle kernels status <slug>` to poll execution state, `kaggle kernels output <slug>` to download artifacts, and `kaggle competitions submit` to finalize predictions.

---

## 5. Memory Architecture & Knowledge Promotion

Memory is structured into 4 distinct layers:
1. **Working Memory:** Ephemeral session context and `blackboard/state.json`.
2. **Project Memory:** Competition-specific artifacts, EDA reports, validation schemes, and the atomic experiment ledger in `knowledge/competitions/<slug>/` and `artifacts/<exp_id>/`.
3. **Strategic Memory:** Cross-competition validated principles stored in `knowledge/strategies/` as versioned YAML records with evidence counts, counterexample logs, and confidence scores.
4. **Meta Memory:** Agent performance scorecards, prompt versions, and scheduler metrics stored in SQLite (`knowledge/meta/experiments.db`).

### Promotion Discipline
- **Weak Hypothesis:** Plausible insight derived from a single experiment ($evidence\_count = 1$).
- **Strong Pattern:** Replicated across $\ge 2$ seeds or $\ge 2$ distinct competitions with $<10\%$ counterexamples.
- **Strategic Rule:** Validated across $\ge 5$ competitions; incorporated into default playbook initialization.
- *Speculation is never auto-promoted.*
