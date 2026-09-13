# Research Report: Autonomous Kaggle Competition Operating System on Antigravity CLI + Gemini 3.8 Flash High

**Date:** September 13, 2026 | **Status:** Research Complete → Architecture Blueprint

---

## 1. Executive Summary

This report delivers a **Kaggle-native autonomous experimentation and competition intelligence system** designed for Antigravity CLI with Gemini 3.8 Flash High as the sole reasoning model. The architecture is grounded in 2026 research on multi-agent orchestration, agent memory systems, self-evolving skill discovery, and Kaggle-proven competition techniques.

**Core finding:** The strongest achievable architecture is **not** a flat swarm or a monolithic supervisor. It is a **hierarchical supervisor with graph-based state management**, augmented by a **persistent meta-knowledge graph** and a **closed-loop skill evolution pipeline**. This architecture minimizes coordination overhead while maximizing parallel experimentation throughput within Kaggle's compute constraints (30 GPU-hours/week, 12-hour session cap).

**Key constraint reality:** Gemini 3.8 Flash High ranks #10 in coding and #17 in agent tasks, with 80.8 coding score and 66.6 agent-task score. Its strength is long-horizon coding and agentic diligence—it "works harder" on complex tasks, calling tools iteratively and executing extra reasoning steps. The architecture must exploit this by decomposing work into well-scoped, verifiable subtasks rather than relying on a single monolithic prompt.

**Primary optimization target:** `Expected Private LB Performance / (Human Effort × Kaggle Compute)`

---

## 2. Current Antigravity Capabilities

### 2.1 CLI Architecture

Antigravity CLI (`agy`) is built in Go for faster execution and supports asynchronous workflows—orchestrating multiple agents in the background without locking the terminal. It shares the same agent harness as Antigravity 2.0 (desktop), ensuring consistent behavior.

**Supported features (migrated from Gemini CLI):**
- Agent Skills (directory-based `SKILL.md` packages)
- Hooks
- Subagents (dynamic and custom)
- Extensions (now Antigravity plugins)

### 2.2 Custom Agents

Custom Agents are file-based configurations with YAML frontmatter, stored in `.agents/agents/` (project) or `~/.gemini/config/agents/` (global). They allow explicit definition of:
- Scoped instructions
- Tool subsets (`view_file`, `replace_file_content`, `manage_task`, etc.)
- MCP server access
- Model selection and permissions

**Critical for this system:** Custom agents let you specify the exact subset of skills relevant for each specialization, solving context bloat.

### 2.3 Skills

Skills use progressive discovery: Antigravity loads only a skill's description until the request matches, then equips the full `SKILL.md`. Skills are agent-triggered, not automatically guaranteed to run—a bootstrap rule is needed for always-on discipline.

### 2.4 Subagents

Subagents can be dispatched with filled prompt templates for independent, adversarial review. Parallel dispatch is supported for independent investigations.

### 2.5 MCP Integration

Antigravity connects MCP servers for live data access. For this system, MCP enables:
- Kaggle API access (competitions, datasets, kernels)
- Experiment tracking (W&B, MLflow, or custom)
- Knowledge graph queries

---

## 3. Gemini 3.8 Flash High: Capabilities and Limitations

| Metric | Score | Rank |
|---|---|---|
| Coding | 80.8 | #10 |
| Agent tasks | 66.6 | #17 |
| Computer use | 73.0 | #15 |
| Instruction following | 73.7 | #17 |
| Speed | 63.8 | #6 |
| HLE-Verified | 54.9% | — |

**Price:** $0.75/M input, $3.75/M output, blended $0.58/M.

**Key behavioral insight:** Gemini 3.8 Flash "works harder" on complex tasks—executing extra reasoning steps and calling tools iteratively. It may use more tokens to maximize performance, especially at higher effort levels.

**Architectural implications:**

1. **Decompose aggressively.** The model's diligence is wasted on poorly scoped tasks. Every agent task must have clear success criteria and bounded scope.
2. **Verify with independent agents.** Coding rank #10 vs. agent-task rank #17 suggests the model is stronger at producing code than at autonomous multi-step planning. Verification must be structural, not just another LLM call.
3. **Use structured outputs.** YAML/JSON schemas for all agent communication eliminate free-form chatter and enable programmatic verification.
4. **Leverage tool-calling strength.** The model excels at iterative tool use—design workflows around tool-calling loops, not one-shot generation.

---

## 4. Kaggle Infrastructure Constraints

| Resource | Limit | Notes |
|---|---|---|
| GPU hours | 30/week (T4×2 or P100) | Resets weekly; single session max 12 hours |
| CPU hours | Unlimited | Not capped |
| RAM | ~29–30 GB | With GPU notebook: ~29 GB |
| Disk | ~20 GB working | Varies by image |
| Session timeout | 12 hours | Both CPU and GPU |
| Internet | Toggle per kernel | Must be disabled for submission in code competitions |

**Critical constraint:** GPU does **not** benefit most pandas/scikit-learn workflows. Kaggle explicitly notes GPU acceleration helps only specific workloads. The system must make accelerator decisions based on actual workload characteristics.

**Compute optimization techniques (evidence-backed):**
- **Polars over pandas:** Zero-copy columnar execution, predicate/projection pushdown on Parquet, 4–6 GB memory for pipelines that OOM pandas at 16 GB.
- **Arrow/Parquet:** Columnar format with row-group statistics for partition pruning.
- **Vectorized operations:** Avoid Python loops; use NumPy/Polars expressions.
- **Caching:** Precompute features; serialize intermediate artifacts.

---

## 5. Multi-Agent Architecture Comparison

| Architecture | Parallelism | Sequential Reasoning | Coordination Overhead | Best For |
|---|---|---|---|---|
| Supervisor | High (+80%) | Degraded (−70%) | Moderate | Parallelizable tasks |
| Hierarchical | High | Moderate | Higher | Large-scale tasks |
| Swarm/P2P | Very High | Poor | Low (no manager) | Independent subtasks |
| Blackboard | Moderate | Good | Low | Shared-state problems |
| Graph-Based | High | Good | Moderate | 2026 dominant pattern |

**Research finding:** Multi-agent systems outperform single agents on parallelizable tasks but degrade by 39–70% on sequential reasoning. Production failures stem from quadratic coordination overhead and error propagation.

**Decision:** Use a **hierarchical supervisor with graph-based state management**, not a flat swarm. The graph provides explicit control flow, typed state, and checkpointing.

---

## 6. Recommended Architecture

### 6.1 Core Loop

```
OBSERVE → HYPOTHESIZE → PLAN → PARALLELIZE → IMPLEMENT →
EXECUTE → MEASURE → CRITIQUE → SELECT → MEMORIZE →
GENERALIZE → EVOLVE → REPEAT
```

### 6.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    KAGGLE COMMANDER (Supervisor)             │
│  Gemini 3.8 Flash High · Custom Agent · Strategic Director   │
│  Tools: manage_task, view_file, run_command, subagent_dispatch│
└──────────────────────────┬──────────────────────────────────┘
                           │ Graph-based state
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   RESEARCH CELL      DATA CELL         MODEL CELL
   ┌──────────┐       ┌──────────┐       ┌──────────┐
   │Competition│       │Data      │       │Model     │
   │Intelligence│       │Forensics │       │Researcher│
   └──────────┘       └──────────┘       └──────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    EXPERIMENT CELL
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   TRAINER          ENSEMBLER          ANALYST
   ┌──────────┐     ┌──────────┐       ┌──────────┐
   │Model     │     │Ensemble  │       │Error     │
   │Trainer   │     │Optimizer │       │Analyst   │
   └──────────┘     └──────────┘       └──────────┘
                           │
                    ADVERSARIAL CELL
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   LEAKAGE HUNTER    CV AUDITOR       ADVERSARIAL
   ┌──────────┐     ┌──────────┐       REVIEWER
   │Leakage   │     │Validation│       ┌──────────┐
   │Detector  │     │Architect │       │Red Team  │
   └──────────┘     └──────────┘       └──────────┘
                           │
                    SUBMISSION CELL
                           │
                    KAGGLE EXECUTION
                           │
                    ARTIFACT INGESTION
                           │
                    KNOWLEDGE EVOLUTION
                           │
                    STRATEGY UPDATE
```

### 6.3 Why This Architecture

1. **Graph-based state management** (2026 dominant pattern) enables checkpointing, resumption, and visual debugging.
2. **Hierarchical supervisor** is the most common enterprise pattern, proven to boost parallel task performance by 80%.
3. **Adversarial verification** is structurally separated from generation—critical because Gemini 3.8 Flash's agent-task rank (#17) trails its coding rank (#10).
4. **Four memory layers** address the context window limitation without dumping transcripts.

---

## 7. Agent Roster (Minimum Effective Set)

### Tier 1: Strategic (Always Active)

| Agent | Role | Model | Tools |
|---|---|---|---|
| **Commander** | Global objective, resource allocation, experiment prioritization, stopping decisions | Gemini 3.8 Flash High | `manage_task`, `view_file`, `run_command`, `subagent_dispatch` |

### Tier 2: Phase Specialists (Invoked by Commander)

| Agent | Role | Key Outputs |
|---|---|---|
| **Competition Researcher** | Rules, metric, submission format, known pitfalls, previous solutions | `competition_intelligence.md` |
| **Data Forensics** | Schema, missingness, duplicates, leakage detection, distribution shift | `data_quality.md`, `leakage_report.md` |
| **Validation Architect** | CV strategy, fold design, leakage tests. **Veto power over invalid designs.** | `validation_config.yaml` |
| **Feature Engineer** | Hypothesis-driven feature generation (not blind enumeration) | Feature pipeline code |
| **Model Researcher** | Model family selection based on competition type and data characteristics | `model_strategy.md` |
| **HPO Agent** | Efficient hyperparameter optimization | Best params + OOF predictions |
| **Ensemble Optimizer** | Correlation analysis, weight search, stacking | Blend weights, final ensemble |
| **Error Analyst** | Residual structure, subgroup failures, calibration, disagreement | `error_analysis.md` |
| **Leakage Hunter** | Target/temporal/group leakage, train-test contamination | Leakage findings |

### Tier 3: Infrastructure (Always Available)

| Agent | Role |
|---|---|
| **Kaggle Executor** | Notebook preparation, `kaggle kernels push`, output download |
| **Artifact Analyst** | Metrics parsing, experiment registry updates, comparison |
| **Memory Curator** | Knowledge base maintenance, deduplication, stale detection |
| **Skill Evolver** | Failure analysis → skill candidate → evaluation → promotion |
| **Final Auditor** | Pre-submission verification. **Can veto submission.** |

**Total: 15 agents.** Research shows single agents max out at 10–15 tools. This roster stays within cognitive-load bounds while covering all critical functions.

---

## 8. Communication Protocol

All inter-agent communication uses **structured YAML artifacts**:

```yaml
status: SUCCESS | PARTIAL | FAILED
confidence: 0.0-1.0
findings:
  - claim: "High-cardinality categorical features present"
    evidence: "17 columns with >1000 unique values"
    source: "data_forensics_report.md:42"
actions_taken:
  - "Ran adversarial validation (AUC=0.52, no shift detected)"
artifacts:
  - path: "reports/data_quality.md"
    type: report
recommendations:
  - priority: HIGH
    action: "Use CatBoost with native categorical handling"
    expected_gain: "+0.003 CV"
risks:
  - "Cardinality may cause overfitting without regularization"
next_tasks:
  - agent: feature_engineer
    task: "Generate target-encoded categorical features"
```

**No free-form agent chatter.** All findings must cite evidence. Confidence must be explicit.

---

## 9. Memory Architecture

### 9.1 Four Layers

| Layer | Scope | Storage | Example |
|---|---|---|---|
| **Working** | Current task state | In-memory JSON | Current experiment config |
| **Project** | Competition-specific | `knowledge/competitions/{name}/` | Dataset signature, validation strategy |
| **Strategic** | General Kaggle knowledge | `knowledge/strategies/` | "CatBoost works well for high-cardinality categoricals" |
| **Meta** | Agent system performance | `knowledge/agent-performance/` | "Validation Architect correctly detected leakage in 8/10 cases" |

### 9.2 Knowledge Record Schema

```yaml
knowledge_record:
  id: "uuid"
  type: "pattern | failure | strategy | skill"
  statement: "..."
  evidence_count: 5
  competitions:
    - "S6E4"
    - "S5E11"
  confidence: 0.85
  counterexamples: 1
  last_validated: "2026-09-10"
  provenance:
    - experiment_id: "exp_042"
      result: "confirmed"
  promotion_status: "validated | hypothesis | speculation"
```

**Promotion rule:** Speculation → Hypothesis requires 1 confirming experiment. Hypothesis → Validated requires 3 confirming experiments across ≥2 competitions. Validated → General Principle requires 5+ experiments with <10% counterexamples.

### 9.3 Context Assembly (Retrieval, Not Dumping)

```
GLOBAL RULES (always)
+ ROLE CONTEXT (agent-specific, from custom agent definition)
+ COMPETITION BRIEF (from competition_intelligence.md)
+ RELEVANT MEMORY (top-k retrieved by similarity to current task)
+ CURRENT TASK (working memory)
+ RELEVANT ARTIFACTS (explicitly referenced files)
+ RECENT EXPERIMENTS (last 5 from experiment registry)
```

This follows the "less context, better agents" principle from 2026 research.

---

## 10. Self-Evolution Architecture

### 10.1 Knowledge Evolution Loop

```
Experiment Complete
        ↓
Structured Knowledge Record Created
        ↓
Promotion Check (evidence count, confidence)
        ↓
If promoted → Strategic Memory Updated
        ↓
If counterexample → Hypothesis Revised
        ↓
If repeated failure → Skill Candidate Flagged
```

### 10.2 Skill Evolution Pipeline

Based on EvoSkill research (2026):

```
Failure Observed
    ↓
Failure Classification (DATA/DEPENDENCY/CODE/MEMORY/VALIDATION/MODEL)
    ↓
Root Cause Analysis
    ↓
Generalizable Pattern?
    ↓ YES
Skill Candidate
    ↓
Skill Draft (SKILL.md with trigger, purpose, evidence)
    ↓
Adversarial Review (independent agent)
    ↓
Versioned Skill (v0.1)
    ↓
A/B Test vs. Baseline
    ↓
Statistical Review
    ↓
PROMOTE (to global or project skills) or REJECT
```

**Skills never auto-install.** Every skill requires:
- Version number
- Purpose and trigger conditions
- Dependencies
- Evidence of benefit (A/B test)
- Known failure modes
- Evaluation cases

### 10.3 Agent Evolution

Track per-agent metrics:

| Metric | Definition | Target |
|---|---|---|
| Task success rate | Completed without human intervention | >90% |
| Experiment quality | % of experiments with positive information gain | >70% |
| Useful discoveries | Findings that influenced decisions | Track trend |
| False positives | Incorrect warnings (e.g., leakage false alarms) | <10% |
| Wasted compute | GPU-hours on uninformative experiments | <20% |
| Runtime | Median task completion time | Track trend |
| Reproducibility | Experiments producing identical results on re-run | 100% |

**Controlled evolution:** No agent self-modifies core orchestration. Changes follow:
```
Candidate Change → Benchmark → Compare vs. Baseline → Evidence Review → Promote or Rollback
```

---

## 11. Experiment Scheduler

### 11.1 Priority Formula

```
priority = (expected_score_gain × confidence × information_gain)
           / (compute_cost × risk)
```

Where:
- `expected_score_gain`: Estimated CV improvement (from similar past experiments)
- `confidence`: 0.0–1.0 based on evidence count
- `information_gain`: How much the result reduces uncertainty (0.0–1.0)
- `compute_cost`: GPU-hours or CPU-hours required
- `risk`: Probability of breaking existing pipeline

### 11.2 Experiment Registry Schema

```yaml
experiment:
  id: "exp_042"
  competition: "S6E4"
  hypothesis: "Target encoding will improve CV by >0.002"
  intervention: "Add target-encoded features for 3 categorical columns"
  dataset_version: "v3"
  feature_version: "v7"
  validation: "5-fold StratifiedKFold, seed=42"
  model: "LightGBM"
  parameters: {num_leaves: 63, learning_rate: 0.05}
  cv_score: 0.9721
  runtime_seconds: 342
  memory_gb: 4.2
  artifacts: ["oof_predictions.csv", "feature_importance.json"]
  conclusion: "CONFIRMED: +0.0032 CV improvement"
  confidence: 0.90
  reproducibility: "seeds=42, deterministic=True"
```

---

## 12. Validation System

### 12.1 Validation as First-Class Artifact

The Validation Architect must answer: **"Why should this CV estimate correlate with private LB performance?"**

If it cannot answer convincingly, modeling must not proceed.

### 12.2 Mandatory Validation Checks

| Check | Method | Threshold |
|---|---|---|
| Adversarial validation | Train classifier to distinguish train vs. test | AUC < 0.60 |
| Leakage detection | Target leakage, temporal leakage, group leakage | No suspicious features |
| Distribution shift | Feature-wise KS test, population stability index | PSI < 0.1 |
| Group integrity | GroupKFold when groups exist | No group overlap |
| Temporal integrity | TimeSeriesSplit for temporal data | No future leakage |

### 12.3 Veto Power

The Validation Architect can veto any modeling pipeline with invalid validation. Veto requires:
- Specific check that failed
- Evidence (metrics, plots)
- Required remediation

---

## 13. Kaggle Automation Flow

### 13.1 CLI Commands (Verified)

```bash
# Download competition data
kaggle competitions download -c <competition-name>
unzip <competition-name>.zip -d ./data

# Initialize notebook
kaggle kernels init -p ./notebooks

# Push and run
kaggle kernels push -p ./notebooks -t 43200 --accelerator nvidiaTeslaT4

# Check status
kaggle kernels status <owner>/<slug>

# Download outputs
kaggle kernels output <owner>/<slug> -p ./artifacts -o

# Submit
kaggle competitions submit -c <competition-name> -f submission.csv -m "Message"
```

### 13.2 Automated Loop

```
LOCAL PROJECT
      ↓
Git commit + push
      ↓
Kaggle Notebook push
      ↓
Kaggle Execution (12-hour max)
      ↓
Artifact download (metrics, OOF, submission)
      ↓
Artifact Ingestion (parse, validate, update registry)
      ↓
Antigravity Analysis (compare vs. previous, decide next)
      ↓
Next experiment
```

**No manual copy-paste.** The loop is fully automated with CLI commands.

---

## 14. GitHub Integration

### 14.1 Branch Strategy

```
main                    # Stable, reproducible baseline
experiments/*           # Feature branches for experiments
  feature/target-encoding
  model/catboost-v2
  ensemble/hillclimb-v3
```

### 14.2 Commit Policy

**Commit:** Meaningful milestones (baseline complete, experiment confirmed, submission ready)
**Never commit:** Credentials, Kaggle API tokens, datasets >100 MB, model binaries unless required

### 14.3 Required Files

```
README.md               # Project overview
COMPETITION.md          # Competition-specific notes
EXPERIMENTS.md          # Experiment log (auto-generated)
CHANGELOG.md            # Milestone history
```

---

## 15. Artifact Contract

### 15.1 Minimum Artifacts (Every Run)

```
artifacts/
├── metrics.json              # All metrics (CV, public LB, private LB if known)
├── predictions.csv           # Test predictions
├── submission.csv            # Formatted submission
├── oof_predictions.csv       # Out-of-fold predictions
├── experiment.json           # Full experiment registry record
├── resource_usage.json       # CPU, GPU, RAM, runtime
├── logs/                     # Training logs
└── reports/                  # Error analysis, feature importance
```

### 15.2 Artifact Schema (Stable Across Competitions)

```json
{
  "schema_version": "1.0",
  "competition": "string",
  "experiment_id": "string",
  "metrics": {
    "cv_score": "float",
    "cv_std": "float",
    "public_lb": "float | null",
    "private_lb": "float | null"
  },
  "validation": {
    "strategy": "string",
    "n_folds": "int",
    "seed": "int"
  },
  "resources": {
    "runtime_seconds": "int",
    "peak_memory_gb": "float",
    "gpu_hours": "float"
  },
  "artifacts": {
    "submission": "path",
    "oof": "path",
    "model_checkpoint": "path | null"
  }
}
```

---

## 16. Compute Optimization System

### 16.1 Accelerator Decision Tree

```
Is the workload primarily:
├── Data processing (pandas/scikit-learn) → CPU only (GPU provides no benefit)
├── Gradient boosting (LightGBM/XGBoost/CatBoost) → CPU (GPU version rarely faster for <1M rows)
├── Neural network training (PyTorch/TensorFlow) → GPU (T4×2)
├── Large-scale feature engineering → CPU with Polars
└── Inference on large test set → GPU if model is neural network
```

### 16.2 Data Processing Stack

| Task | Tool | Rationale |
|---|---|---|
| CSV loading | Polars `scan_csv` | Lazy evaluation, streaming |
| Feature engineering | Polars expressions | Parallel execution, zero-copy |
| Serialization | Parquet | Columnar, compression, predicate pushdown |
| Memory management | Arrow | Zero-copy interchange |
| Categorical encoding | Polars `to_dummies` or CatBoost native | Avoid one-hot explosion |

### 16.3 AI Compute Optimization

| Technique | Implementation |
|---|---|
| Context compression | Summarize experiment logs to 3-line records |
| Prompt reuse | Cache system prompts; only inject task-specific context |
| Parallel agents | Dispatch independent experiments concurrently |
| Structured outputs | YAML schemas reduce parsing overhead |
| Memory retrieval | Top-k similarity search, not full history dump |

---

## 17. Failure Recovery System

### 17.1 Failure Classification

```
DATA_FAILURE          → Re-download, validate integrity
DEPENDENCY_FAILURE    → Pin versions, use uv for speed
CODE_FAILURE          → Debug with minimal repro, fix, re-test
MEMORY_FAILURE        → Reduce batch size, use Polars streaming
TIMEOUT               → Checkpoint, resume, reduce scope
KAGGLE_FAILURE        → Retry with exponential backoff, check quota
VALIDATION_FAILURE    → Re-design CV, add leakage tests
MODEL_FAILURE         → Fall back to simpler model, check data
AGENT_FAILURE         → Re-dispatch with clearer prompt, add context
ORCHESTRATION_FAILURE → Check graph state, resume from checkpoint
ARTIFACT_FAILURE      → Re-run, validate artifact integrity
```

### 17.2 Rule: Never Retry the Same Failed Action

Every retry must change at least one variable: hypothesis, parameters, data version, or validation strategy.

---

## 18. Evaluation Framework

### 18.1 System-Level Metrics

| Metric | Measurement |
|---|---|
| Time-to-good-solution | Hours from start to first competitive baseline |
| Compute efficiency | Private LB score / GPU-hours consumed |
| Failure recovery rate | % of failures auto-recovered |
| Hallucination rate | % of agent claims without evidence |
| Leakage detection rate | % of true leakages detected |
| Reproducibility | % of experiments producing identical results |

### 18.2 Internal Benchmark

Create a suite of synthetic Kaggle-like tasks:
- Tabular binary classification (high cardinality)
- Tabular regression (temporal)
- Image classification (small dataset)
- Text classification (imbalanced)

Run the system on each, measure all metrics above. Compare against baseline (human-guided ML pipeline).

---

## 19. Reusability Architecture

### 19.1 Separation

```
kaggle-agent-core/          # Framework (reusable)
├── agents/                 # Custom agent definitions (.agents/agents/)
├── skills/                 # Global skills (~/.gemini/config/skills/)
├── memory/                 # Memory schemas and retrieval logic
├── orchestration/          # Graph definitions, state management
├── evaluation/             # Benchmark suite
├── schemas/                # YAML/JSON schemas
├── playbooks/              # Default competition playbook
├── templates/              # Notebook, report, config templates
├── scripts/                # Automation scripts
├── policies/               # Stopping rules, compute limits
└── config/                 # Global configuration

competition-project/        # Per-competition (not reusable)
├── data/
├── src/
├── notebooks/
├── experiments/
├── artifacts/
├── reports/
├── knowledge/              # Competition-specific knowledge
└── competition.yaml        # Competition metadata
```

### 19.2 Global vs. Project Skills

| Global Skills | Project Skills |
|---|---|
| `kaggle-competition-analysis` | `competition-rules` |
| `leakage-detection` | `domain-knowledge` |
| `validation-design` | `dataset-specific-analysis` |
| `experiment-management` | `competition-specific-playbook` |
| `ensemble-optimization` | |
| `artifact-analysis` | |
| `kaggle-compute-optimization` | |
| `agent-memory` | |

Antigravity supports global skills (`~/.gemini/config/skills/`) and workspace skills (`.agents/skills/`).

---

## 20. Skill Selection from Catalog (2,121 Skills)

### 20.1 Recommended for Installation (Global)

| Skill | Category | Rationale |
|---|---|---|
| `multi-agent-task-orchestrator` | agent-orchestration | Anti-duplication, quality gates, heartbeat monitoring【Catalog】 |
| `agent-memory` | ai-ml | Persistent, searchable knowledge management【Catalog】 |
| `agent-evaluation` | ai-agents | Versioned cases, explicit verifiers【Catalog】 |
| `context-engineering` | ai-ml | Optimizes agent context setup【Catalog】 |
| `context-compression` | ai-ml | Mandatory for long sessions【Catalog】 |
| `polars` | data-science | Fast in-memory DataFrame library【Catalog】 |
| `uv-package-manager` | development | Extremely fast Python package manager【Catalog】 |
| `using-git-worktrees` | development | Isolated workspaces for parallel experiments【Catalog】 |
| `multi-agent-patterns` | ai-agents | Supervisor, swarm, hierarchical patterns【Catalog】 |
| `dispatching-parallel-agents` | ai-agents | Independent task dispatch【Catalog】 |
| `parallel-agents` | ai-agents | Multi-agent orchestration【Catalog】 |
| `agent-orchestration-improve-agent` | ai-ml | Systematic agent improvement【Catalog】 |
| `leakage-detection` (custom) | — | Not in catalog; create as custom skill |
| `validation-design` (custom) | — | Not in catalog; create as custom skill |
| `experiment-management` (custom) | — | Not in catalog; create as custom skill |

### 20.2 Recommended for Project-Local

| Skill | Rationale |
|---|---|
| `competition-rules` | Competition-specific, not reusable |
| `domain-knowledge` | Domain-specific facts |
| `dataset-specific-analysis` | Dataset-specific quirks |

### 20.3 Rejected Skills (and Why)

| Skill | Reason |
|---|---|
| `agent-squad/*` | Generic coding roles (Alex, Aria, Mason); not Kaggle-specific |
| `langgraph` | Framework dependency; Antigravity has native graph support |
| `crewai` | External framework; not Antigravity-native |
| `autonomous-agents` | Too generic; lacks Kaggle-specific methodology |
| `deep-research` | General research; not competition-focused |
| `computer-vision-expert` | Domain-specific; only useful for CV competitions |

---

## 21. Implementation Roadmap

### Phase 1: Foundation (Week 1–2)

1. Install Antigravity CLI and verify `agy` commands
2. Create `kaggle-agent-core/` directory structure
3. Define custom agents in `.agents/agents/`:
   - `commander.md`
   - `competition-researcher.md`
   - `data-forensics.md`
   - `validation-architect.md`
   - `leakage-hunter.md`
   - `adversarial-reviewer.md`
   - `final-auditor.md`
4. Create global skills for Kaggle-specific capabilities
5. Implement memory schemas and retrieval logic

### Phase 2: Core Loop (Week 3–4)

1. Implement graph-based state management
2. Build experiment registry and artifact ingestion
3. Create Kaggle execution wrapper (`kaggle_executor.py`)
4. Implement artifact contract validation
5. Test on a completed Kaggle Playground competition

### Phase 3: Self-Evolution (Week 5–6)

1. Implement knowledge promotion rules
2. Build skill evolution pipeline (EvoSkill-inspired)
3. Create agent performance tracking
4. Implement A/B testing for architectural changes
5. Run internal benchmark suite

### Phase 4: Hardening (Week 7–8)

1. Add failure recovery for all failure classes
2. Implement stopping policy
3. Add adversarial self-critique before submission
4. Create competition compliance agent
5. Document all workflows and schemas

---

## 22. Benchmark Methodology

### 22.1 Internal Benchmark Tasks

| Task | Type | Size | Metric |
|---|---|---|---|
| Synthetic Tabular A | Binary classification, high cardinality | 100K rows | AUC |
| Synthetic Tabular B | Regression, temporal | 50K rows | RMSE |
| Synthetic Image | 10-class classification | 5K images | Accuracy |
| Synthetic Text | Sentiment, imbalanced | 20K samples | F1 |

### 22.2 Metrics to Track

- Time to first competitive baseline (target: <2 hours)
- Final score vs. human baseline (target: ≥95%)
- GPU-hours consumed (target: <10 per task)
- Agent interactions (target: <50 per task)
- Human interventions required (target: 0 for non-credential actions)

### 22.3 A/B Testing Protocol

For any architectural change:
1. Define candidate change
2. Run both versions on same benchmark tasks
3. Compare metrics (paired t-test)
4. Promote if p < 0.05 and effect size > 5%

---

## 23. Risks and Limitations

### 23.1 Technical Risks

| Risk | Mitigation |
|---|---|
| Gemini 3.8 Flash agent-task rank (#17) | Independent verification; structured outputs |
| Context window limits | Layered memory; retrieval-based context |
| Coordination overhead | Hierarchical (not flat swarm); bounded team size |
| Error propagation | Adversarial verification at each stage |
| Kaggle compute limits | Prioritized experiments; compute budget tracking |

### 23.2 Data Risks

| Risk | Mitigation |
|---|---|
| Leakage | Dedicated Leakage Hunter; adversarial validation |
| Distribution shift | Adversarial validation; PSI monitoring |
| Overfitting public LB | Private-LB-first optimization; CV as primary signal |
| Competition rule violation | Compliance agent checks all rules |

### 23.3 Operational Risks

| Risk | Mitigation |
|---|---|
| Runaway experimentation | Explicit stopping rules; experiment budget |
| Agent hallucination | Evidence requirement; confidence tracking |
| Skill proliferation | Evaluation pipeline before installation |
| Memory bloat | Compression; stale knowledge detection |

---

## 24. Known Limitations

1. **No guaranteed #1 performance.** The system maximizes expected private LB performance, not guarantees victory.
2. **Gemini 3.8 Flash is not GPT-5/Claude Opus-level for planning.** The architecture compensates through decomposition and verification, but some complex planning tasks may require human intervention.
3. **Kaggle API rate limits** constrain submission frequency. The system must batch submissions strategically.
4. **No internet in submission kernels** for code competitions. All dependencies must be pre-installed or attached as datasets.
5. **12-hour session limit** requires checkpointing and resumption. Long-running training must be designed around this.
6. **GPU does not help all workflows.** The system must correctly identify when GPU acceleration is beneficial.
7. **Skill evolution requires human oversight.** Auto-installing skills without review risks introducing harmful behaviors.

---

## 25. Example End-to-End Execution

```bash
# User provides:
# Competition: S6E5 - Predicting Loan Default
# Project directory: ~/kaggle/s6e5-loan-default

$ agy --model gemini-3.8-flash-high

> Competitions to solve: S6E5 - Predicting Loan Default
> Project directory: ~/kaggle/s6e5-loan-default

[Commander Agent] Initializing competition environment...
[Commander Agent] ✓ Environment: Python 3.12, Polars 2.0, LightGBM 4.5
[Commander Agent] ✓ Kaggle CLI authenticated
[Commander Agent] ✓ GPU available: T4×2 (30 hours/week quota)

[Competition Researcher] Analyzing competition rules...
[Competition Researcher] Metric: AUC | Submission: probability | Data: 500K train, 200K test
[Competition Researcher] External data: ALLOWED (must be publicly available)
[Competition Researcher] Internet: DISABLED for submission

[Data Forensics] Profiling dataset...
[Data Forensics] ⚠️ Adversarial validation AUC = 0.58 — moderate distribution shift detected
[Data Forensics] ⚠️ 3 columns with >50% missing values
[Data Forensics] ⚠️ Column 'application_date' shows temporal structure

[Validation Architect] Designing validation strategy...
[Validation Architect] Recommended: TimeSeriesSplit (5 folds) due to temporal structure
[Validation Architect] VETO: StratifiedKFold would cause temporal leakage
[Validation Architect] ✓ Adversarial validation threshold: 0.60 (currently 0.58 — acceptable)

[Commander Agent] Baseline experiment queued...
[Model Researcher] Recommended: LightGBM + CatBoost ensemble
[HPO Agent] Running Optuna (50 trials, 2 GPU-hours)...
[HPO Agent] Best CV AUC: 0.7842

[Error Analyst] Analyzing residuals...
[Error Analyst] High error in low-income subgroup — investigate feature interactions

[Ensemble Optimizer] Building ensemble...
[Ensemble Optimizer] LightGBM + CatBoost correlation: 0.87 — moderate diversity
[Ensemble Optimizer] Hill-climbing weights: LGB 0.6, CatBoost 0.4
[Ensemble Optimizer] Final CV AUC: 0.7891 (+0.0049)

[Adversarial Reviewer] Attacking solution...
[Adversarial Reviewer] CV trustworthy? ✓ TimeSeriesSplit with gap
[Adversarial Reviewer] Leakage? ✓ No target leakage detected
[Adversarial Reviewer] Features available at inference? ✓ All features present in test
[Adversarial Reviewer] APPROVED for submission

[Kaggle Executor] Pushing notebook...
[Kaggle Executor] ✓ Kernel pushed: s6e5-loan-default-v7
[Kaggle Executor] ✓ Runtime: 1h 42m | Public LB: 0.7862

[Artifact Analyst] Ingesting artifacts...
[Artifact Analyst] ✓ Metrics parsed | ✓ Experiment registry updated
[Artifact Analyst] ⚠️ Public LB (0.7862) < CV (0.7891) — normal public/private gap

[Memory Curator] Knowledge extracted:
[Memory Curator] → "TimeSeriesSplit essential for temporal tabular data"
[Memory Curator] → "LightGBM+CatBoost blend effective for loan default"
[Memory Curator] → Confidence: 0.85 | Evidence: 3 experiments

[Commander Agent] Strategy updated. Next experiment: Feature engineering
[Commander Agent] Priority: High | Expected gain: +0.003 | Compute: 1.5 GPU-hours
```

---

## 26. Future Improvements

1. **Cross-competition knowledge graph** — Build relationships between competition types, dataset characteristics, and successful strategies.
2. **Automated feature discovery** — Use LLM to propose and test novel feature combinations.
3. **Test-time augmentation** — For vision competitions, systematic TTA search.
4. **Pseudo-labeling pipeline** — When evidence supports it, semi-supervised learning from test set.
5. **Neural architecture search** — For vision/NLP competitions where architecture matters.
6. **Multi-modal fusion** — For competitions with text+image+tabular data.

---

## 27. Source Summary

| Source | Type | Used For |
|---|---|---|
| Antigravity Custom Agents Blog | Tier 1 (Official) | Agent architecture, file format |
| Gemini 3.8 Flash Blog | Tier 1 (Official) | Model capabilities, pricing |
| ChatBench Benchmarks | Tier 1 (Benchmarks) | Coding/agent ranks |
| Kaggle CLI Quick Reference | Tier 1 (Official) | CLI commands |
| Kaggle Compute Limits | Tier 1 (Official) | GPU quota, session limits |
| Multi-Agent Orchestration Guide | Tier 2 (Technical) | Architecture comparison |
| Openlayer Multi-Agent Guide | Tier 2 (Technical) | Supervisor vs. swarm research |
| Agentic Memory (ACL 2026) | Tier 3 (Peer-reviewed) | Memory architecture |
| EvoSkill (2026) | Tier 3 (Peer-reviewed) | Skill evolution |
| Polars vs Pandas | Tier 4 (Technical) | Data processing |
| Kaggle Rules Discussion | Tier 6 (Community) | External data rules |

---

## 28. Final Deliverable: Complete Architecture Blueprint

The complete system is defined by the following specifications, all contained in this report:

1. **Architecture:** Hierarchical supervisor with graph-based state management
2. **Agent roster:** 15 agents across 3 tiers
3. **Communication:** Structured YAML protocol
4. **Memory:** 4 layers (Working, Project, Strategic, Meta)
5. **Skill evolution:** EvoSkill-inspired pipeline with adversarial review
6. **Experiment scheduler:** Priority formula with expected value calculation
7. **Validation system:** Validation Architect with veto power
8. **Kaggle automation:** CLI-based push/pull/execute loop
9. **GitHub integration:** Branch strategy with meaningful commits
10. **Artifact protocol:** Stable JSON schema across competitions
11. **Compute optimization:** Polars/Arrow/Parquet stack, accelerator decision tree
12. **Failure recovery:** 11 failure classes with specific recovery strategies
13. **Evaluation framework:** Internal benchmark with A/B testing
14. **Reusability:** `kaggle-agent-core/` vs. `competition-project/` separation
15. **Security/compliance:** Competition rules agent with veto power

**The system improves not by accumulating more information, but by accumulating validated decision-making ability.**

---

*End of research report. The architecture is ready for implementation.*
