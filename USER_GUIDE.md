# Kaggle Autonomous Multi-Agent System (KAMAS): User's Operational Guide & Prompt Library

> **System Designation:** Kaggle Autonomous Multi-Agent System (KAMAS)  
> **Host Framework:** Antigravity CLI (`agy`)  
> **Reasoning Core:** `gemini-3.8-flash-high` (Effort: High)  
> **Version:** 1.0.0  
> **Operational Paradigm:** Two-Tier Cognitive–Deterministic Hybrid Architecture

---

## Table of Contents

1. [The Two-Tier Architecture: What agy Does vs. What Python Does](#1-the-two-tier-architecture-what-agy-does-vs-what-python-does)
2. [Prerequisites & Environment Setup](#2-prerequisites--environment-setup)
3. [System Capability Verification](#3-system-capability-verification)
4. [Competition Initialization](#4-competition-initialization)
5. [Complete agy Prompt Template Library (By Role & Phase)](#5-complete-agy-prompt-template-library-by-role--phase)
   - [Phase 1: Reconnaissance (Scout)](#phase-1-reconnaissance-scout)
   - [Phase 2 & 3: Data Ingestion & Forensics (Data Forensics)](#phase-2--3-data-ingestion--forensics-data-forensics)
   - [Phase 4: Adversarial Validation & Drift Analysis (Data Forensics)](#phase-4-adversarial-validation--drift-analysis-data-forensics)
   - [Phase 5: CV Scheme Design (Validation Architect - VETO)](#phase-5-cv-scheme-design-validation-architect---veto)
   - [Phase 6: Data Leakage Pre-Audit (Leakage Compliance - VETO)](#phase-6-data-leakage-pre-audit-leakage-compliance---veto)
   - [Phase 7: End-to-End Minimal Baseline (Runner)](#phase-7-end-to-end-minimal-baseline-runner)
   - [Phase 8: Hypothesis Formulation & EV Ranking (Feature & Model Strategist)](#phase-8-hypothesis-formulation--ev-ranking-feature--model-strategist)
   - [Phase 9: Sandboxed Trial Execution (Experiment Manager & Runner)](#phase-9-sandboxed-trial-execution-experiment-manager--runner)
   - [Phase 10: Ensembling & Hill-Climbing Blending (Blender)](#phase-10-ensembling--hill-climbing-blending-blender)
   - [Phase 11: Pre-Submission Audit (Artifact Analyst - VETO)](#phase-11-pre-submission-audit-artifact-analyst---veto)
   - [Phase 12: Kaggle Cloud Deployment & Tracking (Kaggle Executor)](#phase-12-kaggle-cloud-deployment--tracking-kaggle-executor)
   - [Evolution: Memory Promotion & Postmortem (Memory Curator)](#evolution-memory-promotion--postmortem-memory-curator)
6. [Emergency & Diagnostic Prompt Templates](#6-emergency--diagnostic-prompt-templates)
   - [Self-Healing & Error Recovery](#self-healing--error-recovery)
   - [Kill Switch & Worktree Rollback](#kill-switch--worktree-rollback)
   - [The Expert Council Deliberation (/council)](#the-expert-council-deliberation-council)
7. [Inspecting the Filesystem Blackboard](#7-inspecting-the-filesystem-blackboard)
8. [Querying Project Memory (SQLite)](#8-querying-project-memory-sqlite)
9. [Hardware & Compute Budget Governance](#9-hardware--compute-budget-governance)
10. [Troubleshooting & Verification](#10-troubleshooting--verification)

---

## 1. The Two-Tier Architecture: What agy Does vs. What Python Does

A common misconception is that KAMAS is "only Python execution" or "only an LLM prompt." In reality, KAMAS is a **Two-Tier Cognitive–Deterministic Hybrid Architecture**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   THE COGNITIVE ENGINE: agy CLI                        │
│                   (Gemini 3.8 Flash High Agent)                        │
│                                                                        │
│  • Domain Feature Ideation (Interpreting real-world column semantics)   │
│  • Code Synthesis & Refactoring (Writing modular, vectorized code)     │
│  • Root-Cause Error Diagnosis (Inspecting tracebacks & self-healing)   │
│  • Role-Based Judgment (Scout, Validation Architect, Strategist, etc.) │
│  • Strategic Memory Distillation (Curating playbooks & postmortems)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Reads / Writes structured contracts
                                    ▼ (via experiments/blackboard/)
┌────────────────────────────────────────────────────────────────────────┐
│               THE DETERMINISTIC ENGINE: Python Substrate               │
│                                                                        │
│  • Matrix Math & GBDT Training (LightGBM, XGBoost, CatBoost)           │
│  • Hard Metric Computation (ROC-AUC, QWK, RMSE on validated folds)     │
│  • Leakage & Schema Vetoes (Zero group overlap assertions, SHA-256)    │
│  • Process & Quota Watchdog (Enforcing 10.5h kill & 28h GPU ceiling)   │
│  • Isolated Git Worktrees (Zero corruption of repository main branch)  │
└────────────────────────────────────────────────────────────────────────┘
```

- **`agy` does what Python cannot:** It reads messy competition overviews, invents non-linear interaction features grounded in domain understanding, reads crash stack traces to modify code, and extracts reusable lessons.
- **Python does what `agy` cannot:** It computes exact floating-point metrics, trains 2,000 gradient boosting trees across 5 folds, verifies strict SHA-256 file hashes, and enforces non-negotiable safety vetoes.

---

## 2. Prerequisites & Environment Setup

### Required Tools:

- **Linux x86_64** (Ubuntu 22.04+ / Debian 12 recommended)
- **Python $\ge 3.11$**
- **uv** (high-performance Python package manager)
- **Git**
- **Antigravity CLI (`agy`)** installed and authenticated
- **Kaggle API Credentials** (`~/.kaggle/kaggle.json`)

### Setup Commands:

```bash
# 1. Navigate to the repository
cd /home/kazuha/Documents/kaggle-agents

# 2. Initialize virtual environment and install all dependencies
uv venv
source .venv/bin/activate
uv pip install -e .

# 3. Setup Kaggle API credentials
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## 3. System Capability Verification

Before triggering autonomous trials, verify dependencies, hardware accelerator safety, and filesystem write access:

```bash
.venv/bin/python scripts/probe_capabilities.py
```

### agy Verification Prompt:

If you want `agy` to run the probe and report back its diagnostic assessment:

```text
Run `python scripts/probe_capabilities.py` and inspect the output. Verify that:
1. Python version is >= 3.11 and all core dependencies (polars, pandas, sklearn, lightgbm, pydantic) are functional.
2. The detected hardware accelerator does not violate the Pascal P100 ban.
3. Kaggle API credentials are authenticated.
Report the health status and confirm if the workspace is ready for Phase 1.
```

---

## 4. Competition Initialization

Initialize a new competition workspace:

```bash
.venv/bin/python scripts/init_competition.py \
  --competition-id "playground-series-s6e9" \
  --download-data
```

### agy Initialization Prompt:

```text
Act as the `commander` agent. Initialize competition "playground-series-s6e9":
1. Run `python scripts/init_competition.py --competition-id playground-series-s6e9 --download-data`.
2. Inspect `data/raw/playground-series-s6e9/` to confirm raw files (train.csv, test.csv, sample_submission.csv) exist.
3. Inspect `experiments/blackboard/state.json` to verify the state machine is at INITIALIZED.
Report back with the dataset dimensions and raw column names.
```

---

## 5. Complete agy Prompt Template Library (By Role & Phase)

You can copy and paste these prompt templates directly into `agy` interactive sessions or run them headlessly with `agy -p "<prompt>"`.

---

### Phase 1: Reconnaissance (Scout)

**Role:** `scout`  
**Mission:** Extract rules, determine modality, and write the deterministic metric implementation.

#### agy Prompt Template:

```text
Adopt the persona of the `scout` agent defined in `.agents/agents/scout.md`.
Our competition ID is "playground-series-s6e9".
Tasks:
1. Inspect the competition overview and sample submission in `data/raw/playground-series-s6e9/sample_submission.csv`.
2. Determine:
   - Modality: (Tabular / NLP / Vision / Multimodal)
   - Prediction Type: (Binary classification, multiclass, regression, ranking)
   - Official Evaluation Metric: (e.g. ROC-AUC, RMSE, RMSLE, QWK, LogLoss)
   - Submission Constraints: (Code competition vs CSV upload, max runtime, internet rules)
3. Check `evaluation/metrics.py`. If the official metric is not already registered, implement it with exact edge-case handling (clipping, zero-division protection) and write unit tests for it.
4. Record your findings in `experiments/blackboard/reconnaissance/overview.json`.
```

---

### Phase 2 & 3: Data Ingestion & Forensics (Data Forensics)

**Role:** `data_forensics`  
**Mission:** Standardize raw files to Parquet, profile missingness, and detect anomaly patterns.

#### agy Prompt Template:

```text
Adopt the persona of the `data_forensics` agent defined in `.agents/agents/data_forensics.md`.
Tasks:
1. Standardize raw CSVs from `data/raw/playground-series-s6e9/` into Parquet format under `data/processed/playground-series-s6e9/` using Polars to preserve optimal data types.
2. Conduct a forensic exploratory data analysis (EDA):
   - Row count and column count of train and test.
   - Missing value frequencies and patterns per column.
   - Cardinality and distribution of categorical features.
   - Target variable distribution, skewness, and class imbalance.
   - Potential identifier columns and temporal ordering columns.
3. Write the structured report to `experiments/artifacts/eda_report.json`.
4. Provide a 5-bullet summary highlighting the most critical data quality traps we must design around.
```

---

### Phase 4: Adversarial Validation & Drift Analysis (Data Forensics)

**Role:** `data_forensics`  
**Mission:** Train a classifier distinguishing train from test to detect distribution shift.

#### agy Prompt Template:

```text
Adopt the persona of the `data_forensics` agent defined in `.agents/agents/data_forensics.md`.
Tasks:
1. Execute adversarial validation using `evaluation/adversarial.py` across the processed train and test datasets.
2. Exclude the target column and row ID columns.
3. Train a 3-fold LightGBM classifier predicting whether a sample belongs to test (`is_test=1`) vs train (`is_test=0`).
4. Read the resulting ROC-AUC and feature importance ranking:
   - If AUC < 0.60: Report negligible drift.
   - If 0.60 <= AUC < 0.70: Identify the top 5 drifting features.
   - If AUC >= 0.70: Formulate a warning and recommend feature drops or adversarial sample weighting.
5. If the data passes integrity checks, mark Gate 1 cleared in `experiments/blackboard/state.json`.
```

---

### Phase 5: CV Scheme Design (Validation Architect - VETO)

**Role:** `validation_architect` (HOLDS ABSOLUTE VETO)  
**Mission:** Design leak-free fold partitions. Prevent future leaderboard shake-ups.

#### agy Prompt Template:

```text
Adopt the persona of the `validation_architect` agent defined in `.agents/agents/validation_architect.md`.
You hold absolute VETO power over invalid cross-validation designs.
Tasks:
1. Review `experiments/artifacts/eda_report.json` and the adversarial validation results.
2. Choose the optimal cross-validation scheme:
   - If grouping entities exist (e.g. user_id, patient_id): GroupKFold or StratifiedGroupKFold.
   - If temporal ordering exists: TimeSeriesSplit with expanding window.
   - If standard tabular classification: StratifiedKFold (5 splits, seed 42).
   - If skewed regression: Binned StratifiedKFold.
3. Generate `data/processed/playground-series-s6e9/folds.parquet` containing columns `['id', 'fold']`.
4. Execute invariant assertions:
   - Confirm zero entity overlap between train and validation folds.
   - Confirm all samples are assigned to a fold in [0, 4].
5. If any validation invariant fails, issue an immediate Veto Document to `experiments/blackboard/vetos/`.
```

---

### Phase 6: Data Leakage Pre-Audit (Leakage Compliance - VETO)

**Role:** `leakage_compliance` (HOLDS ABSOLUTE VETO)  
**Mission:** Audit feature transformations and fold isolation for data leakage.

#### agy Prompt Template:

```text
Adopt the persona of the `leakage_compliance` agent defined in `.agents/agents/leakage_compliance.md`.
You hold unconditional VETO authority.
Tasks:
1. Perform static code inspection and numerical correlation scans across the baseline features and folds.
2. Enforce the non-negotiable leakage invariants:
   - No feature may have a Pearson/Spearman correlation >= 0.999 with the target variable.
   - Target encoding, mean encoding, and imputers MUST be fitted strictly inside the training fold loop.
   - Zero test data information may be leaked into feature transformers.
3. If any violation is found, generate `experiments/blackboard/vetos/leakage_veto_<timestamp>.json` and halt pipeline progression.
4. If the audit is 100% clean, clear Quality Gate 2 (`GATE_2_VALIDATION_VETO`) in `experiments/blackboard/state.json`.
```

---

### Phase 7: End-to-End Minimal Baseline (Runner)

**Role:** `runner`  
**Mission:** Execute a leak-free minimal baseline model to prove pipeline mechanics and establish the ground-truth benchmark.

#### agy Prompt Template:

```text
Adopt the persona of the `runner` agent defined in `.agents/agents/runner.md`.
Tasks:
1. Use `templates/train_baseline.py` and `experiments/runner.py`.
2. Train a 5-fold baseline LightGBM model on the processed dataset using the certified folds.
3. Compute out-of-fold predictions and evaluate the exact competition metric.
4. Verify that the following physical artifacts are written to `experiments/artifacts/baseline_lgbm_001/`:
   - `metrics.json` (overall CV score and fold breakdown)
   - `oof_preds.parquet` (columns: id, target, pred)
   - `test_preds.parquet` (columns: id, pred)
   - `feature_importance.json`
5. Record the baseline trial in SQLite (`memory/experiments.db`).
6. Clear Quality Gate 3 (`GATE_3_PIPELINE_SMOKE`) in `experiments/blackboard/state.json`.
Report the baseline CV score and standard deviation across folds.
```

---

### Phase 8: Hypothesis Formulation & EV Ranking (Feature & Model Strategist)

**Role:** `feature_model_strategist`  
**Mission:** Generate domain-grounded feature transformations and model tuning hypotheses ranked by Expected Value.

#### agy Prompt Template:

```text
Adopt the persona of the `feature_model_strategist` agent defined in `.agents/agents/feature_model_strategist.md`.
Tasks:
1. Review `experiments/artifacts/eda_report.json`, baseline feature importances, and `knowledge/tabular_playbook.md`.
2. Formulate 3 distinct, high-impact hypotheses:
   - Hypothesis 1: Domain-specific interaction or aggregation feature block.
   - Hypothesis 2: Out-of-fold target encoding with m-estimate smoothing on high-cardinality categoricals.
   - Hypothesis 3: Orthogonal model family exploration (e.g. CatBoost with depth-wise symmetric trees, or XGBoost).
3. For each hypothesis, calculate the mathematical Expected Value score:
   EV = (Expected_Gain * Confidence * Info_Gain) / (Compute_Cost_Hours * (1 + Risk))
4. Enqueue all hypotheses into `experiments/blackboard/queue.yaml` sorted descending by EV score.
Display the queue with estimated runtime and EV scores.
```

---

### Phase 9: Sandboxed Trial Execution (Experiment Manager & Runner)

**Role:** `experiment_manager` & `runner`  
**Mission:** Pop the highest EV hypothesis, execute training in an isolated Git worktree, and gather verifiable artifacts.

#### agy Prompt Template:

```text
Adopt the persona of the `experiment_manager` and `runner` agents.
Tasks:
1. Inspect `experiments/blackboard/queue.yaml` and pop the highest-ranked hypothesis that fits within the remaining GPU budget (`orchestration/budget.py`).
2. Create an isolated Git worktree at `experiments/worktrees/run_<hypothesis_id>`.
3. Implement the feature transformation or model training script inside the worktree sandbox.
4. Execute 5-fold cross-validation with process watchdog telemetry (`run_sentinel.json`).
5. Ingest the physical artifacts (`metrics.json`, `oof_preds.parquet`, `test_preds.parquet`) into `experiments/artifacts/run_<hypothesis_id>/`.
6. Record the trial in `memory/experiments.db`.
7. Compare the out-of-fold score against our baseline:
   - If CV improved (ΔCV > 0): Mark trial successful and clean up worktree.
   - If CV degraded or crashed: Capture traceback, log error to database, and prune worktree.
Report the empirical CV delta and execution time.
```

---

### Phase 10: Ensembling & Hill-Climbing Blending (Blender)

**Role:** `blender`  
**Mission:** Execute Caruana forward selection with replacement across all certified OOF predictions.

#### agy Prompt Template:

```text
Adopt the persona of the `blender` agent defined in `.agents/agents/blender.md`.
Tasks:
1. Query `memory/experiments.db` for all successful trials and collect their `oof_preds.parquet` arrays.
2. Execute Caruana forward ensemble selection with replacement (50 iterations) using `evaluation/blender.py`.
3. Evaluate Quality Gate 5 (`GATE_5_BLENDING_METRIC`):
   - Does the blended ensemble out-of-fold score beat the single best model by >= +0.0005?
   - If YES: Certify the blend weights and clear Gate 5.
   - If NO: Automatically fall back to the certified single best model (weight = 1.0) and log rationale.
4. Generate the blended test predictions array.
5. Write the candidate submission to `experiments/submissions/sub_latest/submission.csv`.
Display the selected model mixture weights and final ensemble CV score.
```

---

### Phase 11: Pre-Submission Audit (Artifact Analyst - VETO)

**Role:** `artifact_analyst` (HOLDS ABSOLUTE VETO)  
**Mission:** Verify candidate submission file against `sample_submission.csv` down to the byte and compute SHA-256 stamp.

#### agy Prompt Template:

```text
Adopt the persona of the `artifact_analyst` agent defined in `.agents/agents/artifact_analyst.md`.
You hold unconditional VETO authority over submissions.
Tasks:
1. Inspect `experiments/submissions/sub_latest/submission.csv` against `data/processed/playground-series-s6e9/sample_submission.csv`.
2. Run strict verification:
   - Exact row count match (no missing or extra rows).
   - Exact column names in identical order.
   - Target values contain zero NaNs, nulls, infs, or constant trivial values.
   - Values fall within legal metric bounds (e.g. probabilities in [0.0, 1.0]).
   - ID column values and sorting match sample submission identically.
3. Compute the cryptographic SHA-256 hash.
4. If any check fails, issue an immediate Submission Veto.
5. If verified, generate `experiments/artifacts/audit_cert.json` and clear Quality Gate 6 (`GATE_6_SUBMISSION_VETO`).
```

---

### Phase 12: Kaggle Cloud Deployment & Tracking (Kaggle Executor)

**Role:** `kaggle_executor`  
**Mission:** Submit certified bundle via Kaggle CLI and record public Leaderboard feedback.

#### agy Prompt Template:

```text
Adopt the persona of the `kaggle_executor` agent defined in `.agents/agents/kaggle_executor.md`.
Tasks:
1. Verify that `experiments/artifacts/audit_cert.json` exists and confirms Gate 6 clearance.
2. Check `orchestration/budget.py` to ensure we have not exceeded the daily submission limit (maximum 4 made, 1 reserved for emergencies).
3. If this is a code competition:
   - Assemble bundle in `experiments/submissions/sub_latest/`.
   - Verify `kernel-metadata.json` has `enable_internet: false` and `enable_gpu: true`.
   - Push kernel via `kaggle kernels push -p experiments/submissions/sub_latest/`.
   - Poll `kaggle kernels status` until complete.
4. If this is a direct prediction competition:
   - Execute `kaggle competitions submit -c playground-series-s6e9 -f experiments/submissions/sub_latest/submission.csv -m "<blend_description>"`.
5. Retrieve Public Leaderboard score, update `memory/experiments.db` and `experiments/blackboard/state.json`.
Report the Public LB score vs local CV score.
```

---

### Evolution: Memory Promotion & Postmortem (Memory Curator)

**Role:** `memory_curator`  
**Mission:** Synthesize trial outcomes, write postmortems, and promote validated heuristics to Strategic Memory.

#### agy Prompt Template:

```text
Adopt the persona of the `memory_curator` agent defined in `.agents/agents/memory_curator.md`.
Tasks:
1. Query `memory/experiments.db` for all completed runs, feature importance rankings, and CV-to-LB deltas.
2. Write a comprehensive postmortem in `experiments/artifacts/postmortem.md`:
   - Summary of baseline vs final ensemble CV.
   - Which feature transformations yielded statistically significant gains (ΔCV >= +0.0010).
   - Which hypotheses failed and the physical/statistical reasons why.
   - Alignment between local CV and Public Leaderboard.
3. Promote verified techniques to Strategic Memory (`knowledge/tabular_playbook.md`).
4. Record meta-level lessons and prompt tuning recommendations in `knowledge/meta_lessons.md`.
Advance state to TERMINATED.
```

---

## 6. Emergency & Diagnostic Prompt Templates

### Self-Healing & Error Recovery

When a trial crashes or an unexpected exception is logged:

```text
An error occurred during experiment execution.
1. Inspect `experiments/blackboard/active_runs/` for the latest crashed run directory.
2. Read the error log or stack trace.
3. Categorize the root cause:
   - Out of Memory (OOM): Downcast float64 features to float32/float16, reduce batch size or num_leaves.
   - Categorical Dtype Mismatch: Ensure non-numeric columns are cast to `category`.
   - Missing Values: Implement median/mode imputation inside training fold.
4. Modify the code inside the worktree, test the fix on 1 fold, and re-run the trial.
```

---

### Kill Switch & Worktree Rollback

If an agent or trial runs in an infinite loop or exhausts memory:

```text
Act as `commander`. Trigger emergency kill switch:
1. Inspect `experiments/worktrees/` and remove any hung worktrees using `git worktree remove --force`.
2. Inspect `experiments/blackboard/active_runs/` and mark stalled runs as FAILED.
3. Roll back `experiments/blackboard/state.json` to the last certified Quality Gate.
4. Reset process locks and report clean status.
```

---

### The Expert Council Deliberation (`/council`)

When an architectural trade-off is ambiguous (e.g. CV vs Public LB divergence, or whether to use a Tabular Neural Net vs GBDT):

```text
/council Deliberate on the following trade-off:
"Our adversarial validation AUC is 0.74, indicating moderate covariate shift. LightGBM CV is 0.884, but Public LB is 0.871. Should we adversarially re-weight the training samples to mirror the test distribution, drop the top 3 drifting features, or trust standard CV?"
Coordinate insights across:
- Statistician (on sample weighting validity and variance inflation)
- ML Research Scientist (on covariate shift adaptation literature)
- Red Team Reviewer (on failure modes of aggressive feature dropping)
Do not manufacture consensus. Expose the conflict and provide a concrete benchmark hypothesis to test empirically.
```

---

## 7. Inspecting the Filesystem Blackboard

You can inspect the blackboard files at any time during an autonomous run:

```bash
# 1. Check current phase and cleared gates
cat experiments/blackboard/state.json | jq .

# 2. View remaining experimental hypotheses
cat experiments/blackboard/queue.yaml

# 3. View immutable ledger of state transitions
tail -n 20 experiments/blackboard/ledger.jsonl | jq .

# 4. Check for formal safety vetoes
ls -la experiments/blackboard/vetos/
```

---

## 8. Querying Project Memory (SQLite)

Query historical runs, fold metrics, and feature importances:

```bash
# View top 5 runs ranked by cross-validation score
sqlite3 -header -column memory/experiments.db \
  "SELECT run_id, model_family, overall_cv, std_cv, execution_time FROM experiments ORDER BY overall_cv DESC LIMIT 5;"

# View top 10 most impactful features for a specific run
sqlite3 -header -column memory/experiments.db \
  "SELECT feature_name, importance FROM feature_importances WHERE run_id='exp_hypo_interaction_001' ORDER BY importance DESC LIMIT 10;"

# View all submission records and SHA-256 hashes
sqlite3 -header -column memory/experiments.db \
  "SELECT submission_id, run_id, cv_score, public_lb, status, sha256 FROM submissions;"
```

---

## 9. Hardware & Compute Budget Governance

### Governance Constants in `CONFIG.yaml`:

- **Weekly GPU Ceiling:** **28.0 hours** (reserves a 2.0-hour buffer for critical final submissions).
- **Session Timeout Watchdog:** Stated trials are terminated at **10.5 hours** before Kaggle's 12.0-hour hard kill.
- **Daily Submission Limit:** Capped at 4 submissions/day (reserves 1 emergency rollback slot).
- **Hardware Routing:**
  - Tabular Preprocessing / Polars / EDA: CPU worker pool.
  - GBDT Training: Multi-core CPU or Dual `NvidiaTeslaT4`.
  - Transformers / Deep Learning: `NvidiaL4` (24GB VRAM).
  - Banned Accelerator: `NvidiaTeslaP100` (Pascal `sm_60` architecture lacks modern PyTorch CUDA kernels on Kaggle base images).

---

## 10. Troubleshooting & Verification

### Running the Complete Test Suite

Verify that all 14 unit and integration tests pass with 100% success rate:

```bash
.venv/bin/pytest -v tests/
```

### Common Issues & Quick Fixes:

1. **Kaggle CLI Token Error:** Ensure `~/.kaggle/kaggle.json` has permissions `chmod 600 ~/.kaggle/kaggle.json`.
2. **Pandas 3 String Dtype Warnings:** Handled automatically in `evaluation/adversarial.py` and `experiments/runner.py` using `pd.api.types.is_numeric_dtype()`.
3. **Gate 5 Blending Metric Failure:** If an ensemble does not improve CV over the single best model by $+0.0005$, the system automatically falls back to the best single model with weight 1.0.
