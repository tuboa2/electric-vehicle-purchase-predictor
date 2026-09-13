# Kaggle Autonomous Multi-Agent System (KAMAS)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/architecture-Two--Tier%20Supervisor-purple.svg)](ARCHITECTURE.md)
[![Model](https://img.shields.io/badge/reasoning-Gemini%203.8%20Flash%20High-orange.svg)](CONFIG.yaml)
[![Status](https://img.shields.io/badge/status-production-green.svg)](AGENTS.md)

**KAMAS** is an elite, self-evolving, production-grade autonomous multi-agent engineering organization engineered to compete and win on Kaggle. It synthesizes insights from 10 independent research models into a unified, minimal, high-performance architecture driven by Google's **Antigravity CLI** and **Gemini 3.8 Flash High**.

---

## 🎯 Core Engineering Principles

1. **"Agents Do Not Own Truth":** Large language models formulate hypotheses, structure code, analyze errors, and orchestrate workflows. They **never** evaluate their own performance or certify metrics. Ground truth is determined strictly by deterministic Python execution, out-of-fold validation, and cryptographic artifact contracts.
2. **Deterministic Validation Veto:** Specialized safety agents (`validation_architect` and `leakage_compliance`) hold unilateral veto power. Any data leakage or invalid CV split immediately halts the pipeline.
3. **Decoupled Blackboard Architecture:** Inter-agent communication occurs via persistent files on disk (`experiments/blackboard/`), preventing conversational context degradation, token exhaustion, and non-interactive TTY output capture issues.
4. **Isolated Sandboxing via Git Worktrees:** Every trial executes in an isolated Git worktree (`experiments/worktrees/run_<uuid>`), protecting the mainline repository from unverified code mutations.
5. **Rigorous EV Scheduling:** Experiments are dispatched based on mathematical Expected Value ($\text{EV} = \frac{\mathbb{E}[\Delta \text{Metric}] \times \text{Confidence} \times \text{InfoGain}}{\text{ComputeCost} \times (1 + \text{Risk})}$), maximizing leaderboard impact per GPU hour.
6. **Hardware-Aware Quota Governance:** Hard constraints guard Kaggle's 30h/week GPU limits and 12h session caps. Automatic avoidance of outdated architectures (e.g. Pascal P100 with missing PyTorch CUDA kernels) in favor of Dual Tesla T4 or Nvidia L4.

---

## 📁 Repository Structure

```text
├── AGENTS.md                  # Constitutional rules and 11-agent role definitions
├── ARCHITECTURE.md            # Detailed system architecture, FSM, and data flows
├── CONFIG.yaml                # Global configuration, hardware limits, thresholds
├── pyproject.toml             # Python package definitions and test settings
├── README.md                  # System overview and quickstart guide
│
├── .agents/
│   ├── agents/                # YAML-frontmattered agent specifications (11 roles)
│   ├── rules/                 # Kaggle compliance and operational rules
│   └── skills/                # Domain-specific competitive ML skills
│
├── orchestration/             # State machine, blackboard manager, EV scheduler, budget
├── memory/                    # SQLite experiments database, promotion, retrieval engine
├── kaggle/                    # Kaggle CLI wrapper, kernel builder, hardware abstraction
├── experiments/               # Experiment runner, worktree sandbox manager, artifacts
├── evaluation/                # Cross-validation, metrics engine, ablation harness
├── schemas/                   # Pydantic data contracts (tasks, results, states, artifacts)
├── scripts/                   # CLI entrypoints (probe, init, run_pipeline)
├── templates/                 # Reusable kernel metadata, baseline scripts, starter configs
├── knowledge/                 # Longitudinal strategic memory (playbooks, meta lessons)
├── observability/             # Telemetry, run logs, and audit trails
└── tests/                     # Unit and integration test suite (100% pass verification)
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python $\ge 3.11$
- [uv](https://docs.astral.sh/uv/) (recommended package installer)
- Git
- Kaggle API token (`~/.kaggle/kaggle.json`)

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-org/kaggle-agents.git
cd kaggle-agents

# Create virtual environment and sync dependencies
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 3. Verify System Capabilities & Hardware
Run the system capability probe to verify Kaggle CLI authentication, GPU acceleration, and package integrity:
```bash
python scripts/probe_capabilities.py
```

### 4. Initialize a New Competition
```bash
python scripts/init_competition.py --competition-id "playground-series-s4e9"
```

### 5. Launch the Autonomous Pipeline
```bash
python scripts/run_pipeline.py --config CONFIG.yaml --competition "playground-series-s4e9"
```

---

## 🧪 Testing & Verification

KAMAS includes a comprehensive test suite testing validation contracts, blackboard atomic operations, EV scheduling, and ablation suites:

```bash
# Run all unit and integration tests
uv run pytest -v
```

---

## 🧠 The 11-Agent Roster

| Role | Plane | Primary Responsibility |
| :--- | :--- | :--- |
| **`commander`** | Control | Global FSM phase management, kill switch, budget monitoring |
| **`scout`** | Control | Competition reconnaissance, metric reverse-engineering, rules parsing |
| **`data_forensics`** | Evidence | Exploratory data analysis, missingness mapping, adversarial drift validation |
| **`validation_architect`** | Evidence | CV scheme construction (Group, Stratified, TimeSeries), **CV Veto** |
| **`leakage_compliance`** | Evidence | Target correlation scanning, fold leakage verification, **Leakage Veto** |
| **`feature_model_strategist`** | Research | Domain feature engineering, model architecture hypotheses, EV backlog |
| **`experiment_manager`** | Research | EV scheduler, compute budget allocation, parallel dispatch |
| **`runner`** | Research | Git worktree sandbox execution, training loop, OOF/test generation |
| **`blender`** | Research | Out-of-fold blending, hill climbing optimization, stacking meta-learners |
| **`artifact_analyst`** | Verification | Strict artifact schema validation, cryptographic hash checking, **Schema Veto** |
| **`kaggle_executor`** | Verification | Kaggle CLI kernel pushing, offline dataset bundling, submission logging |
| **`memory_curator`** | Evolution | Pattern mining, strategy promotion to `knowledge/`, postmortem analysis |

---

## 📜 Kaggle Fair Play & Compliance

KAMAS strictly adheres to all Kaggle Community Guidelines and Terms of Service:
- **Single Account Only:** Enforces single-account identity; no multiple accounts or private sharing across unmerged teams.
- **Offline Inference Enforced:** Validates that submitted code runs with `internet_allowed=false`.
- **Public External Data:** Any external dataset used is verified to have an accompanying public forum announcement.
- **Pure Cognitive Execution:** Single model pinned to `gemini-3.8-flash-high` with no unauthorized external agent routing.

---

## 📄 License
Apache 2.0. See `LICENSE` for details.
