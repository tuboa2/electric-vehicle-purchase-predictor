# Skill Catalog Audit and Selection Matrix

**Catalog Scope:** Audit of 2,121 available skills and local `.agents/skills/` capabilities.  
**Standing Governance Rule:** Categorical rejection of all delegation skills, third-party model gateways, and non-engineering roleplays to enforce the Antigravity + Gemini 3.8 Flash High platform constraint.

---

## 1. Skill Classification Categories

- **KEEP:** Production-ready skills adopted directly into the global or workspace environment.
- **ADAPT:** Conceptual or generic skills whose core patterns are refactored into the Kaggle-native architecture.
- **CREATE:** Mission-critical Kaggle competition capabilities absent from the catalog that must be built locally.
- **IGNORE:** High-quality skills that belong to irrelevant technical domains (frontend, mobile, cloud infra, business marketing).
- **REJECT:** Disqualified skills that violate core system directives (model delegation, uncontrolled loops, security risks).

---

## 2. Selection Matrix

| Skill Name | Catalog Category | Disposition | Relevant? | Compatible? | Duplicate? | Reusable? | Rationale / Evidence |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| `uv-package-manager` | development | **KEEP** | Yes | Yes | No | Yes | Extremely fast, reproducible Python environment management. Essential for Kaggle notebook packaging. |
| `using-git-worktrees` | development | **KEEP** | Yes | Yes | No | Yes | Isolates parallel experiment branches sharing the same repository without branch-switching collisions. |
| `polars` | data-science | **KEEP** | Yes | Yes | No | Yes | Fast, zero-copy in-memory and streaming DataFrame processing; prevents 16GB RAM crashes on large datasets. |
| `scikit-learn` | ai-ml | **KEEP** | Yes | Yes | No | Yes | Standard foundation for metrics, preprocessing, and linear baseline modeling. |
| `systematic-debugging` | debugging | **KEEP** | Yes | Yes | No | Yes | Structured debugging protocols preventing random trial-and-error code edits. |
| `break-ai-fix-loops` | debugging | **KEEP** | Yes | Yes | No | Yes | Circuit breaker terminating repetitive, unprogressed bug-fixing cycles. |
| `agent-memory` | ai-ml | **ADAPT** | Yes | Yes | Partial | Yes | Pattern is sound, but generic implementation lacks competition schemas. Adapted into 4-layer file/SQLite memory. |
| `context-engineering` | ai-ml | **ADAPT** | Yes | Yes | Partial | Yes | Critical for controlling Gemini 3.8 Flash token usage. Adapted into prompt prefix caching and structured state. |
| `context-compression` | ai-ml | **ADAPT** | Yes | Yes | Partial | Yes | Adapted into LLMLingua-style sliding-window log summarization for long-horizon sessions. |
| `multi-agent-task-orchestrator`| orchestration| **ADAPT** | Yes | Yes | Partial | Yes | Quality gates and anti-duplication mechanisms adapted into the Commander's EV queue manager. |
| `agent-evaluation` | ai-agents | **ADAPT** | Yes | Yes | Partial | Yes | Adapted into the two-track evaluation framework (experiment quality vs. agent quality). |
| `runaway-guard` | safety | **ADAPT** | Yes | Yes | Partial | Yes | Adapted into the compute budget ledger (`config/compute_budget.yaml`) enforcing 30h GPU quota limits. |
| `validation-design` | *(custom)* | **CREATE** | Yes | Yes | No | Yes | **Absent from catalog.** Designs GroupKFold, TimeSeriesSplit, and stratified splits; asserts fold integrity; holds VETO. |
| `adversarial-validation` | *(custom)* | **CREATE** | Yes | Yes | No | Yes | **Absent from catalog.** Trains train-vs-test classifier; measures AUC shift; flags drifting features. |
| `gradient-boosting-suite` | *(custom)* | **CREATE** | Yes | Yes | No | Yes | **Absent from catalog.** Unified reproducible LightGBM, CatBoost, and XGBoost training with OOF Parquet export. |
| `ensemble-hill-climbing` | *(custom)* | **CREATE** | Yes | Yes | No | Yes | **Absent from catalog.** Greedy linear blend weight search over OOF probability arrays with Ridge regularization. |
| `kaggle-cli-automation` | *(custom)* | **CREATE** | Yes | Yes | No | Yes | **Absent from catalog.** Wrapper around `kaggle kernels push/status/output` and `competitions submit` with sentinels. |
| `artifact-analyst` | *(custom)* | **CREATE** | Yes | Yes | No | Yes | **Absent from catalog.** Schema validator verifying `metrics.json`, `oof_predictions.parquet`, and `submission.csv`. |
| `react-patterns`, `nextjs-*` | frontend | **IGNORE** | No | Yes | N/A | No | Web frontend frameworks irrelevant to competitive machine learning. |
| `aws-*`, `azure-*`, `gcp-*` | cloud | **IGNORE** | No | Yes | N/A | No | Enterprise cloud infrastructure skills irrelevant to Kaggle kernel execution. |
| `copywriting-*`, `seo-*` | marketing | **IGNORE** | No | Yes | N/A | No | Content marketing and commercial copy skills irrelevant to data science competitions. |
| `claude-delegate` | orchestration| **REJECT**| Yes | No | Yes | No | **Violates model constraint.** Silently routes tasks to Claude Code. |
| `codex-delegate` | orchestration| **REJECT**| Yes | No | Yes | No | **Violates model constraint.** Silently routes tasks to OpenAI Codex. |
| `cursor-delegate` | orchestration| **REJECT**| Yes | No | Yes | No | **Violates model constraint.** Silently routes tasks to Cursor. |
| `kimi-delegate`, `grok-delegate`| orchestration| **REJECT**| Yes | No | Yes | No | **Violates model constraint.** Silently routes tasks to external vendor CLIs. |
| `routerbase-model-gateway` | gateway | **REJECT**| Yes | No | Yes | No | **Violates model constraint.** Multi-model routing gateway. |
| `sandbase-mcp`, `unified-ai-gateway`| gateway | **REJECT**| Yes | No | Yes | No | **Violates model constraint.** Third-party model routing servers. |
| `yann-lecun*`, `sam-altman` | persona | **REJECT**| No | Yes | N/A | No | Persona roleplay skills offering no empirical ML engineering capabilities. |

---

## 3. Summary Statistics

- **Total Catalog Size:** 2,121 skills
- **Approved / Kept:** 6 skills (`uv-package-manager`, `using-git-worktrees`, `polars`, `scikit-learn`, `systematic-debugging`, `break-ai-fix-loops`)
- **Adapted into Core Architecture:** 6 skills (`agent-memory`, `context-engineering`, `context-compression`, `multi-agent-task-orchestrator`, `agent-evaluation`, `runaway-guard`)
- **Newly Created Local ML Skills:** 6 skills (`validation-design`, `adversarial-validation`, `gradient-boosting-suite`, `ensemble-hill-climbing`, `kaggle-cli-automation`, `artifact-analyst`)
- **Categorically Rejected:** 35+ delegation, proxy, and roleplay skills
- **Ignored:** ~2,060 off-domain skills
