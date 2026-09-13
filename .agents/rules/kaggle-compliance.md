# Kaggle Compliance & Operational Governance Rules

> **Applies to:** All KAMAS agents, tools, scripts, and execution loops  
> **Authority:** Supreme Constitutional Rule (Cannot be overridden by agent prompts)

---

## 1. Kaggle Terms of Service & Fair Play Compliance

1. **Single Account Rule:**
   - The system operates strictly under the user's authenticated Kaggle account.
   - Creating secondary or alternate accounts for extra submission quotas or probe probing is strictly prohibited.
2. **Private Sharing Ban:**
   - Under no circumstances will code, data, models, or predictions be shared privately with any third party outside the registered competition team.
3. **External Data Transparency:**
   - Any external dataset, pretrained weight checkpoint, or supplementary corpus incorporated into a pipeline MUST have a corresponding public announcement thread posted on the official Kaggle competition forum.
   - If no public posting exists, the `validation_architect` and `artifact_analyst` must immediately reject the dataset.
4. **Offline Inference Mandate:**
   - All code competition submissions must run self-contained without internet connectivity (`enable_internet: false` in `kernel-metadata.json`).
   - Wheel dependencies and pretrained checkpoints must be packaged into a private Kaggle Dataset attached to the kernel.

---

## 2. Epistemic & Execution Laws

1. **Prohibition of Metric Self-Certification:**
   - No agent may assert or declare that an experiment succeeded or that a metric improved based on internal LLM reasoning or hypothetical analysis.
   - Scores are only accepted if written to `metrics.json` by the deterministic Python evaluation engine running on validated fold data.
2. **Validation Veto Supremacy:**
   - The `validation_architect` and `leakage_compliance` agents hold unconditional veto authority over the research plane.
   - If data leakage, group overlap, or temporal contamination is identified, the experiment is instantly killed.
3. **Single Reasoning Model Constraint:**
   - The system reasoning engine is pinned exclusively to `gemini-3.8-flash-high`.
   - All delegation skills (`*-delegate`), third-party LLM gateways (`routerbase`, `sandbase`, `unified-ai-gateway`), and external API relays are categorically banned.
4. **Git Sandbox Isolation:**
   - No experimental trial may modify the root Git working tree directly.
   - All code generation, parameter tuning, and execution must occur within ephemeral Git worktrees under `experiments/worktrees/run_<uuid>`.
