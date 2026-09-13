# Natural Language Processing (NLP) Playbook

> **Strategic Memory (Layer 3)**

## 1. Backbone Architectures
- DeBERTa-v3-large / base (strongest default for classification and ranking tasks).
- RoBERTa-large (robust secondary backbone for diverse ensembling).
- Modern lightweight models: Qwen2.5-0.5B / 1.5B or Llama-3.2-1B with LoRA fine-tuning for complex reasoning.

## 2. Training Tactics
- Layer-wise learning rate decay (LLRD).
- Multi-sample dropout.
- Adversarial training (FGM / AWP).
- Sequence length binning to minimize padding and save GPU memory.
