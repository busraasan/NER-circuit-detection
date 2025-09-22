# NER-circuit-detection

Mechanistic interpretability has revealed neat “circuits” in small LMs for structured tasks (e.g., IOI in GPT-2). We test whether similar circuitry appears for a real-world span task—NER—by probing GPT-2 small (and optionally Gemma-2B-Instruct) with activation patching on matched clean/corrupt prompts, evaluating interventions via KL divergence at the answer position. Using NER data derived from Holmes, we see later-layer concentration (≈ L9–L11) but no sparse, reusable circuit: effects are diffuse, prompt-fragile, and sensitive to corruption design. This suggests span extraction in small decoder-only LMs is distributed and context-dependent, and that conclusions hinge on metric and protocol choices.

---

## ✨ Highlights

- **Dataset:** NER-QA constructed from the **Holmes** benchmark’s NER task.  
- **Models:** GPT-2 (small) by default; easily swappable.  
- **Metrics:** token-level EM, **KL divergence** for representational drift.  
- **Analyses:** attention-head scoring, activation/attention **patching heatmaps**, head-type labeling (induction, name-mover, s-induction), and visualization.

---

## 📦 Installation

Create and activate an environment (example with Conda):

```bash
conda create -n llm python=3.11 -y
conda activate llm
pip install -r requirements.txt
```

## 🧠 Data: Holmes → NER-QA
Our dataset is created from the Holmes benchmark’s NER task:
Waldis, A., Perlitz, Y., Choshen, L., Hou, Y., & Gurevych, I. (2024).
Holmes: A Benchmark to Assess the Linguistic Competence of Language Models.
arXiv:2404.18923. https://arxiv.org/abs/2404.18923


