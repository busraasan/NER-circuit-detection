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

## Get the know our codes

To check the accuracies of different models (GPT2-small, Gemma-2B-Instruct) in a fast way for NER task -> notebooks/accuracy_notebook.ipynb

To run path patching for a desired head use follow below parameters 
#### Bash script parameters (edit in `run_kl_divergence.sh` or via env)
| Name | Type | Default | Required? | Description | Example |
|---|---|---:|:---:|---|---|
| `SCRIPT` | string (filename) | `run_advanced_pos_circuit_ner.py` | ✓ | Python entrypoint that runs the experiment. | `export SCRIPT=run_advanced_pos_circuit_ner.py` |
| `PYTHON` | string (executable) | `python` | ✓ | Python interpreter to use. | `export PYTHON=python3.11` |
| `DATASETS` | array\<path\> | `("../data/ner_dataset_15each.json")` | ✓ | One or more JSON datasets to evaluate. | `("../data/ner_train.json" "../data/ner_val.json")` |
| `LAYER_HEAD_PAIRS` | array\<"L,H"\> or `"L,None"` | `("1,4")` | ✓ | Layer/head combos to test. Use `"None"` to omit `--head` (operate on whole layer). | `("9,0" "9,1" "10,None")` |
| `METRIC` | string | `kl_divergence` | ✓ | Metric name forwarded to Python. | `kl_divergence` |
| `NULL_TASK_FLAG` | flag | `--null_task` | — | Adds `--null_task` to Python command. | `--null_task` |
| `ALLOW_MULTI_FLAG` | flag | `--allow_multitoken` | — | Adds `--allow_multitoken` to Python command. | `--allow_multitoken` |
| `SAVE_ROOT` | path | `results/advanced_ner_circuit_dataset_correct_tag/kl_divergence` | ✓ | Root directory for outputs. Per-run subfolders are auto-created. | `results/kl_div_sweeps` |
| `DEVICE_ARG` | string (passthrough) | `--device cuda:0` | ✗ | Device override; appended only if non-empty. | `export DEVICE_ARG="--device cpu"` |
| `MODEL_ARG` | string (passthrough) | `--model_name gpt2-small` | ✗ | Model override; appended only if non-empty. | `export MODEL_ARG="--model_name EleutherAI/pythia-70m"` |
| `COMPONENT_ARG` | string (passthrough) | *(empty)* | ✗ | Optional component (e.g., `z`, `q`, `k`, `v`) if supported. | `export COMPONENT_ARG="--component z"` |
| `SEED_ARG` | string (passthrough) | *(empty)* | ✗ | Random seed if supported. | `export SEED_ARG="--seed 42"` |

**Output folder pattern:**  
`<SAVE_ROOT>/<dataset_name>/layer_<L>_head_<H|none>/`  
Each run writes a `run.log` plus any artifacts your Python script saves (metrics, tensors, figures, …).

