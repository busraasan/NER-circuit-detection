from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import torch
import argparse
import sys, os
sys.path.append(os.path.abspath(".."))  # go up one level to the root
from utils.utils import *
from src.advanced_path_patching import *
import spacy, random
import numpy as np

"""
This script performs path patching to analyze the contribution of specific attention heads
to Named Entity Recognition (NER) tasks using a GPT-2 model.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
parser.add_argument("--head", type=int, default=None)
parser.add_argument("--layer", type=int, default=11)
parser.add_argument("--dataset_path", type=str, default="../data/ner_dataset_15each.json")
parser.add_argument("--component", type=str, default="z")
parser.add_argument("--save_dir", type=str, default="results/ner_small_dataset/kl_divergence")
parser.add_argument("--metric", type=str, default="kl_divergence")
parser.add_argument("--model_name", type=str, default="gpt2-small")
parser.add_argument("--null_task", action="store_true")
parser.add_argument("--allow_multitoken", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_samples", type=int, default=100)

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

device = args.device
receiver_nodes = [(args.layer, args.head)]   # MLP at layer 13
component = args.component
position = -1
freeze_mlps = True
indirect_patch = False
metric=args.metric
seed = args.seed
null_task = args.null_task
allow_multitoken = args.allow_multitoken

model_name = args.model_name
model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device = device
    )

if "json" in args.dataset_path:
    tv_data = load_json(args.dataset_path)
else:
    raise NotImplementedError("Only json datasets are supported currently.")

tag_set = [
    "EVENT", "LOCATION", "MONEY",
    "NATIONALITY, RELIGIOUS, or POLITICAL GROUP",
    "ORGANIZATION", "NUMERICAL", "PERSON",
    "PRODUCT", "TIME"
]

template = """Sentence: Apple announced a new iPhone during its annual product launch event.,
Entity tag: PRODUCT,
Answer: iPhone

Sentence: Barack Obama delivered a keynote speech at the conference.,
Entity tag: PERSON,
Answer: Barack Obama

Sentence: Tesla invested over 2 billion dollars in a new gigafactory in Germany.,
Entity tag: MONEY,
Answer: 2 billion dollars

Sentence: The concert will take place at 8 p.m. on Saturday.,
Entity tag: TIME,
Answer: 8 p.m. on Saturday

Sentence: The Eiffel Tower is located in Paris.,
Entity tag: LOCATION,
Answer: Paris

Sentence: The Olympic Games in Tokyo attracted thousands of visitors despite the pandemic.,
Entity tag: EVENT,
Answer: Olympic Games

Sentence: The recipe calls for 200 grams of sugar and 3 eggs.,
Entity tag: NUMERICAL,
Answer: 200 grams

Sentence: Google has opened a new research center in Zurich to focus on AI development.,
Entity tag: ORGANIZATION,
Answer: Google

Sentence: The American have a long history of culinary excellence.,
Entity tag: NATIONALITY, RELIGIOUS, or POLITICAL GROUP,
Answer: American

Sentence: The Islam religion has over a billion followers worldwide.,
Entity tag: NATIONALITY, RELIGIOUS, or POLITICAL GROUP,
Answer: Islam

Sentence: {sentence}
Entity tag: {tag}
Answer:"""

output = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
random.seed(seed)

# load English model with Penn Treebank tags
nlp = spacy.load("en_core_web_sm")
num_samples = 0
model.eval()
torch.set_grad_enabled(False)
for i, item in enumerate(tv_data):

    item["Sentence"] = item["Sentence"].replace("\"","") + "."
    item["Answer"] = item["Answer"].replace("\"","").lower()
    item["POS tag"] = item["POS tag"].replace("\"","")

    clean_prompt = template.format(sentence=item["Sentence"], tag=item["POS tag"])
    ans_corr = None
    ans_clean = model.to_tokens(item["Answer"], prepend_bos=False).squeeze(-1)

    if null_task and metric == "kl_divergence":
        random_tag = "null"
        ans_corr = item["Answer"]
        ans_corr = model.to_tokens(ans_corr, prepend_bos=False).squeeze(-1)

    elif null_task and metric == "logit_diff":
        corrupted_answers_list = []
        for t in tag_set:
            if t != item["POS tag"]:
                random_tag = t
                ans_corr = first_occurrence(item["Sentence"], random_tag, PTB_TO_COARSE, nlp)
                if ans_corr is not None:
                    ans_corr = model.to_tokens(ans_corr, prepend_bos=False).squeeze(-1)
                    if ans_corr.shape == torch.Size([1]):
                        corrupted_answers_list.append((random_tag, ans_corr))
        random_tag = "null"

    else:
        while ans_corr == None:
            random_tag = random.choice([t for t in tag_set if t != item["POS tag"]])
            ans_corr = first_occurrence(item["Sentence"], random_tag, PTB_TO_COARSE, nlp)
        ans_corr = model.to_tokens(ans_corr, prepend_bos=False).squeeze(-1)

    corrupted_prompt = template.format(
        sentence=item["Sentence"],
        tag=random_tag
    )
    if metric == "logit_diff" and null_task == False:
        if ans_clean.shape != torch.Size([1]) or ans_corr.shape != torch.Size([1]):
            continue

        # Per-example tokenization (no growing lists)
        source_toks = model.to_tokens(clean_prompt, prepend_bos=False).squeeze(-1)
        corr_toks = model.to_tokens(corrupted_prompt, prepend_bos=False).squeeze(-1)

    elif null_task and metric == "logit_diff":

        if ans_clean.shape != torch.Size([1]):
            continue

        if corrupted_answers_list == []:
            continue

        source_toks = model.to_tokens(clean_prompt, prepend_bos=False).squeeze(-1)
        corr_toks = model.to_tokens(corrupted_prompt, prepend_bos=False).squeeze(-1)
    else:
        if null_task and allow_multitoken:
            source_toks = model.to_tokens(clean_prompt+item["Answer"], prepend_bos=False).squeeze(-1)
            corr_toks = model.to_tokens(corrupted_prompt+item["Answer"], prepend_bos=False).squeeze(-1)
        else:
            if ans_clean.shape != torch.Size([1]) or ans_corr.shape != torch.Size([1]):
                continue
            source_toks = model.to_tokens(clean_prompt, prepend_bos=False).squeeze(-1)
            corr_toks = model.to_tokens(corrupted_prompt, prepend_bos=False).squeeze(-1)

    if metric == "kl_divergence" or metric == "tf_loss":
        ans_tokens = torch.stack([torch.tensor((ans_clean))]).to(device)

    elif metric == "logit_diff" and null_task == True:
        ans_tokens_list = []
        for corr in corrupted_answers_list:
            ans_tokens = torch.stack([torch.tensor((ans_clean, corr[1]))]).to(device)
            ans_tokens_list.append(ans_tokens)

        ans_tokens = ans_tokens_list
    else:
        ans_tokens = torch.stack([torch.tensor((ans_clean, ans_corr))]).to(device)
    
    output+=path_patching(model, receiver_nodes, source_toks, corr_toks, ans_tokens, component, position, freeze_mlps, indirect_patch, metric=metric, is_multitoken=allow_multitoken, device=device)
    print(torch.cuda.memory_allocated() // 1e6, "MB")
    del source_toks, corr_toks, ans_tokens
    torch.cuda.empty_cache()
    num_samples += 1
    print(f"Completed {num_samples} samples", end="\r")
    
output/=num_samples
print("OUTPUT", output)

saved_path = save_output(output, receiver_nodes, base="deneme_dataset"+args.dataset_path.split("/")[-1].split(".")[0]+"_layer_head.json", folder=args.save_dir)        # -> ./deneme_1-2_3-4.npy
# Example use:
arr = np.load(saved_path)  # you saved output*100
# show_path_patching_heatmap(arr, title=f"Path patching (%) – {recv_str}", symmetric=True, origin="upper")
# or directly:
show_path_patching_heatmap(saved_path, title=f"Path patching (%)", path=saved_path.replace("npy", "html"))

