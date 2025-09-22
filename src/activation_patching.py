from transformer_lens import HookedTransformer
import torch
import json
import random
from transformer_lens.patching import generic_activation_patch
import pandas as pd
from transformer_lens import patching
import plotly.express as px
import transformer_lens.utils as utils
import numpy as np

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
seed = 42
null_task = True
random.seed(seed)

def imshow(tensor, xaxis="", yaxis="", labels=None, **kwargs):
    # Build a single labels dict
    merged_labels = {"x": xaxis, "y": yaxis}
    if labels is not None:
        merged_labels.update(labels)

    fig = px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels=merged_labels,
        **kwargs
    )
    return fig

dataset_name = "ner_dataset_100each.json"
with open("../../pos_cf_datasets/"+dataset_name, "r", encoding="utf-8") as f:
    tv_data = json.load(f)
tag_set = [
    "EVENT", "LOCATION", "MONEY",
    "NATIONALITY, RELIGIOUS, or POLITICAL GROUP",
    "ORGANIZATION", "NUMERICAL", "PERSON",
    "PRODUCT", "TIME"
]

template = """Sentence: Apple announced a new iPhone during its annual product launch event.,
POS tag: PRODUCT,
Answer: iPhone

Sentence: Barack Obama delivered a keynote speech at the conference.,
POS tag: PERSON,
Answer: Barack Obama

Sentence: Tesla invested over 2 billion dollars in a new gigafactory in Germany.,
POS tag: MONEY,
Answer: 2 billion dollars

Sentence: The concert will take place at 8 p.m. on Saturday.,
POS tag: TIME,
Answer: 8 p.m. on Saturday

Sentence: The Eiffel Tower is located in Paris.,
POS tag: LOCATION,
Answer: Paris

Sentence: The Olympic Games in Tokyo attracted thousands of visitors despite the pandemic.,
POS tag: EVENT, 
Answer: Olympic Games

Sentence: The recipe calls for 200 grams of sugar and 3 eggs.,
POS tag: NUMERICAL, 
Answer: 200 grams

Sentence: Google has opened a new research center in Zurich to focus on AI development.,
POS tag: ORGANIZATION, 
Answer: Google

Sentence: The American have a long history of culinary excellence.,
POS tag: NATIONALITY, RELIGIOUS, or POLITICAL GROUP, 
Answer: American

Sentence: The Islam religion has over a billion followers worldwide.,
POS tag: NATIONALITY, RELIGIOUS, or POLITICAL GROUP, 
Answer: Islam

Sentence: {sentence}
POS tag: {tag}
Answer:"""

all_results = []

for i, item in enumerate(tv_data):

    item["Sentence"] = item["Sentence"].replace("\"","")
    item["Answer"] = item["Answer"].replace("\"","")
    item["POS tag"] = item["POS tag"].replace("\"","")

    clean_prompt = template.format(sentence=item["Sentence"], tag=item["POS tag"])

    corrupted_prompt = template.format(
        sentence=item["Sentence"],
        tag="null"
    )

    # Per-example tokenization (no growing lists)
    clean_tokens    = model.to_tokens(clean_prompt, prepend_bos=False)
    corrupted_tokens= model.to_tokens(corrupted_prompt, prepend_bos=False)

    if clean_tokens.shape[1] != corrupted_tokens.shape[1]:
        print(f"Skipping example {i} due to length mismatch")
        continue

    # One-token gold id (skip if multi-token)
    ans_ids = model.to_tokens(item["Answer"], prepend_bos=False)[0].tolist()
    if len(ans_ids) != 1:
        continue
    gold_id = torch.tensor([ans_ids[0]], device=model.cfg.device)  # [1]

    def get_logit_diff(logits, gold_ids):
        if logits.ndim == 3: logits = logits[:, -1, :]  # [B, V]
        B, V = logits.shape
        rows = torch.arange(B, device=logits.device)
        gold = logits[rows, gold_ids]                   # [B]
        # hardest negative over vocab excluding gold
        logits_no_gold = logits.clone()
        logits_no_gold[rows, gold_ids] = -float("inf")
        neg = logits_no_gold.max(dim=1).values
        return (gold - neg).mean()

    with torch.no_grad():
        clean_logits, clean_cache       = model.run_with_cache(clean_tokens)
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

        CLEAN_BASELINE     = get_logit_diff(clean_logits, gold_id).item()
        CORRUPTED_BASELINE = get_logit_diff(corrupted_logits, gold_id).item()

        def ioi_metric_clean(logits):
            num = get_logit_diff(logits, gold_id)
            return (num - CLEAN_BASELINE) / (CORRUPTED_BASELINE - CLEAN_BASELINE + 1e-8)
            # return (num - CORRUPTED_BASELINE) / (CLEAN_BASELINE - CORRUPTED_BASELINE + 1e-8)

        def ioi_metric(logits):
            num = get_logit_diff(logits, gold_id)
            return (num - CORRUPTED_BASELINE) / (CLEAN_BASELINE - CORRUPTED_BASELINE + 1e-8)

        # z patch setter (per-head at last pos)
        def z_patch_setter(corrupted_act, index_row, clean_act):
            _, H, P = index_row
            corrupted_act[:, P, H, :] = clean_act[:, P, H, :]
            return corrupted_act

        def z_patch_setter_clean(clean_act, index_row, corrupted_act):
            _, H, P = index_row
            clean_act[:, P, H, :] = corrupted_act[:, P, H, :]
            return corrupted_act


        target_pos = -1
        index_df = pd.DataFrame(
            [{"layer": L, "head": H, "pos": target_pos}
             for L in range(model.cfg.n_layers)
             for H in range(model.cfg.n_heads)]
        )

        results_z = generic_activation_patch(
            model=model,
            corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache,
            patching_metric=ioi_metric,   # corrupted; TL computes patched - corrupted
            patch_setter=z_patch_setter,
            activation_name="z",
            index_df=index_df,
        )

    # move to CPU and free big objects
    all_results.append(results_z.to("cpu"))
    del clean_cache, corrupted_cache, clean_logits, corrupted_logits
    torch.cuda.empty_cache()

# stack on CPU
all_results = torch.stack(all_results, dim=0)   # [N, L, H]
mean_results = all_results.mean(dim=0)

N, flat = all_results.shape   # e.g. 16, 144
L, H = model.cfg.n_layers, model.cfg.n_heads   # 18, 8

all_results = all_results.view(N, L, H)   # [N, 18, 8]
mean_results = all_results.mean(dim=0)    # [18, 8]

title = "Activation patching effect (z, last position, averaged over coarse POS dataset)"
fig = imshow(
    mean_results,
    xaxis="Head",
    yaxis="Layer",
    title=title,
    labels={"color": "Logit diff"},
)

n_layers, n_heads = mean_results.shape

fig.update_xaxes(title="Head", tickmode="array", tickvals=list(range(n_heads)))
fig.update_yaxes(title="Layer", tickmode="array", tickvals=list(range(n_layers)))
fig.update_layout(title=title,
                    height=600,                     # keep same vertical size
                width=n_heads * 50)

save_path = f"result_act_patching_clean2corr_dataset_size_{len(tv_data)}_seed_{seed}.html"
if save_path != None:
    fig.write_html(save_path)

