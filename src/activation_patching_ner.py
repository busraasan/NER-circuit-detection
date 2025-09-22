from functools import partial
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
import torch.nn.functional as F

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
seed = 42
null_task = True
metric = "kl_divergence"  # "kl_divergence" or "ioi"
multi_token_ans = False  # skip examples with multi-token answers
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

dataset_name = "ner_dataset_20each.json"
with open("../../pos_cf_datasets/"+dataset_name, "r", encoding="utf-8") as f:
    tv_data = json.load(f)
tag_set = [
    "EVENT", "LOCATION", "MONEY",
    "NATIONALITY, RELIGIOUS, or POLITICAL GROUP",
    "ORGANIZATION", "NUMERICAL", "PERSON",
    "PRODUCT", "TIME"
]

template = """Sentence: Apple announced a new iPhone during its annual product launch event.,
Entity: PRODUCT,
Answer: iPhone

Sentence: Barack Obama delivered a keynote speech at the conference.,
Entity: PERSON,
Answer: Barack Obama

Sentence: Tesla invested over 2 billion dollars in a new gigafactory in Germany.,
Entity: MONEY,
Answer: 2 billion dollars

Sentence: The concert will take place at 8 p.m. on Saturday.,
Entity: TIME,
Answer: 8 p.m. on Saturday

Sentence: The Eiffel Tower is located in Paris.,
Entity: LOCATION,
Answer: Paris

Sentence: The Olympic Games in Tokyo attracted thousands of visitors despite the pandemic.,
Entity: EVENT,
Answer: Olympic Games

Sentence: The recipe calls for 200 grams of sugar and 3 eggs.,
Entity: NUMERICAL,
Answer: 200 grams

Sentence: Google has opened a new research center in Zurich to focus on AI development.,
Entity: ORGANIZATION,
Answer: Google

Sentence: The American have a long history of culinary excellence.,
Entity: NATIONALITY, RELIGIOUS, or POLITICAL GROUP,
Answer: American

Sentence: The Islam religion has over a billion followers worldwide.,
Entity: NATIONALITY, RELIGIOUS, or POLITICAL GROUP,
Answer: Islam

Sentence: {sentence}
Entity: {tag}
Answer:"""

all_results = []

for i, item in enumerate(tv_data):

    item["Sentence"] = item["Sentence"].replace("\"","")
    item["Answer"] = item["Answer"].replace("\"","")
    item["POS tag"] = item["POS tag"].replace("\"","")

    clean_prompt = template.format(sentence=item["Sentence"], tag=item["POS tag"])

    corrupted_prompt = template.format(
        sentence=item["Sentence"],
        tag="NULL"
    )

    # Per-example tokenization (no growing lists)
    if multi_token_ans:
        clean_tokens    = model.to_tokens(clean_prompt+" "+item["Answer"], prepend_bos=False)
        corrupted_tokens= model.to_tokens(corrupted_prompt+" "+item["Answer"], prepend_bos=False)
    else:
        clean_tokens    = model.to_tokens(clean_prompt, prepend_bos=False)
        corrupted_tokens= model.to_tokens(corrupted_prompt, prepend_bos=False)

    # if clean_tokens.shape[1] != corrupted_tokens.shape[1]:
    #     print(f"Skipping example {i} due to length mismatch")
    #     continue

    # One-token gold id (skip if multi-token)
    ans_ids = model.to_tokens(item["Answer"], prepend_bos=False)[0].tolist()

    if multi_token_ans:
        gold_id = torch.tensor(ans_ids, device=model.cfg.device)  # [A]
    else:
        if len(ans_ids) > 1:
            continue
        else:
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
    
    @torch.no_grad()
    def _kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor, dim: int = -1):
        """
        D_KL(P || Q) where P = softmax(p_logits), Q = softmax(q_logits).
        Reduces over vocab only; preserves batch dim.
        """
        
        log_probs_p = torch.log_softmax(p_logits, dim=dim)
        log_probs_q = torch.log_softmax(q_logits, dim=dim)

        return F.kl_div(log_probs_q, log_probs_p, log_target=True, reduction="none").sum(dim=dim)
    
    @torch.no_grad()
    def multitoken_logits_to_KL_div(
        logits: torch.Tensor,
        clean_logits: torch.Tensor,
        answer_tokens,
        per_prompt: bool = True,
        is_multitoken: bool = True,
        reduce_over_span: str = "mean",   # "mean" or "sum"
    ):
        """
        KL over a multi-token answer span using teacher forcing.

        Args:
            logits:        [batch, seq_len, vocab]  (patched/corrupted run)
            clean_logits:  [batch, seq_len, vocab]  (clean/reference run)
            per_prompt:    if True, return [batch]; else return scalar mean over batch.
            reduce_over_span: "mean" (default) or "sum" over the answer span positions.

        Returns:
            Tensor of shape [batch] if per_prompt else a scalar tensor.
        """
        if len(answer_tokens) == 0:
            raise ValueError("answer_token_indices must contain at least one index.")

        # For each answer token at j, the model predicts it at position j-1
        # answer length
        L = len(answer_tokens) if not hasattr(answer_tokens, "shape") else answer_tokens.shape[-1]
        if L <= 0:
            raise ValueError("answer_tokens must have length >= 1")

        Bc, Tc, Vc = clean_logits.shape
        Bp, Tp, Vp = logits.shape
        if Bc != Bp:
            raise ValueError("Batch sizes must match for clean and patched logits.")

        # Need at least one prefix token before the first answer token in both runs
        if Tc - L - 1 < 0 or Tp - L - 1 < 0:
            raise ValueError("Not enough prefix tokens before the answer in one of the runs.")
        kl_list = []

        for k in range(L):
            
            if is_multitoken:
                pos_c = (Tc - L - 1) + k   # predicts clean answer token k
                pos_p = (Tp - L - 1) + k   # predicts patched answer token k
            else:
                pos_c = -1
                pos_p = -1

            kl_k = _kl_from_logits(
                clean_logits[:, pos_c, :],   # P (reference distribution)
                logits[:, pos_p, :],         # Q (current (patched) distribution)
                dim=-1
            )
            kl_list.append(kl_k)

        kl_stack = torch.stack(kl_list, dim=0)  # [span_len, batch]

        if reduce_over_span == "sum":
            kl_over_span = kl_stack.sum(dim=0)   # [batch]
        else:
            kl_over_span = kl_stack.mean(dim=0)  # [batch]

        return kl_over_span if per_prompt else kl_over_span.mean()

    
    @torch.no_grad()
    def probs_at(logits, pos):
        return torch.softmax(logits[:, pos, :], dim=-1)

    with torch.no_grad():
        clean_logits, clean_cache       = model.run_with_cache(clean_tokens)
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

        CLEAN_BASELINE     = get_logit_diff(clean_logits, gold_id).item()
        CORRUPTED_BASELINE = get_logit_diff(corrupted_logits, gold_id).item()

        target_pos = -1
        P_CLEAN     = probs_at(clean_logits,     target_pos).detach()
        P_CORRUPTED = probs_at(corrupted_logits, target_pos).detach()

        def kl_to_clean_metric(logits):
            Q = probs_at(logits, target_pos)
            return -_kl_from_logits(P_CLEAN, Q)

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
        
        index_df = pd.DataFrame(
            [{"layer": L, "head": H, "pos": target_pos}
             for L in range(model.cfg.n_layers)
             for H in range(model.cfg.n_heads)]
        )

        multitoken_logits_to_KL_div_partial = partial(
            multitoken_logits_to_KL_div,
            clean_logits=clean_logits,
            answer_tokens=ans_ids,
            per_prompt=True,
            is_multitoken=True,
            reduce_over_span="mean",
        )

        if metric == "kl_divergence" and multi_token_ans:
            results_z = generic_activation_patch(
                model=model,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                patching_metric=multitoken_logits_to_KL_div_partial,
                patch_setter=z_patch_setter,
                activation_name="z",
                index_df=index_df,
            )

        if metric == "kl_divergence":
            results_z = generic_activation_patch(
                model=model,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                patching_metric=kl_to_clean_metric,  # TL will report patched - corrupted
                patch_setter=z_patch_setter,
                activation_name="z",
                index_df=index_df,
            )
        else:
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

mn = all_results.amin()
mx = all_results.amax()
all_results_01 = (all_results - mn) / (mx - mn + 1e-12)   # -> [0, 1]

all_results = all_results_01.view(N, L, H)   # [N, 18, 8]
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

save_path = f"result_act_patching_NER.html"
if save_path != None:
    fig.write_html(save_path)

