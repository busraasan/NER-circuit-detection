from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import sys, os
sys.path.append(os.path.abspath(".."))  # go up one level to the root
from utils.utils import *

model_name = "gpt2-small"
device ="cuda:0"
model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device = device
    )

# targets = []
# for name, module in model.named_modules():
#     low = name.lower()
#     if any(k in low for k in ["attn", "attention", "self_attn", "self-attention"]):
#         targets.append(name)
# for i, n in enumerate(targets):
#     print(f"{i:03d}  {n}")


attn_cache = {}
layer_i, head_i = 10, 7  # 0-indexed

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

per_sentence_scores = []
model.eval()

ner_solid_samples = load_json("../../pos_cf_datasets/ner_correct.json")

model.eval()
L, H = layer_i, head_i  # 0-based

corr_all_sent   = []   # correlation over all j
corr_answer_only = []  # correlation only over answer-span j (optional)
ans_A_sum, ans_C_sum, ans_C_unrouted_sum = [], [], []
ans_A_mean, ans_C_mean, ans_C_unrouted_mean = [], [], []

for item in ner_solid_samples:
    sent = item["Sentence"].replace('"','') + "."
    ans  = item["Answer"].replace('"','')

    # --- tokenize ---
    tokens = model.to_tokens(sent, prepend_bos=False).to(device)    # [1, seq]
    seq_ids = tokens[0]                                             # [seq] (tensor)

    # --- cache what we need for this head ---
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: n in {
                f"blocks.{L}.attn.hook_pattern",      # attention probs a
                f"blocks.{L}.ln1.hook_normalized",    # x^{(L)} (post-LN1)
            }
        )

    # Attention probabilities for this head: [seq, seq]
    a = cache[f"blocks.{L}.attn.hook_pattern"][0, H]        # rows=t (dest), cols=j (source)
    t_end = a.shape[0] - 1
    attn_row = a[t_end]                                     # [seq]

    # Layer-L input (post-LN1) to attention: [seq, d_model]
    xL = cache[f"blocks.{L}.ln1.hook_normalized"][0]        # [seq, d_model]

    # Head OV circuit
    W_V = model.W_V[L, H]                                   # [d_model, d_head]
    W_O = model.W_O[L, H]                                   # [d_head, d_model]

    # Per-source message content in model space (without routing):
    # M_j = (x_j @ W_V) @ W_O  -> [seq, d_model]
    M_all = (xL @ W_V) @ W_O

    # Route by attention from t_end: o_{t<-j} = a[t_end, j] * M_j
    o_end_from_j = attn_row.unsqueeze(1) * M_all            # [seq, d_model]

    # Build unembedding row for each source token in this sequence
    # TransformerLens: model.W_U has shape [d_model, vocab]
    W_U_seq = model.W_U[:, seq_ids].T                       # [seq, d_model] (one row per j)

    # Copy scores: < o_{t<-j}, W_U[token_j] >
    copy_scores = (o_end_from_j * W_U_seq).sum(dim=-1)      # [seq]
    unrouted_copy = (M_all * W_U_seq).sum(dim=-1)          # <M_j, W_U[token_j]>

    # find span as you already do -> `span` = list of indices or None
    # ---- compute across-examples correlations ----

    # ------- correlation over ALL positions j -------
    # Guard against zero-variance (corr undefined)
    if torch.var(attn_row) > 0 and torch.var(copy_scores) > 0:
        mat = torch.stack([attn_row, copy_scores], dim=0)   # [2, seq]
        corr = torch.corrcoef(mat)[0, 1].item()
        corr_all_sent.append(corr)
    else:
        corr_all_sent.append(np.nan)

    # ------- OPTIONAL: correlation only over answer span j -------
    # Find answer span (try with and without leading space)
    span = None
    cands = [
        model.to_tokens(ans, prepend_bos=False)[0].tolist(),
        model.to_tokens(" " + ans, prepend_bos=False)[0].tolist(),
    ]
    # de-duplicate
    cands = [p for i, p in enumerate(cands) if p and p not in cands[:i]]

    seq_list = seq_ids.tolist()
    for pat in cands:
        Lp = len(pat)
        for s in range(len(seq_list) - Lp + 1):
            if seq_list[s:s+Lp] == pat:
                span = list(range(s, s+Lp))
                break
        if span is not None:
            break

    if span:
        idx = torch.tensor(span, device=attn_row.device, dtype=torch.long)
        attn_span = attn_row.index_select(0, idx)
        copy_span = copy_scores.index_select(0, idx)
        if torch.var(attn_span) > 0 and torch.var(copy_span) > 0:
            mat = torch.stack([attn_span, copy_span], dim=0)
            corr_ans = torch.corrcoef(mat)[0, 1].item()
            corr_answer_only.append(corr_ans)
        else:
            corr_answer_only.append(np.nan)

    if span:
        idx = torch.tensor(span, device=attn_row.device, dtype=torch.long)
        a_span       = attn_row.index_select(0, idx)          # [m]
        c_span       = copy_scores.index_select(0, idx)       # [m] (routed)
        c_unr_span   = unrouted_copy.index_select(0, idx)     # [m]
        m = a_span.numel()

        # Per-example aggregates (choose sum or mean; I save both)
        ans_A_sum.append(a_span.sum().item())
        ans_C_sum.append(c_span.sum().item())
        ans_C_unrouted_sum.append(c_unr_span.sum().item())

        ans_A_mean.append(a_span.mean().item())
        ans_C_mean.append(c_span.mean().item())
        ans_C_unrouted_mean.append(c_unr_span.mean().item())

def safe_corr(x, y):
    x = np.array(x); y = np.array(y)
    if np.std(x) == 0 or np.std(y) == 0 or len(x) < 2: 
        return np.nan
    return np.corrcoef(x, y)[0, 1]

r_sum_routed      = safe_corr(ans_A_sum,  ans_C_sum)
r_sum_unrouted    = safe_corr(ans_A_sum,  ans_C_unrouted_sum)
r_mean_routed     = safe_corr(ans_A_mean, ans_C_mean)
r_mean_unrouted   = safe_corr(ans_A_mean, ans_C_unrouted_mean)

print("Across-examples (answer span):")
print("  SUM  : corr(attn_sum,  copy_sum)      =", r_sum_routed)
print("  SUM  : corr(attn_sum,  unrouted_sum)  =", r_sum_unrouted)
print("  MEAN : corr(attn_mean, copy_mean)     =", r_mean_routed)
print("  MEAN : corr(attn_mean, unrouted_mean) =", r_mean_unrouted)
