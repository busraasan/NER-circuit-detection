from transformer_lens import HookedTransformer
import torch, json, numpy as np

model_name = "gpt2-small"
device = "cuda:1"
model = HookedTransformer.from_pretrained(
    model_name, center_unembed=True, center_writing_weights=True, fold_ln=True, device=device
)
model.eval()

# Load data
with open("../data/ner_correct.json", "r") as f:
    ner_solid_samples = json.load(f)

L_total, H_total = model.cfg.n_layers, model.cfg.n_heads

def find_answer_span(seq_ids, ans_text):
    # Try with and without leading space; return list(range(start, end)) or None
    cands = [
        model.to_tokens(ans_text, prepend_bos=False)[0].tolist(),
        model.to_tokens(" " + ans_text, prepend_bos=False)[0].tolist(),
    ]
    # de-duplicate
    cands = [p for i, p in enumerate(cands) if p and p not in cands[:i]]
    seq_list = seq_ids.tolist()
    for pat in cands:
        Lp = len(pat)
        for s in range(len(seq_list) - Lp + 1):
            if seq_list[s:s+Lp] == pat:
                return list(range(s-1, s+Lp-1))
    return None

# Collect per-head per-example means
per_head_means = { (L,H): [] for L in range(L_total) for H in range(H_total) }

for item in ner_solid_samples:
    sent = item["Sentence"].replace('"','') + "."
    ans  = item["Answer"].replace('"','')

    tokens  = model.to_tokens(sent, prepend_bos=False).to(device)   # [1, T]
    seq_ids = tokens[0]                                          # [T]

    # Cache attention patterns for ALL layers (all heads)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: n.endswith("attn.hook_pattern")
        )

    # Locate answer span once
    span = find_answer_span(seq_ids, ans)
    if span is None:
        continue
    idx = torch.tensor(span, device=device, dtype=torch.long)

    # For each layer/head: take attention from the last position to source tokens, then average over span
    for L in range(L_total):
        A = cache[f"blocks.{L}.attn.hook_pattern"][0]   # [n_heads, T, T]
        attn_from_last = A[:, -1, :]                    # [n_heads, T]
        attn_span = attn_from_last.index_select(1, idx) # [n_heads, |span|]
        means = attn_span.mean(dim=1)                   # [n_heads]
        for H in range(H_total):
            per_head_means[(L,H)].append(float(means[H].item()))

# ---- print results ----
print("Per-head average attention over answer tokens (dataset summary):")
for L in range(L_total):
    for H in range(H_total):
        vals = per_head_means[(L,H)]
        if len(vals) == 0:
            print(f"L{L}H{H}: N=0 (no spans found)")
            continue
        arr = np.array(vals, dtype=float)
        print(f"L{L}H{H}: mean={arr.mean():.4f}  std={arr.std(ddof=1):.4f}  N={len(arr)}")
