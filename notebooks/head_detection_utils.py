from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import sys, os
sys.path.append(os.path.abspath(".."))  # go up one level to the root
from utils.utils import *

"""
This script computes the attention probability (attn weights softmaxed) from the final token 
(the one predicting the next token)
to the answer span, for various NER examples.
"""

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

per_sentence_scores = []
model.eval()

ner_solid_samples = load_json("../data/ner_correct.json")

for item in ner_solid_samples:

    item["Sentence"] = item["Sentence"].replace("\"","") + "."
    item["Answer"] = item["Answer"].replace("\"","")
    item["POS tag"] = item["POS tag"].replace("\"","")

    tokens = model.to_tokens(item["Sentence"], prepend_bos=False).to(device)  # [1, seq]

    names = lambda n: (
        f"blocks.{layer_i}.attn" in n
    )
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=names)

    a = cache[f"blocks.{layer_i}.attn.hook_pattern"][0, head_i]  # [seq, seq]
    ans_ids = model.to_tokens(" "+item["Answer"], prepend_bos=False)
    seq_ids = tokens[0].tolist() 
    # Try both with and without a leading space
    ans = item["Answer"].replace('"','')
    candidates = [
        model.to_tokens(ans, prepend_bos=False)[0].tolist(),
        model.to_tokens(" " + ans, prepend_bos=False)[0].tolist(),
    ]
    # De-duplicate empty/identical patterns
    candidates = [pat for i, pat in enumerate(candidates) if pat and pat not in candidates[:i]]

    def find_span_by_ids(seq_ids, pat):
        L = len(pat)
        for s in range(len(seq_ids) - L + 1):
            if seq_ids[s:s+L] == pat:
                return list(range(s, s+L))
        return None

    span = None
    for pat in candidates:
        span = find_span_by_ids(seq_ids, pat)
        if span is not None:
            break

    print("Span for answer", item["Answer"], "is", span)

    if span is None:
        print("No span match for:", ans)
        continue

    # find contiguous span of the answer
    # find ALL occurrences of the answer subtoken pattern; collect their positions
    src_positions = span
    if src_positions is not None:
        attn_end_to_ans = a[:, src_positions].detach().cpu()

        # destination = last context step (predicts the next token = [end])
        t_end = len(seq_ids) - 1 # destination token

        # store ALL rows → selected source columns (variable length per sentence is fine)
        per_sentence_scores.append(attn_end_to_ans)
    
per_sentence_means = []
for cols_slice in per_sentence_scores:
    t_end = cols_slice.shape[0] - 1
    k = cols_slice.shape[1]
    per_sentence_means.append(np.nan if k == 0 else cols_slice[t_end].max().item())

print("Mean attention prob from [end] to answer:", np.nanmean(per_sentence_means))