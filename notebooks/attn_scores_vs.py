from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import sys, os
sys.path.append(os.path.abspath(".."))  # go up one level to the root
from utils.utils import *
import numpy as np, matplotlib.pyplot as plt

model_name = "gpt2-small"
device ="cuda:1"
model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device = device
    )

attn_cache = {}
strong=moderate=nearzero=valid=anticopy=0  # before the layer/head loops
corr_map = np.full((model.cfg.n_layers, model.cfg.n_heads), np.nan)

for layer_i in range(model.cfg.n_layers):
    for head_i in range(model.cfg.n_heads):
    
# layer_i, head_i = 8, 11  # 0-indexed

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

        Sentence: The Islam religion has over a billion followers worldwide.,
        POS tag: NATIONALITY, RELIGIOUS, or POLITICAL GROUP, 
        Answer: Islam

        Sentence: {sentence}
        POS tag: {tag}
        Answer:"""

        per_sentence_scores = []
        model.eval()

        ner_solid_samples = load_json("../../pos_cf_datasets/ner_correct_alot.json")

        model.eval()
        L, H = layer_i, head_i  # 0-based

        attn_totals = []  # total attention to answer
        logit_totals = []  # total logit towards answer's first token
        for item in ner_solid_samples:

            sent, ans = item["Sentence"].replace('"','')+".", item["Answer"].replace('"','')
            tokens = model.to_tokens(sent, prepend_bos=False).to(device); seq_ids = tokens[0]
            
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in {f"blocks.{L}.attn.hook_pattern", f"blocks.{L}.ln1.hook_normalized"})
            a = cache[f"blocks.{L}.attn.hook_pattern"][0, H]
            t_end = a.shape[0]-1
            attn_row = a[t_end]

            xL = cache[f"blocks.{L}.ln1.hook_normalized"][0]
            M_all = (xL @ model.W_V[L,H]) @ model.W_O[L,H]
            o = attn_row.unsqueeze(1)*M_all

            span=None
            cands=[model.to_tokens(ans,prepend_bos=False)[0].tolist(), model.to_tokens(" "+ans,prepend_bos=False)[0].tolist()]
            cands=[p for i,p in enumerate(cands) if p and p not in cands[:i]]

            seq_list=seq_ids.tolist()
            for pat in cands:
                for s in range(len(seq_list)-len(pat)+1):
                    if seq_list[s:s+len(pat)]==pat: span=list(range(s,s+len(pat))); break
                if span is not None: break
            if not span: continue

            idx = torch.tensor(span, device=attn_row.device)
            attn_totals.append(attn_row.index_select(0,idx).sum().item())
            name_id = model.to_tokens(ans, prepend_bos=False)[0][0].item()
            WU_name = model.W_U[:, name_id]
            logit_totals.append((o.index_select(0,idx) @ WU_name).sum().item())

        q1,q99 = np.percentile(np.array(logit_totals),[1,99])
        mask = (np.array(logit_totals)>=q1)&(np.array(logit_totals)<=q99)
        r = np.corrcoef(np.array(attn_totals)[mask], np.array(logit_totals)[mask])[0,1]

        #r = np.corrcoef(attn_totals, logit_totals)[0,1] if len(attn_totals)>=2 and np.std(attn_totals)>0 and np.std(logit_totals)>0 else float("nan")
        if not np.isnan(r):
            valid+=1
            if r>=0.7: strong+=1
            elif 0.3<=r<0.7: moderate+=1
            if -0.2<=r<=0.2: nearzero+=1
            if r<=-0.3: anticopy+=1

        corr_map[layer_i, head_i] = r
        print(f"Layer {layer_i} Head {head_i}: ", end="")
        print("corr(total attention to answer, total logit toward answer’s first token) =", r, "N =", len(attn_totals))

        if not np.isnan(r) and r>=0.7:
            xs, ys, top = [], [], []
            for item in ner_solid_samples:
                sent, ans = item["Sentence"].replace('"','')+".", item["Answer"].replace('"','')
                tokens = model.to_tokens(sent, prepend_bos=False).to(device); seq_ids = tokens[0]
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in {f"blocks.{L}.attn.hook_pattern", f"blocks.{L}.ln1.hook_normalized"})
                a = cache[f"blocks.{L}.attn.hook_pattern"][0,H]; attn_row = a[a.shape[0]-1]
                o = (attn_row.unsqueeze(1)*((cache[f"blocks.{L}.ln1.hook_normalized"][0]@model.W_V[L,H])@model.W_O[L,H]))
                span=None; cands=[model.to_tokens(ans,False)[0].tolist(), model.to_tokens(" "+ans,False)[0].tolist()]; cands=[p for i,p in enumerate(cands) if p and p not in cands[:i]]
                seq_list=seq_ids.tolist()
                for pat in cands:
                    for s in range(len(seq_list)-len(pat)+1):
                        if seq_list[s:s+len(pat)]==pat: span=list(range(s,s+len(pat))); break
                    if span is not None: break
                if not span: continue
                idx = torch.tensor(span, device=attn_row.device)
                WU_name = model.W_U[:, model.to_tokens(ans, False)[0][0].item()]
                xs += attn_row.index_select(0,idx).tolist()
                contrib = (o.index_select(0,idx) @ WU_name)          # [m]
                ys += contrib.tolist()
                #top.append((contrib.sum().item(), sent, item["Answer"]))
                score = (o.index_select(0,idx) @ WU_name).max().item()   # use .sum().item() for total
                top.append((score, sent, item["Answer"], item.get("POS tag","?")))
                #top.append((contrib.sum().item(), sent, item["Answer"], item.get("POS tag","?")))

            # save scatter
            fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
            ax.scatter(xs, ys, alpha=0.35, s=30)  # was s=10 -> bigger dots

            ax.set_xlabel("Attn Prob on Name", fontsize=13)
            ax.set_ylabel("Dot w Name Embed", fontsize=13)
            ax.set_title(f"Projection of {L}.{H} along name embed vs attention", fontsize=14)

            # tick label sizes
            ax.tick_params(axis="both", labelsize=11)

            # print r on the image (top-left, inside axes)
            ax.text(
                0.02, 0.98,
                f"r = {r:.3f}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=13,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none")
            )

            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(f"figures_gpt/name_mover_scatter_L{L}_H{H}.png", dpi=300)
            plt.close(fig)
            # print top-10 examples by total dot toward name
            for val, sent, ans, tag in sorted(top, key=lambda x: x[0], reverse=True)[:10]:
                print(f"[L{L}H{H}] {val:.3f} :: {sent}  [Answer: {ans}]  [Tag: {tag}]")

print(f"Strong: {100*strong/valid:.1f}%  Moderate: {100*moderate/valid:.1f}%  Near-zero: {100*nearzero/valid:.1f}%  (N={valid}) Anticopy: {100*anticopy/valid:.1f}%")

m = np.ma.masked_invalid(corr_map)
plt.imshow(m, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='corr(attn→answer, logit→answer₁)')
plt.xticks(range(model.cfg.n_heads), range(model.cfg.n_heads))
plt.yticks(range(model.cfg.n_layers), [f"L{i}" for i in range(model.cfg.n_layers)])
plt.xlabel("Head"); plt.ylabel("Layer"); plt.title("Head correlation heatmap")
plt.tight_layout()
plt.savefig("figures_gpt/head_corr_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()