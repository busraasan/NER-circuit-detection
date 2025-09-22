from torch.utils.data import DataLoader, Dataset
from functools import partial
import torch
from rich.progress import track
from transformer_lens import utils
import torch.nn.functional as F

def patch_head_vector_at_pos(
    clean_head_vector,
    hook,
    head_index,
    pos_index,
    corrupted_cache,
    answer_tokens=None,
    is_multitoken=False):

    if pos_index == None:
        clean_head_vector[:, :, head_index, :] = corrupted_cache[hook.name][:, :, head_index, :]
    
    elif is_multitoken:

        if answer_tokens is None:
            raise ValueError("answer_tokens must be provided for multi-token patching")
        L = len(answer_tokens)
        Tc = clean_head_vector.shape[1]  # sequence length
        Tcorr = corrupted_cache[hook.name].shape[1]

        # Compute the answer token positions
        positions = [(Tc - L - 1) + k for k in range(L)]
        positions_corr = [(Tcorr - L - 1) + k for k in range(L)]

        for pos, pos_corr in zip(positions, positions_corr):
            clean_head_vector[:, pos, head_index, :] = corrupted_cache[hook.name][:, pos_corr, head_index, :]
    else:
        clean_head_vector[:, pos_index, head_index, :] = corrupted_cache[hook.name][:, pos_index, head_index, :]
    return clean_head_vector

def cache_activation_hook(
    activation,
    hook,
    my_cache={}):
    #print("HOOK NAME:", hook.name)
    my_cache[hook.name] = activation
    return activation

def patch_full_residual_component(
    corrupted_residual_component, #: Float[torch.Tensor, "batch pos d_model"],
    hook,
    pos_index,
    corrupted_cache,
    answer_tokens=None,
    is_multitoken=False):
    
    if pos_index == None:
        corrupted_residual_component[:, :, :] = corrupted_cache[hook.name][:, :, :]
    elif is_multitoken:

        if answer_tokens is None:
            raise ValueError("answer_tokens must be provided for multi-token patching")
        L = len(answer_tokens)
        Tc = corrupted_residual_component.shape[1]  # sequence length

        # Compute the answer token positions
        positions = [(Tc - L - 1) + k for k in range(L)]

        for pos in positions:
            corrupted_residual_component[:, pos, :] = corrupted_cache[hook.name][:, pos, :]
    else:
        corrupted_residual_component[:, pos_index, :] = corrupted_cache[hook.name][:, pos_index, :]
    return corrupted_residual_component

def logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=True):
    # Only the final logits are relevant for the answer
    final_logits = logits[:, -1, :]

    if isinstance(answer_tokens, list):
        diffs = []
        for ans_tok in answer_tokens:
            ans_logits = final_logits.gather(dim=-1, index=ans_tok)  # [batch, 2]
            diff = ans_logits[:, 0] - ans_logits[:, 1]              # [batch]
            diffs.append(diff)
        # stack and average across list dimension
        answer_logit_diff = torch.stack(diffs, dim=0).mean(dim=0)   # [batch]
    else:
        # Normal single-pair case
        ans_logits = final_logits.gather(dim=-1, index=answer_tokens)  # [batch, 2]
        answer_logit_diff = ans_logits[:, 0] - ans_logits[:, 1]        # [batch]

    if per_prompt:
        return answer_logit_diff
    else:
        return answer_logit_diff.mean()
    
import torch
import torch.nn.functional as F
from functools import partial

@torch.no_grad()
def teacher_forcing_loss_from_indices(
    logits: torch.Tensor,         # [B, S, V]
    targets: torch.Tensor,        # [B, S] (input_ids)
    start_indices: torch.Tensor,  # [B] (where to start predicting)
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """
    Cross-entropy over next-token predictions, starting at start_indices[i] for each item.
    For each i:
        loss_i = CE(logits[i, pos-1:-1, :], targets[i, pos:])
    Returns mean over batch (scalar tensor on logits.device).
    """
    assert logits.ndim == 3 and targets.ndim == 2
    B, S, V = logits.shape
    device = logits.device

    total = torch.zeros((), device=device)
    count = 0

    for i in range(B):
        pos = int(start_indices[i].item())
        # Guard against degenerate ranges
        if pos <= 0 or pos >= S:
            continue

        logit_slice = logits[i, pos-1:-1, :]   # predicts tokens at positions
        label_slice = targets[i, pos:]         # gold tokens at

        if pad_token_id is not None:
            mask = (label_slice != pad_token_id)
            if mask.any():
                loss_i = F.cross_entropy(logit_slice[mask], label_slice[mask], reduction="mean")
                total += loss_i
                count += 1
        else:
            # Flatten over time
            loss_i = F.cross_entropy(
                logit_slice.reshape(-1, V),
                label_slice.reshape(-1),
                reduction="mean"
            )
            total += loss_i
            count += 1

    if count == 0:
        # Fallback (no valid positions) -> zero loss on device
        return torch.zeros((), device=device)

    return total / count


def tf_loss_denoising(
    logits: torch.Tensor,          # [B, S, V] patched run logits (required by act_patch)
    tokens: torch.Tensor,          # [B, S] tokens corresponding to *these* logits (e.g., corrupted_tokens)
    start_indices: torch.Tensor,   # [B]
    corr_loss: torch.Tensor,       # scalar baseline on corrupted run (precomputed)
    clean_loss: torch.Tensor,      # scalar baseline on clean run (precomputed)
    pad_token_id: int | None = None,
    return_tensor: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor | float:
    """
    Compute teacher-forcing loss on the patched logits and calibrate it:
        score = (corr_loss - patched_loss) / (corr_loss - clean_loss)
    Returns a scalar tensor if return_tensor=True, else float.
    """
    patched_loss = teacher_forcing_loss_from_indices(
        logits=logits,
        targets=tokens,
        start_indices=start_indices,
        pad_token_id=pad_token_id,
    )

    denom = (corr_loss - clean_loss)
    #print("corr loss:", corr_loss.item(), "patched_loss:", patched_loss.item(), "clean_loss:", clean_loss.item(), "denom:", denom.item())
    # Avoid divide-by-zero / negative-denominator edge cases
    # if torch.is_tensor(denom):
    #     denom = denom.clamp_min(eps)
    # else:
    #     denom = max(denom, eps)

    score = (corr_loss - patched_loss) / denom
    if score > 100000:
        print("corr_loss", corr_loss.item(), "patched_loss:", patched_loss.item(), "clean_loss:", clean_loss.item(), "denom:", denom.item())
    return score if return_tensor else float(score.item())

def tf_loss_noising(
    logits: torch.Tensor,          # [B, S, V] patched run logits (required by act_patch)
    tokens: torch.Tensor,          # [B, S] tokens corresponding to *these* logits (e.g., corrupted_tokens)
    start_indices: torch.Tensor,   # [B]
    corr_loss: torch.Tensor,       # scalar baseline on corrupted run (precomputed)
    clean_loss: torch.Tensor,      # scalar baseline on clean run (precomputed)
    pad_token_id: int | None = None,
    return_tensor: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor | float:
    """
    Compute teacher-forcing loss on the patched logits and calibrate it:
        score = (clean_loss - patched_loss) / (clean_loss - corr_loss)
    Returns a scalar tensor if return_tensor=True, else float.
    """
    patched_loss = teacher_forcing_loss_from_indices(
        logits=logits,
        targets=tokens,
        start_indices=start_indices,
        pad_token_id=pad_token_id,
    )

    denom = (clean_loss - corr_loss)
    #print("corr loss:", corr_loss.item(), "patched_loss:", patched_loss.item(), "clean_loss:", clean_loss.item(), "denom:", denom.item())
    # Avoid divide-by-zero / negative-denominator edge cases
    # if torch.is_tensor(denom):
    #     denom = denom.clamp_min(eps)
    # else:
    #     denom = max(denom, eps)

    # print("corr loss:", corr_loss.item(), "patched_loss:", patched_loss.item(), "clean_loss:", clean_loss.item(), "denom:", denom.item())
    score = (clean_loss - patched_loss) / denom
    # print("SCORE", score.item())
    if abs(score) > 1000:
        print("corr_loss", corr_loss.item(), "patched_loss:", patched_loss.item(), "clean_loss:", clean_loss.item(), "denom:", denom.item())
    return score if return_tensor else float(score.item())


    
def _kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor, dim: int = -1):
    """
    D_KL(P || Q) where P = softmax(p_logits), Q = softmax(q_logits).
    Reduces over vocab only; preserves batch dim.
    """
    
    log_probs_p = torch.log_softmax(p_logits, dim=dim)
    log_probs_q = torch.log_softmax(q_logits, dim=dim)
    p = log_probs_p.exp()
    topk = 5

    # print("Top tokens in P and Q")
    # values, indices = torch.topk(log_probs_p[0], 5)
    # for v, i in zip(values, indices):
    #     print(f"{i.item():5d} -> {v.item():.4f}")

    # values, indices = torch.topk(log_probs_q[0], 5)
    # for v, i in zip(values, indices):
    #     print(f"{i.item():5d} -> {v.item():.4f}")

    return F.kl_div(log_probs_q, log_probs_p, log_target=True, reduction="none").sum(dim=dim)

def multitoken_patch_effect_KL(
    patched_logits: torch.Tensor,
    clean_logits: torch.Tensor,
    corrupt_logits: torch.Tensor,
    answer_tokens: list[int],
    per_prompt: bool = True,
    is_multitoken: bool = False,
    reduce_over_span: str = "mean",   # "mean" or "sum"
):
    """
    Patching effect based on KL divergence over a multi-token answer span.

    Δ = D_KL(Pcl || P*) - D_KL(Pcl || Ppt)
    """

    if len(answer_tokens) == 0:
        raise ValueError("answer_tokens must contain at least one index.")

    # length of answer
    L = len(answer_tokens) if not hasattr(answer_tokens, "shape") else answer_tokens.shape[-1]
    if L <= 0:
        raise ValueError("answer_tokens must have length >= 1")

    Bc, Tc, Vc = clean_logits.shape
    Bp, Tp, Vp = patched_logits.shape
    Bx, Tx, Vx = corrupt_logits.shape
    if not (Bc == Bp == Bx):
        raise ValueError("Batch sizes must match across runs.")

    if Tc - L - 1 < 0 or Tp - L - 1 < 0 or Tx - L - 1 < 0:
        raise ValueError("Not enough prefix tokens before the answer in one of the runs.")

    diffs = []
    for k in range(L):
        if is_multitoken:
            pos_c = (Tc - L - 1) + k   # predicts token k
            pos_p = (Tp - L - 1) + k
            pos_x = (Tx - L - 1) + k
        else:
            pos_c = pos_p = pos_x = -1

        # KL(Pcl || P*)  -- clean vs corrupt
        kl_clean_corrupt = _kl_from_logits(
            clean_logits[:, pos_c, :], corrupt_logits[:, pos_x, :], dim=-1
        ) # over the vocab
        # KL(Pcl || Ppt) -- clean vs patched
        kl_clean_patched = _kl_from_logits(
            clean_logits[:, pos_c, :], patched_logits[:, pos_p, :], dim=-1
        ) # over the vocab

        # difference
        diffs.append(kl_clean_corrupt - kl_clean_patched) # for each answer token position calculate the difference.

    # [L, batch]
    diffs = torch.stack(diffs, dim=0) 

    if reduce_over_span == "sum":
        out = diffs.sum(dim=0)  # [batch]
    else:
        out = diffs.mean(dim=0)  # [batch]

    return out if per_prompt else out.mean()


def multitoken_logits_to_KL_div(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    answer_tokens,
    per_prompt: bool = True,
    is_multitoken: bool = False,
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

def kl_over_answer_span(
    #same as above but with start indices instead of assuming last L tokens
    clean_logits: torch.Tensor,    # [B, S_c, V]
    patched_logits: torch.Tensor,  # [B, S_p, V]
    start_clean: torch.Tensor,     # [B]  start idx of FIRST answer token in clean targets
    start_patched: torch.Tensor,   # [B]  start idx of FIRST answer token in patched targets
    answer_len: int,               # L
    clamp_min_start: int = 1,      # match your TF behavior (1 if no BOS; 0 if BOS)
    reduce_over_span: str = "mean",# "mean" or "sum"
    per_prompt: bool = True        # return [B] or scalar
):
    """
    Computes KL(P || Q) between clean (P) and patched (Q) distributions at the
    exact positions used by teacher forcing for an L-token answer suffix.
    """
    device = clean_logits.device
    Bc, Sc, Vc = clean_logits.shape
    Bp, Sp, Vp = patched_logits.shape
    assert Bc == Bp and Vc == Vp, "batch/vocab mismatch"

    # Clamp starts like your TF (avoid pos-1 < 0 when no BOS)
    start_clean = torch.clamp(start_clean, min=clamp_min_start)
    start_patched = torch.clamp(start_patched, min=clamp_min_start)

    # positions that *predict* the answer tokens: (start - 1) + k
    offsets = torch.arange(answer_len, device=device)  # [L]
    pos_c = (start_clean.unsqueeze(1) - 1) + offsets   # [B, L]
    pos_p = (start_patched.unsqueeze(1) - 1) + offsets # [B, L]

    # Safety: ensure we never index >= S-1 (last usable logit is S-2)
    if (pos_c.max() >= Sc-0) or (pos_p.max() >= Sp-0):
        raise ValueError("Index out of range; check start indices/answer_len vs sequence lengths.")
    if (pos_c.min() < 0) or (pos_p.min() < 0):
        raise ValueError("Negative index; use BOS or clamp starts consistently with TF.")

    # Gather the logits at those positions → [B, L, V]
    clean_sel   = clean_logits.gather(1, pos_c.unsqueeze(-1).expand(-1, -1, Vc))
    patched_sel = patched_logits.gather(1, pos_p.unsqueeze(-1).expand(-1, -1, Vp))

    # KL(P || Q) per position: sum over vocab
    kl_pos = F.kl_div(
        clean_sel.log_softmax(-1),      # log P
        patched_sel.softmax(-1),        # Q
        reduction="none"
    ).sum(-1)                           # [B, L]

    kl_span = kl_pos.mean(-1) if reduce_over_span == "mean" else kl_pos.sum(-1)  # [B]

    return kl_span if per_prompt else kl_span.mean()

def path_patching(model, receiver_nodes, source_tokens, patch_tokens, ans_tokens, component='z', position=-1, freeze_mlps=False, indirect_patch=False, truncate_to_max_layer=True, metric="logit_diff", is_multitoken=False, device="cuda:0"):
    model.reset_hooks(including_permanent=True)
    """
    Noising: (corrupt -> clean), senders set to corrupt activations
    """
    #print("Patching component...", component)

    # Running cache for clean prompt
    
    clean_logits, clean_cache = model.run_with_cache(source_tokens)
    if metric == "logit_diff":
        clean_logit_diff = logits_to_ave_logit_diff(clean_logits, ans_tokens)

    # label_tokens = ans_tokens[:, 0]
    # logit values (pre-softmax scores) that the model assigned to gold answer token at the final position
    # when we run the clean prompt
    # clean_label_logits = clean_logits[:, -1][list(range(len(clean_logits))), label_tokens]

    # Running cache for corrupt prompt
    corr_logits, corr_cache = model.run_with_cache(patch_tokens)
    if metric == "logit_diff":
        corrupt_logit_diff = logits_to_ave_logit_diff(corr_logits, ans_tokens)

    patched_head_pq_diff = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)

    def add_hook_to_attn(attn_block, hook_fn):
        if component=='v':
            attn_block.hook_v.add_hook(hook_fn)
        elif component=='q':
            attn_block.hook_q.add_hook(hook_fn)
        elif component == 'k':
            attn_block.hook_k.add_hook(hook_fn)
        elif component == 'z':
            attn_block.hook_z.add_hook(hook_fn)
        else:
            raise Exception(f"Component must be q,k,v, or z. You passed {component}")
        
    max_layer = model.cfg.n_layers
    if truncate_to_max_layer: # don’t patch from layers after your receivers, we are interested in how much of s’s signal flows into r
        # Only run the sender–receiver intervention up to the receiver layer(s). 
        # Stop there, let those receivers recompute, and then measure their effect on the final output
        target_layers = [r[0] for r in receiver_nodes]
        for t in target_layers:
            if type(t)==int:
                max_layer = min(t, max_layer)
        if max_layer < model.cfg.n_layers:
            max_layer+=1 # because we want to go up to max layer inclusive

    for layer in track(list(range(max_layer))):
        for head_index in range(model.cfg.n_heads):

            model.reset_hooks(including_permanent=True)
            # Skip if the current head in the layer is in receivers
            if (layer, head_index) in receiver_nodes:
                continue
            
            # “hook the receivers” = attach functions to the receiver modules/heads 
            # so that when the model runs, you intercept and record their activations.
            # cache the values before overwriting them when proceed to a later layer
            receiver_cache = {}
            for recv_layer, recv_head in receiver_nodes:
                # hook for recording activations from receivers
                cache_fn = partial(cache_activation_hook, my_cache=receiver_cache)
                if recv_head is None:
                    # Whole layer is the receiver
                    # In that case, attach the caching hook directly to that module.
                    hook_name = utils.get_act_name("resid_post", recv_layer)
                    model.add_hook(hook_name, cache_fn)
                else:
                    # If receiver is a specific head, then attach the cache hook to the right stream of that head.
                    add_hook_to_attn(model.blocks[recv_layer].attn, cache_fn)

            # Add the hooks for the sender nodes layer, head_index
            # hook function actually patches the activations of the corrupted run to the clean run
            # Sender head --> patch from corrupted
            hook_fn = partial(patch_head_vector_at_pos, head_index=head_index, pos_index=position, corrupted_cache=corr_cache, is_multitoken=is_multitoken, answer_tokens=ans_tokens)
            model.blocks[layer].attn.hook_z.add_hook(hook_fn)

            # You want to isolate the effect of just one sender head. All MLPs and other heads should act exactly the same in the clean run.
            # Forcibly replace with the cached value from the clean run
            receiver_heads  = {(L,h) for (L,h) in receiver_nodes if h is not None}
            receiver_layers = {L     for (L,h) in receiver_nodes if h is None}
            
            for freeze_layer in list(range(model.cfg.n_layers)):

                is_recv_layer = freeze_layer in receiver_layers

                if freeze_mlps:
                    hook_fn = partial(patch_full_residual_component, pos_index=position, corrupted_cache=clean_cache, is_multitoken=is_multitoken, answer_tokens=ans_tokens)
                    model.blocks[freeze_layer].hook_mlp_out.add_hook(hook_fn)

                # For all heads except the sender head you’re currently testing, their z outputs are also patched back to the clean-cache version.
                for freeze_head in range(model.cfg.n_heads):

                    if freeze_layer == layer and freeze_head == head_index:
                        continue

                    if (freeze_layer, freeze_head) in receiver_heads or is_recv_layer:  # don't freeze receivers
                        continue

                    hook_fn = partial(patch_head_vector_at_pos, head_index=freeze_head, pos_index=position, corrupted_cache=clean_cache, is_multitoken=is_multitoken, answer_tokens=ans_tokens) 
                    model.blocks[freeze_layer].attn.hook_z.add_hook(hook_fn)

            # Run with the original tokens with the layer, head_index as a sender node
            # sender = corrupt, others = clean
            interv_logits, interv_cache = model.run_with_cache(source_tokens)

            # When we run this with all hooks,
            # receiver_cache should be filled with receiver activations
            model.reset_hooks()

            # Now patch back in the receiver nodes that are changed by the sender nodes
            fwd_hooks = []
            for recv_layer, recv_head in receiver_nodes:
                if recv_head is None:
                    #print("HOOK", recv_layer, receiver_cache)
                    hook_name = utils.get_act_name("resid_post", recv_layer)
                    hook_fn = partial(patch_full_residual_component, pos_index=position, corrupted_cache=receiver_cache, is_multitoken=is_multitoken, answer_tokens=ans_tokens)
                    fwd_hooks.append((hook_name, hook_fn))
                else:
                    hook_fn = partial(patch_head_vector_at_pos, head_index=recv_head, pos_index=position, corrupted_cache=receiver_cache, is_multitoken=is_multitoken, answer_tokens=ans_tokens)
                    fwd_hooks.append((utils.get_act_name(component, int(recv_layer), component), hook_fn))

            patched_logits = model.run_with_hooks(
                source_tokens,
                fwd_hooks = fwd_hooks,
                return_type="logits"
            )

            if metric == "kl_divergence":

                # for noising
                old = True
                if not old:
                    patched_clean_diff = kl_over_answer_span(
                        clean_logits, patched_logits,
                        start_clean=torch.tensor([max(source_tokens.size(1)-ans_tokens.numel(), 1)], device=device),
                        start_patched=torch.tensor([max(patch_tokens.size(1)-ans_tokens.numel(), 1)], device=device),
                        answer_len=int(ans_tokens.numel()),
                        clamp_min_start=1,
                        reduce_over_span="mean",
                        per_prompt=False
                    )

                    corr_clean_diff = kl_over_answer_span(
                        clean_logits, corr_logits,
                        start_clean=torch.tensor([max(source_tokens.size(1)-ans_tokens.numel(), 1)], device=device),
                        start_patched=torch.tensor([max(patch_tokens.size(1)-ans_tokens.numel(), 1)], device=device),
                        answer_len=int(ans_tokens.numel()),
                        clamp_min_start=1,
                        reduce_over_span="mean",
                        per_prompt=False
                    )

                else:
                    patched_clean_diff = multitoken_logits_to_KL_div(patched_logits, clean_logits, ans_tokens, is_multitoken=is_multitoken)
                    corr_clean_diff = multitoken_logits_to_KL_div(corr_logits, clean_logits, ans_tokens, is_multitoken=is_multitoken)

                # % of behavior harmed
                normalized_kl = patched_clean_diff / (corr_clean_diff + 1e-8)
                patched_head_pq_diff[layer, head_index] = normalized_kl.item() * 100
                del patched_clean_diff, corr_clean_diff, normalized_kl

            elif metric == "tf_loss":

                clean_len = source_tokens.shape[1]
                corr_len  = patch_tokens.shape[1]

                ans_len = int(ans_tokens.numel())

                start_clean = max(clean_len - ans_len, 1)
                start_corr  = max(corr_len  - ans_len, 1)

                # tensors shaped [1] for your loss function
                lt_pos_clean = torch.tensor([start_clean], device=device, dtype=torch.long)
                lt_pos_corr  = torch.tensor([start_corr],  device=device, dtype=torch.long)

                corr_loss = teacher_forcing_loss_from_indices(
                    corr_logits, patch_tokens, lt_pos_corr, pad_token_id=model.tokenizer.pad_token_id
                )
                clean_loss = teacher_forcing_loss_from_indices(
                    clean_logits, source_tokens, lt_pos_clean, pad_token_id=model.tokenizer.pad_token_id
                )
                
                #print("corr loss:", corr_loss.item(), "clean loss:", clean_loss.item())

                patched_tf_loss = tf_loss_noising(
                    patched_logits, source_tokens, 
                    start_indices=lt_pos_clean,
                    corr_loss=corr_loss, 
                    clean_loss=clean_loss, 
                    pad_token_id=model.tokenizer.pad_token_id, 
                    return_tensor=True
                )

                def compare_logits(clean_logits: torch.Tensor, patched_logits: torch.Tensor):
                    assert clean_logits.shape == patched_logits.shape, "shape mismatch"
                    assert clean_logits.dtype == patched_logits.dtype, "dtype mismatch"

                    exact = torch.equal(clean_logits, patched_logits)
                    close = torch.allclose(clean_logits, patched_logits, rtol=1e-5, atol=1e-7)
                    max_abs_diff = (clean_logits - patched_logits).abs().max().item()
                    mean_abs_diff = (clean_logits - patched_logits).abs().mean().item()

                    print(f"exact_equal={exact}")
                    print(f"allclose={close} (rtol=1e-5, atol=1e-7)")
                    print(f"max_abs_diff={max_abs_diff:.3e}, mean_abs_diff={mean_abs_diff:.3e}")

                #compare_logits(clean_logits, patched_logits)

                # print("corr loss:", corr_loss.item(), "patched loss:", patched_tf_loss.item(), "clean loss:", clean_loss.item())
                # print("corr loss:", corr_loss.item(), "patched loss:", patched_tf_loss.item(), "clean loss:", clean_loss.item(), "denom:", (corr_loss - clean_loss).item())
                # % of behavior restored
                patched_head_pq_diff[layer, head_index] = patched_tf_loss.item() * 100
                del corr_loss
                del clean_loss
                del patched_tf_loss

            elif metric == "logit_diff":
                patched_logit_diff = logits_to_ave_logit_diff(patched_logits, ans_tokens)
                patched_logit_diff = (patched_logit_diff - clean_logit_diff)/(corrupt_logit_diff-clean_logit_diff + 1e-8)
                # patched_logit_diff = (patched_logit_diff - corrupt_logit_diff)/(clean_logit_diff-corrupt_logit_diff + 1e-8)

                # 0 -> no harm, 100 -> a lot harm, Can dip <0 (better than clean) or >100 (worse than corrupt)
                patched_head_pq_diff[layer, head_index] = patched_logit_diff.item() * 100
                del patched_logit_diff
            
            del patched_logits
            
    return patched_head_pq_diff