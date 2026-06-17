"""
exp-073 — Robustness of the QK-slope behavioral handle (seed sweep).

Repeats the exp-072 κ-sweep on vicuna-13b-v1.5 across several task seeds to put
error bars on both legs — especially the shallowing leg, whose exp-072 verdict
turned on a single item out of 40.

Reuses the LOCKED exp-072 config verbatim (model, n_doc, positions, target+sham
heads, projector constants) read from
  ../exp-072_cloud_powered_slope_editing/prereg_locked.json
so NO new census is needed. This is characterization, not a new confirmatory
test — but seed 42 must reproduce exp-072 exactly as a numerical-equivalence
check on the one change here: BATCHED inference (left-pad + attention_mask),
the cost fix that makes a multi-seed sweep affordable.

fp32 throughout, to stay comparable to exp-072.

Usage:
  cd research/physics/experiments/exp-073_robustness_seeds
  .venv/bin/python -m modal run robustness_sweep.py
  # optional: --seeds 42,7,123,2024,99  --batch-size 8

Ariel — June 16, 2026.
"""

from __future__ import annotations

import modal

app = modal.App("exp073-robustness-seeds")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.43,<4.50",
        "accelerate",
        "numpy",
        "scipy",
        "sentencepiece",
    )
)

vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

# ── positional projector constants (LOCKED to exp-064 / exp-070 / exp-072) ─────
PROJ_SEQ = 512
PROJ_N = 50
PROJ_SEED = 68
N_PC = 8

# ── task constants (embedded-prose retrieval, exp-069 C6 family; matches exp-072) ──
N_ITEMS = 40
FILLER_PER_DOC = 0

WORD_POOL = [
    "red", "oak", "gem", "ice", "map", "sky", "fig", "ash", "bay", "cup",
    "owl", "elk", "fox", "pen", "key", "jet", "net", "rug", "pot", "den",
    "lab", "fan", "bell", "rope", "coal", "silk", "moss", "clay", "reed", "vine",
    "dog", "cat", "car", "tree", "book", "fish", "bird", "rock", "sand", "gold",
    "iron", "wood", "wine", "milk", "salt", "corn", "rice", "leaf", "wolf", "bear",
    "lion", "goat", "duck", "crow", "hawk", "moth", "frog", "seal", "crab", "lamp",
    "ring", "star", "moon", "lake", "hill", "barn", "gate", "road", "ship", "coin",
    "boat", "fire", "rain", "snow", "wind", "gold", "wing", "horn", "claw", "tail",
]
DIST_WORDS = [
    "sun", "fog", "mud", "tar", "sea", "log", "fur", "oil", "wax", "arc",
    "tor", "sap", "web", "rim", "rod", "ore", "fin", "jug", "nap", "jar",
    "bin", "cog", "dye", "gut", "hub", "ink", "keg", "lid", "mug", "orb",
    "paw", "ram", "sty", "tub", "urn", "vat", "yam", "pad", "cap", "bag",
]
FILLERS_TEMPLATE = [
    "The {n} survey involved several field stations.",
    "Observations for {n} were logged each day.",
    "The {n} report was archived after completion.",
    "Personnel assigned to {n} rotated on a weekly schedule.",
    "Funding for {n} continued through the full period.",
]


# ── architecture helpers (Llama; verbatim from exp-072) ────────────────────────

def _model_dims(model):
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    hidden = cfg.hidden_size
    q_rows = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    head_dim = q_rows // n_heads
    return n_layers, n_heads, hidden, head_dim


def _input_ln(model, layer):
    return model.model.layers[layer].input_layernorm


def _q_proj_weight(model, layer):
    return model.model.layers[layer].self_attn.q_proj.weight


def _get_wq(model, layer, head, head_dim):
    W = _q_proj_weight(model, layer)
    return W[head * head_dim:(head + 1) * head_dim, :].detach().clone()


def _apply_edit(model, layer, head, head_dim, kappa, P_U, W_orig):
    import torch
    hidden = W_orig.shape[1]
    M = torch.eye(hidden, device=W_orig.device, dtype=W_orig.dtype) + (kappa - 1.0) * P_U
    W = _q_proj_weight(model, layer)
    with torch.no_grad():
        W[head * head_dim:(head + 1) * head_dim, :] = W_orig @ M


def _restore(model, layer, head, head_dim, W_orig):
    import torch
    W = _q_proj_weight(model, layer)
    with torch.no_grad():
        W[head * head_dim:(head + 1) * head_dim, :] = W_orig


# ── positional projectors (verbatim from exp-072) ──────────────────────────────

def _compute_projectors(model, needed_layers, device):
    import numpy as np
    import torch
    _, _, hidden, _ = _model_dims(model)
    rng = np.random.default_rng(PROJ_SEED)
    vocab = model.config.vocab_size
    acc = {layer: np.zeros((PROJ_SEQ, hidden), dtype=np.float64) for layer in needed_layers}
    for i in range(PROJ_N):
        ids = torch.tensor(rng.integers(0, vocab, size=(1, PROJ_SEQ)),
                           dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
            for layer in needed_layers:
                h = out.hidden_states[layer]
                ln_out = _input_ln(model, layer)(h)
                acc[layer] += ln_out[0].cpu().float().numpy()
        del out
        if (i + 1) % 10 == 0:
            print(f"    projector pass {i + 1}/{PROJ_N}", flush=True)
    projectors = {}
    for layer in needed_layers:
        xbar = acc[layer] / PROJ_N
        xbar -= xbar.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(xbar, full_matrices=False)
        U = vt[:N_PC].T
        projectors[layer] = (U @ U.T).astype(np.float32)
    return projectors


# ── task construction (verbatim from exp-072 except TASK_SEED is a parameter) ──

def _make_document(name, word, doc_num):
    s = [f"Document {doc_num}: The {name} survey produced a number of findings during its operation."]
    s.append(f"Among the recorded details, the team noted that the {name} signal marker "
             f"was set to {word} throughout the study.")
    for i in range(FILLER_PER_DOC):
        s.append(FILLERS_TEMPLATE[i % len(FILLERS_TEMPLATE)].format(n=name))
    return " ".join(s)


def _make_prompt(query_name, query_word, answer_pos, n_doc, dist_names, dist_order):
    header = ("The following documents contain project registry records. "
              "Read all documents carefully, then answer the question.\n\n")
    docs = []
    dist_ptr = 0
    for slot in range(n_doc):
        doc_num = slot + 1
        if slot == answer_pos:
            docs.append(_make_document(query_name, query_word, doc_num))
        else:
            di = dist_order[dist_ptr]
            docs.append(_make_document(dist_names[di], DIST_WORDS[di % len(DIST_WORDS)], doc_num))
            dist_ptr += 1
    body = "\n\n".join(docs)
    question = (f"\n\nQuestion: According to the documents above, what was the "
                f"{query_name} signal marker set to? The {query_name} signal marker was set to")
    return header + body + question


def _select_words(tokenizer):
    words = []
    for w in WORD_POOL:
        if len(tokenizer.encode(" " + w, add_special_tokens=False)) == 1:
            words.append(w)
        if len(words) >= N_ITEMS:
            break
    return words


def _build_task(tokenizer, n_doc, words, positions, task_seed):
    import numpy as np
    rng = np.random.default_rng(task_seed)
    n_items = len(words)
    query_names = [f"Item-{i + 1:03d}" for i in range(n_items)]
    n_dist = n_doc - 1
    dist_names = [f"Item-{i + 1:03d}" for i in range(n_items, n_items + n_dist + 5)]
    items = []
    for qi in range(n_items):
        dist_pool = list(range(len(dist_names)))
        rng.shuffle(dist_pool)
        dist_sel = dist_pool[:n_dist]
        for pi, pos in enumerate(positions):
            prompt = _make_prompt(query_names[qi], words[qi], pos, n_doc, dist_names, dist_sel)
            enc = tokenizer(prompt, return_tensors="pt")
            ans_ids = tokenizer.encode(" " + words[qi], add_special_tokens=False)
            items.append({
                "qi": qi, "pi": pi,
                "answer_id": ans_ids[0],
                "n_tokens": enc.input_ids.shape[1],
                "input_ids": enc.input_ids,
            })
    return items


def _valley(acc_by_pos):
    primacy, recency, middle = acc_by_pos[0], acc_by_pos[4], acc_by_pos[2]
    denom = max(primacy, recency)
    if denom < 1e-9:
        return float("nan")
    return float(1.0 - middle / denom)


def _batched_accuracy(model, items, tokenizer, device, n_positions, batch_size):
    """Left-padded batched argmax. Left padding keeps the true last token at index
    -1 for every row, so out.logits[:, -1, :] is the next-token distribution.
    fp32 + left-pad should reproduce per-item argmax exactly (mask zeroes pads)."""
    import torch
    pad_id = tokenizer.pad_token_id
    correct = [0] * n_positions
    total = [0] * n_positions
    tokens = [None] * len(items)
    order = list(range(len(items)))
    for start in range(0, len(items), batch_size):
        idx = order[start:start + batch_size]
        seqs = [items[i]["input_ids"][0] for i in idx]
        maxlen = max(s.shape[0] for s in seqs)
        input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for r, s in enumerate(seqs):
            input_ids[r, maxlen - s.shape[0]:] = s
            attn[r, maxlen - s.shape[0]:] = 1
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        with torch.no_grad():
            out = model(input_ids, attention_mask=attn)
        preds = out.logits[:, -1, :].argmax(dim=-1).tolist()
        for r, i in enumerate(idx):
            p = int(preds[r])
            tokens[i] = p
            correct[items[i]["pi"]] += int(p == items[i]["answer_id"])
            total[items[i]["pi"]] += 1
        del out
    acc = [correct[i] / max(total[i], 1) for i in range(n_positions)]
    return acc, correct, total, tokens


@app.function(image=image, gpu="A100-80GB", timeout=14400, volumes={"/cache": vol},
              memory=65536, secrets=[modal.Secret.from_name("huggingface-token")])
def run_robustness(model_name: str, n_doc: int, positions: list,
                   target_heads: list, sham_heads: list, seeds: list,
                   batch_size: int = 8):
    import os
    os.environ["HF_HOME"] = "/cache"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/transformers"
    # Reduce allocator fragmentation: vicuna-13b fp32 + eager attention on long
    # prompts leaves large reserved-but-unallocated blocks (exp-073 first OOM).
    # Must be set before the CUDA context is created (i.e. before torch import).
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import time
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="eager",
    ).to("cuda").eval()
    n_layers, n_heads, hidden, head_dim = _model_dims(model)
    words = _select_words(tok)
    n_pos = len(positions)
    print(f"[{model_name}] {n_layers}L×{n_heads}H hd={head_dim}; {len(words)} words; "
          f"seeds={seeds}; batch={batch_size}", flush=True)

    needed_layers = sorted(set(h["layer"] for h in target_heads + sham_heads))
    print(f"  computing projectors for layers {needed_layers}...", flush=True)
    projectors = _compute_projectors(model, needed_layers, "cuda")
    # The 50 projector passes (output_hidden_states) leave a large cached/
    # fragmented pool; release it before the memory-heavy batched forwards.
    torch.cuda.empty_cache()
    print(f"  cache cleared; reserved={torch.cuda.memory_reserved()/1e9:.1f}GB "
          f"allocated={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    KAPPA_CONDS = [("target", 1.0), ("target", 0.5), ("target", 1.5),
                   ("target", 2.0), ("sham_heads", 1.5)]

    def run_condition(items, edit_heads, kappa):
        orig = {}
        for h in edit_heads:
            l, hd_i = h["layer"], h["head"]
            W_orig = _get_wq(model, l, hd_i, head_dim).to("cuda")
            P_U = torch.tensor(projectors[l], dtype=torch.float32, device="cuda")
            _apply_edit(model, l, hd_i, head_dim, kappa, P_U, W_orig)
            orig[(l, hd_i)] = W_orig
        acc, correct, total, tokens = _batched_accuracy(
            model, items, tok, "cuda", n_pos, batch_size)
        for (l, hd_i), W_orig in orig.items():
            _restore(model, l, hd_i, head_dim, W_orig)
        return acc, correct, total, tokens

    per_seed = []
    for seed in seeds:
        t0 = time.time()
        items = _build_task(tok, n_doc, words, positions, seed)
        print(f"  [seed {seed}] {len(items)} items; running base...", flush=True)
        base_acc, _, _, base_tokens = _batched_accuracy(
            model, items, tok, "cuda", n_pos, batch_size)
        print(f"  [seed {seed}] base done ({time.time()-t0:.0f}s) "
              f"base={[round(a,3) for a in base_acc]}", flush=True)
        conds = {}
        for label, kap in KAPPA_CONDS:
            tc = time.time()
            heads = target_heads if label == "target" else sham_heads
            acc, correct, total, tokens = run_condition(items, heads, kap)
            print(f"    [seed {seed}] {label} k={kap}: V={_valley(acc):.4f} "
                  f"({time.time()-tc:.0f}s)", flush=True)
            key = f"{label}_k{kap}"
            rec = {"label": label, "kappa": kap,
                   "accuracy_by_pos": [round(a, 4) for a in acc],
                   "correct_by_pos": correct, "total_by_pos": total,
                   "V_task": round(_valley(acc), 5)}
            if label == "target" and kap == 1.0:
                rec["token_diff_from_base"] = int(sum(
                    int(b != c) for b, c in zip(base_tokens, tokens)))
            conds[key] = rec
        per_seed.append({"seed": seed,
                         "base_accuracy_by_pos": [round(a, 4) for a in base_acc],
                         "conditions": conds})
        # Incremental persistence: keep completed seeds even if the run is
        # interrupted mid-sweep (flaky connection / timeout).
        import json as _json
        with open("/cache/exp073_partial.json", "w") as f:
            _json.dump({"seeds_done": [s["seed"] for s in per_seed],
                        "per_seed": per_seed}, f, indent=1)
        vol.commit()
        print(f"  seed {seed}: base={[round(a,3) for a in base_acc]} "
              f"k0.5 V={conds['target_k0.5']['V_task']:.4f} "
              f"k1.0 V={conds['target_k1.0']['V_task']:.4f} "
              f"k1.5 V={conds['target_k1.5']['V_task']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ── aggregate: mean ± SE per condition across seeds ──
    def agg(key, field="V_task"):
        vals = [s["conditions"][key][field] for s in per_seed]
        v = np.array(vals, dtype=float)
        return {"mean": round(float(v.mean()), 5),
                "se": round(float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, 5),
                "values": vals}

    summary = {k: agg(f"{lbl}_k{kap}") for lbl, kap in KAPPA_CONDS
               for k in [f"{lbl}_k{kap}"]}
    # shallowing delta distribution: V(0.5) - V(1.0) per seed
    shallow_delta = [s["conditions"]["target_k0.5"]["V_task"]
                     - s["conditions"]["target_k1.0"]["V_task"] for s in per_seed]
    deepen_delta = [s["conditions"]["target_k1.5"]["V_task"]
                    - s["conditions"]["target_k1.0"]["V_task"] for s in per_seed]
    sd = np.array(shallow_delta); dd = np.array(deepen_delta)

    result = {
        "model": model_name, "n_doc": n_doc, "positions": positions, "seeds": seeds,
        "batch_size": batch_size,
        "per_seed": per_seed,
        "summary_V_task": summary,
        "shallowing_delta_V": {  # V(0.5)-V(1.0); NEGATIVE = predicted (valley shallower)
            "per_seed": [round(x, 5) for x in shallow_delta],
            "mean": round(float(sd.mean()), 5),
            "se": round(float(sd.std(ddof=1) / np.sqrt(len(sd))) if len(sd) > 1 else 0.0, 5),
            "n_predicted_sign": int((sd < 0).sum()),
        },
        "deepening_delta_V": {  # V(1.5)-V(1.0); POSITIVE = predicted (valley deeper)
            "per_seed": [round(x, 5) for x in deepen_delta],
            "mean": round(float(dd.mean()), 5),
            "se": round(float(dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else 0.0, 5),
            "n_predicted_sign": int((dd > 0).sum()),
        },
    }
    # Persist the full result server-side on the volume, robust to a flaky client
    # connection (the local entrypoint may never receive the return value).
    import json as _json
    with open("/cache/exp073_results.json", "w") as f:
        _json.dump(result, f, indent=1)
    vol.commit()
    print("  [persisted full result -> volume:/exp073_results.json]", flush=True)
    return result


# ── per-seed worker: one container per seed (robust to client death) ──────────
# Each spawned call runs ONE seed end-to-end and commits its own file
# /cache/exp073_seed{seed}.json. One flaky-connection drop or one container
# failure can no longer wipe out the other seeds, and seeds run concurrently
# instead of 5.5 h serially. Compute path is identical to run_robustness.
@app.function(image=image, gpu="A100-80GB", timeout=7200, volumes={"/cache": vol},
              memory=65536, secrets=[modal.Secret.from_name("huggingface-token")])
def run_one_seed(model_name: str, n_doc: int, positions: list,
                 target_heads: list, sham_heads: list, seed: int,
                 batch_size: int = 4):
    import os
    os.environ["HF_HOME"] = "/cache"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/transformers"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import time
    import json as _json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="eager",
    ).to("cuda").eval()
    n_layers, n_heads, hidden, head_dim = _model_dims(model)
    words = _select_words(tok)
    n_pos = len(positions)
    needed_layers = sorted(set(h["layer"] for h in target_heads + sham_heads))
    print(f"[seed {seed}] {model_name} {n_layers}L×{n_heads}H; projectors {needed_layers}...",
          flush=True)
    projectors = _compute_projectors(model, needed_layers, "cuda")
    torch.cuda.empty_cache()

    KAPPA_CONDS = [("target", 1.0), ("target", 0.5), ("target", 1.5),
                   ("target", 2.0), ("sham_heads", 1.5)]

    def run_condition(items, edit_heads, kappa):
        orig = {}
        for h in edit_heads:
            l, hd_i = h["layer"], h["head"]
            W_orig = _get_wq(model, l, hd_i, head_dim).to("cuda")
            P_U = torch.tensor(projectors[l], dtype=torch.float32, device="cuda")
            _apply_edit(model, l, hd_i, head_dim, kappa, P_U, W_orig)
            orig[(l, hd_i)] = W_orig
        acc, correct, total, tokens = _batched_accuracy(
            model, items, tok, "cuda", n_pos, batch_size)
        for (l, hd_i), W_orig in orig.items():
            _restore(model, l, hd_i, head_dim, W_orig)
        return acc, correct, total, tokens

    t0 = time.time()
    items = _build_task(tok, n_doc, words, positions, seed)
    base_acc, _, _, base_tokens = _batched_accuracy(
        model, items, tok, "cuda", n_pos, batch_size)
    print(f"  [seed {seed}] base done ({time.time()-t0:.0f}s) "
          f"base={[round(a,3) for a in base_acc]}", flush=True)
    conds = {}
    for label, kap in KAPPA_CONDS:
        tc = time.time()
        heads = target_heads if label == "target" else sham_heads
        acc, correct, total, tokens = run_condition(items, heads, kap)
        print(f"    [seed {seed}] {label} k={kap}: V={_valley(acc):.4f} "
              f"({time.time()-tc:.0f}s)", flush=True)
        rec = {"label": label, "kappa": kap,
               "accuracy_by_pos": [round(a, 4) for a in acc],
               "correct_by_pos": correct, "total_by_pos": total,
               "V_task": round(_valley(acc), 5)}
        if label == "target" and kap == 1.0:
            rec["token_diff_from_base"] = int(sum(
                int(b != c) for b, c in zip(base_tokens, tokens)))
        conds[f"{label}_k{kap}"] = rec

    seed_record = {"seed": seed,
                   "base_accuracy_by_pos": [round(a, 4) for a in base_acc],
                   "conditions": conds}
    with open(f"/cache/exp073_seed{seed}.json", "w") as f:
        _json.dump(seed_record, f, indent=1)
    vol.commit()
    print(f"  [seed {seed} persisted -> volume:/exp073_seed{seed}.json "
          f"({time.time()-t0:.0f}s)]", flush=True)
    return seed_record


@app.local_entrypoint()
def spawn(seeds: str = "123,2024,99", batch_size: int = 4):
    """Fire one independent container per seed (robust to client disconnects).
    Returns immediately; poll the volume for /exp073_seed{seed}.json and then
    run the `aggregate` entrypoint. Default seeds are the three missing after
    the serial run banked 42 and 7."""
    import json
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    lock_path = os.path.join(os.path.dirname(here),
                             "exp-072_cloud_powered_slope_editing", "prereg_locked.json")
    with open(lock_path) as f:
        lock = json.load(f)
    seed_list = [int(x) for x in seeds.split(",") if x.strip()]
    print(f"=== exp-073 SPAWN per-seed: {lock['model']} seeds={seed_list} ===")
    calls = []
    for s in seed_list:
        call = run_one_seed.spawn(
            lock["model"], lock["n_doc"], lock["positions"],
            lock["target_heads"], lock["sham_heads"], s, batch_size)
        calls.append((s, call.object_id))
        print(f"  spawned seed {s}: {call.object_id}", flush=True)
    print("\nAll seeds spawned server-side. They will finish and commit "
          "independently even if this client disconnects.")
    print("Retrieve: modal volume get hf-model-cache /exp073_seed<seed>.json")


@app.local_entrypoint()
def main(seeds: str = "42,7,123,2024,99", batch_size: int = 4):
    import json
    import os
    from datetime import datetime, timezone

    here = os.path.dirname(os.path.abspath(__file__))
    lock_path = os.path.join(os.path.dirname(here),
                             "exp-072_cloud_powered_slope_editing", "prereg_locked.json")
    with open(lock_path) as f:
        lock = json.load(f)
    seed_list = [int(x) for x in seeds.split(",") if x.strip()]
    print(f"=== exp-073 ROBUSTNESS: {lock['model']} n_doc={lock['n_doc']} "
          f"seeds={seed_list} ===")
    res = run_robustness.remote(
        lock["model"], lock["n_doc"], lock["positions"],
        lock["target_heads"], lock["sham_heads"], seed_list, batch_size)
    res["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    res["source_prereg_commit"] = lock.get("preregistration_commit", "UNKNOWN")
    out = os.path.join(here, "results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nWrote {out}")

    print("\n=== ROBUSTNESS SUMMARY (V_task mean ± SE across seeds) ===")
    for k, v in res["summary_V_task"].items():
        print(f"  {k:18s}: {v['mean']:.4f} ± {v['se']:.4f}  {v['values']}")
    sh = res["shallowing_delta_V"]; dp = res["deepening_delta_V"]
    print(f"\n  shallowing ΔV (V0.5−V1.0, neg=predicted): {sh['mean']:.4f} ± {sh['se']:.4f}  "
          f"predicted-sign {sh['n_predicted_sign']}/{len(seed_list)}  {sh['per_seed']}")
    print(f"  deepening  ΔV (V1.5−V1.0, pos=predicted): {dp['mean']:.4f} ± {dp['se']:.4f}  "
          f"predicted-sign {dp['n_predicted_sign']}/{len(seed_list)}  {dp['per_seed']}")
    # seed-42 reproduction check vs exp-072 (V: k0.5=0.576, k1.0=0.594, k1.5=0.656)
    s42 = next((s for s in res["per_seed"] if s["seed"] == 42), None)
    if s42:
        print("\n  seed-42 reproduction check (exp-072: k0.5=0.576 k1.0=0.594 k1.5=0.656):")
        print(f"    got k0.5={s42['conditions']['target_k0.5']['V_task']:.4f} "
              f"k1.0={s42['conditions']['target_k1.0']['V_task']:.4f} "
              f"k1.5={s42['conditions']['target_k1.5']['V_task']:.4f} "
              f"(T-D token diff={s42['conditions']['target_k1.0'].get('token_diff_from_base')})")
