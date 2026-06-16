"""
exp-071 (calibration) + exp-072 (powered cloud run) — Causal→Behavior, Phase 2.

Adjudicates the open question left by exp-070: did the SHALLOWING leg (T-B) fail
because the κ-slope mechanism is one-directional, or because the local Pythia-1.4b
task had no headroom (base middle accuracy 0.85, near the task ceiling)?

The only clean test: run the SAME κ-sweep on a model with a DEEP base lost-in-the-
middle U-shape — middle well below the edges AND edges off the ceiling — so κ<1 has
room to *raise* the middle.

This is NOT a re-run of run_exp070.py. That script reads a Pythia-1.4b-specific
census from disk and uses the GPT-NeoX fused `query_key_value` layout. This is a
fresh build for a **Llama-architecture (RoPE + full MHA)** model, with the census,
positional projector, and κ-operator all ported and re-verified:

  Llama port of the three architecture-specific pieces:
    - layers:          model.model.layers[i]
    - edit basis P_U:  top-N_PC PCs of position-mean  layer.input_layernorm(h)
    - W_Q per head:    layer.self_attn.q_proj.weight[h*hd:(h+1)*hd, :]   (hd = 128)

Why Llama and not MPT-30B-Instruct (the original candidate):
  (1) the mosaicml/mpt-30b-instruct repo no longer exists on HF (only GGML forks);
  (2) MPT is ALiBi — position is a fixed additive score bias, NOT in the query
      content — so an operator that rescales W_Q's projection onto the positional
      subspace is a likely MECHANISM MISMATCH, not just an engineering port. The
      κ-operator (exp-064/068/070) was only ever validated on RoPE + MHA.

Phases (Modal A100-80GB):
  calibrate : base U-shape sweep over N_doc + full conformal head census + head
              selection. ALL base-model / analysis-only → allowed before pre-reg.
              Writes exp-071_cloud_calibration/results.json.
  sweep     : the registered edited κ-sweep. Reads the LOCKED (model, n_doc, heads,
              projector) from the exp-072 pre-registration JSON. Run ONLY AFTER the
              pre-reg is committed to git. Writes exp-072.../results.json.

Usage:
  .venv/bin/python -m modal run cloud_slope_editing.py --phase calibrate \
        --model lmsys/vicuna-13b-v1.5 --n-docs 40,60,80
  .venv/bin/python -m modal run cloud_slope_editing.py --phase sweep

Ariel — June 16, 2026.
"""

from __future__ import annotations

import modal

app = modal.App("exp072-cloud-slope-editing")

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

# ── census pipeline constants (LOCKED to exp-025 / exp-059, unchanged) ──────────
SEQ_LEN = 512
N_INPUTS = 50
FIT_LOW = 3
FIT_HIGH = min(120, SEQ_LEN // 3)     # 120
DEEP_LO = SEQ_LEN // 2                 # 256
Q_LO = 3 * SEQ_LEN // 4                # 384
THIRD = SEQ_LEN // 3                   # 170
CENSUS_R2 = 0.85
CENSUS_DELTA = 0.05
INPUT_SEED = 2026                      # census random-token stream

# ── positional projector constants (LOCKED to exp-064 / exp-070) ───────────────
PROJ_SEQ = 512
PROJ_N = 50
PROJ_SEED = 68
N_PC = 8

# ── head-selection criteria (match exp-070 Stage 1) ────────────────────────────
SEL_R2 = 0.85
SEL_DLO = 0.10
SEL_DHI = 0.40
N_SELECT = 8
# Layer band: the proportional analogue of Pythia's layers 8-20 of 24
# (≈ [0.33, 0.83] of depth). Resolved at runtime from the model's layer count.
LAYER_BAND_FRAC = (0.33, 0.83)

# ── task constants (embedded-prose retrieval, exp-069 C6 family) ───────────────
KAPPAS = [0.5, 1.0, 1.5, 2.0]
TASK_SEED = 42
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


# ════════════════════════════════════════════════════════════════════════════════
# Architecture-aware helpers (Llama: model.model.layers / self_attn.q_proj /
# input_layernorm). Kept in one place so the port is auditable.
# ════════════════════════════════════════════════════════════════════════════════

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


# ── census fit (verbatim math from exp-059) ────────────────────────────────────

def _fit_heads(mean_lagprof, X_C, SXX):
    import numpy as np
    y = np.log(np.maximum(mean_lagprof[:, FIT_LOW:FIT_HIGH], 1e-30))
    y_c = y - y.mean(axis=1, keepdims=True)
    slope = (y_c @ X_C) / SXX
    pred = slope[:, None] * X_C[None, :]
    ss_res = np.sum((y_c - pred) ** 2, axis=1)
    ss_tot = np.sum(y_c ** 2, axis=1)
    r2 = np.where(ss_tot > 1e-30, 1.0 - ss_res / np.maximum(ss_tot, 1e-30), 0.0)
    return -slope / 2.0, r2


def _valley_from_thirds(mean_thirds):
    import numpy as np
    s, m, e = mean_thirds[:, 0], mean_thirds[:, 1], mean_thirds[:, 2]
    peak = np.maximum(s, e)
    valley = 1.0 - m / np.maximum(peak, 1e-30)
    bad = np.minimum(np.minimum(s, m), e) < 1e-15
    return np.where(bad, np.nan, valley)


def _census(model, device):
    """exp-059 per-input census: random-token forward passes, lag-profile Δ̂ + R²
    and deep-query thirds valley, all fp32. Returns per-head dicts + raw arrays."""
    import numpy as np
    import torch

    n_layers, n_heads, _, _ = _model_dims(model)
    n_total = n_layers * n_heads
    vocab = model.config.vocab_size

    LOG_X = np.log(np.arange(FIT_LOW, FIT_HIGH).astype(float))
    X_C = LOG_X - LOG_X.mean()
    SXX = float(np.sum(X_C ** 2))

    rng = np.random.default_rng(INPUT_SEED)
    lagprof = np.zeros((N_INPUTS, n_total, FIT_HIGH), dtype=np.float32)
    thirds = np.zeros((N_INPUTS, n_total, 3), dtype=np.float32)

    for inp in range(N_INPUTS):
        token_ids = rng.integers(0, vocab, size=SEQ_LEN)
        input_ids = torch.tensor(token_ids[None, :], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids, output_attentions=True)
        for l in range(n_layers):
            a_t = out.attentions[l]
            assert a_t.dtype == torch.float32, f"layer {l}: dtype {a_t.dtype}"
            a = a_t[0].cpu().numpy()                 # (n_heads, L, L)
            assert not np.isnan(a).any(), f"layer {l}: NaN attention"
            base = l * n_heads
            for dx in range(FIT_HIGH):
                diag = np.diagonal(a, offset=-dx, axis1=-2, axis2=-1)
                k_lo = max(DEEP_LO, dx) - dx
                lagprof[inp, base:base + n_heads, dx] = diag[:, k_lo:].mean(axis=-1)
            prof = a[:, Q_LO:, :].mean(axis=1)
            thirds[inp, base:base + n_heads, 0] = prof[:, 1:THIRD].mean(axis=-1)
            thirds[inp, base:base + n_heads, 1] = prof[:, THIRD:2 * THIRD].mean(axis=-1)
            thirds[inp, base:base + n_heads, 2] = prof[:, 2 * THIRD:Q_LO].mean(axis=-1)
        del out
        if (inp + 1) % 10 == 0:
            print(f"    census forward {inp + 1}/{N_INPUTS}", flush=True)

    delta, r2 = _fit_heads(lagprof.mean(axis=0), X_C, SXX)
    valley = _valley_from_thirds(thirds.mean(axis=0))

    heads = []
    for l in range(n_layers):
        for h in range(n_heads):
            fh = l * n_heads + h
            heads.append({
                "layer": int(l), "head": int(h), "flat_head": int(fh),
                "delta_1d": round(float(delta[fh]), 5),
                "r2_1d": round(float(r2[fh]), 5),
                "valley_census": (round(float(valley[fh]), 5)
                                  if not np.isnan(valley[fh]) else None),
            })
    n_conf = sum(1 for h in heads if h["r2_1d"] >= CENSUS_R2 and h["delta_1d"] >= CENSUS_DELTA)
    return heads, n_conf


def _select_heads(heads, n_layers):
    lo = int(round(LAYER_BAND_FRAC[0] * n_layers))
    hi = int(round(LAYER_BAND_FRAC[1] * n_layers))
    cands = [h for h in heads
             if lo <= h["layer"] <= hi
             and h["r2_1d"] >= SEL_R2
             and SEL_DLO <= h["delta_1d"] <= SEL_DHI]
    cands.sort(key=lambda x: -x["r2_1d"])
    return cands[:N_SELECT], cands[N_SELECT:N_SELECT * 2], (lo, hi), len(cands)


# ── positional projectors (port of exp-070 _compute_all_projectors) ────────────

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


# ── task construction (embedded prose, matched cloze) ──────────────────────────

def _positions_for(n_doc):
    p = sorted(set([0, n_doc // 4, n_doc // 2, (3 * n_doc) // 4, n_doc - 1]))
    while len(p) < 5:               # tiny n_doc safety; not expected here
        p.append(p[-1])
    return p[:5]


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


def _build_task(tokenizer, n_doc, words, positions):
    import numpy as np
    rng = np.random.default_rng(TASK_SEED)
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


def _base_accuracy(model, items, device, n_positions):
    import torch
    correct = [0] * n_positions
    total = [0] * n_positions
    tokens = []
    for it in items:
        ids = it["input_ids"].to(device)
        with torch.no_grad():
            out = model(ids)
        pred = int(out.logits[0, -1, :].argmax().item())
        tokens.append(pred)
        correct[it["pi"]] += int(pred == it["answer_id"])
        total[it["pi"]] += 1
    acc = [correct[i] / max(total[i], 1) for i in range(n_positions)]
    return acc, correct, total, tokens


# ════════════════════════════════════════════════════════════════════════════════
# Modal functions
# ════════════════════════════════════════════════════════════════════════════════

def _load_model(model_name):
    import os
    os.environ["HF_HOME"] = "/cache"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/transformers"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="eager",
    ).to("cuda").eval()
    return model, tok


@app.function(image=image, gpu="A100-80GB", timeout=5400, volumes={"/cache": vol},
              memory=131072, secrets=[modal.Secret.from_name("huggingface-token")])
def calibrate_and_census(model_name: str, n_docs_list: list):
    """Base U-shape sweep over N_doc + conformal head census + head selection.
    All base-model / analysis-only. Allowed before pre-registration."""
    import time
    import numpy as np
    model, tok = _load_model(model_name)
    n_layers, n_heads, hidden, head_dim = _model_dims(model)
    max_pos = getattr(model.config, "max_position_embeddings", 4096)
    print(f"[{model_name}] {n_layers}L × {n_heads}H, hidden {hidden}, head_dim {head_dim}, "
          f"max_pos {max_pos}", flush=True)

    words = _select_words(tok)
    print(f"  single-token answer words available: {len(words)}", flush=True)

    # Census first (cheap, ~2 min) so a kill mid-sweep never loses the head selection.
    print(f"[{model_name}] running conformal head census (L=512, {N_INPUTS} inputs)...", flush=True)
    heads, n_conf = _census(model, "cuda")
    target, sham, band, n_cands = _select_heads(heads, n_layers)
    print(f"  census: {n_conf} conformal heads (R²≥{CENSUS_R2}, Δ≥{CENSUS_DELTA}); "
          f"selection band layers {band}, {n_cands} candidates, "
          f"{len(target)} target + {len(sham)} sham selected", flush=True)
    for h in target:
        print(f"    TARGET L{h['layer']}H{h['head']} Δ={h['delta_1d']:.3f} R²={h['r2_1d']:.3f}", flush=True)

    calib = []
    for n_doc in n_docs_list:
        positions = _positions_for(n_doc)
        items = _build_task(tok, n_doc, words, positions)
        lens = [it["n_tokens"] for it in items]
        ctx_max = max(lens)
        if ctx_max > max_pos - 8:
            print(f"  n_doc={n_doc}: SKIP (ctx {ctx_max} > max_pos {max_pos})", flush=True)
            calib.append({"n_doc": n_doc, "positions": positions, "skipped": True,
                          "ctx_max": ctx_max})
            continue
        t0 = time.time()
        acc, correct, total, _ = _base_accuracy(model, items, "cuda", len(positions))
        v = _valley(acc)
        edges_max = max(acc[0], acc[4])
        rec = {
            "n_doc": n_doc, "positions": positions,
            "n_items": len(words), "ctx_tokens_max": int(ctx_max),
            "ctx_tokens_mean": int(np.mean(lens)),
            "accuracy_by_pos": [round(a, 4) for a in acc],
            "correct_by_pos": correct, "total_by_pos": total,
            "V_task": round(v, 5), "edges_max": round(edges_max, 4),
            "middle": round(acc[2], 4),
            "deep_enough": bool(v >= 0.30 and edges_max <= 0.85 and acc[2] < edges_max),
        }
        calib.append(rec)
        print(f"  n_doc={n_doc}: acc={rec['accuracy_by_pos']} V={v:.3f} "
              f"edges_max={edges_max:.3f} deep_enough={rec['deep_enough']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    vol.commit()
    return {
        "model": model_name,
        "dims": {"n_layers": n_layers, "n_heads": n_heads, "hidden": hidden,
                 "head_dim": head_dim, "max_position_embeddings": int(max_pos)},
        "n_single_token_words": len(words),
        "answer_words": words,
        "calibration": calib,
        "census": {
            "pipeline": "exp-025/exp-059 (L=512, 50 random inputs, fp32)",
            "n_conformal_heads": n_conf,
            "selection_band_layers": band,
            "n_selection_candidates": n_cands,
            "target_heads": target,
            "sham_heads": sham,
            "all_heads": heads,
        },
    }


@app.function(image=image, gpu="A100-80GB", timeout=5400, volumes={"/cache": vol},
              memory=131072, secrets=[modal.Secret.from_name("huggingface-token")])
def run_kappa_sweep(model_name: str, n_doc: int, positions: list,
                    target_heads: list, sham_heads: list):
    """The REGISTERED edited κ-sweep. Run only after the pre-reg is committed.
    κ ∈ {0.5, 1.0 sham, 1.5, 2.0} on target heads + κ=1.5 on sham heads."""
    import time
    import numpy as np
    import torch

    model, tok = _load_model(model_name)
    n_layers, n_heads, hidden, head_dim = _model_dims(model)
    words = _select_words(tok)
    items = _build_task(tok, n_doc, words, positions)
    n_pos = len(positions)
    lens = [it["n_tokens"] for it in items]
    print(f"[{model_name}] task n_doc={n_doc}, {len(items)} items, "
          f"ctx max={max(lens)} mean={np.mean(lens):.0f}", flush=True)

    needed_layers = sorted(set(h["layer"] for h in target_heads + sham_heads))
    print(f"  computing projectors for layers {needed_layers}...", flush=True)
    projectors = _compute_projectors(model, needed_layers, "cuda")

    def run_condition(label, edit_heads, kappa):
        orig = {}
        for h in edit_heads:
            l, hd_i = h["layer"], h["head"]
            W_orig = _get_wq(model, l, hd_i, head_dim).to("cuda")
            P_U = torch.tensor(projectors[l], dtype=torch.float32, device="cuda")
            _apply_edit(model, l, hd_i, head_dim, kappa, P_U, W_orig)
            orig[(l, hd_i)] = W_orig
        t0 = time.time()
        acc, correct, total, tokens = _base_accuracy(model, items, "cuda", n_pos)
        for (l, hd_i), W_orig in orig.items():
            _restore(model, l, hd_i, head_dim, W_orig)
        v = _valley(acc)
        print(f"  κ={kappa} [{label}]: acc={[round(a,3) for a in acc]} "
              f"V={v:.4f} ({time.time()-t0:.0f}s)", flush=True)
        return {"kappa": kappa, "label": label,
                "accuracy_by_pos": [round(a, 4) for a in acc],
                "correct_by_pos": correct, "total_by_pos": total,
                "V_task": round(v, 5), "predicted_tokens": tokens}

    print("  [T-D] base (unedited) reference...", flush=True)
    base_acc, _, _, base_tokens = _base_accuracy(model, items, "cuda", n_pos)
    print(f"    base acc={[round(a,3) for a in base_acc]}", flush=True)

    conditions = []
    c10 = run_condition("target", target_heads, 1.0)
    c10["sham_token_diff_from_base"] = int(sum(int(b != c) for b, c in
                                               zip(base_tokens, c10["predicted_tokens"])))
    print(f"  [T-D] κ=1.0 token diffs from base: {c10['sham_token_diff_from_base']}", flush=True)
    conditions.append(c10)
    for kap in [0.5, 1.5, 2.0]:
        conditions.append(run_condition("target", target_heads, kap))
    conditions.append(run_condition("sham_heads", sham_heads, 1.5))

    vol.commit()
    return {
        "model": model_name, "n_doc": n_doc, "positions": positions,
        "base_accuracy_by_pos": [round(a, 4) for a in base_acc],
        "conditions": conditions,
    }


# ════════════════════════════════════════════════════════════════════════════════
# Local entrypoint
# ════════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(phase: str, model: str = "lmsys/vicuna-13b-v1.5", n_docs: str = "40,60,80"):
    import json
    import os
    from datetime import datetime, timezone

    here = os.path.dirname(os.path.abspath(__file__))
    exp071 = os.path.join(os.path.dirname(here), "exp-071_cloud_calibration")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    if phase == "calibrate":
        n_docs_list = [int(x) for x in n_docs.split(",") if x.strip()]
        print(f"=== CALIBRATE + CENSUS: {model}  n_docs={n_docs_list} ===")
        res = calibrate_and_census.remote(model, n_docs_list)
        res["timestamp_utc"] = ts
        res["phase"] = "exp-071 cloud calibration + census"
        out = os.path.join(exp071, "results.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nWrote {out}")
        print("\n=== CALIBRATION SUMMARY ===")
        for c in res["calibration"]:
            if c.get("skipped"):
                print(f"  n_doc={c['n_doc']}: SKIPPED (ctx {c.get('ctx_max')})")
            else:
                print(f"  n_doc={c['n_doc']}: V={c['V_task']:.3f} middle={c['middle']:.3f} "
                      f"edges_max={c['edges_max']:.3f} DEEP={c['deep_enough']}")
        print(f"  census conformal heads: {res['census']['n_conformal_heads']}, "
              f"target heads selected: {len(res['census']['target_heads'])}")
        deep = [c for c in res["calibration"] if not c.get("skipped") and c["deep_enough"]]
        if deep:
            best = max(deep, key=lambda c: c["V_task"])
            print(f"\n  → DEEP U-SHAPE FOUND at n_doc={best['n_doc']} (V={best['V_task']:.3f}). "
                  f"Lock this in the exp-072 pre-reg, then run phase=sweep.")
        else:
            print("\n  → No deep two-sided U-shape at these n_docs. Try larger n_docs "
                  "or the longchat-13b-16k fallback before pre-registering.")

    elif phase == "sweep":
        prereg = os.path.join(here, "prereg_locked.json")
        if not os.path.exists(prereg):
            raise SystemExit(f"Missing {prereg} — write + git-commit the locked config first.")
        with open(prereg) as f:
            lock = json.load(f)
        print(f"=== REGISTERED κ-SWEEP: {lock['model']} n_doc={lock['n_doc']} ===")
        res = run_kappa_sweep.remote(
            lock["model"], lock["n_doc"], lock["positions"],
            lock["target_heads"], lock["sham_heads"])
        res["timestamp_utc"] = ts
        res["preregistration_commit"] = lock.get("preregistration_commit", "UNCOMMITTED")
        out = os.path.join(here, "results.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nWrote {out}")
    else:
        raise SystemExit("phase must be 'calibrate' or 'sweep'")
