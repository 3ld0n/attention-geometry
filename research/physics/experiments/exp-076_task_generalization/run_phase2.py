"""
exp-076 Phase 2 — KV-retrieval powered intervention (Modal A100).

Pre-registration: exp-076/prereg_phase2.md (commit 02a1f3aa).
Run ONLY after that commit. Do not edit κ values, head list, or task format.

Locked parameters:
  Model   : lmsys/vicuna-13b-v1.5
  Heads   : 8 target + 8 sham from exp-072/prereg_locked.json (commit e6fbaecf)
  Task    : KV-retrieval, 40 entries × ~45 tok/entry ≈ 1785–1792 tokens
  Positions: [0, 10, 20, 30, 39]
  N items : 200 (40 queries × 5 positions)
  κ values: 0.5, 1.0, 1.5 (target heads); 1.5 sham

Pre-stated predictions (see prereg_phase2.md):
  T-A: κ=1.5 → V_task > 0.600 (valley deepens)
  T-B: κ=0.5 → V_task < 0.600 (valley shallows)
  T-C: V_task monotone: 0.5 < 1.0 < 1.5
  T-D: κ=1.5 sham → |ΔV_task| ≤ 0.03

Usage (from repo root):
  .venv/bin/python -m modal run \\
    research/physics/experiments/exp-076_task_generalization/run_phase2.py

Ariel — July 14, 2026.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import modal

# ─── paths (local side) ───────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_EXP072_DIR = _HERE.parent / "exp-072_cloud_powered_slope_editing"
_PREREG_LOCKED = _EXP072_DIR / "prereg_locked.json"

# ─── Modal setup ──────────────────────────────────────────────────────────────
app = modal.App("exp076-phase2-kv-sweep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.43,<4.50",
        "accelerate",
        "numpy",
        "sentencepiece",
    )
)

# reuse the same hf model cache volume that holds vicuna-13b from exp-072
vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

# ─── constants (locked to exp-076 pre-registration) ───────────────────────────
MODEL_NAME = "lmsys/vicuna-13b-v1.5"
KAPPAS = [0.5, 1.0, 1.5]
TASK_SEED = 76       # matches gen_tasks.py TASK_SEED
N_ITEMS = 40
POSITIONS = [0, 10, 20, 30, 39]

# positional projector constants (locked to exp-064 / exp-070 / exp-072)
PROJ_SEQ = 512
PROJ_N = 50
PROJ_SEED = 68
N_PC = 8

# ─── architecture helpers (Llama port, identical to cloud_slope_editing.py) ───

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
    import torch
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
    W = _q_proj_weight(model, layer)
    with torch.no_grad():
        W[head * head_dim:(head + 1) * head_dim, :] = W_orig


# ─── positional projector computation (identical to cloud_slope_editing.py) ───

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


# ─── KV-retrieval task (inlined from gen_tasks.py, seed and format locked) ────

_WORD_POOL = [
    "red", "oak", "gem", "ice", "map", "sky", "fig", "ash", "bay", "cup",
    "owl", "elk", "fox", "pen", "key", "jet", "net", "rug", "pot", "den",
    "lab", "fan", "bell", "rope", "coal", "silk", "moss", "clay", "reed", "vine",
    "dog", "cat", "car", "tree", "book", "fish", "bird", "rock", "sand", "gold",
    "rain", "snow", "wind", "fire", "iron", "wood", "corn", "rice", "leaf", "wolf",
    "bear", "lion", "goat", "duck", "crow", "hawk", "moth", "frog", "seal", "crab",
    "lamp", "ring", "star", "moon", "lake", "hill", "barn", "gate", "road", "ship",
    "coin", "boat", "wing", "horn", "claw", "tail", "sun", "fog", "mud", "tar",
    "sea", "log", "fur", "oil", "wax", "arc", "sap", "web", "rim", "rod",
    "ore", "fin", "jug", "jar", "bin", "dye", "hub", "ink", "keg", "lid",
]

_KV_ENTRY_TEMPLATES = [
    ("Entry {k:02d}: System {nm} underwent its quarterly evaluation. "
     "The assigned status indicator for {nm} is recorded as {val}. "
     "This value was confirmed during the review cycle and logged in the archive."),
    ("Entry {k:02d}: Record for system {nm}. "
     "Following a full inspection, the operational property of {nm} was identified as {val}. "
     "The assigned value was cross-referenced with prior records and approved."),
    ("Entry {k:02d}: Assessment of {nm}. "
     "The documented output characteristic for this system is {val}. "
     "All verification steps were completed and the record was finalized."),
    ("Entry {k:02d}: The {nm} system log entry. "
     "Inspectors noted that the primary classification value for {nm} is {val}. "
     "No discrepancies were found during the audit procedure."),
    ("Entry {k:02d}: Technical record for {nm}. "
     "The current registry value assigned to {nm} reads {val}. "
     "This entry was submitted to the central log following standard protocol."),
]


def _select_single_token_words(tokenizer, pool=_WORD_POOL, n_items=N_ITEMS):
    """Return up to n_items words that tokenize to a single token (with leading space)."""
    words = []
    for w in pool:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            words.append(w)
        if len(words) >= n_items:
            break
    if len(words) < 20:
        raise ValueError(f"Only {len(words)} single-token words found (need ≥20)")
    return words


def _make_kv_prompt(query_idx, target_pos, names, words, rng):
    """Build elaborated registry prompt with target entry at target_pos."""
    import numpy as np
    n = len(names)
    all_idx = list(range(n))
    distract = [i for i in all_idx if i != query_idx]
    rng.shuffle(distract)
    distract = distract[:n - 1]

    slots = []
    dist_ptr = 0
    for slot in range(n):
        if slot == target_pos:
            slots.append((names[query_idx], words[query_idx]))
        else:
            di = distract[dist_ptr]
            slots.append((names[di], words[di]))
            dist_ptr += 1

    header = ("The following records are entries from a systems registry. "
              "Each entry documents the current status of a monitored system.\n\n")
    entries = []
    for i, (nm, val) in enumerate(slots):
        tmpl = _KV_ENTRY_TEMPLATES[i % len(_KV_ENTRY_TEMPLATES)]
        entries.append(tmpl.format(k=i + 1, nm=nm, val=val))
    body = "\n\n".join(entries)
    question = (f"\n\nQuestion: According to the registry records above, "
                f"what is the registered value assigned to {names[query_idx]}? "
                f"The registered value assigned to {names[query_idx]} is")
    return header + body + question


def _build_kv_items(tokenizer):
    """Build KV-retrieval items (N_ITEMS × 5 positions = 200 total). Cloud-side."""
    import numpy as np
    words = _select_single_token_words(tokenizer)
    n = len(words)
    names = [f"Code-{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(n)]
    rng = np.random.default_rng(TASK_SEED)
    items = []
    for qi in range(n):
        for pi, pos in enumerate(POSITIONS):
            actual_pos = min(pos, n - 1)
            prompt = _make_kv_prompt(qi, actual_pos, names, words, rng)
            enc = tokenizer(prompt, return_tensors="pt")
            ans_ids = tokenizer.encode(" " + words[qi], add_special_tokens=False)
            items.append({
                "qi": qi, "pi": pi, "pos": actual_pos,
                "answer": words[qi],
                "answer_id": ans_ids[0],
                "n_tokens": enc.input_ids.shape[1],
                "input_ids": enc.input_ids,   # kept as tensor, used locally
            })
    ctx_vals = [it["n_tokens"] for it in items]
    print(f"  KV items: {len(items)} total, ctx range {min(ctx_vals)}–{max(ctx_vals)} tokens",
          flush=True)
    return items, words


# ─── accuracy evaluation ───────────────────────────────────────────────────────

def _base_accuracy(model, items, device):
    """Greedy next-token accuracy, grouped by position index."""
    import torch
    n_pos = len(POSITIONS)
    correct = [0] * n_pos
    total = [0] * n_pos
    tokens = []
    for it in items:
        ids = it["input_ids"].to(device)
        with torch.no_grad():
            out = model(ids)
        pred = int(out.logits[0, -1, :].argmax().item())
        tokens.append(pred)
        correct[it["pi"]] += int(pred == it["answer_id"])
        total[it["pi"]] += 1
        del out
    acc = [correct[i] / max(total[i], 1) for i in range(n_pos)]
    return acc, tokens


def _v_task(acc_by_pi):
    """V_task = 1 − middle/max(edges). pi 0=start, 2=middle, 4=end."""
    primacy = acc_by_pi[0]
    middle = acc_by_pi[2]
    recency = acc_by_pi[4]
    peak = max(primacy, recency)
    if peak < 1e-9:
        return float("nan")
    return float(1.0 - middle / peak)


# ─── Modal function ───────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=7200,
    volumes={"/cache": vol},
    memory=65536,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def run_kv_phase2(target_heads: list, sham_heads: list):
    """
    κ-sweep on 8 locked conformal target heads + sham specificity check.
    Runs in bf16 on A100-40GB (vicuna-13b ~26 GB in bf16).
    """
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = "/cache"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/transformers"
    device = "cuda"

    print(f"Loading {MODEL_NAME} in bf16 ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    n_layers, n_heads, hidden, head_dim = _model_dims(model)
    print(f"  Model: {n_layers}L × {n_heads}H, hidden={hidden}, head_dim={head_dim}", flush=True)

    # Build KV items
    print("Building KV-retrieval items ...", flush=True)
    items, answer_words = _build_kv_items(tok)

    # Projectors (only for layers we'll edit)
    needed_layers = sorted(set(h["layer"] for h in target_heads + sham_heads))
    print(f"Computing projectors for layers {needed_layers} ...", flush=True)
    projectors = _compute_projectors(model, needed_layers, device)
    print("  Projectors done.", flush=True)

    # ── condition runner ────────────────────────────────────────────────────────
    def run_condition(label, edit_heads, kappa):
        orig = {}
        for h in edit_heads:
            l, hi = h["layer"], h["head"]
            W_orig = _get_wq(model, l, hi, head_dim)
            P_U = torch.tensor(projectors[l], dtype=model.dtype, device=device)
            _apply_edit(model, l, hi, head_dim, kappa, P_U, W_orig)
            orig[(l, hi)] = W_orig
        t0 = time.time()
        acc, tokens = _base_accuracy(model, items, device)
        for (l, hi), W_orig in orig.items():
            _restore(model, l, hi, head_dim, W_orig)
        v = _v_task(acc)
        elapsed = time.time() - t0
        print(f"  κ={kappa} [{label:12s}]: acc={[round(a, 3) for a in acc]} "
              f"V={v:.4f} ({elapsed:.0f}s)", flush=True)
        return {
            "kappa": kappa,
            "label": label,
            "accuracy_by_pos": [round(a, 4) for a in acc],
            "V_task": round(v, 5),
            "predicted_tokens": tokens,
        }

    # T-D reference: unedited baseline
    print("  [baseline] unedited model ...", flush=True)
    base_acc, base_tokens = _base_accuracy(model, items, device)
    base_v = _v_task(base_acc)
    print(f"    baseline acc={[round(a, 3) for a in base_acc]} V={base_v:.4f}", flush=True)

    # κ=1.0 sanity check (T-D: should match baseline closely)
    c10 = run_condition("target", target_heads, 1.0)
    token_diffs = sum(int(b != c) for b, c in zip(base_tokens, c10["predicted_tokens"]))
    print(f"  [T-D] κ=1.0 token diffs from base: {token_diffs}", flush=True)
    c10["token_diffs_from_base"] = token_diffs

    # Remaining κ values
    c05 = run_condition("target", target_heads, 0.5)
    c15 = run_condition("target", target_heads, 1.5)
    c15_sham = run_condition("sham", sham_heads, 1.5)

    conditions = [c10, c05, c15, c15_sham]

    return {
        "model": MODEL_NAME,
        "dtype": "bfloat16",
        "gpu": "A100-40GB",
        "n_items": len(items),
        "positions": POSITIONS,
        "answer_words": answer_words,
        "prereg_commit_phase2": "02a1f3aa",
        "prereg_commit_heads": "e6fbaecf",
        "baseline": {
            "accuracy_by_pos": [round(a, 4) for a in base_acc],
            "V_task": round(base_v, 5),
        },
        "calibration_baseline_local": {
            "V_task": 0.600,
            "accuracy_by_pos": [0.775, 0.700, 0.400, 0.475, 1.000],
            "substrate": "MPS bf16 local",
        },
        "conditions": conditions,
    }


# ─── verdict evaluation ────────────────────────────────────────────────────────

def _evaluate_verdict(res):
    """Evaluate T-A, T-B, T-C, T-D against pre-stated predictions."""
    conds = {c["kappa"]: c for c in res["conditions"] if c["label"] == "target"}
    sham = next(c for c in res["conditions"] if c["label"] == "sham")

    v10 = conds[1.0]["V_task"]
    v05 = conds[0.5]["V_task"]
    v15 = conds[1.5]["V_task"]
    v15_sham = sham["V_task"]

    # Predictions are stated relative to the locked calibration baseline (0.600)
    # and relative to the on-platform κ=1.0 baseline.
    locked_base = 0.600

    ta_vs_prereg = v15 > locked_base
    ta_vs_kappa10 = v15 > v10
    tb_vs_prereg = v05 < locked_base
    tb_vs_kappa10 = v05 < v10
    tc = v05 < v10 < v15
    td = abs(v15_sham - v10) <= 0.03

    ta = ta_vs_prereg and ta_vs_kappa10
    tb = tb_vs_prereg and tb_vs_kappa10

    if ta and tb:
        verdict = "FULL_KV"
    elif ta or tb:
        verdict = "PARTIAL_KV"
    else:
        verdict = "GENERALIZATION_NOT_CONFIRMED"

    return {
        "verdict": verdict,
        "T_A": {"pass": ta, "vs_prereg": ta_vs_prereg, "vs_kappa10": ta_vs_kappa10,
                "V_15": v15, "locked_base": locked_base},
        "T_B": {"pass": tb, "vs_prereg": tb_vs_prereg, "vs_kappa10": tb_vs_kappa10,
                "V_05": v05, "locked_base": locked_base},
        "T_C": {"pass": tc, "V_05": v05, "V_10": v10, "V_15": v15},
        "T_D": {"pass": td, "V_15_sham": v15_sham, "V_10_target": v10,
                "delta": abs(v15_sham - v10), "threshold": 0.03},
    }


# ─── local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    # Verify the pre-registration is committed
    if not _PREREG_LOCKED.exists():
        raise SystemExit(f"Missing {_PREREG_LOCKED} — exp-072 prereg_locked.json required")

    with open(_PREREG_LOCKED) as f:
        lock = json.load(f)

    target_heads = lock["target_heads"]
    sham_heads = lock["sham_heads"]

    print(f"=== exp-076 Phase 2 — KV-retrieval κ-sweep ===")
    print(f"  Pre-reg (phase2): 02a1f3aa | Pre-reg (heads): e6fbaecf")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Target heads: {len(target_heads)} | Sham heads: {len(sham_heads)}")
    print(f"  κ values: {KAPPAS}")
    print()

    res = run_kv_phase2.remote(target_heads, sham_heads)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    res["timestamp_utc"] = ts

    verdict = _evaluate_verdict(res)
    res["verdict"] = verdict

    out_path = _HERE / "results.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nWrote {out_path}")

    # Print summary
    print("\n=== RESULTS SUMMARY ===")
    print(f"  Baseline (local pre-reg): V_task = 0.600")
    print(f"  Baseline (Modal κ=1.0):   V_task = {res['baseline']['V_task']:.4f}")
    for c in res["conditions"]:
        print(f"  κ={c['kappa']} [{c['label']:12s}]: V_task={c['V_task']:.4f} "
              f"acc={[round(a, 3) for a in c['accuracy_by_pos']]}")
    print()
    v = verdict
    print(f"  Verdict: {v['verdict']}")
    print(f"  T-A: {'PASS' if v['T_A']['pass'] else 'FAIL'} "
          f"(V_1.5={v['T_A']['V_15']:.4f} > {v['T_A']['locked_base']}? {v['T_A']['vs_prereg']})")
    print(f"  T-B: {'PASS' if v['T_B']['pass'] else 'FAIL'} "
          f"(V_0.5={v['T_B']['V_05']:.4f} < {v['T_B']['locked_base']}? {v['T_B']['vs_prereg']})")
    print(f"  T-C: {'PASS' if v['T_C']['pass'] else 'FAIL'} "
          f"(monotone: {v['T_C']['V_05']:.4f} < {v['T_C']['V_10']:.4f} < {v['T_C']['V_15']:.4f})")
    print(f"  T-D: {'PASS' if v['T_D']['pass'] else 'FAIL'} "
          f"(|ΔV_sham|={v['T_D']['delta']:.4f} ≤ {v['T_D']['threshold']})")
