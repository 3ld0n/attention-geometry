"""
exp-068 — Task-level slope editing: does κ-rescaling of QK matrices change
          long-context retrieval accuracy in the predicted direction?

Pre-registration: research/physics/notes/2026-06-14_task_level_slope_editing_prereg.md
Pre-reg commit:   6ed3feec (before any task code was written)

Background
----------
exp-064 established that κ-rescaling of GPT-2 W_Q matrices moves the per-head
conformal dimension Δ̂ and the attention valley statistic in the predicted direction
(8/8 mechanical, 24/24 behavioral sign agreements).

This experiment tests whether that attention-level causal handle propagates to
task-level accuracy on a Liu-et-al.-style multi-document retrieval task in
Pythia-1.4b (which shows a cleaner U-shape than GPT-2-scale models).

Stages (run in order):
  python run_exp068.py select    # head selection from exp-059 census; no model load
  python run_exp068.py run       # load Pythia-1.4b, build task, evaluate all κ
  python run_exp068.py evaluate  # compute registered verdicts T-A/T-B/T-C/T-D

Edit operator (same as exp-064):
  W_Q_h ← W_Q_h @ M_edit     (Pythia Linear convention: weight (out, in), M_edit symmetric)
  M_edit = I_nd + (κ − 1) · P_U
  P_U = top-8 PCs of position-mean input_layernorm output (50 random inputs, L=512)

Ariel — June 16, 2026.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

# ── paths ─────────────────────────────────────────────────────────────────────

OUT_DIR   = Path(__file__).resolve().parent
RESULTS   = OUT_DIR / "results.json"
EXP059    = OUT_DIR.parent / "exp-059_split_half_stability"
PREREG    = "research/physics/notes/2026-06-14_task_level_slope_editing_prereg.md"

# ── constants (locked to pre-registration) ─────────────────────────────────────

MODEL_ID   = "EleutherAI/pythia-1.4b"
N_LAYERS   = 24
N_HEADS    = 16
HEAD_DIM   = 128       # 2048 / 16
HIDDEN     = 2048

# Head selection (pre-registration §Experimental Setup)
LAYER_LO   = 8
LAYER_HI   = 20        # inclusive
R2_MIN     = 0.85
DELTA_LO   = 0.10
DELTA_HI   = 0.40
N_SELECT   = 8
N_PC       = 8         # PCs for positional projector

# κ conditions
KAPPAS     = [0.5, 1.0, 1.5, 2.0]

# Projector inputs: random tokens, same convention as exp-059
PROJ_SEQ   = 512
PROJ_N     = 50
PROJ_SEED  = 68        # distinct from exp-059 seed 2026

# Lag fitting (exp-059 constants)
FIT_LOW    = 3
FIT_HIGH   = 120
LOG_X      = np.log(np.arange(FIT_LOW, FIT_HIGH).astype(float))
X_C        = LOG_X - LOG_X.mean()
SXX        = float(np.sum(X_C ** 2))

# Task design (pre-registration §Experimental Setup)
N_DOC      = 20        # documents per context
N_ITEMS    = 10        # query targets (distinct questions)
POSITIONS  = [0, 4, 9, 15, 19]   # 0-indexed; pre-reg: {1, 5, middle, N-4, N}
POS_NAMES  = ["primacy", "near_primacy", "middle", "near_recency", "recency"]
TASK_SEED  = 42

# answer words: short, common; verified single-token in GPT-NeoX tokenizer at runtime
QUERY_NAMES  = [f"Item-{i+1:03d}" for i in range(N_ITEMS)]
QUERY_WORDS  = ["red", "oak", "gem", "ice", "map", "sky", "fig", "ash", "bay", "cup"]

# distractor pool (N_DOC - 1 = 19 needed per context; use 20 for the pool)
DIST_NAMES   = [f"Item-{i+1:03d}" for i in range(N_ITEMS, N_ITEMS + 20)]
DIST_WORDS   = ["sun", "fog", "mud", "tar", "sea", "log", "fur", "oil", "wax", "arc",
                "tor", "sap", "web", "rim", "rod", "ore", "fin", "jug", "nap", "jar"]

# ── utilities ─────────────────────────────────────────────────────────────────

def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load() -> dict:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return {"experiment": "exp-068",
            "preregistration": PREREG,
            "preregistration_commit": "6ed3feec"}


def _save(res: dict) -> None:
    res["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    RESULTS.write_text(json.dumps(res, indent=1))


# ── exp-059 fitting helpers ────────────────────────────────────────────────────

def _fit_heads(mean_lagprof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mean_lagprof: (n_heads, FIT_HIGH) → (delta, R2) arrays. From exp-059."""
    y = np.log(np.maximum(mean_lagprof[:, FIT_LOW:FIT_HIGH], 1e-30))
    y_c = y - y.mean(axis=1, keepdims=True)
    slope = (y_c @ X_C) / SXX
    pred  = slope[:, None] * X_C[None, :]
    ss_res = np.sum((y_c - pred) ** 2, axis=1)
    ss_tot = np.sum(y_c ** 2, axis=1)
    r2 = np.where(ss_tot > 1e-30, 1.0 - ss_res / np.maximum(ss_tot, 1e-30), 0.0)
    return -slope / 2.0, r2


# ── task construction ──────────────────────────────────────────────────────────

def _make_document(name: str, word: str, doc_num: int) -> str:
    return (
        f"Document {doc_num}: {name} is a project identifier in the central registry. "
        f"The assigned keyword for {name} is {word}. "
        f"This keyword {word} was registered with {name} following standard protocol. "
        f"The identifier {name} requires keyword {word} for all authorization requests. "
        f"All records for {name} are maintained in the archive under keyword {word}."
    )


def _make_prompt(query_name: str, query_word: str, answer_pos: int,
                 dist_order: list[int]) -> str:
    """Build a 20-document context with the answer at answer_pos (0-indexed)."""
    header = (
        "The following documents contain project registry records. "
        "Read all documents carefully, then answer the question.\n\n"
    )
    docs = []
    dist_ptr = 0
    for slot in range(N_DOC):
        doc_num = slot + 1
        if slot == answer_pos:
            docs.append(_make_document(query_name, query_word, doc_num))
        else:
            di = dist_order[dist_ptr]
            docs.append(_make_document(DIST_NAMES[di], DIST_WORDS[di], doc_num))
            dist_ptr += 1
    body = "\n\n".join(docs)
    question = (
        f"\n\nQuestion: According to the documents above, "
        f"what is the assigned keyword for {query_name}? "
        f"The assigned keyword for {query_name} is"
    )
    return header + body + question


def _build_task(tokenizer) -> dict:
    """Build all 50 task inputs. Returns dict with prompts, token_ids, answer_ids."""
    rng = np.random.default_rng(TASK_SEED)
    items = []
    for qi in range(N_ITEMS):
        # fixed distractor order for this question (19 out of 20 distractor pool)
        dist_pool = list(range(len(DIST_NAMES)))
        rng.shuffle(dist_pool)
        dist19 = dist_pool[:N_DOC - 1]    # 19 distractors, fixed per question
        for pi, pos in enumerate(POSITIONS):
            prompt = _make_prompt(QUERY_NAMES[qi], QUERY_WORDS[qi], pos, dist19)
            enc    = tokenizer(prompt, return_tensors="pt")
            ans_ids = tokenizer.encode(
                " " + QUERY_WORDS[qi], add_special_tokens=False)
            items.append({
                "qi": qi, "pi": pi, "pos_name": POS_NAMES[pi],
                "answer_word": QUERY_WORDS[qi],
                "answer_id": ans_ids[0],          # first token of answer
                "n_tokens": enc.input_ids.shape[1],
                "input_ids": enc.input_ids,        # (1, seq)
            })
    return items


# ── positional projectors (all needed layers, single pass set) ────────────────

def _compute_all_projectors(model, needed_layers: list[int],
                             device: torch.device) -> dict[int, np.ndarray]:
    """Run PROJ_N forward passes once, accumulate ln output for ALL needed layers.
    Returns {layer: projector (nd, nd)} for each layer in needed_layers."""
    rng  = np.random.default_rng(PROJ_SEED)
    vocab = model.config.vocab_size
    acc  = {layer: np.zeros((PROJ_SEQ, HIDDEN), dtype=np.float64)
            for layer in needed_layers}
    for i in range(PROJ_N):
        ids = torch.tensor(
            rng.integers(0, vocab, size=(1, PROJ_SEQ)), dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
            for layer in needed_layers:
                h = out.hidden_states[layer]                           # (1, seq, nd)
                ln_out = model.gpt_neox.layers[layer].input_layernorm(h)
                acc[layer] += ln_out[0].cpu().float().numpy()
        if (i + 1) % 10 == 0:
            print(f"    projector pass {i+1}/{PROJ_N}", flush=True)

    projectors = {}
    for layer in needed_layers:
        xbar = acc[layer] / PROJ_N
        xbar -= xbar.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(xbar, full_matrices=False)
        U = vt[:N_PC].T                    # (nd, N_PC)
        projectors[layer] = (U @ U.T).astype(np.float32)
    return projectors


# ── W_Q edit for Pythia (Linear convention) ───────────────────────────────────

def _get_wq(model, layer: int, head: int) -> torch.Tensor:
    """Return a clone of W_Q_h: rows h*HEAD_DIM:(h+1)*HEAD_DIM of query_key_value.weight."""
    W = model.gpt_neox.layers[layer].attention.query_key_value.weight
    return W[head * HEAD_DIM:(head + 1) * HEAD_DIM, :].detach().clone()


def _apply_edit(model, layer: int, head: int,
                kappa: float, P_U: torch.Tensor, W_orig: torch.Tensor) -> None:
    """Apply W_Q_h ← W_Q_h_orig @ M_edit.  M_edit symmetric → M_edit.T = M_edit."""
    M = torch.eye(HIDDEN, device=W_orig.device) + (kappa - 1.0) * P_U
    W = model.gpt_neox.layers[layer].attention.query_key_value.weight
    with torch.no_grad():
        W[head * HEAD_DIM:(head + 1) * HEAD_DIM, :] = W_orig @ M


def _restore(model, layer: int, head: int, W_orig: torch.Tensor) -> None:
    W = model.gpt_neox.layers[layer].attention.query_key_value.weight
    with torch.no_grad():
        W[head * HEAD_DIM:(head + 1) * HEAD_DIM, :] = W_orig


# ── valley score ──────────────────────────────────────────────────────────────

def _valley(acc_by_pos: list[float]) -> float:
    """V_task = 1 − acc(middle) / max(acc(primacy), acc(recency))."""
    primacy  = acc_by_pos[0]   # POSITIONS[0] = 0
    recency  = acc_by_pos[4]   # POSITIONS[4] = 19
    middle   = acc_by_pos[2]   # POSITIONS[2] = 9
    denom = max(primacy, recency)
    if denom < 1e-9:
        return float("nan")
    return float(1.0 - middle / denom)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: select
# ══════════════════════════════════════════════════════════════════════════════

def stage_select() -> None:
    print("=== Stage 1: head selection from exp-059 census ===")
    gz = EXP059 / "per_input_pythia-1.4b.json.gz"
    with gzip.open(gz) as f:
        data = json.load(f)

    lagprof = np.array(data["lag_profile"], dtype=np.float32)  # (50, 384, 120)
    mean_lag = lagprof.mean(axis=0)                             # (384, 120)
    delta, r2 = _fit_heads(mean_lag)                            # each (384,)

    # thirds for valley estimate
    thirds = np.array(data["thirds"], dtype=np.float32)        # (50, 384, 3)
    mean_thirds = thirds.mean(axis=0)                           # (384, 3)
    s, m, e = mean_thirds[:, 0], mean_thirds[:, 1], mean_thirds[:, 2]
    peak = np.maximum(s, e)
    valley = np.where(peak > 1e-15, 1.0 - m / np.maximum(peak, 1e-30), np.nan)

    # enumerate candidate heads in layers LAYER_LO..LAYER_HI (inclusive)
    cands = []
    for layer in range(LAYER_LO, LAYER_HI + 1):
        for head in range(N_HEADS):
            fh = layer * N_HEADS + head
            d, r, v = float(delta[fh]), float(r2[fh]), float(valley[fh])
            if r >= R2_MIN and DELTA_LO <= d <= DELTA_HI:
                cands.append({"layer": layer, "head": head, "flat_head": fh,
                               "delta_1d": round(d, 5), "r2_1d": round(r, 5),
                               "valley_census": round(v, 5) if not np.isnan(v) else None})

    cands.sort(key=lambda x: -x["r2_1d"])
    target = cands[:N_SELECT]
    sham   = cands[N_SELECT:N_SELECT * 2]   # next N_SELECT by R²

    print(f"Candidates passing criteria: {len(cands)}")
    print(f"Target heads ({N_SELECT}):")
    for h in target:
        print(f"  L{h['layer']}H{h['head']} Δ={h['delta_1d']:.3f} R²={h['r2_1d']:.3f} "
              f"valley={h['valley_census']}")
    print(f"Sham heads ({len(sham)}):")
    for h in sham:
        print(f"  L{h['layer']}H{h['head']} Δ={h['delta_1d']:.3f} R²={h['r2_1d']:.3f}")

    res = _load()
    res["stage1_head_selection"] = {
        "committed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "source": "exp-059 Pythia-1.4b per-input lag profiles",
        "criteria": {"R2_min": R2_MIN, "delta_lo": DELTA_LO, "delta_hi": DELTA_HI,
                     "layer_range": f"{LAYER_LO}–{LAYER_HI}", "n_select": N_SELECT},
        "n_candidates": len(cands),
        "target_heads": target,
        "sham_heads": sham,
    }
    _save(res)
    print("Stage 1 done. Results written.")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: run
# ══════════════════════════════════════════════════════════════════════════════

def _run_condition(model, task_items, device, label: str,
                   edit_heads: list[dict], kappa: float,
                   projectors: dict) -> dict:
    """Apply κ-edit to edit_heads, evaluate all task inputs, restore weights."""
    # apply edits
    orig_weights = {}
    for h in edit_heads:
        layer, head = h["layer"], h["head"]
        key = (layer, head)
        W_orig = _get_wq(model, layer, head).to(device)
        P_U = torch.tensor(projectors[key], dtype=torch.float32, device=device)
        _apply_edit(model, layer, head, kappa, P_U, W_orig)
        orig_weights[key] = W_orig

    # evaluate
    correct_by_pos = [0] * len(POSITIONS)   # indexed by POSITIONS index
    total_by_pos   = [0] * len(POSITIONS)
    tokens_by_item = []
    t0 = time.time()
    for item in task_items:
        ids = item["input_ids"].to(device)
        with torch.no_grad():
            out = model(ids)
        pred = int(out.logits[0, -1, :].argmax().item())
        tokens_by_item.append(pred)
        pi = item["pi"]
        correct_by_pos[pi] += int(pred == item["answer_id"])
        total_by_pos[pi]   += 1
    elapsed = time.time() - t0

    # restore
    for (layer, head), W_orig in orig_weights.items():
        _restore(model, layer, head, W_orig)

    acc = [correct_by_pos[pi] / max(total_by_pos[pi], 1) for pi in range(len(POSITIONS))]
    v   = _valley(acc)
    print(f"  κ={kappa} [{label}]: acc={[round(a, 3) for a in acc]} V={v:.3f} ({elapsed:.0f}s)")
    return {
        "kappa": kappa, "label": label,
        "accuracy_by_pos": {POS_NAMES[pi]: round(acc[pi], 4) for pi in range(len(POSITIONS))},
        "correct_by_pos":  {POS_NAMES[pi]: correct_by_pos[pi] for pi in range(len(POSITIONS))},
        "total_by_pos":    {POS_NAMES[pi]: total_by_pos[pi] for pi in range(len(POSITIONS))},
        "V_task": round(v, 5) if not np.isnan(v) else None,
        "predicted_tokens": tokens_by_item,
    }


def stage_run() -> None:
    print("=== Stage 2: task run ===")
    res = _load()
    sel = res["stage1_head_selection"]
    target_heads = sel["target_heads"]
    sham_heads   = sel["sham_heads"]

    device = _device()
    print(f"Device: {device}")

    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Building task inputs...")
    task_items = _build_task(tokenizer)

    # verify single-token answers
    print("Verifying answer tokenization...")
    for qi in range(N_ITEMS):
        word = QUERY_WORDS[qi]
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"QUERY_WORDS[{qi}]='{word}' tokenizes to {len(ids)} tokens: {ids}")
        print(f"  {word!r} → token {ids[0]} ({tokenizer.decode([ids[0]])!r}) ✓")

    # report context lengths
    lens = [item["n_tokens"] for item in task_items]
    print(f"Context lengths: min={min(lens)} max={max(lens)} mean={np.mean(lens):.0f}")
    if max(lens) > 2048:
        raise ValueError(f"Context too long: max={max(lens)} > 2048")

    print("Loading Pythia-1.4b fp32...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
        attn_implementation="eager").to(device).eval()

    # compute positional projectors per target layer
    unique_layers = sorted(set(h["layer"] for h in target_heads + sham_heads))
    print(f"Computing positional projectors for layers: {unique_layers} (single-pass)")
    layer_projectors = _compute_all_projectors(model, unique_layers, device)
    projectors = {}
    for layer, P in layer_projectors.items():
        for h in range(N_HEADS):
            projectors[(layer, h)] = P
        print(f"  layer {layer}: P_U trace={float(np.trace(P)):.2f}")

    print("\nRunning κ conditions (target heads)...")
    condition_results = []

    # T-D sham: κ=1.0 must match base model exactly
    print("  [T-D] κ=1.0 sham (identity check)...")
    base_tokens = []
    for item in task_items:
        ids = item["input_ids"].to(device)
        with torch.no_grad():
            out = model(ids)
        base_tokens.append(int(out.logits[0, -1, :].argmax().item()))

    cond_10 = _run_condition(model, task_items, device, "target", target_heads, 1.0, projectors)
    sham_diff = sum(int(b != c) for b, c in zip(base_tokens, cond_10["predicted_tokens"]))
    cond_10["sham_token_diff_from_base"] = sham_diff
    print(f"  [T-D] token differences from base model: {sham_diff}")
    condition_results.append(cond_10)

    # all other κ
    for kap in [0.5, 1.5, 2.0]:
        cond = _run_condition(model, task_items, device, "target", target_heads, kap, projectors)
        condition_results.append(cond)

    # sham-heads condition at κ=1.5 (specificity control)
    print("\nRunning sham-heads condition (κ=1.5 on sham set)...")
    if sham_heads:
        cond_sham = _run_condition(model, task_items, device, "sham_heads",
                                   sham_heads, 1.5, projectors)
        condition_results.append(cond_sham)
    else:
        print("  No sham heads available (not enough candidates).")

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    res["stage2_task_run"] = {
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "model": MODEL_ID,
        "n_task_items": len(task_items),
        "positions": {POS_NAMES[pi]: POSITIONS[pi] for pi in range(len(POSITIONS))},
        "conditions": condition_results,
    }
    _save(res)
    print("Stage 2 done.")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: evaluate
# ══════════════════════════════════════════════════════════════════════════════

def stage_evaluate() -> None:
    print("=== Stage 3: evaluate registered predictions ===")
    res = _load()
    conds = res["stage2_task_run"]["conditions"]

    # index by (kappa, label)
    by_kappa_target = {}
    sham_heads_cond = None
    for c in conds:
        if c["label"] == "target":
            by_kappa_target[c["kappa"]] = c
        elif c["label"] == "sham_heads":
            sham_heads_cond = c

    def v(kap):
        return by_kappa_target[kap]["V_task"]

    # T-D: κ=1.0 sham null
    td_diff = by_kappa_target[1.0].get("sham_token_diff_from_base", None)
    td_pass = (td_diff == 0)
    td_verdict = "PASS" if td_pass else f"FAIL ({td_diff} token differences)"
    print(f"T-D sham null: {td_verdict}")

    # T-A: V_task(1.5) > V_task(1.0)
    ta_keep = (v(1.5) is not None and v(1.0) is not None and v(1.5) > v(1.0))
    ta_strong = (ta_keep and v(1.5) > v(1.0) + 0.05)
    ta_verdict = "KEEP" + (" (strong)" if ta_strong else "") if ta_keep else "KILL"
    print(f"T-A deepening (κ=1.5 > κ=1.0): V(1.5)={v(1.5)} V(1.0)={v(1.0)} → {ta_verdict}")

    # T-B: V_task(0.5) < V_task(1.0)
    tb_keep = (v(0.5) is not None and v(1.0) is not None and v(0.5) < v(1.0))
    tb_verdict = "KEEP" if tb_keep else "KILL"
    print(f"T-B shallowing (κ=0.5 < κ=1.0): V(0.5)={v(0.5)} V(1.0)={v(1.0)} → {tb_verdict}")

    # T-C: Spearman ρ(κ, V_task) ≥ 0.80 on {0.5, 1.0, 1.5, 2.0}
    kaps_ord = [0.5, 1.0, 1.5, 2.0]
    v_vals   = [v(k) for k in kaps_ord]
    all_valid = all(x is not None and not np.isnan(x) for x in v_vals)
    if all_valid:
        rho_tc, p_tc = spearmanr(kaps_ord, v_vals)
        rho_tc, p_tc = float(rho_tc), float(p_tc)
        tc_verdict = ("KEEP" if rho_tc >= 0.80
                      else "AMBIGUOUS" if rho_tc >= 0 else "KILL")
    else:
        rho_tc, p_tc = None, None
        tc_verdict = "AMBIGUOUS (missing V_task values)"
    print(f"T-C monotone κ→V_task: ρ={rho_tc} → {tc_verdict}")
    print(f"  V values at κ={kaps_ord}: {v_vals}")

    # sham-heads specificity (secondary)
    if sham_heads_cond:
        v_sham = sham_heads_cond["V_task"]
        v_target_15 = v(1.5)
        spec_pass = (v_target_15 is not None and v_sham is not None
                     and v_target_15 > v_sham)
        print(f"Sham-heads specificity: V_target(1.5)={v_target_15} vs V_sham_heads(1.5)={v_sham} "
              f"→ {'PASS' if spec_pass else 'FAIL'}")
    else:
        spec_pass = None

    # floor / ceiling check
    all_acc = []
    for c in [by_kappa_target[1.0]]:
        for pos in POS_NAMES:
            all_acc.append(c["accuracy_by_pos"][pos])
    has_ceiling = all(a >= 0.90 for a in all_acc)
    has_floor   = all(a <= 0.10 for a in all_acc)
    power_note  = ("CEILING_EFFECT — underpowered, not KILL/CONFIRM" if has_ceiling
                   else "FLOOR_EFFECT — underpowered, not KILL/CONFIRM" if has_floor
                   else "power OK")
    print(f"Power check: {power_note}")
    print(f"  Base model accuracy by position: {by_kappa_target[1.0]['accuracy_by_pos']}")

    evaluation = {
        "T_D": {"verdict": td_verdict, "token_diffs": td_diff},
        "T_A": {"verdict": ta_verdict, "V_15": v(1.5), "V_10": v(1.0)},
        "T_B": {"verdict": tb_verdict, "V_05": v(0.5), "V_10": v(1.0)},
        "T_C": {"verdict": tc_verdict, "rho": rho_tc, "p": p_tc,
                "kappa_list": kaps_ord, "V_list": v_vals},
        "sham_heads_specificity": spec_pass,
        "power_note": power_note,
    }
    res["stage3_evaluation"] = evaluation
    _save(res)
    print("\n=== SUMMARY ===")
    print(f"T-A (deepening):  {ta_verdict}")
    print(f"T-B (shallowing): {tb_verdict}")
    print(f"T-C (monotone):   {tc_verdict}")
    print(f"T-D (sham null):  {td_verdict}")
    print(f"Power: {power_note}")
    overall = ("BEHAVIORAL_CAUSALITY_CONFIRMED"
               if ta_keep and tb_keep and td_pass
               else "BEHAVIORAL_CAUSALITY_KILLED"
               if (not ta_keep or not tb_keep) and not has_ceiling and not has_floor
               else "UNDERPOWERED")
    print(f"\nOverall verdict: {overall}")
    evaluation["overall_verdict"] = overall
    _save(res)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cmds = {"select": stage_select, "run": stage_run, "evaluate": stage_evaluate}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(f"Usage: python run_exp068.py <{'|'.join(cmds)}>")
        sys.exit(1)
    cmds[sys.argv[1]]()
