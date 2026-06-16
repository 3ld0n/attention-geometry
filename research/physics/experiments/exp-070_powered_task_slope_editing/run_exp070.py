"""
exp-070 — POWERED task-level slope editing: does κ-rescaling of QK matrices change
          long-context retrieval accuracy in the predicted direction, on a task that
          actually shows a base-model U-shape?

Pre-registration: research/physics/notes/2026-06-16_powered_task_slope_editing_prereg.md
Pre-reg commit:   9e6239b9 (committed BEFORE any edited-weight forward pass)

Predecessor: exp-068 (UNDERPOWERED — keyword task at ceiling, no U-shape).
Calibration: exp-069 (found Pythia-1.4b embedded@40doc gives base V_task=0.15).

This script is exp-068's run script with EXACTLY ONE change: the task is the
exp-069 embedded-prose retrieval task at N_doc=40 (40 query items × 5 positions).
Head selection (exp-059 census), positional projector, κ-edit operator, and the
T-A/T-B/T-C/T-D evaluation are reused verbatim.

Stages:
  python run_exp070.py select     # head selection from exp-059 census; no model load
  python run_exp070.py run         # load Pythia-1.4b, build task, evaluate all κ
  python run_exp070.py evaluate    # registered verdicts T-A/T-B/T-C/T-D

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

OUT_DIR = Path(__file__).resolve().parent
RESULTS = OUT_DIR / "results.json"
EXP059 = OUT_DIR.parent / "exp-059_split_half_stability"
PREREG = "research/physics/notes/2026-06-16_powered_task_slope_editing_prereg.md"
PREREG_COMMIT = "9e6239b9"

# ── constants (locked to pre-registration) ─────────────────────────────────────

MODEL_ID = "EleutherAI/pythia-1.4b"
N_LAYERS = 24
N_HEADS = 16
HEAD_DIM = 128
HIDDEN = 2048

# Head selection (unchanged from exp-068)
LAYER_LO = 8
LAYER_HI = 20
R2_MIN = 0.85
DELTA_LO = 0.10
DELTA_HI = 0.40
N_SELECT = 8
N_PC = 8

KAPPAS = [0.5, 1.0, 1.5, 2.0]

PROJ_SEQ = 512
PROJ_N = 50
PROJ_SEED = 68

FIT_LOW = 3
FIT_HIGH = 120
LOG_X = np.log(np.arange(FIT_LOW, FIT_HIGH).astype(float))
X_C = LOG_X - LOG_X.mean()
SXX = float(np.sum(X_C ** 2))

# ── Task design (LOCKED to exp-070 pre-reg: embedded@40doc) ────────────────────

N_DOC = 40
N_ITEMS = 40
POSITIONS = [0, 10, 20, 29, 39]           # primacy / near_primacy / middle / near_recency / recency
POS_NAMES = ["primacy", "near_primacy", "middle", "near_recency", "recency"]
TASK_SEED = 42

# candidate single-token answer words (verified at runtime; first N_ITEMS used)
WORD_POOL = [
    "red", "oak", "gem", "ice", "map", "sky", "fig", "ash", "bay", "cup",
    "owl", "elk", "fox", "pen", "key", "jet", "net", "rug", "pot", "den",
    "lab", "fan", "bell", "rope", "coal", "silk", "moss", "clay", "reed", "vine",
    "dog", "cat", "car", "tree", "book", "fish", "bird", "rock", "sand", "gold",
    "iron", "wood", "wine", "milk", "salt", "corn", "rice", "leaf", "wolf", "bear",
    "lion", "goat", "duck", "crow", "hawk", "moth", "frog", "seal", "crab", "lamp",
    "ring", "star", "moon", "lake", "hill", "barn", "gate", "road", "ship", "coin",
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
# exp-069 config C6 used filler=0 (ctx≈1734 tok, base V_task=0.15). Keep the document
# shape identical so the locked base profile reproduces.
FILLER_PER_DOC = 0


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
    return {"experiment": "exp-070",
            "preregistration": PREREG,
            "preregistration_commit": PREREG_COMMIT}


def _save(res: dict) -> None:
    res["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    RESULTS.write_text(json.dumps(res, indent=1))


# ── exp-059 fitting helpers (verbatim from exp-068) ────────────────────────────

def _fit_heads(mean_lagprof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.log(np.maximum(mean_lagprof[:, FIT_LOW:FIT_HIGH], 1e-30))
    y_c = y - y.mean(axis=1, keepdims=True)
    slope = (y_c @ X_C) / SXX
    pred = slope[:, None] * X_C[None, :]
    ss_res = np.sum((y_c - pred) ** 2, axis=1)
    ss_tot = np.sum(y_c ** 2, axis=1)
    r2 = np.where(ss_tot > 1e-30, 1.0 - ss_res / np.maximum(ss_tot, 1e-30), 0.0)
    return -slope / 2.0, r2


# ── task construction (embedded prose; matched cloze) ──────────────────────────

def _make_document(name: str, word: str, doc_num: int) -> str:
    s = [f"Document {doc_num}: The {name} survey produced a number of findings during its operation."]
    s.append(f"Among the recorded details, the team noted that the {name} signal marker "
             f"was set to {word} throughout the study.")
    for i in range(FILLER_PER_DOC):
        s.append(FILLERS_TEMPLATE[i % len(FILLERS_TEMPLATE)].format(n=name))
    return " ".join(s)


def _make_prompt(query_name: str, query_word: str, answer_pos: int,
                 dist_names: list[str], dist_order: list[int]) -> str:
    header = ("The following documents contain project registry records. "
              "Read all documents carefully, then answer the question.\n\n")
    docs = []
    dist_ptr = 0
    for slot in range(N_DOC):
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


def _select_words(tokenizer) -> list[str]:
    words = []
    for w in WORD_POOL:
        if len(tokenizer.encode(" " + w, add_special_tokens=False)) == 1:
            words.append(w)
        if len(words) >= N_ITEMS:
            break
    if len(words) < N_ITEMS:
        raise ValueError(f"Only {len(words)} single-token words for {N_ITEMS} items")
    return words


def _build_task(tokenizer) -> list[dict]:
    rng = np.random.default_rng(TASK_SEED)
    words = _select_words(tokenizer)
    query_names = [f"Item-{i + 1:03d}" for i in range(N_ITEMS)]
    n_dist = N_DOC - 1
    dist_names = [f"Item-{i + 1:03d}" for i in range(N_ITEMS, N_ITEMS + n_dist + 5)]
    items = []
    for qi in range(N_ITEMS):
        dist_pool = list(range(len(dist_names)))
        rng.shuffle(dist_pool)
        dist_sel = dist_pool[:n_dist]
        for pi, pos in enumerate(POSITIONS):
            prompt = _make_prompt(query_names[qi], words[qi], pos, dist_names, dist_sel)
            enc = tokenizer(prompt, return_tensors="pt")
            ans_ids = tokenizer.encode(" " + words[qi], add_special_tokens=False)
            items.append({
                "qi": qi, "pi": pi, "pos_name": POS_NAMES[pi],
                "answer_word": words[qi],
                "answer_id": ans_ids[0],
                "n_tokens": enc.input_ids.shape[1],
                "input_ids": enc.input_ids,
            })
    return items


# ── positional projectors (verbatim from exp-068) ──────────────────────────────

def _compute_all_projectors(model, needed_layers: list[int],
                            device: torch.device) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(PROJ_SEED)
    vocab = model.config.vocab_size
    acc = {layer: np.zeros((PROJ_SEQ, HIDDEN), dtype=np.float64) for layer in needed_layers}
    for i in range(PROJ_N):
        ids = torch.tensor(rng.integers(0, vocab, size=(1, PROJ_SEQ)),
                           dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
            for layer in needed_layers:
                h = out.hidden_states[layer]
                ln_out = model.gpt_neox.layers[layer].input_layernorm(h)
                acc[layer] += ln_out[0].cpu().float().numpy()
        if (i + 1) % 10 == 0:
            print(f"    projector pass {i+1}/{PROJ_N}", flush=True)
    projectors = {}
    for layer in needed_layers:
        xbar = acc[layer] / PROJ_N
        xbar -= xbar.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(xbar, full_matrices=False)
        U = vt[:N_PC].T
        projectors[layer] = (U @ U.T).astype(np.float32)
    return projectors


# ── W_Q edit (verbatim from exp-068) ───────────────────────────────────────────

def _get_wq(model, layer: int, head: int) -> torch.Tensor:
    W = model.gpt_neox.layers[layer].attention.query_key_value.weight
    return W[head * HEAD_DIM:(head + 1) * HEAD_DIM, :].detach().clone()


def _apply_edit(model, layer, head, kappa, P_U, W_orig) -> None:
    M = torch.eye(HIDDEN, device=W_orig.device) + (kappa - 1.0) * P_U
    W = model.gpt_neox.layers[layer].attention.query_key_value.weight
    with torch.no_grad():
        W[head * HEAD_DIM:(head + 1) * HEAD_DIM, :] = W_orig @ M


def _restore(model, layer, head, W_orig) -> None:
    W = model.gpt_neox.layers[layer].attention.query_key_value.weight
    with torch.no_grad():
        W[head * HEAD_DIM:(head + 1) * HEAD_DIM, :] = W_orig


def _valley(acc_by_pos: list[float]) -> float:
    primacy, recency, middle = acc_by_pos[0], acc_by_pos[4], acc_by_pos[2]
    denom = max(primacy, recency)
    if denom < 1e-9:
        return float("nan")
    return float(1.0 - middle / denom)


# ── Stage 1: select (verbatim from exp-068) ────────────────────────────────────

def stage_select() -> None:
    print("=== Stage 1: head selection from exp-059 census ===")
    gz = EXP059 / "per_input_pythia-1.4b.json.gz"
    with gzip.open(gz) as f:
        data = json.load(f)
    lagprof = np.array(data["lag_profile"], dtype=np.float32)
    mean_lag = lagprof.mean(axis=0)
    delta, r2 = _fit_heads(mean_lag)
    thirds = np.array(data["thirds"], dtype=np.float32)
    mean_thirds = thirds.mean(axis=0)
    s, m, e = mean_thirds[:, 0], mean_thirds[:, 1], mean_thirds[:, 2]
    peak = np.maximum(s, e)
    valley = np.where(peak > 1e-15, 1.0 - m / np.maximum(peak, 1e-30), np.nan)

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
    sham = cands[N_SELECT:N_SELECT * 2]
    print(f"Candidates: {len(cands)}")
    for h in target:
        print(f"  TARGET L{h['layer']}H{h['head']} Δ={h['delta_1d']:.3f} R²={h['r2_1d']:.3f}")
    for h in sham:
        print(f"  SHAM   L{h['layer']}H{h['head']} Δ={h['delta_1d']:.3f} R²={h['r2_1d']:.3f}")
    res = _load()
    res["stage1_head_selection"] = {
        "committed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "source": "exp-059 Pythia-1.4b per-input lag profiles (same as exp-068)",
        "criteria": {"R2_min": R2_MIN, "delta_lo": DELTA_LO, "delta_hi": DELTA_HI,
                     "layer_range": f"{LAYER_LO}-{LAYER_HI}", "n_select": N_SELECT},
        "n_candidates": len(cands), "target_heads": target, "sham_heads": sham,
    }
    _save(res)
    print("Stage 1 done.")


# ── Stage 2: run (verbatim from exp-068, embedded task via _build_task) ─────────

def _run_condition(model, task_items, device, label, edit_heads, kappa, projectors) -> dict:
    orig_weights = {}
    for h in edit_heads:
        layer, head = h["layer"], h["head"]
        W_orig = _get_wq(model, layer, head).to(device)
        P_U = torch.tensor(projectors[(layer, head)], dtype=torch.float32, device=device)
        _apply_edit(model, layer, head, kappa, P_U, W_orig)
        orig_weights[(layer, head)] = W_orig

    correct_by_pos = [0] * len(POSITIONS)
    total_by_pos = [0] * len(POSITIONS)
    tokens = []
    t0 = time.time()
    for item in task_items:
        ids = item["input_ids"].to(device)
        with torch.no_grad():
            out = model(ids)
        pred = int(out.logits[0, -1, :].argmax().item())
        tokens.append(pred)
        pi = item["pi"]
        correct_by_pos[pi] += int(pred == item["answer_id"])
        total_by_pos[pi] += 1
    elapsed = time.time() - t0

    for (layer, head), W_orig in orig_weights.items():
        _restore(model, layer, head, W_orig)

    acc = [correct_by_pos[pi] / max(total_by_pos[pi], 1) for pi in range(len(POSITIONS))]
    v = _valley(acc)
    print(f"  κ={kappa} [{label}]: acc={[round(a,3) for a in acc]} V={v:.4f} ({elapsed:.0f}s)", flush=True)
    return {
        "kappa": kappa, "label": label,
        "accuracy_by_pos": {POS_NAMES[pi]: round(acc[pi], 4) for pi in range(len(POSITIONS))},
        "correct_by_pos": {POS_NAMES[pi]: correct_by_pos[pi] for pi in range(len(POSITIONS))},
        "total_by_pos": {POS_NAMES[pi]: total_by_pos[pi] for pi in range(len(POSITIONS))},
        "V_task": round(v, 5) if not np.isnan(v) else None,
        "predicted_tokens": tokens,
    }


def stage_run() -> None:
    print("=== Stage 2: task run (embedded@40doc) ===")
    res = _load()
    sel = res["stage1_head_selection"]
    target_heads, sham_heads = sel["target_heads"], sel["sham_heads"]
    device = _device()
    print(f"Device: {device}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading tokenizer + building task...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    task_items = _build_task(tokenizer)
    lens = [it["n_tokens"] for it in task_items]
    print(f"N items: {len(task_items)}  ctx tokens: min={min(lens)} max={max(lens)} mean={np.mean(lens):.0f}")
    if max(lens) > 2048:
        raise ValueError(f"Context too long: {max(lens)} > 2048")

    print("Loading Pythia-1.4b fp32...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager").to(device).eval()

    unique_layers = sorted(set(h["layer"] for h in target_heads + sham_heads))
    print(f"Computing positional projectors for layers: {unique_layers}")
    layer_projectors = _compute_all_projectors(model, unique_layers, device)
    projectors = {}
    for layer, P in layer_projectors.items():
        for h in range(N_HEADS):
            projectors[(layer, h)] = P

    conditions = []
    print("  [T-D] base model (κ identity reference)...")
    base_tokens = []
    for item in task_items:
        ids = item["input_ids"].to(device)
        with torch.no_grad():
            out = model(ids)
        base_tokens.append(int(out.logits[0, -1, :].argmax().item()))

    cond_10 = _run_condition(model, task_items, device, "target", target_heads, 1.0, projectors)
    cond_10["sham_token_diff_from_base"] = sum(int(b != c) for b, c in zip(base_tokens, cond_10["predicted_tokens"]))
    print(f"  [T-D] token differences from base: {cond_10['sham_token_diff_from_base']}")
    conditions.append(cond_10)

    for kap in [0.5, 1.5, 2.0]:
        conditions.append(_run_condition(model, task_items, device, "target", target_heads, kap, projectors))

    print("Running sham-heads condition (κ=1.5)...")
    conditions.append(_run_condition(model, task_items, device, "sham_heads", sham_heads, 1.5, projectors))

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    res["stage2_task_run"] = {
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "model": MODEL_ID, "task": "embedded@40doc (exp-069 C6)",
        "n_task_items": len(task_items),
        "positions": {POS_NAMES[pi]: POSITIONS[pi] for pi in range(len(POSITIONS))},
        "conditions": conditions,
    }
    _save(res)
    print("Stage 2 done.")


# ── Stage 3: evaluate (verbatim from exp-068) ──────────────────────────────────

def stage_evaluate() -> None:
    print("=== Stage 3: evaluate registered predictions ===")
    res = _load()
    conds = res["stage2_task_run"]["conditions"]
    by_kappa_target = {}
    sham_heads_cond = None
    for c in conds:
        if c["label"] == "target":
            by_kappa_target[c["kappa"]] = c
        elif c["label"] == "sham_heads":
            sham_heads_cond = c

    def v(kap):
        return by_kappa_target[kap]["V_task"]

    td_diff = by_kappa_target[1.0].get("sham_token_diff_from_base")
    td_pass = (td_diff == 0)
    td_verdict = "PASS" if td_pass else f"FAIL ({td_diff} token differences)"

    ta_keep = (v(1.5) is not None and v(1.0) is not None and v(1.5) > v(1.0))
    ta_strong = (ta_keep and v(1.5) > v(1.0) + 0.05)
    ta_verdict = ("KEEP" + (" (strong)" if ta_strong else "")) if ta_keep else "KILL"

    tb_keep = (v(0.5) is not None and v(1.0) is not None and v(0.5) < v(1.0))
    tb_verdict = "KEEP" if tb_keep else "KILL"

    kaps_ord = [0.5, 1.0, 1.5, 2.0]
    v_vals = [v(k) for k in kaps_ord]
    all_valid = all(x is not None and not np.isnan(x) for x in v_vals)
    if all_valid:
        rho_tc, p_tc = spearmanr(kaps_ord, v_vals)
        rho_tc, p_tc = float(rho_tc), float(p_tc)
        tc_verdict = "KEEP" if rho_tc >= 0.80 else "AMBIGUOUS" if rho_tc >= 0 else "KILL"
    else:
        rho_tc, p_tc, tc_verdict = None, None, "AMBIGUOUS (missing V)"

    if sham_heads_cond:
        v_sham = sham_heads_cond["V_task"]
        spec_pass = (v(1.5) is not None and v_sham is not None and v(1.5) > v_sham)
    else:
        spec_pass = None

    all_acc = [by_kappa_target[1.0]["accuracy_by_pos"][pos] for pos in POS_NAMES]
    has_ceiling = all(a >= 0.90 for a in all_acc)
    has_floor = all(a <= 0.10 for a in all_acc)
    power_note = ("CEILING_EFFECT — underpowered" if has_ceiling
                  else "FLOOR_EFFECT — underpowered" if has_floor else "power OK")

    overall = ("BEHAVIORAL_CAUSALITY_CONFIRMED" if ta_keep and tb_keep and td_pass
               else "BEHAVIORAL_CAUSALITY_KILLED" if (not ta_keep or not tb_keep) and not has_ceiling and not has_floor
               else "UNDERPOWERED")

    evaluation = {
        "T_D": {"verdict": td_verdict, "token_diffs": td_diff},
        "T_A": {"verdict": ta_verdict, "V_15": v(1.5), "V_10": v(1.0)},
        "T_B": {"verdict": tb_verdict, "V_05": v(0.5), "V_10": v(1.0)},
        "T_C": {"verdict": tc_verdict, "rho": rho_tc, "p": p_tc, "kappa_list": kaps_ord, "V_list": v_vals},
        "sham_heads_specificity": spec_pass,
        "power_note": power_note,
        "base_accuracy_by_pos": by_kappa_target[1.0]["accuracy_by_pos"],
        "overall_verdict": overall,
    }
    res["stage3_evaluation"] = evaluation
    _save(res)
    print("\n=== SUMMARY ===")
    print(f"T-A (deepening):  {ta_verdict}  V(1.5)={v(1.5)} V(1.0)={v(1.0)}")
    print(f"T-B (shallowing): {tb_verdict}  V(0.5)={v(0.5)} V(1.0)={v(1.0)}")
    print(f"T-C (monotone):   {tc_verdict}  ρ={rho_tc}  V@κ={v_vals}")
    print(f"T-D (sham null):  {td_verdict}")
    print(f"Sham-heads specificity: {spec_pass}")
    print(f"Power: {power_note}   base acc: {by_kappa_target[1.0]['accuracy_by_pos']}")
    print(f"\nOverall verdict: {overall}")


# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cmds = {"select": stage_select, "run": stage_run, "evaluate": stage_evaluate}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(f"Usage: python run_exp070.py <{'|'.join(cmds)}>")
        sys.exit(1)
    cmds[sys.argv[1]]()
