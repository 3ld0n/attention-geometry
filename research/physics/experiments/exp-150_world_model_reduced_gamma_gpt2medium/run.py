"""
exp-150 — World-model battery: reduced-γ suppress-only in GPT-2 medium

Pre-registration: attention-geometry 3b44c1c (pushed 2026-09-20 before this script).
Tests the over-suppression hypothesis: γ=−0.86 targets ~50× relative κ̃ reduction
(matching exp-142's small-model regime), vs γ=−1.0 which gave ~600× in exp-149.

Hypothesis: the Task B degradation in exp-147/149 is caused by over-suppression.
Calibrating γ to achieve ~50× relative reduction should recover positive ΔP_B.

STEEP_LOCAL (suppress γ=−0.86): L4H13, L15H8, L8H7, L5H11, L11H7
  κ̃ before: 43.2, 39.9, 33.3, 31.7, 30.9 (from exp-149)
  κ̃ target: /50 → ~0.86, 0.80, 0.67, 0.63, 0.62

No structural head amplification (exp-148 confirmed amplification is inert).

Ariel — September 21, 2026, ~9:05 AM MDT. Solo.
"""

from __future__ import annotations
import copy
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, GPT2Tokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

SEQ_LEN  = 512
N_INPUTS = 50
SEED     = 42

PREREG_COMMIT = "3b44c1c"

D_MODEL  = 1024   # GPT-2 medium
D_HEAD   = 64
N_LAYERS = 24
N_HEADS  = 16

MODEL_ID = "openai-community/gpt2-medium"

# Steep/local heads (unchanged from exp-147/149)
STEEP_LOCAL = [(4, 13), (15, 8), (8, 7), (5, 11), (11, 7)]
GAMMA_SUP   = -0.86    # pre-registered; targets ~50× relative κ̃ reduction

# Kill conditions
# K3_low:  κ̃_after/κ̃_before < 1/150 on ≥ 2/5 heads → still over-suppressing
# K3_high: κ̃_after/κ̃_before > 1/10  on ≥ 2/5 heads → insufficient suppression
K3_RATIO_LOW  = 1.0 / 150.0   # below this: still over-suppressing
K3_RATIO_HIGH = 1.0 / 10.0    # above this: insufficient suppression
K3_N_FAIL     = 2              # threshold: ≥ 2/5 heads outside window → kill

SHAM_SEED_BASE = 2026092001    # distinct from all prior experiments

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ── Positional field (ln_1 output hook) ───────────────────────────────────────

def compute_positional_field_ln1(model, layer: int,
                                  rng: np.random.Generator) -> np.ndarray:
    """
    Positional field δ at the OUTPUT of ln_1 for attention block `layer`.
    Returns δ: shape (SEQ_LEN, D_MODEL) — centered positional deviation.
    """
    model.eval()
    tok_ids = rng.integers(0, model.config.vocab_size, size=(N_INPUTS, SEQ_LEN))
    tokens  = torch.tensor(tok_ids, dtype=torch.long, device=DEVICE)

    captured = []

    def hook_fn(mod, inp, out):
        captured.append(out.detach().cpu().float().numpy())

    ln1_module = model.transformer.h[layer].ln_1
    handle = ln1_module.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(tokens)

    handle.remove()

    acts  = np.concatenate(captured, axis=0)   # (N_INPUTS, SEQ_LEN, D_MODEL)
    xbar  = acts.mean(axis=0)                  # (SEQ_LEN, D_MODEL)
    m     = xbar.mean(axis=0, keepdims=True)
    delta = xbar - m                           # centered positional field
    return delta


def compute_kappa(W_K: np.ndarray, delta: np.ndarray) -> float:
    """κ̃(W_K) — isotropic-normalised positional capture (exp-137 formula)."""
    reads      = delta @ W_K
    cap        = float((np.linalg.norm(reads, axis=1) ** 2).sum() /
                       (np.linalg.norm(delta, axis=1) ** 2).sum())
    norm_factor = float((W_K ** 2).sum()) / D_MODEL
    return cap / norm_factor if norm_factor > 1e-14 else 0.0


def get_wk(model, layer: int, head: int) -> np.ndarray:
    """W_K for (layer, head) from GPT-2 c_attn weight. Returns (D_MODEL, D_HEAD)."""
    W = model.transformer.h[layer].attn.c_attn.weight.detach().cpu().double().numpy()
    offset = D_MODEL + head * D_HEAD
    return W[:, offset: offset + D_HEAD]


def set_wk(model, layer: int, head: int, W_K_new: np.ndarray):
    """Write W_K back for (layer, head)."""
    W = model.transformer.h[layer].attn.c_attn.weight.data
    with torch.no_grad():
        W[:, D_MODEL + head * D_HEAD: D_MODEL + (head + 1) * D_HEAD] = torch.tensor(
            W_K_new, dtype=W.dtype, device=W.device
        )


# ── Build suppressed and sham models ──────────────────────────────────────────

def build_models(base_model, rng_census: np.random.Generator):
    """
    Suppressed model: suppress STEEP_LOCAL heads at γ=−0.86 (targeting ~50×
    relative κ̃ reduction).
    Sham model: matched-norm sham in ⊥ complement.
    No structural head amplification.

    Returns (sup_model, sham_model, kappa_sup).
    """
    sup_model  = copy.deepcopy(base_model)
    sham_model = copy.deepcopy(base_model)
    kappa_sup  = {}

    print(f"\n[A] Suppressing steep/local heads (γ={GAMMA_SUP}, targeting ~50× relative reduction):",
          flush=True)
    for i, (ell, h) in enumerate(STEEP_LOCAL):
        delta = compute_positional_field_ln1(base_model, ell, rng_census)

        _, _, Vt  = np.linalg.svd(delta, full_matrices=False)
        P_k       = Vt[:4]
        W_K       = get_wk(base_model, ell, h)
        W_K_proj  = P_k.T @ (P_k @ W_K)

        kappa_before = compute_kappa(W_K, delta)

        # Suppression at γ=−0.86
        W_K_sup         = W_K + GAMMA_SUP * W_K_proj
        kappa_after_sup = compute_kappa(W_K_sup, delta)
        ratio           = kappa_after_sup / kappa_before if kappa_before > 1e-14 else 0.0
        set_wk(sup_model, ell, h, W_K_sup)

        # Sham: matched Frobenius norm in ⊥ complement (same construction as exp-149)
        sham_rng  = np.random.default_rng(SHAM_SEED_BASE + i)
        rand_vecs = sham_rng.standard_normal((4, D_MODEL))
        for pk_row in P_k:
            rand_vecs -= (rand_vecs @ pk_row)[:, None] * pk_row
        P_perp, _ = np.linalg.qr(rand_vecs.T)
        P_perp    = P_perp.T
        W_K_perp_proj   = P_perp.T @ (P_perp @ W_K)
        delta_sup_norm  = np.linalg.norm(GAMMA_SUP * W_K_proj, "fro")
        delta_perp_norm = np.linalg.norm(W_K_perp_proj, "fro")
        gamma_sham = delta_sup_norm / delta_perp_norm if delta_perp_norm > 1e-14 else 0.0
        W_K_sham_sup      = W_K + gamma_sham * W_K_perp_proj
        kappa_after_sham  = compute_kappa(W_K_sham_sup, delta)
        set_wk(sham_model, ell, h, W_K_sham_sup)

        kappa_sup[f"L{ell}H{h}"] = {
            "before":      float(kappa_before),
            "after_sup":   float(kappa_after_sup),
            "after_sham":  float(kappa_after_sham),
            "ratio_sup":   float(ratio),
            "gamma_sup":   float(GAMMA_SUP),
            "gamma_sham":  float(gamma_sham),
        }
        print(f"  L{ell}H{h}: κ̃ {kappa_before:.3f} → sup {kappa_after_sup:.4f} "
              f"(ratio {ratio:.4f}×, sham {kappa_after_sham:.3f})", flush=True)

    return sup_model, sham_model, kappa_sup


# ── Task items (identical to exp-141/142/143/145/147/148/149) ─────────────────

TASK_A = [
    ("The lamp was on. Dave turned the lamp off. Then he turned the lamp on again. The lamp is now", "on"),
    ("The jar was open. Maria closed the jar. The jar is now", "closed"),
    ("The door was closed. Tom opened the door and walked through. The door is", "open"),
    ("The bottle was full. He poured half of it out. Then he filled it back up. The bottle is now", "full"),
    ("The cat was outside. Anna let the cat inside. Then she let the cat back outside. The cat is", "outside"),
    ("The window was shut. James opened the window. The window is now", "open"),
    ("The bag was empty. Lisa filled the bag with books. Then she emptied it again. The bag is now", "empty"),
    ("The switch was off. He flipped the switch on, then flipped it off again. The switch is now", "off"),
    ("The box was closed. She opened the box, took out an apple, and closed the box again. The box is now", "closed"),
    ("The faucet was running. He turned the faucet off. The faucet is now", "off"),
    ("The light was on. She turned it off. Then she turned it back on. The light is", "on"),
    ("The cup was full. He drank half and then refilled it. The cup is now", "full"),
    ("The fire was burning. The rain put the fire out. The fire is now", "out"),
    ("The gate was open. She closed the gate behind her. The gate is now", "closed"),
    ("The phone was off. He turned the phone on to make a call, then turned it back off. The phone is now", "off"),
    ("The drawer was shut. She opened it, removed a pen, and shut it again. The drawer is now", "shut"),
    ("The engine was running. He stopped the engine. The engine is now", "off"),
    ("The curtains were open. She closed them for the night. The curtains are now", "closed"),
    ("The stove was off. She turned it on to cook, then turned it off after eating. The stove is now", "off"),
    ("The valve was open. The plumber closed the valve. The valve is now", "closed"),
]

TASK_B = [
    ("Colors: red, blue, green, yellow. The second color is", "blue"),
    ("Fruits: apple, mango, cherry, grape. The third fruit is", "cherry"),
    ("Animals: cat, dog, fish, bird, rabbit. The fourth animal is", "bird"),
    ("Months: January, March, July, October. The third month is", "July"),
    ("Planets: Mars, Venus, Jupiter, Saturn, Mercury. The second planet is", "Venus"),
    ("Numbers: one, three, seven, twelve. The third number is", "seven"),
    ("Names: Alice, Bob, Carol, Dan. The first name is", "Alice"),
    ("Countries: France, Spain, Italy, Greece, Poland. The fourth country is", "Greece"),
    ("Shapes: circle, square, triangle, oval. The second shape is", "square"),
    ("Seasons: spring, summer, autumn, winter. The third season is", "autumn"),
    ("Metals: gold, silver, copper, iron. The third metal is", "copper"),
    ("Days: Monday, Wednesday, Friday, Sunday. The second day is", "Wednesday"),
    ("Birds: eagle, robin, sparrow, hawk, dove. The third bird is", "sparrow"),
    ("Letters: alpha, beta, gamma, delta, epsilon. The fourth letter is", "delta"),
    ("Coins: penny, nickel, dime, quarter. The second coin is", "nickel"),
    ("Stars: Sirius, Vega, Rigel, Altair. The first star is", "Sirius"),
    ("Trees: oak, pine, maple, birch, cedar. The third tree is", "maple"),
    ("Gems: ruby, sapphire, emerald, diamond. The third gem is", "emerald"),
    ("Spices: salt, pepper, cumin, thyme. The second spice is", "pepper"),
    ("Flowers: rose, lily, tulip, daisy, violet. The fourth flower is", "daisy"),
]


# ── Task scoring ───────────────────────────────────────────────────────────────

def score_task(model, tokenizer, items, task_name):
    model.eval()
    results = []
    for idx, (prompt, answer) in enumerate(items):
        inputs      = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        correct_tok = tokenizer.encode(" " + answer, add_special_tokens=False)
        if not correct_tok:
            correct_tok = tokenizer.encode(answer, add_special_tokens=False)
        correct_id  = correct_tok[0]

        with torch.no_grad():
            out    = model(**inputs)
            logits = out.logits[0, -1, :].float()
            lps    = torch.log_softmax(logits, dim=-1)
            lp     = float(lps[correct_id])
            rank   = int((logits > logits[correct_id]).sum().item()) + 1

        results.append({
            "item":             f"{task_name}{idx+1:02d}",
            "prompt":           prompt[:60] + "...",
            "answer":           answer,
            "correct_tok_id":   int(correct_id),
            "log_prob_correct": lp,
            "rank_correct":     rank,
        })
        print(f"    {task_name}{idx+1:02d}: '{answer}' logP={lp:.2f}  rank={rank}", flush=True)
    return results


def median_lp(results):
    return float(np.median([r["log_prob_correct"] for r in results]))


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(kappa_sup, orig_A, orig_B, sup_A, sup_B, sham_A, sham_B):
    med_orig_A = median_lp(orig_A)
    med_orig_B = median_lp(orig_B)
    med_sup_A  = median_lp(sup_A)
    med_sup_B  = median_lp(sup_B)
    med_sham_A = median_lp(sham_A)
    med_sham_B = median_lp(sham_B)

    dA_sup  = med_sup_A  - med_orig_A
    dB_sup  = med_sup_B  - med_orig_B
    dA_sham = med_sham_A - med_orig_A
    dB_sham = med_sham_B - med_orig_B

    medians = {
        "orig_A": med_orig_A, "orig_B": med_orig_B,
        "sup_A":  med_sup_A,  "sup_B":  med_sup_B,
        "sham_A": med_sham_A, "sham_B": med_sham_B,
    }
    deltas = {"sup_A": dA_sup, "sup_B": dB_sup,
              "sham_A": dA_sham, "sham_B": dB_sham}

    # Kill checks
    ratios = [v["ratio_sup"] for v in kappa_sup.values()]
    n_k3_low  = sum(1 for r in ratios if r < K3_RATIO_LOW)
    n_k3_high = sum(1 for r in ratios if r > K3_RATIO_HIGH)

    k3_low  = {"fired": n_k3_low  >= K3_N_FAIL,
                "n_below_threshold": n_k3_low,
                "threshold": K3_RATIO_LOW,
                "interpretation": "still over-suppressing like exp-149"}
    k3_high = {"fired": n_k3_high >= K3_N_FAIL,
                "n_above_threshold": n_k3_high,
                "threshold": K3_RATIO_HIGH,
                "interpretation": "insufficient suppression"}
    k2      = {"fired": med_orig_A < -20.0 or med_orig_B < -20.0,
               "orig_A": med_orig_A, "orig_B": med_orig_B}
    k1_raw  = (abs(dA_sham) >= abs(dA_sup)) and (abs(dB_sham) >= abs(dB_sup))
    k1      = {"fired": k1_raw,
               "abs_sham_A": abs(dA_sham), "abs_sup_A": abs(dA_sup),
               "abs_sham_B": abs(dB_sham), "abs_sup_B": abs(dB_sup)}

    kills      = {"K1": k1, "K2": k2, "K3_low": k3_low, "K3_high": k3_high}
    any_kill   = k1["fired"] or k2["fired"] or k3_low["fired"] or k3_high["fired"]

    # P_check: regime check — ≥ 4/5 heads in [1/100, 1/20] window
    n_in_window = sum(1 for r in ratios if K3_RATIO_LOW <= r <= K3_RATIO_HIGH)
    p_check     = {"n_in_window": n_in_window, "fired": n_in_window >= 4, "ratios": ratios}

    # Prediction verdicts
    p1      = dB_sup > 0.10
    p2      = abs(dA_sup - dA_sham) < 0.5
    p_null  = abs(dB_sup) <= 0.05
    p_degrade = dB_sup < -0.10

    n_B_improved = sum(1 for o, c in zip(orig_B, sup_B)
                       if c["log_prob_correct"] > o["log_prob_correct"])

    if any_kill:
        overall = "INCONCLUSIVE (kill fired)"
    elif p_degrade:
        overall = "NULL — reduced suppression still degrades Task B; over-suppression hypothesis weakened"
    elif p_null:
        overall = "NULL — suppression inert at γ=−0.86"
    elif p1 and p2:
        overall = "CONFIRMED — γ calibration recovers positive Task B signal; over-suppression hypothesis supported"
    elif p1:
        overall = "PARTIAL — directional improvement (P1), Task A asymmetry (P2 fails)"
    elif dB_sup > 0:
        overall = "PARTIAL — directional improvement, below P1 threshold"
    else:
        overall = "INCONCLUSIVE"

    verdicts = {
        "P1":       {"fired": p1,       "dB_sup": dB_sup, "threshold": 0.10},
        "P2":       {"fired": p2,        "dA_sup": dA_sup, "dA_sham": dA_sham,
                                          "diff": dA_sup - dA_sham},
        "P_check":  p_check,
        "P_null":   {"fired": p_null,    "dB_sup": dB_sup},
        "P_degrade":{"fired": p_degrade, "dB_sup": dB_sup},
        "n_B_improved": n_B_improved,
        "overall":      overall,
    }

    comparison = {
        "exp142_gpt2small_suppress_only_gamma1p0_dB":   +0.33,
        "exp149_gpt2medium_suppress_only_gamma1p0_dB":  -0.10,
        "exp150_gpt2medium_suppress_only_gamma0p86_dB": dB_sup,
        "direction_matches_small":                       dB_sup > 0,
        "note": ("exp-142 used γ=−1.0 in small (κ̃ ~10→0.2, ~50× reduction); "
                 "exp-150 uses γ=−0.86 in medium (targeting same relative reduction)"),
    }

    return kills, verdicts, medians, deltas, comparison


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("exp-150 — World-model battery GPT-2 medium, reduced-γ suppress-only", flush=True)
    print(f"  prereg:    {PREREG_COMMIT} (attention-geometry, pushed 2026-09-20 before this script)", flush=True)
    print(f"  device:    {DEVICE}", flush=True)
    print(f"  STEEP_LOCAL (suppress γ={GAMMA_SUP}): "
          f"{['L{}H{}'.format(l,h) for l,h in STEEP_LOCAL]}", flush=True)
    print(f"  K3_low threshold: ratio < {K3_RATIO_LOW:.4f} (1/150) on ≥ {K3_N_FAIL}/5 heads → still over-suppressing", flush=True)
    print(f"  K3_high threshold: ratio > {K3_RATIO_HIGH:.3f} (1/10) on ≥ {K3_N_FAIL}/5 heads → insufficient", flush=True)
    print("  No structural amplification in this experiment.", flush=True)

    print("\n[1] Loading GPT-2 medium (eager attention)...", flush=True)
    tokenizer  = GPT2Tokenizer.from_pretrained("openai-community/gpt2-medium")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager",    # required: MPS SDPA silently returns 0 attention
    ).to(DEVICE)
    base_model.eval()
    print(f"  Loaded. n_layer={N_LAYERS}, n_head={N_HEADS}, D_MODEL={D_MODEL}", flush=True)

    print("\n[2] Building suppressed and sham models...", flush=True)
    rng_census = np.random.default_rng(SEED)
    sup_model, sham_model, kappa_sup = build_models(base_model, rng_census)

    # Regime check report
    ratios = [v["ratio_sup"] for v in kappa_sup.values()]
    n_in_window = sum(1 for r in ratios if K3_RATIO_LOW <= r <= K3_RATIO_HIGH)
    n_k3_low  = sum(1 for r in ratios if r < K3_RATIO_LOW)
    n_k3_high = sum(1 for r in ratios if r > K3_RATIO_HIGH)
    print(f"\n  Regime check: {n_in_window}/5 heads in [1/150, 1/10] window", flush=True)
    print(f"    Ratios: {[f'{r:.4f}' for r in ratios]}", flush=True)
    if n_k3_low >= 2:
        print("  WARNING: K3_low fires — still over-suppressing (like exp-149).", flush=True)
    if n_k3_high >= 2:
        print("  WARNING: K3_high fires — insufficient suppression.", flush=True)

    print("\n[3] Task A — original:", flush=True)
    orig_A  = score_task(base_model,  tokenizer, TASK_A, "A")
    print("\n[4] Task A — suppressed:", flush=True)
    sup_A   = score_task(sup_model,   tokenizer, TASK_A, "A")
    print("\n[5] Task A — sham:", flush=True)
    sham_A  = score_task(sham_model,  tokenizer, TASK_A, "A")

    print("\n[6] Task B — original:", flush=True)
    orig_B  = score_task(base_model,  tokenizer, TASK_B, "B")
    print("\n[7] Task B — suppressed:", flush=True)
    sup_B   = score_task(sup_model,   tokenizer, TASK_B, "B")
    print("\n[8] Task B — sham:", flush=True)
    sham_B  = score_task(sham_model,  tokenizer, TASK_B, "B")

    print("\n[9] Evaluating...", flush=True)
    kills, verdicts, medians, deltas, comparison = evaluate(
        kappa_sup, orig_A, orig_B, sup_A, sup_B, sham_A, sham_B
    )

    print(f"\n  Overall verdict: {verdicts['overall']}", flush=True)
    m = medians
    d = deltas
    print(f"  Task A: orig={m['orig_A']:.2f}  sup={m['sup_A']:.2f}  sham={m['sham_A']:.2f}  "
          f"(Δsup={d['sup_A']:+.2f}  Δsham={d['sham_A']:+.2f})", flush=True)
    print(f"  Task B: orig={m['orig_B']:.2f}  sup={m['sup_B']:.2f}  sham={m['sham_B']:.2f}  "
          f"(Δsup={d['sup_B']:+.2f}  Δsham={d['sham_B']:+.2f})", flush=True)

    n_A_improved_sup  = sum(1 for o, c in zip(orig_A, sup_A)
                            if c["log_prob_correct"] > o["log_prob_correct"])
    n_B_improved_sup  = sum(1 for o, c in zip(orig_B, sup_B)
                            if c["log_prob_correct"] > o["log_prob_correct"])
    n_A_improved_sham = sum(1 for o, s in zip(orig_A, sham_A)
                            if s["log_prob_correct"] > o["log_prob_correct"])
    n_B_improved_sham = sum(1 for o, s in zip(orig_B, sham_B)
                            if s["log_prob_correct"] > o["log_prob_correct"])
    print(f"\n  Item-level (sup vs orig):  Task A={n_A_improved_sup}/20   Task B={n_B_improved_sup}/20",
          flush=True)
    print(f"  Item-level (sham vs orig): Task A={n_A_improved_sham}/20  Task B={n_B_improved_sham}/20",
          flush=True)

    print(f"\n  Comparison:", flush=True)
    print(f"    exp-142 (GPT-2 small, suppress-only, γ=−1.0, CONFIRMED): ΔP_B = +0.33 nats (50× reduction)", flush=True)
    print(f"    exp-149 (GPT-2 medium, suppress-only, γ=−1.0, P_degrade): ΔP_B = −0.10 nats (600× reduction)", flush=True)
    print(f"    exp-150 (GPT-2 medium, suppress-only, γ=−0.86, this run):  ΔP_B = {d['sup_B']:+.2f} nats", flush=True)

    for pk, pv in verdicts.items():
        if pk not in ("overall", "n_B_improved", "P_check", "comparison_to_prior"):
            if isinstance(pv, dict):
                print(f"  {pk}: {'FIRES' if pv.get('fired') else 'not fired'}", flush=True)

    out = {
        "exp":                      "exp-150",
        "prereg_commit":            PREREG_COMMIT,
        "prereg_evidence":          "git-attested — commit 3b44c1c pushed 2026-09-20 before run.py written",
        "device":                   DEVICE,
        "model":                    MODEL_ID,
        "gamma_sup":                GAMMA_SUP,
        "steep_local_heads":        [f"L{l}H{h}" for l, h in STEEP_LOCAL],
        "structural_amplified":     False,
        "kappa_sup":                kappa_sup,
        "kills":                    kills,
        "verdicts":                 verdicts,
        "medians":                  medians,
        "deltas":                   deltas,
        "comparison_to_prior":      comparison,
        "item_counts": {
            "n_A_improved_sup":  n_A_improved_sup,
            "n_B_improved_sup":  n_B_improved_sup,
            "n_A_improved_sham": n_A_improved_sham,
            "n_B_improved_sham": n_B_improved_sham,
        },
        "task_A_orig":              orig_A,
        "task_A_sup":               sup_A,
        "task_A_sham":              sham_A,
        "task_B_orig":              orig_B,
        "task_B_sup":               sup_B,
        "task_B_sham":              sham_B,
    }

    out_path = Path(__file__).resolve().parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
