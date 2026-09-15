"""
exp-143 — World-model battery: combined manipulation
(amplify Δ-window heads + suppress steep/local heads simultaneously)

Pre-registration: attention-geometry 110832c (pushed before this script).
Analysis-only: GPT-2 small (cached). No new training.

The antagonism model from exp-141/142:
  - Amplifying Δ-window heads (L2H1, L3H4, L5H0, L7H11, L10H8) → Task B +0.27 nats
  - Suppressing steep/local heads (L0H10, L10H5, L8H7, L7H0, L7H9) → Task B +0.33 nats
  - Both shams flat. Both move Task B in the improvement direction.

This experiment applies both simultaneously in a single model copy. If the
antagonism is real, the effects should be additive or super-additive.

Ariel — September 15, 2026, ~2:00 AM MDT, solo.
"""

from __future__ import annotations
import copy
import json
import sys
from pathlib import Path
import importlib.util

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Import shared census constants from exp-112 ───────────────────────────────

HERE = Path(__file__).resolve().parent
EXP107 = HERE.parent / "exp-107_natural_text_bilocal"
EXP112 = HERE.parent / "exp-112_score_drift_decomposition"
sys.path.insert(0, str(EXP107))
spec = importlib.util.spec_from_file_location("exp112", EXP112 / "measure_scores.py")
exp112 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp112)

SEQ_LEN  = exp112.SEQ_LEN    # 512
N_INPUTS = exp112.N_INPUTS   # 50
SEED     = exp112.SEED        # 42

PREREG_COMMIT = "110832c"

# Δ-window structural heads (from exp-137; amplification γ = +2.0)
STRUCTURAL  = [(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)]
GAMMA_AMP   = 2.0

# Steep/local heads (from exp-137; suppression γ = −1.0)
STEEP_LOCAL = [(0, 10), (10, 5), (8, 7), (7, 0), (7, 9)]
GAMMA_SUP   = -1.0

# Kill thresholds
K3_AMP_THRESHOLD = 0.5   # κ̃_K(amp) < 0.5 on ≥ 2/5 STRUCTURAL heads → amplification failed
K3_SUP_THRESHOLD = 1.0   # κ̃_K(sup) > 1.0 on ≥ 2/5 STEEP_LOCAL heads → suppression failed

SHAM_SEED_BASE_AMP = 2026091501   # distinct from exp-141 (2026091201) and exp-142 (2026091301)
SHAM_SEED_BASE_SUP = 2026091510

D_MODEL = 768
D_HEAD  = 64

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ── Positional field (ln_1 output hook) ──────────────────────────────────────

def compute_positional_field_ln1(model: GPT2LMHeadModel,
                                  layer: int,
                                  rng: np.random.Generator) -> np.ndarray:
    """
    Positional field δ at the OUTPUT of ln_1 for attention block `layer`.
    Corrected hook (exp-141): hooks model.transformer.h[layer].ln_1 output.
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
    d = D_MODEL
    reads      = delta @ W_K
    cap        = float((np.linalg.norm(reads, axis=1) ** 2).sum() /
                       (np.linalg.norm(delta, axis=1) ** 2).sum())
    norm_factor = float((W_K ** 2).sum()) / d
    return cap / norm_factor if norm_factor > 1e-14 else 0.0


def get_wk(model: GPT2LMHeadModel, layer: int, head: int) -> np.ndarray:
    """W_K for (layer, head) from GPT-2 c_attn weight. Returns (D_MODEL, D_HEAD)."""
    W = model.transformer.h[layer].attn.c_attn.weight.detach().cpu().double().numpy()
    offset = D_MODEL + head * D_HEAD
    return W[:, offset: offset + D_HEAD]


def set_wk(model: GPT2LMHeadModel, layer: int, head: int, W_K_new: np.ndarray):
    """Write W_K back for (layer, head)."""
    W = model.transformer.h[layer].attn.c_attn.weight.data
    with torch.no_grad():
        W[:, D_MODEL + head * D_HEAD: D_MODEL + (head + 1) * D_HEAD] = torch.tensor(
            W_K_new, dtype=W.dtype, device=W.device
        )


# ── Build combined and sham models ───────────────────────────────────────────

def build_combined_model(base_model: GPT2LMHeadModel,
                          rng_census: np.random.Generator):
    """
    Combined model: amplify STRUCTURAL heads (γ=+2.0) + suppress STEEP_LOCAL heads (γ=−1.0).
    Sham model: matched-norm sham in ⊥ complement for both populations.

    Returns (combined_model, sham_model, kappa_report_amp, kappa_report_sup).
    """
    combined_model = copy.deepcopy(base_model)
    sham_model     = copy.deepcopy(base_model)
    kappa_amp = {}
    kappa_sup = {}

    # ── Step 1: Amplify STRUCTURAL (Δ-window) heads ──────────────────────────
    print("\n[A] Amplifying Δ-window heads (γ=+2.0):", flush=True)
    for i, (ell, h) in enumerate(STRUCTURAL):
        delta = compute_positional_field_ln1(base_model, ell, rng_census)

        _, _, Vt  = np.linalg.svd(delta, full_matrices=False)
        P_k       = Vt[:4]
        W_K       = get_wk(base_model, ell, h)
        W_K_proj  = P_k.T @ (P_k @ W_K)

        kappa_before = compute_kappa(W_K, delta)

        # Amplification
        W_K_amp      = W_K + GAMMA_AMP * W_K_proj
        kappa_after_amp = compute_kappa(W_K_amp, delta)
        set_wk(combined_model, ell, h, W_K_amp)

        # Sham: matched Frobenius norm in ⊥ complement
        sham_rng  = np.random.default_rng(SHAM_SEED_BASE_AMP + i)
        rand_vecs = sham_rng.standard_normal((4, D_MODEL))
        for pk_row in P_k:
            rand_vecs -= (rand_vecs @ pk_row)[:, None] * pk_row
        P_perp, _ = np.linalg.qr(rand_vecs.T)
        P_perp    = P_perp.T
        W_K_perp_proj   = P_perp.T @ (P_perp @ W_K)
        delta_amp_norm  = np.linalg.norm(GAMMA_AMP * W_K_proj, "fro")
        delta_perp_norm = np.linalg.norm(W_K_perp_proj, "fro")
        gamma_sham = delta_amp_norm / delta_perp_norm if delta_perp_norm > 1e-14 else 0.0
        W_K_sham_amp     = W_K + gamma_sham * W_K_perp_proj
        kappa_after_sham = compute_kappa(W_K_sham_amp, delta)
        set_wk(sham_model, ell, h, W_K_sham_amp)

        kappa_amp[f"L{ell}H{h}"] = {
            "before":      float(kappa_before),
            "after_amp":   float(kappa_after_amp),
            "after_sham":  float(kappa_after_sham),
            "gamma_amp":   float(GAMMA_AMP),
            "gamma_sham":  float(gamma_sham),
        }
        print(f"  L{ell}H{h}: κ̃ {kappa_before:.3f} → amp {kappa_after_amp:.3f} "
              f"(sham {kappa_after_sham:.3f})", flush=True)

    # ── Step 2: Suppress STEEP_LOCAL heads ───────────────────────────────────
    print("\n[B] Suppressing steep/local heads (γ=−1.0):", flush=True)
    for i, (ell, h) in enumerate(STEEP_LOCAL):
        delta = compute_positional_field_ln1(base_model, ell, rng_census)

        _, _, Vt  = np.linalg.svd(delta, full_matrices=False)
        P_k       = Vt[:4]
        W_K       = get_wk(base_model, ell, h)
        W_K_proj  = P_k.T @ (P_k @ W_K)

        kappa_before = compute_kappa(W_K, delta)

        # Suppression: W_K_sup = W_K − W_K_proj
        W_K_sup        = W_K + GAMMA_SUP * W_K_proj
        kappa_after_sup = compute_kappa(W_K_sup, delta)
        set_wk(combined_model, ell, h, W_K_sup)

        # Sham: matched Frobenius norm in ⊥ complement
        sham_rng  = np.random.default_rng(SHAM_SEED_BASE_SUP + i)
        rand_vecs = sham_rng.standard_normal((4, D_MODEL))
        for pk_row in P_k:
            rand_vecs -= (rand_vecs @ pk_row)[:, None] * pk_row
        P_perp, _ = np.linalg.qr(rand_vecs.T)
        P_perp    = P_perp.T
        W_K_perp_proj    = P_perp.T @ (P_perp @ W_K)
        delta_sup_norm   = np.linalg.norm(GAMMA_SUP * W_K_proj, "fro")  # = ||W_K_proj||_F
        delta_perp_norm  = np.linalg.norm(W_K_perp_proj, "fro")
        gamma_sham = delta_sup_norm / delta_perp_norm if delta_perp_norm > 1e-14 else 0.0
        W_K_sham_sup      = W_K + gamma_sham * W_K_perp_proj
        kappa_after_sham  = compute_kappa(W_K_sham_sup, delta)
        set_wk(sham_model, ell, h, W_K_sham_sup)

        kappa_sup[f"L{ell}H{h}"] = {
            "before":      float(kappa_before),
            "after_sup":   float(kappa_after_sup),
            "after_sham":  float(kappa_after_sham),
            "gamma_sup":   float(GAMMA_SUP),
            "gamma_sham":  float(gamma_sham),
        }
        print(f"  L{ell}H{h}: κ̃ {kappa_before:.3f} → sup {kappa_after_sup:.4f} "
              f"(sham {kappa_after_sham:.3f})", flush=True)

    return combined_model, sham_model, kappa_amp, kappa_sup


# ── Task items (identical to exp-141/142) ─────────────────────────────────────

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


# ── Task scoring ──────────────────────────────────────────────────────────────

def score_task(model, tokenizer, items, task_name, device=DEVICE):
    model.eval()
    results = []
    for idx, (prompt, answer) in enumerate(items):
        inputs     = tokenizer(prompt, return_tensors="pt").to(device)
        correct_tok = tokenizer.encode(" " + answer, add_special_tokens=False)
        if not correct_tok:
            correct_tok = tokenizer.encode(answer, add_special_tokens=False)
        correct_id = correct_tok[0]

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


# ── Kill checks ───────────────────────────────────────────────────────────────

def check_kills(kappa_amp, kappa_sup, orig_A, orig_B, combined_A, combined_B, sham_A, sham_B):
    kills = {}

    # K3_amp: κ̃_K(amp) < 0.5 on ≥ 2/5 STRUCTURAL heads
    amp_kappas = [v["after_amp"] for v in kappa_amp.values()]
    n_amp_fail = sum(1 for k in amp_kappas if k < K3_AMP_THRESHOLD)
    kills["K3_amp"] = {"fired": n_amp_fail >= 2,
                       "n_below_threshold": n_amp_fail,
                       "threshold": K3_AMP_THRESHOLD,
                       "values": amp_kappas}

    # K3_sup: κ̃_K(sup) > 1.0 on ≥ 2/5 STEEP_LOCAL heads
    sup_kappas = [v["after_sup"] for v in kappa_sup.values()]
    n_sup_fail = sum(1 for k in sup_kappas if k > K3_SUP_THRESHOLD)
    kills["K3_sup"] = {"fired": n_sup_fail >= 2,
                       "n_above_threshold": n_sup_fail,
                       "threshold": K3_SUP_THRESHOLD,
                       "values": sup_kappas}

    med_orig_A  = median_lp(orig_A)
    med_orig_B  = median_lp(orig_B)
    med_comb_A  = median_lp(combined_A)
    med_comb_B  = median_lp(combined_B)
    med_sham_A  = median_lp(sham_A)
    med_sham_B  = median_lp(sham_B)

    dA_comb  = med_comb_A - med_orig_A
    dB_comb  = med_comb_B - med_orig_B
    dA_sham  = med_sham_A - med_orig_A
    dB_sham  = med_sham_B - med_orig_B

    # K2: baseline at floor
    kills["K2"] = {"fired": med_orig_A < -10.0 or med_orig_B < -10.0,
                   "orig_A": med_orig_A, "orig_B": med_orig_B}

    # K1: sham effect ≥ combined effect on both tasks
    k1 = (abs(dA_sham) >= abs(dA_comb)) and (abs(dB_sham) >= abs(dB_comb))
    kills["K1"] = {"fired": k1,
                   "abs_sham_A": abs(dA_sham), "abs_comb_A": abs(dA_comb),
                   "abs_sham_B": abs(dB_sham), "abs_comb_B": abs(dB_comb)}

    medians = {
        "orig_A": med_orig_A, "orig_B": med_orig_B,
        "comb_A": med_comb_A, "comb_B": med_comb_B,
        "sham_A": med_sham_A, "sham_B": med_sham_B,
    }
    deltas = {"comb_A": dA_comb, "comb_B": dB_comb,
              "sham_A": dA_sham, "sham_B": dB_sham}

    # Predictions
    # P1: combined Task B improvement > 0.33 nats
    p1 = dB_comb > 0.33
    # P2: Task A within 0.5 nats of sham
    p2 = abs(dA_comb - dA_sham) < 0.5
    # P3: additivity — Task B ≥ 0.55 nats
    p3 = dB_comb >= 0.55
    # P4 (null): both within 0.5 nats of sham
    p4 = (abs(dA_comb - dA_sham) < 0.5) and (abs(dB_comb - dB_sham) < 0.5)

    any_kill = (kills["K1"]["fired"] or kills["K2"]["fired"] or
                kills["K3_amp"]["fired"] or kills["K3_sup"]["fired"])

    if any_kill:
        overall = "INCONCLUSIVE"
    elif p1 and p2 and p3:
        overall = "CONFIRMED (additive)"
    elif p1 and p2:
        overall = "CONFIRMED (combined > single)"
    elif p4:
        overall = "NULL"
    else:
        overall = "INCONCLUSIVE"

    verdicts = {
        "P1": {"fired": p1, "dB_comb": dB_comb, "threshold": 0.33},
        "P2": {"fired": p2, "dA_comb": dA_comb, "dA_sham": dA_sham,
               "diff": dA_comb - dA_sham},
        "P3": {"fired": p3, "dB_comb": dB_comb, "threshold": 0.55},
        "P4": {"fired": p4},
        "overall": overall,
    }

    # Context: how does combined compare to single-manipulation baselines?
    comparison = {
        "exp141_dB": 0.27,
        "exp142_dB": 0.33,
        "exp143_dB": dB_comb,
        "exceeds_max_single": dB_comb > 0.33,
        "approximate_additivity_expected": 0.60,
    }

    return kills, verdicts, medians, deltas, comparison


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("exp-143 — World-model battery combined manipulation", flush=True)
    print(f"  prereg:  {PREREG_COMMIT} (attention-geometry, pushed before this script)", flush=True)
    print(f"  device:  {DEVICE}", flush=True)
    print(f"  STRUCTURAL (amplify γ=+{GAMMA_AMP}): {['L{}H{}'.format(l,h) for l,h in STRUCTURAL]}", flush=True)
    print(f"  STEEP_LOCAL (suppress γ={GAMMA_SUP}): {['L{}H{}'.format(l,h) for l,h in STEEP_LOCAL]}", flush=True)

    print("\n[1] Loading GPT-2 small...", flush=True)
    tokenizer  = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    base_model = GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2", torch_dtype=torch.float32
    ).to(DEVICE)

    print("\n[2] Building combined and sham models...", flush=True)
    rng_census = np.random.default_rng(SEED)
    combined_model, sham_model, kappa_amp, kappa_sup = build_combined_model(
        base_model, rng_census
    )

    # Kill gates
    amp_kappas_after = [v["after_amp"] for v in kappa_amp.values()]
    sup_kappas_after = [v["after_sup"] for v in kappa_sup.values()]
    n_amp_fail = sum(1 for k in amp_kappas_after if k < K3_AMP_THRESHOLD)
    n_sup_fail = sum(1 for k in sup_kappas_after if k > K3_SUP_THRESHOLD)
    print(f"\n  K3_amp check: {n_amp_fail}/5 STRUCTURAL heads with κ̃_K < {K3_AMP_THRESHOLD}", flush=True)
    print(f"  K3_sup check: {n_sup_fail}/5 STEEP_LOCAL heads with κ̃_K > {K3_SUP_THRESHOLD}", flush=True)
    if n_amp_fail >= 2:
        print("  WARNING: K3_amp fires — amplification may be insufficient.", flush=True)
    if n_sup_fail >= 2:
        print("  WARNING: K3_sup fires — suppression may be insufficient.", flush=True)

    print("\n[3] Task A — entity-state tracking (original):", flush=True)
    orig_A     = score_task(base_model,    tokenizer, TASK_A, "A")
    print("\n[4] Task A — entity-state tracking (combined):", flush=True)
    combined_A = score_task(combined_model, tokenizer, TASK_A, "A")
    print("\n[5] Task A — entity-state tracking (sham):", flush=True)
    sham_A     = score_task(sham_model,    tokenizer, TASK_A, "A")

    print("\n[6] Task B — positional retrieval (original):", flush=True)
    orig_B     = score_task(base_model,    tokenizer, TASK_B, "B")
    print("\n[7] Task B — positional retrieval (combined):", flush=True)
    combined_B = score_task(combined_model, tokenizer, TASK_B, "B")
    print("\n[8] Task B — positional retrieval (sham):", flush=True)
    sham_B     = score_task(sham_model,    tokenizer, TASK_B, "B")

    print("\n[9] Evaluating kills and verdicts...", flush=True)
    kills, verdicts, medians, deltas, comparison = check_kills(
        kappa_amp, kappa_sup, orig_A, orig_B, combined_A, combined_B, sham_A, sham_B
    )

    print(f"\n  Overall verdict: {verdicts['overall']}", flush=True)
    m = medians
    d = deltas
    print(f"  Task A: orig={m['orig_A']:.2f}  combined={m['comb_A']:.2f}  sham={m['sham_A']:.2f}  "
          f"(Δcomb={d['comb_A']:+.2f}  Δsham={d['sham_A']:+.2f})", flush=True)
    print(f"  Task B: orig={m['orig_B']:.2f}  combined={m['comb_B']:.2f}  sham={m['sham_B']:.2f}  "
          f"(Δcomb={d['comb_B']:+.2f}  Δsham={d['sham_B']:+.2f})", flush=True)

    n_A_improved_comb = sum(1 for o, c in zip(orig_A, combined_A)
                            if c["log_prob_correct"] > o["log_prob_correct"])
    n_B_improved_comb = sum(1 for o, c in zip(orig_B, combined_B)
                            if c["log_prob_correct"] > o["log_prob_correct"])
    n_A_improved_sham = sum(1 for o, s in zip(orig_A, sham_A)
                            if s["log_prob_correct"] > o["log_prob_correct"])
    n_B_improved_sham = sum(1 for o, s in zip(orig_B, sham_B)
                            if s["log_prob_correct"] > o["log_prob_correct"])
    print(f"\n  Item-level improvement (combined vs orig):", flush=True)
    print(f"    Task A improved: {n_A_improved_comb}/20", flush=True)
    print(f"    Task B improved: {n_B_improved_comb}/20", flush=True)
    print(f"  Sham item improvement (sham vs orig): A={n_A_improved_sham}/20  B={n_B_improved_sham}/20", flush=True)

    print(f"\n  Comparison to single manipulations:", flush=True)
    print(f"    exp-141 ΔP_B: +0.27 nats (amplify only)", flush=True)
    print(f"    exp-142 ΔP_B: +0.33 nats (suppress only)", flush=True)
    print(f"    exp-143 ΔP_B: {d['comb_B']:+.2f} nats (combined)", flush=True)
    print(f"    Exceeds max single: {comparison['exceeds_max_single']}", flush=True)

    # P1/P2/P3/P4
    for pk, pv in verdicts.items():
        if pk != "overall":
            print(f"  {pk}: {'FIRES' if pv.get('fired') else 'not fired'}", flush=True)

    out = {
        "exp":           "exp-143",
        "prereg_commit": PREREG_COMMIT,
        "prereg_evidence": "git-attested — commit 110832c pushed before run.py written",
        "device":        DEVICE,
        "gamma_amp":     GAMMA_AMP,
        "gamma_sup":     GAMMA_SUP,
        "structural_heads":  [f"L{l}H{h}" for l, h in STRUCTURAL],
        "steep_local_heads": [f"L{l}H{h}" for l, h in STEEP_LOCAL],
        "kappa_amp":     kappa_amp,
        "kappa_sup":     kappa_sup,
        "kills":         kills,
        "verdicts":      verdicts,
        "medians":       medians,
        "deltas":        deltas,
        "comparison_to_singles": comparison,
        "item_counts": {
            "n_A_improved_combined": n_A_improved_comb,
            "n_B_improved_combined": n_B_improved_comb,
            "n_A_improved_sham":     n_A_improved_sham,
            "n_B_improved_sham":     n_B_improved_sham,
        },
        "task_A_orig":     orig_A,
        "task_A_combined": combined_A,
        "task_A_sham":     sham_A,
        "task_B_orig":     orig_B,
        "task_B_combined": combined_B,
        "task_B_sham":     sham_B,
    }

    out_path = HERE / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
