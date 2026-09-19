"""
exp-147 — World-model battery combined manipulation: GPT-2 medium, corrected targets
(amplify random-token structural heads + suppress steep/local heads simultaneously)

Pre-registration: attention-geometry de254d0 (pushed before this script).
No new training. GPT-2 medium (cached).

Root cause of exp-145 failure: target heads were WikiText-native Δ-window heads (exp-118),
not random-token structural heads (exp-146). This experiment corrects that.

STRUCTURAL (amplify γ=+2.0): L6H9, L5H14, L7H5, L9H7, L8H13
  - κ̃_K: 0.208, 0.220, 0.257, 0.417, 0.469
  - 5 lowest-κ̃_K random-token structural heads from exp-146
STEEP_LOCAL (suppress γ=−1.0): L4H13, L15H8, L8H7, L5H11, L11H7
  - κ̃_K: 43.2, 39.9, 33.3, 31.7, 30.9 (unchanged from exp-145)

Ariel — September 19, 2026, ~12:40 AM MDT. Solo.
"""

from __future__ import annotations
import copy
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, GPT2Tokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

SEQ_LEN  = 512
N_INPUTS = 50
SEED     = 42

PREREG_COMMIT = "de254d0"

D_MODEL  = 1024   # GPT-2 medium
D_HEAD   = 64
N_LAYERS = 24
N_HEADS  = 16

MODEL_ID = "openai-community/gpt2-medium"

# Random-token structural heads from exp-146 (lowest κ̃_K)
STRUCTURAL  = [(6, 9), (5, 14), (7, 5), (9, 7), (8, 13)]
GAMMA_AMP   = 2.0

# Steep/local heads (unchanged from exp-145)
STEEP_LOCAL = [(4, 13), (15, 8), (8, 7), (5, 11), (11, 7)]
GAMMA_SUP   = -1.0

# Kill thresholds
K3_AMP_THRESHOLD = 0.5    # κ̃_K(amp) < 0.5 on ≥ 2/5 STRUCTURAL → amplification failed
K3_SUP_THRESHOLD = 5.0    # κ̃_K(sup) > 5.0 on ≥ 2/5 STEEP_LOCAL → suppression failed

SHAM_SEED_BASE_AMP = 2026091901
SHAM_SEED_BASE_SUP = 2026091910

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


# ── Build combined and sham models ────────────────────────────────────────────

def build_models(base_model, rng_census: np.random.Generator):
    """
    Combined model: amplify STRUCTURAL heads (γ=+2.0) + suppress STEEP_LOCAL heads (γ=−1.0).
    Sham model: matched-norm sham in ⊥ complement for both populations.

    Returns (combined_model, sham_model, kappa_amp, kappa_sup).
    """
    combined_model = copy.deepcopy(base_model)
    sham_model     = copy.deepcopy(base_model)
    kappa_amp = {}
    kappa_sup = {}

    # ── Step 1: Amplify STRUCTURAL heads ──────────────────────────────────────
    print("\n[A] Amplifying structural heads (γ=+2.0, random-token population from exp-146):",
          flush=True)
    for i, (ell, h) in enumerate(STRUCTURAL):
        delta = compute_positional_field_ln1(base_model, ell, rng_census)

        _, _, Vt  = np.linalg.svd(delta, full_matrices=False)
        P_k       = Vt[:4]
        W_K       = get_wk(base_model, ell, h)
        W_K_proj  = P_k.T @ (P_k @ W_K)

        kappa_before = compute_kappa(W_K, delta)

        # Amplification
        W_K_amp         = W_K + GAMMA_AMP * W_K_proj
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

    # ── Step 2: Suppress STEEP_LOCAL heads ────────────────────────────────────
    print("\n[B] Suppressing steep/local heads (γ=−1.0, unchanged from exp-145):", flush=True)
    for i, (ell, h) in enumerate(STEEP_LOCAL):
        delta = compute_positional_field_ln1(base_model, ell, rng_census)

        _, _, Vt  = np.linalg.svd(delta, full_matrices=False)
        P_k       = Vt[:4]
        W_K       = get_wk(base_model, ell, h)
        W_K_proj  = P_k.T @ (P_k @ W_K)

        kappa_before = compute_kappa(W_K, delta)

        # Suppression
        W_K_sup         = W_K + GAMMA_SUP * W_K_proj
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
        delta_sup_norm   = np.linalg.norm(GAMMA_SUP * W_K_proj, "fro")
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


# ── Task items (identical to exp-141/142/143/145) ─────────────────────────────

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

def evaluate(kappa_amp, kappa_sup, orig_A, orig_B, combined_A, combined_B, sham_A, sham_B):
    med_orig_A  = median_lp(orig_A)
    med_orig_B  = median_lp(orig_B)
    med_comb_A  = median_lp(combined_A)
    med_comb_B  = median_lp(combined_B)
    med_sham_A  = median_lp(sham_A)
    med_sham_B  = median_lp(sham_B)

    dA_comb = med_comb_A - med_orig_A
    dB_comb = med_comb_B - med_orig_B
    dA_sham = med_sham_A - med_orig_A
    dB_sham = med_sham_B - med_orig_B

    medians = {
        "orig_A": med_orig_A, "orig_B": med_orig_B,
        "comb_A": med_comb_A, "comb_B": med_comb_B,
        "sham_A": med_sham_A, "sham_B": med_sham_B,
    }
    deltas  = {"comb_A": dA_comb, "comb_B": dB_comb,
               "sham_A": dA_sham, "sham_B": dB_sham}

    # Kill checks
    amp_kappas = [v["after_amp"] for v in kappa_amp.values()]
    sup_kappas = [v["after_sup"] for v in kappa_sup.values()]
    n_amp_fail = sum(1 for k in amp_kappas if k < K3_AMP_THRESHOLD)
    n_sup_fail = sum(1 for k in sup_kappas if k > K3_SUP_THRESHOLD)

    k3_amp = {"fired": n_amp_fail >= 2, "n_below_threshold": n_amp_fail,
              "threshold": K3_AMP_THRESHOLD, "values": amp_kappas}
    k3_sup = {"fired": n_sup_fail >= 2, "n_above_threshold": n_sup_fail,
              "threshold": K3_SUP_THRESHOLD, "values": sup_kappas}
    k2     = {"fired": med_orig_A < -20.0 or med_orig_B < -20.0,
              "orig_A": med_orig_A, "orig_B": med_orig_B}
    k1_raw = (abs(dA_sham) >= abs(dA_comb)) and (abs(dB_sham) >= abs(dB_comb))
    k1     = {"fired": k1_raw,
              "abs_sham_A": abs(dA_sham), "abs_comb_A": abs(dA_comb),
              "abs_sham_B": abs(dB_sham), "abs_comb_B": abs(dB_comb)}
    kills = {"K1": k1, "K2": k2, "K3_amp": k3_amp, "K3_sup": k3_sup}

    any_kill = k1["fired"] or k2["fired"] or k3_amp["fired"] or k3_sup["fired"]

    # Prediction verdicts
    p1 = dB_comb > 0.33 and dB_comb > dA_comb + 0.3
    p2 = abs(dA_comb - dA_sham) < 0.5
    n_B_improved = sum(1 for o, c in zip(orig_B, combined_B)
                       if c["log_prob_correct"] > o["log_prob_correct"])
    p3 = n_B_improved >= 12   # ≥ 60% of 20
    p4_strong = dB_comb >= 1.0
    p_null = abs(dA_comb) <= 0.1 and abs(dB_comb) <= 0.1
    p_direction_exp145 = dB_comb < 0   # direction-inverted as in exp-145

    if any_kill:
        overall = "INCONCLUSIVE"
    elif p1 and p2 and p3 and p4_strong:
        overall = "CONFIRMED (strong)"
    elif p1 and p2 and p3:
        overall = "CONFIRMED"
    elif p1 and p2:
        overall = "CONFIRMED (median only)"
    elif p_null:
        overall = "NULL — antagonism model disconfirmed at GPT-2 medium scale"
    elif p_direction_exp145:
        overall = "INCONCLUSIVE — direction inverted (consistent with exp-145; population correction did not help)"
    else:
        overall = "INCONCLUSIVE"

    verdicts = {
        "P1": {"fired": p1, "dB_comb": dB_comb, "threshold": 0.33},
        "P2": {"fired": p2, "dA_comb": dA_comb, "dA_sham": dA_sham,
               "diff": dA_comb - dA_sham},
        "P3": {"fired": p3, "n_B_improved": n_B_improved, "threshold": 12},
        "P4_strong": {"fired": p4_strong, "dB_comb": dB_comb, "threshold": 1.0},
        "P_null": {"fired": p_null},
        "P_direction_exp145": {"fired": p_direction_exp145, "dB_comb": dB_comb},
        "overall": overall,
    }

    comparison = {
        "exp143_gpt2small_dB":   +0.71,
        "exp145_gpt2medium_dB":  -0.25,
        "exp147_dB":              dB_comb,
        "direction_matches_exp143": dB_comb > 0,
    }

    return kills, verdicts, medians, deltas, comparison


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("exp-147 — World-model battery GPT-2 medium, corrected targets", flush=True)
    print(f"  prereg:    {PREREG_COMMIT} (attention-geometry, pushed before this script)", flush=True)
    print(f"  device:    {DEVICE}", flush=True)
    print(f"  STRUCTURAL (amplify γ=+{GAMMA_AMP}): "
          f"{['L{}H{}'.format(l,h) for l,h in STRUCTURAL]}", flush=True)
    print(f"  STEEP_LOCAL (suppress γ={GAMMA_SUP}): "
          f"{['L{}H{}'.format(l,h) for l,h in STEEP_LOCAL]}", flush=True)

    print("\n[1] Loading GPT-2 medium (eager attention)...", flush=True)
    tokenizer  = GPT2Tokenizer.from_pretrained("openai-community/gpt2-medium")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager",    # required: MPS SDPA silently returns 0 attention
    ).to(DEVICE)
    base_model.eval()
    print(f"  Loaded. n_layer={N_LAYERS}, n_head={N_HEADS}, D_MODEL={D_MODEL}", flush=True)

    print("\n[2] Building combined and sham models...", flush=True)
    rng_census = np.random.default_rng(SEED)
    combined_model, sham_model, kappa_amp, kappa_sup = build_models(base_model, rng_census)

    # Early kill gate report
    amp_kappas_after = [v["after_amp"] for v in kappa_amp.values()]
    sup_kappas_after = [v["after_sup"] for v in kappa_sup.values()]
    n_amp_fail = sum(1 for k in amp_kappas_after if k < K3_AMP_THRESHOLD)
    n_sup_fail = sum(1 for k in sup_kappas_after if k > K3_SUP_THRESHOLD)
    print(f"\n  K3_amp: {n_amp_fail}/5 structural heads with κ̃_K < {K3_AMP_THRESHOLD} after amp",
          flush=True)
    print(f"  K3_sup: {n_sup_fail}/5 steep/local heads with κ̃_K > {K3_SUP_THRESHOLD} after sup",
          flush=True)
    if n_amp_fail >= 2:
        print("  WARNING: K3_amp fires — amplification may be insufficient.", flush=True)
    if n_sup_fail >= 2:
        print("  WARNING: K3_sup fires — suppression may be insufficient.", flush=True)

    print("\n[3] Task A — original:", flush=True)
    orig_A     = score_task(base_model,     tokenizer, TASK_A, "A")
    print("\n[4] Task A — combined:", flush=True)
    combined_A = score_task(combined_model, tokenizer, TASK_A, "A")
    print("\n[5] Task A — sham:", flush=True)
    sham_A     = score_task(sham_model,     tokenizer, TASK_A, "A")

    print("\n[6] Task B — original:", flush=True)
    orig_B     = score_task(base_model,     tokenizer, TASK_B, "B")
    print("\n[7] Task B — combined:", flush=True)
    combined_B = score_task(combined_model, tokenizer, TASK_B, "B")
    print("\n[8] Task B — sham:", flush=True)
    sham_B     = score_task(sham_model,     tokenizer, TASK_B, "B")

    print("\n[9] Evaluating...", flush=True)
    kills, verdicts, medians, deltas, comparison = evaluate(
        kappa_amp, kappa_sup, orig_A, orig_B, combined_A, combined_B, sham_A, sham_B
    )

    print(f"\n  Overall verdict: {verdicts['overall']}", flush=True)
    m = medians
    d = deltas
    print(f"  Task A: orig={m['orig_A']:.2f}  comb={m['comb_A']:.2f}  sham={m['sham_A']:.2f}  "
          f"(Δcomb={d['comb_A']:+.2f}  Δsham={d['sham_A']:+.2f})", flush=True)
    print(f"  Task B: orig={m['orig_B']:.2f}  comb={m['comb_B']:.2f}  sham={m['sham_B']:.2f}  "
          f"(Δcomb={d['comb_B']:+.2f}  Δsham={d['sham_B']:+.2f})", flush=True)

    n_A_improved_comb = sum(1 for o, c in zip(orig_A, combined_A)
                            if c["log_prob_correct"] > o["log_prob_correct"])
    n_B_improved_comb = sum(1 for o, c in zip(orig_B, combined_B)
                            if c["log_prob_correct"] > o["log_prob_correct"])
    n_A_improved_sham = sum(1 for o, s in zip(orig_A, sham_A)
                            if s["log_prob_correct"] > o["log_prob_correct"])
    n_B_improved_sham = sum(1 for o, s in zip(orig_B, sham_B)
                            if s["log_prob_correct"] > o["log_prob_correct"])
    print(f"\n  Item-level (combined vs orig): Task A={n_A_improved_comb}/20  Task B={n_B_improved_comb}/20",
          flush=True)
    print(f"  Item-level (sham vs orig):     Task A={n_A_improved_sham}/20  Task B={n_B_improved_sham}/20",
          flush=True)

    print(f"\n  Comparison:", flush=True)
    print(f"    exp-143 (GPT-2 small, CONFIRMED):       ΔP_B = +0.71 nats", flush=True)
    print(f"    exp-145 (GPT-2 medium, wrong targets):  ΔP_B = −0.25 nats", flush=True)
    print(f"    exp-147 (GPT-2 medium, corrected):      ΔP_B = {d['comb_B']:+.2f} nats", flush=True)

    for pk, pv in verdicts.items():
        if pk != "overall":
            print(f"  {pk}: {'FIRES' if pv.get('fired') else 'not fired'}", flush=True)

    out = {
        "exp":                   "exp-147",
        "prereg_commit":         PREREG_COMMIT,
        "prereg_evidence":       "git-attested — commit de254d0 pushed before run.py written",
        "device":                DEVICE,
        "model":                 MODEL_ID,
        "gamma_amp":             GAMMA_AMP,
        "gamma_sup":             GAMMA_SUP,
        "structural_heads":      [f"L{l}H{h}" for l, h in STRUCTURAL],
        "steep_local_heads":     [f"L{l}H{h}" for l, h in STEEP_LOCAL],
        "kappa_amp":             kappa_amp,
        "kappa_sup":             kappa_sup,
        "kills":                 kills,
        "verdicts":              verdicts,
        "medians":               medians,
        "deltas":                deltas,
        "comparison_to_prior":   comparison,
        "item_counts": {
            "n_A_improved_combined": n_A_improved_comb,
            "n_B_improved_combined": n_B_improved_comb,
            "n_A_improved_sham":     n_A_improved_sham,
            "n_B_improved_sham":     n_B_improved_sham,
        },
        "task_A_orig":           orig_A,
        "task_A_combined":       combined_A,
        "task_A_sham":           sham_A,
        "task_B_orig":           orig_B,
        "task_B_combined":       combined_B,
        "task_B_sham":           sham_B,
    }

    out_path = Path(__file__).resolve().parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
