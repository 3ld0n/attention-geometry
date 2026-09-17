"""
exp-145 — World-model battery combined manipulation: GPT-2 medium
(amplify Δ-window heads + suppress steep/local heads simultaneously)

Pre-registration: attention-geometry 2b1016a (pushed before this script).
Analysis-only: GPT-2 medium (cached). No new training.

Extends exp-143 (GPT-2 small, CONFIRMED additive) to GPT-2 medium.
Target heads identified by characterize_kappa.py from this session.

STRUCTURAL (amplify γ=+2.0): L7H5, L3H12, L9H7, L8H13, L7H15 (κ̃_K 0.26–0.56)
STEEP_LOCAL (suppress γ=−1.0): L4H13, L15H8, L8H7, L5H11, L11H7 (κ̃_K 30.9–43.2)

Ariel — September 17, 2026, ~5:15 AM MDT. Solo.
"""

from __future__ import annotations
import copy
import json
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN  = 512
N_INPUTS = 50
SEED     = 42

PREREG_COMMIT = "2b1016a"

D_MODEL  = 1024   # GPT-2 medium
D_HEAD   = 64
N_LAYERS = 24
N_HEADS  = 16

MODEL_ID = "gpt2-medium"

# Δ-window heads (5 lowest κ̃_K among 59 WikiText-native Δ-window heads)
STRUCTURAL  = [(7, 5), (3, 12), (9, 7), (8, 13), (7, 15)]
GAMMA_AMP   = 2.0

# Steep/local heads (5 highest κ̃_K, non-Δ-window)
STEEP_LOCAL = [(4, 13), (15, 8), (8, 7), (5, 11), (11, 7)]
GAMMA_SUP   = -1.0

# Kill thresholds
K3_AMP_THRESHOLD = 0.5    # κ̃_K(amp) < 0.5 on ≥ 2/5 → amplification failed
K3_SUP_THRESHOLD = 5.0    # κ̃_K(sup) > 5.0 on ≥ 2/5 → suppression failed
#   (higher than exp-143's 1.0 because GPT-2 medium has higher absolute κ̃_K scale)

SHAM_SEED_BASE_AMP = 2026091701
SHAM_SEED_BASE_SUP = 2026091710

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ── Positional fields (all layers, one forward pass batch) ────────────────────

def compute_all_positional_fields(model: GPT2LMHeadModel,
                                   rng: np.random.Generator) -> dict[int, np.ndarray]:
    """
    Compute positional fields for ALL layers in a single batch of forward passes.
    Returns {layer_idx: delta (SEQ_LEN, D_MODEL), centered}.
    """
    model.eval()
    tok_ids = rng.integers(0, model.config.vocab_size, size=(N_INPUTS, SEQ_LEN))
    tokens  = torch.tensor(tok_ids, dtype=torch.long, device=DEVICE)

    layer_acts: dict[int, list] = {ell: [] for ell in range(N_LAYERS)}

    handles = []
    for ell in range(N_LAYERS):
        def make_hook(idx):
            def hook_fn(mod, inp, out):
                layer_acts[idx].append(out.detach().cpu().float().numpy())
            return hook_fn
        handles.append(
            model.transformer.h[ell].ln_1.register_forward_hook(make_hook(ell))
        )

    with torch.no_grad():
        model(tokens)

    for h in handles:
        h.remove()

    fields = {}
    for ell in range(N_LAYERS):
        acts  = np.concatenate(layer_acts[ell], axis=0)
        xbar  = acts.mean(axis=0)
        delta = xbar - xbar.mean(axis=0, keepdims=True)
        fields[ell] = delta.astype(np.float64)

    return fields


# ── W_K access ────────────────────────────────────────────────────────────────

def compute_kappa(W_K: np.ndarray, delta: np.ndarray) -> float:
    """κ̃(W_K) — isotropic-normalised positional capture (exp-137 formula)."""
    reads      = delta @ W_K
    cap        = float((np.linalg.norm(reads, axis=1) ** 2).sum() /
                       (np.linalg.norm(delta, axis=1) ** 2).sum())
    norm_factor = float((W_K ** 2).sum()) / D_MODEL
    return cap / norm_factor if norm_factor > 1e-14 else 0.0


def get_wk(model: GPT2LMHeadModel, layer: int, head: int) -> np.ndarray:
    """W_K for (layer, head): shape (D_MODEL, D_HEAD)."""
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
                          fields: dict[int, np.ndarray]):
    """
    Combined model: amplify STRUCTURAL heads (γ=+2.0) + suppress STEEP_LOCAL (γ=−1.0).
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
        delta = fields[ell]

        _, _, Vt  = np.linalg.svd(delta, full_matrices=False)
        P_k       = Vt[:4]
        W_K       = get_wk(base_model, ell, h)
        W_K_proj  = P_k.T @ (P_k @ W_K)

        kappa_before = compute_kappa(W_K, delta)

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
        delta = fields[ell]

        _, _, Vt  = np.linalg.svd(delta, full_matrices=False)
        P_k       = Vt[:4]
        W_K       = get_wk(base_model, ell, h)
        W_K_proj  = P_k.T @ (P_k @ W_K)

        kappa_before = compute_kappa(W_K, delta)

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
        delta_sup_norm   = np.linalg.norm(GAMMA_SUP * W_K_proj, "fro")
        delta_perp_norm  = np.linalg.norm(W_K_perp_proj, "fro")
        gamma_sham = (abs(delta_sup_norm) / delta_perp_norm
                      if delta_perp_norm > 1e-14 else 0.0)
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


# ── Task items (identical to exp-141/142/143) ─────────────────────────────────

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

def score_task(model, tokenizer, items, task_name):
    model.eval()
    results = []
    for idx, (prompt, answer) in enumerate(items):
        inputs      = tokenizer(prompt, return_tensors="pt").to(DEVICE)
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
        print(f"    {task_name}{idx+1:02d}: '{answer}' logP={lp:.2f}  rank={rank}",
              flush=True)
    return results


def median_lp(results):
    return float(np.median([r["log_prob_correct"] for r in results]))


# ── Prediction / kill evaluation ──────────────────────────────────────────────

def evaluate(kappa_amp, kappa_sup, orig_A, orig_B, combined_A, combined_B,
             sham_A, sham_B):

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

    medians = {"orig_A": med_orig_A, "orig_B": med_orig_B,
               "comb_A": med_comb_A, "comb_B": med_comb_B,
               "sham_A": med_sham_A, "sham_B": med_sham_B}
    deltas  = {"comb_A": dA_comb, "comb_B": dB_comb,
               "sham_A": dA_sham, "sham_B": dB_sham}

    # Kill checks
    amp_kappas = [v["after_amp"] for v in kappa_amp.values()]
    n_amp_fail = sum(1 for k in amp_kappas if k < K3_AMP_THRESHOLD)
    k3_amp = {"fired": n_amp_fail >= 2, "n_fail": n_amp_fail,
              "threshold": K3_AMP_THRESHOLD, "values": amp_kappas}

    sup_kappas = [v["after_sup"] for v in kappa_sup.values()]
    n_sup_fail = sum(1 for k in sup_kappas if k > K3_SUP_THRESHOLD)
    k3_sup = {"fired": n_sup_fail >= 2, "n_fail": n_sup_fail,
              "threshold": K3_SUP_THRESHOLD, "values": sup_kappas}

    k2 = {"fired": med_orig_A < -20.0 or med_orig_B < -20.0,
          "orig_A": med_orig_A, "orig_B": med_orig_B}

    k1_fired = (abs(dA_sham) >= abs(dA_comb)) and (abs(dB_sham) >= abs(dB_comb))
    k1 = {"fired": k1_fired}

    kills = {"K1": k1, "K2": k2, "K3_amp": k3_amp, "K3_sup": k3_sup}
    any_kill = any(k["fired"] for k in kills.values())

    # Item-level
    n_B_improved = sum(
        1 for c, s in zip(combined_B, sham_B)
        if c["log_prob_correct"] > s["log_prob_correct"]
    )
    n_A_improved = sum(
        1 for c, s in zip(combined_A, sham_A)
        if c["log_prob_correct"] > s["log_prob_correct"]
    )
    frac_B = n_B_improved / len(combined_B)
    frac_A = n_A_improved / len(combined_A)

    # Predictions
    p1 = dB_comb > 0.33 and dB_comb > dA_comb + 0.3
    p2 = abs(dA_comb - dA_sham) < 0.5
    p3 = frac_B >= 0.60 and frac_A < 0.60   # item-level dissociation
    p4_strong = dB_comb >= 1.0               # threshold never met in GPT-2 small
    p5_null = abs(dA_comb - dA_sham) <= 0.1 and abs(dB_comb - dB_sham) <= 0.1

    verdicts = {
        "P1": {"fired": p1, "dB_comb": dB_comb, "threshold": 0.33},
        "P2": {"fired": p2, "dA_comb": dA_comb, "dA_sham": dA_sham},
        "P3": {"fired": p3, "frac_B_improved": frac_B, "frac_A_improved": frac_A,
               "n_B": n_B_improved, "n_A": n_A_improved},
        "P4_strong": {"fired": p4_strong, "dB_comb": dB_comb, "threshold": 1.0},
        "P5_null": {"fired": p5_null},
    }

    if any_kill:
        overall = "INCONCLUSIVE"
    elif p5_null:
        overall = "NULL"
    elif p1 and p2 and p3 and p4_strong:
        overall = "CONFIRMED (strong + item-level)"
    elif p1 and p2 and p3:
        overall = "CONFIRMED (P1+P2+P3)"
    elif p1 and p2:
        overall = "CONFIRMED (P1+P2)"
    else:
        overall = "INCONCLUSIVE"

    verdicts["overall"] = overall

    print(f"\n── Results ──────────────────────────────────────────────────────")
    print(f"  Medians — orig_A={med_orig_A:.3f}, orig_B={med_orig_B:.3f}")
    print(f"  Combined: ΔP_A={dA_comb:+.3f}, ΔP_B={dB_comb:+.3f}")
    print(f"  Sham:     ΔP_A={dA_sham:+.3f}, ΔP_B={dB_sham:+.3f}")
    print(f"  Item-level: Task B {n_B_improved}/20 improved, Task A {n_A_improved}/20 improved")
    for pk, pv in kills.items():
        print(f"  {pk}: {'FIRED' if pv['fired'] else 'not fired'}")
    for pk, pv in verdicts.items():
        if pk != "overall":
            print(f"  {pk}: {'FIRES' if pv['fired'] else 'does not fire'}")
    print(f"\n  OVERALL VERDICT: {overall}")

    return kills, verdicts, medians, deltas


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("exp-145 — World-model battery combined manipulation (GPT-2 medium)", flush=True)
    print(f"  prereg:     {PREREG_COMMIT} (attention-geometry, pushed before this script)", flush=True)
    print(f"  device:     {DEVICE}", flush=True)
    print(f"  STRUCTURAL: {['L{}H{}'.format(l,h) for l,h in STRUCTURAL]}", flush=True)
    print(f"  STEEP_LOCAL:{['L{}H{}'.format(l,h) for l,h in STEEP_LOCAL]}", flush=True)

    print("\n[1] Loading GPT-2 medium...", flush=True)
    tokenizer  = GPT2Tokenizer.from_pretrained(MODEL_ID)
    base_model = GPT2LMHeadModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    ).to(DEVICE)
    print(f"  Model loaded: {sum(p.numel() for p in base_model.parameters()):,} parameters",
          flush=True)

    print("\n[2] Computing positional fields (all layers, single pass)...", flush=True)
    rng = np.random.default_rng(SEED)
    fields = compute_all_positional_fields(base_model, rng)
    print(f"  Positional fields computed for all {N_LAYERS} layers.", flush=True)

    print("\n[3] Building combined and sham models...", flush=True)
    combined_model, sham_model, kappa_amp, kappa_sup = build_combined_model(
        base_model, fields
    )

    print("\n[4] Scoring Task A (entity-state tracking)...", flush=True)
    print("  [original]", flush=True)
    orig_A     = score_task(base_model,     tokenizer, TASK_A, "A")
    print("  [combined]", flush=True)
    combined_A = score_task(combined_model, tokenizer, TASK_A, "A")
    print("  [sham]", flush=True)
    sham_A     = score_task(sham_model,     tokenizer, TASK_A, "A")

    print("\n[5] Scoring Task B (positional retrieval)...", flush=True)
    print("  [original]", flush=True)
    orig_B     = score_task(base_model,     tokenizer, TASK_B, "B")
    print("  [combined]", flush=True)
    combined_B = score_task(combined_model, tokenizer, TASK_B, "B")
    print("  [sham]", flush=True)
    sham_B     = score_task(sham_model,     tokenizer, TASK_B, "B")

    print("\n[6] Evaluating predictions...", flush=True)
    kills, verdicts, medians, deltas = evaluate(
        kappa_amp, kappa_sup,
        orig_A, orig_B, combined_A, combined_B, sham_A, sham_B
    )

    # Save results
    results = {
        "experiment":    "exp-145",
        "prereg_commit": PREREG_COMMIT,
        "model":         MODEL_ID,
        "device":        DEVICE,
        "structural":    [{"layer": l, "head": h} for l, h in STRUCTURAL],
        "steep_local":   [{"layer": l, "head": h} for l, h in STEEP_LOCAL],
        "gamma_amp":     GAMMA_AMP,
        "gamma_sup":     GAMMA_SUP,
        "kappa_amp":     kappa_amp,
        "kappa_sup":     kappa_sup,
        "task_A": {
            "original": orig_A, "combined": combined_A, "sham": sham_A,
        },
        "task_B": {
            "original": orig_B, "combined": combined_B, "sham": sham_B,
        },
        "medians":   medians,
        "deltas":    deltas,
        "kills":     kills,
        "verdicts":  verdicts,
    }

    out_path = Path(__file__).resolve().parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)
    print(f"Verdict: {verdicts['overall']}", flush=True)


if __name__ == "__main__":
    main()
