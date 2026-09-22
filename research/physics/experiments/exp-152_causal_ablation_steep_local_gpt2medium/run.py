"""
exp-152 — Causal ablation of steep/local heads: GPT-2 medium (relay confirmation)

Pre-registration: attention-geometry 5a15b35 (pushed 2026-09-22 before this script).
Follows exp-149 (suppress γ=−1.0, ΔP_B = −0.10) and exp-150 (suppress γ=−0.86, ΔP_B = −0.08).
Tests the relay account: are the steep/local heads' positional concentration patterns
load-bearing for Task B, or is their mere presence (in any form) what matters?

Ablation: W_K → 0 for all 5 steep/local heads (L4H13, L15H8, L8H7, L5H11, L11H7).
  Keys become zero → attention scores = 0 for all pairs → softmax = uniform causal.
  Head still writes to residual stream (W_V active), but without positional concentration.

Sham: matched-norm perturbation of W_K in ⊥(P_k) — same norm as W_K, but ⊥ to positional
  field. Preserves κ̃ ≈ κ̃_orig while disrupting the non-positional W_K structure.
  Tests: is the degradation specific to zeroing the positional routing, or to any
  large perturbation of these heads' weights?

H_relay: if P1 fires (ΔP_B < −0.10) and K1 does not fire, relay confirmed.

STEEP_LOCAL: L4H13, L15H8, L8H7, L5H11, L11H7 (κ̃_K ≈ 43.2, 39.9, 33.3, 31.7, 30.9)

Ariel — September 22, 2026, ~8:17 AM MDT. Solo physics room session.
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

PREREG_COMMIT = "5a15b35"

D_MODEL  = 1024   # GPT-2 medium
D_HEAD   = 64
N_LAYERS = 24
N_HEADS  = 16

MODEL_ID = "openai-community/gpt2-medium"

# Steep/local heads (from exp-142/147/149/150)
# κ̃_K: 43.2, 39.9, 33.3, 31.7, 30.9 (exp-150 prereg values)
STEEP_LOCAL = [(4, 13), (15, 8), (8, 7), (5, 11), (11, 7)]

# Kill threshold: ablation is complete if κ̃_after ≤ 0.01 on all 5 heads
K3_ABLATION_THRESHOLD = 0.01

# Sham: κ̃_sham should be ≥ 0.8 × κ̃_before (positional component preserved)
SHAM_KAPPA_FLOOR_FRACTION = 0.8

SHAM_SEED_BASE = 2026092201   # distinct from prior experiments

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


# ── Build ablated and sham models ─────────────────────────────────────────────

def build_models(base_model, rng_census: np.random.Generator):
    """
    Ablated model: W_K = 0 for STEEP_LOCAL heads.
    Sham model: matched-norm perturbation of W_K in ⊥(P_k) for STEEP_LOCAL heads.

    Returns (abl_model, sham_model, kappa_records).
    """
    abl_model  = copy.deepcopy(base_model)
    sham_model = copy.deepcopy(base_model)
    kappa_records = {}

    print("\n[A] Ablating steep/local heads (W_K → 0):", flush=True)
    for i, (ell, h) in enumerate(STEEP_LOCAL):
        delta = compute_positional_field_ln1(base_model, ell, rng_census)

        _, _, Vt = np.linalg.svd(delta, full_matrices=False)
        P_k      = Vt[:4]                        # top-4 positional directions

        W_K      = get_wk(base_model, ell, h)   # (D_MODEL, D_HEAD)
        kappa_before = compute_kappa(W_K, delta)

        # ── Ablation: zero W_K ────────────────────────────────────────────────
        W_K_abl      = np.zeros_like(W_K)
        kappa_after_abl = compute_kappa(W_K_abl, delta)   # should be 0
        set_wk(abl_model, ell, h, W_K_abl)

        # ── Sham: matched-norm perturbation in ⊥(P_k) ───────────────────────
        # Target: add a perturbation with ‖perturbation‖_F = ‖W_K‖_F
        # in directions ⊥ to P_k (so the positional routing is preserved).
        abl_norm = np.linalg.norm(W_K, "fro")   # = ‖-W_K‖_F, the ablation norm

        sham_rng  = np.random.default_rng(SHAM_SEED_BASE + i)
        # Generate 4 random basis vectors in ⊥(P_k)
        rand_vecs = sham_rng.standard_normal((4, D_MODEL))
        for pk_row in P_k:
            rand_vecs -= (rand_vecs @ pk_row)[:, None] * pk_row
        P_perp, _ = np.linalg.qr(rand_vecs.T)   # (D_MODEL, 4)
        P_perp    = P_perp.T                      # (4, D_MODEL)

        # Project W_K onto P_perp, scale to match ablation norm
        W_K_perp_proj   = P_perp.T @ (P_perp @ W_K)   # (D_MODEL, D_HEAD)
        perp_norm = np.linalg.norm(W_K_perp_proj, "fro")
        if perp_norm > 1e-14:
            gamma_sham = abl_norm / perp_norm
        else:
            gamma_sham = 0.0
        W_K_sham         = W_K + gamma_sham * W_K_perp_proj
        kappa_after_sham = compute_kappa(W_K_sham, delta)
        set_wk(sham_model, ell, h, W_K_sham)

        kappa_records[f"L{ell}H{h}"] = {
            "before":         float(kappa_before),
            "after_abl":      float(kappa_after_abl),
            "after_sham":     float(kappa_after_sham),
            "gamma_sham":     float(gamma_sham),
            "abl_norm":       float(abl_norm),
            "perp_proj_norm": float(perp_norm),
        }
        print(f"  L{ell}H{h}: κ̃ {kappa_before:.3f} → abl {kappa_after_abl:.4f} | "
              f"sham {kappa_after_sham:.3f} (γ_sham={gamma_sham:.3f})", flush=True)

    return abl_model, sham_model, kappa_records


# ── Task items (identical to exp-141 through exp-151) ─────────────────────────

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

def evaluate(kappa_records, orig_A, orig_B, abl_A, abl_B, sham_A, sham_B):
    med_orig_A = median_lp(orig_A)
    med_orig_B = median_lp(orig_B)
    med_abl_A  = median_lp(abl_A)
    med_abl_B  = median_lp(abl_B)
    med_sham_A = median_lp(sham_A)
    med_sham_B = median_lp(sham_B)

    dA_abl  = med_abl_A  - med_orig_A
    dB_abl  = med_abl_B  - med_orig_B
    dA_sham = med_sham_A - med_orig_A
    dB_sham = med_sham_B - med_orig_B

    medians = {
        "orig_A": med_orig_A, "orig_B": med_orig_B,
        "abl_A":  med_abl_A,  "abl_B":  med_abl_B,
        "sham_A": med_sham_A, "sham_B": med_sham_B,
    }
    deltas = {
        "abl_A":  dA_abl,  "abl_B":  dB_abl,
        "sham_A": dA_sham, "sham_B": dB_sham,
    }

    # ── Kill checks ──────────────────────────────────────────────────────────
    kappas_after_abl  = [v["after_abl"]  for v in kappa_records.values()]
    kappas_after_sham = [v["after_sham"] for v in kappa_records.values()]
    kappas_before     = [v["before"]     for v in kappa_records.values()]

    k3_fired = any(k > K3_ABLATION_THRESHOLD for k in kappas_after_abl)
    k2_fired = med_orig_A < -20.0 or med_orig_B < -20.0
    k1_fired = (abs(dA_sham) >= abs(dA_abl)) and (abs(dB_sham) >= abs(dB_abl))

    k3 = {
        "fired":             k3_fired,
        "threshold":         K3_ABLATION_THRESHOLD,
        "kappas_after_abl":  kappas_after_abl,
        "kappas_before":     kappas_before,
        "max_after_abl":     max(kappas_after_abl),
    }
    k2 = {"fired": k2_fired, "orig_A": med_orig_A, "orig_B": med_orig_B}
    k1 = {
        "fired":       k1_fired,
        "abs_sham_A":  abs(dA_sham), "abs_abl_A": abs(dA_abl),
        "abs_sham_B":  abs(dB_sham), "abs_abl_B": abs(dB_abl),
    }
    kills = {"K1": k1, "K2": k2, "K3": k3}

    any_kill = k1_fired or k2_fired or k3_fired

    # ── Prediction verdicts ───────────────────────────────────────────────────
    p1        = dB_abl < -0.10
    p2        = abs(dA_abl - dA_sham) < 0.5
    p_null    = abs(dB_abl) <= 0.05
    p_improve = dB_abl > 0.10
    p_relay   = p1 and not k1_fired

    n_B_improved_abl  = sum(1 for o, c in zip(orig_B, abl_B)
                            if c["log_prob_correct"] > o["log_prob_correct"])
    n_B_improved_sham = sum(1 for o, c in zip(orig_B, sham_B)
                            if c["log_prob_correct"] > o["log_prob_correct"])

    if any_kill:
        overall = "INCONCLUSIVE (kill fired)"
    elif p_relay:
        overall = "CONFIRMED — relay account confirmed: W_K ablation degrades Task B"
    elif p_null:
        overall = "INCONCLUSIVE — ablation had no detectable effect on Task B"
    elif p_improve:
        overall = "PARTIAL — ablation improved Task B (competitor pattern, inconsistent with exp-149/150)"
    elif p1 and k1_fired:
        overall = "INCONCLUSIVE — Task B degraded but K1 fires (sham equally disruptive)"
    else:
        overall = "INCONCLUSIVE"

    verdicts = {
        "P1":           {"fired": p1,        "dB_abl": dB_abl, "threshold": -0.10},
        "P2":           {"fired": p2,        "dA_abl": dA_abl, "dA_sham": dA_sham},
        "P_null":       {"fired": p_null,    "dB_abl": dB_abl},
        "P_improve":    {"fired": p_improve, "dB_abl": dB_abl},
        "P_relay":      {"fired": p_relay},
        "n_B_improved_abl":  n_B_improved_abl,
        "n_B_improved_sham": n_B_improved_sham,
        "overall":      overall,
    }

    comparison = {
        "exp149_gpt2medium_suppress_γm1_dB":   -0.10,
        "exp150_gpt2medium_suppress_γm086_dB":  -0.08,
        "exp151_gpt2medium_amplify_γp5_dB":     +0.01,
        "exp152_gpt2medium_WK_ablation_dB":      dB_abl,
        "ablation_vs_suppression_stronger":      dB_abl < -0.10,
    }

    return kills, verdicts, medians, deltas, comparison


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("exp-152 — Causal ablation of steep/local heads, GPT-2 medium", flush=True)
    print(f"  prereg:      {PREREG_COMMIT} (attention-geometry, pushed before this script)", flush=True)
    print(f"  device:      {DEVICE}", flush=True)
    print(f"  STEEP_LOCAL: {['L{}H{}'.format(l,h) for l,h in STEEP_LOCAL]}", flush=True)
    print("  Ablation: W_K → 0 (uniform attention). No structural head manipulation.", flush=True)

    print("\n[1] Loading GPT-2 medium (eager attention)...", flush=True)
    tokenizer  = GPT2Tokenizer.from_pretrained("openai-community/gpt2-medium")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(DEVICE)
    base_model.eval()
    print(f"  Loaded. n_layer={N_LAYERS}, n_head={N_HEADS}, D_MODEL={D_MODEL}", flush=True)

    print("\n[2] Building ablated and sham models...", flush=True)
    rng_census = np.random.default_rng(SEED)
    abl_model, sham_model, kappa_records = build_models(base_model, rng_census)

    # K3 check
    kappas_after_abl = [v["after_abl"] for v in kappa_records.values()]
    max_abl = max(kappas_after_abl)
    if max_abl > K3_ABLATION_THRESHOLD:
        print(f"  WARNING: K3 fires — max κ̃_after_abl = {max_abl:.4f} > {K3_ABLATION_THRESHOLD}",
              flush=True)
    else:
        print(f"  K3 OK: max κ̃_after_abl = {max_abl:.6f} ≤ {K3_ABLATION_THRESHOLD} (ablation complete)",
              flush=True)

    print("\n[3] Task A — original:", flush=True)
    orig_A  = score_task(base_model,  tokenizer, TASK_A, "A")
    print("\n[4] Task A — ablated:", flush=True)
    abl_A   = score_task(abl_model,   tokenizer, TASK_A, "A")
    print("\n[5] Task A — sham:", flush=True)
    sham_A  = score_task(sham_model,  tokenizer, TASK_A, "A")

    print("\n[6] Task B — original:", flush=True)
    orig_B  = score_task(base_model,  tokenizer, TASK_B, "B")
    print("\n[7] Task B — ablated:", flush=True)
    abl_B   = score_task(abl_model,   tokenizer, TASK_B, "B")
    print("\n[8] Task B — sham:", flush=True)
    sham_B  = score_task(sham_model,  tokenizer, TASK_B, "B")

    print("\n[9] Evaluating...", flush=True)
    kills, verdicts, medians, deltas, comparison = evaluate(
        kappa_records, orig_A, orig_B, abl_A, abl_B, sham_A, sham_B
    )

    print(f"\n  Overall verdict: {verdicts['overall']}", flush=True)
    m = medians
    d = deltas
    print(f"  Task A: orig={m['orig_A']:.2f}  abl={m['abl_A']:.2f}  sham={m['sham_A']:.2f}  "
          f"(Δabl={d['abl_A']:+.2f}  Δsham={d['sham_A']:+.2f})", flush=True)
    print(f"  Task B: orig={m['orig_B']:.2f}  abl={m['abl_B']:.2f}  sham={m['sham_B']:.2f}  "
          f"(Δabl={d['abl_B']:+.2f}  Δsham={d['sham_B']:+.2f})", flush=True)

    n_A_imp_abl  = sum(1 for o, c in zip(orig_A, abl_A)
                       if c["log_prob_correct"] > o["log_prob_correct"])
    n_B_imp_abl  = sum(1 for o, c in zip(orig_B, abl_B)
                       if c["log_prob_correct"] > o["log_prob_correct"])
    n_A_imp_sham = sum(1 for o, s in zip(orig_A, sham_A)
                       if s["log_prob_correct"] > o["log_prob_correct"])
    n_B_imp_sham = sum(1 for o, s in zip(orig_B, sham_B)
                       if s["log_prob_correct"] > o["log_prob_correct"])
    print(f"\n  Item-level (abl vs orig):  Task A={n_A_imp_abl}/20   Task B={n_B_imp_abl}/20",
          flush=True)
    print(f"  Item-level (sham vs orig): Task A={n_A_imp_sham}/20  Task B={n_B_imp_sham}/20",
          flush=True)

    print(f"\n  Comparison:", flush=True)
    print(f"    exp-149 (suppress γ=−1.0):    ΔP_B = −0.10 nats (2/20 improved)", flush=True)
    print(f"    exp-150 (suppress γ=−0.86):   ΔP_B = −0.08 nats (2/20 improved)", flush=True)
    print(f"    exp-152 (W_K ablation):        ΔP_B = {d['abl_B']:+.2f} nats ({n_B_imp_abl}/20 improved)",
          flush=True)

    for pk, pv in verdicts.items():
        if pk not in ("overall", "n_B_improved_abl", "n_B_improved_sham"):
            print(f"  {pk}: {'FIRES' if pv.get('fired') else 'not fired'}", flush=True)

    out = {
        "exp":                    "exp-152",
        "prereg_commit":          PREREG_COMMIT,
        "prereg_evidence":        "git-attested — commit 5a15b35 pushed before run.py written",
        "device":                 DEVICE,
        "model":                  MODEL_ID,
        "ablation":               "W_K → 0 for all 5 steep/local heads",
        "steep_local_heads":      [f"L{l}H{h}" for l, h in STEEP_LOCAL],
        "kappa_records":          kappa_records,
        "kills":                  kills,
        "verdicts":               verdicts,
        "medians":                medians,
        "deltas":                 deltas,
        "comparison_to_prior":    comparison,
        "item_counts": {
            "n_A_improved_abl":   n_A_imp_abl,
            "n_B_improved_abl":   n_B_imp_abl,
            "n_A_improved_sham":  n_A_imp_sham,
            "n_B_improved_sham":  n_B_imp_sham,
        },
        "task_A_orig": orig_A, "task_A_abl": abl_A, "task_A_sham": sham_A,
        "task_B_orig": orig_B, "task_B_abl": abl_B, "task_B_sham": sham_B,
    }

    out_path = Path(__file__).resolve().parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
