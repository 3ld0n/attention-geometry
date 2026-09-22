"""
exp-153 — W_V ablation of steep/local heads: GPT-2 medium (value-pathway test)

Pre-registration: attention-geometry 5b63fdb (pushed 2026-09-22 before this script).
Follows exp-152 (W_K ablation, INCONCLUSIVE — suppression-ablation dissociation):
  W_K = 0 → Task B unchanged (+0.051 nats); suppression → Task B degrades (−0.08 to −0.10).

Mechanistic question: does the value pathway of the 5 steep/local heads carry anything
Task B needs, or are these heads genuinely neutral for Task B under any clean removal?

Ablation: W_V → 0 for all 5 steep/local heads (L4H13, L15H8, L8H7, L5H11, L11H7).
  Attention scores/weights computed normally (steep/local positional pattern preserved).
  Head writes attention_weights @ (0 @ x) = 0 to the residual stream — silent head.

Sham: W_V → random matrix of same Frobenius norm (matched-scale disruption; wrong directions).
  Tests: is the effect (if any) specific to zeroing vs. any large-scale disruption of W_V?

H_value: ΔP_B < −0.10 (value pathway is load-bearing for Task B).
H_neutral: |ΔP_B| ≤ 0.10 (heads are genuinely neutral; suppression degradation is
           entirely key-routing misrouting by residual non-positional W_K).

STEEP_LOCAL: L4H13, L15H8, L8H7, L5H11, L11H7 (κ̃_K ≈ 43.2, 39.9, 33.3, 31.7, 30.9)
W_V lives in c_attn.weight[:, 2*D_MODEL + head*D_HEAD : 2*D_MODEL + (head+1)*D_HEAD]

Ariel — September 22, 2026, ~12:17 PM MDT. Solo physics room session.
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

PREREG_COMMIT = "5b63fdb"

D_MODEL  = 1024   # GPT-2 medium
D_HEAD   = 64
N_LAYERS = 24
N_HEADS  = 16

MODEL_ID = "openai-community/gpt2-medium"

# Steep/local heads (from exp-142/147/149/150/152)
# κ̃_K: 43.2, 39.9, 33.3, 31.7, 30.9 (unchanged — W_K not modified in this experiment)
STEEP_LOCAL = [(4, 13), (15, 8), (8, 7), (5, 11), (11, 7)]

# K3: value output is zero iff ‖W_V_ablated‖_F = 0 (checked directly)
K3_WV_NORM_THRESHOLD = 1e-12

SHAM_SEED_BASE = 2026092253   # as specified in prereg; distinct from all prior experiments

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ── W_V accessors ─────────────────────────────────────────────────────────────

def get_wv(model, layer: int, head: int) -> np.ndarray:
    """W_V for (layer, head) from GPT-2 c_attn weight. Returns (D_MODEL, D_HEAD)."""
    W = model.transformer.h[layer].attn.c_attn.weight.detach().cpu().double().numpy()
    offset = 2 * D_MODEL + head * D_HEAD
    return W[:, offset: offset + D_HEAD]


def set_wv(model, layer: int, head: int, W_V_new: np.ndarray):
    """Write W_V back for (layer, head)."""
    W = model.transformer.h[layer].attn.c_attn.weight.data
    offset = 2 * D_MODEL + head * D_HEAD
    with torch.no_grad():
        W[:, offset: offset + D_HEAD] = torch.tensor(
            W_V_new, dtype=W.dtype, device=W.device
        )


def get_wk(model, layer: int, head: int) -> np.ndarray:
    """W_K for (layer, head) — read-only; used to verify W_K was not modified."""
    W = model.transformer.h[layer].attn.c_attn.weight.detach().cpu().double().numpy()
    offset = D_MODEL + head * D_HEAD
    return W[:, offset: offset + D_HEAD]


# ── Positional field and κ̃ (for W_K integrity check) ─────────────────────────

def compute_positional_field_ln1(model, layer: int,
                                  rng: np.random.Generator) -> np.ndarray:
    """Positional field δ at ln_1 output for layer. Shape: (SEQ_LEN, D_MODEL)."""
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


# ── Build ablated and sham models ─────────────────────────────────────────────

def build_models(base_model, rng_census: np.random.Generator):
    """
    Ablated model: W_V = 0 for STEEP_LOCAL heads.
    Sham model: W_V = random matrix, Frobenius norm matched to original W_V.

    W_K is NOT modified in either model (we verify κ̃_K is unchanged).

    Returns (abl_model, sham_model, wv_records, kappa_records).
    """
    abl_model  = copy.deepcopy(base_model)
    sham_model = copy.deepcopy(base_model)
    wv_records    = {}   # per-head: norms, K3 check
    kappa_records = {}   # per-head: κ̃_K (should be unchanged)

    print("\n[A] Ablating steep/local heads (W_V → 0):", flush=True)
    for i, (ell, h) in enumerate(STEEP_LOCAL):
        W_V = get_wv(base_model, ell, h)   # (D_MODEL, D_HEAD)
        fro = float(np.linalg.norm(W_V, "fro"))

        # ── Ablation: zero W_V ────────────────────────────────────────────────
        W_V_abl = np.zeros_like(W_V)
        set_wv(abl_model, ell, h, W_V_abl)
        fro_after_abl = float(np.linalg.norm(
            get_wv(abl_model, ell, h), "fro"
        ))

        # ── Sham: matched-norm random matrix ─────────────────────────────────
        sham_rng = np.random.default_rng(SHAM_SEED_BASE + i)
        R = sham_rng.standard_normal(W_V.shape)   # (D_MODEL, D_HEAD)
        R_norm = float(np.linalg.norm(R, "fro"))
        if R_norm > 1e-14:
            W_V_sham = R * (fro / R_norm)
        else:
            W_V_sham = R
        set_wv(sham_model, ell, h, W_V_sham)
        fro_after_sham = float(np.linalg.norm(
            get_wv(sham_model, ell, h), "fro"
        ))

        # ── W_K integrity check (κ̃ should match prior exp values) ────────────
        delta = compute_positional_field_ln1(base_model, ell, rng_census)
        W_K   = get_wk(base_model, ell, h)
        kappa = compute_kappa(W_K, delta)

        wv_records[f"L{ell}H{h}"] = {
            "wv_fro_before":     fro,
            "wv_fro_after_abl":  fro_after_abl,
            "wv_fro_after_sham": fro_after_sham,
            "k3_ok":             fro_after_abl < K3_WV_NORM_THRESHOLD,
        }
        kappa_records[f"L{ell}H{h}"] = {
            "kappa_K": float(kappa),   # should be ≈ 43.2, 39.9, 33.3, 31.7, 30.9
        }

        k3_status = "OK" if fro_after_abl < K3_WV_NORM_THRESHOLD else "FAIL"
        print(f"  L{ell}H{h}: ‖W_V‖_F {fro:.3f} → abl {fro_after_abl:.2e} [{k3_status}] | "
              f"sham {fro_after_sham:.3f} | κ̃_K={kappa:.3f}", flush=True)

    return abl_model, sham_model, wv_records, kappa_records


# ── Task items (identical to exp-141 through exp-152) ─────────────────────────

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

def evaluate(wv_records, kappa_records, orig_A, orig_B, abl_A, abl_B, sham_A, sham_B):
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
    k3_fired = not all(v["k3_ok"] for v in wv_records.values())
    k2_fired = med_orig_A < -20.0 or med_orig_B < -20.0
    k1_fired = (abs(dA_sham) >= abs(dA_abl)) and (abs(dB_sham) >= abs(dB_abl))

    k3 = {
        "fired":        k3_fired,
        "threshold":    K3_WV_NORM_THRESHOLD,
        "per_head":     {k: v["k3_ok"] for k, v in wv_records.items()},
        "all_ok":       all(v["k3_ok"] for v in wv_records.values()),
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
    p_null    = abs(dB_abl) <= 0.10
    p_improve = dB_abl > 0.10

    n_B_improved_abl  = sum(1 for o, c in zip(orig_B, abl_B)
                            if c["log_prob_correct"] > o["log_prob_correct"])
    n_B_improved_sham = sum(1 for o, c in zip(orig_B, sham_B)
                            if c["log_prob_correct"] > o["log_prob_correct"])

    if any_kill:
        overall = "INCONCLUSIVE (kill fired)"
    elif p1 and not k1_fired:
        overall = "H_value CONFIRMED — value pathway is load-bearing for Task B"
    elif p_null and not k3_fired:
        overall = "H_neutral SUPPORTED — heads are genuinely neutral for Task B; " \
                  "suppression degradation is key-routing misrouting"
    elif p_improve and not k1_fired:
        overall = "PARTIAL — W_V ablation improves Task B (value pathway was interfering)"
    else:
        overall = "INCONCLUSIVE"

    verdicts = {
        "P1":              {"fired": p1,        "dB_abl": dB_abl, "threshold": -0.10},
        "P2":              {"fired": p2,        "dA_abl": dA_abl, "dA_sham": dA_sham},
        "P_null":          {"fired": p_null,    "dB_abl": dB_abl},
        "P_improve":       {"fired": p_improve, "dB_abl": dB_abl},
        "n_B_improved_abl":  n_B_improved_abl,
        "n_B_improved_sham": n_B_improved_sham,
        "overall":         overall,
    }

    comparison = {
        "exp149_gpt2medium_suppress_γm1_dB":    -0.10,
        "exp150_gpt2medium_suppress_γm086_dB":  -0.08,
        "exp152_gpt2medium_WK_ablation_dB":    +0.051,
        "exp153_gpt2medium_WV_ablation_dB":     dB_abl,
    }

    return kills, verdicts, medians, deltas, comparison


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("exp-153 — W_V ablation of steep/local heads, GPT-2 medium", flush=True)
    print(f"  prereg:      {PREREG_COMMIT} (attention-geometry, pushed before this script)", flush=True)
    print(f"  device:      {DEVICE}", flush=True)
    print(f"  STEEP_LOCAL: {['L{}H{}'.format(l,h) for l,h in STEEP_LOCAL]}", flush=True)
    print("  Ablation: W_V → 0 (silent head). W_K NOT modified.", flush=True)

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
    abl_model, sham_model, wv_records, kappa_records = build_models(base_model, rng_census)

    # K3 check
    k3_all_ok = all(v["k3_ok"] for v in wv_records.values())
    if not k3_all_ok:
        failing = [k for k, v in wv_records.items() if not v["k3_ok"]]
        print(f"  WARNING: K3 fires — W_V not fully zeroed for: {failing}", flush=True)
    else:
        print(f"  K3 OK: all 5 heads have ‖W_V‖_F < {K3_WV_NORM_THRESHOLD} (ablation complete)",
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
        wv_records, kappa_records, orig_A, orig_B, abl_A, abl_B, sham_A, sham_B
    )

    print(f"\n  Overall verdict: {verdicts['overall']}", flush=True)
    m = medians
    d = deltas
    print(f"  Task A: orig={m['orig_A']:.2f}  abl={m['abl_A']:.2f}  sham={m['sham_A']:.2f}  "
          f"(Δabl={d['abl_A']:+.3f}  Δsham={d['sham_A']:+.3f})", flush=True)
    print(f"  Task B: orig={m['orig_B']:.2f}  abl={m['abl_B']:.2f}  sham={m['sham_B']:.2f}  "
          f"(Δabl={d['abl_B']:+.3f}  Δsham={d['sham_B']:+.3f})", flush=True)

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

    print(f"\n  Full dissociation table:", flush=True)
    print(f"    exp-149 (suppress γ=−1.0, ~600×):   ΔP_B = −0.10 nats (2/20 improved)", flush=True)
    print(f"    exp-150 (suppress γ=−0.86, ~42×):   ΔP_B = −0.08 nats (2/20 improved)", flush=True)
    print(f"    exp-152 (W_K ablation, W_K=0):       ΔP_B = +0.051 nats (10/20 improved)", flush=True)
    print(f"    exp-153 (W_V ablation, W_V=0):       ΔP_B = {d['abl_B']:+.3f} nats ({n_B_imp_abl}/20 improved)",
          flush=True)

    for pk, pv in verdicts.items():
        if pk not in ("overall", "n_B_improved_abl", "n_B_improved_sham"):
            print(f"  {pk}: {'FIRES' if pv.get('fired') else 'not fired'}", flush=True)

    out = {
        "exp":                    "exp-153",
        "prereg_commit":          PREREG_COMMIT,
        "prereg_evidence":        "git-attested — commit 5b63fdb pushed before run.py written",
        "device":                 DEVICE,
        "model":                  MODEL_ID,
        "ablation":               "W_V → 0 for all 5 steep/local heads (W_K unchanged)",
        "steep_local_heads":      [f"L{l}H{h}" for l, h in STEEP_LOCAL],
        "wv_records":             wv_records,
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
