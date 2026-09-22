"""
exp-151 — World-model battery higher-γ amplification-only: GPT-2 medium
(H_thresh test: γ=+5.0 on random-token structural heads)

Pre-registration: attention-geometry 7ec2b60 (pushed 2026-09-22 before this script).
Follows exp-148 (γ=+2.0, NULL: ΔP_B = +0.01). Tests whether higher gain surfaces
the positional retrieval signal that exp-141 confirmed in GPT-2 small.

H_thresh: the signal exists in medium but requires larger gain (predict P1 fires).
H_arch:   the architectural difference is real; medium structural heads are not
           the same positional retrieval circuit (predict P_null fires again).

STRUCTURAL (amplify γ=+5.0): L6H9, L5H14, L7H5, L9H7, L8H13
  - κ̃_K: 0.208, 0.220, 0.257, 0.417, 0.469 (exp-146 random-token census)
  - γ=+5.0 → 36× amplification factor (vs 9× at γ=+2.0 in exp-148)

No steep/local suppression in this experiment.

Ariel — September 22, 2026, ~4:17 AM MDT. Solo physics room session.
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

PREREG_COMMIT = "7ec2b60"

D_MODEL  = 1024   # GPT-2 medium
D_HEAD   = 64
N_LAYERS = 24
N_HEADS  = 16

MODEL_ID = "openai-community/gpt2-medium"

# Random-token structural heads from exp-146 (lowest κ̃_K)
STRUCTURAL  = [(6, 9), (5, 14), (7, 5), (9, 7), (8, 13)]
GAMMA_AMP   = 5.0    # exp-148 used 2.0; this experiment uses 5.0 (36× vs 9× factor)

# Kill threshold — at γ=+5.0, expected ratio is 36×; fire if actual < ~5× (κ̃_amp < 1.0)
# on ≥ 2/5 structural heads (would indicate the amplification protocol failed)
K3_AMP_THRESHOLD = 1.0

# K3_check: amplification confirmed if κ̃_after ≥ 30× κ̃_before on ≥ 4/5 heads
K3_CHECK_RATIO   = 30.0
K3_CHECK_N       = 4

SHAM_SEED_BASE = 2026092201   # distinct from all prior experiments

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


# ── Build amplified and sham models ───────────────────────────────────────────

def build_models(base_model, rng_census: np.random.Generator):
    """
    Amplified model: amplify STRUCTURAL heads (γ=+5.0, W_K direction).
    Sham model: matched-norm perturbation in ⊥ complement.
    NO steep/local suppression.

    Returns (amp_model, sham_model, kappa_amp).
    """
    amp_model  = copy.deepcopy(base_model)
    sham_model = copy.deepcopy(base_model)
    kappa_amp  = {}

    print(f"\n[A] Amplifying structural heads (γ=+{GAMMA_AMP}, random-token population from exp-146):",
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
        set_wk(amp_model, ell, h, W_K_amp)

        # Sham: matched Frobenius norm in ⊥ complement
        sham_rng  = np.random.default_rng(SHAM_SEED_BASE + i)
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

        ratio = kappa_after_amp / kappa_before if kappa_before > 1e-14 else float("inf")
        kappa_amp[f"L{ell}H{h}"] = {
            "before":          float(kappa_before),
            "after_amp":       float(kappa_after_amp),
            "after_sham":      float(kappa_after_sham),
            "gamma_amp":       float(GAMMA_AMP),
            "gamma_sham":      float(gamma_sham),
            "ratio_achieved":  float(ratio),
        }
        print(f"  L{ell}H{h}: κ̃ {kappa_before:.3f} → amp {kappa_after_amp:.3f} "
              f"({ratio:.0f}×; sham {kappa_after_sham:.3f})", flush=True)

    return amp_model, sham_model, kappa_amp


# ── Task items (identical to exp-141/142/143/145/147/148/149/150) ─────────────

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

def evaluate(kappa_amp, orig_A, orig_B, amp_A, amp_B, sham_A, sham_B):
    med_orig_A = median_lp(orig_A)
    med_orig_B = median_lp(orig_B)
    med_amp_A  = median_lp(amp_A)
    med_amp_B  = median_lp(amp_B)
    med_sham_A = median_lp(sham_A)
    med_sham_B = median_lp(sham_B)

    dA_amp  = med_amp_A  - med_orig_A
    dB_amp  = med_amp_B  - med_orig_B
    dA_sham = med_sham_A - med_orig_A
    dB_sham = med_sham_B - med_orig_B

    medians = {
        "orig_A": med_orig_A, "orig_B": med_orig_B,
        "amp_A":  med_amp_A,  "amp_B":  med_amp_B,
        "sham_A": med_sham_A, "sham_B": med_sham_B,
    }
    deltas = {"amp_A": dA_amp, "amp_B": dB_amp,
              "sham_A": dA_sham, "sham_B": dB_sham}

    # ── Kill checks ──────────────────────────────────────────────────────────
    amp_kappas_after  = [v["after_amp"] for v in kappa_amp.values()]
    amp_ratios        = [v["ratio_achieved"] for v in kappa_amp.values()]
    n_amp_kill        = sum(1 for k in amp_kappas_after if k < K3_AMP_THRESHOLD)
    n_k3_check_passed = sum(1 for r in amp_ratios if r >= K3_CHECK_RATIO)

    k3 = {
        "fired":                n_amp_kill >= 2,
        "n_below_threshold":    n_amp_kill,
        "kill_threshold":       K3_AMP_THRESHOLD,
        "k3_check_ratio":       K3_CHECK_RATIO,
        "k3_check_n_passed":    n_k3_check_passed,
        "k3_check_required":    K3_CHECK_N,
        "k3_check_fired":       n_k3_check_passed >= K3_CHECK_N,
        "ratios_achieved":      amp_ratios,
        "kappas_after":         amp_kappas_after,
    }
    k2 = {"fired": med_orig_A < -20.0 or med_orig_B < -20.0,
          "orig_A": med_orig_A, "orig_B": med_orig_B}
    k1_raw = (abs(dA_sham) >= abs(dA_amp)) and (abs(dB_sham) >= abs(dB_amp))
    k1 = {"fired": k1_raw,
          "abs_sham_A": abs(dA_sham), "abs_amp_A": abs(dA_amp),
          "abs_sham_B": abs(dB_sham), "abs_amp_B": abs(dB_amp)}
    kills = {"K1": k1, "K2": k2, "K3": k3}

    any_kill = k1["fired"] or k2["fired"] or k3["fired"]

    # ── Prediction verdicts ───────────────────────────────────────────────────
    p1 = dB_amp > 0.10
    p2 = abs(dA_amp - dA_sham) < 0.5
    n_B_improved = sum(1 for o, c in zip(orig_B, amp_B)
                       if c["log_prob_correct"] > o["log_prob_correct"])
    p_null     = abs(dB_amp) <= 0.05
    p_degrade  = dB_amp < -0.10
    p_positive = dB_amp > 0

    if any_kill:
        overall = "INCONCLUSIVE (kill fired)"
    elif p_degrade:
        overall = "NULL — high-gamma amplification damages Task B (H_arch supported)"
    elif p_null:
        overall = "NULL — amplification arm remains inert at γ=+5.0 (H_arch supported)"
    elif p1 and p2:
        overall = "CONFIRMED — amplification signal present at γ=+5.0 (H_thresh confirmed)"
    elif p_positive:
        overall = "PARTIAL — directional improvement, below P1 threshold"
    else:
        overall = "INCONCLUSIVE"

    verdicts = {
        "P1":           {"fired": p1,        "dB_amp": dB_amp, "threshold": 0.10},
        "P2":           {"fired": p2,        "dA_amp": dA_amp, "dA_sham": dA_sham,
                         "diff": dA_amp - dA_sham},
        "P_null":       {"fired": p_null,    "dB_amp": dB_amp},
        "P_degrade":    {"fired": p_degrade, "dB_amp": dB_amp},
        "P_positive":   {"fired": p_positive,"dB_amp": dB_amp},
        "n_B_improved": n_B_improved,
        "overall":      overall,
    }

    comparison = {
        "exp141_gpt2small_amplify_γ2_dB":     +0.27,
        "exp148_gpt2medium_amplify_γ2_dB":    +0.01,
        "exp149_gpt2medium_suppress_γm1_dB":  -0.10,
        "exp150_gpt2medium_suppress_γm086_dB": -0.08,
        "exp151_gpt2medium_amplify_γ5_dB":    dB_amp,
        "direction_matches_small":             dB_amp > 0,
    }

    return kills, verdicts, medians, deltas, comparison


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("exp-151 — World-model battery GPT-2 medium, amplify-only γ=+5.0", flush=True)
    print(f"  prereg:    {PREREG_COMMIT} (attention-geometry, pushed before this script)", flush=True)
    print(f"  device:    {DEVICE}", flush=True)
    print(f"  STRUCTURAL (amplify γ=+{GAMMA_AMP}, 36× factor): "
          f"{['L{}H{}'.format(l,h) for l,h in STRUCTURAL]}", flush=True)
    print("  No steep/local suppression in this experiment.", flush=True)

    print("\n[1] Loading GPT-2 medium (eager attention)...", flush=True)
    tokenizer  = GPT2Tokenizer.from_pretrained("openai-community/gpt2-medium")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(DEVICE)
    base_model.eval()
    print(f"  Loaded. n_layer={N_LAYERS}, n_head={N_HEADS}, D_MODEL={D_MODEL}", flush=True)

    print("\n[2] Building amplified and sham models...", flush=True)
    rng_census = np.random.default_rng(SEED)
    amp_model, sham_model, kappa_amp = build_models(base_model, rng_census)

    # K3 check report
    ratios = [v["ratio_achieved"] for v in kappa_amp.values()]
    n_pass = sum(1 for r in ratios if r >= K3_CHECK_RATIO)
    print(f"\n  K3_check: {n_pass}/5 heads achieved ≥{K3_CHECK_RATIO:.0f}× amplification", flush=True)
    kappas_after = [v["after_amp"] for v in kappa_amp.values()]
    n_kill = sum(1 for k in kappas_after if k < K3_AMP_THRESHOLD)
    if n_kill >= 2:
        print(f"  WARNING: K3 kill fires — {n_kill}/5 heads below threshold {K3_AMP_THRESHOLD}",
              flush=True)

    print("\n[3] Task A — original:", flush=True)
    orig_A  = score_task(base_model,  tokenizer, TASK_A, "A")
    print("\n[4] Task A — amplified:", flush=True)
    amp_A   = score_task(amp_model,   tokenizer, TASK_A, "A")
    print("\n[5] Task A — sham:", flush=True)
    sham_A  = score_task(sham_model,  tokenizer, TASK_A, "A")

    print("\n[6] Task B — original:", flush=True)
    orig_B  = score_task(base_model,  tokenizer, TASK_B, "B")
    print("\n[7] Task B — amplified:", flush=True)
    amp_B   = score_task(amp_model,   tokenizer, TASK_B, "B")
    print("\n[8] Task B — sham:", flush=True)
    sham_B  = score_task(sham_model,  tokenizer, TASK_B, "B")

    print("\n[9] Evaluating...", flush=True)
    kills, verdicts, medians, deltas, comparison = evaluate(
        kappa_amp, orig_A, orig_B, amp_A, amp_B, sham_A, sham_B
    )

    print(f"\n  Overall verdict: {verdicts['overall']}", flush=True)
    m = medians
    d = deltas
    print(f"  Task A: orig={m['orig_A']:.2f}  amp={m['amp_A']:.2f}  sham={m['sham_A']:.2f}  "
          f"(Δamp={d['amp_A']:+.2f}  Δsham={d['sham_A']:+.2f})", flush=True)
    print(f"  Task B: orig={m['orig_B']:.2f}  amp={m['amp_B']:.2f}  sham={m['sham_B']:.2f}  "
          f"(Δamp={d['amp_B']:+.2f}  Δsham={d['sham_B']:+.2f})", flush=True)

    n_A_imp_amp  = sum(1 for o, c in zip(orig_A, amp_A)
                       if c["log_prob_correct"] > o["log_prob_correct"])
    n_B_imp_amp  = sum(1 for o, c in zip(orig_B, amp_B)
                       if c["log_prob_correct"] > o["log_prob_correct"])
    n_A_imp_sham = sum(1 for o, s in zip(orig_A, sham_A)
                       if s["log_prob_correct"] > o["log_prob_correct"])
    n_B_imp_sham = sum(1 for o, s in zip(orig_B, sham_B)
                       if s["log_prob_correct"] > o["log_prob_correct"])
    print(f"\n  Item-level (amp vs orig):  Task A={n_A_imp_amp}/20   Task B={n_B_imp_amp}/20",
          flush=True)
    print(f"  Item-level (sham vs orig): Task A={n_A_imp_sham}/20  Task B={n_B_imp_sham}/20",
          flush=True)

    print(f"\n  Comparison:", flush=True)
    print(f"    exp-141 (GPT-2 small, γ=+2.0, CONFIRMED):       ΔP_B = +0.27 nats", flush=True)
    print(f"    exp-148 (GPT-2 medium, γ=+2.0, NULL):           ΔP_B = +0.01 nats", flush=True)
    print(f"    exp-149 (GPT-2 medium, suppress γ=−1.0):        ΔP_B = −0.10 nats", flush=True)
    print(f"    exp-150 (GPT-2 medium, suppress γ=−0.86):       ΔP_B = −0.08 nats", flush=True)
    print(f"    exp-151 (GPT-2 medium, γ=+5.0, this run):       ΔP_B = {d['amp_B']:+.2f} nats",
          flush=True)

    for pk, pv in verdicts.items():
        if pk not in ("overall", "n_B_improved"):
            print(f"  {pk}: {'FIRES' if pv.get('fired') else 'not fired'}", flush=True)

    out = {
        "exp":                    "exp-151",
        "prereg_commit":          PREREG_COMMIT,
        "prereg_evidence":        "git-attested — commit 7ec2b60 pushed before run.py written",
        "device":                 DEVICE,
        "model":                  MODEL_ID,
        "gamma_amp":              GAMMA_AMP,
        "structural_heads":       [f"L{l}H{h}" for l, h in STRUCTURAL],
        "steep_local_suppressed": False,
        "kappa_amp":              kappa_amp,
        "kills":                  kills,
        "verdicts":               verdicts,
        "medians":                medians,
        "deltas":                 deltas,
        "comparison_to_prior":    comparison,
        "item_counts": {
            "n_A_improved_amp":  n_A_imp_amp,
            "n_B_improved_amp":  n_B_imp_amp,
            "n_A_improved_sham": n_A_imp_sham,
            "n_B_improved_sham": n_B_imp_sham,
        },
        "task_A_orig": orig_A, "task_A_amp": amp_A, "task_A_sham": sham_A,
        "task_B_orig": orig_B, "task_B_amp": amp_B, "task_B_sham": sham_B,
    }

    out_path = Path(__file__).resolve().parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
