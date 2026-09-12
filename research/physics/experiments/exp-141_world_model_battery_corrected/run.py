"""
exp-141 — World-model battery: corrected ln_1 hook protocol

Pre-registration: attention-geometry a448cb0 (pushed before this script).
Analysis-only: GPT-2 small (cached). No new training.

Correction vs exp-140: positional field computed on ln_1(h) output, not raw
residual stream h before ln_1. This restores consistency with exp-137's κ̃
framework and makes the W_K amplification target the subspace the heads see.

Same tasks, thresholds, and sham design as exp-140 (pre-reg 10b7bb4).

Ariel — September 12, 2026, ~5:30 PM MDT, solo.
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

SEQ_LEN   = exp112.SEQ_LEN    # 512
N_INPUTS  = exp112.N_INPUTS   # 50
SEED      = exp112.SEED        # 42

PREREG_COMMIT = "a448cb0"
STRUCTURAL = [(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)]
GAMMA          = 2.0
GAMMA_RETRY    = 4.0
K3_THRESHOLD   = 0.5
SHAM_SEED_BASE = 2026091201   # different base than exp-140 to avoid any seed collision
D_MODEL = 768
D_HEAD  = 64
N_HEADS = 12

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ── Positional field — KEY FIX: hook ln_1 output, not block input ─────────────

def compute_positional_field_ln1(model: GPT2LMHeadModel,
                                  layer: int,
                                  rng: np.random.Generator) -> np.ndarray:
    """
    Compute the positional field δ at the OUTPUT of ln_1 for attention block `layer`.

    exp-140 BUG: hooked `model.transformer.h[layer]` and captured `inp[0]` —
    which is the raw residual stream h BEFORE ln_1 is applied internally.

    exp-141 FIX: hook `model.transformer.h[layer].ln_1` and capture the OUTPUT.
    This gives ln_1(h), which is the tensor that W_Q, W_K, W_V actually see.

    Returns δ: shape (SEQ_LEN, D_MODEL) — the centered positional deviation.
    """
    model.eval()
    tok_ids = rng.integers(0, model.config.vocab_size, size=(N_INPUTS, SEQ_LEN))
    tokens  = torch.tensor(tok_ids, dtype=torch.long, device=DEVICE)

    captured = []

    def hook_fn(mod, inp, out):
        # out is the OUTPUT of ln_1 — the normalised residual stream
        captured.append(out.detach().cpu().float().numpy())

    ln1_module = model.transformer.h[layer].ln_1
    handle = ln1_module.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(tokens)

    handle.remove()

    # captured: list of (N_INPUTS, SEQ_LEN, D_MODEL)
    acts = np.concatenate(captured, axis=0)   # (N_INPUTS, SEQ_LEN, D_MODEL)
    xbar  = acts.mean(axis=0)                 # (SEQ_LEN, D_MODEL)  position-mean field
    m     = xbar.mean(axis=0, keepdims=True)
    delta = xbar - m                          # centered positional field
    return delta


def compute_kappa(W_K: np.ndarray, delta: np.ndarray) -> float:
    """κ̃(W_K) = isotropic-normalised positional capture (same formula as exp-137)."""
    d = D_MODEL
    reads = delta @ W_K   # (SEQ_LEN, D_HEAD)
    cap = float((np.linalg.norm(reads, axis=1) ** 2).sum() /
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


# ── Build amplified and sham models ──────────────────────────────────────────

def build_amplified_model(base_model: GPT2LMHeadModel,
                           rng_census: np.random.Generator,
                           gamma: float = GAMMA):
    """
    Amplify the positional read gain of each Δ-window head using the
    ln_1(h)-based positional field (exp-141 correction).

    Returns (amp_model, sham_model, kappa_report).
    """
    amp_model  = copy.deepcopy(base_model)
    sham_model = copy.deepcopy(base_model)
    kappa_report = {}

    for i, (ell, h) in enumerate(STRUCTURAL):
        # [FIX] Use ln_1 output for positional field
        delta = compute_positional_field_ln1(base_model, ell, rng_census)

        # Top-4 PC directions of the positional field
        _, _, Vt = np.linalg.svd(delta, full_matrices=False)
        P_k = Vt[:4]   # (4, D_MODEL)

        W_K = get_wk(base_model, ell, h)   # (D_MODEL, D_HEAD)

        W_K_proj = P_k.T @ (P_k @ W_K)    # positional projection

        kappa_before = compute_kappa(W_K, delta)

        # --- Amplification ---
        W_K_amp        = W_K + gamma * W_K_proj
        kappa_after_amp = compute_kappa(W_K_amp, delta)
        set_wk(amp_model, ell, h, W_K_amp)

        # --- Sham: matched Frobenius norm in the orthogonal complement ---
        sham_rng  = np.random.default_rng(SHAM_SEED_BASE + i)
        rand_vecs = sham_rng.standard_normal((4, D_MODEL))
        for pk_row in P_k:
            rand_vecs -= (rand_vecs @ pk_row)[:, None] * pk_row
        P_perp, _ = np.linalg.qr(rand_vecs.T)
        P_perp = P_perp.T

        W_K_perp_proj = P_perp.T @ (P_perp @ W_K)
        delta_amp_norm  = np.linalg.norm(gamma * W_K_proj, "fro")
        delta_perp_norm = np.linalg.norm(W_K_perp_proj, "fro")
        gamma_sham = delta_amp_norm / delta_perp_norm if delta_perp_norm > 1e-14 else 0.0

        W_K_sham        = W_K + gamma_sham * W_K_perp_proj
        kappa_after_sham = compute_kappa(W_K_sham, delta)
        set_wk(sham_model, ell, h, W_K_sham)

        kappa_report[f"L{ell}H{h}"] = {
            "before":      float(kappa_before),
            "after_amp":   float(kappa_after_amp),
            "after_sham":  float(kappa_after_sham),
            "gamma":       float(gamma),
            "gamma_sham":  float(gamma_sham),
            "delta_amp_norm": float(delta_amp_norm),
        }
        print(f"  L{ell}H{h}: κ̃ {kappa_before:.3f} → amp {kappa_after_amp:.3f} "
              f"(sham {kappa_after_sham:.3f})", flush=True)

    return amp_model, sham_model, kappa_report


# ── Task items (same as exp-140) ──────────────────────────────────────────────

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
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
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
            "item":            f"{task_name}{idx+1:02d}",
            "prompt":          prompt[:60] + "...",
            "answer":          answer,
            "correct_tok_id":  int(correct_id),
            "log_prob_correct": lp,
            "rank_correct":    rank,
        })
        print(f"    {task_name}{idx+1:02d}: '{answer}' logP={lp:.2f}  rank={rank}", flush=True)
    return results


def median_lp(results):
    return float(np.median([r["log_prob_correct"] for r in results]))


# ── Verdict evaluation (same criteria as exp-140 pre-reg) ─────────────────────

def evaluate_verdicts(kappa_report, orig_A, orig_B, amp_A, amp_B, sham_A, sham_B):
    verdicts = {}

    amp_kappas = [v["after_amp"] for v in kappa_report.values()]
    k3_fail    = sum(1 for k in amp_kappas if k < K3_THRESHOLD)
    verdicts["K3"] = {"fired": k3_fail >= 2, "n_below_threshold": k3_fail,
                      "values": amp_kappas}

    med = {
        "orig_A": median_lp(orig_A), "orig_B": median_lp(orig_B),
        "amp_A":  median_lp(amp_A),  "amp_B":  median_lp(amp_B),
        "sham_A": median_lp(sham_A), "sham_B": median_lp(sham_B),
    }
    dA_amp  = med["amp_A"]  - med["orig_A"]
    dB_amp  = med["amp_B"]  - med["orig_B"]
    dA_sham = med["sham_A"] - med["orig_A"]
    dB_sham = med["sham_B"] - med["orig_B"]
    verdicts["medians"] = med
    verdicts["deltas"]  = {"amp_A": dA_amp, "amp_B": dB_amp,
                           "sham_A": dA_sham, "sham_B": dB_sham}

    k2 = med["orig_A"] < -10.0 or med["orig_B"] < -10.0
    verdicts["K2"] = {"fired": k2, "orig_A": med["orig_A"], "orig_B": med["orig_B"]}

    k1 = (abs(dA_sham) > abs(dA_amp)) and (abs(dB_sham) > abs(dB_amp))
    verdicts["K1"] = {"fired": k1,
                      "abs_sham_A": abs(dA_sham), "abs_amp_A": abs(dA_amp),
                      "abs_sham_B": abs(dB_sham), "abs_amp_B": abs(dB_amp)}

    p1 = (dA_amp < -1.0) and (dA_amp < dB_amp - 1.5)
    p2 = (dB_amp >  1.0) and (dB_amp > dA_amp + 1.5)
    p3 = (abs(dA_amp - dA_sham) < 0.5) and (abs(dB_amp - dB_sham) < 0.5)
    verdicts["P1"] = {"fired": p1, "dA_amp": dA_amp, "dB_amp": dB_amp}
    verdicts["P2"] = {"fired": p2, "dA_amp": dA_amp, "dB_amp": dB_amp}
    verdicts["P3"] = {"fired": p3, "delta_A_amp_vs_sham": dA_amp - dA_sham,
                      "delta_B_amp_vs_sham": dB_amp - dB_sham}

    any_kill = verdicts["K1"]["fired"] or verdicts["K2"]["fired"] or verdicts["K3"]["fired"]
    if any_kill:
        overall = "INCONCLUSIVE"
    elif p1 and p2:
        overall = "CONFIRMED"
    elif p1:
        overall = "PARTIAL"
    elif p2:
        overall = "PARTIAL"
    elif p3:
        overall = "INCONCLUSIVE"
    else:
        overall = "INCONCLUSIVE"
    verdicts["overall"] = overall
    return verdicts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("exp-141 — World-model battery (corrected ln_1 hook protocol)", flush=True)
    print(f"  prereg:  {PREREG_COMMIT}", flush=True)
    print(f"  device:  {DEVICE}", flush=True)
    print(f"  gamma:   {GAMMA}", flush=True)
    print(f"  fix:     positional field from ln_1 output, not raw residual stream", flush=True)

    print("\n[1] Loading GPT-2 small...", flush=True)
    tokenizer  = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    base_model = GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2", torch_dtype=torch.float32
    ).to(DEVICE)

    print("\n[2] Building amplified and sham models (ln_1-corrected field)...", flush=True)
    rng_census = np.random.default_rng(SEED)
    amp_model, sham_model, kappa_report = build_amplified_model(
        base_model, rng_census, gamma=GAMMA
    )

    # K3 gate — retry at higher gamma if needed
    n_below = sum(1 for v in kappa_report.values() if v["after_amp"] < K3_THRESHOLD)
    if n_below >= 2:
        print(f"\n  K3 gate: {n_below}/5 heads below {K3_THRESHOLD}. Retrying at γ={GAMMA_RETRY}...", flush=True)
        rng2 = np.random.default_rng(SEED)
        amp_model, sham_model, kappa_report = build_amplified_model(
            base_model, rng2, gamma=GAMMA_RETRY
        )
        n_below2 = sum(1 for v in kappa_report.values() if v["after_amp"] < K3_THRESHOLD)
        if n_below2 >= 2:
            print(f"  K3 still fires at γ={GAMMA_RETRY}. Flagging INCONCLUSIVE.", flush=True)

    # Sanity check: are κ̃ baselines in exp-137 range (0.05–0.33)?
    baselines = {k: v["before"] for k, v in kappa_report.items()}
    print(f"\n  κ̃ baseline check (expect ~0.05–0.33 per exp-137):", flush=True)
    for k, v in baselines.items():
        flag = "" if 0.0 < v < 0.5 else " ← WARNING: out of expected range"
        print(f"    {k}: {v:.4f}{flag}", flush=True)

    print("\n[3] Task A — entity-state tracking (original):", flush=True)
    orig_A = score_task(base_model, tokenizer, TASK_A, "A")
    print("\n[4] Task A — entity-state tracking (amplified):", flush=True)
    amp_A  = score_task(amp_model,  tokenizer, TASK_A, "A")
    print("\n[5] Task A — entity-state tracking (sham):", flush=True)
    sham_A = score_task(sham_model, tokenizer, TASK_A, "A")

    print("\n[6] Task B — positional retrieval (original):", flush=True)
    orig_B = score_task(base_model, tokenizer, TASK_B, "B")
    print("\n[7] Task B — positional retrieval (amplified):", flush=True)
    amp_B  = score_task(amp_model,  tokenizer, TASK_B, "B")
    print("\n[8] Task B — positional retrieval (sham):", flush=True)
    sham_B = score_task(sham_model, tokenizer, TASK_B, "B")

    print("\n[9] Evaluating verdicts...", flush=True)
    verdicts = evaluate_verdicts(kappa_report, orig_A, orig_B, amp_A, amp_B, sham_A, sham_B)

    print(f"\n  Overall verdict: {verdicts['overall']}", flush=True)
    m = verdicts["medians"]
    d = verdicts["deltas"]
    print(f"  Task A: orig={m['orig_A']:.2f}  amp={m['amp_A']:.2f}  sham={m['sham_A']:.2f}  "
          f"(Δamp={d['amp_A']:+.2f}  Δsham={d['sham_A']:+.2f})", flush=True)
    print(f"  Task B: orig={m['orig_B']:.2f}  amp={m['amp_B']:.2f}  sham={m['sham_B']:.2f}  "
          f"(Δamp={d['amp_B']:+.2f}  Δsham={d['sham_B']:+.2f})", flush=True)

    # Item-level binomial
    n_A_improved = sum(1 for o, a in zip(orig_A, amp_A)
                       if a["log_prob_correct"] > o["log_prob_correct"])
    n_B_improved = sum(1 for o, a in zip(orig_B, amp_B)
                       if a["log_prob_correct"] > o["log_prob_correct"])
    print(f"\n  Item-level improvement (amp vs orig):", flush=True)
    print(f"    Task A: {n_A_improved}/20", flush=True)
    print(f"    Task B: {n_B_improved}/20", flush=True)

    out = {
        "exp":           "exp-141",
        "prereg_commit": PREREG_COMMIT,
        "prereg_evidence": "git-attested — commit a448cb0 pushed before run.py written",
        "fix_applied":   "ln_1 output hook (model.transformer.h[layer].ln_1 forward hook, output captured)",
        "device":        DEVICE,
        "gamma":         GAMMA,
        "structural_heads": [f"L{ell}H{h}" for ell, h in STRUCTURAL],
        "kappa_report":  kappa_report,
        "kappa_baseline_check": baselines,
        "verdicts":      verdicts,
        "item_counts": {
            "n_A_improved": n_A_improved,
            "n_B_improved": n_B_improved,
        },
        "task_A_orig":   orig_A,
        "task_A_amp":    amp_A,
        "task_A_sham":   sham_A,
        "task_B_orig":   orig_B,
        "task_B_amp":    amp_B,
        "task_B_sham":   sham_B,
    }

    out_path = HERE / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
