"""
exp-161 — Census query-pool test

Pre-registration: attention-geometry 64b914b (committed before this script).

Does the fitted census exponent σ_pos vary when the query pool moves to different
absolute positions? A relative-lag law predicts pool-invariance; absolute-key-
position drift predicts pool-dependence.

Four pools (all with seq_len=1024):
  A: queries 64–319   (early)
  B: queries 256–511  (standard — baseline comparison)
  C: queries 512–767  (mid-late)
  D: queries 768–1023 (late)

Heads: 5 structural Δ-window heads + 4 top text-native Δ-window heads (9 total).

Ariel — 2026-09-25, solo physics room, ~1:10 AM MDT.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Constants ──────────────────────────────────────────────────────────────────

PREREG_COMMIT = "64b914b"

# Heads to measure
STRUCTURAL_HEADS = [(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)]
TEXT_NATIVE_TOP4 = [(10, 10), (9, 6), (10, 1), (10, 2)]
TARGET_HEADS = STRUCTURAL_HEADS + TEXT_NATIVE_TOP4  # 9 heads total

# Census protocol
SEQ_LEN   = 1024   # extended from standard 512 to reach pools C and D
N_INPUTS  = 200    # random sequences (more than standard 50 for stability)
SEED      = 42
FIT_LO    = 8
FIT_HI    = 256    # same fitting range for all pools

# Query pools: (start, end) — inclusive
POOLS = {
    "A": (64, 319),    # early
    "B": (256, 511),   # standard baseline
    "C": (512, 767),   # mid-late
    "D": (768, 1023),  # late
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ── Lag profile for a single attention tensor, restricted to a query pool ──────

def lag_profile_pool(att: np.ndarray, q_start: int, q_end: int) -> tuple:
    """
    att: (n_heads, L, L) — single sequence attention tensor
    q_start, q_end: inclusive query-position range for this pool

    Returns (prof, counts) where:
      prof[head, dx] = mean attention at lag dx over queries i in [q_start, q_end]
      counts[dx]     = number of contributing (i, j) pairs at lag dx

    Uses np.diagonal for vectorized access — much faster than explicit i-loop.
    """
    n_heads, L, _ = att.shape
    q_end_eff = min(q_end, L - 1)
    prof = np.zeros((n_heads, FIT_HI + 2))
    counts = np.zeros(FIT_HI + 2, dtype=np.int64)

    for dx in range(FIT_HI + 2):
        # np.diagonal(att, offset=-dx) shape: (n_heads, L-dx)
        # diag[h, k] = att[h, k+dx, k]  →  query position = k+dx, key = k
        # We want k+dx in [q_start, q_end_eff], i.e. k in [q_start-dx, q_end_eff-dx]
        if dx > q_end_eff:
            continue
        k_lo = max(0, q_start - dx)
        k_hi = q_end_eff - dx          # inclusive
        if k_lo > k_hi:
            continue
        diag = np.diagonal(att, offset=-dx, axis1=-2, axis2=-1)  # (n_heads, L-dx)
        slice_ = diag[:, k_lo: k_hi + 1]                         # (n_heads, n_valid)
        if slice_.shape[1] == 0:
            continue
        prof[:, dx] = slice_.mean(axis=-1)
        counts[dx] = slice_.shape[1]

    return prof, counts


# ── Power-law fit ──────────────────────────────────────────────────────────────

def fit_power_law(profile: np.ndarray, fit_lo: int = FIT_LO, fit_hi: int = FIT_HI):
    """
    Fit A(dx) ~ dx^(-2*sigma) by OLS in log-log over [fit_lo, fit_hi].
    Returns (sigma, R2) or (None, None) if insufficient data.
    """
    lags = np.arange(fit_lo, fit_hi + 1)
    y = profile[fit_lo:fit_hi + 1]
    ok = y > 1e-15
    if ok.sum() < 5:
        return None, None
    lx = np.log(lags[ok].astype(float))
    ly = np.log(y[ok])
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, ly, rcond=None)
    resid = ly - X @ c
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 1e-30 else 0.0
    sigma = float(-c[1] / 2)
    return sigma, r2


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Loading GPT-2 small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2", torch_dtype=torch.float32, attn_implementation="eager"
    ).to(DEVICE).eval()
    n_layers = model.config.n_layer
    n_heads_model = model.config.n_head
    vocab_size = model.config.vocab_size
    print(f"  {n_layers}L / {n_heads_model}H, vocab {vocab_size}")

    # Pre-allocate per-pool mean lag profiles
    pool_profs = {
        name: np.zeros((n_layers, n_heads_model, FIT_HI + 2))
        for name in POOLS
    }
    pool_counts = {name: np.zeros(FIT_HI + 2, dtype=np.int64) for name in POOLS}

    rng = np.random.default_rng(SEED)
    print(f"Running {N_INPUTS} forward passes on SEQ_LEN={SEQ_LEN}...")
    for inp_idx in range(N_INPUTS):
        token_ids = rng.integers(0, vocab_size, size=(1, SEQ_LEN))
        ids = torch.tensor(token_ids, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            out = model(ids, output_attentions=True)

        # Convert each layer to CPU once, then accumulate all pools
        for lay in range(n_layers):
            a = out.attentions[lay]
            assert a.dtype == torch.float32, (
                f"Layer {lay}: attention dtype {a.dtype} — need fp32"
            )
            a_np = a[0].cpu().numpy()  # (n_heads, SEQ_LEN, SEQ_LEN) — done once
            for name, (q_start, q_end) in POOLS.items():
                prof_seq, cnts = lag_profile_pool(a_np, q_start, q_end)
                pool_profs[name][lay] += prof_seq
                if lay == 0:
                    pool_counts[name] += cnts

        del out
        if (inp_idx + 1) % 50 == 0:
            print(f"  forward {inp_idx + 1}/{N_INPUTS}  elapsed={time.time()-t0:.1f}s",
                  flush=True)

    # Average over inputs
    for name in POOLS:
        pool_profs[name] /= N_INPUTS

    # ── Fit and report ─────────────────────────────────────────────────────────
    print("\n=== Results ===")
    results_per_pool = {}
    for pool_name, (q_start, q_end) in POOLS.items():
        head_results = []
        for (lay, h) in TARGET_HEADS:
            prof = pool_profs[pool_name][lay, h]
            sigma, r2 = fit_power_law(prof)
            head_results.append({
                "layer": lay, "head": h,
                "sigma_pos": round(sigma, 4) if sigma is not None else None,
                "r2": round(r2, 4) if r2 is not None else None,
            })
            label = "(structural)" if (lay, h) in STRUCTURAL_HEADS else "(text-native)"
            print(f"  Pool {pool_name} L{lay}H{h} {label}: "
                  f"σ={sigma:.4f}" if sigma else f"  Pool {pool_name} L{lay}H{h}: no fit")
        results_per_pool[pool_name] = {
            "q_start": q_start,
            "q_end": q_end,
            "heads": head_results,
        }

    # ── Kill condition evaluation ──────────────────────────────────────────────

    # H_pool_invariant: Δσ > 0.05 across B/C/D on ≥ 3/5 structural heads → K1
    k1_checks = []
    for (lay, h) in STRUCTURAL_HEADS:
        sigmas_bcd = []
        for pool_name in ["B", "C", "D"]:
            for hr in results_per_pool[pool_name]["heads"]:
                if hr["layer"] == lay and hr["head"] == h:
                    if hr["sigma_pos"] is not None:
                        sigmas_bcd.append(hr["sigma_pos"])
        if len(sigmas_bcd) >= 2:
            delta_sigma = max(sigmas_bcd) - min(sigmas_bcd)
            k1_fires = delta_sigma > 0.05
            k1_checks.append({
                "head": f"L{lay}H{h}",
                "sigma_BCD": sigmas_bcd,
                "delta_sigma": round(delta_sigma, 4),
                "k1_fires": k1_fires,
            })

    k1_count = sum(1 for c in k1_checks if c["k1_fires"])
    H_pool_invariant_status = "DEAD" if k1_count >= 3 else "NOT_DEAD"

    # H_direction: σ_pos(A) > σ_pos(B) > σ_pos(C) > σ_pos(D) for structural heads
    direction_checks = []
    for (lay, h) in STRUCTURAL_HEADS:
        pool_sigmas = {}
        for pool_name in ["A", "B", "C", "D"]:
            for hr in results_per_pool[pool_name]["heads"]:
                if hr["layer"] == lay and hr["head"] == h and hr["sigma_pos"] is not None:
                    pool_sigmas[pool_name] = hr["sigma_pos"]
        if len(pool_sigmas) == 4:
            monotone_decreasing = (
                pool_sigmas["A"] > pool_sigmas["B"] and
                pool_sigmas["B"] > pool_sigmas["C"] and
                pool_sigmas["C"] > pool_sigmas["D"]
            )
            direction_checks.append({
                "head": f"L{lay}H{h}",
                "sigma_ABCD": [pool_sigmas.get(k) for k in ["A", "B", "C", "D"]],
                "monotone_decreasing": monotone_decreasing,
            })

    monotone_count = sum(1 for d in direction_checks if d["monotone_decreasing"])
    H_direction_status = (
        "CONFIRMED" if monotone_count >= 3
        else "DEAD" if monotone_count == 0
        else "PARTIAL"
    )

    # H_baseline_reproducibility: Pool B vs exp-118 structural head σ_pos
    # exp-118 did not record per-head σ_pos for structural heads (it measured Δ-window).
    # Approximate reference: exp-007 / published census σ_pos ≈ 0.249 (median Δ).
    # Pool B should reproduce approximately this value for structural heads.
    # K3: Δσ_pool_B vs 0.249 > 0.04 for majority of structural heads.
    baseline_ref = 0.249
    k3_checks = []
    for (lay, h) in STRUCTURAL_HEADS:
        for hr in results_per_pool["B"]["heads"]:
            if hr["layer"] == lay and hr["head"] == h and hr["sigma_pos"] is not None:
                delta = abs(hr["sigma_pos"] - baseline_ref)
                k3_checks.append({
                    "head": f"L{lay}H{h}",
                    "sigma_pool_B": hr["sigma_pos"],
                    "delta_from_baseline": round(delta, 4),
                    "k3_fires": delta > 0.04,
                })

    k3_count = sum(1 for c in k3_checks if c["k3_fires"])
    H_baseline_status = "OK" if k3_count < 3 else "PROTOCOL_CONCERN"

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n=== Kill conditions ===")
    print(f"K1 (H_pool_invariant DEAD): {k1_count}/5 structural heads Δσ>0.05 → "
          f"H_pool_invariant: {H_pool_invariant_status}")
    print(f"H_direction ({monotone_count}/5 monotone decreasing): {H_direction_status}")
    print(f"K3 (baseline check): {k3_count}/5 heads off → {H_baseline_status}")

    results = {
        "exp": "exp-161",
        "prereg_commit": PREREG_COMMIT,
        "date": "2026-09-25",
        "model": "gpt2",
        "seq_len": SEQ_LEN,
        "n_inputs": N_INPUTS,
        "seed": SEED,
        "fit_lo": FIT_LO,
        "fit_hi": FIT_HI,
        "target_heads": {
            "structural": [{"layer": l, "head": h} for l, h in STRUCTURAL_HEADS],
            "text_native_top4": [{"layer": l, "head": h} for l, h in TEXT_NATIVE_TOP4],
        },
        "pools": results_per_pool,
        "kill_conditions": {
            "K1_H_pool_invariant": {
                "per_head": k1_checks,
                "heads_firing": k1_count,
                "verdict": H_pool_invariant_status,
            },
            "K2_H_direction": {
                "per_head": direction_checks,
                "monotone_count": monotone_count,
                "verdict": H_direction_status,
            },
            "K3_H_baseline_reproducibility": {
                "per_head": k3_checks,
                "heads_firing": k3_count,
                "verdict": H_baseline_status,
            },
        },
        "elapsed_s": round(time.time() - t0, 1),
    }

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
