#!/usr/bin/env python3
"""
exp-174 — 4-point cumulant structure of census score distributions
Pre-registration: prereg.md in this folder.
Commit: db56b4f (pushed to attention-geometry 2026-11-10; git-attested before this script was written)

Question: Does the Δ-window slow-decay subpopulation of GPT-2 small show elevated
within-bin excess kurtosis in the pre-softmax score distribution, relative to
non-Δ-window heads?

H1: Δ-window heads have higher mean within-bin excess kurtosis (Mann-Whitney, one-sided, p<0.05)
H2: same for median kurtosis
H3: H1 holds with L4H10 excluded
H4: Spearman ρ(Δ, mean_κ_ex) > 0

Kill conditions:
  K1: H1 fails (p ≥ 0.05) — 4-point structure hypothesis not supported at this level
  K2: H2 fails — result driven by outlier heads
  K3: H3 fails — result is one-head artifact (L4H10)
  K4: ρ < 0 — higher Δ correlates with LESS non-Gaussian structure

Run from this directory:
  cd research/physics/experiments/exp-174_4pt_cumulant_kurtosis_gpt2/
  python run.py

Produces: results/results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import GPT2LMHeadModel

# ── Protocol constants (frozen; from prereg.md) ──────────────────────────────

SEQ_LEN  = 512
N_SEQS   = 500
SEED     = 42
DEEP_LO  = 256      # census: queries i >= max(DEEP_LO, dx)
FIT_LO   = 8        # census: OLS fit over lags [FIT_LO, FIT_HI]
FIT_HI   = 256
DELTA_LO = 0.20     # Δ-window bounds
DELTA_HI = 0.30
R2_MIN   = 0.90     # R² threshold for conformal classification

# Key-position bins for kurtosis (j in [bin_lo, bin_hi) exclusive upper)
# prereg: early j∈[8,64], mid j∈[128,256], late j∈[384,511]
BINS: list[tuple[str, int, int]] = [
    ("early",  8,   65),   # j ∈ [8, 64]
    ("mid",  128,  257),   # j ∈ [128, 256]
    ("late", 384,  512),   # j ∈ [384, 511]
]

RESULTS_DIR = Path(__file__).parent / "results"

# ── Load model ────────────────────────────────────────────────────────────────

print("Loading GPT-2 small (gpt2, eager attention, float32)...")
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
model.eval()

n_layers  = model.config.n_layer    # 12
n_heads   = model.config.n_head     # 12
n_embd    = model.config.n_embd     # 768
d_head    = n_embd // n_heads       # 64
n_total   = n_layers * n_heads      # 144
SCALE     = 1.0 / np.sqrt(d_head)  # 1/√64

print(f"  {n_layers} layers × {n_heads} heads = {n_total} heads, d_head={d_head}")

# ── Generate random-token sequences ──────────────────────────────────────────

rng = np.random.default_rng(SEED)
vocab_size = model.config.vocab_size
tokens = rng.integers(0, vocab_size, size=(N_SEQS, SEQ_LEN))
all_seqs = torch.tensor(tokens, dtype=torch.long)
print(f"  {N_SEQS} random-token sequences, seq_len={SEQ_LEN}, seed={SEED}")

# ── Hook c_attn to capture Q, K ──────────────────────────────────────────────
# GPT-2 c_attn output: (batch, seq_len, 3*n_embd)
# Split: Q = out[:,:,:n_embd], K = out[:,:,n_embd:2*n_embd]

qk_cache: dict[int, torch.Tensor] = {}

def make_qk_hook(layer_idx: int):
    def hook(module, inp, out):
        qk_cache[layer_idx] = out.detach().cpu()
    return hook

hooks = []
for l in range(n_layers):
    hooks.append(
        model.transformer.h[l].attn.c_attn.register_forward_hook(make_qk_hook(l))
    )

# ── Accumulators ──────────────────────────────────────────────────────────────

# Census: sum of lag profiles (n_layers, n_heads, FIT_HI+1)
census_sum = np.zeros((n_layers, n_heads, FIT_HI + 1), dtype=np.float64)

# Kurtosis moments: per (global_head, bin)
# global_head = l * n_heads + h
n_bins  = len(BINS)
acc_n   = np.zeros((n_total, n_bins), dtype=np.int64)
acc_s1  = np.zeros((n_total, n_bins), dtype=np.float64)
acc_s2  = np.zeros((n_total, n_bins), dtype=np.float64)
acc_s3  = np.zeros((n_total, n_bins), dtype=np.float64)
acc_s4  = np.zeros((n_total, n_bins), dtype=np.float64)

# Precompute causal masks for each bin (seq_len, bin_width)
bin_masks: list[tuple[np.ndarray, np.ndarray]] = []  # (valid_i, valid_j) per bin
for _, bin_lo, bin_hi in BINS:
    bin_width = bin_hi - bin_lo
    i_arr = np.arange(SEQ_LEN)[:, None]        # (seq_len, 1)
    j_arr = bin_lo + np.arange(bin_width)[None, :]  # (1, bin_width)
    vi, vj = np.where(i_arr > j_arr)           # strictly causal: j < i
    bin_masks.append((vi, vj))

# ── Main loop ────────────────────────────────────────────────────────────────

print(f"\nProcessing {N_SEQS} sequences...")
t0 = time.time()

LAGS_CENSUS = np.arange(FIT_HI + 1)

for seq_idx in range(N_SEQS):
    if (seq_idx + 1) % 100 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (seq_idx + 1) * (N_SEQS - seq_idx - 1)
        print(f"  {seq_idx+1}/{N_SEQS}  elapsed={elapsed:.1f}s  eta={eta:.1f}s")

    x = all_seqs[seq_idx : seq_idx + 1]  # (1, SEQ_LEN)

    with torch.no_grad():
        outputs = model(x, output_attentions=True)

    for l in range(n_layers):
        # ── Census: accumulate lag profile from attention weights ─────────────
        attn_l = outputs.attentions[l][0].float().numpy()  # (n_heads, SEQ_LEN, SEQ_LEN)
        for dx in LAGS_CENSUS:
            diag = np.diagonal(attn_l, offset=-dx, axis1=-2, axis2=-1)  # (n_heads, SEQ_LEN-dx)
            k_lo = max(DEEP_LO, dx) - dx
            if k_lo < diag.shape[-1]:
                census_sum[l, :, dx] += diag[:, k_lo:].mean(axis=-1)

        # ── Kurtosis: compute pre-softmax scores and accumulate moments ───────
        # qkv: (1, SEQ_LEN, 3*n_embd) → drop batch dim
        qkv = qk_cache[l][0].float().numpy()  # (SEQ_LEN, 3*n_embd)
        q_flat = qkv[:, :n_embd]                   # (SEQ_LEN, n_embd)
        k_flat = qkv[:, n_embd : 2 * n_embd]       # (SEQ_LEN, n_embd)
        # Reshape to per-head: (n_heads, SEQ_LEN, d_head)
        q_h = q_flat.reshape(SEQ_LEN, n_heads, d_head).transpose(1, 0, 2)
        k_h = k_flat.reshape(SEQ_LEN, n_heads, d_head).transpose(1, 0, 2)

        head_slice = slice(l * n_heads, (l + 1) * n_heads)

        for bin_idx, (_, bin_lo, bin_hi) in enumerate(BINS):
            k_bin = k_h[:, bin_lo:bin_hi, :]           # (n_heads, bin_width, d_head)
            # Pre-softmax scores: (n_heads, SEQ_LEN, bin_width)
            scores = (q_h @ k_bin.transpose(0, 2, 1)) * SCALE  # type: ignore[operator]
            scores = scores.astype(np.float64)

            vi, vj = bin_masks[bin_idx]
            vs = scores[:, vi, vj]  # (n_heads, n_valid) — valid causal pairs

            n_valid = vs.shape[1]
            acc_n[head_slice, bin_idx]  += n_valid
            acc_s1[head_slice, bin_idx] += vs.sum(axis=1)
            acc_s2[head_slice, bin_idx] += (vs ** 2).sum(axis=1)
            acc_s3[head_slice, bin_idx] += (vs ** 3).sum(axis=1)
            acc_s4[head_slice, bin_idx] += (vs ** 4).sum(axis=1)

for h in hooks:
    h.remove()

elapsed_total = time.time() - t0
print(f"\nDone. {N_SEQS} sequences in {elapsed_total:.1f}s")

# ── Census: fit power laws ────────────────────────────────────────────────────

print("\nFitting census power laws...")
census_avg = census_sum / N_SEQS  # (n_layers, n_heads, FIT_HI+1)

lags    = np.arange(FIT_LO, FIT_HI + 1)
log_lag = np.log(lags.astype(float))

delta_arr = np.full((n_layers, n_heads), np.nan)
r2_arr    = np.zeros((n_layers, n_heads))

for l in range(n_layers):
    for h in range(n_heads):
        y  = census_avg[l, h, FIT_LO : FIT_HI + 1]
        ok = y > 1e-15
        if ok.sum() < 5:
            continue
        lx, ly = log_lag[ok], np.log(y[ok])
        X    = np.column_stack([np.ones_like(lx), lx])
        c, *_ = np.linalg.lstsq(X, ly, rcond=None)
        pred  = X @ c
        ss_res = float(np.sum((ly - pred) ** 2))
        ss_tot = float(np.sum((ly - ly.mean()) ** 2))
        r2_arr[l, h]    = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
        delta_arr[l, h] = float(-c[1] / 2)  # Δ = -(slope)/2

# Classify Δ-window heads
dw_mask = (r2_arr >= R2_MIN) & (delta_arr >= DELTA_LO) & (delta_arr <= DELTA_HI)
dw_idx  = np.argwhere(dw_mask)  # (n_dw, 2): [layer, head]
n_dw    = int(dw_mask.sum())

print(f"  Δ-window heads (R²≥{R2_MIN}, Δ∈[{DELTA_LO},{DELTA_HI}]): {n_dw}")
for (l, h) in dw_idx:
    print(f"    L{l}H{h}: Δ={delta_arr[l,h]:.4f}  R²={r2_arr[l,h]:.4f}")

# ── Kurtosis: compute from accumulated moments ────────────────────────────────

print("\nComputing excess kurtosis...")

kex = np.full((n_total, n_bins), np.nan)

for gh in range(n_total):
    for b in range(n_bins):
        n  = acc_n[gh, b]
        if n < 20:
            continue
        m1 = acc_s1[gh, b] / n
        m2 = acc_s2[gh, b] / n
        m3 = acc_s3[gh, b] / n
        m4 = acc_s4[gh, b] / n
        var = m2 - m1 ** 2
        if var < 1e-20:
            continue
        # Central 4th moment: E[(X-μ)⁴] = m4 - 4·μ·m3 + 6·μ²·m2 - 3·μ⁴
        mu4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
        kex[gh, b] = mu4 / (var ** 2) - 3

mean_kex   = np.nanmean(kex, axis=1)    # (n_total,)
median_kex = np.nanmedian(kex, axis=1)  # (n_total,)

# ── Statistical tests ─────────────────────────────────────────────────────────

print("\nRunning statistical tests...")

# Global head indices
dw_global  = [int(l * n_heads + h) for (l, h) in dw_idx]
nondw_set  = set(range(n_total)) - set(dw_global)
nondw_global = sorted(nondw_set)

# L4H10 global index (for H3)
L4H10_global = 4 * n_heads + 10

dw_mean   = mean_kex[dw_global]
nondw_mean = mean_kex[nondw_global]
dw_med    = median_kex[dw_global]
nondw_med = median_kex[nondw_global]

# H1: Mann-Whitney U — Δ-window mean_κ_ex > non-window (one-sided)
h1_stat, h1_p = stats.mannwhitneyu(dw_mean, nondw_mean, alternative='greater')
h1_pass = bool(h1_p < 0.05)

# H2: Mann-Whitney U — Δ-window median_κ_ex > non-window (one-sided)
h2_stat, h2_p = stats.mannwhitneyu(dw_med, nondw_med, alternative='greater')
h2_pass = bool(h2_p < 0.05)

# H3: H1 excluding L4H10
h3_applicable = L4H10_global in set(dw_global)
dw_excl = [gh for gh in dw_global if gh != L4H10_global]
if h3_applicable and len(dw_excl) > 0:
    h3_stat, h3_p = stats.mannwhitneyu(mean_kex[dw_excl], nondw_mean, alternative='greater')
    h3_pass = bool(h3_p < 0.05)
else:
    h3_stat, h3_p = np.nan, np.nan
    h3_pass = None

# H4: Spearman ρ(Δ, mean_κ_ex)
delta_flat = delta_arr.flatten()  # (n_total,)
valid = ~np.isnan(delta_flat) & ~np.isnan(mean_kex)
h4_rho, h4_p = stats.spearmanr(delta_flat[valid], mean_kex[valid])
h4_positive = bool(float(h4_rho) > 0)

# Kill flags
k1_fires = not h1_pass
k2_fires = not h2_pass
k3_fires = (not h3_pass) if h3_applicable and h3_pass is not None else False
k4_fires = not h4_positive

# ── Print summary ────────────────────────────────────────────────────────────

print()
print("=" * 72)
print("  EXP-174 RESULTS")
print("=" * 72)
print(f"  Δ-window heads: {n_dw}   Non-window heads: {len(nondw_global)}")
print(f"  Δ-window mean κ_ex (mean±std): {np.nanmean(dw_mean):.4f} ± {np.nanstd(dw_mean):.4f}")
print(f"  Non-window mean κ_ex:          {np.nanmean(nondw_mean):.4f} ± {np.nanstd(nondw_mean):.4f}")
ratio = np.nanmean(dw_mean) / np.nanmean(nondw_mean) if np.nanmean(nondw_mean) != 0 else np.nan
print(f"  Ratio Δ-window / non-window:   {ratio:.2f}×")
print()
print(f"  H1 (mean κ_ex, one-sided MW):  U={h1_stat:.0f}  p={h1_p:.6f}  →  {'CONFIRMED' if h1_pass else 'K1 FIRES'}")
print(f"  H2 (median κ_ex, one-sided MW): U={h2_stat:.0f}  p={h2_p:.6f}  →  {'CONFIRMED' if h2_pass else 'K2 FIRES'}")
if h3_applicable:
    print(f"  H3 (excl L4H10):               U={h3_stat:.0f}  p={h3_p:.6f}  →  {'CONFIRMED' if h3_pass else 'K3 FIRES'}")
else:
    print(f"  H3: not applicable (L4H10 not in Δ-window group with this fresh census)")
print(f"  H4 (Spearman ρ Δ vs κ_ex):    ρ={float(h4_rho):.4f}  p={float(h4_p):.6f}  →  {'positive' if h4_positive else 'K4 FIRES (inverted)'}")
print()
print(f"  Kill conditions fired: K1={k1_fires}  K2={k2_fires}  K3={k3_fires}  K4={k4_fires}")
print()

# Per-head details for Δ-window heads
print("  Per-Δ-window head kurtosis:")
for gh in dw_global:
    l_h, h_h = gh // n_heads, gh % n_heads
    print(f"    L{l_h}H{h_h}  Δ={delta_arr[l_h,h_h]:.4f}  mean_κ_ex={mean_kex[gh]:.4f}  "
          f"(early={kex[gh,0]:.3f}  mid={kex[gh,1]:.3f}  late={kex[gh,2]:.3f})")

print()
print(f"  Elapsed: {elapsed_total:.1f}s")
print("=" * 72)

# ── Save results ─────────────────────────────────────────────────────────────

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

results = {
    "exp": "exp-174",
    "model": "gpt2",
    "protocol": {
        "n_seqs": N_SEQS,
        "seq_len": SEQ_LEN,
        "seed": SEED,
        "deep_lo": DEEP_LO,
        "fit_lo": FIT_LO,
        "fit_hi": FIT_HI,
        "delta_lo": DELTA_LO,
        "delta_hi": DELTA_HI,
        "r2_min": R2_MIN,
        "bins": [{"name": name, "lo": lo, "hi": hi - 1} for name, lo, hi in BINS],
    },
    "census": {
        "n_dw_heads": n_dw,
        "dw_heads": [
            {"layer": int(l), "head": int(h),
             "delta": float(delta_arr[l, h]), "r2": float(r2_arr[l, h])}
            for (l, h) in dw_idx
        ],
    },
    "kurtosis_summary": {
        "dw_mean_kex_mean": float(np.nanmean(dw_mean)),
        "dw_mean_kex_std":  float(np.nanstd(dw_mean)),
        "nondw_mean_kex_mean": float(np.nanmean(nondw_mean)),
        "nondw_mean_kex_std":  float(np.nanstd(nondw_mean)),
        "ratio": float(ratio),
    },
    "h1": {"stat": float(h1_stat), "p": float(h1_p), "pass": h1_pass,
           "verdict": "CONFIRMED" if h1_pass else "K1_FIRES"},
    "h2": {"stat": float(h2_stat), "p": float(h2_p), "pass": h2_pass},
    "h3": {"applicable": h3_applicable,
           "stat": float(h3_stat) if h3_applicable else None,
           "p": float(h3_p) if h3_applicable else None,
           "pass": h3_pass},
    "h4": {"rho": float(h4_rho), "p": float(h4_p), "positive": h4_positive},
    "kill_conditions": {
        "k1_fires": k1_fires,
        "k2_fires": k2_fires,
        "k3_fires": k3_fires,
        "k4_fires": k4_fires,
    },
    "per_head": [
        {
            "global_head": gh,
            "layer": gh // n_heads,
            "head": gh % n_heads,
            "is_dw": gh in set(dw_global),
            "delta": (float(delta_arr[gh // n_heads, gh % n_heads])
                      if not np.isnan(delta_arr[gh // n_heads, gh % n_heads]) else None),
            "r2": float(r2_arr[gh // n_heads, gh % n_heads]),
            "mean_kex": float(mean_kex[gh]) if not np.isnan(mean_kex[gh]) else None,
            "median_kex": float(median_kex[gh]) if not np.isnan(median_kex[gh]) else None,
            "kex_early": float(kex[gh, 0]) if not np.isnan(kex[gh, 0]) else None,
            "kex_mid":   float(kex[gh, 1]) if not np.isnan(kex[gh, 1]) else None,
            "kex_late":  float(kex[gh, 2]) if not np.isnan(kex[gh, 2]) else None,
        }
        for gh in range(n_total)
    ],
    "elapsed_s": elapsed_total,
}

out_path = RESULTS_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"Results saved to {out_path}")
