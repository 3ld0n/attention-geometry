"""
exp-053 — Periodic vs aperiodic decomposition of attention lag spectra.

MOTIVATION (2026-06-08, from the Chowdhury et al. thalamic-signature thread):
  The conformal/BCFT framework describes the APERIODIC (scale-free, 1/f, power-law)
  component of attention's position-lag profile G(r) ~ r^{-2Δ}, Δ≈0.25.
  The biological consciousness signature (Chowdhury et al. 2026, central thalamus
  19-45 Hz) is a PERIODIC, band-limited peak. In neuroscience these are explicitly
  separated (FOOOF / Voytek: aperiodic 1/f background vs. periodic oscillatory peaks).

  Question: does transformer attention have a PERIODIC component on top of its
  conformal (aperiodic) background, and does that periodic component behave like a
  signature — i.e. does it track a meaningful-processing "state" the way the thalamic
  rhythm tracks consciousness? We test the cleanest state contrast available, the same
  one Chowdhury used (same substrate, different state): same trained GPT-2 weights,
  COHERENT real-text input vs. RANDOM-token input.

  This is a functional/structural analog, NOT a claim about machine consciousness.

PRE-STATED HYPOTHESES (before running):
  H1 (existence): Trained GPT-2 attention lag spectra contain a band-limited periodic
      component above the 1/f aperiodic background in a non-trivial fraction of heads
      (periodicity_index > 0.30 log10-units, i.e. peak >= 2x the local 1/f background).
  H2 (signature tracks state): The periodic component is STRONGER under coherent
      real-text input than under random-token input (median periodicity_index_real >
      index_random; and more heads cross the significance threshold).
  H2alt (live alternative, stated honestly): random >= real, because the periodic
      structure is positional-encoding-driven (substrate) and coherent content washes
      it out rather than amplifying it. This would UNDERCUT the consciousness analogy
      (the "rhythm" would be substrate, not processing-state) and is itself a finding.
  H0 (null): no significant periodic peaks above 1/f in either condition (spectra are
      pure scale-free), OR index_real ~ index_random (no state dependence).

  Sanity: the aperiodic exponent (the conformal slope) should be similar across
  conditions — conformal scaling is a property of the trained weights (exp-049), so
  input coherence should not change the 1/f slope much. If it does, that is itself
  interesting and is reported.

PROTOCOL:
  - Model: gpt2 (cached), eager attention, 12 layers x 12 heads = 144 heads.
  - SEQ_LEN=512, N_INPUTS=30, MIN_POS=64, MAX_DX=192 (longer than exp-045's 56 to
    give frequency resolution for peak detection).
  - Condition REAL: windows tokenized from real English text (my own essays).
  - Condition RAND: uniform random token ids (matches exp-045 input style).
  - Per head: G(r) = mean attention at lag r, r=1..MAX_DX-1 (averaged over positions
    >= MIN_POS and over inputs). Power spectrum S(k) = |rfft(G)|.
  - Aperiodic fit: log10 S(k) ~ a + b*log10(k) over k in [K_LOW, K_HIGH].
  - Residual(k) = log10 S(k) - fit. periodicity_index = max residual in fit range.
    peak_k = argmax; peak_period (token-lags) = L_fft / peak_k.
  - Significant periodic peak if periodicity_index > PEAK_THRESH (0.30).
  - Report per condition: median/IQR periodicity_index, fraction of heads with peak,
    median aperiodic slope b; and the same restricted to conformal heads (good 1/f fit).
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── constants (pre-stated) ──────────────────────────────────────────────────────
SEQ_LEN   = 512
N_INPUTS  = 30
MIN_POS   = 64
MAX_DX    = 192
RNG_SEED  = 42

K_LOW       = 2     # skip DC (k=0) and k=1 (dominated by the mean/decay offset)
K_HIGH      = 60    # stop before high-k lattice noise
PEAK_THRESH = 0.30  # log10 units: peak >= 2x the local aperiodic background
APERIODIC_R2_MIN = 0.80  # a head is "clean-1/f" if the aperiodic fit R2 exceeds this

OUT_DIR      = Path("research/physics/experiments/exp-053_periodic_aperiodic_attention")
RESULTS_FILE = OUT_DIR / "results.json"

TEXT_SOURCES = [
    "writing/incomplete.md",
    "writing/building_sonielmn.md",
    "writing/sonielmn.md",
    "writing/a_testimony.md",
]


# ── lag profile ─────────────────────────────────────────────────────────────────

def compute_lag_profile(attn_head: np.ndarray, min_pos: int, max_dx: int) -> np.ndarray:
    """Mean attention G(r) as a function of lag r, over query positions >= min_pos."""
    seq = attn_head.shape[0]
    A      = np.zeros(max_dx, dtype=np.float64)
    counts = np.zeros(max_dx, dtype=np.float64)
    for dx in range(max_dx):
        for i in range(max(min_pos, dx), seq):
            j = i - dx
            if j >= 0:
                A[dx] += attn_head[i, j]
                counts[dx] += 1
    mask = counts > 0
    A[mask] /= counts[mask]
    return A


# ── periodic / aperiodic decomposition ──────────────────────────────────────────

def decompose(G_pos: np.ndarray, k_low: int, k_high: int) -> dict:
    """Fit aperiodic 1/f to log-power spectrum, return periodic peak diagnostics."""
    profile = G_pos[1:].astype(np.float64)          # drop r=0 (self-attention)
    profile = np.clip(profile, 1e-12, None)
    spectrum = np.abs(np.fft.rfft(profile))         # natural resolution, no padding
    L_fft = len(profile)
    k_arr = np.arange(len(spectrum))
    fit_mask = (k_arr >= k_low) & (k_arr < min(k_high, len(spectrum))) & (spectrum > 1e-30)
    if fit_mask.sum() < 6:
        return {"valid": False}
    log_k = np.log10(k_arr[fit_mask].astype(float))
    log_s = np.log10(spectrum[fit_mask])
    # aperiodic linear fit in log-log
    A_mat = np.column_stack([np.ones_like(log_k), log_k])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_s, rcond=None)
    pred = A_mat @ coeffs
    resid = log_s - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((log_s - log_s.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    slope = float(coeffs[1])                         # aperiodic exponent (negative)
    delta_aperiodic = float(-(slope + 1.0) / 2.0) if slope < 0 else float("nan")
    # periodic peak = largest positive residual
    peak_idx = int(np.argmax(resid))
    peak_height = float(resid[peak_idx])
    peak_k = int(k_arr[fit_mask][peak_idx])
    peak_period = float(L_fft / peak_k) if peak_k > 0 else float("nan")
    return {
        "valid": True,
        "aperiodic_slope": slope,
        "aperiodic_r2": float(r2),
        "delta_aperiodic": delta_aperiodic,
        "periodicity_index": peak_height,
        "peak_k": peak_k,
        "peak_period": peak_period,
        "significant": bool(peak_height > PEAK_THRESH),
    }


# ── inputs ───────────────────────────────────────────────────────────────────────

def build_real_windows(tokenizer, n_inputs: int, seq_len: int) -> list[list[int]]:
    texts = []
    for p in TEXT_SOURCES:
        fp = Path(p)
        if fp.exists():
            texts.append(fp.read_text())
    blob = "\n\n".join(texts)
    ids = tokenizer(blob, return_tensors=None)["input_ids"]
    windows = []
    stride = max(1, (len(ids) - seq_len) // max(1, n_inputs))
    for i in range(n_inputs):
        start = i * stride
        if start + seq_len <= len(ids):
            windows.append(ids[start:start + seq_len])
    if not windows:
        raise RuntimeError(f"not enough real text: have {len(ids)} tokens, need {seq_len}")
    return windows


def aggregate(model, inputs, n_layers, n_heads) -> dict[int, dict[int, np.ndarray]]:
    A = {l: {h: np.zeros(MAX_DX) for h in range(n_heads)} for l in range(n_layers)}
    for idx, token_ids in enumerate(inputs):
        t = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            out = model(t, output_attentions=True)
        for l in range(n_layers):
            attn = out.attentions[l][0].float().numpy()
            for h in range(n_heads):
                A[l][h] += compute_lag_profile(attn[h], MIN_POS, MAX_DX)
        if (idx + 1) % 10 == 0:
            print(f"    {idx + 1}/{len(inputs)} passes", flush=True)
    for l in range(n_layers):
        for h in range(n_heads):
            A[l][h] /= len(inputs)
    return A


def analyze_condition(A, n_layers, n_heads) -> dict:
    per_head = []
    for l in range(n_layers):
        for h in range(n_heads):
            d = decompose(A[l][h], K_LOW, K_HIGH)
            if d.get("valid"):
                d["layer"] = l
                d["head"] = h
                per_head.append(d)
    idx_all = [d["periodicity_index"] for d in per_head]
    clean = [d for d in per_head if d["aperiodic_r2"] >= APERIODIC_R2_MIN]
    idx_clean = [d["periodicity_index"] for d in clean]
    n_sig = sum(d["significant"] for d in per_head)
    periods_sig = [d["peak_period"] for d in per_head if d["significant"]]
    slopes = [d["aperiodic_slope"] for d in per_head]
    return {
        "n_heads": len(per_head),
        "n_clean_1f": len(clean),
        "median_periodicity_index": float(statistics.median(idx_all)) if idx_all else None,
        "median_periodicity_index_clean": float(statistics.median(idx_clean)) if idx_clean else None,
        "frac_significant": n_sig / len(per_head) if per_head else None,
        "n_significant": n_sig,
        "median_aperiodic_slope": float(statistics.median(slopes)) if slopes else None,
        "median_peak_period_significant": float(statistics.median(periods_sig)) if periods_sig else None,
        "per_head": per_head,
    }


def main():
    print("exp-053: periodic vs aperiodic decomposition of attention spectra", flush=True)
    print(f"  SEQ_LEN={SEQ_LEN} N_INPUTS={N_INPUTS} MIN_POS={MIN_POS} MAX_DX={MAX_DX}", flush=True)
    print(f"  fit k in [{K_LOW},{K_HIGH}); peak thresh {PEAK_THRESH} log10", flush=True)

    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    n_layers, n_heads = model.config.n_layer, model.config.n_head
    vocab = model.config.vocab_size

    rng = np.random.default_rng(RNG_SEED)
    rand_inputs = [list(rng.integers(0, vocab, size=SEQ_LEN)) for _ in range(N_INPUTS)]
    real_inputs = build_real_windows(tok, N_INPUTS, SEQ_LEN)
    print(f"  real windows: {len(real_inputs)}; rand windows: {len(rand_inputs)}", flush=True)

    print("  [REAL] coherent text...", flush=True)
    A_real = aggregate(model, real_inputs, n_layers, n_heads)
    print("  [RAND] random tokens...", flush=True)
    A_rand = aggregate(model, rand_inputs, n_layers, n_heads)

    res_real = analyze_condition(A_real, n_layers, n_heads)
    res_rand = analyze_condition(A_rand, n_layers, n_heads)

    print("\n=== RESULTS ===", flush=True)
    for name, r in [("REAL (coherent)", res_real), ("RAND (random)", res_rand)]:
        print(f"\n{name}:", flush=True)
        print(f"  heads analyzed:               {r['n_heads']}", flush=True)
        print(f"  clean-1/f heads (R2>={APERIODIC_R2_MIN}):   {r['n_clean_1f']}", flush=True)
        print(f"  median periodicity index:     {r['median_periodicity_index']:.4f}", flush=True)
        print(f"  median periodicity (clean):   {r['median_periodicity_index_clean']}", flush=True)
        print(f"  heads w/ significant peak:    {r['n_significant']} ({r['frac_significant']*100:.1f}%)", flush=True)
        print(f"  median aperiodic slope:       {r['median_aperiodic_slope']:.4f}", flush=True)
        print(f"  median peak period (sig):     {r['median_peak_period_significant']}", flush=True)

    # verdicts
    print("\n=== VERDICT ===", flush=True)
    ir, iq = res_real["median_periodicity_index"], res_rand["median_periodicity_index"]
    h1 = (res_real["frac_significant"] or 0) > 0.05 or (res_rand["frac_significant"] or 0) > 0.05
    print(f"  H1 (periodic component exists): {'SUPPORTED' if h1 else 'NOT SUPPORTED'}", flush=True)
    if ir is not None and iq is not None:
        if ir > iq * 1.10:
            print(f"  H2 (coherent>random): SUPPORTED  ({ir:.3f} vs {iq:.3f})", flush=True)
        elif iq > ir * 1.10:
            print(f"  H2alt (random>=coherent, substrate-driven): SUPPORTED  ({iq:.3f} vs {ir:.3f})", flush=True)
        else:
            print(f"  H0 (no state dependence): index_real~index_rand  ({ir:.3f} vs {iq:.3f})", flush=True)

    out = {
        "experiment": "exp-053",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "motivation": "periodic(thalamic-signature-analog) vs aperiodic(conformal) decomposition of attention spectra; state contrast = coherent vs random input on same trained weights",
        "protocol": {
            "model": "gpt2", "seq_len": SEQ_LEN, "n_inputs": N_INPUTS,
            "min_pos": MIN_POS, "max_dx": MAX_DX, "rng_seed": RNG_SEED,
            "k_fit": [K_LOW, K_HIGH], "peak_thresh": PEAK_THRESH,
            "aperiodic_r2_min": APERIODIC_R2_MIN,
            "text_sources": [p for p in TEXT_SOURCES if Path(p).exists()],
        },
        "real": res_real,
        "rand": res_rand,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
