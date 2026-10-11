"""
exp-171 — Residual stream decomposition at structural head inputs.

Pre-registration: prereg.md (committed e309533 before this script was written).

Context:
  exp-170 showed that for L3H4 and L10H8, S_wpe_raw does NOT cross zero near 511,
  yet S_abskey does. LN acting on the full residual stream creates the zero crossing.

  This experiment isolates the mechanism:
  - kbar_delta: LN applied to (residual − wpe) — accumulated writes only
  - S_delta: whether accumulated writes contribute negatively near 511
  - RMS profile: whether LN denominator has structure at position ~511

Usage:
    .venv/bin/python3 research/physics/experiments/exp-171_residual_stream_decomposition/run.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Config ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
EXPS_DIR   = SCRIPT_DIR.parent
EXP164_DIR = EXPS_DIR / "exp-164_l3h4_bilinear_decomposition"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEQ_LEN    = 1024
N_SEQS     = 100
BATCH_SIZE = 10
SEED       = 42

D_HEAD  = 64
D_MODEL = 768

HEAD_MAP = {
    "L2H1":  (2,  1),
    "L3H4":  (3,  4),
    "L5H0":  (5,  0),
    "L7H11": (7, 11),
    "L10H8": (10, 8),
}
STRUCTURAL_HEADS = list(HEAD_MAP.keys())

MQ_START, MQ_END = 256, 1024       # mean-key / mean-query pooling window
ZERO_LO, ZERO_HI = 400, 624        # ±100 window around position 511


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_random_seqs(tokenizer, n: int, seq_len: int, seed: int) -> torch.Tensor:
    rng     = np.random.default_rng(seed)
    exclude = {tokenizer.eos_token_id, tokenizer.bos_token_id,
               tokenizer.pad_token_id} - {None}
    valid   = [i for i in range(tokenizer.vocab_size) if i not in exclude]
    ids     = rng.choice(valid, size=(n, seq_len), replace=True)
    return torch.tensor(ids, dtype=torch.long)


def apply_ln(h: np.ndarray, ln_w: np.ndarray, ln_b: np.ndarray,
             eps: float = 1e-5) -> np.ndarray:
    """
    Manual LayerNorm over last dimension.
    h: [..., D_MODEL]
    Returns: [..., D_MODEL]
    """
    mean = h.mean(axis=-1, keepdims=True)
    var  = ((h - mean) ** 2).mean(axis=-1, keepdims=True)
    return (h - mean) / np.sqrt(var + eps) * ln_w + ln_b


def centered_rms(h: np.ndarray) -> np.ndarray:
    """
    Centered RMS (= LN denominator, pre gamma/beta) along last dimension.
    h: [B, T, D] → rms: [B, T]
    """
    mean = h.mean(axis=-1, keepdims=True)
    return np.sqrt(((h - mean) ** 2).mean(axis=-1))


def find_zero_crossings(profile: np.ndarray) -> list[int]:
    crossings = []
    for a in range(len(profile) - 1):
        if profile[a] * profile[a + 1] < 0:
            crossings.append(a)
    return crossings


def in_window(crossings: list[int]) -> list[int]:
    return [c for c in crossings if ZERO_LO <= c <= ZERO_HI]


# ── Data collection ────────────────────────────────────────────────────────────

def extract_model_params(model):
    """
    Extract wpe, ln_1 weights, W_K and b_K per structural head.
    Returns:
      wpe: [SEQ_LEN, D_MODEL]
      ln_params: dict{head_name: (ln_w, ln_b)} — ln_1 weight/bias for head's layer
      wk: dict{head_name: [D_MODEL, D_HEAD]}
      bk: dict{head_name: [D_HEAD]}
    """
    with torch.no_grad():
        wpe = model.transformer.wpe.weight.cpu().float().numpy()   # [1024, 768]

    ln_params = {}
    wk        = {}
    bk        = {}

    with torch.no_grad():
        for h_name, (layer, head) in HEAD_MAP.items():
            block  = model.transformer.h[layer]
            ln_mod = block.ln_1
            ln_params[h_name] = (
                ln_mod.weight.cpu().float().numpy(),   # [768]
                ln_mod.bias.cpu().float().numpy(),     # [768]
            )

            c_attn_w = block.attn.c_attn.weight  # [768, 2304]
            c_attn_b = block.attn.c_attn.bias    # [2304]

            col_lo = D_MODEL + head * D_HEAD
            col_hi = D_MODEL + (head + 1) * D_HEAD
            wk[h_name] = c_attn_w[:, col_lo:col_hi].cpu().float().numpy()  # [768, 64]
            bk[h_name] = c_attn_b[col_lo:col_hi].cpu().float().numpy()     # [64]

    return wpe, ln_params, wk, bk


def collect_residual_streams(model, tokenizer, wpe: np.ndarray):
    """
    Run all batches; for each structural head's layer, accumulate:
      kbar_full_sum[h]:  sum_b(LN(h_full) @ W_K + b_K)[a]
      kbar_delta_sum[h]: sum_b(LN(h_full - wpe) @ W_K + b_K)[a]
      rms_sum[h]:        sum_b(centered_rms(h_full))[a]

    Also accumulate the centered-rms² for each position.
    """
    _, ln_params, wk, bk = extract_model_params(model)

    # Accumulators
    kbar_full_sum  = {h: np.zeros((SEQ_LEN, D_HEAD), dtype=np.float64) for h in HEAD_MAP}
    kbar_delta_sum = {h: np.zeros((SEQ_LEN, D_HEAD), dtype=np.float64) for h in HEAD_MAP}
    rms_sum        = {h: np.zeros(SEQ_LEN, dtype=np.float64) for h in HEAD_MAP}
    seq_count      = 0

    all_seqs = make_random_seqs(tokenizer, N_SEQS, SEQ_LEN, seed=SEED)

    # Pre-cast wpe to float32
    wpe_f32 = wpe.astype(np.float32)

    model.eval()
    with torch.no_grad():
        for b0 in range(0, N_SEQS, BATCH_SIZE):
            b1  = min(b0 + BATCH_SIZE, N_SEQS)
            Bsz = b1 - b0
            batch = all_seqs[b0:b1]

            # Hook residual streams at ln_1 input for each needed layer
            captured_residual: dict[int, np.ndarray] = {}
            needed_layers = sorted({l for l, _ in HEAD_MAP.values()})
            handles = []

            for l_idx in needed_layers:
                def make_hook(l_i):
                    def hook(module, inputs, output):
                        # inputs[0]: [B, T, D_MODEL] — residual stream before ln_1
                        captured_residual[l_i] = inputs[0].cpu().float().numpy()
                    return hook
                handles.append(
                    model.transformer.h[l_idx].ln_1.register_forward_hook(make_hook(l_idx))
                )

            model(batch)

            for h in handles:
                h.remove()

            # Process each head
            for h_name, (l_idx, head) in HEAD_MAP.items():
                h_full = captured_residual[l_idx]         # [B, T, D_MODEL]
                h_delta = h_full - wpe_f32[np.newaxis, :, :]  # [B, T, D_MODEL]

                ln_w, ln_b = ln_params[h_name]
                W_K = wk[h_name]   # [D_MODEL, D_HEAD]
                b_K = bk[h_name]   # [D_HEAD]

                # Apply LN, project to key space
                k_full  = apply_ln(h_full,  ln_w, ln_b) @ W_K + b_K   # [B, T, D_HEAD]
                k_delta = apply_ln(h_delta, ln_w, ln_b) @ W_K + b_K   # [B, T, D_HEAD]

                # RMS denominator (before gamma/beta)
                rms_batch = centered_rms(h_full.astype(np.float64))    # [B, T]

                kbar_full_sum[h_name]  += k_full.astype(np.float64).sum(axis=0)
                kbar_delta_sum[h_name] += k_delta.astype(np.float64).sum(axis=0)
                rms_sum[h_name]        += rms_batch.sum(axis=0)

            seq_count += Bsz
            del captured_residual
            print(f"  Batch {b0}–{b1-1} / {N_SEQS} done", flush=True)

    kbar_full  = {h: kbar_full_sum[h]  / seq_count for h in HEAD_MAP}
    kbar_delta = {h: kbar_delta_sum[h] / seq_count for h in HEAD_MAP}
    rms_mean   = {h: rms_sum[h]        / seq_count for h in HEAD_MAP}
    return kbar_full, kbar_delta, rms_mean


# ── Analysis ───────────────────────────────────────────────────────────────────

def compute_s_abskey(kbar: np.ndarray, m_q: np.ndarray) -> np.ndarray:
    """
    S[a] = m_q · (kbar[a] − m_k) / sqrt(D_HEAD)
    where m_k = mean(kbar[MQ_START:MQ_END]).
    """
    m_k = kbar[MQ_START:MQ_END].mean(axis=0)
    dk  = kbar - m_k[np.newaxis, :]                # [T, D_HEAD]
    return (dk @ m_q) / np.sqrt(D_HEAD)            # [T]


def rms_slope(rms: np.ndarray) -> float:
    """OLS slope of rms ~ a (linear fit over all positions)."""
    a    = np.arange(len(rms), dtype=float)
    X    = np.column_stack([a, np.ones(len(a))])
    coef = np.linalg.lstsq(X, rms, rcond=None)[0]
    return float(coef[0])


def fraction_negative(s: np.ndarray) -> float:
    """Fraction of entries < 0 in s[ZERO_LO:ZERO_HI]."""
    window = s[ZERO_LO:ZERO_HI]
    return float((window < 0).mean())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 72)
    print("exp-171 — Residual stream decomposition at structural head inputs")
    print("Pre-registration: e309533 (committed before this script)")
    print("=" * 72)

    # ── Load reference qbar from exp-164 ──────────────────────────────────────
    print("\nLoading kbar_qbar.npz from exp-164 ...")
    ref_data = np.load(EXP164_DIR / "kbar_qbar.npz")
    ref_kbar = {h: ref_data[f"kbar_{h}"] for h in STRUCTURAL_HEADS}
    ref_mq   = {h: ref_data[f"qbar_{h}"][MQ_START:MQ_END].mean(axis=0)
                for h in STRUCTURAL_HEADS}
    print(f"  Loaded reference kbar and qbar for {len(STRUCTURAL_HEADS)} heads")

    # Pre-compute reference S_abskey for H2 consistency check
    ref_s_full = {}
    for h in STRUCTURAL_HEADS:
        ref_s_full[h] = compute_s_abskey(ref_kbar[h], ref_mq[h])

    # ── Load GPT-2 small ───────────────────────────────────────────────────────
    print("\nLoading GPT-2 small ...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model     = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    print(f"  Loaded. n_layers={model.config.n_layer}, n_heads={model.config.n_head}")

    # Extract wpe
    with torch.no_grad():
        wpe = model.transformer.wpe.weight.cpu().float().numpy()
    print(f"  wpe shape: {wpe.shape}")

    # ── Run forward passes ────────────────────────────────────────────────────
    print(f"\nCollecting residual streams: N={N_SEQS} sequences, seed={SEED} ...")
    kbar_full, kbar_delta, rms_mean = collect_residual_streams(model, tokenizer, wpe)

    # ── Compute S profiles ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Analysis")
    print("=" * 72)

    results_per_head = {}

    for h in STRUCTURAL_HEADS:
        m_q = ref_mq[h]   # [D_HEAD] — from exp-164

        s_full  = compute_s_abskey(kbar_full[h],  m_q)
        s_delta = compute_s_abskey(kbar_delta[h], m_q)

        # H2: consistency with exp-164
        max_dev_full = float(np.abs(s_full - ref_s_full[h]).max())
        h2_pass      = max_dev_full < 0.20

        # H1: S_delta sign change in window [ZERO_LO, ZERO_HI] for this head
        delta_crossings    = find_zero_crossings(s_delta)
        delta_in_window    = in_window(delta_crossings)
        h1_this_head       = len(delta_in_window) > 0

        # H4: fraction of positions in window where S_delta < 0
        frac_neg_delta     = fraction_negative(s_delta)

        # H3: RMS slope
        rms_slope_val      = rms_slope(rms_mean[h])
        h3_this_head       = rms_slope_val > 1e-4

        # Useful extra: S_full crossings for reference
        full_crossings     = find_zero_crossings(s_full)
        full_in_window     = in_window(full_crossings)

        # Delta profile stats in window
        delta_window_vals  = s_delta[ZERO_LO:ZERO_HI]
        delta_window_mean  = float(delta_window_vals.mean())
        delta_window_min   = float(delta_window_vals.min())
        delta_window_max   = float(delta_window_vals.max())

        # RMS profile stats
        rms_early  = float(rms_mean[h][:256].mean())
        rms_mid    = float(rms_mean[h][256:512].mean())
        rms_late   = float(rms_mean[h][512:].mean())
        rms_min    = float(rms_mean[h].min())
        rms_max    = float(rms_mean[h].max())

        print(f"\n  {h} (layer {HEAD_MAP[h][0]}):")
        print(f"    H2 (consistency): max_dev_full = {max_dev_full:.4f} → {'PASS' if h2_pass else 'FAIL (K1)'}")
        print(f"    S_full  crossings in [{ZERO_LO},{ZERO_HI}]: {full_in_window}")
        print(f"    S_delta crossings in [{ZERO_LO},{ZERO_HI}]: {delta_in_window}  → H1 {'CONFIRMED' if h1_this_head else 'not confirmed'}")
        print(f"    S_delta in window: mean={delta_window_mean:+.4f}, min={delta_window_min:+.4f}, max={delta_window_max:+.4f}")
        print(f"    Fraction S_delta < 0 in window: {frac_neg_delta:.2%}  → H4 {'CONFIRMED' if frac_neg_delta > 0.5 else 'not confirmed'}")
        print(f"    RMS profile: early={rms_early:.3f}, mid={rms_mid:.3f}, late={rms_late:.3f}")
        print(f"    RMS slope: {rms_slope_val:.6f}/pos → H3 {'CONFIRMED' if h3_this_head else 'not confirmed'}")

        results_per_head[h] = {
            "h2_max_dev_full":      round(max_dev_full, 6),
            "h2_pass":              h2_pass,
            "k1_fires":             not h2_pass,
            "s_full_crossings_all":   full_crossings[:10],
            "s_full_in_window":       full_in_window,
            "s_delta_crossings_all":  delta_crossings[:10],
            "s_delta_in_window":      delta_in_window,
            "h1_confirmed":           h1_this_head,
            "delta_window_mean":    round(delta_window_mean, 6),
            "delta_window_min":     round(delta_window_min, 6),
            "delta_window_max":     round(delta_window_max, 6),
            "frac_neg_delta_in_window": round(frac_neg_delta, 4),
            "h4_confirmed":           frac_neg_delta > 0.5,
            "rms_early_mean":       round(rms_early, 5),
            "rms_mid_mean":         round(rms_mid, 5),
            "rms_late_mean":        round(rms_late, 5),
            "rms_min":              round(rms_min, 5),
            "rms_max":              round(rms_max, 5),
            "rms_linear_slope":     round(rms_slope_val, 8),
            "h3_confirmed":           h3_this_head,
        }

    # ── Aggregate verdicts ────────────────────────────────────────────────────
    # H1: S_delta has zero crossing in window for L3H4 AND L10H8
    h1_confirmed = (results_per_head["L3H4"]["h1_confirmed"]
                    and results_per_head["L10H8"]["h1_confirmed"])

    # H2: all heads pass consistency check (K1 = any head fails)
    k1_fires = any(results_per_head[h]["k1_fires"] for h in STRUCTURAL_HEADS)

    # H3: RMS slope > threshold for ≥ 3/5 heads
    h3_n_confirmed = sum(results_per_head[h]["h3_confirmed"] for h in STRUCTURAL_HEADS)
    h3_confirmed   = h3_n_confirmed >= 3

    # H4: frac_neg_delta > 0.5 for L3H4 AND L10H8
    h4_confirmed = (results_per_head["L3H4"]["h4_confirmed"]
                    and results_per_head["L10H8"]["h4_confirmed"])

    # K2: S_delta shows no sign change in window for either L3H4 or L10H8
    k2_fires = not (results_per_head["L3H4"]["h1_confirmed"]
                    or results_per_head["L10H8"]["h1_confirmed"])

    print("\n" + "=" * 72)
    print("AGGREGATE VERDICTS")
    print("=" * 72)
    print(f"  H1 (S_delta crosses zero in [{ZERO_LO},{ZERO_HI}] for L3H4 AND L10H8): {h1_confirmed}")
    print(f"  H2 / K1 (consistency, all heads max_dev < 0.20): K1 {'FIRES' if k1_fires else 'does not fire'}")
    print(f"  H3 (RMS slope > 1e-4 for ≥3/5 heads): {h3_confirmed}  ({h3_n_confirmed}/5 confirmed)")
    print(f"  H4 (>50% negative delta in window for L3H4 AND L10H8): {h4_confirmed}")
    print(f"  K2 fires (no window crossing for either L3H4 or L10H8): {k2_fires}")

    # ── Save results ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0

    results = {
        "experiment":        "exp-171",
        "date":              "2026-10-11",
        "prereg_commit":     "e309533",
        "prereg_evidence":   "git-attested — e309533 committed and pushed before run.py was written",
        "analysis_only":     True,
        "model":             "gpt2",
        "n_seqs":            N_SEQS,
        "seq_len":           SEQ_LEN,
        "seed":              SEED,
        "zero_window":       [ZERO_LO, ZERO_HI],
        "per_head":          results_per_head,
        "verdicts": {
            "H1_s_delta_crosses_in_window_L3H4_and_L10H8": h1_confirmed,
            "H2_K1_fires_consistency_fail":                  k1_fires,
            "H3_rms_slope_n_confirmed":                      h3_n_confirmed,
            "H3_rms_increasing_confirmed":                   h3_confirmed,
            "H4_majority_negative_delta_L3H4_and_L10H8":    h4_confirmed,
            "K2_fires_no_delta_crossing_for_either_target": k2_fires,
        },
        "elapsed_s": round(elapsed, 1),
    }

    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
