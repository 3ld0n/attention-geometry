"""
exp-163: Pool-stable mechanism via kbar at SEQ_LEN=1024

Pre-registration: attention-geometry fff5fc2 (committed and pushed before this script)

Two parts:
  Part 1 (H1): Collect kbar at SEQ_LEN=1024. Compute S_abskey[h, a] for all positions.
               For each structural head × pool, fit OLS slope of S_abskey ~ log(a).
               Compare cross-pool slope range between pool-stable and pool-sensitive heads.

  Part 2 (H2): Direct census replication at SEQ_LEN=1024 with pools B/C/D.
               Measure σ for each structural head × query pool. Compute Δσ_census.

Structural Δ-window heads (from exp-007, exp-127):
  Pool-sensitive: L2H1 (layer 2, head 1), L5H0 (layer 5, head 0), L10H8 (layer 10, head 8)
  Pool-stable:    L3H4 (layer 3, head 4), L7H11 (layer 7, head 11)
  (Layer, Head indexed from 0)

Kill conditions:
  K1: cross-pool slope range does NOT discriminate pool-stable from pool-sensitive
  K2: ρ(cross-pool slope range, Δσ_exp161) ≤ 0
  K3: Δσ_census > 0.030 for pool-stable heads (replication fails)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Config ───────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent
SEQ_LEN     = 1024
N_SEQS      = 200
BATCH_SIZE  = 10        # 10 seqs × 1024 tokens; attention = ~500MB per batch
SEED        = 42

# Structural heads: (layer, head) indexed from 0
HEAD_MAP = {
    "L2H1":  (2,  1),
    "L3H4":  (3,  4),
    "L5H0":  (5,  0),
    "L7H11": (7, 11),
    "L10H8": (10, 8),
}
POOL_STABLE    = ["L3H4", "L7H11"]
POOL_SENSITIVE = ["L2H1", "L5H0", "L10H8"]

# Query/key pool definitions (inclusive lower, exclusive upper)
POOLS = {
    "B": (256, 512),
    "C": (512, 768),
    "D": (768, 1024),
}

# Census lag fit range
LAG_MIN = 8
LAG_MAX = 256

# m_q averaging window: avoid position-0 wpe artifact (found in exp-162)
MQ_POS_START = 256
MQ_POS_END   = 1024

# exp-161 pool-sensitivity values (Δσ across B/C/D)
EXP161_DELTA_SIGMA = {
    "L2H1":  0.047,
    "L3H4":  0.014,
    "L5H0":  0.052,
    "L7H11": 0.015,
    "L10H8": 0.064,
}


# ── Data collection ───────────────────────────────────────────────────────────

def make_random_seqs(tokenizer, n, seq_len, seed=42):
    rng     = np.random.default_rng(seed)
    exclude = {tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id} - {None}
    valid   = [i for i in range(tokenizer.vocab_size) if i not in exclude]
    ids     = rng.choice(valid, size=(n, seq_len), replace=True)
    return torch.tensor(ids, dtype=torch.long)


def run_all_batches(model, tokenizer):
    """
    Single-pass collection.  For each batch:
      • One forward pass with output_attentions=True, plus a c_attn hook
        to capture Q and K matrices for the 5 structural heads.
      • Accumulate kbar/qbar sums and census lag histograms.
    """
    n_layers = model.config.n_layer   # 12
    n_heads  = model.config.n_head    # 12
    d_head   = model.config.n_embd // n_heads  # 64

    # ── Accumulators ──────────────────────────────────────────────────────────
    # kbar / qbar: only need the 5 structural heads
    kbar_sum   = {h: np.zeros((SEQ_LEN, d_head), dtype=np.float64) for h in HEAD_MAP}
    qbar_sum   = {h: np.zeros((SEQ_LEN, d_head), dtype=np.float64) for h in HEAD_MAP}
    seq_count  = 0   # total sequences processed

    # census lag histograms: attn_sum[head][pool] = np.zeros(SEQ_LEN)
    attn_sum   = {h: {p: np.zeros(SEQ_LEN, dtype=np.float64) for p in POOLS}
                  for h in HEAD_MAP}
    attn_cnt   = {h: {p: np.zeros(SEQ_LEN, dtype=np.int64)   for p in POOLS}
                  for h in HEAD_MAP}

    all_seqs = make_random_seqs(tokenizer, N_SEQS, SEQ_LEN, seed=SEED)

    model.eval()
    with torch.no_grad():
        for b0 in range(0, N_SEQS, BATCH_SIZE):
            b1   = min(b0 + BATCH_SIZE, N_SEQS)
            Bsz  = b1 - b0
            batch = all_seqs[b0:b1]  # [Bsz, SEQ_LEN] — stays on CPU

            # ── Hook: capture Q and K for structural heads ─────────────────
            captured = {}   # layer_idx -> {"q": arr, "k": arr}

            def make_qk_hook(l_idx, num_heads, head_dim):
                def hook(module, inputs, output):
                    # inputs[0] is hidden_states [Bsz, T, C]
                    hidden = inputs[0]
                    B2, T, C = hidden.shape
                    qkv = module.c_attn(hidden)   # [B, T, 3C]
                    split = C                      # split_size = n_embd
                    q, k, _ = qkv.split(split, dim=-1)
                    def to_heads(x):
                        x = x.view(B2, T, num_heads, head_dim)
                        return x.permute(0, 2, 1, 3).cpu().float().numpy()
                        # -> [B, n_heads, T, d_head]
                    captured[l_idx] = {
                        "q": to_heads(q),
                        "k": to_heads(k),
                    }
                return hook

            handles = []
            needed_layers = set(l for l, _ in HEAD_MAP.values())
            for l_idx in needed_layers:
                h_mod = model.transformer.h[l_idx].attn
                handles.append(h_mod.register_forward_hook(
                    make_qk_hook(l_idx, n_heads, d_head)))

            # ── Forward pass ──────────────────────────────────────────────
            out = model(batch, output_attentions=True)

            for h in handles:
                h.remove()

            # attn_weights: list of [B, n_heads, T_q, T_k] per layer
            attn_w = [a.cpu().float().numpy() for a in out.attentions]
            del out

            # ── Accumulate kbar / qbar for structural heads ────────────────
            for hname, (l_idx, h_idx) in HEAD_MAP.items():
                cap = captured[l_idx]
                # k_batch: [B, n_heads, T, d_head] → take head h_idx → [B, T, d_head]
                k_h = cap["k"][:, h_idx, :, :]   # [B, T, d_head]
                q_h = cap["q"][:, h_idx, :, :]   # [B, T, d_head]
                kbar_sum[hname] += k_h.sum(axis=0)  # sum over B → [T, d_head]
                qbar_sum[hname] += q_h.sum(axis=0)

            seq_count += Bsz

            # ── Census accumulation for structural heads ───────────────────
            for hname, (l_idx, h_idx) in HEAD_MAP.items():
                A = attn_w[l_idx][:, h_idx, :, :]  # [B, T_q, T_k]
                for pname, (q_lo, q_hi) in POOLS.items():
                    # For each lag l in [LAG_MIN, LAG_MAX]:
                    # attention at lag l = A[:, q, q-l] for valid q in [q_lo, q_hi)
                    A_pool = A[:, q_lo:q_hi, :]   # [B, n_pool, T_k]
                    n_pool = q_hi - q_lo
                    q_abs  = np.arange(q_lo, q_hi)  # absolute query positions

                    for lag in range(LAG_MIN, LAG_MAX + 1):
                        # valid queries: q >= lag (so k = q - lag >= 0)
                        valid = q_abs >= lag
                        if not valid.any():
                            continue
                        qi = np.where(valid)[0]    # indices into pool dimension
                        ki = q_abs[valid] - lag    # key positions
                        # A_pool[:, qi, ki] → [B, n_valid]
                        attn_vals = A_pool[:, qi, ki]   # [B, n_valid]
                        attn_sum[hname][pname][lag]  += float(attn_vals.mean())
                        attn_cnt[hname][pname][lag]  += 1

            del attn_w, captured

            print(f"  Batch {b0}–{b1-1} / {N_SEQS} done", flush=True)

    # ── Normalize ────────────────────────────────────────────────────────────
    kbar = {h: kbar_sum[h] / seq_count for h in HEAD_MAP}
    qbar = {h: qbar_sum[h] / seq_count for h in HEAD_MAP}

    attn_lag_mean = {}
    for hname in HEAD_MAP:
        attn_lag_mean[hname] = {}
        for pname in POOLS:
            s = attn_sum[hname][pname]
            c = attn_cnt[hname][pname]
            with np.errstate(divide="ignore", invalid="ignore"):
                attn_lag_mean[hname][pname] = np.where(c > 0, s / c, np.nan)

    return kbar, qbar, attn_lag_mean


# ── Analysis ──────────────────────────────────────────────────────────────────

def compute_s_abskey(kbar_h, qbar_h):
    """
    S_abskey[a] = m_q · dk[a] / sqrt(d_head)
    where dk[a] = kbar[a] - m_k  (position deviation from mean key direction)
    and m_q = mean of qbar over positions MQ_POS_START..MQ_POS_END-1.

    Consistent with exp-138 and exp-162's definition:
      m_k = mean over positions of kbar[a]
      dk[a] = kbar[a] - m_k
      S_abskey[a] = m_q @ dk[a] / sqrt(d_head)
    """
    d_head   = kbar_h.shape[-1]
    m_k      = kbar_h.mean(axis=0)                             # [d_head]
    dk       = kbar_h - m_k[np.newaxis, :]                    # [SEQ_LEN, d_head]
    m_q      = qbar_h[MQ_POS_START:MQ_POS_END].mean(axis=0)   # [d_head]
    s_abskey = (dk @ m_q) / np.sqrt(d_head)                   # [SEQ_LEN]
    return s_abskey


def fit_pool_slope(s_abskey, pool_lo, pool_hi):
    """OLS slope of S_abskey[a] ~ slope * log(a) within pool."""
    a    = np.arange(pool_lo, pool_hi)
    y    = s_abskey[pool_lo:pool_hi]
    X    = np.column_stack([np.log(a), np.ones(len(a))])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(coef[0])


def fit_census_sigma(attn_lag_mean_pool, lag_min=LAG_MIN, lag_max=LAG_MAX):
    """Log-log OLS fit; returns (sigma, r2, n_pts)."""
    lags = np.arange(lag_min, lag_max + 1)
    y    = attn_lag_mean_pool[lag_min:lag_max + 1]
    mask = np.isfinite(y) & (y > 0)
    lags, y = lags[mask], y[mask]
    if len(lags) < 10:
        return np.nan, np.nan, int(len(lags))
    slope, _, r, *_ = stats.linregress(np.log(lags), np.log(y))
    return float(-slope / 2.0), float(r**2), int(len(lags))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 72)
    print("exp-163 — Pool-stable mechanism via kbar at SEQ_LEN=1024")
    print("Pre-reg commit: fff5fc2  (git-attested, pushed before this script)")
    print("=" * 72)

    # Load model
    print("\nLoading GPT-2 small (cpu, float32, eager attn)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model     = GPT2LMHeadModel.from_pretrained(
        "gpt2", attn_implementation="eager"
    )
    model.eval()
    n_heads = model.config.n_head
    d_head  = model.config.n_embd // n_heads
    print(f"  n_layers={model.config.n_layer}, n_heads={n_heads}, d_head={d_head}")

    # Collect kbar, qbar, census lag means
    print(f"\nRunning {N_SEQS} sequences at SEQ_LEN={SEQ_LEN}, batch={BATCH_SIZE}...")
    kbar, qbar, attn_lag_mean = run_all_batches(model, tokenizer)
    print(f"  kbar shape per head: {kbar['L2H1'].shape}")

    # ── Part 1: S_abskey profile and pool slopes (H1) ────────────────────────
    print("\n" + "=" * 72)
    print("PART 1 — S_abskey profile and pool slopes (H1)")
    print("=" * 72)

    h1_results = {}
    for hname in HEAD_MAP:
        s = compute_s_abskey(kbar[hname], qbar[hname])
        pool_slopes   = {}
        for pname, (p_lo, p_hi) in POOLS.items():
            pool_slopes[pname] = fit_pool_slope(s, p_lo, p_hi)
        slope_vals         = list(pool_slopes.values())
        cross_pool_range   = max(slope_vals) - min(slope_vals)

        # Sample S_abskey at key positions
        sample_positions = [0, 127, 255, 383, 511, 639, 767, 895, 1023]
        s_sample = {f"a{p}": float(s[p]) for p in sample_positions}

        h1_results[hname] = {
            "pool_slopes":      pool_slopes,
            "cross_pool_range": cross_pool_range,
            "s_abskey_sample":  s_sample,
        }

    # Print pool-slope table
    print("\n  S_abskey pool slopes  (OLS: S_abskey[a] ~ slope·log(a)):")
    hdr = f"  {'Head':<10} | {'Slope B':>9} {'Slope C':>9} {'Slope D':>9} | {'Range':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for hname in ["L2H1", "L3H4", "L5H0", "L7H11", "L10H8"]:
        r   = h1_results[hname]
        sB, sC, sD = r["pool_slopes"]["B"], r["pool_slopes"]["C"], r["pool_slopes"]["D"]
        rng = r["cross_pool_range"]
        tag = "(stable)" if hname in POOL_STABLE else "(sensitive)"
        print(f"  {hname:<10} | {sB:>9.4f} {sC:>9.4f} {sD:>9.4f} | {rng:>8.4f}  {tag}")

    # Spearman correlation with exp-161 Δσ
    head_order  = list(EXP161_DELTA_SIGMA.keys())
    ranges      = [h1_results[h]["cross_pool_range"] for h in head_order]
    exp161_vals = [EXP161_DELTA_SIGMA[h]              for h in head_order]
    rho, pval   = stats.spearmanr(ranges, exp161_vals)
    print(f"\n  Spearman ρ(cross_pool_range, Δσ_exp161) = {rho:.3f}  "
          f"(p={pval:.3f}, n=5)")

    stable_ranges    = [h1_results[h]["cross_pool_range"] for h in POOL_STABLE]
    sensitive_ranges = [h1_results[h]["cross_pool_range"] for h in POOL_SENSITIVE]
    mean_stable      = float(np.mean(stable_ranges))
    mean_sensitive   = float(np.mean(sensitive_ranges))
    print(f"\n  Mean cross-pool range — stable: {mean_stable:.4f},  "
          f"sensitive: {mean_sensitive:.4f}")

    k1_fires = (mean_stable >= mean_sensitive)
    k2_fires = (float(rho) <= 0.0)
    print(f"\n  K1 fires (range doesn't discriminate): {k1_fires}")
    print(f"  K2 fires (ρ ≤ 0):                      {k2_fires}")

    # S_abskey profile sample
    print("\n  S_abskey[a] sampled values:")
    hdr2 = (f"  {'Head':<10} | {'a=0':>7} {'a=127':>7} {'a=255':>7} "
            f"{'a=383':>7} {'a=511':>7} {'a=639':>7} {'a=767':>7} {'a=1023':>8}")
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for hname in ["L2H1", "L3H4", "L5H0", "L7H11", "L10H8"]:
        s = h1_results[hname]["s_abskey_sample"]
        print(f"  {hname:<10} | {s['a0']:>7.3f} {s['a127']:>7.3f} {s['a255']:>7.3f} "
              f"{s['a383']:>7.3f} {s['a511']:>7.3f} {s['a639']:>7.3f} "
              f"{s['a767']:>7.3f} {s['a1023']:>8.3f}")

    # ── Part 2: Direct census replication (H2) ───────────────────────────────
    print("\n" + "=" * 72)
    print("PART 2 — Direct census replication at SEQ_LEN=1024 (H2)")
    print("=" * 72)

    h2_results = {}
    for hname in HEAD_MAP:
        pool_sigmas = {}
        for pname in POOLS:
            sig, r2, n_lags = fit_census_sigma(attn_lag_mean[hname][pname])
            pool_sigmas[pname] = {"sigma": sig, "r2": r2, "n_lags": n_lags}
        sig_vals = [pool_sigmas[p]["sigma"] for p in POOLS
                    if not np.isnan(pool_sigmas[p]["sigma"])]
        dsigma = (max(sig_vals) - min(sig_vals)) if len(sig_vals) == 3 else float("nan")
        h2_results[hname] = {
            "pool_sigmas":        pool_sigmas,
            "delta_sigma_census": dsigma,
        }

    hdr3 = (f"  {'Head':<10} | {'σ_B':>8} {'R²_B':>6} | {'σ_C':>8} {'R²_C':>6} "
            f"| {'σ_D':>8} {'R²_D':>6} | {'Δσ':>8}")
    print("\n  Census σ by query pool (direct replication):")
    print(hdr3)
    print("  " + "-" * (len(hdr3) - 2))
    for hname in ["L2H1", "L3H4", "L5H0", "L7H11", "L10H8"]:
        r   = h2_results[hname]
        pB  = r["pool_sigmas"]["B"]
        pC  = r["pool_sigmas"]["C"]
        pD  = r["pool_sigmas"]["D"]
        tag = "(stable)" if hname in POOL_STABLE else "(sensitive)"
        print(f"  {hname:<10} | {pB['sigma']:>8.4f} {pB['r2']:>6.3f} | "
              f"{pC['sigma']:>8.4f} {pC['r2']:>6.3f} | "
              f"{pD['sigma']:>8.4f} {pD['r2']:>6.3f} | "
              f"{r['delta_sigma_census']:>8.4f}  {tag}")

    stable_dc    = [h2_results[h]["delta_sigma_census"] for h in POOL_STABLE]
    sensitive_dc = [h2_results[h]["delta_sigma_census"] for h in POOL_SENSITIVE]
    k3_fires     = any(d > 0.030 for d in stable_dc if not np.isnan(d))
    print(f"\n  K3 fires (pool-stable Δσ_census > 0.030): {k3_fires}")
    for h, d in zip(POOL_STABLE, stable_dc):
        print(f"    {h}: Δσ_census = {d:.4f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"Runtime: {elapsed:.0f}s")

    if not k1_fires and not k2_fires:
        h1_v = "CONFIRMED"
        h1_d = (f"Pool-stable heads lower cross-pool slope range "
                f"(mean {mean_stable:.4f} < {mean_sensitive:.4f}); ρ={rho:.3f} > 0.")
    elif k1_fires:
        h1_v = "FALSIFIED"
        h1_d = (f"K1 fires: stable mean range {mean_stable:.4f} ≥ "
                f"sensitive {mean_sensitive:.4f}.")
    else:
        h1_v = "INCONCLUSIVE"
        h1_d = f"K1 not fired but K2 fires: ρ={rho:.3f} ≤ 0."

    h2_v = ("INCONCLUSIVE (K3)" if k3_fires else "CONFIRMED (K3 not fired)")

    print(f"\n  H1 verdict: {h1_v}")
    print(f"  H1 detail:  {h1_d}")
    print(f"  H2 verdict: {h2_v}")

    # ── Save results ──────────────────────────────────────────────────────────
    def nan_to_none(x):
        if isinstance(x, float) and np.isnan(x):
            return None
        return x

    results = {
        "experiment":      "exp-163",
        "date":            "2026-09-30",
        "prereg_commit":   "fff5fc2",
        "prereg_evidence": "git-attested — registration pushed before run.py was written",
        "model":           "gpt2",
        "seq_len":         SEQ_LEN,
        "n_seqs":          N_SEQS,
        "seed":            SEED,
        "elapsed_s":       round(elapsed, 1),
        "h1": {
            hname: {
                "pool_slopes":      h1_results[hname]["pool_slopes"],
                "cross_pool_range": round(h1_results[hname]["cross_pool_range"], 6),
                "s_abskey_sample":  h1_results[hname]["s_abskey_sample"],
            }
            for hname in HEAD_MAP
        },
        "h2": {
            hname: {
                "pool_sigmas": {
                    pname: {
                        "sigma":  nan_to_none(round(v["sigma"], 6))
                                  if isinstance(v["sigma"], float) else v["sigma"],
                        "r2":     nan_to_none(round(v["r2"],   4))
                                  if isinstance(v["r2"],   float) else v["r2"],
                        "n_lags": v["n_lags"],
                    }
                    for pname, v in h2_results[hname]["pool_sigmas"].items()
                },
                "delta_sigma_census": nan_to_none(
                    round(h2_results[hname]["delta_sigma_census"], 6)
                    if isinstance(h2_results[hname]["delta_sigma_census"], float)
                    else h2_results[hname]["delta_sigma_census"]
                ),
            }
            for hname in HEAD_MAP
        },
        "spearman": {"rho": round(float(rho), 4), "pval": round(float(pval), 4)},
        "kill_conditions": {
            "K1_fires": bool(k1_fires),
            "K2_fires": bool(k2_fires),
            "K3_fires": bool(k3_fires),
        },
        "h1_verdict": h1_v,
        "h2_verdict": h2_v,
    }

    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
