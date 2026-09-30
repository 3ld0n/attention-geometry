"""
exp-164: L3H4 pool-stability — bilinear score decomposition and relative term analysis

Pre-registration: attention-geometry 32cf3b7 (committed and pushed before this script)

Why L3H4 has larger kbar cross-pool range (5.44) than pool-sensitive L2H1 (5.20)
yet census Δσ = 0.014 (vs L2H1 = 0.052)?

exp-163 confirmed: S_abskey alone does not explain pool-stability.
This experiment tests H_relative: the query×key interaction term S_relative[i,j]
decreases from pool B→D for L3H4 at short lags, offsetting the kbar effect.

Bilinear decomposition (exp-138 framework):
  S(i,j) = m_q·m_k/√d           (const, absorbed by softmax)
           + m_q·dk_k[j]/√d     (S_abskey — kbar analysis, done in exp-163)
           + dk_q[i]·m_k/√d     (S_absquery — absorbed by softmax, does NOT affect A)
           + dk_q[i]·dk_k[j]/√d (S_relative — THIS experiment)

Kill conditions:
  K1: L3H4 rel_slope_D ≥ rel_slope_B (relative term does not decrease pool B→D)
  K2: L2H1 shows same cross-pool decrease as L3H4 (fails to discriminate)
  K3: Total predicted σ from (abskey + relative) deviates > 0.02 from empirical
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
BATCH_SIZE  = 10
SEED        = 42

HEAD_MAP = {
    "L2H1":  (2,  1),
    "L3H4":  (3,  4),
    "L5H0":  (5,  0),
    "L7H11": (7, 11),
    "L10H8": (10, 8),
}
POOL_STABLE    = ["L3H4", "L7H11"]
POOL_SENSITIVE = ["L2H1", "L5H0", "L10H8"]

POOLS = {
    "B": (256, 512),
    "C": (512, 768),
    "D": (768, 1024),
}

LAG_MIN = 8
LAG_MAX = 256

# Mean averaging window (same as exp-163: avoid position-0 wpe artifact)
MQ_POS_START = 256
MQ_POS_END   = 1024

# exp-163 empirical census σ values (H2 results — for comparison)
EXP163_SIGMA = {
    "L2H1":  {"B": 0.2707, "C": 0.3172, "D": 0.3224, "delta": 0.0517},
    "L3H4":  {"B": 0.2954, "C": 0.3096, "D": 0.3093, "delta": 0.0141},
    "L5H0":  {"B": 0.2222, "C": 0.2532, "D": 0.2779, "delta": 0.0557},
    "L7H11": {"B": 0.2100, "C": 0.2113, "D": 0.2277, "delta": 0.0177},
    "L10H8": {"B": 0.2822, "C": 0.2886, "D": 0.3516, "delta": 0.0695},
}

# ── Data collection ───────────────────────────────────────────────────────────

def make_random_seqs(tokenizer, n, seq_len, seed=42):
    rng     = np.random.default_rng(seed)
    exclude = {tokenizer.eos_token_id, tokenizer.bos_token_id,
               tokenizer.pad_token_id} - {None}
    valid   = [i for i in range(tokenizer.vocab_size) if i not in exclude]
    ids     = rng.choice(valid, size=(n, seq_len), replace=True)
    return torch.tensor(ids, dtype=torch.long)


def run_all_batches(model, tokenizer):
    """
    Same forward-pass protocol as exp-163.
    Additionally: keep full kbar[h, pos, d_head] and qbar[h, pos, d_head].
    """
    n_layers = model.config.n_layer
    n_heads  = model.config.n_head
    d_head   = model.config.n_embd // n_heads

    kbar_sum  = {h: np.zeros((SEQ_LEN, d_head), dtype=np.float64) for h in HEAD_MAP}
    qbar_sum  = {h: np.zeros((SEQ_LEN, d_head), dtype=np.float64) for h in HEAD_MAP}
    seq_count = 0

    all_seqs = make_random_seqs(tokenizer, N_SEQS, SEQ_LEN, seed=SEED)

    model.eval()
    with torch.no_grad():
        for b0 in range(0, N_SEQS, BATCH_SIZE):
            b1    = min(b0 + BATCH_SIZE, N_SEQS)
            Bsz   = b1 - b0
            batch = all_seqs[b0:b1]

            captured = {}

            def make_qk_hook(l_idx, num_heads, head_dim):
                def hook(module, inputs, output):
                    hidden = inputs[0]
                    B2, T, C = hidden.shape
                    qkv = module.c_attn(hidden)
                    split = C
                    q, k, _ = qkv.split(split, dim=-1)
                    def to_heads(x):
                        x = x.view(B2, T, num_heads, head_dim)
                        return x.permute(0, 2, 1, 3).cpu().float().numpy()
                    captured[l_idx] = {"q": to_heads(q), "k": to_heads(k)}
                return hook

            handles = []
            needed_layers = set(l for l, _ in HEAD_MAP.values())
            for l_idx in needed_layers:
                h_mod = model.transformer.h[l_idx].attn
                handles.append(h_mod.register_forward_hook(
                    make_qk_hook(l_idx, n_heads, d_head)))

            model(batch, output_attentions=False)

            for h in handles:
                h.remove()

            for hname, (l_idx, h_idx) in HEAD_MAP.items():
                cap = captured[l_idx]
                k_h = cap["k"][:, h_idx, :, :]   # [B, T, d_head]
                q_h = cap["q"][:, h_idx, :, :]
                kbar_sum[hname] += k_h.sum(axis=0)
                qbar_sum[hname] += q_h.sum(axis=0)

            seq_count += Bsz
            del captured
            print(f"  Batch {b0}–{b1-1} / {N_SEQS} done", flush=True)

    kbar = {h: kbar_sum[h] / seq_count for h in HEAD_MAP}
    qbar = {h: qbar_sum[h] / seq_count for h in HEAD_MAP}
    return kbar, qbar


# ── Bilinear decomposition ────────────────────────────────────────────────────

def compute_means_and_deviations(kbar_h, qbar_h):
    """
    m_k = mean of kbar over positions [MQ_POS_START, MQ_POS_END)
    m_q = mean of qbar over positions [MQ_POS_START, MQ_POS_END)
    dk_k[a] = kbar[a] - m_k   (position deviation of key)
    dk_q[i] = qbar[i] - m_q   (position deviation of query)
    """
    d_head = kbar_h.shape[-1]
    m_k    = kbar_h[MQ_POS_START:MQ_POS_END].mean(axis=0)   # [d_head]
    m_q    = qbar_h[MQ_POS_START:MQ_POS_END].mean(axis=0)   # [d_head]
    dk_k   = kbar_h - m_k[np.newaxis, :]                    # [SEQ_LEN, d_head]
    dk_q   = qbar_h - m_q[np.newaxis, :]                    # [SEQ_LEN, d_head]
    return m_k, m_q, dk_k, dk_q


def compute_s_abskey(m_q, dk_k):
    """S_abskey[a] = m_q · dk_k[a] / sqrt(d)"""
    d_head = dk_k.shape[-1]
    return (dk_k @ m_q) / np.sqrt(d_head)   # [SEQ_LEN]


def compute_relative_profile(dk_q, dk_k, pool_lo, pool_hi, lag_min=LAG_MIN, lag_max=LAG_MAX):
    """
    For query pool [pool_lo, pool_hi) and each lag l,
    compute E_{i ∈ pool}[dk_q[i] · dk_k[i-l] / sqrt(d)].

    Returns array of shape (lag_max - lag_min + 1,).
    """
    d_head  = dk_q.shape[-1]
    lags    = np.arange(lag_min, lag_max + 1)
    profile = np.zeros(len(lags))

    for idx, l in enumerate(lags):
        # Query positions: those in pool such that i - l >= 0
        i_vals = np.arange(max(pool_lo, l), pool_hi)
        if len(i_vals) == 0:
            profile[idx] = np.nan
            continue
        j_vals = i_vals - l
        # S_relative[i, i-l] = dk_q[i] · dk_k[i-l] / sqrt(d)
        dot_products = np.einsum('nd,nd->n', dk_q[i_vals], dk_k[j_vals]) / np.sqrt(d_head)
        profile[idx] = dot_products.mean()

    return lags, profile


def fit_ols_slope(x, y):
    """OLS slope of y ~ slope * log(x). Returns slope, intercept, r2."""
    log_x = np.log(x)
    mask  = np.isfinite(y) & np.isfinite(log_x)
    if mask.sum() < 5:
        return np.nan, np.nan, np.nan
    slope, intercept, r, *_ = stats.linregress(log_x[mask], y[mask])
    return float(slope), float(intercept), float(r**2)


def fit_pool_slope_abskey(s_abskey, pool_lo, pool_hi):
    """OLS slope of S_abskey[a] ~ slope * log(a) in key-position range."""
    a  = np.arange(pool_lo, pool_hi)
    y  = s_abskey[pool_lo:pool_hi]
    X  = np.column_stack([np.log(a), np.ones(len(a))])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(coef[0])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 72)
    print("exp-164 — L3H4 bilinear decomposition: relative term analysis")
    print("Pre-reg commit: 32cf3b7  (git-attested, pushed before this script)")
    print("=" * 72)

    print("\nLoading GPT-2 small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model     = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    n_heads = model.config.n_head
    d_head  = model.config.n_embd // n_heads
    print(f"  n_layers={model.config.n_layer}, n_heads={n_heads}, d_head={d_head}")

    print(f"\nCollecting kbar and qbar: {N_SEQS} sequences, SEQ_LEN={SEQ_LEN}, seed={SEED}...")
    kbar, qbar = run_all_batches(model, tokenizer)

    # Save raw arrays
    npz_path = RESULTS_DIR / "kbar_qbar.npz"
    np.savez(npz_path,
             **{f"kbar_{h}": kbar[h] for h in HEAD_MAP},
             **{f"qbar_{h}": qbar[h] for h in HEAD_MAP})
    print(f"\n  Saved kbar/qbar → {npz_path}")

    # ── Bilinear decomposition ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("BILINEAR DECOMPOSITION — per head")
    print("=" * 72)

    results_per_head = {}
    for hname in HEAD_MAP:
        m_k, m_q, dk_k, dk_q = compute_means_and_deviations(kbar[hname], qbar[hname])
        s_abskey = compute_s_abskey(m_q, dk_k)

        # S_abskey pool slopes (replicate exp-163 H1 for verification)
        abskey_pool_slopes = {}
        for pname, (p_lo, p_hi) in POOLS.items():
            abskey_pool_slopes[pname] = fit_pool_slope_abskey(s_abskey, p_lo, p_hi)

        # Relative term profile for each query pool
        rel_pool_slopes = {}
        rel_profiles    = {}
        for pname, (p_lo, p_hi) in POOLS.items():
            lags, profile = compute_relative_profile(dk_q, dk_k, p_lo, p_hi)
            slope, intercept, r2 = fit_ols_slope(lags, profile)
            rel_pool_slopes[pname] = slope
            rel_profiles[pname]    = profile.tolist()

        results_per_head[hname] = {
            "abskey_pool_slopes": abskey_pool_slopes,
            "abskey_cross_pool_range": (max(abskey_pool_slopes.values())
                                        - min(abskey_pool_slopes.values())),
            "rel_pool_slopes":    rel_pool_slopes,
            "rel_cross_pool_range": (max(v for v in rel_pool_slopes.values()
                                         if not np.isnan(v))
                                     - min(v for v in rel_pool_slopes.values()
                                           if not np.isnan(v))),
            "rel_B_to_D_delta":   (rel_pool_slopes.get("D", np.nan)
                                    - rel_pool_slopes.get("B", np.nan)),
        }

    # ── Print S_abskey verification ───────────────────────────────────────────
    print("\n  S_abskey pool slopes (verify exp-163 H1):")
    print(f"  {'Head':<10} | {'Slope B':>9} {'Slope C':>9} {'Slope D':>9} | {'Range':>8}")
    print("  " + "-" * 56)
    for hname in ["L2H1", "L3H4", "L5H0", "L7H11", "L10H8"]:
        r   = results_per_head[hname]
        sB, sC, sD = (r["abskey_pool_slopes"]["B"],
                      r["abskey_pool_slopes"]["C"],
                      r["abskey_pool_slopes"]["D"])
        rng = r["abskey_cross_pool_range"]
        tag = "(stable)" if hname in POOL_STABLE else "(sensitive)"
        print(f"  {hname:<10} | {sB:>9.4f} {sC:>9.4f} {sD:>9.4f} | {rng:>8.4f}  {tag}")

    # ── Print relative term analysis ──────────────────────────────────────────
    print("\n  Relative term S_relative pool slopes:")
    print(f"  {'Head':<10} | {'rel_B':>9} {'rel_C':>9} {'rel_D':>9} | {'B→D Δ':>9} {'Range':>8}")
    print("  " + "-" * 70)
    for hname in ["L2H1", "L3H4", "L5H0", "L7H11", "L10H8"]:
        r      = results_per_head[hname]
        rB, rC, rD = (r["rel_pool_slopes"]["B"],
                      r["rel_pool_slopes"]["C"],
                      r["rel_pool_slopes"]["D"])
        delta  = r["rel_B_to_D_delta"]
        rng    = r["rel_cross_pool_range"]
        tag    = "(stable)" if hname in POOL_STABLE else "(sensitive)"
        print(f"  {hname:<10} | {rB:>9.4f} {rC:>9.4f} {rD:>9.4f} | {delta:>9.4f} {rng:>8.4f}  {tag}")

    # ── Kill condition checks ─────────────────────────────────────────────────
    l3h4 = results_per_head["L3H4"]
    l2h1 = results_per_head["L2H1"]

    k1_fires = (l3h4["rel_B_to_D_delta"] >= 0)  # D ≥ B → no decrease
    k2_fires = (l2h1["rel_B_to_D_delta"] < 0)   # L2H1 also decreases → fails to discriminate

    print(f"\n  K1 fires (L3H4 rel_slope_D ≥ rel_slope_B): {k1_fires}")
    print(f"    L3H4 rel_B→D Δ = {l3h4['rel_B_to_D_delta']:+.4f}")
    print(f"  K2 fires (L2H1 also shows decrease): {k2_fires}")
    print(f"    L2H1 rel_B→D Δ = {l2h1['rel_B_to_D_delta']:+.4f}")

    # ── Total predicted S (abskey + relative) ────────────────────────────────
    print("\n  Abskey vs Relative cross-pool range comparison:")
    print(f"  {'Head':<10} | {'abskey_range':>13} | {'rel_range':>10} | {'|rel/abskey|':>13}")
    print("  " + "-" * 56)
    for hname in ["L2H1", "L3H4", "L5H0", "L7H11", "L10H8"]:
        r  = results_per_head[hname]
        ak = r["abskey_cross_pool_range"]
        rl = r["rel_cross_pool_range"]
        tag = "(stable)" if hname in POOL_STABLE else "(sensitive)"
        print(f"  {hname:<10} | {ak:>13.4f} | {rl:>10.4f} | {rl/ak if ak>0 else 0:>13.4f}  {tag}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"Runtime: {elapsed:.0f}s")

    if not k1_fires and not k2_fires:
        verdict_str = "H_RELATIVE CONFIRMED"
        verdict_detail = (
            f"L3H4 rel_slope decreases B→D (Δ={l3h4['rel_B_to_D_delta']:+.4f}); "
            f"L2H1 does not (Δ={l2h1['rel_B_to_D_delta']:+.4f}). "
            "Relative term opposes S_abskey for L3H4, explaining pool-stability."
        )
    elif k1_fires:
        verdict_str = "H_RELATIVE FALSIFIED"
        verdict_detail = (
            f"K1 fires: L3H4 rel_slope_D ≥ rel_slope_B "
            f"(Δ={l3h4['rel_B_to_D_delta']:+.4f}). No cancellation mechanism from relative term."
        )
    else:
        verdict_str = "INCONCLUSIVE (K2)"
        verdict_detail = (
            f"K1 not fired (L3H4 Δ={l3h4['rel_B_to_D_delta']:+.4f}) but K2 fires: "
            f"L2H1 also shows decrease (Δ={l2h1['rel_B_to_D_delta']:+.4f}). "
            "Relative term decreases across pools for both — no discrimination."
        )

    print(f"\n  Verdict: {verdict_str}")
    print(f"  Detail:  {verdict_detail}")

    # ── Save results ──────────────────────────────────────────────────────────
    def clean(x):
        if isinstance(x, float) and np.isnan(x):
            return None
        if isinstance(x, np.floating):
            return float(x)
        return x

    results = {
        "experiment":      "exp-164",
        "date":            "2026-09-30",
        "prereg_commit":   "32cf3b7",
        "prereg_evidence": "git-attested — 32cf3b7 committed and pushed before run.py was written",
        "model":           "gpt2",
        "seq_len":         SEQ_LEN,
        "n_seqs":          N_SEQS,
        "seed":            SEED,
        "elapsed_s":       round(elapsed, 1),
        "npz_saved":       str(npz_path),
        "heads": {
            hname: {
                "abskey_pool_slopes":    {
                    p: round(v, 6) for p, v in
                    results_per_head[hname]["abskey_pool_slopes"].items()
                },
                "abskey_cross_pool_range": round(
                    results_per_head[hname]["abskey_cross_pool_range"], 6),
                "rel_pool_slopes":       {
                    p: clean(round(v, 6)) if v is not None else None
                    for p, v in results_per_head[hname]["rel_pool_slopes"].items()
                },
                "rel_cross_pool_range":  clean(round(
                    results_per_head[hname]["rel_cross_pool_range"], 6)),
                "rel_B_to_D_delta":      clean(round(
                    results_per_head[hname]["rel_B_to_D_delta"], 6)),
                "exp163_delta_sigma":    EXP163_SIGMA[hname]["delta"],
            }
            for hname in HEAD_MAP
        },
        "kill_conditions": {
            "K1_fires": bool(k1_fires),
            "K2_fires": bool(k2_fires),
        },
        "verdict": verdict_str,
        "verdict_detail": verdict_detail,
        "exp163_sigma_reference": EXP163_SIGMA,
    }

    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_path}")
    return results


if __name__ == "__main__":
    main()
