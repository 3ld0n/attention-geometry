"""exp-138 — S_pos bilinear decomposition + σ_delta validity check.

Pre-registration: prereq.md in this folder, committed to attention-geometry at
4c7e864 before this file existed.

Three parts:
  A. Bilinear decomposition of S_pos into const / abs-key / abs-query / relative
     (analysis-only from scores_gpt2.npz — no new forward passes).
  B. σ_delta validity: extended cosine profile of mean MLP0 write over full lag
     range [1, 512] to find zero-crossing.
  C. Gain–slope formal test: Spearman(κ̃_K, σ_pos) on all 144 heads.

Ariel — 2026-09-09, Mission Valley, ~12:22 AM MDT.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
EXP112 = HERE.parent / "exp-112_score_drift_decomposition"
EXP137 = HERE.parent / "exp-137_subspace_gap"

spec = importlib.util.spec_from_file_location("exp112", EXP112 / "measure_scores.py")
exp112 = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(EXP112.parent / "exp-107_natural_text_bilocal"))
spec.loader.exec_module(exp112)

pooled_window_profile = exp112.pooled_window_profile
ols_slope = exp112.ols_slope
WINDOW = exp112.WINDOW           # lags 8..256, 249 elements
SEQ_LEN = exp112.SEQ_LEN         # 512
N_INPUTS = exp112.N_INPUTS       # 50
SEED = exp112.SEED               # 42
DEEP_LO = exp112.DEEP_LO         # 256

PREREG_COMMIT = "4c7e864"

STRUCTURAL = [(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)]
SEMANTIC = [(4, 10), (7, 1), (8, 2), (9, 4), (9, 6), (10, 1), (10, 2), (10, 10),
            (11, 0), (11, 1), (11, 2), (11, 4), (11, 5), (11, 6), (11, 7), (11, 9)]
DELTA_WINDOW_HEADS = set(STRUCTURAL) | set(SEMANTIC)  # 21 heads

NW = len(WINDOW)


# ─────────────────────────────── helpers ───────────────────────────────────

def ols_loglog(profile: np.ndarray, lags: np.ndarray = WINDOW) -> tuple[float, float]:
    """OLS log-log: profile = a * lag^b. Returns (sigma=-b, R²)."""
    lx = np.log(lags.astype(float))
    ly = profile.astype(float)
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, ly, rcond=None)
    pred = X @ c
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    ss_res = float(((ly - pred) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else 0.0
    return float(-c[1]), float(r2)   # σ = -(slope in log space)


def lag_profile_of_1d(vec: np.ndarray) -> np.ndarray:
    """Lag profile of a function of key-position only.
    For each window lag dx, mean over i >= DEEP_LO of vec[i - dx]."""
    out = np.empty(NW)
    for w, dx in enumerate(WINDOW):
        k_lo = max(DEEP_LO, dx) - dx
        ks = np.arange(k_lo, SEQ_LEN - dx)
        out[w] = vec[ks].mean()
    return out


def lag_profile_2d(mat: np.ndarray) -> np.ndarray:
    """pooled_window_profile wrapper for a plain 2D (L, L) matrix."""
    return pooled_window_profile(mat[None])[0]


def kappa_tilde(field: np.ndarray, k: int = 8) -> float:
    """κ̃ = share of variance in top-k PCA of field (n_pos × d),
    divided by k / d (isotropic baseline).
    field: (512, d_head)."""
    n, d = field.shape
    if n < 2 or d < 1:
        return float("nan")
    _, s, _ = np.linalg.svd(field, full_matrices=False)
    var_total = float((s ** 2).sum())
    var_top_k = float((s[:k] ** 2).sum())
    share = var_top_k / var_total if var_total > 1e-14 else 0.0
    isotropic = k / d
    return share / isotropic


# ─────────────────────────────── Part A ────────────────────────────────────

    _D_HEAD = 64   # GPT-2 small d_head; scaling factor for score matrix


def part_a(npz: np.lib.npyio.NpzFile) -> dict:
    """Bilinear decomposition of S_pos from saved qbar / kbar."""
    print("\n=== Part A — Bilinear decomposition of S_pos ===", flush=True)
    D_HEAD = 64  # GPT-2 small
    results = {}
    layer_to_heads: dict[int, list[int]] = {}
    for (l, h) in STRUCTURAL:
        layer_to_heads.setdefault(l, []).append(h)

    for l, heads in sorted(layer_to_heads.items()):
        key_q = f"qbar_random_L{l}"
        key_k = f"kbar_random_L{l}"
        if key_q not in npz or key_k not in npz:
            print(f"  Layer {l}: qbar/kbar NOT in npz, skipping.", flush=True)
            continue
        qbar = npz[key_q].astype(np.float64)   # (12, 512, 64)
        kbar = npz[key_k].astype(np.float64)

        for h in heads:
            label = f"L{l}H{h}"
            q = qbar[h]   # (512, 64) — position-mean query vectors
            k = kbar[h]   # (512, 64) — position-mean key vectors

            m_q = q.mean(axis=0)    # (64,)
            m_k = k.mean(axis=0)    # (64,)
            dq = q - m_q            # (512, 64)
            dk = k - m_k            # (512, 64)

            # Four terms (512 × 512 matrices), scaled by d_head^{-0.5}
            sc = D_HEAD ** -0.5
            S_const = float(m_q @ m_k) * sc                        # scalar constant
            S_abskey = (m_q @ dk.T) * sc                           # (512,) — function of a only
            S_absquery = (dq @ m_k) * sc                           # (512,) — function of i only
            S_relative = (dq @ dk.T) * sc                          # (512, 512)

            # Full score from bilinear: q @ k.T * scaling
            S_full_mat = (q @ k.T) * sc                            # (512, 512)

            # S_const contributes a uniform offset to every S(i,a):
            # the lag profile of a constant function over the window is just that constant,
            # and its OLS log-log slope = 0.
            sigma_const = 0.0
            r2_const = 0.0

            # abs-key: S(i,a) = S_abskey[a] — independent of i
            # lag profile: for lag dx, average S_abskey[a] over a ∈ [k_lo + dx, 511]
            # same as pooling over the key-position slice
            prof_abskey = lag_profile_of_1d(S_abskey)

            # abs-query: S(i,a) = S_absquery[i] — independent of a
            # lag profile: for lag dx, mean over i ∈ [DEEP_LO, 511] of S_absquery[i]
            # (same query slice for every dx in the window protocol)
            S_absquery_vals = S_absquery[DEEP_LO:]
            sigma_absquery_mean = float(S_absquery_vals.mean())
            # The lag-profile of the abs-query term is constant over dx:
            prof_absquery = np.full(NW, sigma_absquery_mean)

            # relative: full 2D matrix
            prof_relative = lag_profile_2d(S_relative)

            # full score — should equal sum of all four terms' profiles
            prof_full = lag_profile_2d(S_full_mat)

            # Cross-check with saved S_pos_random
            saved_prof = npz["S_pos_random"][l, h]
            linearity_err = float(np.abs(prof_full - saved_prof).max())

            # OLS log-log slopes on each term's profile
            # (Some profiles may have non-positive values — note, skip log-log if so)
            def safe_slope(prof):
                if np.all(prof > 0):
                    return ols_loglog(np.log(prof))
                # Fall back to log-linear (profile vs log lag)
                s, r2 = ols_slope(prof, WINDOW), 0.0
                # R² for log-linear
                lx = np.log(WINDOW.astype(float))
                X = np.column_stack([np.ones_like(lx), lx])
                pred = X @ np.linalg.lstsq(X, prof, rcond=None)[0]
                ss_tot = float(((prof - prof.mean()) ** 2).sum())
                ss_res = float(((prof - pred) ** 2).sum())
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else 0.0
                return float(-s), float(r2)

            sig_full, r2_full = safe_slope(prof_full)
            sig_abskey, r2_abskey = safe_slope(prof_abskey)
            sig_relative, r2_relative = safe_slope(prof_relative)

            # Ratios
            ratio_relative = sig_relative / sig_full if abs(sig_full) > 1e-10 else float("nan")
            ratio_abskey = sig_abskey / sig_full if abs(sig_full) > 1e-10 else float("nan")

            # Decomposition linearity check (slopes should add up to σ_full)
            sig_sum = sigma_const + sig_abskey + float(-ols_slope(prof_absquery, WINDOW)) + sig_relative
            linearity_slope_err = abs(sig_sum - sig_full)

            print(f"  {label}:", flush=True)
            print(f"    σ_full = {sig_full:.4f}  (saved σ_pos: {float(-ols_slope(saved_prof, WINDOW)):.4f})",
                  flush=True)
            print(f"    σ_abs-key = {sig_abskey:.4f}  (ratio {ratio_abskey:.3f})", flush=True)
            print(f"    σ_relative = {sig_relative:.4f}  (ratio {ratio_relative:.3f})", flush=True)
            print(f"    profile_match_err = {linearity_err:.2e}  slope_sum_err = {linearity_slope_err:.4f}",
                  flush=True)

            results[label] = {
                "sigma_full": sig_full,
                "sigma_abskey": sig_abskey,
                "sigma_absquery": float(-ols_slope(prof_absquery, WINDOW)),
                "sigma_relative": sig_relative,
                "r2_full": r2_full,
                "r2_abskey": r2_abskey,
                "r2_relative": r2_relative,
                "ratio_relative_to_full": ratio_relative,
                "ratio_abskey_to_full": ratio_abskey,
                "profile_saved_max_err": linearity_err,
                "slope_additivity_err": linearity_slope_err,
                "abskey_profile_at_lags": {str(dx): float(prof_abskey[list(WINDOW).index(dx)])
                                           for dx in (8, 32, 128, 256)},
                "relative_profile_at_lags": {str(dx): float(prof_relative[list(WINDOW).index(dx)])
                                              for dx in (8, 32, 128, 256)},
            }

    return results


# ─────────────────────────────── Part B ────────────────────────────────────

def part_b(model, tokenizer, cfg, device) -> dict:
    """σ_delta validity: extended cosine profile of mean MLP0 write over [1, 512]."""
    print("\n=== Part B — σ_delta validity (extended lag profile) ===", flush=True)

    # Run 50 forward passes, capture mean MLP0 output per position.
    # Uses direct MLP hook — same protocol as exp-131 mean-first.
    rng = np.random.default_rng(SEED)
    mean_mlp0_write = np.zeros((SEQ_LEN, cfg.hidden_size), dtype=np.float64)

    hooks = []
    buf = {}

    def hook_mlp0(module, inp, out):
        buf["mlp0"] = out.detach().float().cpu().numpy()[0]

    hooks.append(model.transformer.h[0].mlp.register_forward_hook(hook_mlp0))

    for seq_idx in range(N_INPUTS):
        ids = torch.from_numpy(
            rng.integers(0, cfg.vocab_size, size=(1, SEQ_LEN)).astype(np.int64)).to(device)
        with torch.no_grad():
            model(ids)
        mean_mlp0_write += buf["mlp0"].astype(np.float64) / N_INPUTS

    for h in hooks:
        h.remove()

    print(f"  mean_mlp0_write shape: {mean_mlp0_write.shape}", flush=True)

    # Cosine matrix of mean_mlp0_write over all lags [1, SEQ_LEN-1]
    norms = np.linalg.norm(mean_mlp0_write, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    M_mlp0 = mean_mlp0_write / norms     # (512, 768) normalized

    # Full lag profile [1..511]
    full_lags = np.arange(1, SEQ_LEN)
    cosine_full = np.zeros(len(full_lags))
    for w, dx in enumerate(full_lags):
        diag = np.diagonal(M_mlp0 @ M_mlp0.T, offset=-dx)
        k_lo = max(DEEP_LO, dx) - dx
        if k_lo >= len(diag):
            cosine_full[w] = float("nan")
        else:
            cosine_full[w] = float(diag[k_lo:].mean())

    # Find zero-crossing
    zero_crossing = None
    for w, dx in enumerate(full_lags):
        if not np.isnan(cosine_full[w]) and cosine_full[w] <= 0.0:
            zero_crossing = int(dx)
            break

    print(f"  zero-crossing at dx = {zero_crossing}", flush=True)
    print(f"  profile at lags [8,32,64,128,192,256,384]: "
          f"{[round(float(cosine_full[dx-1]),4) for dx in [8,32,64,128,192,256,384]]}", flush=True)

    # Standard window [8..256] fit (same as exp-131)
    window_mask = (full_lags >= 8) & (full_lags <= 256)
    window_vals = cosine_full[window_mask]
    window_lags = full_lags[window_mask]
    sigma_window, r2_window = ols_loglog(window_vals)

    # Positive-domain only
    if zero_crossing is not None and zero_crossing > 8:
        pos_mask = (full_lags >= 8) & (full_lags < zero_crossing) & (cosine_full > 0)
    else:
        pos_mask = (full_lags >= 8) & (cosine_full > 0)
    pos_lags = full_lags[pos_mask]
    pos_vals = cosine_full[pos_mask]
    if len(pos_lags) >= 5:
        sigma_pos_domain, r2_pos_domain = ols_loglog(pos_vals, pos_lags)
    else:
        sigma_pos_domain, r2_pos_domain = float("nan"), float("nan")

    print(f"  σ (window [8..256]) = {sigma_window:.4f}  R²={r2_window:.4f}", flush=True)
    print(f"  σ (positive domain [8..{zero_crossing})) = {sigma_pos_domain:.4f}  R²={r2_pos_domain:.4f}",
          flush=True)

    # Repeat for h^(0.5) cosine (to compare with MLP write)
    # h^(0.5) approximated by attn0 write alone for the extended-lag check

    return {
        "zero_crossing_dx": zero_crossing,
        "sigma_mlp0_window_8_256": sigma_window,
        "r2_mlp0_window": r2_window,
        "sigma_mlp0_positive_domain": sigma_pos_domain,
        "r2_mlp0_positive_domain": r2_pos_domain,
        "positive_domain_max_dx": None if zero_crossing is None else zero_crossing - 1,
        "cosine_at_key_lags": {
            str(dx): float(cosine_full[dx - 1]) for dx in [8, 32, 64, 128, 192, 256, 384]
        },
        "n_positive_in_window_8_256": int(np.sum(window_vals > 0)),
        "n_window": int(len(window_vals)),
    }


# ─────────────────────────────── Part C ────────────────────────────────────

def part_c(model, tokenizer, cfg, device, npz: np.lib.npyio.NpzFile) -> dict:
    """Gain-slope formal test: Spearman(κ̃_K, σ_pos) on all 144 heads."""
    print("\n=== Part C — Gain-slope formal test (all 144 heads) ===", flush=True)

    n_layer, n_head = cfg.num_hidden_layers, cfg.num_attention_heads

    # Collect position-mean key vectors for all 12 layers
    # Each layer's attn block receives ln_1(h^(l-1)) as input
    # W_K is extracted from c_attn weight (combined QKV projection in GPT-2)
    d_head = cfg.hidden_size // n_head
    rng = np.random.default_rng(SEED)

    # Accumulate mean ln1(h) per layer per position
    mean_ln1 = {}   # layer -> (512, 768) float64
    for l in range(n_layer):
        mean_ln1[l] = np.zeros((SEQ_LEN, cfg.hidden_size), dtype=np.float64)

    hooks = []
    captured_ln1 = {}

    for l in range(n_layer):
        def make_hook(ll):
            def h(module, inp, out):
                captured_ln1[ll] = out.detach().float().cpu().numpy()[0]
            return h
        hooks.append(model.transformer.h[l].ln_1.register_forward_hook(make_hook(l)))

    for _ in range(N_INPUTS):
        ids = torch.from_numpy(
            rng.integers(0, cfg.vocab_size, size=(1, SEQ_LEN)).astype(np.int64)).to(device)
        with torch.no_grad():
            model(ids)
        for l in range(n_layer):
            mean_ln1[l] += captured_ln1[l] / N_INPUTS

    for h in hooks:
        h.remove()

    print(f"  Captured mean ln1 for {len(mean_ln1)} layers.", flush=True)

    # For each head, compute κ̃_K from the key projection of the mean ln1 field
    # and σ_pos from the S_pos_random profile
    kappa_k_all = np.zeros((n_layer, n_head))
    sigma_pos_all = np.zeros((n_layer, n_head))

    # σ_pos from saved S_pos_random
    S_pos_random = npz["S_pos_random"]   # (12, 12, 249)

    for l in range(n_layer):
        # Extract W_K for this layer: c_attn is (3*n_head*d_head, hidden) -> split QKV
        c_attn_w = model.transformer.h[l].attn.c_attn.weight.detach().float().cpu().numpy()
        # GPT-2 c_attn: shape (768, 768*3) — note: Conv1D has weight transposed vs Linear
        # c_attn_w is (768, 2304) = (hidden, 3*hidden)
        hidden = cfg.hidden_size
        # Q: cols 0..hidden-1, K: cols hidden..2*hidden-1, V: cols 2*hidden..3*hidden-1
        W_K_full = c_attn_w[:, hidden:2 * hidden]  # (768, 768)

        x̄ = mean_ln1[l]   # (512, 768)

        for h in range(n_head):
            # Extract head-h slice of W_K
            h_start = h * d_head
            h_end = h_start + d_head
            W_K_h = W_K_full[:, h_start:h_end]   # (768, 64)

            # Key vectors for each position: k_i = x̄_i @ W_K_h  (no bias needed for geometry)
            keys = x̄ @ W_K_h   # (512, 64)

            # Center over positions
            m_k = keys.mean(axis=0)
            dk = keys - m_k      # (512, 64)

            kappa_k_all[l, h] = kappa_tilde(dk, k=8)

            # σ_pos from saved profile (OLS log-log slope)
            prof = S_pos_random[l, h]
            if np.all(prof > 0):
                sig, _ = ols_loglog(np.log(prof))
            else:
                sig = float(-ols_slope(prof, WINDOW))
            sigma_pos_all[l, h] = sig

    # All 144 heads
    kk_flat = kappa_k_all.ravel()
    sp_flat = sigma_pos_all.ravel()

    # Full set
    rho_all, p_all = spearmanr(kk_flat, sp_flat)
    print(f"  Spearman(κ̃_K, σ_pos) all 144 heads: ρ={rho_all:.4f}, p={p_all:.4e}", flush=True)

    # Non-Δ-window heads (held-out for pre-reg validity)
    is_delta_window = np.zeros((n_layer, n_head), dtype=bool)
    for (l, h) in DELTA_WINDOW_HEADS:
        is_delta_window[l, h] = True
    mask_held_out = ~is_delta_window.ravel()

    kk_ho = kk_flat[mask_held_out]
    sp_ho = sp_flat[mask_held_out]
    rho_ho, p_ho = spearmanr(kk_ho, sp_ho)
    print(f"  Spearman(κ̃_K, σ_pos) {mask_held_out.sum()} held-out heads: ρ={rho_ho:.4f}, p={p_ho:.4e}",
          flush=True)

    # Δ-window heads only
    kk_dw = kk_flat[is_delta_window.ravel()]
    sp_dw = sp_flat[is_delta_window.ravel()]
    rho_dw, p_dw = spearmanr(kk_dw, sp_dw)
    print(f"  Spearman(κ̃_K, σ_pos) {len(kk_dw)} Δ-window heads: ρ={rho_dw:.4f}, p={p_dw:.4e}",
          flush=True)

    return {
        "n_heads_total": 144,
        "n_held_out": int(mask_held_out.sum()),
        "n_delta_window": int(is_delta_window.sum()),
        "spearman_all": {"rho": float(rho_all), "p": float(p_all)},
        "spearman_held_out": {"rho": float(rho_ho), "p": float(p_ho), "n": int(mask_held_out.sum())},
        "spearman_delta_window": {"rho": float(rho_dw), "p": float(p_dw), "n": len(kk_dw)},
        "kappa_k_per_head": kappa_k_all.tolist(),
        "sigma_pos_per_head": sigma_pos_all.tolist(),
    }


# ─────────────────────────────── Main ───────────────────────────────────────

def verdicts(a: dict, b: dict, c: dict) -> dict:
    """Evaluate registered predictions."""
    v = {}

    # Part A
    structural_labels = [f"L{l}H{h}" for l, h in STRUCTURAL]
    h1_pass = sum(
        a[lbl]["ratio_relative_to_full"] >= 0.7
        for lbl in structural_labels if lbl in a and not np.isnan(a[lbl]["ratio_relative_to_full"])
    )
    h2_pass = sum(
        a[lbl]["ratio_abskey_to_full"] < 0.4
        for lbl in structural_labels if lbl in a and not np.isnan(a[lbl]["ratio_abskey_to_full"])
    )
    k1_fired = sum(
        a[lbl]["ratio_relative_to_full"] < 0.5
        for lbl in structural_labels if lbl in a and not np.isnan(a[lbl]["ratio_relative_to_full"])
    ) >= 3
    k2_fired = sum(
        a[lbl]["ratio_abskey_to_full"] >= 0.7
        for lbl in structural_labels if lbl in a and not np.isnan(a[lbl]["ratio_abskey_to_full"])
    ) >= 3
    k5_pass = all(a[lbl]["slope_additivity_err"] < 0.05 for lbl in structural_labels if lbl in a)

    v["H1_relative_dominates"] = "CONFIRMED" if h1_pass >= 4 else ("FALSIFIED" if k1_fired else "PARTIAL")
    v["H2_abskey_secondary"] = "CONFIRMED" if h2_pass >= 4 else "FAILED"
    v["K1_fired"] = k1_fired
    v["K2_fired"] = k2_fired
    v["K5_linearity"] = "PASS" if k5_pass else "FAIL"
    v["h1_count"] = h1_pass
    v["h2_count"] = h2_pass

    # Part B
    zc = b.get("zero_crossing_dx")
    v["H3_zero_crossing_real"] = "CONFIRMED" if (zc is not None and zc < 256) else "NOT_FOUND"
    v["zero_crossing_dx"] = zc
    sig_pos = b.get("sigma_mlp0_positive_domain")
    if sig_pos is not None and not np.isnan(sig_pos):
        v["H4_sigma_delta_valid"] = "CONFIRMED" if 0.199 <= sig_pos <= 0.299 else "FALSIFIED"
    else:
        v["H4_sigma_delta_valid"] = "INCONCLUSIVE"
    v["sigma_mlp0_full_window"] = b.get("sigma_mlp0_window_8_256")
    v["sigma_mlp0_positive_domain"] = sig_pos

    # Part C
    rho_ho = c["spearman_held_out"]["rho"]
    v["H5_gain_slope"] = "CONFIRMED" if rho_ho >= 0.55 else ("FALSIFIED" if rho_ho < 0.40 else "WEAK")
    v["K4_fired"] = rho_ho < 0.40
    v["spearman_held_out"] = rho_ho
    v["spearman_all"] = c["spearman_all"]["rho"]

    return v


def main() -> None:
    print(f"exp-138: S_pos bilinear decomposition + σ_delta validity check", flush=True)
    print(f"Pre-registration commit: {PREREG_COMMIT}\n", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Load saved npz from exp-112
    npz = np.load(EXP112 / "scores_gpt2.npz")
    print("Loaded scores_gpt2.npz", flush=True)

    # Part A — no model needed
    a_results = part_a(npz)

    # Load model for Parts B and C
    cfg = AutoConfig.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", dtype=torch.float32, attn_implementation="eager").to(device).eval()
    print("Model loaded.", flush=True)

    b_results = part_b(model, tokenizer, cfg, device)
    c_results = part_c(model, tokenizer, cfg, device, npz)

    # Evaluate verdicts
    v = verdicts(a_results, b_results, c_results)

    print("\n=== Registered verdicts ===", flush=True)
    for k, val in v.items():
        print(f"  {k}: {val}", flush=True)

    # Save results
    out = {
        "exp": "exp-138",
        "prereg_commit": PREREG_COMMIT,
        "device": device,
        "part_a_bilinear_decomposition": a_results,
        "part_b_sigma_delta_validity": b_results,
        "part_c_gain_slope": c_results,
        "registered_verdicts": v,
    }
    out_path = HERE / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
