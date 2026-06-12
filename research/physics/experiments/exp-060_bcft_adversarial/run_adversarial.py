"""
exp-060 — Adversarial model comparison for the BCFT image form (Phase 0 item 0.2).

Pre-registered BEFORE any fitting in
notes/2026-06-10_bcft_adversarial_preregistration.md (workspace notes, June 10, 2026).
Everything below implements that document exactly; see it for the decision rule.

Families: GPT-2 (all layers) and GPT-Neo-2.7B (global-attention layers only).
Pipeline: mean attention over 50 random-token inputs, L=256, seed 42, fp32, eager.
Conformal heads: 1D lag-profile fit (queries i ≥ 32, lags [3,50)), R² ≥ 0.90, Δ ≥ 0.05.
2D domain per head: pairs (i,j), i ≥ 32, Δx ∈ [3,50), j ≥ 0, A > 0.

Models (log-space SSE):
  M0 derived: C·Δx^{-2Δ}·(1+λ·η^{2Δ})                 k=3
  a1: C·Δx^{-2Δ} + s·1[j=0]                            k=3
  a4: C·Δx^{-2Δ} + s·1[j<=4]                           k=3
  b:  C·Δx^{-2Δ}·(1+b·e^{-j/ξ})                        k=4
  c:  C·Δx^{-2Δ}·(1+λ·η^γ)   (diagnostic, nests M0)    k=4
  d:  C1·Δx^{-2Δ1} + C2·Δx^{-2Δ2}                      k=4

Metric: AIC = n·ln(SSE/n)+2k, BIC = n·ln(SSE/n)+k·ln(n); vote share over
{M0,a1,a4,b,d}; γ-diagnostic ρ(γ̂, 2Δ̂_c); residual-vs-j structure of vote winner.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM

OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "results.json"

SEQ_LEN = 256
N_INPUTS = 50
MIN_POS = 32
FIT_LOW = 3
FIT_HIGH = 50          # exclusive
RNG_SEED = 42
CONF_R2 = 0.90
CONF_DELTA = 0.05
N_RESTARTS = 5
VOTE_MODELS = ["M0", "a1", "a4", "b", "d"]

FAMILIES = {
    "gpt2": {"hf": "gpt2", "global_only": False},
    "gpt-neo-2.7B": {"hf": "EleutherAI/gpt-neo-2.7B", "global_only": True},
}


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── measurement ────────────────────────────────────────────────────────────────

def measure_family(hf_name: str, global_only: bool, device):
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, dtype=torch.float32, attn_implementation="eager",
        trust_remote_code=True).to(device).eval()
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer")
    n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head")
    vocab = cfg.vocab_size

    if global_only:
        att_types = getattr(cfg, "attention_layers", None)
        keep_layers = [l for l in range(n_layers) if att_types[l] == "global"]
    else:
        keep_layers = list(range(n_layers))

    A_sum = np.zeros((len(keep_layers), n_heads, SEQ_LEN, SEQ_LEN), dtype=np.float64)
    lag_per_input = np.zeros((N_INPUTS, len(keep_layers), n_heads, FIT_HIGH), dtype=np.float32)

    rng = np.random.default_rng(RNG_SEED)
    t0 = time.time()
    for inp in range(N_INPUTS):
        ids = torch.tensor(rng.integers(0, vocab, size=(1, SEQ_LEN)), dtype=torch.long,
                           device=device)
        with torch.no_grad():
            out = model(ids, output_attentions=True)
        for li, l in enumerate(keep_layers):
            a_t = out.attentions[l]
            assert a_t.dtype == torch.float32, f"layer {l} dtype {a_t.dtype}"
            a = a_t[0].cpu().numpy()
            assert not np.isnan(a).any(), f"layer {l} NaN"
            A_sum[li] += a
            for dx in range(FIT_HIGH):
                diag = np.diagonal(a, offset=-dx, axis1=-2, axis2=-1)
                k_lo = max(MIN_POS, dx) - dx
                lag_per_input[inp, li, :, dx] = diag[:, k_lo:].mean(axis=-1)
        del out
        if (inp + 1) % 10 == 0:
            print(f"    forward {inp+1}/{N_INPUTS} ({time.time()-t0:.0f}s)", flush=True)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    A_mean = A_sum / N_INPUTS
    return A_mean, lag_per_input, keep_layers, n_heads


def fit_1d(profile: np.ndarray):
    lags = np.arange(FIT_LOW, FIT_HIGH)
    y = profile[FIT_LOW:FIT_HIGH]
    valid = y > 1e-15
    if valid.sum() < 10:
        return None, None
    lx, ly = np.log(lags[valid].astype(float)), np.log(y[valid])
    X = np.column_stack([np.ones_like(lx), lx])
    c, _, _, _ = np.linalg.lstsq(X, ly, rcond=None)
    pred = X @ c
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    return float(-c[1] / 2), float(r2)


# ── 2D domain and model forms ──────────────────────────────────────────────────

def domain_arrays(A: np.ndarray):
    ii, jj, aa = [], [], []
    for i in range(MIN_POS, A.shape[0]):
        for dx in range(FIT_LOW, min(FIT_HIGH, i + 1)):
            j = i - dx
            v = A[i, j]
            if v > 0:
                ii.append(i); jj.append(j); aa.append(v)
    i = np.array(ii, float); j = np.array(jj, float); a = np.array(aa, float)
    return i, j, a, np.log(a), i - j, (i - j) / (i + j)


EPS = 1e-30


def make_models(i, j, dx, eta, logA, d1d, C1d):
    """Each model: (k, param_names, inits(list of arrays), bounds(lo,hi), logf(params))."""
    lC = np.log(max(C1d, 1e-10))
    n = len(logA)

    def logf_M0(p):
        lc, d, lam = p
        return lc - 2 * d * np.log(dx) + np.log(np.maximum(1 + lam * eta ** (2 * d), EPS))

    def logf_a(k0):
        spike = (j <= k0).astype(float)
        def f(p):
            lc, d, s = p
            return np.log(np.maximum(np.exp(lc) * dx ** (-2 * d) + s * spike, EPS))
        return f

    def logf_b(p):
        lc, d, bb, lxi = p
        return lc - 2 * d * np.log(dx) + np.log(np.maximum(1 + bb * np.exp(-j / np.exp(lxi)), EPS))

    def logf_c(p):
        lc, d, lam, g = p
        return lc - 2 * d * np.log(dx) + np.log(np.maximum(1 + lam * eta ** g, EPS))

    def logf_d(p):
        lc1, d1, lc2, d2 = p
        return np.log(np.maximum(np.exp(lc1) * dx ** (-2 * d1) + np.exp(lc2) * dx ** (-2 * d2), EPS))

    LO_C, HI_C = np.log(1e-12), np.log(1e3)
    models = {
        "M0": dict(k=3, f=logf_M0,
                   inits=[[lC, d1d, 0.5], [lC, d1d, 2.0]],
                   lo=[LO_C, 0.001, -0.99], hi=[HI_C, 3.0, 1e3]),
        "a1": dict(k=3, f=logf_a(0),
                   inits=[[lC, d1d, 0.5], [lC, d1d, 2.0]],
                   lo=[LO_C, 0.001, 0.0], hi=[HI_C, 3.0, 1e3]),
        "a4": dict(k=3, f=logf_a(4),
                   inits=[[lC, d1d, 0.5], [lC, d1d, 2.0]],
                   lo=[LO_C, 0.001, 0.0], hi=[HI_C, 3.0, 1e3]),
        "b": dict(k=4, f=logf_b,
                  inits=[[lC, d1d, 0.5, np.log(2)], [lC, d1d, 2.0, np.log(10)]],
                  lo=[LO_C, 0.001, -0.99, np.log(0.5)], hi=[HI_C, 3.0, 1e3, np.log(256)]),
        "c": dict(k=4, f=logf_c,
                  inits=[[lC, d1d, 0.5, 2 * d1d], [lC, d1d, 2.0, 2 * d1d]],
                  lo=[LO_C, 0.001, -0.99, 0.01], hi=[HI_C, 3.0, 1e3, 6.0]),
        "d": dict(k=4, f=logf_d,
                  inits=[[lC, d1d, lC + np.log(0.1), 0.05],
                         [lC, d1d, lC + np.log(0.1), 0.5]],
                  lo=[LO_C, 0.001, LO_C, 0.001], hi=[HI_C, 3.0, HI_C, 3.0]),
    }
    return models, n


def fit_head(A: np.ndarray, d1d: float, rng: np.random.Generator):
    i, j, a, logA, dx, eta = domain_arrays(A)
    if len(logA) < 100:
        return None
    # crude C init from 1D: median(A · dx^{2Δ})
    C1d = float(np.median(a * dx ** (2 * d1d)))
    models, n = make_models(i, j, dx, eta, logA, d1d, C1d)

    out = {"n_pairs": int(n)}
    for name, m in models.items():
        best = None
        seeds = []
        for base in m["inits"]:
            seeds.append(np.array(base, float))
        while len(seeds) < N_RESTARTS:
            base = np.array(m["inits"][len(seeds) % len(m["inits"])], float)
            jit = base + rng.normal(0, 0.3, size=len(base))
            seeds.append(np.clip(jit, np.array(m["lo"]) + 1e-9, np.array(m["hi"]) - 1e-9))
        for x0 in seeds[:N_RESTARTS]:
            x0 = np.clip(x0, np.array(m["lo"]) + 1e-9, np.array(m["hi"]) - 1e-9)
            try:
                res = least_squares(lambda p: m["f"](p) - logA, x0,
                                    bounds=(m["lo"], m["hi"]), method="trf",
                                    max_nfev=2000)
            except Exception:
                continue
            sse = float(np.sum(res.fun ** 2))
            if best is None or sse < best[0]:
                best = (sse, res.x)
        if best is None:
            out[name] = None
            continue
        sse, p = best
        k = m["k"]
        aic = n * np.log(max(sse / n, 1e-300)) + 2 * k
        bic = n * np.log(max(sse / n, 1e-300)) + k * np.log(n)
        ss_tot = float(np.sum((logA - logA.mean()) ** 2))
        out[name] = {"sse": sse, "aic": float(aic), "bic": float(bic),
                     "r2_log": 1 - sse / ss_tot if ss_tot > 0 else None,
                     "params": [float(v) for v in p]}
    # residual-by-j bins for M0 (and computed later for vote winner)
    return out


def residual_bins(A: np.ndarray, model_name: str, params, d1d: float):
    i, j, a, logA, dx, eta = domain_arrays(A)
    C1d = float(np.median(a * dx ** (2 * d1d)))
    models, _ = make_models(i, j, dx, eta, logA, d1d, C1d)
    resid = models[model_name]["f"](np.array(params)) - logA
    bins = {"j=0": j == 0, "j=1-4": (j >= 1) & (j <= 4), "j=5-16": (j >= 5) & (j <= 16),
            "j=17-64": (j >= 17) & (j <= 64), "j>=65": j >= 65}
    return {k: float(np.mean(resid[m])) if m.any() else None for k, m in bins.items()}


# ── family pipeline ────────────────────────────────────────────────────────────

def run_family(fam: str, device) -> dict:
    spec = FAMILIES[fam]
    print(f"\n===== {fam} =====", flush=True)
    A_mean, lag_per_input, keep_layers, n_heads = measure_family(
        spec["hf"], spec["global_only"], device)

    # save per-input lag profiles (constraint §2.3)
    pp = OUT_DIR / f"per_input_lag_{fam}.json.gz"
    with gzip.open(pp, "wt") as f:
        json.dump({"experiment": "exp-060", "family": fam, "layers_kept": keep_layers,
                   "axes": ["input", "kept_layer_idx", "head", "lag 0..49"],
                   "rng_seed": RNG_SEED,
                   "profiles": np.round(lag_per_input.astype(float), 12).tolist()}, f)
    print(f"  per-input lag profiles → {pp.name}", flush=True)

    # conformal selection from the mean lag profile
    mean_lag = lag_per_input.mean(axis=0)
    conformal = []
    for li, l in enumerate(keep_layers):
        for h in range(n_heads):
            d, r2 = fit_1d(mean_lag[li, h])
            if d is not None and r2 >= CONF_R2 and d >= CONF_DELTA:
                conformal.append((li, l, h, d, r2))
    print(f"  conformal heads: {len(conformal)}", flush=True)

    rng = np.random.default_rng(60)
    per_head = []
    t0 = time.time()
    for idx, (li, l, h, d1d, r2) in enumerate(conformal):
        fits = fit_head(A_mean[li, h], d1d, rng)
        if fits is None:
            continue
        rec = {"layer": int(l), "head": int(h), "delta_1d": d1d, "r2_1d": r2, **fits}
        per_head.append(rec)
        if (idx + 1) % 25 == 0:
            print(f"    fitted {idx+1}/{len(conformal)} heads ({time.time()-t0:.0f}s)", flush=True)

    # votes
    votes = {c: {m: 0 for m in VOTE_MODELS} for c in ("aic", "bic")}
    for rec in per_head:
        for crit in ("aic", "bic"):
            vals = {m: rec[m][crit] for m in VOTE_MODELS if rec.get(m)}
            if vals:
                votes[crit][min(vals, key=vals.get)] += 1
    n_voted = sum(votes["aic"].values())
    vote_share = {c: {m: votes[c][m] / n_voted for m in VOTE_MODELS} for c in ("aic", "bic")}

    # γ diagnostic
    gam, twod_c, twod_1d = [], [], []
    for rec in per_head:
        if rec.get("c"):
            p = rec["c"]["params"]      # [lC, d, lam, gamma]
            gam.append(p[3]); twod_c.append(2 * p[1]); twod_1d.append(2 * rec["delta_1d"])
    rho_c, p_c = spearmanr(gam, twod_c)
    rho_1d, p_1d = spearmanr(gam, twod_1d)
    med_abs = float(np.median(np.abs(np.array(gam) - np.array(twod_c))))

    # residual structure of the AIC vote winner
    winner = max(vote_share["aic"], key=vote_share["aic"].get)
    res_acc = {}
    for rec in per_head:
        if not rec.get(winner):
            continue
        li = keep_layers.index(rec["layer"])
        rb = residual_bins(A_mean[li, rec["head"]], winner, rec[winner]["params"],
                           rec["delta_1d"])
        for k, v in rb.items():
            if v is not None:
                res_acc.setdefault(k, []).append(v)
    res_mean = {k: float(np.mean(v)) for k, v in res_acc.items()}

    return {
        "n_conformal": len(conformal), "n_fitted": len(per_head),
        "votes": votes, "vote_share": vote_share, "aic_vote_winner": winner,
        "gamma_diagnostic": {
            "n": len(gam),
            "rho_gamma_vs_2delta_c": float(rho_c), "p": float(p_c),
            "rho_gamma_vs_2delta_1d": float(rho_1d), "p_1d": float(p_1d),
            "median_abs_gamma_minus_2delta": med_abs,
            "gamma": [round(g, 4) for g in gam],
            "two_delta_c": [round(t, 4) for t in twod_c],
        },
        "winner_residual_by_j": res_mean,
        "per_head": per_head,
    }


def main():
    t0 = time.time()
    device = _device()
    print(f"exp-060: device={device}", flush=True)
    result = {
        "experiment": "exp-060",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "phase0_item": "0.2",
        "preregistration": "notes/2026-06-10_bcft_adversarial_preregistration.md",
        "protocol": {"seq_len": SEQ_LEN, "n_inputs": N_INPUTS, "min_pos": MIN_POS,
                     "fit_range_exclusive": [FIT_LOW, FIT_HIGH], "rng_seed": RNG_SEED,
                     "conformal": {"R2": CONF_R2, "delta_min": CONF_DELTA},
                     "n_restarts": N_RESTARTS, "loss": "SSE of log A",
                     "dtype": "float32 (verified at extraction)"},
        "families": {},
    }
    import sys
    fams = sys.argv[1:] if len(sys.argv) > 1 else list(FAMILIES)
    if RESULTS_PATH.exists():
        result = json.loads(RESULTS_PATH.read_text())
    for fam in fams:
        result["families"][fam] = run_family(fam, device)
        result["elapsed_seconds"] = time.time() - t0
        RESULTS_PATH.write_text(json.dumps(result, indent=1))
        f = result["families"][fam]
        print(json.dumps({"family": fam, "vote_share": f["vote_share"],
                          "gamma": {k: v for k, v in f["gamma_diagnostic"].items()
                                    if k not in ("gamma", "two_delta_c")},
                          "winner": f["aic_vote_winner"],
                          "winner_residual_by_j": f["winner_residual_by_j"]}, indent=1),
              flush=True)
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=OUT_DIR).decode().strip()
    except Exception:
        pass
    RESULTS_PATH.write_text(json.dumps(result, indent=1))
    print(f"\nDONE_EXP060 — wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
