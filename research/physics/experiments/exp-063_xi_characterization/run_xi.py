"""
exp-063 — ξ characterization (Phase 1.2; SESSION_BRIEF_PHASE1 §3.1.2).

The exp-060 kill's positive residue: the winning boundary form
A = C·Δx^{−2Δ}·(1 + b·e^{−j/ξ}) decays in ABSOLUTE key position with
ξ ≈ tens of tokens (GPT-2 median ≈ 17). Nothing in the framework predicts
that scale. This is characterization, not hypothesis confirmation: fit
form (b) only; no new adversarial claims.

Parts:
  A (archived)   ξ̂ distribution from exp-060 results.json; correlations with
                 layer and Δ̂_1D.
  B (local)      GPT-2 re-run of the exp-060 pipeline saving PER-INPUT 2D
                 profiles per conformal head → per-input ξ̂ stability; plus
                 mean per-head attention entropy.
  C (cross-model) same pipeline, form (b) only, on cached models with
                 different training context windows:
                   gpt2-medium, gpt2-large (ctx 1024),
                   pythia-410m, pythia-1.4b (ctx 2048),
                   + pythia-2.8b / gpt-neo-2.7B (global layers) when cached.
                 Question: does median ξ track training context, stay
                 constant in tokens, or neither?

Protocol identical to exp-060 (frozen): L=256, 50 random inputs, seed 42,
fp32 + dtype/NaN asserts, conformal = 1D lag fit (i ≥ 32, lags [3,50)),
R² ≥ 0.90, Δ ≥ 0.05; 2D domain (i ≥ 32, Δx ∈ [3,50), A > 0); SSE in log A.

Usage:
  python run_xi.py archived          # part A
  python run_xi.py gpt2              # part B (per-input stability + entropy)
  python run_xi.py <model_key>       # part C, one model at a time (resumable)
  python run_xi.py summary           # cross-model table from accumulated results
"""

from __future__ import annotations

import gzip
import json
import sys
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
EXP060 = OUT_DIR.parent / "exp-060_bcft_adversarial" / "results.json"

SEQ_LEN = 256
N_INPUTS = 50
MIN_POS = 32
FIT_LOW, FIT_HIGH = 3, 50
RNG_SEED = 42
CONF_R2, CONF_DELTA = 0.90, 0.05
N_RESTARTS = 5

MODELS = {
    "gpt2":         {"hf": "gpt2", "ctx": 1024, "global_only": False},
    "gpt2-medium":  {"hf": "gpt2-medium", "ctx": 1024, "global_only": False},
    "gpt2-large":   {"hf": "openai-community/gpt2-large", "ctx": 1024, "global_only": False},
    "pythia-410m":  {"hf": "EleutherAI/pythia-410m", "ctx": 2048, "global_only": False},
    "pythia-1.4b":  {"hf": "EleutherAI/pythia-1.4b", "ctx": 2048, "global_only": False},
    "pythia-2.8b":  {"hf": "EleutherAI/pythia-2.8b", "ctx": 2048, "global_only": False},
    "gpt-neo-2.7B": {"hf": "EleutherAI/gpt-neo-2.7B", "ctx": 2048, "global_only": True},
}


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {"experiment": "exp-063", "protocol": "exp-060 frozen pipeline, form (b) only",
            "parts": {}}


def _save(res: dict) -> None:
    res["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    RESULTS_PATH.write_text(json.dumps(res, indent=1))


# ── shared fitting ──────────────────────────────────────────────────────────────

def fit_1d(profile: np.ndarray):
    lags = np.arange(FIT_LOW, FIT_HIGH)
    y = profile[FIT_LOW:FIT_HIGH]
    valid = y > 1e-15
    if valid.sum() < 10:
        return None, None
    lx, ly = np.log(lags[valid].astype(float)), np.log(y[valid])
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, ly, rcond=None)
    pred = X @ c
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return float(-c[1] / 2), (1 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0)


def domain_arrays(A: np.ndarray):
    ii, jj, aa = [], [], []
    for i in range(MIN_POS, A.shape[0]):
        for dx in range(FIT_LOW, min(FIT_HIGH, i + 1)):
            j = i - dx
            v = A[i, j]
            if v > 0:
                ii.append(i); jj.append(j); aa.append(v)
    i = np.array(ii, float); j = np.array(jj, float); a = np.array(aa, float)
    return j, np.log(a), i - j


EPS = 1e-30


def fit_b(A: np.ndarray, d1d: float, rng: np.random.Generator):
    """Form (b): log A = lC − 2Δ·log Δx + log(1 + b·e^{−j/ξ}). Returns dict or None."""
    j, logA, dx = domain_arrays(A)
    if len(logA) < 100:
        return None
    C1d = float(np.median(np.exp(logA) * dx ** (2 * d1d)))
    lC = np.log(max(C1d, 1e-10))

    def f(p):
        lc, d, bb, lxi = p
        return (lc - 2 * d * np.log(dx)
                + np.log(np.maximum(1 + bb * np.exp(-j / np.exp(lxi)), EPS)) - logA)

    lo = [np.log(1e-12), 0.001, -0.99, np.log(0.5)]
    hi = [np.log(1e3), 3.0, 1e3, np.log(256)]
    inits = [[lC, d1d, 0.5, np.log(2)], [lC, d1d, 2.0, np.log(10)],
             [lC, d1d, 5.0, np.log(30)]]
    best = None
    for t in range(N_RESTARTS):
        x0 = np.array(inits[t % len(inits)], float)
        if t >= len(inits):
            x0 = x0 + rng.normal(0, 0.3, size=4)
        x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        try:
            r = least_squares(f, x0, bounds=(lo, hi), method="trf", max_nfev=2000)
        except Exception:
            continue
        sse = float(np.sum(r.fun ** 2))
        if best is None or sse < best[0]:
            best = (sse, r.x)
    if best is None:
        return None
    sse, p = best
    ss_tot = float(np.sum((logA - logA.mean()) ** 2))
    return {"params": [float(v) for v in p], "xi": float(np.exp(p[3])),
            "b_amp": float(p[2]), "delta_2d": float(p[1]), "sse": sse,
            "r2_log": 1 - sse / ss_tot if ss_tot > 0 else None,
            "n_pairs": int(len(logA))}


# ── part A: archived exp-060 fits ───────────────────────────────────────────────

def part_archived() -> None:
    src = json.loads(EXP060.read_text())["families"]["gpt2"]["per_head"]
    rows = []
    for rec in src:
        if not rec.get("b"):
            continue
        lxi = rec["b"]["params"][3]
        rows.append({"layer": rec["layer"], "head": rec["head"],
                     "delta_1d": rec["delta_1d"], "xi": float(np.exp(lxi)),
                     "b_amp": rec["b"]["params"][2], "r2_log": rec["b"]["r2_log"]})
    xi = np.array([r["xi"] for r in rows])
    layer = np.array([r["layer"] for r in rows], float)
    delta = np.array([r["delta_1d"] for r in rows])
    bamp = np.array([r["b_amp"] for r in rows])
    at_bound = xi >= 255.0
    summary = {
        "n_heads": len(rows),
        "xi_median": float(np.median(xi)),
        "xi_iqr": [float(np.percentile(xi, 25)), float(np.percentile(xi, 75))],
        "xi_at_upper_bound_n": int(at_bound.sum()),
        "xi_median_interior": float(np.median(xi[~at_bound])),
        "rho_xi_layer": _sp(xi, layer), "rho_xi_delta": _sp(xi, delta),
        "rho_xi_bamp": _sp(xi, bamp),
        "rho_xi_layer_interior": _sp(xi[~at_bound], layer[~at_bound]),
        "rho_xi_delta_interior": _sp(xi[~at_bound], delta[~at_bound]),
        "per_head": rows,
    }
    res = _load_results()
    res["parts"]["A_archived_gpt2"] = summary
    _save(res)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_head"}, indent=1))


def _sp(a, b):
    rho, p = spearmanr(a, b)
    return {"rho": float(rho), "p": float(p), "n": int(len(a))}


# ── parts B/C: measurement ─────────────────────────────────────────────────────

def measure(model_key: str, keep_per_input: bool):
    spec = MODELS[model_key]
    device = _device()
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf"], dtype=torch.float32, attn_implementation="eager",
        trust_remote_code=True).to(device).eval()
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None) or cfg.n_layer
    n_heads = getattr(cfg, "num_attention_heads", None) or cfg.n_head
    vocab = cfg.vocab_size
    if spec["global_only"]:
        att_types = cfg.attention_layers
        keep_layers = [l for l in range(n_layers) if att_types[l] == "global"]
    else:
        keep_layers = list(range(n_layers))

    A_sum = np.zeros((len(keep_layers), n_heads, SEQ_LEN, SEQ_LEN), dtype=np.float64)
    A_per_input = (np.zeros((N_INPUTS, len(keep_layers), n_heads, SEQ_LEN, SEQ_LEN),
                            dtype=np.float32) if keep_per_input else None)
    entropy_sum = np.zeros((len(keep_layers), n_heads), dtype=np.float64)

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
            if keep_per_input:
                A_per_input[inp, li] = a.astype(np.float32)
            # mean row entropy over deep queries i >= MIN_POS
            rows = np.clip(a[:, MIN_POS:, :], 1e-30, None)
            entropy_sum[li] += -(rows * np.log(rows)).sum(-1).mean(-1)
        del out
        if (inp + 1) % 10 == 0:
            print(f"    forward {inp+1}/{N_INPUTS} ({time.time()-t0:.0f}s)", flush=True)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return (A_sum / N_INPUTS, A_per_input, entropy_sum / N_INPUTS,
            keep_layers, n_heads)


def run_model(model_key: str) -> None:
    keep_per_input = model_key == "gpt2"   # part B stability needs per-input 2D
    print(f"===== {model_key} (per_input={keep_per_input}) =====", flush=True)
    A_mean, A_pi, entropy, keep_layers, n_heads = measure(model_key, keep_per_input)

    conformal = []
    for li, l in enumerate(keep_layers):
        for h in range(n_heads):
            prof = np.zeros(FIT_HIGH)
            for dx in range(FIT_HIGH):
                diag = np.diagonal(A_mean[li, h], offset=-dx)
                k_lo = max(MIN_POS, dx) - dx
                prof[dx] = diag[k_lo:].mean()
            d, r2 = fit_1d(prof)
            if d is not None and r2 >= CONF_R2 and d >= CONF_DELTA:
                conformal.append((li, l, h, d, r2))
    print(f"  conformal heads: {len(conformal)}", flush=True)

    rng = np.random.default_rng(63)
    rows = []
    t0 = time.time()
    for idx, (li, l, h, d1d, r2) in enumerate(conformal):
        fb = fit_b(A_mean[li, h], d1d, rng)
        if fb is None:
            continue
        row = {"layer": int(l), "head": int(h), "delta_1d": d1d, "r2_1d": r2,
               "entropy": float(entropy[li, h]), **fb}
        if keep_per_input:
            xis = []
            for inp in range(N_INPUTS):
                f_i = fit_b(A_pi[inp, li, h].astype(np.float64), d1d, rng)
                xis.append(f_i["xi"] if f_i else None)
            ok = [x for x in xis if x is not None]
            row["xi_per_input"] = [round(x, 3) if x is not None else None for x in xis]
            row["xi_pi_median"] = float(np.median(ok)) if ok else None
            row["xi_pi_iqr"] = ([float(np.percentile(ok, 25)),
                                 float(np.percentile(ok, 75))] if ok else None)
        rows.append(row)
        if (idx + 1) % 10 == 0:
            print(f"    fitted {idx+1}/{len(conformal)} ({time.time()-t0:.0f}s)", flush=True)

    xi = np.array([r["xi"] for r in rows])
    interior = xi < 255.0
    summary = {
        "hf": MODELS[model_key]["hf"], "train_ctx": MODELS[model_key]["ctx"],
        "n_conformal": len(conformal), "n_fitted": len(rows),
        "xi_median": float(np.median(xi)) if len(xi) else None,
        "xi_iqr": ([float(np.percentile(xi, 25)), float(np.percentile(xi, 75))]
                   if len(xi) else None),
        "xi_median_interior": float(np.median(xi[interior])) if interior.any() else None,
        "n_at_bound": int((~interior).sum()),
        "rho_xi_layer": _sp(xi, np.array([r["layer"] for r in rows], float)),
        "rho_xi_delta": _sp(xi, np.array([r["delta_1d"] for r in rows])),
        "rho_xi_entropy": _sp(xi, np.array([r["entropy"] for r in rows])),
        "per_head": rows,
    }
    res = _load_results()
    res["parts"].setdefault("C_models", {})[model_key] = summary
    _save(res)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_head"}, indent=1))


def summary_table() -> None:
    res = _load_results()
    print(f"{'model':14s} {'ctx':>5s} {'n_conf':>6s} {'xi_med':>8s} {'xi_med_int':>10s} {'at_bound':>8s}")
    for k, v in res["parts"].get("C_models", {}).items():
        print(f"{k:14s} {v['train_ctx']:5d} {v['n_conformal']:6d} "
              f"{v['xi_median']:8.2f} {v['xi_median_interior']:10.2f} {v['n_at_bound']:8d}")


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "archived":
        part_archived()
    elif arg == "summary":
        summary_table()
    else:
        run_model(arg)
