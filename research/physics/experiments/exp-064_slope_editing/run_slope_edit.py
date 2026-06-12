"""
exp-064 — Causal slope-editing pilot (Phase 1.4 / RESEARCH_PLAN 2.3).

Pre-registration of record: notes/2026-06-11_slope_editing_preregistration.md
(committed 2026-06-11 BEFORE any weight edit). This script implements it exactly.

Stages (run in order; `predict` writes the committed sign vector BEFORE `edit`):
  python run_slope_edit.py predict   # head selection + per-head predicted signs
  python run_slope_edit.py edit      # edits + re-measurement (Δ̂, score slope, valley)
  python run_slope_edit.py evaluate  # decision statistics per prereg §2

Edit: W_Q_h <- (I + (κ−1)·P_U)·W_Q_h, P_U = top-8 PCs of position-mean ln_1 output
(50 random inputs, L=256, seed 42). κ ∈ {0.5, 1.0 (sham), 1.5, 2.0}.
Selection: the 8 conformal heads with highest 1D R² among Δ̂ ∈ [0.15, 0.35]
in the exp-060 GPT-2 conformal list.
Measurement: exp-060 pipeline constants (L=256, 50 inputs, seed 42, MIN_POS=32,
lags [3,50), fp32 asserts). Valley: deep queries i ≥ 3L/4 = 192, key thirds
(exp-061 geometry ported to L=256), valley = 1 − M/max(S, E).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import GPT2LMHeadModel

OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "results.json"
EXP060 = OUT_DIR.parent / "exp-060_bcft_adversarial" / "results.json"

SEQ_LEN = 256
N_INPUTS = 50
MIN_POS = 32
FIT_LOW, FIT_HIGH = 3, 50
RNG_SEED = 42
KAPPAS = [0.5, 1.0, 1.5, 2.0]
N_PC = 8
N_SELECT = 8
DELTA_RANGE = (0.15, 0.35)
HEAD_DIM = 64

# valley geometry, exp-061 ported to L = 256
Q_LO = 3 * SEQ_LEN // 4                      # 192
THIRD = SEQ_LEN // 3                         # 85
J_START = np.arange(1, THIRD)                # 1..84
J_MID = np.arange(THIRD, 2 * THIRD)          # 85..169
J_END = np.arange(2 * THIRD, Q_LO)           # 170..191


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _load() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {"experiment": "exp-064",
            "preregistration": "notes/2026-06-11_slope_editing_preregistration.md"}


def _save(res: dict) -> None:
    res["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    RESULTS_PATH.write_text(json.dumps(res, indent=1))


# ── exp-061 implied-valley machinery (L = 256 geometry) ─────────────────────────

def masses(delta: float) -> dict:
    out = {}
    qq = np.arange(Q_LO, SEQ_LEN)[:, None].astype(float)
    for name, jr in [("S", J_START), ("M", J_MID), ("E", J_END)]:
        jj = jr[None, :].astype(float)
        out[name + "0"] = float(((qq - jj) ** (-2.0 * delta)).mean())
        out[name + "1"] = float(((qq + jj) ** (-2.0 * delta)).mean())
    return out


def model_valley(delta: float, lam: float) -> float:
    m = masses(delta)
    S = m["S0"] + lam * m["S1"]
    M = m["M0"] + lam * m["M1"]
    E = m["E0"] + lam * m["E1"]
    return 1.0 - M / max(S, E)


# ── stage 1: predict (BEFORE any edit) ──────────────────────────────────────────

def predict() -> None:
    src = json.loads(EXP060.read_text())["families"]["gpt2"]["per_head"]
    cands = [r for r in src if DELTA_RANGE[0] <= r["delta_1d"] <= DELTA_RANGE[1]]
    cands.sort(key=lambda r: -r["r2_1d"])
    sel = cands[:N_SELECT]
    rows = []
    for r in sel:
        lam = r["M0"]["params"][2] if r.get("M0") else 0.0
        d0 = r["delta_1d"]
        v0 = model_valley(d0, lam)
        preds = {}
        for k in KAPPAS:
            if k == 1.0:
                preds[str(k)] = {"dv_nominal": 0.0, "sign": 0}
                continue
            dv = model_valley(k * d0, lam) - v0
            preds[str(k)] = {"dv_nominal": float(dv), "sign": int(np.sign(dv))}
        rows.append({"layer": r["layer"], "head": r["head"], "delta_1d": d0,
                     "r2_1d": r["r2_1d"], "lambda_M0": lam,
                     "v_hat_baseline": float(v0), "predicted": preds})
    res = _load()
    res["stage1_predictions_committed"] = {
        "committed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "note": "computed BEFORE any weight edit (prereg §2.2); nominal Δ_edit = κ·Δ̂",
        "selection_rule": f"top-{N_SELECT} r2_1d among conformal heads with Δ̂ in {DELTA_RANGE}",
        "heads": rows,
    }
    _save(res)
    for r in rows:
        print(f"L{r['layer']}H{r['head']} Δ={r['delta_1d']:.3f} λ={r['lambda_M0']:.3f} "
              f"v̂0={r['v_hat_baseline']:.3f} signs:",
              {k: v["sign"] for k, v in r["predicted"].items()})


# ── shared measurement ─────────────────────────────────────────────────────────

def fit_1d_from_attn(A_mean: np.ndarray):
    prof = np.zeros(FIT_HIGH)
    for dx in range(FIT_HIGH):
        diag = np.diagonal(A_mean, offset=-dx)
        k_lo = max(MIN_POS, dx) - dx
        prof[dx] = diag[k_lo:].mean()
    lags = np.arange(FIT_LOW, FIT_HIGH)
    y = prof[FIT_LOW:FIT_HIGH]
    valid = y > 1e-15
    lx, ly = np.log(lags[valid].astype(float)), np.log(y[valid])
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, ly, rcond=None)
    pred = X @ c
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1 - float(np.sum((ly - pred) ** 2)) / ss_tot if ss_tot > 1e-30 else 0.0
    return float(-c[1] / 2), r2


def valley_from_attn(A_mean: np.ndarray) -> float:
    deep = A_mean[Q_LO:, :]
    S = deep[:, J_START].mean()
    M = deep[:, J_MID].mean()
    E = deep[:, J_END].mean()
    return float(1.0 - M / max(S, E))


def score_slope(scores_mean: np.ndarray):
    """OLS slope of mean score vs log lag over [FIT_LOW, FIT_HIGH) (exp-056)."""
    prof = np.zeros(FIT_HIGH)
    for dx in range(FIT_LOW, FIT_HIGH):
        diag = np.diagonal(scores_mean, offset=-dx)
        k_lo = max(MIN_POS, dx) - dx
        prof[dx] = diag[k_lo:].mean()
    lags = np.arange(FIT_LOW, FIT_HIGH).astype(float)
    y = prof[FIT_LOW:FIT_HIGH]
    X = np.column_stack([np.ones_like(lags), np.log(lags)])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(c[1])


def measure_head(model, layer: int, head: int, device, inputs: torch.Tensor):
    """Mean attention (L,L) + mean pre-softmax score (L,L) for one head."""
    blk = model.transformer.h[layer]
    A_sum = np.zeros((SEQ_LEN, SEQ_LEN))
    S_sum = np.zeros((SEQ_LEN, SEQ_LEN))
    sd = blk.attn.head_dim if hasattr(blk.attn, "head_dim") else HEAD_DIM
    for b in range(inputs.shape[0]):
        ids = inputs[b:b + 1].to(device)
        with torch.no_grad():
            out = model(ids, output_attentions=True, output_hidden_states=True)
            a_t = out.attentions[layer]
            assert a_t.dtype == torch.float32
            a = a_t[0, head].cpu().numpy()
            assert not np.isnan(a).any()
            A_sum += a
            # recompute this head's raw scores through ln_1 (exp-056 method)
            x = blk.ln_1(out.hidden_states[layer])
            qkv = blk.attn.c_attn(x)
            nd = model.config.n_embd
            q = qkv[..., :nd][0, :, head * sd:(head + 1) * sd]
            k = qkv[..., nd:2 * nd][0, :, head * sd:(head + 1) * sd]
            s = (q @ k.T / np.sqrt(sd)).cpu().numpy()
            S_sum += s
    return A_sum / inputs.shape[0], S_sum / inputs.shape[0]


def positional_projector(model, layer: int, device, inputs: torch.Tensor) -> np.ndarray:
    """Top-N_PC PCs of the position-mean ln_1 output at `layer` -> projector (nd, nd)."""
    blk = model.transformer.h[layer]
    nd = model.config.n_embd
    acc = np.zeros((SEQ_LEN, nd))
    for b in range(inputs.shape[0]):
        ids = inputs[b:b + 1].to(device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
            acc += blk.ln_1(out.hidden_states[layer])[0].cpu().numpy()
    xbar = acc / inputs.shape[0]
    xbar = xbar - xbar.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(xbar, full_matrices=False)
    U = vt[:N_PC].T                                      # (nd, N_PC)
    return U @ U.T


def edit_and_measure() -> None:
    res = _load()
    sel = res["stage1_predictions_committed"]["heads"]
    device = _device()
    rng = np.random.default_rng(RNG_SEED)
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2", dtype=torch.float32, attn_implementation="eager").to(device).eval()
    vocab = model.config.vocab_size
    nd = model.config.n_embd
    inputs = torch.tensor(rng.integers(0, vocab, size=(N_INPUTS, SEQ_LEN)),
                          dtype=torch.long)

    measured = []
    t0 = time.time()
    for hrec in sel:
        layer, head = hrec["layer"], hrec["head"]
        P = positional_projector(model, layer, device, inputs)
        W = model.transformer.h[layer].attn.c_attn.weight      # (nd, 3nd)
        col0 = head * HEAD_DIM
        Wq_orig = W[:, col0:col0 + HEAD_DIM].detach().clone()
        row = {"layer": layer, "head": head, "kappa": {}}
        for kap in KAPPAS:
            M_edit = torch.eye(nd) + (kap - 1.0) * torch.tensor(P, dtype=torch.float32)
            with torch.no_grad():
                W[:, col0:col0 + HEAD_DIM] = (M_edit.to(device) @ Wq_orig)
            A_mean, S_mean = measure_head(model, layer, head, device, inputs)
            d_hat, r2 = fit_1d_from_attn(A_mean)
            row["kappa"][str(kap)] = {
                "delta_hat": d_hat, "r2": r2,
                "score_slope": score_slope(S_mean),
                "valley": valley_from_attn(A_mean),
            }
            print(f"L{layer}H{head} κ={kap}: Δ̂={d_hat:.4f} r2={r2:.3f} "
                  f"α={row['kappa'][str(kap)]['score_slope']:.4f} "
                  f"valley={row['kappa'][str(kap)]['valley']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        with torch.no_grad():
            W[:, col0:col0 + HEAD_DIM] = Wq_orig               # restore
        measured.append(row)
        res["stage2_measurements"] = measured
        _save(res)
    print("EDIT_MEASURE_DONE")


def evaluate() -> None:
    res = _load()
    sel = {(h["layer"], h["head"]): h for h in res["stage1_predictions_committed"]["heads"]}
    rows = res["stage2_measurements"]
    mech_pass = {k: 0 for k in ("1.5", "2.0", "0.5")}
    sham_dd, pred_dv, meas_dv, signs_ok, n_signs = [], [], [], 0, 0
    details = []
    for r in rows:
        key = (r["layer"], r["head"])
        base = r["kappa"]["1.0"]
        h = sel[key]
        lam = h["lambda_M0"]
        v0_meas = base["valley"]
        d0_meas = base["delta_hat"]
        sham_dd.append(abs(d0_meas - h["delta_1d"]))
        for kap in ("0.5", "1.5", "2.0"):
            m = r["kappa"][kap]
            dd = m["delta_hat"] - d0_meas
            expected = np.sign(float(kap) - 1.0)
            if np.sign(dd) == expected:
                mech_pass[kap] += 1
            # behavioral: predicted Δv̂ from realized Δ̂_edit (prereg §2.2)
            dv_hat = model_valley(m["delta_hat"], lam) - model_valley(d0_meas, lam)
            dv_meas = m["valley"] - v0_meas
            pred_dv.append(dv_hat)
            meas_dv.append(dv_meas)
            n_signs += 1
            if np.sign(dv_hat) == np.sign(dv_meas):
                signs_ok += 1
            details.append({"layer": r["layer"], "head": r["head"], "kappa": float(kap),
                            "d_delta_hat": float(dd), "dv_hat": float(dv_hat),
                            "dv_measured": float(dv_meas)})
    rho, p = spearmanr(pred_dv, meas_dv)
    n = len(rows)
    mech_ok = mech_pass["1.5"] >= 6 and mech_pass["2.0"] >= 6
    works = bool(mech_ok and rho >= 0.5 and signs_ok >= 16)
    summary = {
        "mechanical": {"n_heads": n, "sign_correct": mech_pass,
                       "sham_abs_delta_change_median": float(np.median(sham_dd)),
                       "pass": bool(mech_ok)},
        "behavioral": {"rho_dvhat_dvmeas": float(rho), "p": float(p),
                       "sign_agreement": f"{signs_ok}/{n_signs}"},
        "verdict": "HANDLE WORKS" if works else "HANDLE FAILS (per pre-registered rule)",
        "details": details,
    }
    res["stage3_evaluation"] = summary
    _save(res)
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=1))


if __name__ == "__main__":
    {"predict": predict, "edit": edit_and_measure, "evaluate": evaluate}[sys.argv[1]]()
