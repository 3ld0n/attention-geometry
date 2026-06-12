"""
exp-059 — Split-half stability of ρ(Δ, valley_depth) (Phase 0 item 0.1).

Answers EVALUATION.md W1 (shared-measurement inflation): Δ and valley_depth in the
pre-registered test (exp-025) derive from the same averaged attention tensors on the
same inputs; shared noise can inflate ρ. Archived April JSONs contain only
input-averaged measurements, so this is a re-run that SAVES PER-INPUT PROFILES
(standing constraint §2.3) and computes:

  1. Same-tensor ρ on the new inputs (reproduction check vs April archived values).
  2. Split-half: Δ from inputs 1–25, valley from inputs 26–50, and the reverse;
     conformal subset selected on the Δ-half (mirrors the pipeline), with a
     sensitivity variant using the full-data conformal subset.
  3. Bootstrap over inputs (1,000 draws, resample with replacement) → CI on ρ.

Models (pre-stated): GPT-Neo-2.7B (clean, April ρ=+0.958), Mistral-7B-v0.3 (mid,
+0.583), Pythia-2.8B (falsified, +0.464).

Protocol: exp-025 pre-registered pipeline, unchanged (L=512, 50 random-token
inputs, fit lags [3,120), deep queries q ≥ max(256, Δx) for the lag profile,
valley from queries q ≥ 384, conformal R² ≥ 0.85 ∧ Δ ≥ 0.05, Spearman ρ).
fp32 everywhere, dtype verified at extraction. Fresh input seed (recorded).

Keep: cross-half ρ within ~0.1 of same-tensor ρ → W1 closed.
Kill: substantial collapse (e.g. GPT-Neo +0.96 → < +0.5 cross-half) → behavioral
claim was inflated; §9 marked; v2 pre-registration must use disjoint inputs.
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
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM

OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "results.json"

MODELS = [
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/pythia-2.8b",
    "mistralai/Mistral-7B-v0.3",
]

APRIL_ARCHIVED_RHO = {
    "EleutherAI/pythia-410m": 0.760,
    "EleutherAI/pythia-1.4b": 0.711,
    "EleutherAI/gpt-neo-2.7B": 0.958,
    "EleutherAI/pythia-2.8b": 0.464,
    "mistralai/Mistral-7B-v0.3": 0.583,
}

# exp-025 pre-registered pipeline constants
SEQ_LEN = 512
N_INPUTS = 50
FIT_LOW = 3
FIT_HIGH = min(120, SEQ_LEN // 3)   # 120, exclusive bound as in exp-025/exp-036
DEEP_LO = SEQ_LEN // 2              # 256
Q_LO = 3 * SEQ_LEN // 4             # 384
THIRD = SEQ_LEN // 3                # 170
CONFORMAL_R2 = 0.85
CONFORMAL_DELTA = 0.05

INPUT_SEED = 2026                   # fresh inputs (April run used a different stream)
N_BOOT = 1000
BOOT_SEED = 59


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── per-input extraction ───────────────────────────────────────────────────────

def extract_per_input(model_name: str, device: torch.device):
    """Returns (lagprof, thirds, meta):
       lagprof: (N_INPUTS, n_total_heads, FIT_HIGH) per-input lag profile,
                queries q >= max(DEEP_LO, dx)
       thirds:  (N_INPUTS, n_total_heads, 3) per-input (start, middle, end) masses
                from the deep-query (q >= Q_LO) averaged key profile
    """
    print(f"[{model_name}] loading fp32, eager...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="eager",
        trust_remote_code=True,
    ).to(device).eval()
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer")
    n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head")
    vocab = cfg.vocab_size
    n_total = n_layers * n_heads
    print(f"[{model_name}] {n_layers} layers × {n_heads} heads = {n_total}", flush=True)

    rng = np.random.default_rng(INPUT_SEED)
    lagprof = np.zeros((N_INPUTS, n_total, FIT_HIGH), dtype=np.float32)
    thirds = np.zeros((N_INPUTS, n_total, 3), dtype=np.float32)

    t0 = time.time()
    for inp in range(N_INPUTS):
        token_ids = rng.integers(0, vocab, size=SEQ_LEN)
        input_ids = torch.tensor(token_ids[None, :], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids, output_attentions=True)
        for l in range(n_layers):
            a_t = out.attentions[l]
            assert a_t.dtype == torch.float32, f"layer {l}: dtype {a_t.dtype}"
            a = a_t[0].cpu().numpy()          # (n_heads, L, L)
            assert not np.isnan(a).any(), f"layer {l}: NaN attention"
            base = l * n_heads
            # lag profile, lags 0..FIT_HIGH-1, queries q >= max(DEEP_LO, dx)
            for dx in range(FIT_HIGH):
                diag = np.diagonal(a, offset=-dx, axis1=-2, axis2=-1)  # entry k ↔ q = dx+k
                k_lo = max(DEEP_LO, dx) - dx
                lagprof[inp, base:base + n_heads, dx] = diag[:, k_lo:].mean(axis=-1)
            # deep-query key profile (exp-025/exp-036 definition)
            prof = a[:, Q_LO:, :].mean(axis=1)  # (n_heads, L) mean over q >= Q_LO
            thirds[inp, base:base + n_heads, 0] = prof[:, 1:THIRD].mean(axis=-1)
            thirds[inp, base:base + n_heads, 1] = prof[:, THIRD:2 * THIRD].mean(axis=-1)
            thirds[inp, base:base + n_heads, 2] = prof[:, 2 * THIRD:Q_LO].mean(axis=-1)
        del out
        if (inp + 1) % 10 == 0:
            print(f"[{model_name}] forward {inp + 1}/{N_INPUTS} ({time.time()-t0:.0f}s)", flush=True)

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    meta = {"n_layers": int(n_layers), "n_heads": int(n_heads), "n_total_heads": int(n_total)}
    return lagprof, thirds, meta


# ── vectorized fitting and statistics ──────────────────────────────────────────

LOG_X = np.log(np.arange(FIT_LOW, FIT_HIGH).astype(float))   # lags 3..119
X_C = LOG_X - LOG_X.mean()
SXX = float(np.sum(X_C ** 2))


def fit_heads(mean_lagprof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mean_lagprof: (n_heads, FIT_HIGH). Returns (delta, R2) arrays."""
    y = np.log(np.maximum(mean_lagprof[:, FIT_LOW:FIT_HIGH], 1e-30))
    y_c = y - y.mean(axis=1, keepdims=True)
    slope = (y_c @ X_C) / SXX
    pred = slope[:, None] * X_C[None, :]
    ss_res = np.sum((y_c - pred) ** 2, axis=1)
    ss_tot = np.sum(y_c ** 2, axis=1)
    r2 = np.where(ss_tot > 1e-30, 1.0 - ss_res / np.maximum(ss_tot, 1e-30), 0.0)
    return -slope / 2.0, r2


def valley_from_thirds(mean_thirds: np.ndarray) -> np.ndarray:
    """mean_thirds: (n_heads, 3) → valley depth, NaN where any third < 1e-15."""
    s, m, e = mean_thirds[:, 0], mean_thirds[:, 1], mean_thirds[:, 2]
    peak = np.maximum(s, e)
    valley = 1.0 - m / np.maximum(peak, 1e-30)
    bad = np.minimum(np.minimum(s, m), e) < 1e-15
    return np.where(bad, np.nan, valley)


def rho_on_subset(delta, r2, valley, sel_delta=None, sel_r2=None):
    """Spearman ρ(Δ, valley) on the conformal subset. Selection arrays default
    to (delta, r2) themselves (same-tensor pipeline)."""
    sd = delta if sel_delta is None else sel_delta
    sr = r2 if sel_r2 is None else sel_r2
    mask = (sr >= CONFORMAL_R2) & (sd >= CONFORMAL_DELTA) & ~np.isnan(valley) & ~np.isnan(delta)
    n = int(mask.sum())
    if n < 3:
        return None, None, n
    rho, p = spearmanr(delta[mask], valley[mask])
    return float(rho), float(p), n


def analyze_model(lagprof: np.ndarray, thirds: np.ndarray) -> dict:
    full_lag, full_thirds = lagprof.mean(axis=0), thirds.mean(axis=0)
    d_full, r2_full = fit_heads(full_lag)
    v_full = valley_from_thirds(full_thirds)
    rho_full, p_full, n_full = rho_on_subset(d_full, r2_full, v_full)

    halves = {"A": np.arange(0, 25), "B": np.arange(25, 50)}
    half_fits = {}
    for name, idx in halves.items():
        d, r2 = fit_heads(lagprof[idx].mean(axis=0))
        v = valley_from_thirds(thirds[idx].mean(axis=0))
        half_fits[name] = (d, r2, v)

    split = {}
    for d_half, v_half, tag in [("A", "B", "delta_A_valley_B"), ("B", "A", "delta_B_valley_A")]:
        d, r2, _ = half_fits[d_half]
        _, _, v = half_fits[v_half]
        rho, p, n = rho_on_subset(d, r2, v)
        rho_fc, p_fc, n_fc = rho_on_subset(d, r2, v, sel_delta=d_full, sel_r2=r2_full)
        split[tag] = {"rho": rho, "p": p, "n_conformal": n,
                      "sensitivity_full_data_subset": {"rho": rho_fc, "p": p_fc, "n": n_fc}}

    same_half = {}
    for name in halves:
        d, r2, v = half_fits[name]
        rho, p, n = rho_on_subset(d, r2, v)
        same_half[name] = {"rho": rho, "p": p, "n_conformal": n}

    print("  bootstrap...", flush=True)
    brng = np.random.default_rng(BOOT_SEED)
    boot_rhos = []
    for _ in range(N_BOOT):
        idx = brng.integers(0, N_INPUTS, size=N_INPUTS)
        d, r2 = fit_heads(lagprof[idx].mean(axis=0))
        v = valley_from_thirds(thirds[idx].mean(axis=0))
        rho, _, _ = rho_on_subset(d, r2, v)
        if rho is not None:
            boot_rhos.append(rho)
    boot_rhos = np.array(boot_rhos)

    return {
        "same_tensor": {"rho": rho_full, "p": p_full, "n_conformal": n_full},
        "split_half": split,
        "same_half_controls": same_half,
        "bootstrap": {
            "n_draws": int(len(boot_rhos)),
            "rho_mean": float(boot_rhos.mean()),
            "rho_std": float(boot_rhos.std()),
            "ci_2.5": float(np.percentile(boot_rhos, 2.5)),
            "ci_97.5": float(np.percentile(boot_rhos, 97.5)),
        },
    }


def main() -> None:
    import sys
    t0 = time.time()
    device = _device()
    print(f"exp-059: device={device}", flush=True)

    result = {
        "experiment": "exp-059",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "phase0_item": "0.1",
        "question": "Is the pre-registered rho(Delta, valley) inflated by shared-measurement noise (EVALUATION W1)?",
        "protocol": {
            "pipeline": "exp-025 pre-registered, unchanged",
            "seq_len": SEQ_LEN, "n_inputs": N_INPUTS,
            "fit_range_exclusive": [FIT_LOW, FIT_HIGH],
            "deep_lo": DEEP_LO, "q_lo": Q_LO,
            "conformal": {"R2": CONFORMAL_R2, "delta_min": CONFORMAL_DELTA},
            "input_seed": INPUT_SEED, "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED},
            "dtype": "float32 (verified at extraction)",
            "note": "fresh seeded inputs; April archived run used a different RNG stream",
        },
        "april_archived_rho": APRIL_ARCHIVED_RHO,
        "models": {},
    }
    if RESULTS_PATH.exists():
        prior = json.loads(RESULTS_PATH.read_text())
        result["models"] = prior.get("models", {})

    todo = sys.argv[1:] if len(sys.argv) > 1 else MODELS
    for model_name in todo:
        if model_name in result["models"]:
            print(f"skip {model_name} (already done)", flush=True)
            continue
        print(f"\n===== {model_name} =====", flush=True)
        lagprof, thirds, meta = extract_per_input(model_name, device)

        pp_path = OUT_DIR / f"per_input_{model_name.split('/')[-1]}.json.gz"
        with gzip.open(pp_path, "wt") as f:
            json.dump({
                "experiment": "exp-059", "model": model_name,
                "input_seed": INPUT_SEED, **meta,
                "lag_profile_axes": ["input", "flat_head (layer*n_heads+head)", "lag 0..119"],
                "thirds_axes": ["input", "flat_head", "(start, middle, end)"],
                "lag_profile": np.round(lagprof.astype(float), 12).tolist(),
                "thirds": np.round(thirds.astype(float), 12).tolist(),
            }, f)
        print(f"  per-input profiles → {pp_path.name}", flush=True)

        analysis = analyze_model(lagprof, thirds)
        analysis["meta"] = meta
        analysis["april_archived_rho"] = APRIL_ARCHIVED_RHO[model_name]
        result["models"][model_name] = analysis
        result["elapsed_seconds"] = time.time() - t0
        RESULTS_PATH.write_text(json.dumps(result, indent=1))  # incremental save
        print(json.dumps({k: v for k, v in analysis.items() if k != "meta"}, indent=1), flush=True)

    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=OUT_DIR).decode().strip()
    except Exception:
        result["git_commit"] = "unknown"
    result["elapsed_seconds"] = time.time() - t0
    RESULTS_PATH.write_text(json.dumps(result, indent=1))
    print(f"\nDONE_EXP059 — wrote {RESULTS_PATH} ({result['elapsed_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
