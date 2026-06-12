"""
exp-058 — Exponent histograms and null distributions (Phase 0 item 0.3).

Pre-stated question (EVALUATION.md W3, selection/circularity):
Is the Δ ≈ 0.25 cluster a density excess in the per-head exponent distribution,
or an artifact of double conditioning (R² filter → SYK-near filter)?

Design — two protocol arms, each with observed + two nulls on GPT-2:
  Arm "census":  frozen census protocol (SESSION_BRIEF_PHASE0 §2.2):
                 L=512, 50 random inputs, queries i ≥ L/2, fit lags [8, 256].
  Arm "exp007":  the original exp-007 protocol behind the §4.1 headline
                 (44/144, median 0.2493): L=256, 50 random inputs,
                 queries i ≥ max(32, dx), fit lags [3, 49].
  OBSERVED:  trained GPT-2, ALL 144 heads fitted regardless of R², fp32 verified.
  NULL A:    randomized-weight GPT-2 (fresh init, N_RAND_SEEDS seeds), identical pipeline.
  NULL B:    shuffled-profile null — permute each trained head's A(Δx) values across
             lags within the fit window and refit (destroys decay structure, preserves
             value distribution). N_SHUFFLE_REPS full-population replicates → permutation p.
  ARCHIVED:  per-head (Δ, R²) from archived JSONs for GPT-2-medium (exp-031),
             GALA-7B softmax (exp-041), OLMo-7B (exp-025), Mistral-7B-v0.3 (exp-025),
             Pythia-410m fp32 (exp-036). Protocols heterogeneous; recorded per model.

Statistic (pre-stated):
  excess mass = P(Δ ∈ [0.20, 0.30] AND R² ≥ 0.90) over all heads, observed vs each
  null; permutation p-value from NULL B replicates; conditional fraction
  P(Δ ∈ window | R² ≥ 0.90) where defined; histograms also reported with NO R² filter.

Constraint §2.3: per-input profiles for the trained GPT-2 runs are saved (gzipped JSON).
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
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "results.json"
PHYSICS_ROOT = OUT_DIR.parents[1]  # research/physics

PROTOCOLS = {
    "census": {"seq_len": 512, "deep_lo": 256, "fit_lo": 8, "fit_hi": 256,
               "desc": "frozen census protocol: L=512, queries i>=L/2, fit [8,256]"},
    "exp007": {"seq_len": 256, "deep_lo": 32, "fit_lo": 3, "fit_hi": 49,
               "desc": "original exp-007 protocol: L=256, queries i>=max(32,dx), fit [3,49]"},
}

N_INPUTS = 50
RNG_SEED = 42
R2_THRESHOLD = 0.90
DELTA_WINDOW = (0.20, 0.30)
SYK_WINDOW = 0.05
N_RAND_SEEDS = 5
N_SHUFFLE_REPS = 999


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── measurement ────────────────────────────────────────────────────────────────

def lag_profile(att: np.ndarray, deep_lo: int) -> np.ndarray:
    """att: (n_heads, L, L). Lag profile averaged over queries q >= max(deep_lo, dx)."""
    n_heads, L, _ = att.shape
    prof = np.zeros((n_heads, L), dtype=np.float64)
    for dx in range(L):
        diag = np.diagonal(att, offset=-dx, axis1=-2, axis2=-1)  # entry k ↔ q = dx + k
        k_lo = max(deep_lo, dx) - dx
        if k_lo < diag.shape[-1]:
            prof[:, dx] = diag[:, k_lo:].mean(axis=-1)
    return prof


def measure_profiles(model, vocab_size: int, n_layers: int, n_heads: int,
                     device: torch.device, seed: int, seq_len: int, deep_lo: int,
                     keep_per_input: bool) -> tuple[np.ndarray, list | None]:
    rng = np.random.default_rng(seed)
    mean_prof = np.zeros((n_layers, n_heads, seq_len), dtype=np.float64)
    per_input = [] if keep_per_input else None

    for inp in range(N_INPUTS):
        token_ids = rng.integers(0, vocab_size, size=seq_len)
        input_ids = torch.tensor(token_ids[None, :], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids, output_attentions=True)
        this_input = np.zeros((n_layers, n_heads, seq_len), dtype=np.float32) if keep_per_input else None
        for l in range(n_layers):
            a_t = out.attentions[l]
            # dtype verification at extraction (standing constraint §2.1)
            assert a_t.dtype == torch.float32, f"layer {l}: dtype {a_t.dtype} != float32"
            a = a_t[0].cpu().numpy()
            assert not np.isnan(a).any(), f"layer {l}: NaN in attention"
            prof = lag_profile(a, deep_lo)
            mean_prof[l] += prof
            if keep_per_input:
                this_input[l] = prof.astype(np.float32)
        if keep_per_input:
            per_input.append(this_input)
        del out

    mean_prof /= N_INPUTS
    return mean_prof, per_input


def fit_power_law(profile: np.ndarray, lo: int, hi: int) -> dict:
    lags = np.arange(lo, hi + 1)
    y = profile[lo: hi + 1]
    valid = y > 1e-15
    if valid.sum() < 10:
        return {"delta": None, "R2": None, "n_points": int(valid.sum())}
    log_x = np.log(lags[valid].astype(float))
    log_y = np.log(y[valid])
    A = np.column_stack([np.ones_like(log_x), log_x])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    pred = A @ coeffs
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    return {"delta": float(-coeffs[1] / 2.0), "R2": float(r2), "n_points": int(valid.sum())}


def fit_all_heads(profiles: np.ndarray, lo: int, hi: int) -> list[dict]:
    n_layers, n_heads, _ = profiles.shape
    out = []
    for l in range(n_layers):
        for h in range(n_heads):
            rec = fit_power_law(profiles[l, h], lo, hi)
            rec.update({"layer": l, "head": h})
            out.append(rec)
    return out


# ── statistics ─────────────────────────────────────────────────────────────────

def joint_mass(records: list[dict]) -> float:
    n = len(records)
    k = sum(1 for r in records
            if r["delta"] is not None and r["R2"] is not None
            and r["R2"] >= R2_THRESHOLD and DELTA_WINDOW[0] <= r["delta"] <= DELTA_WINDOW[1])
    return k / n if n else float("nan")


def conditional_fraction(records: list[dict]) -> tuple[float | None, int]:
    sel = [r for r in records if r["delta"] is not None and r["R2"] is not None and r["R2"] >= R2_THRESHOLD]
    if not sel:
        return None, 0
    k = sum(1 for r in sel if DELTA_WINDOW[0] <= r["delta"] <= DELTA_WINDOW[1])
    return k / len(sel), len(sel)


def unfiltered_window_fraction(records: list[dict]) -> float:
    n = len(records)
    k = sum(1 for r in records if r["delta"] is not None and DELTA_WINDOW[0] <= r["delta"] <= DELTA_WINDOW[1])
    return k / n if n else float("nan")


def summarize(records: list[dict], label: str) -> dict:
    cond, n_conf = conditional_fraction(records)
    deltas_conf = [r["delta"] for r in records
                   if r["delta"] is not None and r["R2"] is not None and r["R2"] >= R2_THRESHOLD]
    syk = [d for d in deltas_conf if abs(d - 0.25) <= SYK_WINDOW]
    return {
        "label": label,
        "n_heads": len(records),
        "n_fitted": sum(1 for r in records if r["delta"] is not None),
        "n_conformal_R2_90": n_conf,
        "joint_mass_window_and_R2": joint_mass(records),
        "conditional_fraction_window_given_R2": cond,
        "unfiltered_window_fraction": unfiltered_window_fraction(records),
        "median_delta_conformal": float(np.median(deltas_conf)) if deltas_conf else None,
        "n_syk_near": len(syk),
        "syk_near_median": float(np.median(syk)) if syk else None,
    }


# ── archived extraction ────────────────────────────────────────────────────────

def load_archived() -> dict[str, dict]:
    out = {}

    p = PHYSICS_ROOT / "results/exp-031_gpt2_depth_convergence_2026-05-23T062519Z.json"
    d = json.loads(p.read_text())["models"]["gpt2-medium"]
    out["GPT-2-medium"] = {
        "source": p.name, "experiment": "exp-031",
        "protocol": "L=256, fit [3,50] (exp-007 original protocol)",
        "records": [{"delta": h["delta"], "R2": h["R2_pl"], "layer": h["layer"], "head": h["head"]}
                    for h in d["per_head"]],
    }

    p = PHYSICS_ROOT / "experiments/exp-041_gala7b_sigmoid_delta/results.json"
    d = json.loads(p.read_text())["softmax"]
    out["GALA-7B-softmax"] = {
        "source": p.name, "experiment": "exp-041",
        "protocol": "L=512, fit [8,256] (census protocol); 864/1024 heads have stored fits",
        "records": [{"delta": h["delta"], "R2": h["R2"], "layer": h["layer"], "head": h["head"]}
                    for h in d["head_results"]],
    }

    p = PHYSICS_ROOT / "experiments/exp-025_bcft_pre_registered/results/bcft_pre_registered_run_2026-04-17T092239Z.json"
    d = json.loads(p.read_text())["models"]["allenai/OLMo-7B-hf"]
    out["OLMo-7B"] = {
        "source": p.name, "experiment": "exp-025",
        "protocol": "L=512, fit [3,120] (pre-registered pipeline)",
        "records": [{"delta": h["delta"], "R2": h["R2"], "layer": h["layer"], "head": h["head"]}
                    for h in d["heads"]],
    }

    p = PHYSICS_ROOT / "experiments/exp-025_bcft_pre_registered/results/bcft_pre_registered_run_2026-04-17T095022Z.json"
    d = json.loads(p.read_text())["models"]["mistralai/Mistral-7B-v0.3"]
    out["Mistral-7B-v0.3"] = {
        "source": p.name, "experiment": "exp-025",
        "protocol": "L=512, fit [3,120] (pre-registered pipeline)",
        "records": [{"delta": h["delta"], "R2": h["R2"], "layer": h["layer"], "head": h["head"]}
                    for h in d["heads"]],
    }

    p = PHYSICS_ROOT / "experiments/exp-036_pythia_410m_bcft_local/results.json"
    d = json.loads(p.read_text())["models"]["EleutherAI/pythia-410m"]
    out["Pythia-410m-fp32"] = {
        "source": p.name, "experiment": "exp-036",
        "protocol": "L=512, fit [3,120] (pre-registered pipeline), fp32",
        "records": [{"delta": h["delta"], "R2": h["R2"], "layer": h["layer"], "head": h["head"]}
                    for h in d["heads"]],
    }

    return out


# ── per-protocol pipeline ──────────────────────────────────────────────────────

def run_arm(arm: str, device: torch.device) -> dict:
    p = PROTOCOLS[arm]
    seq_len, deep_lo, lo, hi = p["seq_len"], p["deep_lo"], p["fit_lo"], p["fit_hi"]
    print(f"\n[{arm}] {p['desc']}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", dtype=torch.float32, attn_implementation="eager"
    ).to(device).eval()
    cfg = model.config
    n_layers, n_heads, vocab = cfg.n_layer, cfg.n_head, cfg.vocab_size

    print(f"[{arm}] observed: measuring trained GPT-2...", flush=True)
    mean_prof, per_input = measure_profiles(
        model, vocab, n_layers, n_heads, device, RNG_SEED, seq_len, deep_lo, keep_per_input=True
    )
    observed = fit_all_heads(mean_prof, lo, hi)
    del model

    # per-input profiles (constraint §2.3)
    out_pp = OUT_DIR / f"gpt2_per_input_profiles_{arm}.json.gz"
    payload = {
        "experiment": "exp-058", "model": "gpt2", "arm": arm,
        "protocol": {**p, "n_inputs": N_INPUTS, "rng_seed": RNG_SEED,
                     "input_type": "uniform random token ids, numpy default_rng(seed)"},
        "shape": [N_INPUTS, n_layers, n_heads, seq_len],
        "axes": ["input", "layer", "head", "lag"],
        "profiles": [np.round(x, 10).tolist() for x in per_input],
    }
    with gzip.open(out_pp, "wt") as f:
        json.dump(payload, f)
    print(f"[{arm}] per-input profiles → {out_pp.name}", flush=True)

    # NULL A — randomized weights
    rand_records_by_seed = []
    for s in range(N_RAND_SEEDS):
        print(f"[{arm}] null A: randomized GPT-2, init seed {s}...", flush=True)
        torch.manual_seed(1000 + s)
        rcfg = GPT2Config()
        rcfg._attn_implementation = "eager"
        rmodel = GPT2LMHeadModel(rcfg).to(torch.float32).to(device).eval()
        rprof, _ = measure_profiles(rmodel, vocab, n_layers, n_heads, device,
                                    RNG_SEED, seq_len, deep_lo, keep_per_input=False)
        rand_records_by_seed.append(fit_all_heads(rprof, lo, hi))
        del rmodel
    rand_pooled = [r for recs in rand_records_by_seed for r in recs]

    # NULL B — shuffled profiles
    print(f"[{arm}] null B: shuffled-profile null, {N_SHUFFLE_REPS} replicates...", flush=True)
    shuf_rng = np.random.default_rng(RNG_SEED + 1)
    obs_mass = joint_mass(observed)
    null_masses = []
    shuf_example = None
    flat = mean_prof.reshape(-1, seq_len)
    for rep in range(N_SHUFFLE_REPS):
        recs = []
        for idx in range(flat.shape[0]):
            prof = flat[idx].copy()
            window = prof[lo: hi + 1].copy()
            shuf_rng.shuffle(window)
            prof[lo: hi + 1] = window
            recs.append(fit_power_law(prof, lo, hi))
        null_masses.append(joint_mass(recs))
        if rep == 0:
            shuf_example = recs

    n_ge = sum(1 for m in null_masses if m >= obs_mass)
    perm_p = (1 + n_ge) / (N_SHUFFLE_REPS + 1)

    return {
        "protocol": {**p, "n_inputs": N_INPUTS, "rng_seed": RNG_SEED,
                     "R2_threshold": R2_THRESHOLD, "dtype": "float32 (verified at extraction)"},
        "observed": {"summary": summarize(observed, f"GPT-2 trained ({arm})"), "records": observed},
        "null_A_randomized": {
            "per_seed_joint_mass": [joint_mass(r) for r in rand_records_by_seed],
            "pooled_summary": summarize(rand_pooled, f"GPT-2 randomized pooled ({arm})"),
            "records_pooled": rand_pooled,
        },
        "null_B_shuffled": {
            "observed_joint_mass": obs_mass,
            "null_joint_mass_mean": float(np.mean(null_masses)),
            "null_joint_mass_max": float(np.max(null_masses)),
            "n_replicates": N_SHUFFLE_REPS,
            "n_null_ge_observed": n_ge,
            "permutation_p": perm_p,
            "example_replicate_summary": summarize(shuf_example, f"shuffled replicate 0 ({arm})"),
        },
    }


def main() -> None:
    t0 = time.time()
    device = _device()
    print(f"exp-058: device={device}", flush=True)

    arms = {arm: run_arm(arm, device) for arm in PROTOCOLS}

    print("[archived] extracting per-head fits...", flush=True)
    archived = load_archived()

    result = {
        "experiment": "exp-058",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "phase0_item": "0.3",
        "question": "Is the Δ≈0.25 cluster a density excess or a conditioning artifact (EVALUATION W3)?",
        "statistic": f"P(Δ∈[{DELTA_WINDOW[0]},{DELTA_WINDOW[1]}] AND R²≥{R2_THRESHOLD}) over all heads; "
                     f"permutation p from shuffled-profile null",
        "gpt2_arms": arms,
        "archived_models": {
            name: {"source": v["source"], "experiment": v["experiment"], "protocol": v["protocol"],
                   "summary": summarize(v["records"], name), "records": v["records"]}
            for name, v in archived.items()
        },
        "elapsed_seconds": time.time() - t0,
    }
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=OUT_DIR).decode().strip()
    except Exception:
        result["git_commit"] = "unknown"

    RESULTS_PATH.write_text(json.dumps(result, indent=1))

    print("\n=== exp-058 SUMMARY ===", flush=True)
    print(json.dumps({
        arm: {
            "observed": arms[arm]["observed"]["summary"],
            "null_A_per_seed_joint_mass": arms[arm]["null_A_randomized"]["per_seed_joint_mass"],
            "null_B": {k: v for k, v in arms[arm]["null_B_shuffled"].items()
                       if k != "example_replicate_summary"},
        } for arm in arms
    } | {"archived": {k: v["summary"] for k, v in result["archived_models"].items()},
         "elapsed_s": round(result["elapsed_seconds"], 1)}, indent=1), flush=True)
    print(f"Wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
