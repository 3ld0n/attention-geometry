"""
exp-074 — P-B2b: Context-length dilution at intermediate depth.

Pre-registration: notes.md in this directory (written before this script).

Hypothesis: prim_window(8) decreases with N at intermediate k ∈ {4,6,8} where
there is headroom (prim_window ≈ 0.05–0.28 at N=512), unlike k=24 where
prim_window ≈ 1.0 makes the effect trivially small (exp-067 P-B2 saturation).

Design:
  - Pythia-1.4b (cached)
  - N ∈ {256, 512, 1024, 2048}
  - For each N: measure all 24 layer profiles, compose to k ∈ {4, 6, 8, 12}
  - Report prim_window(8) in a k×N matrix, ρ(N, prim_window) per k row
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.signal import fftconvolve
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM

OUT_DIR = Path(__file__).resolve().parent

MODEL_NAME = "EleutherAI/pythia-1.4b"
N_INPUTS = 50
N_VALUES = [256, 512, 1024, 2048]
K_CHECKPOINTS = [4, 6, 8, 12]
WINDOW_W = 8

# Pre-stated thresholds (from notes.md)
KEEP_RHO = -0.80
KILL_RHO = -0.50
KEEP_RANGE = 0.02
KILL_RANGE = 0.005


def lag_profile_from_attn(avg_attn: np.ndarray, N: int) -> np.ndarray:
    """
    avg_attn: (N, N) mean attention matrix (averaged over heads and inputs).
    Returns profile (N,): profile[u] = mean attention at lag u = q - j,
    averaged over deep queries q >= N//2.

    Convention: lag u=0 is self-attention (j=q), u=1 is one step back, etc.
    profile is normalized to sum=1.
    """
    q_lo = N // 2
    profile = np.zeros(N)
    for u in range(N):
        q_min = max(q_lo, u)
        if q_min >= N:
            break
        q_arr = np.arange(q_min, N)
        j_arr = q_arr - u
        profile[u] = avg_attn[q_arr, j_arr].mean()
    s = profile.sum()
    if s > 0:
        profile /= s
    return profile


def compose_step(row: np.ndarray, profile: np.ndarray, N: int) -> np.ndarray:
    """
    Apply one layer's row-stochastic kernel to the current row.

    M[q,j] ∝ profile[q-j], j=0..q.
    new_row[j] = sum_{q=j}^{N-1} row[q] * M[q,j]
               = sum_{u=0}^{N-1-j} norm_row[j+u] * profile[u]

    Implemented via FFT convolution.
    """
    cum_prof = np.cumsum(profile)
    safe_cum = np.where(cum_prof > 1e-15, cum_prof, 1.0)
    norm_row = row / safe_cum
    self_loop_mask = cum_prof <= 1e-15
    self_loop_mass = row * self_loop_mask.astype(float)

    # fftconvolve(a[::-1], b)[N-1-j] = sum_v a[j+v] * b[v]
    corr = fftconvolve(norm_row[::-1], profile, mode="full")
    new_row = corr[:N][::-1].copy()
    new_row += self_loop_mass
    np.clip(new_row, 0, None, out=new_row)
    s = new_row.sum()
    if s > 1e-10:
        new_row /= s
    return new_row


def measure_profiles(model, N: int, n_inputs: int, device: str) -> list[np.ndarray]:
    """
    Run n_inputs random-token forward passes of length N.
    Returns list of n_layers lag profiles (each length N, normalized).
    """
    n_layers = model.config.num_hidden_layers
    vocab_size = model.config.vocab_size
    avg_attn = [np.zeros((N, N), dtype=np.float64) for _ in range(n_layers)]

    for _ in range(n_inputs):
        input_ids = torch.randint(0, vocab_size, (1, N), device=device)
        with torch.no_grad():
            out = model(input_ids, output_attentions=True)
        for l, attn in enumerate(out.attentions):
            avg_attn[l] += attn[0].mean(dim=0).cpu().float().numpy()

    profiles = []
    for l in range(n_layers):
        avg_attn[l] /= n_inputs
        profiles.append(lag_profile_from_attn(avg_attn[l], N))
    return profiles


def compose_to_checkpoints(
    profiles: list[np.ndarray], N: int, k_checkpoints: list[int]
) -> dict[int, dict]:
    """Compose layers 1..k, recording prim_window at each k in k_checkpoints."""
    row = np.zeros(N)
    row[N - 1] = 1.0  # start: all mass at the deepest query position

    results = {}
    for k in range(1, len(profiles) + 1):
        row = compose_step(row, profiles[k - 1], N)
        if k in k_checkpoints:
            results[k] = {
                "prim_window": float(row[:WINDOW_W].sum()),
                "prim_decile": float(row[: max(1, N // 10)].sum()),
            }
    return results


def main() -> None:
    t_start = time.time()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    print(f"\nLoading {MODEL_NAME}...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        output_attentions=True,
    )
    model.eval().to(device)
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers}L  loaded in {time.time()-t0:.1f}s", flush=True)

    k_valid = [k for k in K_CHECKPOINTS if k <= n_layers]

    # Matrix: prim_window[k][N]
    pw_matrix: dict[int, list[float]] = {k: [] for k in k_valid}
    pd_matrix: dict[int, list[float]] = {k: [] for k in k_valid}

    for N in N_VALUES:
        print(f"\n--- N={N} ---", flush=True)
        t1 = time.time()
        profiles = measure_profiles(model, N, N_INPUTS, device)
        print(f"  profiles measured in {time.time()-t1:.1f}s", flush=True)

        ck = compose_to_checkpoints(profiles, N, k_valid)
        for k in k_valid:
            pw = ck[k]["prim_window"]
            pd = ck[k]["prim_decile"]
            pw_matrix[k].append(pw)
            pd_matrix[k].append(pd)
            print(f"  k={k:2d}: prim_window={pw:.5f}  prim_decile={pd:.5f}", flush=True)

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("\n" + "=" * 60, flush=True)
    print("=== RESULTS ===", flush=True)
    print(f"N values: {N_VALUES}", flush=True)

    analysis: dict = {}
    for k in k_valid:
        pw_vals = np.array(pw_matrix[k])
        rho, p = spearmanr(N_VALUES, pw_vals)
        rho, p = float(rho), float(p)
        pw_range = float(pw_vals.max() - pw_vals.min())
        pw_list = [round(x, 6) for x in pw_vals.tolist()]

        # Verdict
        h1_keep = rho <= KEEP_RHO and pw_range > KEEP_RANGE
        h1_kill = rho > KILL_RHO or pw_range <= KILL_RANGE
        if h1_keep:
            verdict = "H1 KEEP"
        elif h1_kill:
            verdict = "H1 KILL"
        else:
            verdict = "H1 AMBIGUOUS"

        analysis[k] = {
            "prim_window_by_N": pw_list,
            "prim_decile_by_N": [round(x, 6) for x in pd_matrix[k]],
            "rho_N_prim_window": rho,
            "p": p,
            "range": pw_range,
            "verdict": verdict,
        }

        print(
            f"  k={k:2d}: prim_window={pw_list}  ρ={rho:+.4f}  range={pw_range:.5f}  → {verdict}",
            flush=True,
        )

    elapsed = time.time() - t_start

    # H2 gradient check (descriptive)
    ranges_by_k = [analysis[k]["range"] for k in k_valid]
    # Expect ranges to decrease as k increases (more saturation at higher k)
    h2_monotone = all(
        ranges_by_k[i] >= ranges_by_k[i + 1] for i in range(len(ranges_by_k) - 1)
    )
    print(
        f"\nH2 (gradient): ranges by k = {[f'{r:.5f}' for r in ranges_by_k]} — monotone={h2_monotone}",
        flush=True,
    )

    results = {
        "experiment": "exp-074",
        "title": "P-B2b: Context-length dilution at intermediate depth (Pythia-1.4b)",
        "preregistration": "research/physics/experiments/exp-074_pb2_intermediate_depth/notes.md",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "N_values": N_VALUES,
        "k_checkpoints": k_valid,
        "window_W": WINDOW_W,
        "n_inputs": N_INPUTS,
        "thresholds": {
            "keep_rho": KEEP_RHO,
            "kill_rho": KILL_RHO,
            "keep_range": KEEP_RANGE,
            "kill_range": KILL_RANGE,
        },
        "analysis_by_k": {str(k): analysis[k] for k in k_valid},
        "h2_gradient_monotone": h2_monotone,
        "ranges_by_k": {str(k): analysis[k]["range"] for k in k_valid},
        "elapsed_s": round(elapsed, 1),
    }

    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[{elapsed:.0f}s] Results → {out_path}", flush=True)


if __name__ == "__main__":
    main()
