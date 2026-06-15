"""
exp-067 — v2 pre-registration: Test B (depth-accumulated primacy laws).

Pre-registration: research/physics/notes/2026-06-14_v2_preregistration.md

Tests:
  P-B1 (Law A, primary): primacy grows with depth k.
      Spearman ρ(k, prim_decile) ≥ 0.90 on Pythia-1.4b.
  P-B2 (Law B, primary): start-window mass decreases with context N.
      Spearman ρ(N, prim_window(8)) ≤ −0.90 on Pythia-1.4b.
  P-B3 (Law A, robustness): same as P-B1 on GPT-2-medium. ρ ≥ 0.80.

Fresh models: Pythia-1.4b and GPT-2-medium were NOT used in exp-065/066
(LongChat was the derivation model).

All computation uses vectorized numpy. Composition tracked via single-row
propagation: row_k = row_{k-1} @ M_k, O(N log N) per step via scipy.signal.
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

ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent

# Pre-registration thresholds (from 2026-06-14_v2_preregistration.md)
KEEP_B1_RHO = 0.90
KILL_B1_RHO = 0.50
KEEP_B2_RHO = -0.90
KILL_B2_RHO = -0.50
KEEP_B3_RHO = 0.80

N_INPUTS = 50
SEQ_LEN_DEPTH = 512
SEQ_LENS_CONTEXT = [256, 512, 1024, 2048]
WINDOW_W = 8
K_CHECKPOINTS = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24]


def lag_profile_from_attn(avg_attn: np.ndarray, N: int) -> np.ndarray:
    """
    avg_attn: (N, N) mean attention matrix (averaged over heads and inputs).
    Returns profile (N,): profile[u-1] = mean attention at lag u = q - j,
    averaged over deep queries q >= N//2.
    """
    q_lo = N // 2
    profile = np.zeros(N)
    for u in range(1, N):
        q_min = max(q_lo, u)
        if q_min >= N:
            break
        q_arr = np.arange(q_min, N)
        j_arr = q_arr - u
        profile[u - 1] = avg_attn[q_arr, j_arr].mean()
    # Normalize
    s = profile.sum()
    if s > 0:
        profile /= s
    return profile


def compose_step(row: np.ndarray, profile: np.ndarray, N: int) -> np.ndarray:
    """
    Apply one layer's row-stochastic kernel to the current row vector.

    M[q, j] ∝ profile[q-j] for j = 0..q (lag = q-j = 1..q+1; profile[lag-1]).
    Actually profile index: lag = q - j, so j = q - lag, profile[lag - 1] = profile[q-j-1].
    But profile[u-1] where u = lag = q - j.

    new_row[j] = sum_{q >= j} row[q] * M[q, j]
              = sum_{q >= j} row[q] * profile[q-j-1] / cumsum_profile[q]
    Wait: let's use 0-indexed lags in profile. profile[0] is lag-1 (closest key).
    Lag = q - j means: j = q - lag, for lag in 1..q. profile[lag-1] = profile[q-j-1].

    Hmm: redefine. From exp-066 convention: lag u = q - j (not +1). So u = 0 is
    self-attention (j=q). profile[u] = weight at lag u, for u = 0..N-1.

    M[q, j] ∝ profile[q-j], j = 0..q.

    new_row[j] = sum_{q=j}^{N-1} (row[q] / sum_q') * profile[q - j]
    where sum_q' = sum_{u=0}^{q} profile[u] = cum_profile[q].

    Using cross-correlation:
    new_row[j] = sum_{u=0}^{N-1-j} norm_row[j+u] * profile[u]
    (substituting u = q - j)
    = correlate(norm_row, profile)[N-1+j]  (full mode, at lag j)

    Actually np.correlate(a, b, mode='full')[k] = sum_i a[i] * b[i - (k - (len(a)-1))]
    At k = N-1+j: sum_i norm_row[i] * b[i - j]. Let v = i - j:
    = sum_v norm_row[v+j] * profile[v].  Yes!
    """
    # Cumulative profile for per-row normalization
    cum_prof = np.cumsum(profile)  # shape (N,)
    # Avoid division by zero; if cum_prof[q] = 0, self-loop at q
    safe_cum = np.where(cum_prof > 1e-15, cum_prof, 1.0)
    norm_row = row / safe_cum  # norm_row[q] = row[q] / sum(profile[:q+1])
    # Zero out where cum_prof = 0 (self-loop; mass stays at q, no redistribution)
    # Actually handle self-loop by adding back: if cum_prof[q] = 0, keep mass at q
    self_loop_mask = cum_prof <= 1e-15
    self_loop_mass = row * self_loop_mask.astype(float)

    # Vectorized cross-correlation
    # np.correlate(a, b, 'full') at index k = N-1+j gives sum_i a[i]*b[i-j]
    # We need sum_v norm_row[j+v]*profile[v] = correlate at index N-1+j
    corr = fftconvolve(norm_row[::-1], profile, mode='full')
    # corr has length 2N-1; index for lag j is N-1+j... let me verify
    # fftconvolve(a[::-1], b) at index i: sum_k a[N-1-k] * b[i-k]
    # Let k' = N-1-k: sum_{k'} a[k'] * b[i - (N-1-k')] = sum_{k'} a[k'] * b[i-N+1+k']
    # We want sum_v norm_row[j+v] * profile[v].
    # Set k' = j+v, v = k'-j: sum_{k'>=j} norm_row[k'] * profile[k'-j]
    # = sum_{k'>=j} norm_row[k'] * b[k'-j]   (b = profile)
    # In the convolution: at i = ? sum_k' a[k'] * b[i-N+1+k'] = sum_k' norm_row[k'] * profile[i-N+1+k']
    # We need i-N+1+k' = k'-j → i-N+1 = -j → i = N-1-j.
    # So: corr[N-1-j] = sum_{k'>=j} norm_row[k'] * profile[k'-j]  (for the truncated, valid part)
    # → new_row[j] = corr[N-1-j]
    new_row = corr[N - 1 - np.arange(N)[::-1]]  # corr[N-1] for j=0, corr[0] for j=N-1
    # Equivalently: new_row = corr[np.arange(N-1, 2*N-1)]? Let me recheck.
    # new_row[j] = corr[N-1-j]. For j=0: corr[N-1]. For j=N-1: corr[0].
    # So new_row[j] = corr[N-1-j], i.e., new_row = corr[N-1::-1]  (reversed first N elements)
    # Wait: corr[N-1-j] for j=0..N-1 = corr[N-1], corr[N-2], ..., corr[0] = corr[:N][::-1]
    new_row = corr[:N][::-1].copy()

    # Add back self-loop mass
    new_row += self_loop_mass

    # Clip small negatives from numerical noise
    np.clip(new_row, 0, None, out=new_row)

    # Renormalize (should sum to ~1 already; numerical safeguard)
    s = new_row.sum()
    if s > 1e-10:
        new_row /= s

    return new_row


def measure_profiles_all_layers(
    model: object,
    N: int,
    n_inputs: int,
    device: str,
) -> list[np.ndarray]:
    """
    Run n_inputs random-token forward passes, return list of n_layers lag profiles.
    Each profile is a length-N array normalized to sum=1.
    """
    n_layers = model.config.num_hidden_layers
    vocab_size = model.config.vocab_size

    # Accumulate averaged attention (one (N,N) per layer)
    avg_attn = [np.zeros((N, N), dtype=np.float64) for _ in range(n_layers)]
    for _ in range(n_inputs):
        input_ids = torch.randint(0, vocab_size, (1, N), device=device)
        with torch.no_grad():
            out = model(input_ids, output_attentions=True)
        for l, attn in enumerate(out.attentions):
            # attn: (1, n_heads, N, N)
            avg_attn[l] += attn[0].mean(dim=0).cpu().float().numpy()

    profiles = []
    for l in range(n_layers):
        avg_attn[l] /= n_inputs
        profiles.append(lag_profile_from_attn(avg_attn[l], N))
    return profiles


def run_depth_sweep(
    profiles: list[np.ndarray],
    N: int,
    k_checkpoints: list[int],
) -> dict:
    """Compose layers 1..k, record prim_decile at each k in k_checkpoints."""
    n_layers = len(profiles)
    row = np.zeros(N)
    row[N - 1] = 1.0  # start: all mass at deepest query position

    by_k = {}
    for k in range(1, n_layers + 1):
        row = compose_step(row, profiles[k - 1], N)
        if k in k_checkpoints:
            by_k[k] = {
                "prim_decile": float(row[:max(1, N // 10)].sum()),
                "prim_window": float(row[:WINDOW_W].sum()),
            }
    return by_k


def run_model(model_name: str, run_b2: bool) -> dict:
    short = model_name.split("/")[-1]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n{'='*60}", flush=True)
    print(f"Model: {short}  (device={device})", flush=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        output_attentions=True,
    )
    model.eval().to(device)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"  {n_layers}L/{n_heads}H  loaded in {time.time()-t0:.1f}s", flush=True)

    model_result: dict = {"n_layers": n_layers, "n_heads": n_heads}
    k_list = [k for k in K_CHECKPOINTS if k <= n_layers]

    # --- P-B1 / P-B3: depth sweep at SEQ_LEN_DEPTH ---
    print(f"  [depth sweep] N={SEQ_LEN_DEPTH}, measuring profiles...", flush=True)
    t1 = time.time()
    profiles_depth = measure_profiles_all_layers(model, SEQ_LEN_DEPTH, N_INPUTS, device)
    print(f"    profiles measured in {time.time()-t1:.1f}s", flush=True)
    by_k = run_depth_sweep(profiles_depth, SEQ_LEN_DEPTH, k_list)

    ks = np.array(k_list)
    prims = np.array([by_k[k]["prim_decile"] for k in k_list])
    rho_b1, p_b1 = spearmanr(ks, prims)
    rho_b1, p_b1 = float(rho_b1), float(p_b1)

    keep_thresh = KEEP_B1_RHO if run_b2 else KEEP_B3_RHO
    label = "P-B1" if run_b2 else "P-B3"
    if rho_b1 >= keep_thresh:
        verdict_b1 = f"KEEP ({label})"
    elif rho_b1 < KILL_B1_RHO:
        verdict_b1 = f"KILL ({label})"
    else:
        verdict_b1 = f"AMBIGUOUS ({label})"

    print(f"  {label}: ρ(k, prim_decile) = {rho_b1:+.4f}  → {verdict_b1}", flush=True)
    for k in k_list:
        print(f"    k={k:2d}: prim_decile={by_k[k]['prim_decile']:.5f}  prim_window={by_k[k]['prim_window']:.5f}")

    model_result["pb1"] = {
        "N": SEQ_LEN_DEPTH, "k_values": k_list,
        "prim_decile_by_k": [by_k[k]["prim_decile"] for k in k_list],
        "prim_window_by_k": [by_k[k]["prim_window"] for k in k_list],
        "rho_k_prim_decile": rho_b1, "p": p_b1,
        "verdict": verdict_b1,
    }

    # --- P-B2: context-length sweep (Pythia-1.4b only) ---
    if run_b2:
        print(f"  [context sweep] N ∈ {SEQ_LENS_CONTEXT}...", flush=True)
        b2_by_N: dict = {}
        for N in SEQ_LENS_CONTEXT:
            print(f"    N={N}...", flush=True)
            t2 = time.time()
            profs_N = measure_profiles_all_layers(model, N, N_INPUTS, device)
            k_full = list(range(1, n_layers + 1))
            by_k_N = run_depth_sweep(profs_N, N, k_full)
            pw = by_k_N[n_layers]["prim_window"]
            pd = by_k_N[n_layers]["prim_decile"]
            b2_by_N[N] = {"prim_window": pw, "prim_decile": pd}
            print(f"      prim_window(8)={pw:.5f}  prim_decile={pd:.5f}  ({time.time()-t2:.1f}s)", flush=True)

        Ns = np.array(SEQ_LENS_CONTEXT)
        pws = np.array([b2_by_N[N]["prim_window"] for N in SEQ_LENS_CONTEXT])
        rho_b2, p_b2 = spearmanr(Ns, pws)
        rho_b2, p_b2 = float(rho_b2), float(p_b2)
        if rho_b2 <= KEEP_B2_RHO:
            verdict_b2 = "KEEP (P-B2)"
        elif rho_b2 > KILL_B2_RHO:
            verdict_b2 = "KILL (P-B2)"
        else:
            verdict_b2 = "AMBIGUOUS (P-B2)"

        print(f"  P-B2: ρ(N, prim_window(8)) = {rho_b2:+.4f}  → {verdict_b2}", flush=True)
        model_result["pb2"] = {
            "N_values": SEQ_LENS_CONTEXT,
            "prim_window_by_N": [b2_by_N[N]["prim_window"] for N in SEQ_LENS_CONTEXT],
            "prim_decile_by_N": [b2_by_N[N]["prim_decile"] for N in SEQ_LENS_CONTEXT],
            "by_N": {str(N): b2_by_N[N] for N in SEQ_LENS_CONTEXT},
            "rho_N_prim_window": rho_b2, "p": p_b2,
            "verdict": verdict_b2,
        }

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return model_result


def main() -> None:
    t_start = time.time()
    results: dict = {
        "experiment": "exp-067-test-b",
        "preregistration": "research/physics/notes/2026-06-14_v2_preregistration.md",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "keep_b1_rho": KEEP_B1_RHO, "kill_b1_rho": KILL_B1_RHO,
            "keep_b2_rho": KEEP_B2_RHO, "kill_b2_rho": KILL_B2_RHO,
            "keep_b3_rho": KEEP_B3_RHO,
        },
        "models": {}
    }

    results["models"]["EleutherAI/pythia-1.4b"] = run_model(
        "EleutherAI/pythia-1.4b", run_b2=True)

    results["models"]["openai-community/gpt2-medium"] = run_model(
        "openai-community/gpt2-medium", run_b2=False)

    elapsed = time.time() - t_start
    out_path = OUT_DIR / "results_test_b.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[{elapsed:.0f}s] Results → {out_path}")

    print("\n=== SUMMARY ===")
    for m, mr in results["models"].items():
        short = m.split("/")[-1]
        pb1 = mr.get("pb1", {})
        print(f"  {short}:")
        print(f"    P-B1/B3: ρ(k, prim_decile)={pb1.get('rho_k_prim_decile','?'):+.4f}  {pb1.get('verdict','?')}")
        if "pb2" in mr:
            pb2 = mr["pb2"]
            print(f"    P-B2: ρ(N, prim_window)={pb2.get('rho_N_prim_window','?'):+.4f}  {pb2.get('verdict','?')}")


if __name__ == "__main__":
    main()
