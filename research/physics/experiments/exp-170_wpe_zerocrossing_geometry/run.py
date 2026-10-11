"""
exp-170 — wpe zero-crossing geometry (analysis-only).

Context:
  exp-169 showed wpe-through-LN accounts for 74-102% of S_abskey trough depth.
  exp-168 computed S_wpe_lin[a] = qbar · (wpe[a] @ W_K^T) / sqrt(d) — the linear
  (pre-LN) wpe contribution — which crosses zero near position 511. This experiment
  traces *why*: is the zero crossing a direction phenomenon (cosine between k_wpe and
  qbar changes sign) or something else?

Pre-registration: prereg.md (committed 9f6cfeb before this script was written).

Usage:
    .venv/bin/python3 research/physics/experiments/exp-170_wpe_zerocrossing_geometry/run.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
EXPS_DIR   = SCRIPT_DIR.parent

EXP164_DIR = EXPS_DIR / "exp-164_l3h4_bilinear_decomposition"
EXP168_DIR = EXPS_DIR / "exp-168_ln_recency_hj_fit"

KBAR_PATH    = EXP164_DIR / "kbar_qbar.npz"
EXP168_RES   = EXP168_DIR / "results" / "results.json"

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────

STRUCTURAL_HEADS = ["L2H1", "L3H4", "L5H0", "L7H11", "L10H8"]

HEAD_INDEX = {
    "L2H1":  (2, 1),
    "L3H4":  (3, 4),
    "L5H0":  (5, 0),
    "L7H11": (7, 11),
    "L10H8": (10, 8),
}

D_HEAD  = 64
D_MODEL = 768
SEQ_LEN = 1024

MQ_START, MQ_END = 256, 1024   # mean query pooling window (exp-164 protocol)
ZERO_CROSS_LO, ZERO_CROSS_HI = 400, 624   # ±100 window around 511

# ── helpers ───────────────────────────────────────────────────────────────────

def load_kbar_qbar() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load kbar and qbar from exp-164 npz; compute mean query per head."""
    data = np.load(KBAR_PATH)
    kbar = {h: data[f"kbar_{h}"] for h in STRUCTURAL_HEADS}
    qbar = {h: data[f"qbar_{h}"][MQ_START:MQ_END].mean(axis=0)
            for h in STRUCTURAL_HEADS}
    return kbar, qbar


def extract_wpe_wk(model) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Extract wpe [SEQ_LEN, D_MODEL] and W_K per head [D_MODEL, D_HEAD]."""
    import torch
    with torch.no_grad():
        wpe = model.transformer.wpe.weight.cpu().numpy()  # [1024, 768]

        wk = {}
        for h_name, (layer, head) in HEAD_INDEX.items():
            c_attn_w = model.transformer.h[layer].attn.c_attn.weight  # [768, 2304]
            col_start = D_MODEL + head * D_HEAD
            col_end   = D_MODEL + (head + 1) * D_HEAD
            wk[h_name] = c_attn_w[:, col_start:col_end].cpu().numpy()  # [768, 64]

    return wpe, wk


def find_zero_crossings(profile: np.ndarray) -> list[int]:
    """Find positions where sign changes: sign(profile[a]) != sign(profile[a+1])."""
    crossings = []
    for a in range(len(profile) - 1):
        if profile[a] * profile[a + 1] < 0:
            crossings.append(a)
    return crossings


def monotone_check(arr: np.ndarray) -> bool:
    """True if arr has no local extremum (fully monotone)."""
    diffs = np.diff(arr)
    return bool(np.all(diffs >= 0) or np.all(diffs <= 0))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import torch
    from transformers import GPT2LMHeadModel

    t0 = time.time()
    print("exp-170 | wpe zero-crossing geometry")
    print("Pre-registration: prereg.md (9f6cfeb)")
    print()

    # ── Step 1: load kbar/qbar from exp-164 ───────────────────────────────────
    print("Loading kbar_qbar.npz from exp-164 ...")
    kbar, qbar = load_kbar_qbar()
    for h in STRUCTURAL_HEADS:
        assert kbar[h].shape == (SEQ_LEN, D_HEAD), f"K2-shape: kbar[{h}] bad"
        assert qbar[h].shape == (D_HEAD,), f"K2-shape: qbar[{h}] bad"
    print(f"  Loaded qbar for {len(STRUCTURAL_HEADS)} heads OK")

    # ── Step 2: load GPT-2 small weights ──────────────────────────────────────
    print("\nLoading GPT-2 small weights ...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    wpe, wk = extract_wpe_wk(model)
    print(f"  wpe shape: {wpe.shape}")

    # ── Step 3: load exp-168 S_wpe values for K1 cross-check ─────────────────
    print("\nLoading exp-168 results for K1 cross-check ...")
    with open(EXP168_RES) as f:
        exp168_res = json.load(f)
    # exp-168 stores per_head results with s_wpe as a per-position list
    exp168_s_wpe = {}
    for h in STRUCTURAL_HEADS:
        # exp-168 results may store this; if not, skip K1 (field may vary)
        hdata = exp168_res.get("per_head", {}).get(h, {})
        if "s_wpe" in hdata:
            exp168_s_wpe[h] = np.array(hdata["s_wpe"])
        else:
            print(f"  Note: exp-168 did not store per-position s_wpe for {h}; K1 skipped for this head")

    # ── Step 4: compute geometry per head ─────────────────────────────────────
    print("\nComputing wpe geometry ...")

    # wpe norm profile (H3)
    norm_wpe = np.linalg.norm(wpe, axis=1)   # [SEQ_LEN]
    wpe_norm_monotone = monotone_check(norm_wpe)
    wpe_norm_extrema = []
    dwpe = np.diff(norm_wpe)
    for a in range(len(dwpe) - 1):
        if dwpe[a] * dwpe[a + 1] < 0:
            wpe_norm_extrema.append(a + 1)

    print(f"\n  wpe norm profile:")
    print(f"    min={norm_wpe.min():.4f}, max={norm_wpe.max():.4f}, "
          f"mean={norm_wpe.mean():.4f}")
    print(f"    monotone: {wpe_norm_monotone}")
    print(f"    n_local_extrema: {len(wpe_norm_extrema)}")
    if wpe_norm_extrema:
        print(f"    extrema positions (first 10): {wpe_norm_extrema[:10]}")

    # Per-head analysis
    per_head: dict[str, dict] = {}
    for h in STRUCTURAL_HEADS:
        q = qbar[h]                      # [64]
        q_hat = q / np.linalg.norm(q)   # unit query direction

        # k_wpe[a] = wpe[a] @ W_K^T  (W_K stored as [D_MODEL, D_HEAD])
        k_wpe = wpe @ wk[h]             # [1024, 64]

        # S_wpe_raw (H4) — same formula as exp-168
        s_wpe_raw = (k_wpe @ q) / np.sqrt(D_HEAD)  # [1024]

        # norm of k_wpe (H2)
        norm_k = np.linalg.norm(k_wpe, axis=1)     # [1024]

        # cosine between k_wpe[a] and q (H1)
        k_wpe_hat = k_wpe / norm_k[:, np.newaxis]   # [1024, 64] unit vectors
        cos_profile = k_wpe_hat @ q_hat             # [1024]

        # Zero crossings
        s_wpe_crossings    = find_zero_crossings(s_wpe_raw)
        cos_crossings      = find_zero_crossings(cos_profile)
        norm_k_crossings   = find_zero_crossings(norm_k)  # should be empty (K2)

        # Crossings within the ±100 window around 511
        s_wpe_in_window   = [c for c in s_wpe_crossings if ZERO_CROSS_LO <= c <= ZERO_CROSS_HI]
        cos_in_window     = [c for c in cos_crossings    if ZERO_CROSS_LO <= c <= ZERO_CROSS_HI]

        # K1: cross-check with exp-168 S_wpe
        k1_max_diff = None
        if h in exp168_s_wpe:
            diff = np.abs(s_wpe_raw - exp168_s_wpe[h])
            k1_max_diff = float(diff.max())
            if k1_max_diff > 1e-4:
                print(f"  K1 FIRES for {h}: max diff = {k1_max_diff:.6f} — data integrity issue")

        # K2: no zero crossing in norm_k
        if norm_k_crossings:
            print(f"  K2 FIRES for {h}: norm_k has zero crossings at {norm_k_crossings[:5]}")

        # H1: cosine crosses zero within ±100 of 511
        h1 = len(cos_in_window) > 0
        # H4: S_wpe_raw crosses zero within ±100 of 511
        h4 = len(s_wpe_in_window) > 0

        print(f"\n  {h}:")
        print(f"    norm_k: min={norm_k.min():.4f}, max={norm_k.max():.4f}")
        print(f"    S_wpe_raw zero crossings: {s_wpe_crossings[:5]}, in-window: {s_wpe_in_window}")
        print(f"    cos_profile zero crossings: {cos_crossings[:5]}, in-window: {cos_in_window}")
        print(f"    H1 (cos crosses in window): {h1}")
        print(f"    H4 (S_wpe crosses in window): {h4}")
        if k1_max_diff is not None:
            print(f"    K1 max diff vs exp-168: {k1_max_diff:.2e}")

        # Trough characterization
        trough_region = s_wpe_raw[1:511]
        trough_min_pos = int(np.argmin(trough_region)) + 1
        trough_depth   = float(-trough_region.min())  # depth = -min (trough is negative)
        first_crossing = s_wpe_crossings[0] if s_wpe_crossings else None

        per_head[h] = {
            "norm_k_min":             float(norm_k.min()),
            "norm_k_max":             float(norm_k.max()),
            "norm_k_mean":            float(norm_k.mean()),
            "norm_k_zero_crossings":  norm_k_crossings,
            "s_wpe_raw_zero_crossings":  s_wpe_crossings,
            "s_wpe_raw_in_window":       s_wpe_in_window,
            "cos_zero_crossings":        cos_crossings,
            "cos_in_window":             cos_in_window,
            "h1_confirmed":              h1,
            "h4_confirmed":              h4,
            "trough_min_position":       trough_min_pos,
            "trough_depth_raw":          trough_depth,
            "first_s_wpe_crossing":      first_crossing,
            "k1_max_diff_vs_exp168":     k1_max_diff,
        }

    # ── Step 5: H3 verdict ────────────────────────────────────────────────────
    h3 = len(wpe_norm_extrema) > 0

    # ── Step 6: aggregate verdicts ────────────────────────────────────────────
    h1_confirmed = all(per_head[h]["h1_confirmed"] for h in STRUCTURAL_HEADS)
    h4_confirmed = all(per_head[h]["h4_confirmed"] for h in STRUCTURAL_HEADS)
    k2_fired     = any(per_head[h]["norm_k_zero_crossings"] for h in STRUCTURAL_HEADS)

    print("\n── Summary ────────────────────────────────────────────────────────────")
    print(f"  H1 (cos crosses within ±100 of 511, all heads): {h1_confirmed}")
    print(f"  H2 (norm not the driver — no zero crossings in norm_k): {not k2_fired}")
    print(f"  H3 (wpe norm has local extrema, not monotone): {h3}")
    print(f"  H4 (S_wpe_raw crosses within ±100 of 511, all heads): {h4_confirmed}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "experiment": "exp-170",
        "prereg_commit": "9f6cfeb",
        "prereg_evidence": "git-attested — prereg.md committed before run.py was written",
        "elapsed_s": round(elapsed, 1),
        "wpe_norm_profile": {
            "min":           float(norm_wpe.min()),
            "max":           float(norm_wpe.max()),
            "mean":          float(norm_wpe.mean()),
            "is_monotone":   wpe_norm_monotone,
            "n_local_extrema": len(wpe_norm_extrema),
            "extrema_positions_first_20": wpe_norm_extrema[:20],
        },
        "per_head": per_head,
        "verdicts": {
            "H1_cos_crosses_in_window_all_heads": h1_confirmed,
            "H2_norm_not_the_driver":             not k2_fired,
            "H3_wpe_norm_has_extrema":            h3,
            "H4_s_wpe_raw_crosses_in_window_all": h4_confirmed,
        },
    }

    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
