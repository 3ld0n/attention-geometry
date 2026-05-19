"""Per-layer BCFT diagnostic across the Pythia family + matched controls.

Reads cached per-head BCFT data from research/physics/results/ (the JSON
files written by bcft_pre_registered_run.py) and computes per-layer
Spearman correlation between Δ and valley_depth, restricted to conformal
heads (R² >= 0.85, Δ >= 0.05).

Purpose: localize where the BCFT prediction succeeds and fails inside each
model. Pythia-2.8B was the only falsifying case in the pre-registered run;
this diagnostic is intended to characterize the failure structure rather
than re-test the framework.

Output:
  - Console table of per-layer ρ for each model.
  - JSON with per-layer values for downstream plotting/writeup.
  - PNG plot: per-layer ρ vs relative depth, one line per model.

Usage:
  .venv/bin/python3 research/physics/pythia_per_layer_diagnostic.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "research" / "physics" / "results"

CONFORMAL_R2 = 0.85
CONFORMAL_DELTA = 0.05
MIN_HEADS_PER_LAYER = 5  # below this, ρ is not meaningful


MODELS_OF_INTEREST = [
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/gpt-neo-2.7B",
    "Qwen/Qwen2-7B",
    "allenai/OLMo-7B-hf",
    "mistralai/Mistral-7B-v0.3",
]


def load_all_results() -> dict:
    """Merge all bcft_pre_registered_run_*.json files into one dict by model."""
    merged: dict = {}
    for f in sorted(RESULTS_DIR.glob("bcft_pre_registered_run_*.json")):
        with open(f) as fp:
            data = json.load(fp)
        for name, md in data.get("models", {}).items():
            merged[name] = md
    return merged


def per_layer_correlation(model_data: dict) -> list[dict]:
    """For each layer, compute Spearman ρ between Δ and valley_depth on
    conformal heads. Returns a list of dicts, one per layer."""
    n_layers = model_data["n_layers"]
    heads = model_data.get("heads", [])

    by_layer: dict[int, list[dict]] = {l: [] for l in range(n_layers)}
    for h in heads:
        by_layer[h["layer"]].append(h)

    rows = []
    for l in range(n_layers):
        all_heads = by_layer[l]
        conformal = [
            h for h in all_heads
            if h["R2"] >= CONFORMAL_R2 and h["delta"] >= CONFORMAL_DELTA
        ]
        n_conf = len(conformal)
        if n_conf >= MIN_HEADS_PER_LAYER:
            deltas = np.array([h["delta"] for h in conformal])
            valleys = np.array([h["valley_depth"] for h in conformal])
            rho, p = spearmanr(deltas, valleys)
            rho_val = float(rho)
            p_val = float(p)
        else:
            rho_val = float("nan")
            p_val = float("nan")
        rows.append({
            "layer": l,
            "rel_depth": l / max(1, n_layers - 1),
            "n_heads_total": len(all_heads),
            "n_conformal": n_conf,
            "rho": rho_val,
            "p": p_val,
        })
    return rows


def format_layer_table(model_name: str, model_data: dict, rows: list[dict]) -> str:
    """Pretty-print per-layer results."""
    lines = []
    n_layers = model_data["n_layers"]
    n_conformal_total = model_data["n_conformal_heads"]
    pooled_rho = model_data["spearman_rho"]
    pooled_p = model_data["spearman_p"]
    verdict = model_data["verdict"]
    lines.append(
        f"\n=== {model_name} "
        f"(layers={n_layers}, conformal_total={n_conformal_total}, "
        f"pooled ρ={pooled_rho:+.4f}, p={pooled_p:.2e}, {verdict}) ==="
    )
    lines.append(f"  {'L':>3}  {'rel':>5}  {'n_conf':>6}  {'ρ':>8}  {'p':>10}")
    for r in rows:
        rho = r["rho"]
        p = r["p"]
        rho_s = f"{rho:+.3f}" if rho == rho else "  nan "
        p_s = f"{p:.2e}" if p == p else "      n/a "
        flag = ""
        if rho == rho:
            if rho < 0 and p < 0.05:
                flag = "  *NEG*"
            elif rho >= 0.50 and p < 1e-3:
                flag = "  STRONG"
        lines.append(
            f"  {r['layer']:>3}  {r['rel_depth']:>5.2f}  "
            f"{r['n_conformal']:>6}  {rho_s:>8}  {p_s:>10}{flag}"
        )
    return "\n".join(lines)


def summarize(model_name: str, rows: list[dict]) -> dict:
    """Compute summary stats: fraction of layers positive, fraction
    significantly negative, mean ρ, etc."""
    valid = [r for r in rows if r["rho"] == r["rho"]]
    if not valid:
        return {
            "model": model_name,
            "n_layers_with_signal": 0,
            "frac_positive": float("nan"),
            "frac_strong_positive": float("nan"),
            "frac_significant_negative": float("nan"),
            "mean_rho": float("nan"),
            "median_rho": float("nan"),
        }
    rhos = np.array([r["rho"] for r in valid])
    ps = np.array([r["p"] for r in valid])
    n = len(rhos)
    return {
        "model": model_name,
        "n_layers_with_signal": n,
        "frac_positive": float(np.mean(rhos > 0)),
        "frac_strong_positive": float(np.mean((rhos >= 0.50) & (ps < 1e-3))),
        "frac_significant_negative": float(np.mean((rhos < 0) & (ps < 0.05))),
        "mean_rho": float(np.mean(rhos)),
        "median_rho": float(np.median(rhos)),
        "first_negative_rel_depth": (
            float(min((r["rel_depth"] for r in valid if r["rho"] < 0), default=float("nan")))
        ),
    }


def maybe_plot(per_model_rows: dict[str, list[dict]], output_path: Path):
    """Save a per-layer ρ plot (one line per model). Skip cleanly if
    matplotlib isn't available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    pythia_models = [m for m in per_model_rows if "pythia" in m.lower()]
    other_models = [m for m in per_model_rows if "pythia" not in m.lower()]

    for m in pythia_models:
        rows = per_model_rows[m]
        x = [r["rel_depth"] for r in rows if r["rho"] == r["rho"]]
        y = [r["rho"] for r in rows if r["rho"] == r["rho"]]
        ax.plot(x, y, "o-", label=m.split("/")[-1], linewidth=2)

    for m in other_models:
        rows = per_model_rows[m]
        x = [r["rel_depth"] for r in rows if r["rho"] == r["rho"]]
        y = [r["rho"] for r in rows if r["rho"] == r["rho"]]
        ax.plot(x, y, "o--", label=m.split("/")[-1], alpha=0.5)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(0.50, color="green", linewidth=0.5, linestyle=":",
               label="pre-reg threshold ρ=0.50")
    ax.set_xlabel("Relative depth (0 = first layer, 1 = last)")
    ax.set_ylabel("Per-layer Spearman ρ(Δ, valley_depth)")
    ax.set_title("Per-layer BCFT correlation across model families")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print(f"Plot saved to: {output_path}")


def main():
    if not RESULTS_DIR.exists():
        print(f"No results dir at {RESULTS_DIR}", file=sys.stderr)
        sys.exit(1)

    all_results = load_all_results()
    missing = [m for m in MODELS_OF_INTEREST if m not in all_results]
    if missing:
        print(f"WARNING: missing models {missing} — they'll be skipped")

    per_model_rows: dict[str, list[dict]] = {}
    summaries: list[dict] = []

    for m in MODELS_OF_INTEREST:
        if m not in all_results:
            continue
        rows = per_layer_correlation(all_results[m])
        per_model_rows[m] = rows
        print(format_layer_table(m, all_results[m], rows))
        summaries.append(summarize(m, rows))

    print("\n\n=== SUMMARY ===")
    print(
        f"  {'model':<35}  {'n_lyr':>5}  {'frac+':>6}  "
        f"{'frac_strong+':>12}  {'frac_sig-':>9}  {'mean_ρ':>7}  {'median_ρ':>8}"
    )
    for s in summaries:
        print(
            f"  {s['model']:<35}  {s['n_layers_with_signal']:>5}  "
            f"{s['frac_positive']:>6.2f}  {s['frac_strong_positive']:>12.2f}  "
            f"{s['frac_significant_negative']:>9.2f}  "
            f"{s['mean_rho']:>+7.3f}  {s['median_rho']:>+8.3f}"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_json = RESULTS_DIR / f"pythia_per_layer_diagnostic_{timestamp}.json"
    with open(out_json, "w") as f:
        json.dump({
            "timestamp_utc": timestamp,
            "conformal_R2_threshold": CONFORMAL_R2,
            "conformal_delta_threshold": CONFORMAL_DELTA,
            "min_heads_per_layer": MIN_HEADS_PER_LAYER,
            "per_model_rows": per_model_rows,
            "summaries": summaries,
        }, f, indent=2)
    print(f"\nJSON saved to: {out_json}")

    out_png = RESULTS_DIR / f"pythia_per_layer_diagnostic_{timestamp}.png"
    maybe_plot(per_model_rows, out_png)


if __name__ == "__main__":
    main()
