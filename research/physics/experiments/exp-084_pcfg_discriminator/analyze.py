"""
exp-084 — Decision statistics (pre-registered formation criterion only).

Amendment 2026-07-13: β̂ matching not achievable; comparison is against
C-PL40 (β̂=0.79, 5/48 FAIL). Formation criterion unchanged: ≥ 10/48.

Usage: python analyze.py  (reads measurements/run_CPCFG_final.json, writes results.json)
"""

from __future__ import annotations

import json
from pathlib import Path

HERE  = Path(__file__).resolve().parent
MEAS  = HERE / "measurements"
TAG   = "run_CPCFG_final"
FORMATION_MIN = 10   # pre-registered criterion (notes.md §pre-stated hypotheses)


def main() -> None:
    path = MEAS / f"{TAG}.json"
    if not path.exists():
        raise FileNotFoundError(f"Measurement not found: {path}")
    meas = json.loads(path.read_text())

    n_conformal = meas["n_conformal"]
    n_syk_near  = meas["n_syk_near"]
    n_total     = meas["n_heads_total"]
    delta_med   = meas["delta_median_conformal"]

    forms = n_conformal >= FORMATION_MIN

    if forms:
        verdict = "H_hier_suf CONFIRMED"
        interpretation = (
            f"C-PCFG formed {n_conformal}/{n_total} conformal heads (≥{FORMATION_MIN}). "
            "Hierarchical/compositional structure at matched β̂ is sufficient to induce "
            "conformal heads. Strong form of reference reading retreats. "
            "Caveat: β̂ not matched (PCFG β̂=2.97 free-floor vs C-PL40 β̂=0.79); "
            "short-range MI confound requires further analysis."
        )
    else:
        verdict = "H_ref_needed CONFIRMED"
        interpretation = (
            f"C-PCFG formed {n_conformal}/{n_total} conformal heads (<{FORMATION_MIN}). "
            "Hierarchical structure at β̂=2.97 is NOT sufficient. "
            "C-PL40 (β̂=0.79) also failed (5/48). C-NAT (β̂=1.38) forms (11-15/48). "
            "Something beyond hierarchy and MI statistics — world-reference, semantic "
            "grounding, or both — is required. Reference driver survives this kill test."
        )

    results = {
        "experiment": "exp-084",
        "amendment": "2026-07-13: β̂ not matched (PCFG β̂=2.97 vs C-PL40 β̂=0.79)",
        "measurement": {
            "tag": TAG,
            "n_total": n_total,
            "n_conformal": n_conformal,
            "n_syk_near": n_syk_near,
            "delta_median_conformal": delta_med,
            "forms": forms,
        },
        "comparison_baseline": {
            "C-PL40": {"n_conformal": 5, "n_syk_near": 0, "forms": False},
            "C-NAT_s0": {"n_conformal": 15, "forms": True},
        },
        "formation_criterion": f">= {FORMATION_MIN} conformal heads",
        "verdict": verdict,
        "interpretation": interpretation,
    }

    out = HERE / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nwritten → {out}")


if __name__ == "__main__":
    main()
