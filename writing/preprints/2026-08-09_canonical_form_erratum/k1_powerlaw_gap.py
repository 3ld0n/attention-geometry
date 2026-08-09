"""K1 — entropy gap of an exactly normalized power-law distribution.

Tests claim C1 of notes/2026-08-09_theory_of_A_entropy_gap_and_sum_rule.md
against the canonical-form paper's Sec. 8.3 formula H_gap = 2*Delta*log n.

Kill condition (committed before running): for s = 0.5, if the OLS slope of
H_gap vs log n over n in [4, 256] (the paper's own measured range) is
>= 0.25, C1 is DEAD. Slope < 0.10 confirms C1. Between: ambiguous.

Pure math. No model, no data files.
"""
import json

import numpy as np

OUT = {}
for s in (0.3, 0.5, 0.7, 1.0):
    rows = []
    for n in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        r = np.arange(1, n + 1, dtype=float)
        w = r ** (-s)
        alpha = w / w.sum()
        H = float(-(alpha * np.log(alpha)).sum())
        rows.append({"n": n, "H": H, "H_gap": float(np.log(n) - H)})
    # OLS slope of H_gap vs log n over the paper's measured range n in [4, 256]
    sub = [row for row in rows if 4 <= row["n"] <= 256]
    x = np.log([row["n"] for row in sub])
    y = np.array([row["H_gap"] for row in sub])
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    # asymptotic constant predicted by C1 for s < 1
    pred = float(np.log(1 - s) + s / (1 - s)) if s < 1 else None
    OUT[f"s={s}"] = {
        "gap_by_n": rows,
        "ols_slope_gap_vs_logn_n4_256": float(coef[1]),
        "ols_intercept": float(coef[0]),
        "paper_predicted_slope_2Delta": s,
        "C1_predicted_asymptotic_gap": pred,
    }

s5 = OUT["s=0.5"]["ols_slope_gap_vs_logn_n4_256"]
verdict = ("C1 CONFIRMED" if s5 < 0.10 else
           "C1 DEAD" if s5 >= 0.25 else "AMBIGUOUS")
OUT["verdict"] = {"slope_at_s0.5": s5, "verdict": verdict}

print(json.dumps(OUT["verdict"], indent=1))
for s in (0.3, 0.5, 0.7, 1.0):
    o = OUT[f"s={s}"]
    print(f"s={s}: measured slope={o['ols_slope_gap_vs_logn_n4_256']:.4f}  "
          f"paper predicts {o['paper_predicted_slope_2Delta']:.2f}  "
          f"C1 asymptotic gap={o['C1_predicted_asymptotic_gap']}")
    tail = o["gap_by_n"][-1]
    print(f"   H_gap(n=4096) = {tail['H_gap']:.4f}")

with open(__file__.replace(".py", ".json"), "w") as f:
    json.dump(OUT, f, indent=1)
