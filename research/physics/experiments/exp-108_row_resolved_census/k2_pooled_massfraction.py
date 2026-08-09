"""K2 — where does the pooled A-profile's mass sit?

Tests claim C2 (composite row: n-independent near field + power-law tail)
against the exp-104 saved pooled profiles. First inspection of these values.

Registered prediction: on the SYK-window heads (|Delta_A - 0.25| <= 0.05,
R2_A >= 0.90 from results_gpt2.json), the near-field mass fraction (lags 0-7
of the normalized pooled profile, final 8 lag bins reported separately as
sink-adjacent) lies in [0.3, 0.7]. Kill: < 0.15 or > 0.85 on the population.
"""
import json
from pathlib import Path

import numpy as np

E104 = Path(__file__).resolve().parent.parent / "exp-104_bilocal_object_identification"
prof = np.load(E104 / "profiles_gpt2.npz")["A"]          # (12, 12, 512), summed over 50 inputs
res = json.loads((E104 / "results_gpt2.json").read_text())

R2_MIN, D_LO, D_HI = 0.90, 0.20, 0.30
syk = [(h["layer"], h["head"], h["delta_A"], h["r2_A"]) for h in res["heads"]
       if h["r2_A"] is not None and h["r2_A"] >= R2_MIN
       and h["delta_A"] is not None and D_LO <= h["delta_A"] <= D_HI]
conf = [(h["layer"], h["head"], h["delta_A"], h["r2_A"]) for h in res["heads"]
        if h["r2_A"] is not None and h["r2_A"] >= R2_MIN
        and h["delta_A"] is not None and h["delta_A"] >= 0.05]

def massfractions(l, h):
    p = prof[l, h].astype(float)
    total = p.sum()
    a = p / total
    return {
        "raw_sum_over_50_inputs": float(total),
        "near_field_lags_0_7": float(a[:8].sum()),
        "fit_window_lags_8_256": float(a[8:257].sum()),
        "mid_lags_257_503": float(a[257:504].sum()),
        "sink_adjacent_final8": float(a[504:].sum()),
    }

out = {"population_def": {"r2_min": R2_MIN, "delta_window": [D_LO, D_HI]},
       "n_syk_window": len(syk), "heads": []}
print(f"SYK-window heads (n={len(syk)}), conformal heads (n={len(conf)})")
nf = []
for l, h, d, r2 in syk:
    m = massfractions(l, h)
    m.update({"layer": l, "head": h, "delta_A": d, "r2_A": r2})
    out["heads"].append(m)
    nf.append(m["near_field_lags_0_7"])
    print(f"L{l}H{h}  Delta_A={d:.3f}  near0-7={m['near_field_lags_0_7']:.3f}  "
          f"window={m['fit_window_lags_8_256']:.3f}  mid={m['mid_lags_257_503']:.3f}  "
          f"sink8={m['sink_adjacent_final8']:.3f}  rawsum={m['raw_sum_over_50_inputs']:.1f}")

med = float(np.median(nf)) if nf else None
verdict = (None if med is None else
           "C2 DEAD (too little near-field mass)" if med < 0.15 else
           "C2 DEAD (too much near-field mass)" if med > 0.85 else
           "C2 SURVIVES" if 0.3 <= med <= 0.7 else
           "OUTSIDE PREDICTION BAND but not killed")
out["median_near_field_syk_window"] = med
out["verdict"] = verdict
print(f"\nmedian near-field mass (SYK window): {med}\nverdict: {verdict}")

# context: same table for the whole conformal population
cf = [massfractions(l, h)["near_field_lags_0_7"] for l, h, _, _ in conf]
out["conformal_population_near_field"] = {
    "n": len(cf), "median": float(np.median(cf)) if cf else None,
    "q25": float(np.percentile(cf, 25)) if cf else None,
    "q75": float(np.percentile(cf, 75)) if cf else None,
}
print(f"conformal population near-field: median={out['conformal_population_near_field']['median']:.3f} "
      f"IQR=[{out['conformal_population_near_field']['q25']:.3f}, "
      f"{out['conformal_population_near_field']['q75']:.3f}]")

with open(Path(__file__).with_suffix(".json"), "w") as f:
    json.dump(out, f, indent=1)
