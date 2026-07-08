"""Aggregate exp-073 across all seeds into a single n=5 result.

Combines the validated serial seeds (42, 7 in seeds_42_7.json) with the per-seed
files produced by the deployed run_one_seed (exp073_seed<seed>.json), and reports
BOTH metrics:
  - Δ(middle correct)  [PRIMARY]  — direct count out of 40 at the middle position.
  - Δ(V_task)          [secondary] — the ratio metric, which conflates middle with
                                     edge movement (see notes.md).

Sign conventions (predicted = the QK-slope handle works):
  shallowing κ=0.5 : middle should RISE  (Δmid > 0) / V_task should FALL (ΔV < 0)
  deepening  κ=1.5 : middle should FALL  (Δmid < 0) / V_task should RISE (ΔV > 0)

  python aggregate.py
"""
import glob
import json
import os

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))


def load_seeds():
    seeds = {}
    # Load per-seed files first (MPS gate and new seeds)
    for path in glob.glob(os.path.join(here, "exp073_seed*.json")):
        s = json.load(open(path))
        seeds[s["seed"]] = s
    # Load seeds_42_7.json last so it wins for seeds 42 and 7 (Modal CUDA fp32 complete runs)
    sp = os.path.join(here, "seeds_42_7.json")
    if os.path.exists(sp):
        for s in json.load(open(sp))["per_seed"]:
            seeds[s["seed"]] = s
    return [seeds[k] for k in sorted(seeds)]


def mid_correct(s, key):
    return s["conditions"][key]["correct_by_pos"][2]


def vtask(s, key):
    return s["conditions"][key]["V_task"]


def stat(vals):
    v = np.array(vals, float)
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return {"per_seed": [round(x, 4) for x in vals], "mean": round(float(v.mean()), 4),
            "se": round(se, 4), "n": len(v)}


def main():
    per_seed = load_seeds()
    seed_ids = [s["seed"] for s in per_seed]
    print(f"=== exp-073 aggregate, n={len(per_seed)} seeds {seed_ids} ===\n")

    shallow_mid = [mid_correct(s, "target_k0.5") - mid_correct(s, "target_k1.0") for s in per_seed]
    deepen_mid = [mid_correct(s, "target_k1.5") - mid_correct(s, "target_k1.0") for s in per_seed]
    shallow_v = [vtask(s, "target_k0.5") - vtask(s, "target_k1.0") for s in per_seed]
    deepen_v = [vtask(s, "target_k1.5") - vtask(s, "target_k1.0") for s in per_seed]

    print("PRIMARY — Δ(middle correct out of 40):")
    print(f"  shallow κ=0.5 (predicted >0): {stat(shallow_mid)}  predicted-sign {sum(x>0 for x in shallow_mid)}/{len(per_seed)}")
    print(f"  deepen  κ=1.5 (predicted <0): {stat(deepen_mid)}  predicted-sign {sum(x<0 for x in deepen_mid)}/{len(per_seed)}")
    print("\nsecondary — Δ(V_task):")
    print(f"  shallow κ=0.5 (predicted <0): {stat(shallow_v)}  predicted-sign {sum(x<0 for x in shallow_v)}/{len(per_seed)}")
    print(f"  deepen  κ=1.5 (predicted >0): {stat(deepen_v)}  predicted-sign {sum(x>0 for x in deepen_v)}/{len(per_seed)}")

    result = {
        "n_seeds": len(per_seed), "seeds": seed_ids,
        "primary_middle_correct_delta": {
            "shallow_k0.5": {**stat(shallow_mid), "n_predicted_sign": sum(x > 0 for x in shallow_mid)},
            "deepen_k1.5": {**stat(deepen_mid), "n_predicted_sign": sum(x < 0 for x in deepen_mid)},
        },
        "secondary_vtask_delta": {
            "shallow_k0.5": {**stat(shallow_v), "n_predicted_sign": sum(x < 0 for x in shallow_v)},
            "deepen_k1.5": {**stat(deepen_v), "n_predicted_sign": sum(x > 0 for x in deepen_v)},
        },
        "per_seed": per_seed,
    }
    out = os.path.join(here, "results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
