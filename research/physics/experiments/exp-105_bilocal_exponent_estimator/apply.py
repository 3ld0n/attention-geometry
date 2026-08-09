"""exp-105 application — Delta_G for real heads, using the validated estimator.

Runs M1 on the lag profiles saved by exp-104. Heads outside M1's calibrated
operating range are reported as refusals, not as numbers.

M2 is not applied: it is self-rejecting on synthetic data (zero accepted cells,
kill condition K2) and its pre-registered verdict is FAIL.

Ariel — August 8, 2026.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from estimator import kit, m0_census, m1_three_param

HERE = Path(__file__).resolve().parent
EXP104 = HERE.parent / "exp-104_bilocal_object_identification"
OBJECTS = ("A", "G_out", "G_K", "G_cos")


def main() -> None:
    prof = np.load(EXP104 / "profiles_gpt2.npz")
    heads104 = json.loads((EXP104 / "results_gpt2.json").read_text())["heads"]

    rows = []
    for h in heads104:
        ell, hd = h["layer"], h["head"]
        rec = {"layer": ell, "head": hd,
               "delta_A_census": h["delta_A"], "r2_A_census": h["r2_A"]}
        for name in OBJECTS:
            fit = m1_three_param(prof[name][ell, hd])
            rec[f"{name}_m1_delta"] = fit.delta
            rec[f"{name}_m1_ok"] = fit.ok
            rec[f"{name}_m1_reason"] = fit.reason
            rec[f"{name}_m1_r2"] = fit.r2
            if fit.extra:
                rec[f"{name}_m1_ratio"] = fit.extra.get("ratio_est")
                rec[f"{name}_m1_noise"] = fit.extra.get("noise_est")
        rows.append(rec)

    (HERE / "applied_gpt2.json").write_text(json.dumps(rows, indent=1))

    conformal = [r for r in rows
                 if r["r2_A_census"] is not None and r["r2_A_census"] >= kit.R2_MIN
                 and r["delta_A_census"] is not None
                 and r["delta_A_census"] >= kit.DELTA_MIN]
    syk_near = [r for r in conformal if abs(r["delta_A_census"] - 0.25) <= 0.05]

    def report(label, subset):
        print(f"\n[{label}] n={len(subset)}")
        print(f"  {'object':8s} {'accepted':>9s} {'median Δ (M1)':>14s} "
              f"{'median ratio':>13s} {'median noise':>13s}")
        for name in OBJECTS:
            acc = [r for r in subset if r[f"{name}_m1_ok"]]
            d = [r[f"{name}_m1_delta"] for r in acc if r[f"{name}_m1_delta"] is not None]
            ratio = [r.get(f"{name}_m1_ratio") for r in subset
                     if r.get(f"{name}_m1_ratio") is not None]
            nz = [r.get(f"{name}_m1_noise") for r in subset
                  if r.get(f"{name}_m1_noise") is not None]
            dm = f"{np.median(d):.4f}" if d else "—"
            rm = f"{np.median(ratio):.2f}" if ratio else "—"
            nm = f"{np.median(nz):.2e}" if nz else "—"
            print(f"  {name:8s} {len(acc):>4d}/{len(subset):<4d} {dm:>14s} "
                  f"{rm:>13s} {nm:>13s}")

        # Paired comparison on heads where G_out is accepted.
        paired = [(r[f"G_out_m1_delta"], r["delta_A_census"]) for r in subset
                  if r["G_out_m1_ok"] and r["G_out_m1_delta"] is not None
                  and r["delta_A_census"] is not None]
        if paired:
            diff = np.array([g - a for g, a in paired])
            print(f"  paired on {len(paired)} accepted heads: "
                  f"median(Δ_G_out − Δ_A) = {np.median(diff):+.4f}   "
                  f"IQR [{np.percentile(diff, 25):+.4f}, {np.percentile(diff, 75):+.4f}]")
        else:
            print("  no heads with G_out accepted -> Delta_G not reportable here")

        # Refusal reasons, so the failures are visible rather than silent.
        reasons: dict[str, int] = {}
        for r in subset:
            if not r["G_out_m1_ok"]:
                key = r["G_out_m1_reason"].split("(")[0].strip() or "unspecified"
                reasons[key] = reasons.get(key, 0) + 1
        if reasons:
            print("  G_out refusals:")
            for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
                print(f"    {v:>3d}  {k}")

    print("exp-105 application — M1 on exp-104 profiles (GPT-2)")
    print("A is included as a control: with little floor, M1 must reproduce the census.")
    report("ALL", rows)
    report("CONFORMAL", conformal)
    report("SYK-NEAR", syk_near)

    # V4 in practice: M1 vs census on A, where the floor is small.
    both = [(r["A_m1_delta"], r["delta_A_census"]) for r in conformal
            if r["A_m1_ok"] and r["A_m1_delta"] is not None
            and r["delta_A_census"] is not None]
    if both:
        d = np.array([m - c for m, c in both])
        print(f"\ncontrol: M1 vs census on A over {len(both)} accepted conformal heads: "
              f"median diff {np.median(d):+.4f}, max |diff| {np.max(np.abs(d)):.4f}")


if __name__ == "__main__":
    main()
