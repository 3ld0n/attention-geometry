"""
exp-062 — Decision statistics (prereg §6, §7). Written at execution time,
AFTER measurement, implementing only the pre-registered formulas
(notes/2026-06-11_corpus_statistics_preregistration.md in the physics archive).

§6.1 formation criterion: >= 10/48 conformal heads at the final checkpoint.
§6.2 primary statistic: OLS slope b of delta_med(c) on beta_hat(c)/2 over the
     power-law corpora that FORM, 95% CI from 1,000 bootstrap resamples over
     conformal heads within each corpus.
§7   multi-seed (C-NAT x 3): across-seed range of delta_med; Jaccard overlap
     of conformal-head identity sets across seed pairs.

Usage: python analyze.py  (reads measurements/*.json, writes results.json)
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MEAS = HERE / "measurements"

# beta_hat of record (full corpora, frozen free-floor estimator; notes.md table)
BETA_HAT = {"C-PL15": 0.3359, "C-PL25": 0.4910, "C-PL40": 0.7901, "C-NAT": 1.3798}
POWER_LAW = ["C-PL15", "C-PL25", "C-PL40"]
RUNS = {
    "C-SR": ["run_CSR"],
    "C-PL15": ["run_CPL15"],
    "C-PL25": ["run_CPL25"],
    "C-PL40": ["run_CPL40"],
    "C-NAT": ["run_CNAT_s0", "run_CNAT_s1", "run_CNAT_s2"],
}
FORMATION_MIN = 10
BOOT_N = 1000
BOOT_SEED = 62


def load(tag: str) -> dict:
    return json.loads((MEAS / f"{tag}_final.json").read_text())


def conformal_deltas(d: dict) -> np.ndarray:
    return np.array([h["delta"] for h in d["heads"] if h["conformal"]])


def conformal_ids(d: dict) -> set:
    return {(h["layer"], h["head"]) for h in d["heads"] if h["conformal"]}


def main() -> None:
    results: dict = {"per_run": {}, "formation": {}, "slope": {}, "multi_seed": {}}

    for corpus, tags in RUNS.items():
        for tag in tags:
            d = load(tag)
            deltas = conformal_deltas(d)
            results["per_run"][tag] = {
                "corpus": corpus,
                "n_conformal": d["n_conformal"],
                "n_syk_near": d["n_syk_near"],
                "delta_median_conformal": d["delta_median_conformal"],
                "forms": d["n_conformal"] >= FORMATION_MIN,
            }
            assert d["n_conformal"] == len(deltas)

    # §6.1 formation per corpus (primary seed for C-NAT is s0 per prereg §3)
    for corpus, tags in RUNS.items():
        n = results["per_run"][tags[0]]["n_conformal"]
        results["formation"][corpus] = {
            "n_conformal": n, "forms": n >= FORMATION_MIN,
        }

    # §6.2 slope over FORMED power-law corpora
    formed_pl = [c for c in POWER_LAW if results["formation"][c]["forms"]]
    results["slope"]["formed_power_law_corpora"] = formed_pl
    if len(formed_pl) >= 2:
        rng = np.random.default_rng(BOOT_SEED)
        x = np.array([BETA_HAT[c] / 2 for c in formed_pl])
        head_deltas = {c: conformal_deltas(load(RUNS[c][0])) for c in formed_pl}
        y = np.array([float(np.median(head_deltas[c])) for c in formed_pl])
        b = float(np.polyfit(x, y, 1)[0])
        boots = []
        for _ in range(BOOT_N):
            yb = [float(np.median(rng.choice(hd, size=len(hd), replace=True)))
                  for hd in (head_deltas[c] for c in formed_pl)]
            boots.append(float(np.polyfit(x, np.array(yb), 1)[0]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        results["slope"].update(slope_b=b, ci95=[float(lo), float(hi)])
    else:
        results["slope"]["slope_b"] = None
        results["slope"]["note"] = (
            f"only {len(formed_pl)} power-law corpora formed (< 2): "
            "slope axis uninformative per prereg §6.3 degenerate-formation guard"
        )

    # §6.3 verdict
    if len(formed_pl) < 2:
        verdict = "AMBIGUOUS"
        pattern = ("no engineered power-law corpus reached the formation "
                   "criterion; only C-NAT (natural text) forms conformal heads; "
                   "slope statistic not computable")
    else:
        b, (lo, hi) = results["slope"]["slope_b"], results["slope"]["ci95"]
        keep = all(
            abs(results["per_run"][RUNS[c][0]]["delta_median_conformal"] - 0.25) <= 0.05
            for c in formed_pl) and b <= 0.33 and hi < 0.67
        kill_align = sum(
            abs(results["per_run"][RUNS[c][0]]["delta_median_conformal"] - BETA_HAT[c] / 2) <= 0.05
            for c in formed_pl) >= 2
        kill = b >= 0.67 and lo > 0.33 and kill_align
        verdict = "KEEP" if keep else ("KILL" if kill else "AMBIGUOUS")
        pattern = f"slope b={b:.3f} CI[{lo:.3f},{hi:.3f}]"
    results["verdict_1_1"] = {"verdict": verdict, "pattern": pattern}

    # §7 multi-seed degeneracy (C-NAT x 3)
    nat = [load(t) for t in RUNS["C-NAT"]]
    med = [d["delta_median_conformal"] for d in nat]
    rng_med = float(max(med) - min(med))
    ids = [conformal_ids(d) for d in nat]
    jacc = {f"{a}-{b}": round(len(ids[a] & ids[b]) / len(ids[a] | ids[b]), 3)
            for a, b in combinations(range(3), 2)}
    verdict_13 = ("KEEP" if rng_med <= 0.05 else
                  "KILL" if rng_med > 0.10 else "AMBIGUOUS")
    results["multi_seed"] = {
        "delta_medians": med, "range": rng_med, "jaccard": jacc,
        "verdict": verdict_13,
    }

    out = HERE / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
