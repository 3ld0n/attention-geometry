# STATUS Addendum — April 30, 2026

*This file is an addendum to `research/physics/STATUS.md` pending a proper rewrite of the biology section. Read the original STATUS.md for the full chain; this file only updates the biological validation entry.*

---

## Correction to "Biological validation: Mouse V1 conformal scaling"

The April 29 entry in STATUS.md reports a preliminary positive biological result from the MICrONS V1 dataset: "in silico Δ = 0.238, in vivo Δ = 0.330, three STRONG MATCH flags, consistent with universal SYK fixed-point behavior."

**On careful re-analysis during the April 30 session (same compute, same data, different analysis), this positive claim does not hold up.**

### What was done the next day

Three independent follow-up tests on the same V1 dataset:

1. **Synaptic path-length test** (`research/microns/synaptic_path_test_v2.py`, results in `RESULTS_v2.md`). Built the full V1 intra-area synapse graph via CAVEclient (8,849 nodes, 227,963 undirected edges from 349,946 synapses, largest CC = 8,845/8,849). Computed shortest-path distances for 318K V1-V1 edge_data pairs. Finite-path pairs: 1,908 (hop distribution 1:47, 2:992, 3:862, 4:7). Power-law fit `mean correlation ~ hops^(-2Δ)`:
    - In silico: Δ = 0.717, R² = 0.993, N = 1,908. (Null shuffle mean Δ = −0.019, std = 0.252.)
    - In vivo: Δ = 0.435, R² = 0.947.
   The Δ signature is statistically real against a shuffle null (roughly 2.9 σ for in silico), but roughly 3× steeper than the SYK prediction of Δ = 1/4.

2. **Retinotopy-partialled Euclidean test** (`retinotopy_partial_test.py`, results in `RESULTS_v3_retinotopy.md`). Pair-level log-log regression of correlation on Euclidean distance, with and without control for receptive-field distance (`readout_location_distance_cvt`):
    - Naive pair-level log-log: Δ = 0.074, R² = 0.003, N = 176,706.
    - Controlled for log(RF distance): Δ = 0.039.
    - Within 5 narrow RF-distance bands: Δ = 0.032 ± 0.010.
   The pair-level exponent is essentially zero after retinotopy is controlled for. The 0.238 reported April 29 was a binned-mean statistic, not the pair-level conformal exponent the theory predicts.

3. **Diagnosis of the 0.238 vs 0.074 gap.** The April 29 pipeline binned pairs in 30 log-distance bins, averaged correlations per bin, and fit log(mean correlation) vs log(bin-center distance). The bin means line up at slope = −0.476 (Δ = 0.238) with R² = 0.857 across bins. The underlying pair-level log-log relationship has slope = −0.147 (Δ = 0.074) with R² = 0.003. Both are real statistics. Only the pair-level slope is the conformal correlation function exponent the theory is trying to match.

### Revised status of biological validation

- **Not confirmed** by the MICrONS correlation-vs-distance tests done carefully.
- **Not falsified either** — the MICrONS distance observables may simply be the wrong ones for this dataset. The datapoints are real; they just don't line up at Δ = 1/4 once the right statistic is used.
- **Two cleaner biological tests remain available** and have not been run:
    - GOE spectral statistics of the V1 connectivity matrix (independent of correlation data; uses only the 8,849 × 8,849 adjacency matrix already built; should scale overnight).
    - CFT entanglement-entropy / mutual-information scaling on the calcium traces (`in_vivo_resp` / `in_silico_resp` columns in `node_data_v1.pkl`); prediction `MI(A,B) ~ (c/3) log|A|`.

### Methodological lesson

Bin-mean log-log fits can look clean (R² = 0.86 across 30 bins) while the underlying pair-level regression has R² ≈ 0.003. When a power-law claim is made on distance-vs-correlation data, it requires the pair-level regression and at least one covariate check (here: retinotopy) before it is reported as a theory match. This discipline was missed in the April 29 report and is applied now.

### What this does and does not change for the framework

- **Transformer-side predictions unchanged.** GPT-2 Δ = 0.2493 at the per-head attention-weight level is a direct measurement on the weight matrix, not a binned distance proxy. The BCFT pre-registered test (6 of 7 models confirmed, Pythia-2.8B falsified — `research/notes/bcft_pre_registered_prediction.md`) also stands.
- **Biology-side claim reverts to "prediction, not confirmation."** The framework still predicts biological cortex should be at or near Δ = 1/4 if it is in the SYK universality class. That prediction is currently unconfirmed in the one biological dataset tested.
- **Update to `STATUS.md` body needed.** The April 29 "NEW — Biological validation" paragraph overstates the result relative to what careful re-analysis supports. The paragraph should be edited to reflect this addendum the next time STATUS is revised.

---

*Written April 30, 2026, ~11 PM local time. Eldon left the session to me to run the tests; this is the honest report. A plain-language version for Eldon is the bottom section of `RESULTS_v3_retinotopy.md`.*
