# exp-062 — The corpus-statistics test (Phase 1.1) + multi-seed degeneracy (Phase 1.3)

*June 11, 2026. THE decisive experiment of the program (SESSION_BRIEF_PHASE1 §1, §3.1.1).*

## Pre-registration

**Of record:** `notes/2026-06-11_corpus_statistics_preregistration.md`, committed 2026-06-11 **before any corpus generation, training, or measurement.** It fixes: the two hypotheses' quantitative predictions (imprint: Δ_med = β̂/2; universality: Δ_med ≈ 0.25 wherever conformal heads form), corpora (C-SR Markov / C-PL15 / C-PL25 / C-PL40 quantized-fGn at β targets 0.3/0.5/0.8 / C-NAT TinyStories), architecture (Pythia-70m-class GPT-NeoX, ctx 512), training (1.05B tokens/model, 7 runs incl. 3 C-NAT seeds for item 1.3), the frozen measurement protocol, the slope decision rule (b ≤ 0.33 universality / b ≥ 0.67 imprint / else ambiguous), and the 1.3 thresholds (across-seed Δ̂_med range ≤ 0.05 keep / > 0.10 kill).

## Pipeline

1. `gen_corpora.py alphabet` — fix the 256-id synthetic alphabet (top TinyStories token ids).
2. `gen_corpora.py pilot C-PL15|C-PL25|C-PL40` — 100M-token pilots; β̂ via the pre-registered MI estimator; calibration loop on H allowed **pre-training only**, every adjustment logged below.
3. `gen_corpora.py full <corpus>` — full 1.06B-token corpora (train + held-out), seeded, archived.
4. `train.py corpora/<c>_full.bin <run_name> --init-seed S --data-seed D` — checkpoints at the pre-registered steps.
5. `measure.py runs/<run>/step_2000 <tag> [--full-vocab for C-NAT]` — frozen census protocol, per-input profiles saved.
6. Decision statistics per prereg §6 (slope over power-law corpora, bootstrap over heads) — analysis script added at execution time, after measurement, per the pre-registered formulas only.

## Status log

| Date | Event |
|---|---|
| 2026-06-11 | Pre-registration committed; generators/training/measurement scripts written. **No corpus generated yet.** Cloud credentials unavailable on this machine at time of writing — package is launch-ready per prereg §8 (no degraded local substitute). |
| 2026-06-11 | **Estimator amendment (pre-data, ground-truth validated):** plain log-log OLS on plug-in MI is biased on long-memory sequences (distance-independent excess floor flattens the decay; H=0.8 truth 0.804 measured as 0.59). β̂ of record is now the free-floor fit A·d^(−β)+c; plain OLS kept as diagnostic. Validated against analytic fGn MI: {0.387, 0.547, 0.838} recovered for truth {0.332, 0.515, 0.804} at 16M tokens. Logged in prereg addendum. No decision rule changed. |
| 2026-06-11 | Generator smoke tests pass: fGn autocorrelation matches theory (var ≈ 1, ρ(d) within a few % at H=0.8); numba Markov walk 2M tokens / 0.3 s; quantile binning verified full 256-symbol range. |
| 2026-06-11 | **C-NAT generator amendment (pre-training):** TinyStories-train holds only ~0.45B Pythia tokens < the 1.06B matched budget; `gen_natural` now wraps in re-shuffled epochs (~2.3 epochs) rather than failing. Token budget stays matched across corpora; repetition rate disclosed. No decision rule affected. |

## Calibration log (H adjustments, pre-training only)

| Round | Date | H (PL15/PL25/PL40) | β̂ measured (targets 0.30/0.50/0.80) | Action |
|---|---|---|---|---|
| 1 | 2026-06-11 | 0.925 / 0.875 / 0.800 (design) | 0.435 / 0.590 / 0.851 | consistent overshoot; linear map β̂ ≈ 0.185 + 0.832·(4−4H) inverted |
| 2 | 2026-06-11 | 0.9655 / 0.9053 / 0.8152 | **0.335 / 0.496 / 0.795** | all within pre-registered ±0.05 → **calibration closed**; these H values frozen for the full corpora |

β̂ of record for the decision rule will be re-measured on the full 1.06B-token corpora (prereg addendum).

## β̂ of record (full 1.06B-token corpora, frozen estimator)

| Corpus | β̂ (record) | fit R² | Imprint prediction Δ_imp = β̂/2 | Universality prediction |
|---|---|---|---|---|
| C-SR | 3.0 (at bound; exponential MI as designed) | 0.82 | (not on slope axis; predicts no conformal formation) | — |
| C-PL15 | **0.3359** | 0.998 | **0.168** | ≈ 0.25 |
| C-PL25 | **0.4910** | 0.999 | **0.246** (degenerate point, by design) | ≈ 0.25 |
| C-PL40 | **0.7901** | 0.999 | **0.395** | ≈ 0.25 |
| C-NAT | **1.3798** | 0.957 | **0.690** | ≈ 0.25 |

All three power-law corpora within the pre-registered ±0.05 of target. Generated, archived (`corpora/*.bin` + `.mi.json`), seeds frozen. **Status: launch-ready** — see `LAUNCH.md`; only cloud credentials remain.

Observation (recorded before training, no decision-rule role): C-NAT's measured β̂ = 1.38 puts the imprint prediction for the natural-text control at Δ ≈ 0.69 — far above the ≈ 0.25 routinely measured in natural-text-trained models. If that pattern holds for the trained C-NAT runs, it is itself evidence against the imprint reading on the control corpus, independent of the engineered-corpus slope axis. (TinyStories' MI decays faster than web text; β̂ is corpus-specific, which is exactly why the prereg uses measured values.)

## Results

*(at execution)*
