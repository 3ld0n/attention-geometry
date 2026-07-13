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
| 2026-07-09 | **CLOUD LAUNCH.** Modal payment method added by Eldon — block lifted after 28 days. Volume `exp062-data` created; corpora (sha256 manifest `corpora_sha256.txt`) + train.py/measure.py uploaded. Launcher `modal_exp062.py` written — transport only: runs the pre-registered train.py/measure.py verbatim from the volume; image pins local library versions (numpy 2.4.6, scipy 1.17.1, torch 2.12.0, transformers 5.8.1) so the data-seed batch stream is identical to what a local run would produce. Chain: remote sha256 verify → 7 parallel A100-40GB training runs (resumable from volume checkpoints) → fp32 measurement → results pulled local. No protocol parameter changed. |
| 2026-07-09 | First train attempt OOM'd on A100-40GB (launcher had `--micro-batch 128`; the fused CE loss needs 12.3 GiB at that size). Fixed to the train.py default 64 (gradient accumulation only — effective batch unchanged at the pre-registered 1024) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Relaunched; runs that had saved step_0 resumed from it (byte-identical to fresh start: init save + deterministic data stream). No protocol parameter changed. |
| 2026-07-09 | **All 7 runs trained + measured clean** (~75 min/run). `analyze.py` written after measurement, prereg formulas only. **Verdicts: 1.1 AMBIGUOUS (slope axis uninformative — no power-law corpus formed); 1.3 KEEP (seed range 0.017).** See Results. |

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

*Measured July 9, 2026 (cloud; 7 runs, A100-40GB, all losses converged: C-SR 10.92→3.70, C-PL15 →4.63, C-PL25 →5.07, C-PL40 →5.34, C-NAT →1.43). Decision statistics per prereg §6/§7 only — `analyze.py`, `results.json`.*

### Formation (prereg §6.1: ≥ 10/48 conformal at step 2000)

| Corpus | conformal heads | forms? | Δ̂_med (conformal) |
|---|---|---|---|
| C-SR | 0/48 | no | — |
| C-PL15 | 2/48 | no | (1.39; n=2, not meaningful) |
| C-PL25 | 0/48 | no | — |
| C-PL40 | 5/48 | no | (0.45; n=5, not meaningful) |
| C-NAT s0/s1/s2 | **15 / 11 / 13** /48 | **yes ×3** | **0.174 / 0.167 / 0.158** |

### Verdict, Phase 1.1 (prereg §6.3): **AMBIGUOUS — slope axis uninformative**

**No engineered power-law corpus reached the formation criterion.** The slope
statistic b cannot be computed (0 of 3 power-law points; degenerate-formation
guard applies a fortiori). Reported exactly as measured, no reframing.

What the pattern does say, scored against the §2 row-level predictions:

- **C-SR row: imprint's one confirmed cell.** The Markovian control formed
  nothing, as the imprint hypothesis predicted (universality was silent on
  formation).
- **The imprint hypothesis's central mechanism fails on its home turf.**
  Corpora *engineered to have* power-law MI at β = 0.34/0.49/0.79 — the exact
  statistical property imprint says gets mirrored into attention — produced
  essentially no conformal population (0–5 heads vs. 11–15 for natural text at
  identical architecture, budget, and optimizer). Matching the two-point MI
  statistics of language is not sufficient to induce the conformal phase.
- **Neither hypothesis, as pre-registered, predicted this.** Imprint predicted
  formation tracking β; universality predicted ≈ 0.25 *wherever heads form* and
  was silent on formation — but the C-NAT Δ̂_med ≈ 0.166 is NOT within the
  ±0.05 of 0.25 that a KEEP would have required had the slope been computable.
  (Note: at this 70m/ctx-512/1B-token scale the SYK-near population may simply
  not have matured; the conformal population that does form sits at Δ ~ 0.10–0.17
  in layer 0 plus scattered steeper heads deeper — see per-head data.)
- **Escalation (per §6.3):** the resolving experiment is a corpus point with
  *hierarchical/compositional* structure but controlled statistics (imprint's
  quantitative form is dead as stated; the live question is what property of
  natural text — beyond pairwise MI decay — drives formation). A cheaper probe:
  higher-order Markov / PCFG corpora at matched β̂.

### Verdict, Phase 1.3 multi-seed (prereg §7): **KEEP (attractor)**

Across-seed range of Δ̂_med = **0.017** ≤ 0.05, while conformal-head identities
vary (Jaccard 0.37/0.56/0.41 across seed pairs — well below 1). The exponent
value is seed-stable; the specific heads carrying it are not. The attractor
reading survives its kill test.

### Files

- `results.json` — decision statistics (analyze.py, prereg formulas only)
- `measurements/run_*_final.json` (+ per-input gzips) — frozen-protocol output
- Volume `exp062-data`: all 16-step checkpoint trajectories for all 7 runs
  (emergence trajectory analysis available without retraining)
