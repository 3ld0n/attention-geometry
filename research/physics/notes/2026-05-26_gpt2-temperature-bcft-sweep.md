# GPT-2 temperature–BCFT sweep — exp-034 summary

*May 26, 2026. Physics room session.*

REBUS/Junction 1 predicted that lowering precision (increasing T) would drift Δ from 0.25 and weaken BCFT λ. **exp-034** tested this locally on GPT-2 without the 50 GB sigmoid download.

## Headline numbers

- **T=1.0:** Δ_med=**0.243** (exp-007: 0.249) — protocol validated.
- **T=0.5:** Δ_med=**0.468**, λ_med=**0.017** — high precision pushes Δ up and collapses boundary parameter.
- **T=2.0:** Δ_med=**0.328**; on the 23 heads conformal at T=1.0, Δ_med→**0.118**, λ_med→**4.26**.

## Verdict

Drift **confirmed** — SYK Δ≈1/4 is specific to trained T=1, not temperature-generic.

Simple “high T → weak λ” **falsified**. The coupling is richer: low T kills λ; high T on conformal heads drives Δ down while λ rises (possible boundary-dominated regime).

## Room implications

- Parallel local test **done** — sigmoid remains Tier 1 mechanism falsification.
- **exp-033 update:** axlearn now `infrastructure_ready_download_pending` after `tokamax` + `cloudpickle` install; only blocker is ~50 GB GCS download (+ Eldon OK).

Script: `research/physics/experiments/exp-034_gpt2_temperature_bcft/run_temperature_bcft_sweep.py`
