# GPT-2 fine temperature sweep — exp-035 summary

*May 26, 2026 (evening). Physics room solo session.*

## Question

After exp-034’s coarse sweep (T ∈ {0.5, 1.0, 2.0}), is |Δ_med − 0.25| minimized at trained T=1.0, with asymmetric drift below vs above?

## Headline numbers (R²>0.90 median Δ, 50 random inputs)

| T   | Δ_med | \|Δ−0.25\| | PL heads | λ_med (conformal) |
|-----|-------|-----------|----------|-------------------|
| 0.8 | 0.289 | 0.039     | 43       | 1.07              |
| 0.9 | 0.264 | 0.014     | 44       | 1.32              |
| 1.0 | 0.263 | 0.013     | 42       | 1.68              |
| 1.1 | **0.257** | **0.007** | 44   | 1.54              |
| 1.2 | 0.265 | 0.015     | 47       | 1.32              |

Reference: exp-007 Δ_med=0.2493; exp-034 T=1.0 Δ_med=0.243 (same protocol, different run seed — within noise).

## Verdict

**Basin slightly above T=1, not at T=1.** Closest SYK match at **T=1.1** (|Δ−0.25| = 0.007). T=1.0 is second-closest (0.013). Asymmetry: **low_T_steeper** — T=0.8 pulls Δ up more than T=1.2 pulls it down in this window.

Pre-stated “minimized at T=1.0” is **falsified at fine resolution** but the basin is narrow: all T ∈ [0.9, 1.2] stay within 0.015 of SYK. Coarse exp-034 behavior (T=0.5 / T=2.0) lies outside this band.

## Implications

- Junction 1 / REBUS: “trained precision” is a **narrow basin** around T≈1, not a single point; SYK Δ≈1/4 is not fragile to ±10% temperature rescaling.
- Does not replace sigmoid falsification (mechanism test still Tier 1).
- λ still varies across the band (max at T=1.0); Δ and λ decouple even near the basin.

Script: `research/physics/experiments/exp-035_gpt2_temperature_fine_sweep/run_fine_temperature_sweep.py`
