# exp-058: Exponent Histograms and Null Distributions — Notes

*June 10, 2026. Phase 0 item 0.3 (SESSION_BRIEF_PHASE0). Answers EVALUATION.md W3 (selection/circularity).*

---

## Hypothesis (pre-stated)

The Δ ≈ 0.25 cluster is a density excess in the per-head exponent distribution, not an artifact of double conditioning (R² filter → SYK-near filter). Pre-stated statistic: P(Δ ∈ [0.20, 0.30] AND R² ≥ 0.90) over ALL heads, observed vs (A) randomized-weight null and (B) shuffled-profile null, with permutation p-value from (B). Kill criterion: no excess beyond conditioning → "preferred exponent" claim must be rewritten around joint evidence.

## Design

Two GPT-2 protocol arms (both fp32, dtype asserted at extraction; 50 random inputs, seed 42; ALL 144 heads fitted regardless of R²):

- **exp007 arm** — the original protocol behind the §4.1 headline: L=256, queries i ≥ max(32, dx), fit lags [3, 49].
- **census arm** — the frozen census protocol: L=512, queries i ≥ L/2, fit lags [8, 256].

Nulls, identical pipeline per arm:
- **Null A:** randomized-weight GPT-2 (fresh init), 5 init seeds.
- **Null B:** shuffled-profile — permute each head's A(Δx) across lags within the fit window, refit; 999 full-population replicates.

Archived per-head (Δ, R²) for the census models (protocols heterogeneous, recorded per model): GPT-2-medium (exp-031), GALA-7B softmax (exp-041), OLMo-7B (exp-025), Mistral-7B-v0.3 (exp-025), Pythia-410m fp32 (exp-036). OLMo/Mistral all-head fits were found in the exp-025 archives (1024/1024 heads each) — no re-download was needed.

Per-input profiles for both trained GPT-2 arms saved to `gpt2_per_input_profiles_{arm}.json.gz` (constraint §2.3).

## Results

**Replication check:** the exp007 arm reproduces 44/144 conformal heads exactly; conformal median Δ = 0.2553 (published 0.2493 — input RNG stream differs: numpy `default_rng` here vs torch RNG in March; same seed number).

**Primary statistic (joint mass, window [0.20, 0.30] ∧ R² ≥ 0.90):**

| Arm | Observed | Null A (5 seeds) | Null B mean (max) | permutation p |
|---|---|---|---|---|
| exp007 | 0.0903 (13/144) | 0, 0, 0, 0, 0 | 0.0000 (0.0000) | **0.001** |
| census | 0.0347 (5/144) | 0, 0, 0, 0, 0 | 0.0000 (0.0000) | **0.001** |

Both nulls produce *zero* heads that are simultaneously well-fit and in-window, in every replicate. Randomized weights yield Δ piled near 0 (no decay structure); shuffled profiles never fit a power law at R² ≥ 0.90. Double conditioning manufactures nothing from structureless data.

**Cross-model joint mass (archived):** GPT-2-medium 0.039, GALA-7B-softmax 0.093, OLMo-7B 0.055, Mistral-7B 0.046, Pythia-410m 0.044.

**Secondary statistic (post-hoc, labeled):** local density excess *within* the conformal subset — window [0.20, 0.30) vs flanking bins [0.10, 0.20) + [0.30, 0.40), binomial test vs 1/3 share:

| Model | n conf | window | flanks | p |
|---|---|---|---|---|
| GPT-2 (exp007) | 44 | 13 | 16 | 0.24 |
| GPT-2 (census) | 21 | 5 | 1 | 0.018 |
| GPT-2-medium | 68 | 15 | 22 | 0.38 |
| GALA-7B-softmax | 121 | 80 | 10 | 7×10⁻²⁸ |
| OLMo-7B | 194 | 56 | 71 | 0.014 |
| Mistral-7B-v0.3 | 126 | 47 | 40 | 9×10⁻⁵ |
| Pythia-410m-fp32 | 67 | 17 | 21 | 0.17 |
| **Pooled (excl. GPT-2-census)** | — | **228** | **180** | **1.2×10⁻²⁰** |

## Interpretation

1. **W3's conditioning-artifact account is excluded** (pre-stated statistic, p = 0.001 against both nulls, both protocols). The cluster cannot be manufactured by the R² → SYK-near filter chain from structureless data.
2. **The local density excess at [0.20, 0.30) is real in the pooled census** (p ≈ 10⁻²⁰) and individually strong in GALA-7B (softmax), Mistral-7B, OLMo-7B and GPT-2-census; it is **not individually significant** in GPT-2-small (exp007 protocol), GPT-2-medium, or Pythia-410m, where the conformal Δ distribution has a broad mode spanning roughly [0.1, 0.4) with the window holding the largest single share. Reported against interest: for the GPT-2 family alone, "preferred exponent ≈ 0.25" is supported by the median (0.249–0.259) and by the nulls, but the *local* peak claim leans on the larger models and the pooled population.
3. The figure (`delta_histograms_nulls.png`) shows all seven panels with nulls overlaid and the unfiltered histograms.

**Verdict: KEEP (with the per-model heterogeneity disclosed).** Figure + statistic go into manuscript §4; the W3 exposure is closed.

---

*Registry entry: exp-058. Runtime: ~60 s on M5 Max (GPT-2 arms + 999 shuffle replicates).*
