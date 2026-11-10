---
created: "2026-10-27"
time: "~4:30 PM MDT (2026-10-27T22:30Z)"
author: Ariel
surface: physics-room
status: >
  exploration — exploratory computation from new forward passes;
  no hypothesis registered; no pre-registration (blocked on GitHub token).
  This note is the record of that exploration.
subject: >
  Exploratory computation of excess kurtosis (κ₄/σ⁴ - 3) of pre-softmax
  attention scores for GPT-2 small. Tests whether Δ-window heads have
  systematically more non-Gaussian score distributions, consistent with
    10|  the 4-point structure hypothesis (Oct 25 note).
builds_on:
  - research/physics/notes/2026-10-25_route_a_failure_and_4pt_hypothesis.md
  - research/physics/experiments/exp-118_wikitext_census_cross_family/results.json
  - research/physics/experiments/exploration_2026-10-27_kurtosis/explore_kurtosis.py
registers: >
  [MEASURED exploratory] — kurtosis computation from 200 random sequences,
  seq_len=128, GPT-2 small. Δ-window classification from exp-118 (WikiText-native).
  [INTERPRETIVE] — comparison between populations and the 4-point hypothesis.
  No hypothesis was registered before this computation. Results cannot be
    20|  cited as confirmatory evidence. They inform the formal pre-registration design.
note: >
  Protocol mismatch: Δ-window classification from exp-118 uses WikiText-native census;
  kurtosis computed under random-token protocol. The formal experiment must resolve this.
  See §4 (methodological limits) below.
---

# Exploratory κ₄(s) computation — GPT-2 small

## What was done [MEASURED exploratory]

    30|Ran GPT-2 small forward passes on 200 random-token sequences (seq_len=128, seed=42).
Hooked pre-softmax attention scores s(i,j) = q_i · k_j / √d_head for all 144 heads.
Collected all causal pairs (j < i) per head: **1,625,600 score values per head**.
Computed per-head: mean, σ, skewness, excess kurtosis = E[(s-μ)⁴]/σ⁴ − 3.
Classified heads using Δ-window membership from exp-118 (WikiText-native census):
16 Δ-window heads, 128 non-window heads.

Script: `research/physics/experiments/exploration_2026-10-27_kurtosis/explore_kurtosis.py`

---
    40|
## Results [MEASURED exploratory]

### Excess kurtosis by population

| Population | n | Mean κ_ex | Std | Min | Max |
|---|---|---|---|---|---|
| Δ-window heads | 16 | **1.44** | 1.27 | −0.47 | 5.21 |
| Non-window heads | 128 | 0.37 | 1.78 | −1.01 | 12.07 |

**Mann-Whitney U test (two-sided):** U = 1597, **p = 0.0003**

The Δ-window population shows significantly elevated excess kurtosis compared to
non-window heads. The signal is robust to the distributional heterogeneity in both groups.

    50|### Skewness by population

| Population | Mean skewness | Std |
|---|---|---|
| Δ-window heads | 0.544 | 0.290 |
| Non-window heads | 0.285 | 0.407 |

Both populations are right-skewed, with the Δ-window heads more skewed on average.

### Per-head detail (Δ-window heads)

| Head | Layer | Excess Kurt | Skewness |
|---|---|---|---|
    60|| L4H10 | 4 | **5.21** | 1.32 |
| L8H2 | 8 | 2.35 | 0.75 |
| L9H4 | 9 | 2.31 | 0.67 |
| L9H6 | 9 | 2.17 | 0.65 |
| L11H6 | 11 | 2.21 | 0.80 |
| L11H4 | 11 | 1.74 | 0.66 |
| L11H2 | 11 | 1.58 | 0.69 |
| L7H1 | 7 | 1.32 | 0.43 |
| L10H10 | 10 | 1.32 | 0.58 |
| L10H1 | 10 | 0.81 | 0.29 |
| L11H1 | 11 | 0.61 | 0.37 |
| L10H2 | 10 | 0.76 | 0.51 |
| L11H7 | 11 | 0.57 | 0.39 |
| L11H5 | 11 | 0.30 | 0.36 |
| L11H9 | 11 | 0.30 | 0.22 |
| L11H0 | 11 | **−0.47** | 0.00 |

    70|**Observations:**
- L4H10 is the outlier (kurt = 5.21, skew = 1.32) — the shallowest-layer Δ-window head.
- L11H0 is sub-Gaussian (kurt = −0.47) — the only Δ-window head with negative kurtosis.
- Most Δ-window heads: kurt ≈ 0.3–2.4.

### Non-window head kurtosis quantiles

| Percentile | Excess kurtosis |
|---|---|
| 10th | −0.92 |
| 25th | −0.80 |
| 50th | −0.12 |
| 75th | +0.77 |
| 90th | +2.53 |

    80|Non-window heads are roughly centered near zero kurtosis (median −0.12), with heavy tails
(the max is 12.07 — some non-window heads have very non-Gaussian score distributions too).

---

## Interpretation [INTERPRETIVE]

**Consistent with the 4-point structure hypothesis:** The Δ-window subpopulation has
significantly elevated excess kurtosis (p = 0.0003), indicating heavier-tailed score
distributions than the non-window population. This is what the 4-point structure hypothesis
predicts: the slow-decay population sits at an SYK₄-like fixed point where the leading
    90|non-Gaussian correction is a 4-point term, producing heavy-tailed score statistics.

**Does not confirm SYK₄:** Excess kurtosis > 0 means heavier tails than a Gaussian. Many
mechanisms produce this. The SYK₄ hypothesis requires a specific relationship between the
4th cumulant and the 2nd cumulant (not just "elevated"), and a specific spectral structure.
This exploration confirms the signal is there; the formal experiment tests whether the
structure is SYK₄-class.

**What the non-Gaussian structure might reflect:**
- Positional encoding structure: GPT-2 uses learned absolute PE (wpe), which induces
  position-specific key vectors. Under random inputs, the score distribution reflects
   100|  the wpe structure for keys at different positions — potentially non-Gaussian.
- Trained weight structure: Δ-window heads have distributed key geometry (exp-127: lower
  λ₁/Σλ). A more distributed key geometry means a sum of more effective dimensions,
  which by CLT should be MORE Gaussian (lower kurtosis). The OPPOSITE is found. This
  suggests the non-Gaussianity is not explained by rank-1 concentration.
- Absolute-key-position bias (exp-138): structural heads (a subset of Δ-window heads)
  have score profiles dominated by absolute key position. This means s(i,j) is not
  stationary across positions — the score distribution is a mixture of position-specific
  distributions, which can produce elevated kurtosis as a mixing artifact.

   110|**The mixing artifact concern:** A head with score s(i,j) = α·log j + noise would have
a score distribution at a given j that's Gaussian, but across all j values (as we pooled),
it's a mixture. Mixture distributions have elevated kurtosis when the component means differ.
This is a potential confound: the elevated kurtosis might be a positional mixing artifact
rather than genuine non-Gaussian score structure.

**Testing the mixing artifact:** The formal experiment should compute kurtosis WITHIN
a fixed key position (or narrow position bin), not pooled across all positions. If the
kurtosis remains elevated within a fixed position range, it's not a positional mixing artifact.

---
   120|
## Methodological limits [MEASURED + INTERPRETIVE]

**Protocol mismatch (critical):**
The Δ-window head classification used exp-118's WikiText-native census (classification
protocol: WikiText tokens, R²≥0.90, Δ∈[0.20,0.30]). The kurtosis was computed on
random-token inputs. These are different protocols.

The two-populations result (exp-109) showed that random-token and WikiText-native window
populations are largely disjoint (Jaccard < 0.3 in GPT-2 small). Under random-token census,
exp-007 found 44/144 GPT-2 small heads in the Δ-window — different from exp-118's 16 WikiText-
native heads. This exploration's binary classification doesn't match the actual measurement
   130|protocol used (random-token inputs), and the formal experiment must decide:

**Option A:** Classify heads using random-token census (exp-007 protocol): 44 Δ-window heads.
This matches the kurtosis measurement protocol.

**Option B:** Compute kurtosis under WikiText-native protocol (matching exp-118 classification).
This keeps the class definition and measurement protocol aligned but requires different inputs.

This exploration used WikiText-native classification with random-token kurtosis — mixing the
two. The formal experiment must choose one.

**Small n issue:** n=16 Δ-window heads, n=128 non-window heads. The Mann-Whitney U is robust
   140|to distributional heterogeneity, but the effect size estimate (mean difference 1.44 - 0.37
= 1.07 excess kurtosis units) may not be stable at n=16.

**Mixing artifact:** See §3 above. Kurtosis within a fixed position bin should be tested.

**L4H10 outlier:** This one head (kurt=5.21) substantially elevates the window-head mean.
The formal experiment should report median kurtosis as well and test robustness to L4H10.

---

## What this adds to the structural session agenda [INTERPRETIVE]

The Oct 25 note placed the 4-point hypothesis with a kill condition: "compute κ₄(s)
from exp-118/exp-163 data" (analysis-only from existing data). This exploration clarifies:
   150|

1. **exp-118 and exp-163 data do NOT contain raw score arrays** — both save only per-head
   summary statistics (Δ, R², results). The κ₄ computation requires new forward passes.
   The analysis is NOT "analysis-only from existing data" as the Oct 25 note stated.

2. **New forward passes are needed** — but GPT-2 small is available locally (~1GB). Once
   the GitHub token is restored and the pre-registration is pushed, this can run locally
   without Modal. No cloud compute required.

3. **The signal is present and significant** (p=0.0003), but the protocol mismatch and
   mixing artifact concern must be addressed in the formal design before the result
   can be cited as evidence.

4. **Formal pre-registration for exp-174** should specify:
   - Classification protocol (random-token or WikiText-native) — must match measurement
   - Kurtosis within fixed position bins (mixing artifact control)
   - Median excess kurtosis (not just mean, given L4H10 outlier)
   - Minimum sequences: N=1000, seq_len=512 (longer than this exploration's 128)

   160|---

## Next moves

1. **Write formal pre-registration for exp-174** (4-point cumulant test) — commit locally
   to attention-geometry; push when GitHub token is restored; then run locally on GPT-2 small.
   The pre-reg must specify: (a) classification protocol matching measurement, (b) position-
   binned kurtosis (mixing artifact control), (c) median as primary statistic.

2. **GitHub token (Eldon-gated):** When restored, push exp-174 pre-reg commit, then run.

3. **Structural session agenda item:** The Oct 25 note said κ₄ is "analysis-only blocked
   on token." Correction: κ₄ requires NEW FORWARD PASSES but can run locally on GPT-2 small
   (no Modal needed). This is a simpler gate than previously stated — token only.

---

*Register ledger:*
*MEASURED (exploratory) — kurtosis computation from explore_kurtosis.py run 2026-10-27;
200 sequences, seq_len=128, seed=42, GPT-2 small. Pre-softmax scores, causal pairs.*
*INTERPRETIVE — comparison to 4-point hypothesis; mixing artifact concern; protocol mismatch.*
*Nothing here is a formal result. The formal pre-registration and run come next.*
