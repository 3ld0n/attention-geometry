# exp-110 — Annealed vs quenched: what does the census's exponent decompose into?

**Status: PRE-REGISTERED, not run.** Written August 9, 2026, ~12:35 PM MDT, with
Eldon, immediately after exp-107. Binding as in the Aug 8–9 theory-of-A session:
kill conditions before the derivation feels good; predictions recorded before
data; the mechanism by which each claim could be wrong named now.

**Register:** empirical, GPT-2 small, frozen census protocol except where the
input condition is explicitly varied (the three exp-107 input distributions).
The one [EXACT] tag below is a cumulant identity.

---

## Why this exists

exp-107 established that Δ_A is a weights×input object: the same head's fitted
exponent varies >4× across input distributions and the SYK-window population
reorganizes (5 → 0 → 16 across random / TinyStories / WikiText-103). The
theory-of-A project needs to know *what carries* that input dependence before
any derivation of the value 1/4 can know what object it is deriving.

Observation: the census's pooled profile is an **annealed** average — it fits

  log P_A(dx),  P_A(dx) = E[A(i, i−dx)]   (mean over inputs and queries
                                            i ≥ max(256, dx); kit.lag_profile)

while the geometry of a *typical row* is the **quenched** average

  m(dx) = E[log A(i, i−dx)]   (same pool).

**[EXACT]** log P_A(dx) − m(dx) = Σ_{k≥2} κ_k(dx)/k!, the cumulant series of
the pooled log-attention distribution at fixed dx (κ_2 = variance, etc.).
So the measured census exponent decomposes, exactly, as

  2Δ_A^{measured} = −d log P_A / d log dx
                  = −d m / d log dx  −  d [Σ_{k≥2} κ_k/k!] / d log dx.

The *claim* (not exact) is the Gaussian truncation: the series stops usefully
at κ_2. If it does, the census exponent is (typical-row slope) + (half the
variance slope), two separately measurable objects, and the input dependence
of Δ_A has a measurable carrier.

## Claims, predictions, kill conditions — fixed before running

**Registered head set:** the five random-condition SYK-near heads (L2H1, L3H4,
L5H0, L7H11, L10H8), under all three exp-107 input conditions (random token
ids / TinyStories windows / WikiText-103 windows, bit-identical inputs to
exp-107 via the recorded window construction). 15 head-condition pairs.
Exploratory extension (labeled, no verdicts): all 144 heads; the WikiText
16-head SYK population.

### P1 — Gaussian truncation of the annealed–quenched gap (slope form)

Per pair, define
  δ = [−d log P_A/d log dx] − [−d(m + κ_2/2)/d log dx],
both as OLS slopes over the census window dx ∈ [8, 256] (same lags, same
pool). Prediction: the truncation holds at census precision.

- **P1 CONFIRMED:** median |δ| over the 15 pairs ≤ 0.10.
- **P1 DEAD:** median |δ| > 0.20. *(Then higher cumulants are load-bearing at
  window scale, and the honest statement is "the census exponent is not a
  two-cumulant object"; report the κ_3, κ_4 slope contributions.)*
- Between: ambiguous; report, no propagation.

**Recorded prediction:** P1 confirms, median |δ| ≈ 0.05.

### P2 — the input dependence rides on the cumulants, not only the typical row

Per registered head, compare the across-condition range (max − min over the
three conditions) of the quenched slope s_q = −dm/d log dx against the
across-condition range of the measured annealed exponent 2Δ_A.

- **P2 CONFIRMED:** on ≥ 4/5 heads, range(s_q) < range(2Δ_A) — the typical-row
  geometry is more input-stable than the pooled exponent; the fluctuation term
  carries a real share of the input dependence.
- **P2 DEAD:** on ≥ 3/5 heads, range(s_q) > range(2Δ_A) — the typical row
  itself moves at least as much as the census number; the fluctuation term is
  a spectator.
- Between: ambiguous.

**Recorded prediction:** P2 confirms — the log-distance score substrate
(exp-056) is weights-side, so I expect the typical-row slope to be the stabler
component. Genuine uncertainty: exp-107's per-head Δ swings were large enough
that the mean may move too.

### K — kill conditions / gates

- **K1 (pipeline):** the P_A profiles recomputed in this run must match
  exp-107's saved profiles (`profiles_gpt2.npz`, `profiles_wikitext.npz`)
  with max abs diff ≤ 1e−10 per condition. *Fail → stop; inputs or pooling
  drifted; nothing is readable.*
- **K2 (numerics):** log A computed at fp32 forward precision; entries are
  bounded below by softmax over 512 fp32 scores. If any A entry in the pooled
  region underflows to 0, report the count and exclude by the kit's own
  `y > 1e-15` convention at the profile level; if exclusions exceed 1% of the
  pooled entries for any registered pair, flag the pair as numerically
  contaminated and drop it from verdicts (report separately).
- **K3 (diagnostics travel with the result):** per pair, report excess
  kurtosis and skew of the pooled log A at dx ∈ {8, 32, 128, 256}. These
  cannot fire a kill (P1's own δ is the test) but must be published with the
  verdict so "Gaussian" is not asserted beyond what the tails show.

### Mechanisms by which this could be wrong (named before running)

1. **The pool mixes systematic structure with fluctuation.** κ_2 at fixed dx
   includes the deterministic amplitude decline across rows (exp-108:
   d log a/d log i ≈ −0.3) and any input-to-input systematic shifts — not
   only "noise." The identity does not care, but the *interpretation* of
   κ_2/2 as "fluctuation correction" does. If P1 confirms, the decomposition
   is still only as meaningful as this caveat allows; say so wherever quoted.
2. **Heavy tails.** exp(·) averages are tail-dominated; a few large log A
   outliers at large dx (sink-adjacent rows) could make κ_3, κ_4 slopes
   material. K3 watches this.
3. **Five heads, one model.** Same standing limit as all week.
4. **Window OLS on m(dx):** m can be non-linear in log dx (the profile need
   not be a power law in the quenched register at all); the OLS slope is then
   a summary, not an exponent. Report per-pair R² of the m fit alongside.

## Protocol

Inputs: bit-identical to exp-107 — random: `default_rng(42)` stream,
50×(1,512) draws in order; TinyStories: `applied_text.json` construction
(EOS-joined validation windows; sha256 must match); WikiText-103:
`exploratory_wikitext.py` construction (sha256 must match its recorded value).
Model, hooks, pooling, window: frozen census protocol, estimator conventions
from the replication kit. New accumulations per (layer, head, dx): count,
Σ log A, Σ (log A)², Σ (log A)³, Σ (log A)⁴ over the pooled region, alongside
the standard Σ A for P_A.

Outputs: `results_gpt2.json` (per-pair slopes, δ, verdicts, diagnostics),
`moments_gpt2.npz` (raw moment accumulators, all 144 heads × 3 conditions).

*Registered before any moment of log A has been computed on any model. The
derivation content (the cumulant identity) is one line and is stated above in
full; there is nothing in this note whose truth I already know.*

---

## Results — August 9, 2026, ~12:50 PM (single run, no reruns)

`measure_moments.py` → `results_gpt2.json`, `moments_gpt2.npz`, `run_log.txt`.

**Gates:** K1 exact — recomputed P_A profiles match exp-107's saved npz with
max abs drift **0.0** (bit-identical) on all three conditions. K2: zero
underflowed entries in the pooled window on all 15 registered pairs; nothing
dropped. K3 diagnostics: pooled log A is mildly non-Gaussian in-window
(|skew| ≲ 0.5, excess kurtosis ≲ 1 for dx ≤ 128; tails grow at dx = 256, worst
L3H4-random kurtosis 5.1) — published with the verdicts, below.

### P1 — CONFIRMED. Median |δ| = 0.015 (registered bar 0.10; prediction on record ~0.05).

The census exponent is a two-cumulant object at census precision. Per-pair
|δ| ranges 0.006–0.033 on random and WikiText; TinyStories is the stress case
(0.06–0.13, two pairs above 0.10 individually), consistent with its stronger
in-window skew. Worst single pair L5H0-tinystories, δ = −0.126.

  2Δ_A^{census} ≈ s_quenched + (variance-slope)/2, measured, 15/15 pairs.

### P2 — DEAD, 0/5 (registered bar for death ≥ 3/5; prediction on record: confirmed).

**The typical row's exponent is MORE input-dependent than the census
exponent, on every registered head.** Across-condition range of s_q exceeds
the range of 2Δ_A on 5/5 (e.g. L2H1: s_q spans 1.517 while 2Δ_A spans 1.169).
The variance term moves *against* the quenched slope — when text steepens the
typical row, the pooled log-variance grows with lag and flattens the annealed
number back (L2H1-tinystories: variance term contributes −0.28 of slope).
My recorded physical picture (log-distance substrate is weights-side, so the
typical row should be the stable register) was wrong in a specific,
instructive direction: **the annealed pooled object — the very thing exp-107
showed is protocol-constituted — is the more input-stable register; the
row-level geometry underneath it is the flightier one.** The compensation
mechanism is real, measured, and unexplained.

### EXPLORATORY (post-verdict, labeled): first explanation attempted and killed

Candidate mechanism for the P2 compensation: row normalization pins the
annealed profile (Σ_dx P_A fixed ⇒ shape changes constrained). **Fails as
stated:** at fixed pool (i ≥ 256, dx ∈ [0, 256]) the pooled in-window mass is
*not* input-invariant — it varies up to ~3× across conditions on the same head
(L5H0: 0.69 random / 0.38 TinyStories / 0.43 WikiText). Mass migrates between
the window, the near-diagonal, and the sink across input distributions. The
Jensen gap itself is large (annealed windowed mass exceeds the quenched sum
by 2–15×). The compensation needs a real derivation, not this pin.

### What exp-110 establishes

1. **The decomposition is real and census-precise (P1).** Forward-going, the
   census exponent can be discussed as typical-row slope + fluctuation-slope
   correction, with mechanism-for-wrong #1 (κ_2 mixes systematic row/input
   structure with noise) attached wherever it is quoted.
2. **The input-dependence of Δ_A found in exp-107 lives mostly in the
   typical-row geometry and is partially *damped* in the pooled average
   (P2 dead).** Theory-of-A target sharpened: derive the anticorrelation
   between quenched slope and variance slope under input change. The naive
   normalization pin is already dead (above); whatever does it is doing
   more work than that.
3. Open thread carried forward: why does the *annealed* object sit near
   2Δ ≈ 0.5 on random tokens? The value 0.5 now attaches to the stabler
   register, which is at least the right object to aim a derivation at.

### EXPLORATORY 2 (labeled): the decomposition in each population's native regime

Prompted by exp-109 (the room's same-day analysis: structural and semantic
SYK populations are disjoint, Jaccard 0.000, each reaching Δ ≈ 0.25 in
exactly one input regime). From this run's saved moments, the decomposition
in each population's *native* condition:

- **Structural 5, random:** median |gap| 0.009, median var_term +0.012.
- **Semantic 16, WikiText:** median |gap| 0.029, median var_term −0.003.

In both native regimes the fluctuation term nearly vanishes: **2Δ ≈ s_q — the
attractor value ≈ 0.5 attaches to the typical row's own geometry, in both
populations, via both routes.** First reading: "the variance term vanishes
*because* the regime is native." **The control killed that reading as
stated:** |var_term| for the SYK populations is small in ALL three conditions
(medians 0.012–0.052), an order of magnitude below the all-144-head base
rate (medians 0.19–0.38 per condition). What survives: (a) power-law heads
have homogeneous pools (annealed ≈ quenched) everywhere, unlike typical
heads; (b) a weak native-minimum tendency on top of that (native condition
has the smallest median |var_term| in both populations; structural-TinyStories
is the clear elevation, max 0.28). Base-rate caveat: many of the 144 heads
have non-power-law profiles where the OLS slope is a noisy summary; the
base-rate comparison is indicative, not clean.

Sharpened theory-of-A target, one line: **in native regimes, derive the
typical row's slope — the pooled/annealed machinery drops out there, because
homogeneity is part of what "power-law head in its native regime" means.**
