# exp-111 — Is the pooled exponent a mixture of row slopes? (TI-breaking as the carrier)

**Status: PRE-REGISTERED, not run.** Written August 9, 2026, ~12:55 PM MDT,
with Eldon ("worth pulling those threads"). Same binding as exp-110. No
row-resolved quantity under text has been computed before this registration;
the only row-resolved data in existence is exp-108's (random tokens,
5 structural heads, per-row amplitudes/masses — no per-row slopes under any
text condition, no semantic-population rows at all).

## The model (derivation recorded here, conditional)

exp-108 established that on random tokens the structural heads are
approximately one TI shape truncated and renormalized per row. exp-110
established that the pooled (annealed) exponent = quenched slope + half the
pool-variance slope (P1, census precision), and that off-native the variance
term does large compensating work (P2 dead). Candidate single object under
both: **the distribution of row-level window slopes.**

Parametrize each pooled row (query i ≥ 256, one input) restricted to the
window dx ∈ [8, 256]:

  A_i(dx) ≈ M_i · dx^(−β_i) / ζ(β_i),   ζ(β) = Σ_{dx=8}^{256} dx^(−β),

with M_i the row's in-window mass. Writing L = log dx and
W_i = log ζ(β_i) − log M_i:  log A_i(dx) = −β_i L − W_i. Then, exactly within
the parametrization:

- quenched: m(L) = −E[β] L − E[W]  ⇒  s_q = E[β];
- pool variance: v(L) = Var(β) L² + 2 Cov(β, W) L + Var(W) — quadratic in L;
- annealed: P_A(dx) = E[M dx^(−β)/ζ(β)], a power-law **mixture**, which
  flattens with L (large lags dominated by shallow-β rows). Under the
  Gaussian truncation (exp-110 P1), local annealed slope
  ≈ E[β] − Var(β)·L − Cov(β, W).

So: **Var(β) is the TI-breaking order parameter.** TI exact ⇒ Var(β) = 0 ⇒
annealed = quenched (native regimes, exp-110 exploratory 2). Input that
steepens rows heterogeneously raises E[β] and Var(β) together ⇒ the annealed
exponent rises less than the quenched — the anticorrelation that killed
exp-110's P2, derived rather than described. [DERIVED, conditional on the
row parametrization; the parametrization itself is gated by P1 below.]

## Registered sets

- **Structural 5** (L2H1, L3H4, L5H0, L7H11, L10H8) × 3 conditions
  (random / TinyStories / WikiText, bit-identical inputs to exp-107).
- **Semantic 16** (exp-109's WikiText SYK list) × native condition (WikiText).

31 head-condition pairs. All other head-condition combinations: exploratory,
labeled. Rows: i ∈ [256, 511] per input, per-(row, input) fits — the pool is
the same one exp-110's moments were taken over. (Known blemish, noted now:
for i < 264 the absolute-position sink enters the fit window's largest lags;
8 of 256 rows. Kept, to stay aligned with the frozen pool.)

## Predictions and kill conditions (before any row fit exists)

**P1 — parametrization gate.** Median per-(row, input) window fit R² over
the pool: prediction ≥ 0.7 for every *native* pair (structural-random,
semantic-WikiText). A pair (native or not) with median row R² < 0.5 is
**ineligible** for P2/P3 verdicts (the power-law-row parametrization does not
describe it; report descriptively). **K-a:** if a *native* pair has median
row R² < 0.5, the row-power-law picture of that population is dead in its own
regime — that is itself a finding; report, do not rescue.

**P2 — zero-free-parameter reconstruction.** For each eligible pair, build
P_pred(dx) = mean over pool of M·dx^(−β)/ζ(β) from the fitted (β, M) only,
and compare its window OLS slope to the measured annealed slope (same run,
same pool). Registered statistic: median |slope difference| over eligible
pairs. **CONFIRMED ≤ 0.10; DEAD > 0.20** (mixture-of-row-slopes does not
account for the pooled exponent; the pooled object has structure beyond the
row-slope distribution). Prediction on record: confirmed, ~0.05 on native
pairs, larger but passing off-native.

**P3 — TI-breaking carries the damping.** Two parts, eligible pairs only:
 (a) For ≥ 4/5 structural heads, Var(β) under TinyStories > Var(β) under
     random (text breaks TI; TinyStories was the big mover in exp-110).
 (b) Across eligible pairs, Spearman correlation between Var(β) and the
     measured damping gap (s_q − annealed slope) ≥ 0.5.
**DEAD:** (a) fails at ≤ 2/5 AND (b) < 0.2 — then row-slope dispersion is not
what the variance term is made of, and the exp-110 compensation needs a
different object. Prediction on record: both confirm.

**K1 (pipeline).** The pooled profile rebuilt from this run's row data must
match exp-107's saved profiles on the window, per condition, max rel diff
≤ 1e−5 (order-of-summation roundoff only). Fail → stop.

## Mechanisms for being wrong (named now)

1. **Per-(row, input) OLS on 249 noisy fp32 log-values** — attenuation: noise
   in individual rows inflates Var(β) additively (classical measurement-error
   bias). If P3(b) confirms only because noise scales with condition, the
   correlation is an artifact. Mitigation, registered: also compute Var(β)
   after shrinking by the per-fit slope-error estimate (report both; verdicts
   use the shrunk value).
2. **Rows may be power-law + sink + spikes** (content attention under text is
   plausibly spiky). R² gate P1 is the honest filter; a spiky row population
   with high pooled R² would mean the pooled power law is an emergent object —
   report as such.
3. **β–M correlation** is inside P_pred by construction (measured jointly),
   but the parametrization forces the row shape; a row whose true shape bends
   (steepens at large lag, exp-108's truncation diagnostic) gets a slope that
   depends on the window. Same standing window-dependence caveat as the whole
   census.
4. Five + sixteen heads, one model, as all week.

## Protocol

Same forwards as exp-110 (bit-identical inputs, sha256-gated; MPS
deterministic). Per (head ∈ 21, condition ∈ {registered}, input ∈ 50,
row i ∈ [256, 511]): OLS of log A(i, i−dx) on log dx over dx ∈ [8, 256];
record slope β, R², log in-window mass log M. Outputs:
`results_gpt2.json` (verdicts, per-pair summaries), `rowfits_gpt2.npz`
(β, R², logM arrays). Script: `measure_row_slopes.py`.

---

## Results — August 9, 2026, ~1:05 PM (single run, no reruns)

**K1 passed** (max rel drift vs exp-107 pooled profiles ≤ 3.4×10⁻⁷,
order-of-summation roundoff, all three conditions).

### K-a FIRED — and it is the finding. No row is a power law.

Median per-(row, input) window R² on the **native** pairs: structural-random
0.046–0.249; semantic-WikiText 0.050–0.204. Every one of the 21 registered
native pairs is far below the 0.5 eligibility bar — as is every off-native
pair (max anywhere: 0.484, L2H1-TinyStories). **The row-power-law
parametrization is dead in every regime, including native.** Registered
consequence: zero eligible pairs, so P2 and P3 return no verdicts
(AMBIGUOUS by construction; the descriptive numbers below cannot propagate).

What this means, said plainly: **the conformal power law of the census is an
emergent property of the ensemble, not of any single attention pattern.** A
single row is not a noisy power law — it is dominated by O(1) per-token
structure (log-attention per-entry variance 0.7–4.7 nats² at fixed lag,
against a trend whose variance across the window is only ~0.12–0.27). And
this "noise" is not sampling error — each row is an exact, deterministic
distribution; the scatter is token-realization dependence. The power law
lives in the marginal over token realizations: E[log A] (quenched) or
E[A] (annealed, the census), objects of the weights×input-distribution pair.
This is the row-level ground of exp-107's protocol-constitution finding.

### Bookkeeping fact (noticed at analysis, not a finding)

E[β] over the pool equals s_q *identically* — the OLS slope is a linear
functional of log A, so averaging commutes with fitting. The mean_beta
column is a within-run consistency check (it matches exp-110's quenched
slopes to 3 decimals), not evidence for the mixture model.

### Descriptive only (ineligible, cannot propagate): the reconstruction was
### nonetheless close

P_pred rebuilt from per-row (β, M) lands within |slope diff| ≤ 0.074 of the
measured annealed slope on all 31 registered pairs (median ≈ 0.033) — but
with median row R² ≈ 0.1 the fitted βs are trend+noise summaries, and the
agreement partly reflects the linearity identity above plus small curvature;
it is not a confirmation of the mixture model and is recorded only so the
number exists.

### EXPLORATORY (labeled): the noise picture is quantitatively consistent,
### and it sets a coherence scale

From exp-110's saved moments (no new runs), under independent-noise
signal-averaging, predicted single-row R² = Var_L(m)/(Var_L(m)+E_L[v])
matches the measured median row R² closely on native pairs (L2H1 0.208
predicted / 0.219 measured; L5H0 0.045/0.046; L7H11 0.052/0.054; L10H8
0.109/0.105; L4H10 0.106/0.121; L9H6 0.040/0.053). The rows-to-average
before the power law is visible at R² = 0.9: **k\* ≈ 30–220 rows** depending
on head. Caveat: independence across pooled rows is assumed, not tested;
rows within one input share content.

### What exp-111 establishes

1. **The conformal profile is ensemble-emergent.** No individual attention
   row, in any input regime, native or not, is described by the power law
   (K-a, registered). Forward-going documents must not say "the head's rows
   follow dx^(−2Δ)"; the true statement is "the head's ensemble-averaged
   profile follows dx^(−2Δ), emerging after ~30–220 rows of pooling."
2. **The mixture-of-row-slopes model died at its gate.** Rows do not carry
   well-defined individual exponents to disperse; the exp-110 variance term
   is per-token scatter structure, not slope dispersion. The damping
   anticorrelation still lacks a derivation — but the right variables are
   now the lag-profile of the token-scatter variance, not Var(β).
3. **Sharpened theory-of-A target (v3):** derive the *ensemble-mean* score
   drift ≈ −0.5·log dx of the weights×native-input pair — and why the
   token-scatter variance is lag-uniform enough in native regimes for the
   census exponent to sit on the drift (exp-110 exploratory: |var_term|
   ≈ 0.01–0.05 on these populations vs 0.19–0.38 base rate).
