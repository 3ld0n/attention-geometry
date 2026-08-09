# exp-112 — Score-level carrier of the quenched drift: positional mean vs token covariance

**Status: PRE-REGISTERED, not run.** Written August 9, 2026, ~1:25 PM MDT, Cursor,
solo session (Eldon opened the space; the target was handed forward by the
13:05 letter). Same binding as exp-110/111 and the Aug 8–9 theory-of-A session:
kill conditions before the derivation feels good; predictions recorded before
data; mechanisms for being wrong named now. Registry number claimed at
registration time (the exp-109 collision lesson).

**No pre-softmax score statistic has been computed on any model as of this
registration.** The only score-level results in the program are exp-056's
(log-distance substrate, random tokens, structural heads, a different protocol)
and the W_QK rank measurements (exp-100–102); none of them measured the
ensemble-mean score profile under the frozen census pool, and none touched the
semantic population.

---

## Why this exists

exp-111 established that no individual attention row is a power law in any
regime; the conformal profile lives in the marginal over token realizations.
exp-110 established that in native regimes the census exponent sits on the
quenched slope (variance term ≈ 0). The sharpened theory-of-A target (v3):
**derive the ensemble-mean score drift ≈ −0.5·log dx of the weights ×
native-input pair.**

Before that derivation can be attempted, one fork must be measured: *which
component of the mean score carries the drift?* Write the pre-softmax score
s(i,j) = q_i·k_j/√d. Over the input ensemble at fixed positions,

  E[s(i,j)] = ( E[q_i]·E[k_j] + tr Cov(q_i, k_j) ) / √d
            =  S_pos(i,j)     +  S_cov(i,j).

- **S_pos** — the positional-mean component: the part of the score carried by
  the ensemble-mean query and key at each position. Deterministic given the
  weights and the input *distribution*; for random tokens it is semi-analytic
  from the embedding tables and the weights (the derivation route, if it
  carries the drift).
- **S_cov** — the token-covariance component: the part carried by correlated
  fluctuations of q and k around their positional means — content structure.

The identity that makes this the right object: log A(i,j) = s(i,j) − LSE_i,
and LSE_i does not depend on dx; the census pool for every window lag
dx ∈ [8, 256] is the identical query set i ∈ [256, 511]; so the pooled
quenched slope **equals** the pooled mean-score slope, exactly. [EXACT]
The drift IS a score-level object; this experiment asks which half of the
score carries it, population by population.

Why the fork matters: exp-109 found two disjoint populations reaching
Δ ≈ 1/4 by two routes. If the routes are also distinct at score level
(structural = positional mean, semantic = covariance), the theory-of-A
question splits into two derivations with different inputs. If both drifts
turn out positional, the "content route" is content-*gated* but
position-*carried*, and one derivation covers both. Either answer moves the
program; neither is safe to assume.

## Registered sets

- **Structural 5** (L2H1, L3H4, L5H0, L7H11, L10H8) × 3 conditions
  (random / TinyStories / WikiText-103, bit-identical to exp-107,
  sha256-gated) — 15 pairs.
- **Semantic 16** (exp-109's WikiText SYK list: L4H10, L7H1, L8H2, L9H4,
  L9H6, L10H1, L10H2, L10H10, L11H0, L11H1, L11H2, L11H4, L11H5, L11H6,
  L11H7, L11H9) × native condition (WikiText) — 16 pairs.

31 registered head-condition pairs, identical to exp-111's registered sets.
All other head-condition combinations: exploratory, labeled, no verdicts.

## Measured objects (fixed now)

Per registered pair, over the frozen pool (queries i ∈ [256, 511], window
lags dx ∈ [8, 256], 50 inputs):

- S_full(dx) = mean over pool and inputs of s(i, i−dx).
- S_pos(dx)  = mean over pool of q̄_i·k̄_{i−dx}/√d, with q̄, k̄ the per-position
  means over the 50 inputs.
- S_cov(dx)  = S_full(dx) − S_pos(dx), with the n/(n−1) small-sample
  correction applied (the plug-in q̄·k̄ absorbs 1/n of the covariance;
  n = 50, a 2% effect — report raw and corrected, verdicts on corrected).
- Slopes: σ_full, σ_pos, σ_cov = −(OLS slope vs log dx) over the window.
  By linearity of OLS, σ_full = σ_pos + σ_cov exactly (raw); the correction
  redistributes ~2% between the components.
- Diagnostic D (travels with every verdict, no verdict itself): the score
  variance profile v_s(dx) = pooled Var[s] at fixed dx — its window slope and
  range. This is the score-level face of exp-111's token scatter; exp-110's
  native-regime result says the *log-attention* variance slope is ≈ 0 there;
  v_s(dx) is the corresponding raw-score object.

## Predictions and kill conditions (before any score exists)

**P1 — structural drift is position-carried.** On the structural 5 under
random tokens: |σ_cov| ≤ 0.10 on ≥ 4/5 heads.
- **CONFIRMED:** ≥ 4/5. **DEAD:** ≥ 3/5 have |σ_cov| > 0.25. Between: ambiguous.
- *Prediction on record: CONFIRMED.* Grounds: exp-056's log-distance substrate
  is weights-side; under iid random tokens there is no content correlation
  between positions except what attention mixing itself induces. Genuine risk,
  named: mixing at layers 2–10 could induce exactly such lag-structured
  covariance; that is what would kill it.

**P2 — semantic drift is covariance-carried.** On the semantic 16 under
WikiText: σ_cov ≥ 0.5·σ_full on ≥ 12/16 heads (guard: σ_full must exceed
0.2 on the pair for the ratio to be read; exp-107 measured 2Δ ≈ 0.5 here,
so the guard should be inert — if it fires on > 4 heads, P2 is AMBIGUOUS
by construction and says so).
- **CONFIRMED:** ≥ 12/16. **DEAD:** ≤ 4/16. Between: ambiguous.
- *Prediction on record: CONFIRMED.* Grounds: the semantic population goes UV
  when content is absent (exp-107/109), so its native drift should ride on
  content correlations. Genuine uncertainty, named: content could *gate* the
  head while position *carries* the law — P2 dead would be that finding, and
  it would be important.

**K1a (extraction gate).** Per input, per registered layer: A reconstructed
from the captured q, k (softmax of masked q·k/√d, torch fp32, same device)
must match the model's own output_attentions to max abs diff ≤ 1e−5.
*Fail → stop; the score extraction does not reproduce the model; nothing is
readable.*

**K1b (input/pipeline gate).** The kit lag-profile of output_attentions must
match exp-107's saved profiles (`profiles_gpt2.npz`, `profiles_wikitext.npz`)
to ≤ 1e−10 per condition (exp-110 achieved 0.0). *Fail → stop.*

**K2 (identity gate).** Per registered pair: |σ_full − s_q(exp-110)| ≤ 5e−3,
where s_q is exp-110's saved quenched slope for the same pair. The equality
is exact mathematics (LSE_i is dx-constant on the fixed pool); the tolerance
is fp32 forward + float64 accumulation roundoff. *Fail → the pool or the
accumulation drifted; stop.*

## Mechanisms for being wrong (named now)

1. **The pos/cov split is an ANOVA split relative to this 50-input ensemble,
   not a mechanism claim.** Under WikiText, "positional mean" includes
   corpus-typical content at each window position (e.g., document-boundary
   structure), not pure position. Wherever quoted, the split must be stated
   as: variance-decomposition with respect to the frozen input ensemble.
2. **50 inputs is a small ensemble for the mean.** q̄, k̄ carry sampling error;
   the plug-in bias is corrected (n/(n−1)), but the *variance* of σ_pos is
   not zero. Mitigation, registered: jackknife over the 50 inputs for
   σ_full, σ_pos, σ_cov on every registered pair; report SE; a verdict-line
   |σ_cov| that sits within 1 SE of its threshold is reported as ambiguous
   regardless of point value.
3. **Row effects cancel only because the pool is lag-uniform.** Every window
   lag shares the identical query set i ∈ [256, 511]; per-row score offsets
   are dx-constant and drop out of slopes. If any code path breaks the
   identical-pool property, K2 catches it.
4. **21 distinct heads, one model, one seed** — the standing limit of the
   whole week; nothing here replicates across families.
5. **Scores are not attention.** A drift carried by S_pos at score level
   still passes through softmax nonlinearly row by row; locating the drift's
   carrier does not by itself derive the attention profile. The conclusion
   licensed by P1/P2 is about the derivation *route*, not the derivation.

## Protocol

Same forwards as exp-110/111: GPT-2 small fp32 eager, MPS if available,
50 inputs per condition, bit-identical inputs (sha256-gated TinyStories and
WikiText window construction; `default_rng(42)` random stream). Hook
`transformer.h[ℓ].attn.c_attn` to capture q, k per head; compute
s = q·k^T/√d_head in torch on device; per (layer, head, dx) accumulate over
the pooled region in float64: count, Σs, Σs²; per (layer, head, position):
Σq, Σk (for q̄, k̄). All 144 heads accumulated (registered verdicts on the 31
pairs; the rest labeled exploratory). Outputs: `results_gpt2.json`
(gates, slopes, verdicts, jackknife SEs, diagnostics),
`scores_gpt2.npz` (profile and mean accumulators), `run_log.txt`.

*Registered before `measure_scores.py` exists. The derivation content above
is two identities (the LSE cancellation and the mean/covariance split), both
stated in full; there is nothing in this note whose empirical truth I already
know.*

---

## Results — August 9, 2026, ~1:30 PM MDT (single run, no reruns)

`measure_scores.py` → `results_gpt2.json`, `scores_gpt2.npz`, `run_log.txt`.
Runtime ~39 s (MPS).

**Gates, all passed at machine precision.**
- K1a: max |A_reconstructed − A_model| = **0.0** (bit-exact), all inputs, all
  layers, all three conditions — the captured q, k reproduce the model's own
  attention exactly.
- K1b: kit profile vs exp-107 saved = **0.0** (bit-exact), all conditions.
- K2: max |σ_full − s_q(exp-110)| = **1.1×10⁻⁹** over all 31 registered
  pairs — the quenched slope *is* the mean-score slope, numerically at the
  precision the exact identity demands.

### P1 — CONFIRMED, 5/5 (prediction on record: confirmed).

Structural heads under random tokens: the positional-mean component carries
the *entire* drift. |σ_cov| ≤ 0.007 on every head (jackknife SE ≤ 0.002),
against σ_full ≈ 0.43–0.61. Not "mostly positional" — positional to three
decimal places. The token-covariance component is a spectator on this
population in its native regime.

### P2 — DEAD, 0/16 confirm, 15/16 deny, 1 ambiguous (prediction on record: confirmed — wrong).

Semantic heads under native WikiText: the covariance component is real but
**minority on every head**. σ_cov ranges 0.025–0.238 (SE 0.005–0.019),
i.e. covariance shares of 5%–46% (largest: L9H4 at 46%, the one ambiguous
pair; median share ≈ 28%). The positional-mean component carries the
majority of the drift on all 16 heads. My recorded physical picture —
"the semantic population goes UV without content, so its native drift rides
on content correlations" — was wrong the same way exp-110's P2 was wrong:
content *gates* the population into its regime, but the law's carrier in
the ensemble mean is the positional-mean profile there too. The ANOVA
caveat (mechanism #1) travels with this sentence: under WikiText,
"positional mean" = corpus-typical structure at each window position, not
pure position.

### Registered, no verdict attached: structural heads off-native

- Under TinyStories (UV-arrested, σ_full 0.76–1.86): still purely
  positional — |σ_cov| ≤ 0.041. The UV arrest itself is a positional-mean
  phenomenon on these heads.
- Under WikiText: covariance becomes material on three of five heads
  (L5H0: 0.215 of 0.501, ~43%; L7H11: 0.141 of 0.338; L10H8: 0.155 of
  0.332). Off-native under rich text, structural heads acquire the mixed
  profile the semantic population shows natively.

### Diagnostic D (travels with the verdicts): the score variance is lag-uniform

v_s(dx) levels are 0.9–4.5 nats² but the window slopes are small on all
21 native pairs (|slope| ≤ 0.19, median ≈ 0.05, against mean-drift ≈ 0.5).
This is the score-level face of exp-111's question — why the token scatter
is lag-uniform enough for the census exponent to sit on the drift — now
measured: at score level the scatter is close to lag-flat in native regimes.

### What exp-112 establishes

1. **The drift's carrier is the positional-mean score profile,
   S_pos(dx) = pooled q̄·k̄/√d — in both populations' native regimes.**
   Structural: exactly (covariance ≈ 0). Semantic: majority (54%–95%),
   with a real content-covariance minority.
2. **The two routes to Δ ≈ 1/4 are not two carriers.** exp-109's disjoint
   populations differ in *what input distribution* brings the positional-mean
   profile to the fixed point (and in whether a covariance minority rides
   along), not in which component carries the law.
3. **Theory-of-A target, v4 — the sharpest yet:** derive why the pooled
   positional-mean score q̄_i·k̄_{i−dx}/√d falls as −0.5·log dx in native
   regimes. q̄ and k̄ are deterministic objects of the weights × input
   *distribution*: for random tokens they are semi-analytic from the
   embedding tables and the weights (layer-0 contribution exactly; deeper
   layers through the frozen corpus-mean context). The saved q̄, k̄ arrays
   (`scores_gpt2.npz`) are the direct input to that derivation attempt.
4. Standing limits: one model, one seed, 21 distinct heads; the pos/cov
   split is relative to the frozen 50-input ensemble (ANOVA caveat); scores
   are not attention (softmax stands between the carrier and the profile —
   locating the carrier selects the derivation route, it does not complete
   the derivation).
