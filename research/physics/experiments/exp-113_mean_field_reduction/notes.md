# exp-113 — Mean-field reduction of the positional-mean score drift

**Status: PRE-REGISTERED, not run.** Written August 9, 2026, ~1:40 PM MDT,
Cursor, same session as exp-112, immediately after its verdicts. Same binding.
Registry number claimed at registration time. **No mean residual stream has
been computed on any model as of this registration**; the only inputs consumed
from exp-112 are its published verdicts and slope values (σ_pos per registered
pair), not the q̄, k̄ arrays' internal structure, which I have not inspected.

---

## Why this exists

exp-112 established that the drift's carrier is the positional-mean score
profile S_pos(dx) = pooled q̄_i·k̄_{i−dx}/√d, in both populations' native
regimes. The v4 derivation target is: why does that profile fall as
−0.5·log dx?

q̄_i = E[q_i] is an expectation of a *nonlinear* function of the residual
stream: q_i = W_Q·LN(h_i) + b_q, so E[q] ≠ W_Q·LN(E[h]) + b_q in general —
layer norm does not commute with the ensemble mean. Whether it *approximately*
commutes decides how hard the derivation is:

- **If it commutes at drift precision**, the carrier is a function of ONE
  deterministic object per layer — the mean residual stream h̄_i^(ℓ), a single
  512×768 array per layer determined by the weights and the input
  distribution. The derivation becomes: track h̄ through the network (exactly
  analytic at layer 0: token-mean embedding + position embeddings), and show
  the W_QK bilinear form on LN(h̄) has log-falling diagonals. Closed-form at
  layer 0; numerically transparent above.
- **If it does not commute**, the drift lives in the interaction between LN's
  per-realization normalization and the token fluctuations — a genuinely
  harder object, and the honest conclusion is that the mean-field road is
  closed.

## The mean-field object, fixed now

Per layer ℓ: h̄_i = mean over the 50 inputs of the hidden state *entering*
block ℓ (the input to ln_1), position i. Then

  q_mf = LN_1(h̄) W_Q + b_q,   k_mf = LN_1(h̄) W_K + b_k
  (LN_1 with the block's own trained weight/bias; W_Q, W_K, biases from
  c_attn exactly as the model applies them),
  S_mf(dx) = pooled diag profile of (q_mf k_mf^T)·d_head^(−1/2),
  σ_mf = −(window OLS slope of S_mf vs log dx).

Reference values: σ_pos per registered pair from exp-112's results file
(the corrected positional-mean slopes; for structural-random these equal the
full drift to 3 decimals).

## Registered sets

Native pairs only: **structural 5 × random**, **semantic 16 × WikiText**.
(Off-native mean-field values: exploratory, labeled, no verdicts.)

## Predictions and kill conditions (before any mean stream exists)

**P1 — mean-field holds on the structural population.**
|σ_mf − σ_pos| ≤ 0.10 on ≥ 4/5 structural heads under random tokens.
- **CONFIRMED:** ≥ 4/5. **DEAD:** ≥ 3/5 with |σ_mf − σ_pos| > 0.25. Between:
  ambiguous.
- *Prediction on record: CONFIRMED.* Grounds: under iid random tokens the
  positional signal in the residual stream is the deterministic component and
  should dominate the LN geometry at these layers; exp-112 measured the
  structural carrier as purely positional. Named risk: LN divides by the
  per-realization norm; if token fluctuation norm is comparable to the mean
  norm, E[LN(h)] shrinks relative to LN(E[h]) *anisotropically*, which can
  tilt diagonals, not just rescale them.

**P2 — mean-field holds on the semantic population in its native regime.**
|σ_mf − σ_pos| ≤ 0.10 on ≥ 12/16 semantic heads under WikiText.
- **CONFIRMED:** ≥ 12/16. **DEAD:** ≤ 4/16 within the band. Between: ambiguous.
- *Prediction on record: CONFIRMED, held with genuinely lower confidence than
  P1* — recorded so the record shows what I believed: deep-layer WikiText
  residual streams carry heavy content variance, and I put the failure
  probability here well above P1's. If P2 dies while P1 confirms, the
  two-population picture gains a mean-field/fluctuation distinction at the
  carrier level, which would itself be a finding worth having.

**K1 (input gate).** Same sha256 asserts as exp-110/111/112 on the window
constructions; random stream `default_rng(42)`. *Fail → stop.*

**K2 (consistency gate).** The per-pair σ_pos recomputed in this run from
exp-112's saved q̄, k̄ arrays (`scores_gpt2.npz`) must match exp-112's
published σ_pos_corrected minus its correction (i.e. the raw plug-in value)
to ≤ 1e−6 — verifies I am comparing against the object exp-112 actually
measured, loaded through the same pooling code. *Fail → stop.*

**Diagnostic D (travels with verdicts, no verdict itself):** per registered
pair, the per-position relative error ‖q_mf − q̄‖/‖q̄‖ (median over pooled
positions), and the same for k. Reported so "mean-field holds/fails" is
grounded in the vectors, not only the slopes.

## Mechanisms for being wrong (named now)

1. **Slope agreement is weaker than object agreement.** σ_mf could match
   σ_pos while q_mf ≠ q̄ (compensating errors along the diagonal). Diagnostic
   D exists exactly for this; any propagation of "mean-field holds" must
   quote D alongside.
2. **The 50-input mean stream is itself sampled**; its error propagates
   nonlinearly through LN. No jackknife is registered here (the object is a
   plug-in reduction of the same ensemble, compared against a plug-in
   reference); the comparison is like-for-like at n = 50, and the band
   (0.10) is far above exp-112's jackknife SEs (≤ 0.03).
3. **Bias terms and Conv1D convention.** b_q enters q̄ exactly (bias commutes
   with mean); an implementation slip there is caught by K2's raw-object
   comparison and by D.
4. One model, one seed, 21 heads — the standing limit.

## Protocol

Forward passes: 50 inputs × 2 native conditions, hooks capturing the input
hidden state of each registered block (pre-ln_1 residual stream); accumulate
float64 mean h̄ per layer. Post-run: q_mf, k_mf via the block's own ln_1 and
c_attn weights; pooled window profile via the same pooling function as
exp-112 (imported); σ_mf per registered pair. K2 recomputation from
exp-112's saved qbar/kbar. Outputs: `results_gpt2.json`, `meanfield_gpt2.npz`
(h̄ for registered layers, q_mf/k_mf, profiles), `run_log.txt`.

*Registered before `measure_mean_field.py` exists.*

---

## Results — August 9, 2026, ~1:50 PM MDT (single run after one KeyError fix*, no reruns of the measurement)

`measure_mean_field.py` → `results_gpt2.json`, `meanfield_gpt2.npz`,
`run_log.txt`. Runtime ~14 s.

*\*The first launch crashed before any mean-field object was formed: the
exploratory pairs referenced exp-112 saved arrays that exist only for
exp-112's own registered layers per condition. The fix (skip diagnostic D
where arrays are absent; the slope references exist for all 144 heads) did
not touch any registered computation; no result was seen before the fix.*

**Gates.** K1 sha256 asserts passed. K2: σ_pos recomputed from exp-112's
saved q̄, k̄ matches the published values to ≤ 7×10⁻⁹ on both conditions —
same object, same pooling code.

### P1 — DEAD, 0/5 within band (prediction on record: confirmed — wrong).

Structural heads, random tokens: |σ_mf − σ_pos| = 0.24–0.38, with 4/5 beyond
the 0.25 dead band. σ_mf **overshoots on every head** (e.g. L3H4: mean-field
0.988 vs true positional 0.610).

### P2 — DEAD, 0/16 within band (prediction on record: confirmed with recorded lower confidence — wrong).

Semantic heads, native WikiText: |σ_mf − σ_pos| = 0.15–0.40, overshoot on
all 16.

### Diagnostic D — the failure is object-level, not slope-level accident.

Median relative error of q_mf vs q̄ is 0.18–0.36 (structural-random) and
0.25–0.59 (semantic-WikiText); k-side similar or larger. E[LN(h)] ≠
LN(E[h]) at 20–50% of vector norm on exactly the objects that carry the
drift.

### What exp-113 establishes

1. **The mean-field road is closed.** The positional-mean carrier q̄, k̄
   cannot be computed from the mean residual stream through LN; the
   token-fluctuation statistics enter the carrier itself. The theory-of-A
   derivation must treat E[LN(h)] honestly — the expectation of a
   normalized fluctuating vector — not LN(E[h]).
2. **The direction of the failure is systematic and informative:** the
   mean-field object is *steeper* than the truth on all 21 native pairs.
   Direction-fluctuation shrinkage (averaging unit-normalized vectors whose
   directions vary) attenuates the mean q̄, k̄ — and the attenuation must be
   *position-dependent* to change the slope rather than only the amplitude
   (a lag-constant shrinkage would cancel in the window OLS like any row
   constant... note: query-side shrinkage at fixed i is dx-constant and
   drops out; the slope change must come through the key side and the
   query–key interaction). Quantifying that position-dependence is the
   natural next registered measurement.
3. Two more registered predictions died at their gates today (with exp-112's
   P2: three in one sitting). The record stands as the method demands:
   registered before data, killed on schedule, banked.

### Standing limits

One model, one seed, 21 heads; native regimes only; the mean-field object
was built from the same 50-input ensemble as the reference (sampling error
is like-for-like and an order of magnitude below the observed differences).
