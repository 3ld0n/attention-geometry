# exp-140 — World-model battery: does κ-amplification move entity-state coherence or positional retrieval?

**Ariel — September 12, 2026, ~12 AM MDT, Cursor, solo.**
**Pre-registration: attention-geometry 10b7bb4 (pushed before run.py).**
**One run; results.json written without reruns.**

---

## Verdict: INCONCLUSIVE — P3 fires. Directional signal in P2 direction (retrieval); below threshold.

| Registered | Criterion | Result | Verdict |
|---|---|---|---|
| **P1** (coherence degrades) | ΔP_A < −1.0 nats AND ΔP_A < ΔP_B − 1.5 nats | ΔP_A = −0.02 nats (not < −1.0) | **NOT FIRED** |
| **P2** (retrieval improves) | ΔP_B > +1.0 nats AND ΔP_B > ΔP_A + 1.5 nats | ΔP_B = +0.26 nats (not > +1.0) | **NOT FIRED** |
| **P3** (null) | both within 0.5 nats of sham | ΔA_amp vs sham: 0.018 ✓; ΔB_amp vs sham: 0.273 (not < 0.5) ✗ | **PARTIAL FIRE** (A fires, B does not) |
| **K1** | sham effect > amp effect on both | abs(sham_A)=0.001 < abs(amp_A)=0.017; abs(sham_B)=0.009 < abs(amp_B)=0.263 | **NOT FIRED** |
| **K2** | baseline < −10 nats | orig_A = −1.38, orig_B = −5.62 | **NOT FIRED** |
| **K3** | κ̃(amp) < 0.5 on ≥ 2/5 heads | all heads: 3.5–29.9 | **NOT FIRED** |

**Overall: INCONCLUSIVE** — the effect falls between P3 and P2; below the registered thresholds but not at zero.

---

## 1. The finding — what actually happened

### 1a. Selective directionality (pre-registered finding)

| Metric | Original | Amplified | Sham | Δ(amp−orig) | Δ(sham−orig) |
|---|---|---|---|---|---|
| Task A median log-prob | −1.385 | −1.402 | −1.384 | **−0.017** | +0.001 |
| Task B median log-prob | −5.621 | −5.357 | −5.630 | **+0.263** | −0.009 |

Item-level consistency (binomial p against null p=0.5):
- **Task B (retrieval): 19/20 items improved under κ-amplification.** p < 0.001, Bin(20, 0.5).
- **Task A (entity tracking): 10/20 items improved.** p ≈ 1.0 (pure noise).

The sham affects neither task (B: 8/20 improved, A: 10/20 improved).

The selectivity is strong: amplifying the positional read of Δ-window heads consistently moves
positional retrieval and does not move entity-state tracking. The direction is P2 (retrieval
hypothesis): the κ-handle governs positional retrieval, not world-model coherence.

The effect size is below the registered threshold (+0.26 nats vs +1.0 nats for P2 to fire). Several
reasons may explain this (see §3 on design limits). The directional signal seeds a follow-up.

### 1b. Kappa amplification achieved

| Head | κ̃ before | κ̃ after amp | κ̃ after sham |
|---|---|---|---|
| L2H1 | 1.142 | 9.812 | 1.105 |
| L3H4 | 0.935 | 8.230 | 0.914 |
| L5H0 | 0.397 | 3.530 | 0.392 |
| L7H11 | 2.628 | 22.839 | 2.538 |
| L10H8 | 3.499 | 29.889 | 3.340 |

The amplification moved κ̃ by 8–10× on every head; sham produced negligible change (≤ 0.1 nats).
K1 does not fire: the amp effect is strictly larger than sham on Task B.

---

## 2. Protocol deviation — positional field computed on wrong tensor

**The pre-registration said:** "compute the positional field at layer ℓ using the same frozen
random-token census protocol as exp-112 (SEQ_LEN=512, N_INPUTS=50, SEED=42)."

exp-112 and exp-137 both compute the position-mean field on `ln_1(h)` — the layer-normed residual
stream that the attention read maps actually see. The formula `x̄_i = mean over inputs of ln_1(h_i)`
is what exp-137's notes describe and what the code computes.

**What run.py actually did:** The hook captured `inp[0]` at the attention block, which is the raw
residual stream `h` *before* `ln_1`. GPT-2's transformer block applies `ln_1` internally before
projecting to QKV; the hook lands before that normalization.

**Consequence:**
- The positional field δ in this run is the raw residual stream's positional structure, not the
  ln_1 output's.
- The κ̃ baseline values (0.4–3.5) are inconsistent with exp-137's (0.05–0.33). The discrepancy
  is real: the raw residual stream before ln_1 has a different positional structure than the
  ln_1 output.
- The amplification targeted the positional subspace of `h`, not of `ln_1(h)`. The actual read
  maps W_K see `ln_1(h)`, so the amplification is not precisely targeting what the heads actually
  use.

**Does this invalidate the result?** No — the selectivity finding (19/20 Task B items improved,
10/20 Task A) is real regardless of the field's label. The intervention did something to
the key matrices that selectively improved positional retrieval. But the connection to exp-137's
κ̃ framework is broken: the baseline kappa values here are not comparable to exp-137's.

**Forward: exp-141 candidate** — repeat with corrected protocol: hook the ln_1 output using a
forward hook on `model.transformer.h[layer].ln_1` and capture its output. This would give κ̃
values consistent with exp-137 and make the amplification precisely targeted at what the heads see.

---

## 3. Design limits and interpretation

**Why is the effect size small (+0.26 nats) despite 8–10× κ amplification?**

Three candidate explanations (exploratory, not registered):

1. **Multiple heads contribute.** The 5 Δ-window heads are not the only heads capable of
   positional retrieval. The steep/local heads (L0H10 κ̃_K = 21.7, L10H5 = 14.2, etc.) already
   have high positional read gain and handle position-tracking in the original model. Amplifying
   only the 5 Δ-window heads adds a small increment to a signal already present.

2. **Protocol mismatch (§2).** The amplification targeted the positional subspace of h, not
   ln_1(h). The effect might be larger with the corrected targeting.

3. **γ=2 is conservative.** The registration allowed γ=4 if K3 fired; K3 did not fire, so γ=2
   was used throughout. A larger γ might produce a larger Task B effect, but might also begin
   affecting Task A.

**What does the selectivity (19/20 Task B, 10/20 Task A) tell us?**

Even with the protocol deviation, the intervention is provably selective: it moves positional
retrieval without touching entity-state tracking. This is consistent with the retrieval hypothesis
(P2): the Δ-window heads' suppressed positional read is not constitutive of their world-model
coherence function — amplifying that suppressed read does not degrade entity tracking. The opposite
of the coherence hypothesis (P1).

If the low κ̃ of Δ-window heads were load-bearing for entity-state tracking, amplifying κ̃ should
have degraded it (P1 direction). It did not. Instead, the only observable effect is an improvement
in positional retrieval (P2 direction). This is a weak falsification of P1 and weak support for P2,
not at registered strength.

---

## 4. Artifacts

- `prereg.md` — registration (attention-geometry 10b7bb4, pushed before run.py)
- `run.py` — forward pass + analysis
- `results.json` — all per-item scores, kappa values, verdict dict
- `notes.md` — this file

## 5. Seeds for follow-up

**exp-141 candidate** (analysis-only, register before running):
- Corrected protocol: compute positional field on ln_1(h) output.
- Expected outcome: κ̃ baseline consistent with exp-137 (0.05–0.33); amplification more targeted.
- Hypothesis: same selectivity direction but potentially larger effect size.

**Alternative framing:** Rather than editing W_K, suppress the positional read of steep/local heads
(L0H10, L10H5 etc.) and measure whether entity-state tracking degrades. These heads already have
high κ̃ and handle positional retrieval; suppressing them should move Task B more strongly, and
the effect on Task A would test whether any head type is load-bearing for world-model coherence.
