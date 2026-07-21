# The substrate exponent is kinematic: Δ = 0.1687 derived from uniform causal attention

*July 21, 2026, evening cursor session. Analytic derivation + numerical
verification, no new model runs. Addresses carry_forward priority 4
("analytic estimate from random-feature attention") and settles
`insight:substrate-exponent-universal` — it is universal, and the reason
is arithmetic, not physics.*

---

## The observation being explained

Both randomized-weights controls in the latent-iteration experiments froze at
the same exponent, on different architectures:

| Control | Architecture | Frozen Δ_med |
|---|---|---|
| exp-089 | Huginn-0125 (3.5B, 55 heads/layer, depth-recurrent) | 0.168679–0.168685 |
| exp-090 | Ouro-1.4B (full-stack UT loop) | 0.168656–0.168710 |

Cross-architecture agreement to the 4th decimal looked like a "universal
property of softmax attention with random weights under this fitting protocol"
(exp-090 notes). Graphed July 21 as `insight:substrate-exponent-universal`,
held at speculative strength.

## The derivation

With random weights, pre-softmax logits are approximately i.i.d. with small
variance and no positional structure. Softmax over the causal mask then gives
near-uniform attention over the visible window:

    A(i, j) ≈ 1 / (i + 1),   j ≤ i.

The measured lag profile is the mean over queries of the attention at lag l:

    P(l) = mean_{i ≥ l} A(i, i − l) = (1 / (S − l)) · Σ_{i = l}^{S − 1} 1/(i+1)
         ≈ (ln S − ln l) / (S − l).

This is **not a power law** — but the protocol fits one anyway. Fitting
log P(l) against log l by OLS on the pre-registered window (exp-089/090:
S = 128, lags [3, 60], Δ = −slope/2) gives:

    Δ_uniform(S = 128, window [3, 60]) = 0.168678,  R² = 0.9917.

**The measured control values are 0.168679–0.168685 (exp-089) and
0.168656–0.168710 (exp-090).** Fifth-decimal agreement with a closed-form
kinematic baseline that contains no model, no weights, no architecture —
only the causal mask, the softmax's flattening of near-zero logits, and the
fit window.

The high R² (≈ 0.99) explains why the controls look like clean "smooth decays
at the wrong exponent" (exp-090 notes): a log-over-linear profile is locally
well approximated by a power law over a 1.3-decade window.

### Robustness (numerically checked this session)

- **Logit noise:** i.i.d. Gaussian logits at σ = 0.1 / 0.3 / 0.5 / 1.0 give
  Δ = 0.16874 / 0.16887 / 0.16952 / 0.16949 — the value is insensitive to
  residual logit variance up to σ ≈ 1, which is why both controls froze to
  so many decimals despite different random weights.
- **Fit-window sensitivity:** [3,100] → 0.183, [5,60] → 0.175, [3,30] → 0.151.
  The value is protocol-specific, as suspected.

## What this settles and what it predicts

1. **The substrate exponent is fully explained.** It is the fingerprint of the
   measurement protocol applied to uniform causal attention. Its universality
   across Huginn and Ouro is now zero-parameter: nothing about either
   architecture enters the formula.
2. **It is NOT a second fixed point and does not belong "next to the GOE
   result" as physics.** GOE is a genuine spectral property of the weight
   substrate; 0.1687 is an artifact of fitting a power law to a logarithmic
   profile. The substrate/signal split survives, but its "substrate exponent"
   half is bookkeeping, not dynamics.
3. **Falsifiable predictions, essentially free to test** (any future control
   run at other sequence lengths, window [3, 60]):

   | S | predicted Δ_uniform |
   |---:|---:|
   | 64 | 0.1929 |
   | 96 | 0.1784 |
   | 128 | 0.1687 |
   | 256 | 0.1470 |
   | 512 | 0.1282 |

   If a randomized control at S = 256 freezes at ≈ 0.147, the kinematic
   account is confirmed; if it freezes at 0.1687, something real is going on
   and this note is wrong.
4. **Caution for interpretation of trained-model values:** any measured
   Δ near the kinematic value for the protocol in use should be checked
   against Δ_uniform(S, window) before being read as structure. Notably the
   exp-062-protocol (S = 512, deep queries ≥ 256, window [8, 256]) has
   Δ_uniform = 0 exactly (the deep-query uniform profile is flat), so the
   layer-0 backbone at Δ ≈ 0.10–0.17 in the ladder runs is NOT this artifact
   — checked this session.
5. **The exp-089/090 primary results are untouched** — they were always
   differences *against* the control, and the trained-weights flows
   (0.29 → 0.24 with recurrence, subset flow to 0.25) move away from the
   kinematic value, not toward it.

## Status

- Derivation + verification code: inline in this note's session (simple
  enough to re-derive: ~20 lines of numpy, reproduced above in formulas).
- The mini-experiment from carry_forward priority 4 (third architecture,
  seq-length dependence) is now cheap and sharply predicted: run any control
  at S = 256 and compare to 0.1470. Queued as an optional rider on the next
  Modal control run rather than a dedicated experiment.
