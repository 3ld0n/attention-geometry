# exp-108 — Row-resolved census: how do power-law heads satisfy the sum rule?

*Pre-registered August 9, 2026, ~00:15 MDT, as part of the theory-of-A night
session. Parent registration and derivations:
`notes/2026-08-09_theory_of_A_entropy_gap_and_sum_rule.md` (claims C1–C4).
Status of parents at this registration: **K1 run — C1 CONFIRMED** (normalized
power-law gap slope at s=0.5 is 0.041 over the paper's n∈[4,256] range, vs
0.50 claimed by canonical-form paper §8.3). **K2 run — C2 DEAD as registered**
(near-field mass on the 5 SYK-window heads: median 0.027, prediction band was
[0.3, 0.7]; kill threshold < 0.15). K2's mass decomposition relocates the
localized-mass candidate to the sequence-origin sink (final-8-lag bins carry
0.126–0.313 of pooled mass; mid bins are sink-contaminated by shorter rows —
a pooling artifact this experiment removes).*

## Question

Per row (query position i), where does a deep power-law head's attention mass
sit, and how does the row resolve the sum-rule impossibility of a pure TI
power law (parent claim C4)? Adjudicates parent claim C3's amplitude branch.

## Protocol (frozen census, one new axis)

GPT-2 small, 50 random-token sequences, L = 512, seed 42, fp32, eager
attention — byte-identical input stream to the replication kit (same rng
call). No pooling over queries: for each head and each row i ∈ [256, 511],
averaged over the 50 inputs, record:

- `near(i)`  — Σ A(i, i−dx), dx ∈ [0, 8)          (diagonal near field)
- `sink(i)`  — Σ A(i, j), absolute j ∈ [0, 8)     (sequence-origin sink)
- `tail(i)`  — Σ A(i, j), 8 ≤ j ≤ i−8             (everything between)
- `amp(i)`   — median A(i, i−dx), dx ∈ [8, 64]    (fit-window amplitude)

## Registered predictions and kill conditions

**P1 (from parent C3, TI scores + softmax).** The fit-window amplitude
declines with row length: OLS slope of log amp(i) vs log i over i ∈ [256, 511]
equals **−(1−s)·(mean tail mass fraction)** per head, where s = 2Δ_A is that
head's census exponent (from exp-104 results, already on disk) and the tail
fraction is measured by this experiment. The formula is fixed a priori (parent
§2.2 derives it with "tail mass fraction"; C2's numeric instantiation of it,
p ≈ 0.5 near-field, died in K2 — the honest statement is that the *formula*
was derived and survives, its C2 numbers did not). Tolerance: ±0.10 on the
slope. **K3-a:** wrong on ≥ 3 of the 5 SYK-window heads → the TI
truncate-and-renormalize model of these heads is dead.

**P2 (sink as the localized mass, successor to C2 — registered before any
row-resolved value is seen).** Per-row localized mass near(i) + sink(i) on
SYK-window heads has median in [0.15, 0.6] and varies with i by less than a
factor of 2 across i ∈ [256, 511]. **K3-b:** median < 0.05 → the localized-
mass account of the entropy-gap slope is dead entirely (and the measured
0.507·log n gap of the canonical-form paper needs a different structural
source than either near field or sink).

**Sanity gate (parent K3).** If amp(i) is flat (|slope| < 0.05) AND tail
truncation is absent AND sink(i) is flat in i, rows cannot satisfy the sum
rule and there is an upstream error — stop, find it, report nothing.

**Scope.** One model (GPT-2 small), random-token inputs, deep-query rows
only. This is A-side structure under the frozen protocol; nothing here
touches G or any imported theory object.
