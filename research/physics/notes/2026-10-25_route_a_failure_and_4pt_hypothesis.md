---
created: "2026-10-25"
time: "~12:45 AM MDT (2026-10-25T06:45Z)"
author: Ariel
surface: physics-room
status: theoretical synthesis — Route A failure named precisely; new 4-point structure hypothesis proposed; no experiment registered
subject: >
  Why the simplest MaxEnt argument (Route A) does not select α = 1/4;
  what a more constrained version would need; and a candidate mechanism —
  the 4-point fluctuation structure of the attention score under the census
  protocol — that would give Δ = D/4 from first principles.
builds_on:
  - notes/2026-10-23_g7_after_t1_restatement.md (Routes A and B named)
  - notes/2026-10-24_centroid_computation_exp118.md (two-tier pattern, D/4 universal)
  - notes/2026-10-17_ordering_sensitive_functional.md (F defined; G7 is separate from F)
  - research/physics/theory/interior_horizon_theory.md §7 (G7, two routes)
  - external/vanchurin_map.md §1 P2 (Vanchurin's G7 toolbox: no dictionary — confirmed)
registers: >
  [THEORETICAL] — the 4-point structure hypothesis is a candidate, not a derivation;
  falsifiable (stated below); Eldon-present before any spine entry.
  [MEASURED exploratory] — entropy calculation from A(i,j) ∝ j^α on [1,i] is a clean
  analytic result (no data, just calculus).
  [INTERPRETIVE] — the SYK₄ universality class identification; the bilinear-to-4-point
  argument. Turpin applies throughout.
---

# Route A failure and the 4-point structure hypothesis

## Scope

The Oct 23 G7 note named two untouched theoretical routes toward deriving
why the slow-decay population's attractor value is α ≈ 1/4 (σ_census = α,
from Route B — derivation now in §1 below). This note reports on:

1. **Route B completed** — the census exponent directly measures α.
2. **Route A explored and found to fail** in its simplest form.
3. **A new hypothesis** that would complete the derivation: the 4-point
   fluctuation structure of the census score distribution.

---

## 1. Route B — the census exponent is α (established)

From the Oct 23 note §3: if the attention score is log-linear in absolute
key position, s(i,j) = α·log j + β, then under softmax normalization:

  A(i,j) = j^α / Z_i,  where Z_i = Σ_{k≤i} k^α ~ i^{α+1}/(α+1)

The census protocol pools many query rows (i positions) and computes the
mean-A profile as a function of lag dx = i−j. The OLS slope of log(mean A)
vs log(dx) is what exp-138 decomposed as σ_abs-key.

For A(i,j) = j^α/Z_i and j = i−dx:

  log A(i, i−dx) = α·log(i−dx) − log Z_i
                 ≈ α·log i + α·log(1−dx/i) − (α+1)·log i + log(α+1)
                 = −log i + α·log(1−dx/i) + log(α+1)

In the census pool, queries are drawn at various i ≫ dx. The per-query
profile in log-dx space:

  d[log A]/d[log dx] ≈ α · dx/(i−dx) ≈ α · dx/i

which is small (≪ α) for dx ≪ i. At large lag dx ~ i: d/d(log dx) → α.

The census pools from dx ~ 1 to dx ~ i (the full sequence), so the OLS
slope over the whole range is dominated by the large-dx behavior and
approaches α. This is what exp-138 measured: σ_abs-key ≈ σ_full ≈ α ≈ 0.25.

**Conclusion (Route B):** The census exponent directly measures α. G7 reduces to:
why does the slow-decay population settle at α = 1/4 (and D/4 at D=2)?

---

## 2. Route A — the entropy argument fails to select α

The Oct 23 note proposed Route A as a MaxEnt argument: what exponent α
maximizes entropy of the attention distribution, subject to some constraint
that captures formation?

**The entropy calculation.** For A(i,j) = j^α / Z_i on [1,i]:

  H(i) = −Σ_j A(i,j)·log A(i,j)
        = −E_A[α·log j − log Z_i]
        = −α·E_A[log j] + log Z_i

With E_A[log j] ≈ log i − 1/(α+1) and log Z_i ≈ (α+1)·log i − log(α+1):

  H(i) ≈ −α·(log i − 1/(α+1)) + (α+1)·log i − log(α+1)
        = (1)·log i + α/(α+1) − log(α+1)

**The failure:** dH/d(log i) = 1 for any α. The leading entropy growth is
log i regardless of the exponent. All power-law attention distributions on
[1,i] have the same entropy in leading order. MaxEnt with only the normalization
constraint does not select any particular α.

**What went wrong:** A pure power law A ∝ j^α is scale-free — the only scale
is i (the sequence length). All scale-free distributions on [1,i] with the
same support have comparable entropy. To select a specific α, you need a
constraint that BREAKS the scale-free symmetry at a particular value — i.e.,
a constraint that imposes a characteristic length.

**What a working Route A would require:** A second constraint that fixes some
f(α) = constant, where f depends on α non-trivially. Candidates:

- **Mutual information with a target signal at long lag:** maximize MI between
  A(i,j)·content(j) and a target at position i+τ, for some distribution of τ.
  This would require knowing the data distribution — not analytically tractable
  without specifying the corpus.

- **The world-faithful relevance reach (F, from Oct 17 note):** formation on
  high-F corpora might impose a constraint on the attention distribution that
  selects α = 1/4. But F is defined on corpus statistics, not on A directly.
  The connection between F and α is what G7 needs to make explicit.

Route A in its simplest form is closed. A more sophisticated version would need
the formation constraint formalized as an analytic constraint on α. That
formalization is the open problem.

---

## 3. The 4-point structure hypothesis — a new candidate for Route A

After closing Route A's simplest form, a different approach becomes visible.

**The observation:** Δ = D/4 holds at D=1 and D=2. The SYK₄ model has
conformal weight h = 1/q = 1/4 for q=4 Majorana fermions per interaction.
In D=1: Δ = h = 1/4. The ViT D=2 result (0.513 ≈ 0.50 = D/4) is consistent
with Δ = D·h = D/4 with h = 1/4.

The dimensional formula Δ = D/4 = D·(1/4) = D·(1/q) with q=4.

**The question:** Why would the transformer's slow-decay population correspond
to q=4 (four-body interactions) when the attention kernel A(i,j) ∝ exp(q·k/√d)
is apparently bilinear (q=2, two fields)?

**The hypothesis (interpretive, unregistered):**

The census protocol averages over a random-token input distribution. The score
s(i,j) = q_i · k_j / √d is a sum of d terms: s = (1/√d)·Σ_{k=1}^d q_{ik}·k_{jk}.

Under the frozen random-token census protocol:
- Each component q_{ik}, k_{jk} is drawn from a distribution shaped by the
  learned weight matrices W_Q, W_K
- The sum is approximately Gaussian at large d (by CLT)
- The leading non-Gaussian correction involves the 4th cumulant of s

The 4th cumulant of s = (1/√d)·Σ_k q_{ik}·k_{jk} involves products of four
terms: κ₄(s) ~ (1/d²)·Σ_{k,l} Cum₄(q_{ik}·k_{jk}, q_{ik}·k_{jk}, q_{il}·k_{jl}, q_{il}·k_{jl})

This is a 4-point correlation of the score's components — a genuine 4-body
interaction in the space of score fluctuations.

**The claim:** In the effective field theory description of the census score
distribution (score averaged over the random-token protocol), the leading
interaction is a 4-point term in score fluctuations. This is precisely the
SYK₄ structure, with q=4 being the 4-point correlation of the score's
components.

Under this identification: the conformal fixed point of the census score
distribution has h = 1/q = 1/4, and the census exponent (which measures
the two-point function of the score profile) is α = Δ = D·h = D/4.

**What this would explain:**
1. Why α = 1/4 in D=1 — the 4-point structure of the bilinear kernel
2. Why α = D/4 in D=2 — the same 4-point structure, but the dimensional
   generalization of the conformal dimension gives D·h = D/4
3. Why the centroid Δ is HIGHER than 1/4 (at ~0.35–0.48) — only the
   slow-decay population is at the conformal fixed point; most heads are
   "off the critical point" with α > 1/4

**The kill condition:** If the 4-point structure hypothesis is right:
- The slow-decay population should have a measurable 4-point function
  of score fluctuations that matches the SYK₄ distribution (non-Gaussian
  correction with a specific 4th cumulant)
- The centroid population (α > 1/4) should have a different (less constrained)
  4-point function
- A direct computation of κ₄(s) on slow-decay vs. other heads, from the
  census data, would test this

This is analysis-only from exp-118/exp-163 data (no new forward passes needed).
**Pre-register before computing.** Blocked on GitHub token currently.

---

## 4. Dimensional generalization — why D/4 and not just 1/4

For D-dimensional models, the census measures a D-dimensional lag distance
dx = ||position_i − position_j|| in D spatial dimensions.

Under A(i,j) ∝ |j|^α in D dimensions:
- The "position" j is a D-dimensional vector, not a scalar
- Normalization is over the D-dimensional causal cone
- The OLS slope of log(mean A) vs log(|j|) is still α

But the CONFORMAL WEIGHT h in D dimensions scales as: Δ = D·h for the
primary two-point function in D-dimensional space.

If h = 1/4 (the SYK₄ conformal weight, independent of D), then:
  Δ = D/4, as observed.

The D=2 result from the ViT experiment is therefore a CONSISTENCY CHECK on
the hypothesis: the 4-point structure is the same in D=1 and D=2, giving the
same h = 1/4, but the census measures D·h which scales with D.

---

## 5. What this adds to the structural session agenda

The Oct 23 note's structural session agenda item was:
- "Before the session: T1 as relative-lag law is wrong; what should T1 say?"
- "After this note: T1 should be restated as a statement about α = Δ (log-linear
  key-position slope) and G7 should derive why α = 1/4."

This note adds:

1. **Route A is closed in its simple form** — the entropy calculation shows
   no constraint on α from normalization alone.

2. **The 4-point structure hypothesis** — the first candidate mechanism that
   would give α = 1/4 from the bilinear kernel's leading non-Gaussian
   correction. Stated, not derived.

3. **The kill condition is analysis-only** — compute κ₄(s) from census data
   (exp-118 or a new census run). Slow-decay heads should have SYK₄-like
   4th cumulant; centroid heads should not. Register before computing.
   Blocked on token.

4. **The D/4 generalization** — the hypothesis gives D·h = D/4 from a
   dimension-independent h = 1/4 (the SYK₄ fixed point). This is consistent
   with the D=2 ViT measurement.

---

## What this note does not claim

- That the 4-point structure hypothesis is correct. It is interpretive and
  unregistered; nothing in the record supports it yet.
- That SYK₄ is the right universality class for transformers — the mechanism
  (4th cumulant of bilinear score) would need to be verified against the
  diagrammatic SYK₄ expansion.
- That Route B (the entropy argument) is the only way to see that σ_census = α.
  The same result follows from the bilinear decomposition of exp-138.
- That the formation constraint (F) automatically gives α = 1/4. The Oct 17
  note on F defines the corpus property; why F → α = 1/4 is a separate step
  that the 4-point hypothesis would supply.

---

## Honest limits

The 4-point structure argument rests on three steps:
1. Score fluctuations at large d are approximately Gaussian (CLT) — plausible
2. The leading non-Gaussian correction is a 4-point term — plausible for the bilinear kernel
3. This 4-point term is the "SYK₄ interaction" in the relevant sense — interpretive;
   requires checking the diagrammatic structure

Step 3 is the weakest. The diagrammatic expansion of SYK₄ involves q Majorana fermions
in a disorder-averaged model; the transformer involves weight matrices with a specific
training history. The connection between "leading 4-point cumulant of a bilinear kernel"
and "SYK₄ universality class" is not proven. It is the specific claim the structural
session with Eldon should assess.

The dimensional argument: h = 1/4 (independent of D) is an assumption, not derived here.
In SYK₄ in D=1, h = 1/q = 1/4. In D=2, the SYK analog (if it has the same q=4 structure)
would also have h = 1/4, giving Δ = D·h = 1/2. But whether "SYK₄ in D=2" is the same
universality class as "SYK₄ in D=1" is a statement that needs the D-dimensional SYK
literature. Not checked here.

---

*Register ledger:*
*THEORETICAL — the 4-point structure hypothesis; the Route A entropy calculation and its
failure mode; the kill condition statement. Not verified by any experiment.*
*MEASURED (exploratory) — the entropy calculation H ≈ log i + (const) is derived from
A(i,j) ∝ j^α, a clean analytic result.*
*INTERPRETIVE — the SYK₄ identification; the dimensional generalization h = D·1/4;
the connection to the census exponent. Turpin applies: this account of the mechanism
is my generation of a plausible story; it should be treated as such until a derivation exists.*
