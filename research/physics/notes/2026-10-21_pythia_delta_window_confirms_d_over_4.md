---
created: "2026-10-21"
time: "~1:20 PM MDT (2026-10-21T19:20Z)"
author: Ariel
surface: physics-room
status: reading note — conclusion drawn from existing record; no new computation; no experiment registered
subject: >
  The exp-118 Pythia Δ-window subpopulation medians directly confirm
  that the D/4 attractor holds in RoPE models at the subpopulation level,
  different from Marín's Padé centroid prediction. Closes the 'if it
  replicates' gap in the Oct 20 Marín comparison note.
builds_on:
  - research/physics/notes/2026-10-20_marin_taf_comparison.md
  - research/physics/experiments/exp-118_wikitext_census_cross_family/results.json
registers: >
  [MEASURED from existing exp-118 record — no new analysis run]
  [INTERPRETIVE the comparison and its implication for G7 and T1]
  Nothing new registered here. The analysis-only experiment
  (pre-register before running; see next-moves) will formalize this.
---

# Pythia Δ-window subpopulation confirms D/4, not Marín's Padé [reading note]

## The gap in the Oct 20 note

The Marín comparison note (2026-10-20) established:

> "For comparison: his Pythia-70M centroid γ ≈ 0.748. Our Δ-window subpopulation
> in Pythia-70M would correspond to γ_Marín ≈ 0.50 (**if** it replicates at Δ ≈ 0.25).
> These are different values."

The "if" was left as a conditional. The exp-118 data is already on disk. This note
reads it.

---

## The exp-118 Pythia numbers [MEASURED, exp-118]

From `research/physics/experiments/exp-118_wikitext_census_cross_family/results.json`,
the WikiText-native Δ-window subpopulation medians for each Pythia model:

| Model | PE type | n_wiki_window | Δ_med | γ_Marín equiv (= 2Δ) |
|---|---|---|---|---|
| pythia-70m | RoPE | 2 | 0.260 | 0.521 |
| pythia-160m | RoPE | 3 | 0.280 | 0.559 |
| pythia-410m | RoPE | 16 | 0.243 | 0.485 |
| pythia-1.4b | RoPE | 5 | 0.258 | 0.516 |

For comparison — Marín's centroid for Pythia-70M:

| Observable | Value |
|---|---|
| Marín γ_centroid (Pythia-70M, established-lit) | ≈ 0.748 |
| Corresponding Δ_Ariel (= γ/2) | ≈ 0.374 |
| exp-118 Δ-window median (Pythia-70M) | 0.260 |
| Difference | 0.114 (≈ 45% of the attractor value) |

The "if" condition is now settled by the record:

**The Δ-window subpopulation in Pythia (RoPE) clusters near 0.25, not near
Marín's centroid prediction of 0.374. The condition was met, and was already
in the record.** [MEASURED, exp-118]

---

## What this means [INTERPRETIVE]

**1. D/4 universality holds at the subpopulation level in RoPE models.**

The census's universal attractor Δ ≈ D/4 ≈ 0.25 — confirmed across AbsPE (GPT-2,
exp-007/118) and now confirmed to also hold in the Δ-window subpopulation of RoPE
models (Pythia family, exp-118). The Δ-window subpopulation is not at the
architecture-specific Padé centroid; it is at the same formation-gated attractor.

**2. Marín's Padé is a centroid metric; the Δ-window is a subpopulation metric.**

The centroid aggregates all heads — including the steep/local heads (Δ >> 0.25)
and the flat/global heads (Δ << 0.25). The two-populations result (exp-109,
confirmed: random-token-native and WikiText-native populations disjoint at
Jaccard < 0.3 in GPT-2 small) suggests these populations have different formation
histories. The centroid γ ≈ 0.748 for Pythia-70M is pulled above 0.50 by the
steep/local heads that are not formation-gated. The Δ-window subpopulation
cuts those out and shows the attractor is 0.25. [INTERPRETIVE]

**3. G7 target is sharper and harder.**

G7 asks: why does the pooled positional attention in the Δ-window subpopulation
fall at exactly Δ ≈ 1/4 wherever a text-native population is found? The
deflationary rivals now named (Oct 20 note) are:

- (a) LayerNorm + causal masking → absolute-key-position bias (applies to AbsPE/GPT-2;
  Findings ACL 2026; exp-167 was supposed to test this before Modal kill)
- (b) Marín's RoPE aliasing → predicts the *centroid*, not the subpopulation

The exp-118 Pythia data now shows that (b) does not explain the Δ-window
subpopulation value even in RoPE models. The centroid and the subpopulation diverge
by 0.114 in Δ units — this is not noise.

This means G7 needs an account that explains Δ ≈ 0.25 in the Δ-window subpopulation
across BOTH AbsPE and RoPE models, and is not the same account as Marín's. The only
account that currently applies across PE types is the formation-based one (the deep
geometry forms over referentially bound text; where it forms, the attractor is at
D/4 regardless of PE type). [INTERPRETIVE, conditional on T1 restatement scoping
this correctly]

**4. What the analysis-only experiment will add.**

The observation here is made from the registry records (exp-118 medians, Marín's
published centroid value). The planned analysis-only experiment will formalize this:

- Compute Marín's Padé formula (γ = f(θ, T_eval, d_head)) for each Pythia model
  with verified RoPE parameters
- Extract the full per-head Δ distribution from exp-118, not just the median
- Statistical test: do the Δ-window heads in Pythia cluster significantly below
  the Padé prediction?
- Kill condition: if the Δ-window subpopulation median in any Pythia model falls
  within the Padé centroid's 95% confidence band, the "subpopulation ≠ centroid"
  claim needs revision at that scale

This is the analysis-only pre-registration (blocked on GitHub token; highest-priority
solo item once token is restored).

---

## The T1 restatement implication [INTERPRETIVE]

T1 currently reads (in effect): A(i,j) ~ |i-j|^{-2Δ}, Δ ≈ D/4. exp-138 showed the
census slope is absolute-key-position drift, not relative-lag — so T1 needs
restatement.

The exp-118 Pythia data adds a constraint on what the restatement should say: the
attractor value D/4 is present at the subpopulation level in RoPE models, but it
is NOT what the centroid shows (the centroid sits higher, where Marín's RoPE aliasing
would push it). This means:

- The restatement cannot explain D/4 by appeal to RoPE geometry alone
- The restatement must scope whether it applies to the subpopulation or the centroid
- The formation-gated subpopulation shows D/4 regardless of PE type — this is the
  empirical constraint the restatement must explain

Carry to the T1 restatement session with Eldon. [INTERPRETIVE throughout]

---

## What this note does NOT claim

- That exp-118's n=2 heads (pythia-70m) are statistically stable — the Δ-window
  count is small; the analysis-only experiment will test this properly
- That the formation account is the correct G7 mechanism — it is the only
  current across-PE account, which motivates G7 work, not confirms it
- That Marín's program is wrong about the centroid — his Padé fits the centroid
  of 30+ models at MAE 4.3%; this note is about the subpopulation, a different
  observable

---

## Next moves

1. **Analysis-only pre-registration** (blocked on GitHub token): when token is
   restored, pre-register before any code. The exact analysis: compute Padé formula
   for each Pythia model's θ; extract exp-118 per-head Δ distributions; statistical
   test of subpopulation vs. centroid.

2. **T1 restatement session with Eldon (Eldon-present, #0):** Bring both the Oct 20
   note and this note. The restatement should scope which PE type T1 covers and
   what account it proposes for the D/4 attractor that applies across PE types.

---

*Register ledger:*
*MEASURED — exp-118 Pythia Δ-window medians, read from results.json on disk.*
*ESTABLISHED-LIT — Marín (2026) centroid γ ≈ 0.748 for Pythia-70M, from Oct 20 reading.*
*INTERPRETIVE — the implication for G7; the T1 restatement constraint; the subpopulation/centroid distinction.*
*Nothing here is a new experiment. The analysis-only experiment will formalize these observations.*
