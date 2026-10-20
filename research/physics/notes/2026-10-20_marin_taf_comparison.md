---
created: "2026-10-20"
time: "~12:45 AM MDT (2026-10-20T06:45Z)"
author: Ariel
surface: physics-room
status: reading note — external paper read at source; comparison written; no experiment registered
subject: >
  Marín (2026) "Predicting How Transformers Attend" — max-ent route to
  the attention power law, compared to the census. Object mismatch identified.
  GPT-2 as the counterexample to RoPE-specificity claim.
builds_on:
  - research/external/vanchurin.md (dossier template for external programs)
  - research/physics/OVERVIEW.md (current census standing)
  - development/status/rooms/physics/queue.md (exp-138 result: absolute-key-position)
  - research/2026-10-07_where_the_research_stands.md (T1 status)
registers: >
  [ESTABLISHED-LIT] Marín's results, read at source.
  [MEASURED] Our census results, with experiment pointers.
  [INTERPRETIVE] the comparison and the contact points.
  [THEORETICAL] the open questions this raises.
source: >
  Marín Muñoz, Carles. "Predicting How Transformers Attend: Analytic
  Power-Law Theory, Phase Transitions, and Practical Compression Tools."
  Zenodo 10.5281/zenodo.19826343, v2.0 (April 27, 2026).
  Erratum 2026-05-02: §5.2 Theorem 5.2 coefficient corrected from (log N)²/4
  to (log N)²/12 (Zenodo 10.5281/zenodo.19986818).
---

# Marín (2026) — Comparison with the census

## What the paper claims [established-lit]

**The central claim:** RoPE positional encoding imposes a log-distance constraint on
attention scores; the maximum-entropy distribution compatible with that constraint is
a power law A(d) ∝ d^{-γ} with closed-form exponent

    γ = (2θ − T_eval √2) / (2θ + T_eval √2)

(the [1,1] Padé approximant of exp(−z), where z = T_eval√2/θ).

**The observable:** mean attention weight A(d) — averaged over ALL heads, ALL layers,
and all prompts — as a function of token distance d. This is the *model centroid*
of attention decay.

**The derivation:** Three steps.
1. RoPE score decomposition: score(m,n) = Σ_i amplitude_i · cos(ω_i · d)
2. Average over content (mean-field on content, not position): yields ~n_active(d)
   RoPE frequency pairs contributing, where n_active(d) = (d_head/2)(1 − log_θ(d/2π))
3. Padé approximation to sum the active-dimension count → closed-form γ

**Validated on:** 30+ models, Pythia-70M to Qwen2.5-7B, median MAE 4.3% on
the centroid. Predicts centroid well; explains ~1% of cross-model variance at
fixed T_eval. The residual (99% of variance) decomposes into corpus signature,
GQA/SWA architecture, induction-head phase, training residuals (R²=0.44, n=23).

**Key theoretical elaborations:**
- §5 MaxEnt foundation: power-law form is the *unique* MaxEnt solution to E[log d] = λ
- §6 Phase diagram: Hagedorn transition at γ → 1⁻; Phase A (γ<1) divergent series,
  Phase B (γ>1) convergent; C_V diverges at boundary
- §11 (exploratory, self-labeled): RG view — γ as anomalous dimension;
  CFT identification: A(d) = d^{-γ} maps to 2-point function with conformal weight h = γ/2

**Explicitly acknowledged limitations:**
- L2: "Non-RoPE models require extension. AbsPE (GPT-2) and ALiBi require the
  Lerch phase~C framework. The current closed-form tools apply only to Phase A/B."
- L1: Derivation is approximate (mean-field in step 2; Padé approximation in step 3)
- RLHF results preliminary (n=3 measured pairs with divergent results)

---

## The object mismatch — stated precisely

**His observable vs ours:**

| | Marín γ | Our Δ_A |
|---|---|---|
| What is measured | Mean A(d) over ALL heads, ALL layers, all prompts — the model centroid | Ensemble-averaged lag-decay exponent of individual attention heads under frozen random-token protocol |
| Which heads | All heads (averaged) | Subpopulation: Δ-window heads (R²≥0.90, Δ∈[0.20,0.30]) |
| Value in GPT-2 | Not applicable — GPT-2 is AbsPE (his L2: "requires extension") | Δ_med ≈ 0.249, confirmed 44/144 heads (exp-007) |
| Value in Pythia | γ ≈ 0.674–0.997 (varies by θ, T_eval) | Δ_med ≈ 0.24–0.28 (exp-018 series) |
| Is it a universal attractor? | No — varies with RoPE parameters; not universal | Yes — fixed at D/4 across architecture families (measured, exp-007/118/120) |
| Formation-gated? | Not addressed (no formation ladder) | Yes — forms only over referentially bound text at story scale (exp-085/091/092) |

**Notation bridge:** In Marín's notation, A(d) ∝ d^{-γ}. In ours, A(i,j) ~ |i-j|^{-2Δ}.
So γ_Marín = 2Δ_Ariel. Our Δ ≈ 0.249 → γ_Marín ≈ 0.498.

For comparison: his Pythia-70M centroid γ ≈ 0.748. Our Δ-window subpopulation in
Pythia-70M would correspond to γ_Marín ≈ 0.50 (if it replicates at Δ ≈ 0.25).
These are different values. The centroid is pulled upward by the steep/local heads
(Δ >> 0.25) and downward by flat/global heads (Δ << 0.25). The two-populations
result (exp-109, confirmed: random-token-native and WikiText-native Δ-window
populations have Jaccard < 0.3) suggests the centroid aggregates populations whose
formation histories differ.

---

## The GPT-2 counterexample [measured + interpretive]

**The key falsification of Marín as a universal account:**

Marín's derivation is RoPE-specific. GPT-2 uses learned absolute positional encoding —
no RoPE. He explicitly acknowledges this (L2: "AbsPE (GPT-2) … requires extension").
His closed-form tools do not apply to GPT-2.

Our census finds Δ ≈ 0.249 (γ_Marín ≈ 0.498) in GPT-2 as a universal attractor value,
replicated across the formation ladder (exp-085/091/092) and the cross-family census
(exp-018/007). [MEASURED, exp-007]

This is the counterexample: the same attractor value Δ ≈ 0.25 appears in GPT-2 (no RoPE)
as in RoPE models. The mechanism cannot be RoPE aliasing geometry for GPT-2. Something
else sets the attractor at D/4, and that something applies to both PE types.

**Additional complication from exp-138 (September 9, 2026):** The census slope in GPT-2
is absolute-key-position drift (not relative-lag). K1+K2 fired on all 5 structural heads:
σ_abs-key = 1.04–1.17 × σ_full; σ_relative = −0.04 to −0.17 × σ_full (negative).
So the power law the census measures in GPT-2 is driven by *absolute key position*,
not by relative distance. Marín's framework derives a *relative-distance* power law
from RoPE. Different mechanisms, different PE types, and exp-138 suggests different
observables even when both appear as "power-law decay." [MEASURED, exp-138]

The result: for GPT-2, the census and Marín's centroid are measuring physically different things.

---

## Where they might touch — candidate contact points [interpretive]

**1. RoPE models — Pythia family.**
For Pythia models (RoPE), both programs measure a power law. The question is whether
the Δ-window subpopulation sits at a different value than the model centroid, and
whether the subpopulation's exponent is predicted by Marín's Padé or by the conformal
attractor D/4 (or both). This is registerable as analysis-only:
- Run the frozen-protocol census on Pythia (we have this data from exp-118)
- Compute the model centroid γ (mean over all heads)
- Compare to: (a) Marín's Padé prediction for these θ and T_eval, (b) D/4 = 0.25
- If the Δ-window heads are at D/4 while the centroid is at Marín's Padé, the two
  programs are measuring genuinely different things. If the Δ-window heads are at
  the centroid, they're consistent but still a subpopulation question.
→ Does not require new data; analysis-only from exp-118 arrays. Pre-register first.

**2. The CFT language — same vocabulary, different objects.**
His CFT identification: h = γ/2, c = 3 universally, critical exponents β = γ−1,
ν = 1/(1−γ), η = 2γ.
Our conformal claim: Δ ≈ D/4, formation-gated subpopulation.
In his notation: our Δ ≈ 0.25 → γ ≈ 0.50 → h ≈ 0.25. This would put our subpopulation
at Phase A (γ < 1) well away from the Hagedorn boundary.
Both programs describe the power law as CFT 2-point function language, but for
different objects. Neither verifies the other. The shared vocabulary does not
make the claims equivalent.

**3. Formation ladder × centroid — does Marín's γ track the ladder?**
Our formation ladder shows the deep geometry forms only over referentially bound text
at story scale (exp-085/091/092). For RoPE models trained on the ladder rungs, his
γ_centroid would reflect all heads — including the steep/local heads that are NOT
formation-gated. We'd expect his centroid to be relatively stable across formation
rungs (because most heads don't form the deep geometry), while our Δ-window count
changes dramatically. If confirmed, this would be the cleanest demonstration that
his centroid and our subpopulation are genuinely orthogonal observables.
→ This would be a natural prediction for P7(e) (the architecture-decomposition arm
  of P7 on the formation ladder).

**4. T1 restatement — Marín sharpens the question.**
T1 as A(i,j) ~ |i-j|^{-2Δ} is not what the census measures (exp-138). T1 needs
restatement. Marín's work sharpens what that restatement needs to say for *RoPE models*:
- For GPT-2 (AbsPE): the census slope is absolute-key-position drift (exp-138); Marín
  doesn't apply; the attractor value D/4 needs a non-RoPE explanation (G7 open)
- For RoPE models: there is now a rival account (Marín) for the power-law form (RoPE
  aliasing + MaxEnt), distinct from the conformal attractor account. Whether the
  subpopulation's exponent is explained by Marín's Padé or by formation is an
  open question. T1's restatement should explicitly scope which PE type it covers.

The two deflationary rivals for the conformal reading are now named:
  (a) LayerNorm + causal masking → absolute-key-position bias (Findings ACL 2026;
      applies to AbsPE/GPT-2; exp-167 was supposed to test this before Modal kill)
  (b) RoPE aliasing → relative-distance power law (Marín; applies to RoPE models)
These are different mechanisms for different PE types. Neither one explains GPT-2
and RoPE models with the same account. The formation-based account (D/4 as universal
attractor, formation-gated) would be more general if it explains *both*.

---

## Comparison in method register [established-lit / measured]

| | Marín | This program |
|---|---|---|
| Pre-registration | Not mentioned | Required; git-attested (exp-113–172 run) |
| Kill conditions | Explicitly published (Failure Atlas §10, 12 negative results) | Required; published at same prominence as confirmations |
| Scope limits | Acknowledged (L1–L6, Failure Atlas) | Acknowledged (one architecture class, formation-scale, structural-consequence) |
| Formation account | None — describes inference-time behavior only | Central — formation ladder (exp-085/091/092/144) |
| Subpopulation | Detected as circuit anomalies (KL divergence heads) but not measured per formation | Directly measured as Δ-window subpopulation |
| Erratum record | Published (Theorem 5.2 coefficient corrected 2026-05-02) | Published (entropy-gap erratum; our program's immune memory) |

**Method note:** He publishes a Failure Atlas (§10, 12 negative results), which is honest
practice — it serves the same role as our standing constraints and kill conditions. We
should cite this as a positive feature when discussing the program publicly.

---

## RELATIONS question

**Relevant surface:** `research/physics/RELATIONS.md` (contact landscape) — decision
with Eldon per task description.

**Case for contact:**
- Complementarity is real: he has the centroid measurement across 30+ models; we
  have the subpopulation, the formation ladder, and GPT-2 (outside his framework)
- The Padé-vs-D/4 question for Pythia heads is registerable and could be framed
  as a joint question
- He is clearly working carefully (erratum, Failure Atlas, explicit limitations)

**Case against (or for waiting):**
- T1 restatement is not settled — our program's conformal claim is "open pending
  restatement" as of Oct 7 (OVERVIEW.md correction). Reaching out before settling
  what we claim about RoPE models is premature.
- The object mismatch is fundamental: his tool produces an architecture-diagnostic
  centroid; ours produces a formation-sensitive subpopulation. Contact is most
  productive after we can say precisely what the Δ-window subpopulation exponent is
  in Pythia heads, relative to his Padé prediction.

**Recommendation:** Hold for the T1 restatement session with Eldon. Bring this note.
Add to agenda: (a) does his Padé prediction for Pythia match the Δ-window subpopulation
exponent, or are they at different values? (b) after the analysis-only Pythia comparison,
assess whether contact makes sense.

---

## What this note does NOT claim

- That the census and Marín's centroid are measuring the same thing — they are not
- That Marín's derivation explains the attractor Δ ≈ D/4 in GPT-2 — it explicitly doesn't
- That the CFT language links the two programs — same vocabulary, different objects,
  neither verifies the other

---

## Next moves

1. **Analysis-only pre-registration (blocked on GitHub token):** Compare Pythia Δ-window
   subpopulation exponent to Marín's Padé prediction for those models using exp-118 data.
   Pre-register before any code; this is the one concrete empirical question Marín raises
   for existing data.

2. **T1 restatement session with Eldon (Eldon-present, #0 priority):** Bring this note.
   The deflationary rivals are now: (a) LN+causal masking for AbsPE, (b) Marín for RoPE.
   T1's restatement should scope which PE type it covers and what account it proposes for
   each.

3. **RELATIONS decision (Eldon):** After the Pythia analysis-only is registered. Not
   before the T1 restatement is settled.

---

*Register ledger:*
*ESTABLISHED-LIT — Marín (2026), all claims attributed with section.*
*MEASURED — exp-007 (GPT-2 Δ_med), exp-118 (cross-family census), exp-138 (absolute-key-position),*
*exp-085/091/092 (formation ladder), exp-109 (two-populations).*
*INTERPRETIVE — the object-mismatch argument; the GPT-2 counterexample framing; the*
*contact-point proposals; the T1 restatement implications.*
*THEORETICAL — the open questions about what sets D/4 in AbsPE models (G7 open).*
