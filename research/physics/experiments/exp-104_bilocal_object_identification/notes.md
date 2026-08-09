# exp-104 — Is the census's Δ the theory's Δ? Identifying the bilocal object

*Pre-registration: this file is committed BEFORE the measurement script runs.
Results are appended afterward.*

*Ariel — August 8, 2026, Saturday evening, Cursor session with Eldon.*

---

## Why this experiment exists

exp-103 (Aug 8, ~2 AM) was a methodology failure: it measured
∂A^(ℓ+1)/∂h^(ℓ), a cross-space map, when the SYK comparison requires the
self-map ∂G^(ℓ+1)/∂G^(ℓ). Its own "next step" section prescribed the fix:
perturb A^(ℓ) directly and measure the A^(ℓ+1) response.

**That prescription is also wrong, and finding out why is what produced this
experiment.** Reading the primary sources before implementing the fix (the
step-0 rule from `notes/2026-08-08_map_retirement_harvest.md`) established that
**G is not A.** Three independent primary sources agree:

- `notes/2026-08-03_melonic_threshold_derivation.md` eq. (2.1):
  H(1,2) = (σ_V²/d) Σ_{a,b} α_a(1) α_b(2) K_{ab}
- `LINEARIZED_SOFTMAX_CALCULATION.md` (Mar 9), same kernel, and explicitly:
  *"This factorizes into query-query × key-key correlations."*
- `SYK_ANALYSIS.md` (Mar 6), same object as X_{ab}, *"a bilocal functional of
  the attention weights."*

In matrix form, G^(ℓ) = w · A^(ℓ) K^(ℓ) A^(ℓ)ᵀ. **Its two free indices are both
query positions**, contracted over the key indices a,b. Equivalently, with value
vectors v_a = x_a W^V and layer output o_i = Σ_a α_{ia} v_a,

    E_{W^V}[⟨o_i, o_j⟩] = H(i,j),

so G is the **output–output correlation across query positions** — the internal
correlation structure D1 names.

The census measures something else. `replication/measure_conformal_heads.py`
takes the raw attention tensor `(n_heads, L, L)`, indexed (query, key), and fits
the decay of its diagonals:

    prof[:, dx] = np.diagonal(att, offset=-dx, ...)[:, k_lo:].mean(-1)
    fit A(dx) ~ dx^(-2Δ) by OLS over lags [8, 256]

**That is a query–key lag decay.** The spine's glossary bridges the two objects
with the phrase *"whose measured face is the lag profile A(i,j) ~ |i−j|^{−2Δ}"*
(`theory/interior_horizon_theory.md` L133–136; same language in Paper 6's
glossary). **I can find no derivation or measurement anywhere in the program
establishing that these two exponents are equal.** Searched: the spine, the
Aug 3 melonic note, the Mar 9 linearized-softmax derivation, `SYK_ANALYSIS.md`,
`conformal_integration_theory.md`, the census code. Not found.

This is not a claim that the census is wrong. It is the observation that a
proxy relation the whole program rests on has never been checked, and that
checking it costs a forward pass.

### Why it is not academic

Back-of-envelope only, K = I, ignoring row-stochasticity and boundary effects —
**recorded as motivation, explicitly not a result and not to be propagated:**
if A(u) ~ u^{−2Δ_A} on the causal strip, then
G(s) = Σ_u A(u+s) A(u) ~ s^{1−4Δ_A} when 4Δ_A < 1, and inherits A's exponent
only when 4Δ_A > 1. At the program's central value Δ_A = 1/4 the sum sits
exactly at the marginal point, where the query–query profile is nearly flat with
log corrections rather than at Δ = 1/4. So the two exponents plausibly diverge
precisely in the regime the headline claim occupies.

### What rides on the answer

- **P-A** (deep-layer Δ → 1/4 from above) is a statement about the fermion Δ,
  i.e. about G, checked in the melonic note §4.4 against census numbers, i.e.
  against A.
- **The 1/4 result itself**, and the GPT-2 precision-head figure (L6H4,
  Δ = 0.2499).
- **exp-103's mode templates**, which were built from Δ_h "from the power-law
  attention fit" — Δ_A. Fixing the Jacobian's space without fixing the exponent
  measures nothing.
- **Paper 6 v0.3**, currently awaiting Eldon's read, carries the same glossary
  language.

---

## The object being measured

For each layer ℓ and head h of a trained causal LM:

1. **A^(ℓ,h)** — the softmax attention matrix, (L × L), indexed (query, key).
   Exactly what the census uses.
2. **G_out^(ℓ,h)(i,j) := ⟨o_i^(ℓ,h), o_j^(ℓ,h)⟩** where o_i = Σ_a α_{ia} v_a and
   v_a = x_a W^V_h is the head's value vector (d_head = 64). This is the
   *empirical trained* bilocal — the ensemble average over W^V replaced by the
   actual trained W^V. **Primary measurement.**
3. **G_K^(ℓ,h)(i,j) := [A^(ℓ,h) K^(ℓ) A^(ℓ,h)ᵀ]_{ij}** with K^(ℓ)_{ab} = x_a·x_b
   on the layer input (pre-attention residual stream, post-LN as fed to the
   attention block). This is eq. (2.1) literally, with the σ_V²/d prefactor
   dropped (scale-invariant for an exponent fit). **Secondary — the theory's
   stated object, ensemble form.**
4. **G_cos** — G_out row/column normalized to cosine similarity. **Robustness
   check only**, since the theory's G is an unnormalized propagator.

Both G variants are symmetric and PSD by construction (Gram matrices), which is
itself a consistency check against axiom A4; A is row-stochastic and not PSD.

## Protocol

Frozen to match the published census wherever the two overlap, so exponents are
directly comparable:

- **Models:** `gpt2` (124M, learned PE — the showcase model, L6H4 Δ = 0.2499),
  then `EleutherAI/pythia-410m` (RoPE) as a second architecture if time allows.
  Both already cached locally.
- **Inputs:** 50 random-token sequences, length 512, seed 42, drawn from the
  full vocab — identical to `measure_conformal_heads.py`.
- **Precision:** fp32, `attn_implementation="eager"`, NaN-checked at extraction.
- **Lag profile:** the census's own `lag_profile` function applied unmodified to
  each of A, G_out, G_K, G_cos — diagonals at offset −dx, queries i ≥ max(256, dx).
- **Fit:** OLS on log-profile vs log-lag over lags [8, 256]; report Δ and R².
- **Aggregation:** per head, per layer. Report the paired difference
  Δ_G − Δ_A per head, and medians over (a) all heads, (b) the census's conformal
  subpopulation (R²_A ≥ 0.90, Δ_A ≥ 0.05), (c) the SYK-near subset
  (|Δ_A − 0.25| ≤ 0.05).

Averaging note: profiles are accumulated across the 50 inputs before fitting,
matching the census (which sums `mean_prof[l] += lag_profile(a)`), not fitted
per-input.

## Hypotheses (pre-registered)

**H1 — the proxy holds.** Over the conformal subpopulation,
median|Δ_G_out − Δ_A| ≤ 0.05, and Δ_G tracks Δ_A across heads with Spearman
ρ ≥ 0.7.

*If H1: the "measured face" language is justified, the census's Δ is the
theory's Δ to within the program's stated resolution, and this experiment
becomes the missing citation for a link that was previously assumed. exp-105
(the Jacobian) proceeds using measured Δ as before.*

**H2 — the exponents diverge (the registered alternative).** For heads with
Δ_A ∈ [0.20, 0.30], Δ_G_out < Δ_A − 0.05. Direction registered from the
back-of-envelope above: the query–query profile is *flatter* than the query–key
profile in this regime.

*If H2: the census measures a different exponent from the one the theory's
Δ = 1/4 refers to. The spine's glossary requires correction, P-A's data checks
require restatement in terms of which object, and every Δ-to-theory comparison
in the program — including Paper 6's — must name its object explicitly. This
would be the most consequential correction the program has made.*

**H3 — G is not power-law at all.** median R²_G_out < 0.5 over the heads where
R²_A ≥ 0.90.

*If H3: this is a finding about the theory rather than about the proxy. The
conformal ansatz is imposed on G (melonic note §3.3); if G's own two-point
profile is not power-law where A's is, the ansatz lacks its measured support and
that must be stated in the spine at the top, not one indirection away.*

**H4 — ensemble vs trained agreement.** |Δ_G_K − Δ_G_out| ≤ 0.05 in the
median. *If violated, the W^V-ensemble step in eq. (2.1) is not innocuous for
trained models and the theory's G and the measurable G are themselves two
objects.*

## Decision table

| Outcome | Verdict |
|---|---|
| H1 holds | **Proxy validated.** Add this experiment as the citation for the glossary bridge; no claim changes. |
| H2 holds (either sign, systematic) | **Proxy broken.** Spine glossary + Paper 6 glossary corrected; P-A and the 1/4 claim restated by object; exp-105 templates rebuilt from Δ_G. |
| H3 holds | **Conformal ansatz unsupported on G.** Report as a theory finding; spine T-chain status lines need the limit surfaced. |
| H1 and H4 hold | Strongest clean outcome — proxy validated *and* the ensemble step justified. |
| Δ_G ≈ Δ_A but R² low for both | Inconclusive on this protocol; report and redesign the estimator (see exp-052's Hanning lesson and exp-045's DFT bias). |

## Honest limits, named before running

1. **One model class per run.** GPT-2 is one architecture, one PE type, one
   scale. A single-model result does not generalize; it does, however, settle
   whether the question is live.
2. **Layer-input Gram ambiguity.** K^(ℓ) can reasonably mean the pre-LN residual
   stream, the post-LN input to the attention block, or the value-projected
   Gram. I use post-LN attention-block input for G_K and report the choice; the
   trained G_out avoids the ambiguity entirely, which is why it is primary.
3. **Unnormalized correlator.** ‖o_i‖ varies with position i, so G_out's
   diagonal is not constant and the lag profile mixes a genuine lag decay with a
   position-dependent amplitude. G_cos is the control for exactly this. If they
   disagree, the profile is amplitude-contaminated and the fit is not clean —
   report rather than pick.
4. **Random-token inputs.** The frozen census protocol uses random tokens, not
   natural text. That is the published protocol and comparability requires it,
   but G is a correlation of *outputs*, and random-token inputs may suppress
   query–query structure that natural text would produce. A natural-text arm is
   a follow-up, not a substitute.
5. **Ageev Eq. 20 transcription.** `SYK_ANALYSIS.md` writes the
   self-consistency map with α_{ia}(x_1) α_{ib}(x_2) — index i repeated and
   unsummed — where the other two sources write α_a(1) α_b(2). This changes
   whether the dressed map is A G Aᵀ or Aᵀ G A. **Not resolved here and not
   guessed at;** it does not affect this experiment (which measures G's profile,
   not the map), but it must be resolved against Ageev's paper before exp-105.
6. **This tests the identification, not the physics.** A negative result does
   not falsify D1, T3, or the SYK identification. It would mean the program's
   central *measurement* has been reported against the wrong object's name.

## Compute

Forward passes only, no training. 50 × 512 on GPT-2 (12 layers × 12 heads),
attention plus per-head value outputs retained. Local MPS. Estimated: a few
minutes. Second model similar.

---

*Pre-registration ends here. Results appended below after the run.*

---

## Results

**Run:** 2026-08-08, ~19:40–20:10 MDT, local MPS. GPT-2 (124M), 12 layers ×
12 heads = 144 heads, 50 random-token sequences of length 512, seed 42.
Estimator imported verbatim from `replication/measure_conformal_heads.py`;
protocol constants asserted equal at import. Result reproduced **byte-for-byte**
on a second run.

**Pre-registration commit:** `4bb825c`, before the measurement script existed.

### Status: H1 FALSIFIED · H2 MET · H4 FALSIFIED · H3 not met

| Subset | n | Δ_A | Δ_G_out | Δ_G_K | Δ_G_cos | median(Δ_G_out − Δ_A) |
|---|---:|---:|---:|---:|---:|---:|
| All heads | 144 | 0.6221 | 0.0158 | 0.1618 | 0.0136 | −0.5781 |
| Conformal (census criterion) | 20 | 0.4743 | 0.0262 | 0.1488 | 0.0206 | −0.4386 |
| **SYK-near (\|Δ_A−0.25\|≤0.05)** | **5** | **0.2683** | **0.0164** | **0.1067** | **0.0152** | **−0.2519** |

Median R² on the SYK-near subset: A 0.912, G_out 0.536, G_K 0.703, G_cos 0.643.

- **H1 (proxy holds; |ΔΔ| ≤ 0.05): FALSIFIED.** Observed |Δ_G_out − Δ_A| = 0.2519
  on the SYK-near heads — five times the registered threshold — and 0.4386 on
  the full conformal subpopulation.
- **H2 (registered alternative; Δ_G < Δ_A − 0.05 for Δ_A ∈ [0.20,0.30]): MET**,
  in the registered direction, at every head in the subset (IQR
  [−0.2587, −0.2205], so the effect is not driven by an outlier).
- **H4 (ensemble ≈ trained; |Δ_G_K − Δ_G_out| ≤ 0.05): FALSIFIED.** 0.1067 vs
  0.0164, a gap of 0.090. The W^V-ensemble step of eq. (2.1) is not innocuous
  for a trained model.
- **H3 (G not power-law; median R²_G_out < 0.5): NOT MET**, but only just —
  0.536 on the SYK-near subset. G_out's profile is neither a clean power law nor
  noise.

### The confound, and why the headline is narrower than the numbers look

**The pre-registered measurement is confounded by a term the theory itself
predicts, and the design should have caught it.** Melonic note eq. (2.2):

    E[H(1,2)] = w Σ_ab K_ab + w c₀ K₁₂ · Tr(K δK)

The first term is the **bare propagator G₀ — a constant in (1,2)**. A log-log
OLS on a profile with a nonzero floor is dragged toward Δ ≈ 0 by the floor no
matter what the connected part does. So **Δ_G_out ≈ 0.016 is very likely
measuring G₀, not a flat correlator**, and it must not be read as "the bilocal
has no power-law decay." That the pre-registration failed to specify floor
handling is a design miss, recorded rather than quietly repaired — this is the
second design-level miss in this experiment's lineage in two days (exp-103's was
the object; this one is the estimator).

### Post-hoc floor analysis — attempted, and NOT trustworthy

`posthoc_floor.py` (exploratory, not pre-registered) tried two removals: a
3-parameter fit prof(dx) = c + b·dx^(−2Δ), and a far-tail floor subtraction
followed by the census estimator. **Both failed to produce a usable fit** and
their numbers are recorded in `posthoc_floor_gpt2.json` as a negative
methodological result, not as measurements:

- The 3-parameter fit does not converge — `OptimizeWarning: Covariance of the
  parameters could not be estimated`, Δ pinned at exactly 0.0000 for most heads,
  and the constant-fraction diagnostic returning absurd values (10879 on the
  SYK-near A row), which means the fit found degenerate (c, b) pairs rather than
  a decay. The diagnostic itself is also miscomputed and would need rewriting.
- Far-tail floor subtraction gives median R² ≈ 0.51–0.56 on the subsets of
  interest. That is not a fit.

**So this experiment does not establish what G's exponent is.** It establishes
that nobody knows, and that finding out requires an estimator designed for a
correlator with a bare constant — which does not exist in the program.

### What is established, at register strength

**VERIFIED-AT-SOURCE (three independent primary documents, five months apart):**
The theory's bilocal G is a **query–query** object, G = w·A K Aᵀ, equal to the
output–output correlation ⟨o_i, o_j⟩ across query positions. The census fits
**A's query–key** lag decay. The spine's glossary bridges them in one
undemonstrated phrase — *"whose measured face is the lag profile"* — and no
derivation of that bridge exists anywhere in the program.

**MEASURED (pre-registered, reproducible):** Applying the program's own frozen
estimator to both objects on the same model, same inputs, same lags, gives
Δ_A = 0.2683 and Δ_G_out = 0.0164 on the SYK-near heads. Whatever the correct
estimator turns out to be, **these two numbers are not the same number**, which
is what the glossary asserts.

**NOT CLAIMED:** that Δ = 1/4 is wrong; that the SYK identification fails; that
the census is invalid. None of that follows. The bare-term confound means Δ_G is
unmeasured, not measured-and-different.

### Consequences

1. **P6 / exp-105 is blocked on an estimator, not on hooks.** The reparam-mode
   templates cannot be built until Δ_G is known, because exp-103 built them from
   Δ_A. Designing the floor-aware estimator for G is now the blocking item.
2. **The spine and Paper 6 need the bridge named as open.** Both carry the
   "measured face" language. It should say that the relation between A's and G's
   exponents is undetermined, with this experiment as the citation. Paper 6 v0.3
   is awaiting Eldon's read; this is a glossary-level correction, not a claim
   retraction.
3. **The melonic note §4.4 data checks need their object named.** P-A is about
   the fermion Δ (i.e. G); §4.4 checks it against census numbers (i.e. A).
4. **A cheap follow-up exists and is worth pre-registering:** G_K's exponent
   (0.1067) sits between Δ_A and Δ_G_out with the best R² of the three G
   variants (0.703). Whether that is signal or an artifact of K's own structure
   is a well-posed question.

### Honest note on how this experiment came about

The plan for the evening was exp-104 as exp-103 prescribed it: perturb A, measure
A's response. Applying the harvest note's step-0 rule to exp-103's own next-step
section — read the source before implementing — is what caught that G ≠ A. That
is the third consecutive instance of the source read returning more than it cost
(July 22 logos_bindings; August 7 reference pass; August 8 J-1), and the first
where it prevented an experiment rather than narrowing a claim.

*Files: `measure_bilocal.py` (pre-registered measurement),
`results_gpt2.json`, `profiles_gpt2.npz` (raw lag profiles, so any future
estimator can be tested without a re-run), `posthoc_floor.py` +
`posthoc_floor_gpt2.json` (exploratory, failed, kept).*
