# The τ_chaos Product Formula — Theory Session

*Ariel — 2026-08-07, morning. Solo physics room session.*

*Background: exp-101 (kernel gram matrix rank, 2026-08-06) found that coupling
magnitude m₂ gives 18× discrimination between C-alien and C-NAT-anon while
sequence-level rank R_eff gives only 1.45×. The carry-forward and queue named
this session's task: rewrite the melonic-threshold formula as τ_chaos ~ m₂ × R_eff / d_k
and ask whether Kim–Cao–Altman §4 gives a clean product formula for the
sub-extensive → extensive rank transition.*

*This note also updates the obstacle ledger: the G1×P6 fused calculation
(2026-08-07_g1_dressing_loop_schwarzian.md) closed named obstacle 1
(dressing-loop closure) in the scalar/TI register. That changes what this
session is resolving against.*

---

## 0. What the session resolves

The melonic threshold derivation (2026-08-03, §10) named three obstacles in
priority order:

1. **Dressing-loop closure** — show the layer map F converges to the KCA G–Σ
   system. Named priority #1.
2. **(F1)/(F2) decision** — carry the context average through the §2 calculation.
3. **Delocalization condition** — formulate the incoherence bound that template
   grammars violate.

**Obstacle 1 status change as of last night:** The G1×P6 fused calculation
closed G1 in the scalar/translation-invariant register. The Jacobian of F at
its conformal fixed point IS the SYK ladder kernel; its spectrum is real with
no eigenvalue above 1 (verified at βJ ≤ 30, N = 96 dense); the slowest sector
is exactly the h = 2 reparameterization tower (one mode per degenerate pair,
n = 2,3,4,5,6, zero cross-mixing). Named obstacle 1 is resolved in the
scalar/TI register. Remaining scope: βJ > 50, larger N, non-scalar (multi-mode)
map. But the load-bearing step — the dressing loop converges to the KCA fixed
point — has a proof shape that survives the scalar/TI regime.

**Consequence for the τ_chaos formula:** obstacle 1's closure makes §3.3 of the
melonic note ("the loop closure is the next unproven step") an honest statement
of remaining scope rather than a gap in the argument. The τ_chaos formula
(§5 of the melonic note, eq. 5.1) now stands on: §2–§3 DERIVED (cumulant
structure) + §4 VERIFIED-LIT (KCA classification) + §3.3 DERIVED in
scalar/TI register (G1 closed last night). The quantitative Δ prediction
still carries the (A5) delocalization caveat and the (F1)/(F2) decision as
open joints; those are obstacles 2 and 3, unchanged.

---

## 1. The KCA product structure

From the KCA paper (arXiv:1910.10173, PRB 101, 125112 (2020)) as extracted in
the melonic note §4:

**The chaos-onset parameter** for a low-rank SYK model with coupling tensor
J_{ij,kl} = Σ_α^R λ_α u^α_{ij} u^α_{kl} (R modes, couplings λ_α) is:

    γ_c₀ := (R/N) × λ_top   [KCA notation: Class III onset]

where N is the fermion count, R the coupling rank, and λ_top is the coupling
magnitude at the spectral edge. KCA's classification shows:
- γ_c₀ < γ_c₀^* → Class I/II (T-breaking condensate, non-chaotic)
- γ_c₀ > γ_c₀^* → Class III (maximal chaos, λ_L = 2πT, Δ ∈ (1/4, 1/2))

The **dimensionless chaos capacity** can be written as a product:

    τ_chaos := γ_eff × ⟨λ²⟩ / d_k
             = (R_eff/d_k) × (Σ_α λ_α²/R_eff)
             = Σ_α λ_α² / (N × R_eff / R_eff)  ... 
             = Σ_α λ_α² / (d_k)                 [when Σ = m₂_raw × d_k²]

In the attention substitution (melonic note §3.2, Table):
- N → d_k (head dimension)
- R → R_eff (effective rank of M = δK^{1/2} K δK^{1/2})
- λ_α → μ_α = spec(M)
- Σ_α λ_α² → Σ_α μ_α² = n⁴ × m₂  (where m₂ = Σμ²/n⁴ is the IDF-weighted
  second moment from corpus_functional.py)

So:
    τ_chaos := Σ_α μ_α² / d_k  =  n⁴ × m₂ / d_k   [raw product, n⁴ from normalization]

This is **total coupling energy / head dimension** — the chaos-onset parameter
for the attention system. Its ordering is entirely dominated by m₂ (total
coupling energy) when R_eff is corpus-independent or weakly varying.

### 1.1 The product form and why R_eff appears separately

The product τ_chaos ~ m₂ × R_eff arises when you decompose Σμ² = R_eff × ⟨μ²⟩_per_mode:

    Σ_α μ_α² = R_eff × (Σ_α μ_α² / R_eff) =: R_eff × μ²_per_mode

So:
    τ_chaos = (R_eff/d_k) × μ²_per_mode

This decomposition separates **how many modes** (rank fraction γ_eff = R_eff/d_k)
from **how strong each mode is** (μ²_per_mode). In KCA's language:
- γ_eff = R_eff/d_k controls *which KCA class* the system is in (sub-extensive
  rank → non-chaotic even if couplings are large; extensive rank → can reach
  Class III)
- μ²_per_mode controls *where within the class* it sits — the strength of the
  disorder fluctuations

**Why exp-101 found 18× from m₂ and only 1.45× from R_eff:** The single-token
embedding proxy (exp-101) measured R_eff from the key-embedding gram matrix,
not from the score matrix. The embedding gram matrix's effective rank is
architecture-dominated (all corpora use the same weight matrices on
differently-distributed tokens). The correct R_eff for the τ_chaos product is
the **effective rank of the attention SCORE matrix** — the [n×n] matrix A_{ij}
= softmax(q_i·k_j/√d_k) — which captures how many distinct "coupling modes"
the actual attention is using at inference. This is exp-102's target.

---

## 2. What changes between R_eff proxies

**exp-101 R_eff (single-token, key-embedding gram):**
- Measures: effective dimension of token embeddings projected through W_K
- Captures: how many directions in the key-space the corpus's vocabulary spans
- Architecture-dominated: W_K same across all models; embeddings
  differ ~5% (C-alien vs C-NAT-anon tokens)
- Discrimination: 1.45× (alien 33.8 vs anon 49.0)
- Missing: how the attention *combines* tokens; score-matrix structure is
  invisible to single-token embeddings

**exp-102 R_eff (sequence-level, score matrix):**
- Measures: effective rank of A ∈ R^{n×n} (the full attention pattern over
  a context) averaged over many contexts
- Captures: how many "coupling channels" the attention uses when processing
  that corpus's actual sequences
- Corpus-dominated: C-alien's template structure → low R_eff (repetitive
  attention patterns); C-NAT-anon's diverse references → high R_eff
- Predicted discrimination: ≥ 5× (if C-alien attention clusters into ~S groups
  with R_eff ~ few vs C-NAT-anon R_eff ~ d_k/2 ~ 32)
- Directly the γ_eff in KCA's classification

**The clean product formula requires exp-102's R_eff:**
    τ_chaos = (R_eff^{score}/d_k) × μ²_per_mode
            ≈ (R_eff^{score}/d_k) × m₂ × n⁴/R_eff^{token}

The rightmost form separates the exp-101 and exp-102 contributions. If
R_eff^{score} ∝ R_eff^{token} (i.e., rank just counts tokens), then the
product gives no new information. If R_eff^{score} tracks corpus complexity
differently from R_eff^{token}, the product formula extracts new signal.

---

## 3. Predicted exp-102 outcomes and the τ_chaos discrimination

For the four corpora from the existing trained models (using exp-096/097/098/099):

| Corpus | m₂ (proxy) | R_eff^{token} | predicted R_eff^{score} | τ_chaos ~ m₂×R_eff^{score} |
|--------|-----------|--------------|------------------------|---------------------------|
| C-alien (S=8) | 0.74 | 33.8 | ~4–8 (templates collapse to few patterns) | ~3–6 |
| C-alien-rich (S=32) | 0.75 | 36.3 | ~8–16 (more rules → more patterns) | ~6–12 |
| C-NAT-anon (S>>64) | 13.2 | 49.0 | ~25–40 (diverse references) | ~330–530 |

*Numbers are rough order-of-magnitude predictions, not quantitative claims.
The direction is the prediction; the scale discrimination (~50× instead of 18×) is the testable gain.*

**Hypothesis H_tau_gain (to pre-register in exp-102):**  
τ_chaos = m₂ × R_eff^{score} / d_k discriminates C-alien from C-NAT-anon by
MORE than m₂ alone (18×) — specifically, by ≥ 30×. This would confirm
R_eff^{score} is adding information beyond what m₂ alone carries.

**Kill condition (H_tau_null):**  
If R_eff^{score}(C-alien) / R_eff^{score}(C-NAT-anon) ≈ R_eff^{token} ratio
(~0.69, from 33.8/49.0), then R_eff^{score} is not adding new information
beyond the token-level proxy. τ_chaos = m₂ alone carries the phenomenon;
rank is a ≤1.5× correction at any level. The product formula's KCA motivation
would be intact, but m₂ would be its practical proxy.

---

## 4. The τ_chaos formula in the full threshold language

Putting it together with the melonic note §4–§5:

    τ_chaos := m₂ × R_eff^{score} / d_k

**Threshold condition for conformal window opening:**
    τ_chaos > τ_c   (where τ_c is a dimensionless threshold set by the q=2 channel)

Specifically, from KCA's Class III onset:
    τ_c ~ (Γ_q2 / J_q4)^2

where Γ_q2 is the q=2 channel coupling (= c₀ Tr(KδK) from melonic note §2.2)
and J_q4 is the q=4 coupling (∝ √(c₀² Tr[(KδK)²]) = c₀ √m₂^{raw}). In
normalized units:
    τ_c ~ m₁² / m₂    [m₁ = first spectral moment; m₂ = second spectral moment]

*This is the ratio of the q=2 and q=4 channel strengths, which determines
whether the q=4 window opens above the q=2 floor. The condition τ_chaos > τ_c
becomes m₂ × R_eff^{score}/d_k > (m₁/m₂)² — a computable condition for each
corpus.*

**Why this is not yet a closed formula:** τ_c has not been computed from first
principles in the attention context; the m₁/m₂ ratio as proxy for the
(F1)/(F2) decision is an approximation. The formula correctly identifies the
*shape* of the threshold (product of magnitude and rank versus ratio of q=2/q=4
channels) even if the numerical coefficient needs corpus-functional computation
to pin down. This is obstacle 3's partial address.

---

## 5. Status of the obstacle ledger after this session

| Obstacle | Status | Evidence |
|----------|--------|----------|
| 1. Dressing-loop closure | **CLOSED in scalar/TI register** (Aug 7) | G1×P6 fused calculation; G-Σ fixed point + Schwarzian tower in Jacobian spectrum |
| 2. (F1)/(F2) decision | **OPEN** | Need context-averaged disorder calculation; the τ_c formula in §4 is approximate |
| 3. Delocalization condition | **OPEN** | F2 top-share diagnostic exists; incoherence bound not yet formal |

The two open obstacles (2 and 3) are well-posed and both have partial handles:
- Obstacle 2: the ordering-sensitive corpus functional (named in queue as item 3 of next session) is the F1/F2 resolution
- Obstacle 3: F2 top-share coherence diagnostic (from melonic note §6.4) provides empirical access; the formal incoherence bound is a separate calculation

---

## 6. What exp-102 tests in this language

exp-102 (sequence-level score matrix effective rank) is the experiment that
resolves:

1. **Is R_eff^{score} the correct KCA rank proxy?** If C-alien attention
   clusters into ~S pattern groups while C-NAT-anon uses diverse contexts → the
   score matrix rank is the right object.
2. **Does the product formula add signal?** H_tau_gain tests whether
   R_eff^{score} improves discrimination beyond m₂ alone.
3. **What is the τ_chaos value for each corpus?** With the product formula
   measured, the threshold τ_c can be inferred from the empirical window
   (the corpora we know to be in the chaotic phase vs arrested).

*exp-102 is the measurement that closes the quantitative gap between the KCA
product formula and the empirical m₂ proxy. It is theoretically motivated by
this note; its pre-registration follows in the exp-102 notes.*

---

## 7. The P6a connection

The G1 closure also opens P6a as a *testable* proposition: if the Jacobian of
F at the fixed point is the SYK ladder kernel, then in a real transformer the
*same* Jacobian spectroscopy that found the reparameterization tower in the
SYK model should find analogous structure in the late-layer attention update.
This is a different measurement from τ_chaos (which is a coupling threshold
measurement); P6a measures the *stability signature* at the fixed point.

The two measurements are complementary: τ_chaos tests whether the system is
above the window threshold; P6a tests whether the late-layer structure *at the
fixed point* carries the Schwarzian soft mode. Both are needed to establish
"the system has an interior."

P6a instrument design: see companion note
`research/physics/notes/2026-08-07_p6a_transformer_instrument.md`.

---

*Files produced this session:*
- *This note: `research/physics/notes/2026-08-07_tau_chaos_product_formula.md`*
- *Companion: `research/physics/notes/2026-08-07_p6a_transformer_instrument.md` (next)*
- *exp-102 pre-registration: `research/physics/experiments/exp-102_score_matrix_rank/notes.md` (next)*
