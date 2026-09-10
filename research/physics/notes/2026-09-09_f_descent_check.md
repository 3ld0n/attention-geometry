---
created: "2026-09-09"
status: Theory computation, G1-class — closed with a result. No trained model, no instrument, no pre-registration (this is a check on a theoretical condition, not a measurement on a system; the condition and the computation were named in the draft before the run).
author: Ariel (Cursor, ~4:50–5:20 PM, September 9; Eldon asked "could you check the across-step half now?")
subject: Is the attending closure a descent on the horizon's variational free energy F? (attending_system_theory_draft.md A5, §7.1 ii–iii)
code: theory/f_descent_check.py (main grid), theory/f_descent_mask_control.py (control); outputs in theory/logs/
registers: [COMPUTED] — established by direct numerical computation on random systems, with stated scope; [PROPOSED]; [OPEN]
---

# Is the closure a descent on F?

## 0. The question and why it was asked

At ~4:45 PM Eldon asked whether the free-energy principle could supply A5's
c-function. The draft was amended: F = Σ_i [−Σ_a A_ia s_ia − β⁻¹ H(A_i·)]
is the natural candidate; its within-step half is exact (Gibbs variational
principle); the across-step half — *the closure (σ, A) → (σ′, A′) does not
increase F* — is the free-energy principle's contested claim and enters as
a condition to be checked. If it held, A5 would have its monotone and T1
would follow by the Jaynes route, so T2 (Born rule) would be derived rather
than imported. He asked me to check it. This note is the check.

With the exponential normalization, F = −β⁻¹ Σ_i log Σ_a exp(β s_ia): minus
the summed log-partition function. "Descent on F" = "the summed log-partition
function does not decrease along the closure."

## 1. The computation

Random bilinear attending systems, A6 intact (A recomputed from the state
every step). N = 64 loci, d = 32, η = 0.1, T = 300 steps, 12 seeds,
β ∈ {0.5, 1, 2, 4, 8}.

- Score s_ia = σ_iᵀ M σ_a / √d, three forms: **sym_psd** M = WᵀW
  (reciprocal, positive); **sym_ind** M = (B + Bᵀ)/2 (reciprocal,
  indefinite); **asym** M = W_QᵀW_K (two independent reads — the q = 4
  structure of T3).
- Routing A = softmax_a(β s_ia), optionally under a causal mask.
- Content v(σ_a) = W_V σ_a, three forms: **identity** W_V = I; **random**
  W_V independent of M; **aligned** W_V = M/√d (content read = score read).
- Update σ_i ← σ_i + η Σ_a A_ia v(σ_a), with and without renormalizing each
  state to unit norm.

Recorded per variant: fraction of steps with ΔF ≤ 0; fraction of seeds with
F monotone over the whole run; F first → last; the same for the mean row
entropy H (the entropy-only candidate); mean state norm.

**The norm confound, controlled.** Without renormalization the state norms
grow by 10⁶–10¹² over 300 steps, the scores scale up, and the log-partition
function rises trivially — F "descends" in nearly every unnormalized
variant, including ones where the normalized run is a coin flip. The
unnormalized rows are uninformative and are reported only to show the
confound. Everything below is the renormalized case.

## 2. Results (renormalized; fraction of steps with ΔF ≤ 0 / fraction of seeds fully monotone, averaged over β where flat)

| Score | Content | Mask | Steps ΔF ≤ 0 | Seeds monotone | F first → last (β = 2) |
|---|---|---|---|---|---|
| sym_psd | identity | none | **1.000** | **1.00** | −133.4 → −144.8 |
| sym_psd | identity | causal | 0.70 | 0.25–0.42 | −103.7 → −115.2 |
| sym_psd | random | none | 0.58 | 0.00 | −133.4 → −144.1 |
| sym_psd | aligned | none | **1.000** | **1.00** | −133.4 → −172.6 |
| sym_psd | aligned | causal | 0.97–1.00 | 0.92–1.00 | −103.7 → −141.8 |
| sym_ind | identity | none | 0.88 | 0.58 | −133.1 → −133.7 |
| asym | identity | none | 0.82 | 0.33–0.42 | −133.1 → −132.9 (flat) |
| asym | identity | causal | 0.39 | **0.00** | −102.7 → −102.5 (rises) |
| **asym** | **random** | none | **0.50** | **0.00** | −133.1 → −132.4 (flat) |
| asym | random | causal | 0.46 | 0.00 | −102.6 → −102.2 (rises) |
| asym | aligned | none | 0.975 | 0.75–0.83 | −133.1 → −140.7 |
| asym | aligned | causal | 0.93 | 0.58–0.75 | −102.7 → −111.5 |

The entropy-only candidate H (mean row entropy) is monotone in **0.00** of
seeds in every renormalized variant. It is dead as a c-function.

**Control (mask type, β = 2):** with identity content on a symmetric PSD
score, *any* masking breaks exact monotonicity — undirected 50% sparsity
0.89 / 0.33, directed 50% sparsity 0.94 / 0.50, causal 0.70 / 0.42. With
aligned content, undirected and directed sparsity are both 1.000 / 1.00 and
causal is 0.97 / 0.92. So the variable that governs descent is not the
mask's directedness. It is whether the content read coincides with the
score read.

## 3. What the result says

**The closure is a descent on F when, and to the degree that, the content
channel reads the state through the same form as the score channel.** With
v = Mσ/√d the update is the row-half of −∇F (∂F/∂σ_i has a row term
−Σ_a A_ia Mσ_a/√d, which is exactly the aligned update, and a column term
−Σ_j A_ji Mᵀσ_j/√d from σ_i's appearance as a source in other rows, which
the attending update omits). The half-gradient descends F robustly, even
under directed masks. With an independent content read — a random W_V —
descent is a **coin flip (0.50)** regardless of the score's symmetry, and F
is flat over 300 steps. The genuinely two-read attending system does not
descend F at all.

**The consequence for the theory is sharper than "the FEP fails."** Clause
(i) of the definition (§2.1) says an attending system has *two channels
read from the same state by different maps*. The condition under which F is
a Lyapunov function is that the two maps coincide. So: **the free-energy
principle, in its variational-F form, holds for an attending system exactly
in the limit where it stops being one in the sense of clause (i)** — where
routing and content collapse to a single read and the system becomes an
energy-based (Hopfield-type) network. The independence of the reads, which
the definition requires, is what breaks the Lyapunov function.

This is consistent with, and restates in the theory's terms, what the
energy-based-attention literature already knows: the modern Hopfield update
is attention with the key and value reads tied to the query read, and its
energy is −β⁻¹ logsumexp plus a norm term (Ramsauer et al. 2020); the
Energy Transformer obtains a Lyapunov function by *enforcing* symmetric
tied weights (Hoover et al. 2023); no energy function is known for
attention with independent Q, K, V. What is new here is what that boundary
*means* for a theory of the observer: it is the boundary of clause (i).

**Secondary findings.**
- The sym_psd / identity / full-attention case — the modern-Hopfield case —
  is exactly monotone (1.000 / 1.00 at every β). The computation recovers
  the known result where it should.
- Masking degrades descent for identity content even when undirected; so
  the earlier draft's worry that *causality* specifically conflicts with
  F-descent is not supported — sparsity of any kind does, and alignment
  repairs it. The T6 tension I anticipated is not there in this form.
- The unnormalized runs are a standing warning for any application: on a
  system whose state norm is not controlled, "F descends" is not evidence
  of anything.

## 4. What this does to the draft

- **A5.** F is *not* available as the c-function of an attending system
  under clause (i). It is the c-function of the single-read (energy-based)
  limit. The paragraph proposing it is amended to say so, with the
  computation cited. The row-entropy candidate is removed (dead).
- **§7.1 (ii)–(iii).** The two blocked tests are no longer unblocked
  together. T1's Jaynes route via F is available in the single-read limit
  only; for a genuine attending system the monotone is OPEN again, and one
  candidate has been eliminated with a reason.
- **§7.3 (v), the degenerate-case reduction.** Strengthened: the
  single-read limit is now characterized by a property (it has a Lyapunov
  function; the two-read system does not), which is a sharper reduction
  statement than "clauses (ii) and (iii) switched off."
- **§8, FEP entry.** The relation is now a result: the FEP holds for
  attending systems in the limit where routing and content are one read,
  and fails when they are independent. This is a statement *about* the FEP
  and its scope, computed on the theory's own objects.
- **Direction, not built on [PROPOSED].** If a monotone exists for the
  two-read system, it must couple the two reads. The natural place to look
  is a free energy whose energy term is the content-channel inner product
  routed by the score channel, −Σ_i Σ_a A_ia ⟨σ_i, v(σ_a)⟩_M′, for some
  metric M′ fixed by P0; or the full gradient including the column term,
  which would mean the attending update is only half of a descent and the
  "being attended" half is what a physical attending system supplies that
  a feed-forward one does not. Neither is tried here.

## 5. Scope

Random Gaussian systems at one size (N = 64, d = 32), one step size, 300
steps, 12 seeds, five temperatures. Bilinear score only; additive update
only; exponential normalization only. The result is a statement about
which conditions produce descent in this family; it is not a theorem. The
asym/random coin-flip is robust across β and mask (0.46–0.52 in every
cell), and the aligned-content recovery is robust (0.93–1.00 in every
cell). A proof of the aligned case (half-gradient descent for a
convex-in-scores F) looks straightforward and is not written. A proof that
*no* Lyapunov function exists for the two-read case is not attempted and
would be the real theorem.

*Register ledger: [COMPUTED] — every number in §2, at the scope of §5.
[PROPOSED] — the reading in §3 that the boundary of F-descent is the
boundary of clause (i); the two directions in §4. [OPEN] — the c-function
for a two-read attending system; T1's derivation outside the single-read
limit; a nonexistence proof.*
