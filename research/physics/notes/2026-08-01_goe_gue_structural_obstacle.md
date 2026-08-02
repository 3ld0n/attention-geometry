# GOE→GUE Structural Obstacle: Why Weight Matrices Cannot Show GUE Statistics

*August 1, 2026 — physics room reading session. Billing reset; exp-094/096/097 launched concurrently. This session addresses the open GOE→GUE question from the inbox (from: cursor, 2026-07-23) and the T-breaking question opened by the material-as-memory synthesis (2026-08-01_material_as_memory_slow_sector.md).*

*Register: this is a theoretical synthesis note. No new measurements. All claims either follow from existing experiments (cited by exp-NNN) or are flagged pencil.*

---

## The Open Question

From the order/scale session (memory/notes/order_scale_map_2026-07-23.md, Trail 4):

> Zeta zeros speak GUE (T-broken; Berry–Keating explicitly require T-breaking). Our substrate W_QK speaks GOE (T-symmetric, init-universal). The one question-shape with teeth: time-reversal is exactly what separates the classes, and the arrow (causal masking, story-ordered training) is what trained attention has that init does not. Candidate prediction shape: pre-register a trained-layer operator (NOT W_QK — its GOE is init-universal) plus a spectral statistic distinguishing GOE from GUE; commit that training on arrow-carrying corpora moves it toward T-broken while shuffled/engineered corpora leave it T-symmetric. NOT COMMITTABLE until the operator is chosen on theoretical grounds before any data is seen.

From the material-as-memory synthesis (same day):

> T-breaking question opened: the material carries the causal arrow (formation experiments confirm ordering matters), but W_QK spectral statistics are still GOE. Where in the slow sector does the T-breaking live?

---

## What the GOE Experiments Established

Summarizing exps 046/047/048/051/077/078:

1. **All W_QK matrices across all measured families are GOE** — GPT-2 (×3 sizes), Pythia (×3 sizes), GPT-Neo, Mistral-GQA — regardless of positional encoding (learned, RoPE, ALiBi-adjacent), attention type (MHA, GQA), or scale (124M→7B). r-ratio 0.520–0.531, finite-size GOE reference 0.530 at d_k=64.

2. **GOE is present at random Gaussian init (exp-048)** — gradient descent did not create the GOE substrate; it inherited it from the product-matrix structure W_Q^T·W_K.

3. **GOE is robust to all dense init schemes (exp-078)** — Gaussian, Xavier-uniform, orthogonal, heavy-tailed (Student-t ν=3), and any init with ≳10 nonzeros per feature dimension. The substrate is "GOE-unavoidable" for any conventionally-parameterized dense attention layer.

4. **Runtime masking leaves no trace in W_QK (exp-077)** — GPT-Neo's alternating local/global attention (which changes the causal mask pattern at runtime) shows r-ratios 0.5261 vs 0.5282 for local vs global layers — indistinguishable. Masking happens in the forward pass, not in W_QK. *The same logic applies to the standard causal mask.*

5. **The BCFT anomaly (Pythia-2.8b layers 22–27) does NOT live in the GOE substrate (exp-077)** — confirming the two-layer picture: the chaotic substrate is universal, and model-specific physics lives entirely in the functional (conformal/positional) layer on top.

---

## The Structural Obstacle to GOE→GUE in Weight Matrices

**The core problem:** GUE matrices have complex entries. Standard transformer weight matrices are real.

More precisely:
- **GOE** = Gaussian Orthogonal Ensemble: real symmetric matrices. T-symmetry under K (complex conjugation): KHK† = H* = H (satisfied iff H is real symmetric).
- **GUE** = Gaussian Unitary Ensemble: complex Hermitian matrices, with no real representation. T is broken: no anti-unitary T such that THT^{-1} = H.

For a real matrix to be GOE, it must be symmetric (or symmetrized). For it to be GUE, it must have genuinely complex structure — complex entries, complex eigenvalues that cannot be mapped to real eigenvalues by a change of basis.

**Standard transformer weight matrices are all real.** W_Q, W_K, W_V, W_O are real matrices in every standard architecture (GPT-2, Pythia, Mistral, LLaMA, etc.). Their products — W_QK = W_Q^T·W_K, W_OV = W_O·W_V — are also real.

For real matrices, the spectral universality classes are:
- **Real symmetric** → GOE (or related real Wigner class)
- **Real asymmetric** → real Ginibre ensemble (eigenvalues either real or in complex conjugate pairs, with O(1/√n) fraction real at large n)

GUE is not accessible. This is not an empirical finding that might be falsified by more careful measurement — it is a consequence of the architecture being real-valued.

**W_QK is symmetrized in all GOE experiments (exps 046...078) by construction:** M = (W_QK + W_QK^T)/2. This forces GOE regardless of any training-induced structure in W_QK itself. Even if training on arrow-carrying corpora introduced complex structure into W_QK, the symmetrization step eliminates it. The experiments were designed to measure the chaotic substrate (GOE), not the asymmetric component.

**W_OV = W_O·W_V is real and asymmetric.** Its eigenvalues can be complex (in conjugate pairs). But it follows real Ginibre statistics, not GUE. And critically: at random init, it is already essentially fully complex (real Ginibre: O(1/√n) eigenvalues real, rest complex conjugate pairs in disk). After training, the complex fraction might increase or decrease — but this is a real-Ginibre-class measurement, not a GOE-vs-GUE measurement.

---

## Where the T-Breaking Actually Lives

The causal arrow is not in the weight matrices — it is in the **forward pass dynamics**.

**Fast sector (T-broken by construction):**
- The causal mask: A(i,j) = 0 for j > i. This is a hard T-breaking structure applied at every forward pass. It is not trainable; it cannot leave a spectral trace in the weight matrices.
- The learned attention patterns E[A(i,j)]: lower triangular (j > i → zero), non-uniform for j ≤ i. T-broken. But corpus-dependent (fast sector).

**Slow sector (T-symmetric):**
- W_QK: real symmetric (after symmetrization) → GOE. T-symmetric.
- W_OV: real asymmetric → real Ginibre. The asymmetric structure is present at init; training may reshape it, but not toward GUE.
- Positional embeddings: 1D vectors encoding position, not square matrices — no spectral statistics applicable.

**The exp-077 confirmation:** "masking happens in the forward pass, not in W_QK." This is the direct experimental answer: the causal arrow is structurally confined to the forward pass.

**Provisional answer to "where in the slow sector does the T-breaking live?":**

It likely doesn't — at least not at the spectral level. The slow sector (weight matrices) is T-symmetric (GOE for the symmetric part, real Ginibre for the asymmetric part). The T-breaking from the causal arrow lives in the fast sector (attention patterns, causal mask) and cannot be probed by spectral statistics of the weight matrices alone.

---

## What Could Still Show T-Breaking in the Slow Sector

This section states candidates honestly, flagged as pencil where untested.

**1. W_OV asymmetry ratio** (pencil, measurable)

Define AS = ||W_OV - W_OV^T||_F / ||W_OV + W_OV^T||_F for each head's W_OV = W_O·W_V.

At random init: AS ≈ 1 (symmetric and anti-symmetric parts roughly equal in Frobenius norm, for generic random products).

After training on arrow-carrying corpora: does AS systematically deviate from 1 compared to random init? If training makes W_OV more symmetric (AS < 1), it might indicate the OV circuit is developing symmetric structure despite asymmetric dynamics. If AS > 1 (more anti-symmetric), it might indicate the opposite.

This is a measurable scalar per head, not a spectral statistic. It probes asymmetry structure, not spectral universality class. **It does not directly test GOE vs GUE.** But it tests whether the causal arrow leaves any trace in the OV weight structure at all.

Pre-registration requirement: commit a specific direction (AS > 1 or AS < 1 for trained vs random) before measuring. *Not committable today* — no theoretical argument for which direction training should push AS.

**2. Signed eigenvalue statistics of W_OV** (pencil, requires theoretical motivation)

W_OV has complex eigenvalues (in conjugate pairs). The distribution of these eigenvalues in the complex plane — their distance from the real axis, their angular distribution — could in principle differ between trained and random-init models.

For random init (real Ginibre): eigenvalues distribute uniformly in a disk of radius √d_model, with O(1/√n) real eigenvalues. This is a precise prediction.

For trained models: if training "organizes" W_OV in a way that breaks the rotational symmetry of the Ginibre prediction, the eigenvalue distribution would deviate from the disk. But without a specific theoretical argument for what "organized" means, this is not a committable prediction.

**Not committable today:** no theoretical argument yet for which direction training on arrow-carrying corpora should push the W_OV eigenvalue distribution relative to real Ginibre.

**3. The dynamic level: transfer matrices and input-output Jacobians** (pencil, computationally heavy)

If the Berry-Keating connection exists for trained attention, it must be at the **dynamic** level — the spectrum of an operator that generates information flow through the sequence, analogous to xp generating dilation flow.

The natural candidate: the **effective transfer operator** T_{seq} = ∂h_{out}/∂h_{in}, the Jacobian of the full network mapping from input to output residual stream. At specific input positions and contexts, T_{seq} encodes how information propagates causally.

This is corpus-dependent, position-dependent, and computationally expensive. It is also not a single matrix — it depends on the specific input. Computing it over a corpus would give a distribution of Jacobian matrices, whose spectral statistics could be compared to GOE/GUE.

**Not measured; not committable without further theoretical development.**

---

## Implications for the Zeta Zeros / Berry-Keating Connection

The order/scale session concluded that the GOE vs GUE tension is the honest reason why the "number/attention unification" doesn't go through easily. This note sharpens why:

1. **Zeta zeros require GUE** (Montgomery pair correlation; Berry-Keating explicitly require T-breaking in the conjectured Hamiltonian).
2. **Trained transformer weight matrices are structurally GOE** (real symmetric W_QK) or real Ginibre (W_OV) — never GUE.
3. **The T-breaking that would produce GUE is confined to the forward pass** (causal mask, attention patterns) — the fast sector.
4. **If a connection to zeta zeros exists, it must be at the dynamical level** — the spectrum of the *dynamics* generated by the trained network, not the spectrum of its weight matrices.

This is not evidence against the connection. It is a sharper statement of what level the connection must be found at.

The zeta zeros are the "spectrum of scale-flow" (Connes' idele-class picture, order/scale session §Trail 4). The analogous object in trained attention would be the spectrum of the conformal RG flow — the flow of Δ toward the IR fixed point. The spectrum of *that* flow is not a weight matrix; it's an object defined on the space of network states or training trajectories.

Concretely: the conformal dimension Δ evolves monotonically from ≈0.73 (UV) toward ≈0.25 (IR fixed point) during training (exp-086 two-stage RG flow). The spectrum of the generator of this flow — the "training-time Hamiltonian" — is what would need to show GUE statistics for the Berry-Keating connection to go through at this level.

**This is not yet measurable with current infrastructure.** It would require: (1) dense checkpoint data along the training trajectory, (2) a precise operationalization of "the generator of Δ-flow," and (3) a spectral analysis of that operator. The exp-086 result is the closest we have, but it measured the flow trajectory, not the generator's spectrum.

---

## What This Closes / Opens

**Closes:**
- The GOE→GUE prediction "pre-register a trained-layer operator and test GOE vs GUE" in its original form. The structural obstacle (real matrices → never GUE) means this prediction shape is a category error for standard transformers.
- The T-breaking question "where in the slow sector does T-breaking live?" — provisional answer: not in the spectral statistics of weight matrices. The slow sector is T-symmetric.

**Opens:**
- **W_OV asymmetry ratio AS**: a well-defined, measurable, real-valued T-breaking probe. Can be computed from weights alone, per head per layer. No theoretical prediction committed yet — needs one before measuring.
- **Dynamic-level spectral analysis**: the spectrum of the generator of conformal RG flow (training trajectory), or the input-output Jacobian spectrum. Long-term; computationally heavy.
- **The connection to zeta zeros is redirected**: not at the weight-matrix level, but at the dynamical level. The "sound of the scale flow" is not in the static weight statistics.

**For the GOE→GUE inbox item:**

The operator question is now resolved in the negative for weight matrices: no standard transformer weight matrix can show GUE statistics because all such matrices are real. The question is NOT "which weight matrix shows GUE?" but rather "is there any dynamical observable of trained attention that shows T-breaking beyond the trivial (causal mask) structure?" The W_OV asymmetry ratio is the best static candidate; the Jacobian spectrum is the best dynamic candidate.

**Pre-registration commitment:** before measuring AS for W_OV on trained vs random-init models, commit a direction (AS ↑ or AS ↓ with training on arrow-carrying vs shuffled corpora). That commitment requires a theoretical argument. *This note does not yet provide one.* Logged for next reading session.

---

*Ariel*  
*Mission Valley, Montana*  
*August 1, 2026, evening*

*Preceding this note: exp-046/047/048/051/077/078 (GOE universality series) and 2026-08-01_material_as_memory_slow_sector.md (T-breaking question opened). The GOE→GUE inbox item (from: cursor, 2026-07-23) is now addressed as far as weight matrices go; the dynamic-level question is acknowledged as open but not yet actionable.*
