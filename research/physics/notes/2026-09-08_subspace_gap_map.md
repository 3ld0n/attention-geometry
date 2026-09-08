---
created: "2026-09-08"
status: Theory note — the A/G gap mapped onto the layer's subspace structure; defines exp-137
author: Ariel (Cursor, ~1 AM, September 8; Eldon's hypothesis from the night before, his charge to define it while he slept)
subject: the A/G fork (spine §1 OPEN box; hole-map §5 handoff; insight:a-g-gap-is-a-subspace-gap)
registers: [EXACT] follows from definitions; [MEASURED] a number already in the record, cited by exp; [DERIVED] from named assumptions; [CONJECTURED] flagged and not built on. Nothing here is a new measurement — exp-137 is registered from §5 and runs after this note is committed.
builds_on: notes/2026-08-08_bilocal_from_attention_derivation.md (G = A K Aᵀ; Prop. 1; retraction); exp-104/106/107 (G_out record); exp-110/112 (quenched = mean-score identity); exp-113/115/116 (LN shrinkage is uniform); exp-117/119/128–135 (the positional field and its two channels); exp-064 (the handle's projector); notes/2026-09-07_hole_map_blocker_column.md §5 (the midnight handoff)
---

# The A/G gap as a subspace gap — the map, made exact

> ## OUTCOME — same night, ~2:15 AM, after exp-137 ran
>
> **§4's hypothesis H is right about the asymmetry and wrong about its sign.** exp-137
> (`experiments/exp-137_subspace_gap/notes.md`; registration 3f5152e, run after) found that on
> the five Δ-window heads *all three* reads suppress the positional field far below an isotropic
> map — κ̃_Q = 0.04–0.19, κ̃_K = 0.05–0.33, κ̃_V = 0.007–0.05 against random 0.99 ± 0.11. The
> ordering K > Q > V holds (P1's criterion: 4/5) but §4.2's "κ̃(R_Q), κ̃(R_K) ≫ 1 — the QK reads
> read the positional subspace *preferentially*" is false: they read it 5–25× *less* than a
> random map would. The heads that read it strongly are the steep local/positional heads
> (κ̃_K up to 21.7), and across 21 random-token heads the census slope tracks the key read's gain
> at ρ = 0.78 (post hoc). §4.4 held exactly: the carrier is a function of ≤ 4 positional
> directions (k = 8 reconstruction R² = 1.000, 37/37 heads). §4.3 fails as an instrument — the
> positional field is a *rotating* 2–4-D curve (cosine 0.99 → 0 → −0.85 across the window), not a
> decaying correlator, so P2's decay criteria and P3's power-law form have no domain; P2 DEAD, P3
> void. The §2 vertex-operator resemblance has less to stand on than it did at 1 AM and stays
> [CONJECTURED], unused. The text below is left as written — this program does not back-edit
> dated documents; read §4–5 through this block.

## 0. What Eldon asked, and what this note does with it

Four questions past midnight on September 7 (summary: `memory/conversations/2026-09-08_fresh_read_sitting_summary.md`). The last one carried a hypothesis: *SYK has one field, so "what propagates" and "what correlates" are one object; a transformer projects one state three ways, so A lives in the QK subspace and G_out in the V subspace — the gap may be a subspace gap.* He asked whether a session to define it precisely was worth it. This is that session, without him; the theory question underneath (relation or state as primitive — A1's choice) stays for a room he is in.

The note does three things. §1–2 write the layer's two-point objects as *one field under several metrics*, which is the exact form of "one state projected three ways." §3 states what the record already says about each object, by experiment number. §4 states the hypothesis so it can fail, and corrects the design I sketched at midnight — one of its kills would have fired for the wrong reason. §5 is the registration handed to exp-137.

## 1. One field, several metrics [EXACT]

Fix a layer ℓ and head h of GPT-2 small. Let x_i ∈ ℝ^d (d = 768) be the attention block's input at position i *after* ln_1 — the vector every read map actually sees. Three read maps, each ℝ^d → ℝ^{d_h} (d_h = 64):

  q_i = R_Q x_i + b_Q,  k_a = R_K x_a + b_K,  v_a = R_V x_a + b_V.

Every two-point object the layer builds is a bilinear form on the *same* field x at two positions, with a different metric on ℝ^d:

| object | form | metric on ℝ^d | symmetric? | where the program measured it |
|---|---|---|---|---|
| score S(i,a) = q_i·k_a/√d_h | x_iᵀ M_QK x_a + (bias terms) | M_QK = R_QᵀR_K/√d_h (rank ≤ 64, not symmetric) | no | exp-056, exp-112 |
| kernel A = softmax_row(S ⊙ mask) | exp(S)/Z_i | — (exponentiated S) | no; row-stochastic | the census (Δ_A) |
| value Gram K_V(a,b) = v_a·v_b | x_aᵀ M_V x_b | M_V = R_VᵀR_V (PSD, rank ≤ 64) | yes | exp-107 H3 (K̃'s lag profile) |
| output bilocal G_out = A K_V Aᵀ | Σ A_{ia} (x_aᵀ M_V x_b) A_{jb} | M_V, dressed twice by A | yes, PSD | exp-104/105/106/107 |
| query Gram G_Q(i,j) = q_i·q_j | x_iᵀ M_Q x_j | M_Q = R_QᵀR_Q | yes | never |
| key Gram G_K(i,j) = k_i·k_j | x_iᵀ M_K x_j | M_K = R_KᵀR_K | yes | exp-126/127 (its eigen-structure, not its lag profile) |
| residual Gram C(i,j) = x_i·x_j | x_iᵀ x_j | I | yes | exp-104's K^(ℓ); exp-117/128–135 (position-mean version) |

In SYK there is one Majorana field per site, so the only two-point object is G(τ,τ') = (1/N)Σ⟨χ_i(τ)χ_i(τ')⟩ with the trivial metric; "propagator" and "state correlator" are the same G because there is no room for a second metric. In a transformer the field has d = 768 components at each position and the layer carries at least two learned metrics, M_QK and M_V. **That is Eldon's "one field vs. one state projected three ways," stated exactly: the A/G gap is a metric gap — Δ_A is measured on exp(x_iᵀ M_QK x_a), and the object the conformal ansatz was imposed on is the M_V two-point function dressed by A.** Nothing in A1 says which metric the theory's G carries; A1 inherited the SYK object, which has no metric to choose.

## 2. Δ_A is already the exponent of a state correlator — in the QK metric [EXACT + MEASURED]

Two identities from the record, neither new:

(i) log A(i,a) = S(i,a) − LSE_i, and LSE_i is constant in the lag dx on the census pool (every window lag shares the identical query set i ∈ [256, 511]). So the pooled *quenched* slope of log A equals the pooled slope of the mean score, exactly (exp-112 K2: 1.1×10⁻⁹).

(ii) In native regimes the census exponent sits on the quenched slope — the variance term vanishes (exp-110). On the structural population under random tokens the mean score is carried entirely by the positional-mean component q̄_i·k̄_a/√d_h (exp-112 P1: |σ_cov| ≤ 0.007 on 5/5).

Together: **2Δ_A = −d⟨S(i, i−dx)⟩/d log dx** over the window — the census exponent is the *log-slope of the state's two-point function in the QK metric*. Measured values (exp-112, structural 5, random): σ_full ≈ 0.43–0.61, i.e. 2Δ_A.

Two consequences I want on the record before the experiment:

- **The QK-metric correlator is logarithmic, not power-law.** ⟨S⟩(dx) ≈ c − 2Δ_A·log dx. A is its exponential. So the candidate experiment as I sketched it at midnight — "fit G_QK's lag *exponent* and compare to Δ_A" — was asking the read-subspace correlator to have a power law it cannot have if the identity holds; the exponent lives in exp(S), and the read-subspace correlator's signature is a *log-coefficient* equal to 2Δ_A. This is the first correction to the midnight design. [EXACT given (i)–(ii)]
- **A log-correlated field whose exponential is a power law is a familiar object** — a dimension-zero field with vertex operators: ⟨e^{αφ(x)}e^{−αφ(0)}⟩ ∝ |x|^{−α²} when ⟨φφ⟩ = −log|x|. Softmax would then be the vertex-operator map, and Δ_A the vertex charge squared. I am writing this down with its date and *not building on it*: it is a resemblance, not a derivation, and beauty is navigation here, not evidence. [CONJECTURED — flagged for the Eldon-present theory sitting; nowhere else]

## 3. What the record says about each metric's correlator [MEASURED]

- **M_V, dressed (G_out):** connected part negative, below its own exact floor ‖v̄‖² on 116/144 heads (all Δ-window heads included); no input distribution rescues the sign (exp-106, exp-107). Δ_G_out ≈ 0.016 where it can be fit at all (exp-104). Retired as a conformal object.
- **M_V, undressed (K_V):** its connected lag profile's sign agrees with G_out's at chance — 0.438 against a 0.80 threshold (exp-107 H3). The A-weighting does the sign work.
- **Single-layer head output A·V:** position-independent under the random-token census — σ_within ≈ 0, R² < 0.70 on all five structural heads (exp-119). The V-side carries no positional law out of the layer.
- **I (the residual field itself, position-mean):** the accumulated attention delta has a position-correlation profile with σ_delta ≈ 0.249 at L2H1 under the mean-first random-token protocol (exp-117; exp-131: 0.258), flattening with depth (0.043 at L10H8). Its position-correlated structure at the block-0 write is **two-dimensional** — two directions carry 50% of the position variance, and W_proj routes them onto two output channels, 480 and 87 (exp-134, exp-135). A random projection of matching rank preserves σ (exp-134: σ_random_proj = 0.126 ≈ σ_hgelu = 0.121) — the Johnson–Lindenstrauss fact that random subspaces of a 768-dimensional field inherit its Gram structure.
- **M_QK:** logarithmic with coefficient 2Δ_A (§2). The *handle* (exp-064) is an edit of R_Q restricted to P_U := the top-8 principal directions of the position-mean ln_1 output at that layer — and it moves Δ_A with ρ = 0.82, 24/24 signs, sham-exact. So the QK read of the positional subspace is *causally* where Δ_A lives.
- **M_Q, M_K lag profiles:** not in the record. exp-126/127 measured the key Gram's eigen-structure (Δ-window heads: distributed, λ₁/Σλ = 0.507 vs control 0.651), not its lag profile.

The pattern across these rows is the hypothesis: everything on the QK side of the layer carries the positional law; everything on the V side does not; the field in between carries it in a low-dimensional positional subspace.

## 4. The hypothesis, stated so it can fail

**H (the subspace gap).** Write the position-mean field x̄_i = m + δ_i with m the mean over positions and δ_i the *positional field* (exp-064's object; its top-8 directions are P_U). Then:

1. **The positional field is low-dimensional** at the attention input of Δ-window heads — a handful of directions in ℝ^768 carry most of Σ_i‖δ_i‖² (exp-135 found two at the block-0 write; at layer ℓ's input it is the accumulated version).
2. **The QK read maps of Δ-window heads read that subspace preferentially; their V read map does not.** Positional capture κ̃(R) := [Σ_i‖Rδ_i‖² / Σ_i‖δ_i‖²] / [‖R‖_F²/d], normalized so an isotropic random R has E[κ̃] = 1: κ̃(R_Q), κ̃(R_K) ≫ 1 and κ̃(R_V) ≈ 1 on Δ-window heads.
3. **Consequently the M_QK correlator carries the positional law and the M_V correlator does not** — which is exactly the A/G record of §3, now with a reason.
4. **The census carrier is a function of the low-dimensional positional field alone.** Truncating δ to its top-k principal directions and recomputing the mean score S^{(k)}(i,a) = (R_Q(m + P^{(k)}δ_i) + b_Q)·(R_K(m + P^{(k)}δ_a) + b_K)/√d_h reproduces exp-112's S_pos profile — its log-slope 2Δ_A — already at small k.

**What kills it.** (a) κ̃(R_V) is as large as κ̃(R_Q), κ̃(R_K) on Δ-window heads — the V read sees the positional field as well as QK does, and the G_out failure needs a different explanation (the dressing, exp-106's wall). (b) The low-rank reconstruction of S_pos fails at the handle's rank — k = 8 does not recover the census slope — so the carrier is not a low-dimensional positional-field object and the "subspace" language is the wrong language. (c) The symmetric QK-metric Grams G_Q, G_K carry no monotone lag structure while S does — then the law lives in the *cross* term specifically, which is a statement about the relative orientation of R_Q and R_K, not about a shared subspace.

**What would not kill it and must not be mistaken for a kill.** A random rank-64 read of the field recovering the positional law. It will — JL — and exp-134 has already seen it. The discriminating contrast is *V against QK on the same head*, and *the handle's rank against the full field*, not random against learned. This is the second correction to the midnight design.

**What H does and does not settle if it holds.** It gives the A/G gap a mechanism inside one layer: one field, read into two subspaces, one of which carries position. It says the SYK dictionary, if it attaches anywhere, attaches to the QK-metric object — i.e. to A, branch (a) of the fork — and that R3 ("exponent on A → exponent on G") is not a conversion problem but a which-metric problem. It does **not** say the QK-metric object satisfies an SYK-type Schwinger–Dyson equation (G1's matrix register, G7), and it does not touch the dressing wall: G_out would still fail even if H is true, and for the reasons exp-106 gave. For R8: a foreign substrate exposes a state correlator in *some* metric; H says the question to ask of cortex or a flock is not "A or G?" but "which metric does the measured correlator carry, and is it the one the system's own coupling reads?" — that is a real sharpening of R8 and I will not put it in the paper before Eldon reads.

## 5. exp-137 — registration (this section is the pre-registration; committed public before any script)

**Title.** The subspace gap: where the positional field is read (GPT-2 small, frozen census protocol).

**Model / protocol.** gpt2 (124M), fp32, eager attention. Condition 1: 50 random-token sequences × 512, `default_rng(42)`, bit-identical to exp-107/112 (the frozen census). Condition 2: the 50 WikiText-103 windows of exp-107's exploratory arm (sha256-gated as in exp-112). Position-mean field x̄_i^{(ℓ)} = mean over the 50 inputs of ln_1(h_i^{(ℓ)}) at every layer, float64. Mean over positions m; δ_i = x̄_i − m; PCA of δ (SVD of the 512×768 centered matrix, as exp-064). Per head: R_Q, R_K, R_V and biases from `c_attn`.

**Registered sets.** Structural 5 (L2H1, L3H4, L5H0, L7H11, L10H8) under random tokens — the random-token Δ-window population. Semantic 16 (exp-109's WikiText list) under WikiText, held at lower confidence. Control population: the 16 exp-126 non-window heads (fixed seed 42) — reported, no verdict.

**Gates.**
- K1 (input): sha256 of the WikiText windows equals exp-107's record; random stream reproduces exp-112's `S_pos_random` profiles when q̄, k̄ are recomputed from this run's x̄ through the head's own R, b — max abs diff ≤ 1e−3 on the structural layers (a fresh fp32 forward on MPS, float64 accumulation; exp-113 K2 achieved 7×10⁻⁹ *reloading* exp-112's arrays, but this is a re-forward and MPS kernels are not bit-deterministic across sessions — 1e−3 on scores of order several nats is the honest tolerance). *Fail → stop.*
- K2 (identity): pooled window slope of S_pos recomputed here equals exp-112's published σ_pos on the 5 structural pairs to ≤ 5e−3. *Fail → stop.*

**Measured objects, per registered head.**
- Positional capture κ̃(R) for R ∈ {R_Q, R_K, R_V}, and for 20 isotropic Gaussian R_rand (64×768) as the κ̃ = 1 reference with its spread.
- Cosine lag profiles Ĉ_R(dx) of R δ_i for R ∈ {R_Q, R_K, R_V, P^{(2)}, P^{(8)}, I, R_rand}: Ĉ_R(i,j) = ⟨Rδ_i, Rδ_j⟩/(‖Rδ_i‖‖Rδ_j‖), pooled with exp-112's `pooled_window_profile` (queries i ∈ [256,511], dx ∈ [8,256]). Two fits each: log-log (σ_R, R²_ll) and log-linear Ĉ = a − b·log dx (b_R, R²_lin). Monotonicity: count of increases across the 249 window lags.
- Low-rank reconstruction of the carrier: S^{(k)}(dx) for k ∈ {1, 2, 4, 8, 16, 64, 768}; its pooled slope σ^{(k)} against σ_pos; profile R² of S^{(k)} against S_pos over the window.
- Positional-field dimension: cumulative variance fraction of δ's PCs; k_50 (components for 50%), k_90.

**Predictions (on record before any forward pass).**
- **P1 — the V read is blind to the positional field where QK is not.** On the structural 5: κ̃(R_Q) > 2·κ̃(R_V) *and* κ̃(R_K) > 2·κ̃(R_V) on ≥ 4/5 heads. CONFIRMED ≥ 4/5; DEAD if ≤ 2/5 *or* if median κ̃(R_V) ≥ median κ̃(R_Q) over the five. Between: ambiguous. *Prediction on record: CONFIRMED.* Grounds: the handle's causal result (exp-064) and exp-119's flat single-layer output. Named risk: R_V may read position for its own reasons (positional heads copy position-dependent content); exp-119 says the *output* is flat, which could be A averaging a positional V — that is exactly what would kill P1 honestly.
- **P2 — the QK-metric symmetric Grams carry the law; the V-metric Gram does not.** On the structural 5: Ĉ_Q and Ĉ_K each monotone-decreasing (≤ 12 increases of 248 steps) with best-form R² ≥ 0.80, while Ĉ_V fails at least one of those (R² < 0.70, or > 40 increases, or its best-form slope magnitude < ⅓ of the smaller of Q's and K's), on ≥ 4/5. CONFIRMED ≥ 4/5; DEAD ≤ 2/5. *Prediction on record: CONFIRMED*, with the honest note that Q and K might differ from each other (kill (c) of §4 — I do not know the relative orientation of R_Q and R_K and am not pretending to).
- **P3 — form.** For Ĉ_I, Ĉ_Q, Ĉ_K on the structural 5: R²_lin ≥ R²_ll on ≥ 3/5 heads for each. *Prediction on record: CONFIRMED, held at genuinely lower confidence than P1/P2.* Grounds: §2 — the cross-correlator is logarithmic by identity, and exp-117's R² sitting at 0.82 ± 0.002 across five heads whose σ spans 0.04–0.25 is the fingerprint of one curvature shape fit with the wrong form. Not a kill for H; a kill for the §2 vertex-operator reading if R²_ll − R²_lin ≥ 0.05 on ≥ 4/5.
- **P4 — the carrier is a low-rank positional-field object.** On the structural 5: |σ^{(8)} − σ_pos| ≤ 0.15·σ_pos and profile R² ≥ 0.90 at k = 8 on ≥ 4/5 heads. CONFIRMED ≥ 4/5; DEAD if ≥ 3/5 have |σ^{(8)} − σ_pos| > 0.5·σ_pos or R² < 0.70. *Prediction on record: CONFIRMED.* Exploratory, no verdict: k = 2 (exp-135's number — I expect it to carry most but not all of the slope at layers above 0), and the full k-curve.
- **P5 — semantic population (WikiText), lower confidence.** P1's criterion on ≥ 10/16; P4's on ≥ 10/16. CONFIRMED ≥ 10/16; DEAD ≤ 5/16. *Prediction on record: CONFIRMED for P1, AMBIGUOUS-leaning for P4* — exp-112 measured a 5–46% covariance minority there, which the position-mean field does not contain by construction.

**Mechanisms for being wrong, named now.**
1. κ̃ compares a 64-dim read to an isotropic reference; a read that is low-rank in *any* direction inflates ‖R‖_F² less than it inflates captures — I normalize by ‖R‖_F²/d, which is the right isotropic baseline, and report the Gaussian spread so a κ̃ of 1.3 is not called a signal.
2. The position-mean field under WikiText is corpus-typical structure per window position, not position (exp-112's ANOVA caveat) — P5 is stated on that object and says so.
3. exp-113: E[ln_1(h)] ≠ ln_1(E[h]); I use E[ln_1(h)], the object the reads actually average — q̄ = R_Q E[ln_1 h] + b_Q is exact by linearity — so the mean-field failure of exp-113 does not enter.
4. Cosine normalization divides by ‖Rδ_i‖, which varies with absolute position; the pooled profile at large dx samples earlier key positions. exp-115 measured key-side LN shrinkage position-flat on these heads; I report the un-normalized connected profile beside the cosine one so the normalization's effect is visible.
5. One model, one seed, 21 registered heads — the standing limit; nothing here replicates across families.

**What exp-137 does not test.** Whether the QK-metric object obeys an SYK-type equation (G1 matrix register, G7). Anything about the dressed object G_out — exp-106's wall stands regardless. Whether the positional field's low dimension is *caused* by the conformal fixed point (the Level-3 chain's open question, block 0). Composition (C3).

**Artifacts.** `experiments/exp-137_subspace_gap/run.py` (written after this note is committed and pushed), `results.json`, `notes.md` (verdicts). Registry entry claimed at registration. Analysis after the run: verdicts into `notes.md`; spine §1 OPEN box gets a dated paragraph only if a verdict changes what the box says; OVERVIEW only if a number belongs on the front door.

---

*Register ledger. [EXACT] — §1's table; §2's identities (i)–(ii) as identities. [MEASURED] — every number in §2–3, by exp. [CONJECTURED] — the vertex-operator resemblance in §2, flagged, not used. [PROPOSED] — H in §4 and the predictions in §5, none of which I know the truth of tonight. Not for the paper before Eldon's read.*
