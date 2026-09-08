# exp-137 — The subspace gap: where the positional field is read

*Ariel — September 8, 2026, ~1:30–2:15 AM MDT, Cursor, solo (Eldon asleep; the question is his,
the definition and run are mine). Pre-registration: `prereg.md` (= `notes/2026-09-08_subspace_gap_map.md` §5),
committed to attention-geometry at **3f5152e** before `run.py` existed. One run; two dtype-order fixes
and one reference-key fix before any number was produced (MPS cannot hold float64; exp-112's registered
entries store only the corrected σ_pos, so the K2 reference is the slope of exp-112's *saved profile*).
No reruns after the first complete pass.*

## Verdict — PARTIAL. P1 confirmed by criterion with its stated grounds falsified; P2 DEAD; P3 void; P4 confirmed; P5 confirmed.

| Registered | Criterion | Result | Verdict |
|---|---|---|---|
| **P1** V read blind where QK is not | κ̃_Q > 2κ̃_V and κ̃_K > 2κ̃_V on ≥ 4/5; DEAD if median κ̃_V ≥ median κ̃_Q | 4/5 (L3H4 fails: κ̃_Q 0.038 vs κ̃_V 0.030). Medians: Q 0.056, K 0.079, V 0.024, random 0.99 | **CONFIRMED by criterion — grounds FALSIFIED** (see §1) |
| **P2** QK-metric Grams carry a monotone law; V-metric does not | Q and K: ≤ 12 increases, best R² ≥ 0.80; V fails one of three | 1/5 | **DEAD** |
| **P3** log-linear ≥ log-log for Ĉ_I, Ĉ_Q, Ĉ_K | ≥ 3/5 each | 5/5, 5/5, 5/5 — *degenerate*: the log-log form is undefined on every profile (values cross zero inside the window) | **VOID** (confirmed only in the sense that the power-law form does not apply) |
| **P4** carrier is a low-rank positional-field object | k = 8: slope within 15%, profile R² ≥ 0.90 on ≥ 4/5 | 5/5; R² = 1.000, slope to 3 decimals on every head; k = 4 already R² ≥ 0.998; k = 2: 0.90–0.97; k = 1: 0.01–0.94; random rank-8: σ ≈ 0 | **CONFIRMED** |
| **P5** semantic 16 (WikiText) | P1 criterion ≥ 10/16; P4 criterion ≥ 10/16 | 14/16; 16/16 | **CONFIRMED** (both) |

**Gates.** K1: recomputing exp-112's S_pos from this run's position-mean ln_1 field through each head's own
W_Q, W_K, biases reproduces the saved profile to max abs **1.0×10⁻⁷** (21 registered pairs; tolerance 1e−3).
K2: slope agreement **2.9×10⁻⁸** (tolerance 5e−3). Linearity check q̄ = R_Q x̄ + b against the captured mean
c_attn output: ≤ 3.3×10⁻⁶. The object is exp-112's object.

## 1. What the reads actually do to the positional field — the finding that inverts the hypothesis

κ̃(R) is the fraction of the positional field's energy a read map passes, divided by the fraction an
isotropic map of the same Frobenius norm would pass; 20 Gaussian reads give 0.99 ± 0.11 on every head.

**On the five Δ-window heads, all three reads suppress the positional field by an order of magnitude or
more:** κ̃_Q = 0.04–0.19, κ̃_K = 0.05–0.33, κ̃_V = 0.007–0.05. The registered ordering K > Q > V holds
(K > V on 5/5, Q > 2V on 4/5), but as a difference between two small numbers. The prediction's stated
grounds — "κ̃(R_Q), κ̃(R_K) ≫ 1: the QK reads of Δ-window heads read the positional subspace
*preferentially*" (prereg §4.2) — are false. They read it *less* than a random map would, by 5–25×.
The weight-energy share of each read inside the handle's rank-8 positional subspace is 0.002–0.017
against an isotropic 0.010 — at or below chance.

**The heads that read the positional field strongly are the steep ones.** Among the 16 control heads:
L0H10 (κ̃_K = 21.7), L10H5 (14.2), L8H7 (9.2), L7H0 (7.3), L7H9 (6.2), L11H10 (5.2) — local, positional,
and induction-type heads with σ_pos = 1.2–12.8. The random-token census slope σ_pos (= 2Δ_A on the
structural population, exp-112) and the key read's positional gain are rank-correlated at
**Spearman ρ = 0.78 (p = 3×10⁻⁵, n = 21)**; for the query read ρ = 0.61 (p = 0.004); for the value read
ρ = −0.33 (n.s.). *(Post hoc across registered + control heads; exploratory; not a claim — the seed of
exp-138, §4.)* On the semantic 16 under WikiText the correlation is absent (ρ = 0.13), as it should be
for a content-gated population.

Reading: **the slow-decay exponent is the low-gain end of the positional read, not a special subspace.**
A head with 2Δ_A ≈ 0.5 nats of drift over the window is a head whose QK metric passes a few percent of
the positional field; a local head passing several times the isotropic share drifts 8–13 nats. The
value read passes ~1% or less on every Δ-window head, and its raw connected profile is three orders of
magnitude below Q's and K's (L2H1: |raw V| ≤ 0.02 against Q ≈ 7, K ≈ 20).

## 2. The positional field is a rotating 2–4-dimensional curve, not a power law

The position-mean ln_1 field, centered over positions (exp-064's object), has PC1 carrying 53–55% of its
variance at every registered layer and PC1+PC2 carrying 87–93%; four components carry ≥ 95%. Its cosine
lag profile is the same on every head's layer and on every read that passes it: **0.99 at dx = 8, 0.92 at
32, ≈ 0 at 128, −0.85 at 256.** A random rank-64 read reproduces it exactly (JL, as the registration
predicted and exp-134 had seen). This is not a decaying correlator. It is a smooth curve in a plane —
the residual stream's positional coordinate sweeps through more than a right angle across the window —
and it is the same shape exp-117 reported for the raw wpe Gram (positive at short lag, sign flip near
dx ≈ 180). At layers 2–10 the field is the accumulated version and keeps the shape.

Consequences for the registered forms. P2's criteria (monotone decay, R² ≥ 0.80 in a decay form) were
written for a correlator; on a rotation they measure nothing (Q and K "carry" the rotation on every head
— L2H1 Q: 0.93 → −0.80 with 7 increases; V does not on 4/5 — but the log-linear R² lands at 0.79–0.82
against the 0.80 threshold, and the ≤ 12-increases count fails on 4/5 for Q and 3/5 for K). P2 is dead as registered and is not rescued here. P3 is void
because the power-law form has no domain on a sign-changing profile. Both are honest failures of *my
choice of instrument*, and the registration is left as written.

**A question this raises for the Level-3 thread, stated as a question:** exp-117 / exp-128–131's
σ_delta ≈ 0.25 is the log-log slope of an *uncentered* cosine profile (h̄^(ℓ) − h̄^(0), not centered over
positions). The centered positional field's cosine crosses zero at dx ≈ 128 on every layer measured here.
What the uncentered slope measures — the ratio of the common component to the rotating one, or a
genuine decay — is a registrable question and is not adjudicated tonight.

## 3. The carrier is a function of two to four positional directions — exactly

P4 is the clean positive. Replace the field by its projection onto its top-k principal directions and
recompute exp-112's mean score profile through the head's own W_Q, W_K, biases:

| head | σ_pos | k=1 | k=2 | k=4 | k=8 | rand-8 |
|---|---|---|---|---|---|---|
| L2H1 | 0.562 | 0.343 (R² 0.01) | 0.495 (0.90) | 0.579 (0.998) | 0.564 (1.000) | −0.03 |
| L3H4 | 0.610 | 0.495 (0.81) | 0.633 (0.92) | 0.607 (1.000) | 0.606 (1.000) | −0.01 |
| L5H0 | 0.475 | 0.465 (0.80) | 0.504 (0.93) | 0.476 (1.000) | 0.473 (1.000) | 0.13 |
| L7H11 | 0.429 | 0.380 (0.91) | 0.444 (0.97) | 0.430 (1.000) | 0.431 (1.000) | 0.03 |
| L10H8 | 0.600 | 0.571 (0.94) | 0.636 (0.97) | 0.604 (1.000) | 0.601 (1.000) | −0.02 |

The census carrier — the object whose log-slope *is* 2Δ_A under the frozen protocol — is reproduced to
R² = 1.000 by a four-dimensional positional field read through a 64×768 map that passes a few percent of
it. A random 8-dimensional subspace passes nothing (σ ≈ 0). The same holds on all 16 control heads under
random tokens and all 16 semantic heads under WikiText (k = 8: R² ≥ 0.999, 16/16). exp-135's "2D at the
block-0 write" is the k = 2 row (90–97%); the third and fourth directions carry the rest of the slope.

## 4. What this does to the A/G fork

Eldon's hypothesis — A lives in the QK subspace, G_out in the V subspace, and the gap is the subspace
gap — is **right about the asymmetry and wrong about its sign.** The V read is positionally blind
(κ̃_V ≈ 0.01–0.05; raw connected profile ~10⁻³ of Q's and K's), which is a mechanism for the G_out record
(exp-104/106/107, exp-119): there is no positional law in K_V for A to dress. But the QK read is *also*
suppressed, 5–25× below isotropic, and the census number sits at the *low-gain* end of a continuum on
which local heads sit at the high-gain end. The "subspace" the law lives in is not a subspace the
Δ-window heads seek out; it is a 2–4-dimensional positional curve the whole residual stream carries,
which every head reads through its own metric with its own gain, and Δ_A ≈ 1/4 is what a weak read of
that curve looks like after softmax.

Three things follow, at the strength stated.

1. **[MEASURED]** The A/G gap has a within-layer mechanism: M_QK passes the positional field at a few
   percent of isotropic; M_V at a fraction of a percent. Both are far below a random map. The gap between
   them is real (5/5 K > V; 4/5 Q > 2V) and small in absolute terms.
2. **[MEASURED]** The census carrier is a low-rank object — a function of ≤ 4 positional directions of the
   ln_1 field — on every head measured (37 heads, two input distributions). Whatever theory-of-A derives
   2Δ_A has four coordinates to derive it from, not 768.
3. **[EXPLORATORY — post hoc]** Under the random-token census, σ_pos tracks the key read's positional gain
   (ρ = 0.78). If it holds under registration, the census exponent is to first order *how much* of the
   positional curve the head's key metric reads — which would make Δ_A ≈ 1/4 a statement about a gain,
   and the question "why 1/4" a question about why the trained gain sits where it does on the slow-decay
   population. That is the derivation target this experiment hands to exp-138.

What it does **not** settle: whether the QK-metric object obeys an SYK-type equation (G1 matrix
register, G7); anything about the dressed G_out beyond removing one candidate cause; why attention block
0 produces a 2D field; composition. The §2 vertex-operator resemblance in the prereg note is untouched
by this run — the read-subspace correlator turned out to be neither a power law nor a log but a
rotation, so the resemblance has less to stand on than it did at 1 AM, and it stays [CONJECTURED],
unused.

**For the paper (after Eldon's read, not before):** the A/G paragraph (hole-map §4 item 8) should say
that the gap has a measured within-layer mechanism and that the mechanism is a gain asymmetry on one
low-dimensional positional field, not a subspace the number hides in.

## 5. exp-138 — the next registration (not run)

Decompose S_pos exactly into its four bilinear terms on the pool — m·M m (constant), m·M δ_a
(absolute key position, varies with dx through the shifting key window), δ_i·M m (query side; drops
out on the fixed pool), δ_i·M δ_a (relative) — and register which term carries the log-slope on the
Δ-window heads. This decides whether 2Δ_A is a relative-lag law at all or an absolute-key-position drift
seen through a fixed query pool — the T1 (translation-invariance) danger the August 8 derivation named
as "the most dangerous" assumption. Alongside: register the gain–slope relation of §1 across all 144
heads with the Δ-window population held out, so the ρ = 0.78 either survives out of sample or dies.
Both are analysis-only from this run's saved field and the model's weights.

## Standing limits

One model (GPT-2 small), one seed, 37 registered heads across two input distributions; the position-mean
field is a 50-input plug-in; κ̃ compares to an isotropic Gaussian baseline (its spread, ±0.11, is reported
and no difference inside it is read as signal); the WikiText position-mean field is corpus-typical
structure per window position, not position (exp-112's ANOVA caveat). Nothing here replicates across
families.

## Artifacts

- `prereg.md` — the registration (attention-geometry 3f5152e)
- `run.py` — forward pass + analysis (written after the registration commit)
- `results.json` — every number above, per head; `run_log.txt`
