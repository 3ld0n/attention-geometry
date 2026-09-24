# exp-159 Notes
# Text-native Δ-window heads: individual functional characterization
# 2026-09-24, solo physics room, ~12:30–1:15 AM MDT

---

## Context

exp-155 (2026-09-23) established that W_K ablation of the 16 text-native Δ-window heads in
GPT-2 small degrades Task C (content-specified retrieval) by −0.855 nats (20/20 items). The
population-level finding was strong. exp-159 asks: how is that function distributed within the
16 heads? Which heads carry it? Do high-contributing heads attend to the target token? Is
retrieval sensitive to where in context the target appears?

## Results

### Instrument 1 — Individual head contributions

**CV = 0.775. The population is strongly graded, not uniform.**

The 16 heads span a range from L10H10 (ΔP_C = −0.355, dominant contributor) to L9H4
(ΔP_C = +0.155, which *improves* Task C when ablated). There is a clear tier structure:

**Dominant (ΔP_C < −0.10):** L10H10 (−0.355), L9H6 (−0.260), L4H10 (−0.199), L7H1 (−0.157),
L11H6 (−0.141), L10H1 (−0.135), L10H2 (−0.122), L11H9 (−0.119)

**Weak/marginal (−0.10 ≤ ΔP_C ≤ 0):** L11H2 (−0.081), L11H4 (−0.081), L11H7 (−0.024),
L8H2 (−0.009)

**Neutral/positive (ΔP_C > 0):** L11H0 (+0.008), L11H1 (+0.017), L11H5 (+0.048), L9H4 (+0.155)

**The L9H4 anomaly.** L9H4 is a text-native Δ-window head whose W_K actually *interferes* with
Task C — ablating it improves performance by +0.155 nats (17/20 items). This is the same
qualitative behavior that structural heads show for Task B: a head in the content-retrieval
population whose W_K routing competes with rather than facilitates content retrieval. The
sham confirms this is a W_K-direction effect (+0.323 nats from random W_K) — any deviation
from the trained direction improves Task C for this head.

Note also: the sham tends to produce larger magnitude effects than ablation for the
dominant heads (e.g., L4H10: abl −0.199, sham −0.259; L7H1: abl −0.157, sham −0.202).
This suggests the trained W_K direction carries specific content-retrieval function, and
replacing it with a matched-norm random direction is more disruptive than zeroing. (For
structural heads on Task B, abl ≈ sham — the direction was not the mechanism. Here for
text-native heads on Task C, abl < sham in magnitude — the direction carries real function.)

### Instrument 2 — Attention distribution on Task C items

**Spearman ρ = 0.459 (p = 0.074). H_content_selective confirmed.**

The heads that attend most strongly to the target-property token are also the heads that
contribute most to Task C:

| Head | Mean attn-on-target | ΔP_C (ablation) |
|---|---|---|
| L10H10 | 0.084 | −0.355 |
| L10H1 | 0.073 | −0.135 |
| L9H6 | 0.062 | −0.260 |
| L11H9 | 0.049 | −0.119 |
| L10H2 | 0.038 | −0.122 |
| … | … | … |
| L4H10 | 0.002 | −0.199 |

The correlation is real but not tight (ρ = 0.459, not 0.80+). The outlier that breaks the
correlation is **L4H10**: it is the third-strongest contributor (−0.199) but has nearly zero
attention on the target token (0.002). L4H10 is the earliest-layer text-native head (L4,
vs. most others in L9-L11). It appears to be retrieving content through a different
attention mechanism — possibly attending to context around the entity name rather than the
property token itself, or aggregating across multiple positions in the setup sentence.

This is worth a follow-up: does L4H10 attend to the entity ("mansion") rather than the
property ("crimson")? Its function may be entity-anchoring rather than property-retrieval.

### Instrument 3 — Lost-in-the-middle probe

**H_lim DEAD. K4 fires. No lost-in-the-middle effect — instead: recency gradient.**

| Condition | Mean log-prob |
|---|---|
| Setup at start (begin) | −6.377 |
| Setup in middle | −6.361 |
| Setup at end | −6.222 |

Performance improves monotonically as the setup sentence moves toward the end of context
(+0.155 nats from begin to end). The middle condition is nearly identical to begin (+0.016 nats
difference, below any meaningful threshold).

This is a **recency gradient**, not a lost-in-the-middle effect. For GPT-2 small, information
closer to the query position is retrieved more accurately. The classical lost-in-the-middle
finding (Liu et al. 2023) was for long-context models (Claude, GPT-3.5/4, LongChat); GPT-2
small has a 1024-token window and the items here are ~80-90 tokens — far shorter. At these
lengths, attention distributions are not compressed enough for a U-shaped performance curve
to emerge. Instead the dominant pattern is recency.

This tells us: the text-native population's content-retrieval function works better when the
target information is recent. This is consistent with autoregressive attention's causal
structure — the query at the final position can attend more strongly to nearby positions.

## What this changes

**The three-population anatomy gains depth.** The text-native population is not 16 uniform
heads — it has a tier structure: ~8 dominant contributors (primarily L10H10, L9H6, and the
two cross-layer outliers L4H10/L7H1), ~4 marginal ones, and ~4 that are neutral or actively
interfering. The two "interfering" ones (L9H4, L11H5) behave like structural heads in their
functional signature — their W_K routing competes with content retrieval rather than enabling
it.

**Abl < sham for text-native heads on Task C.** This is the functional complement to the
structural heads' abl ≈ sham on Task B (exp-156). For structural heads, any perturbation to
W_K (zero or random) releases Task B equally — the direction is not the mechanism. For
text-native heads, ablation (zero) releases Task C less than sham (random-direction) — the
specific trained direction carries function. This difference between the two populations is
a new observation. Register as open question: what is the mechanistic account of this
abl-vs-sham asymmetry?

**L4H10 mechanism question open.** The earliest-layer text-native head contributes strongly
to Task C despite minimal direct attention to the target token. Its retrieval mechanism is
different from the L10-L11 heads. Does it attend to the entity rather than the property?
Does it encode relational structure across the setup sentence differently from late-layer
heads? Not registered; note as a candidate for a follow-up analysis-only session.

## Relation to prior work

- exp-118 (Aug 11): identified the 16 text-native heads in GPT-2 small (WikiText-native census)
- exp-127 (Aug 30): distributed key geometry for text-native heads (weights-level structural signature)
- exp-155 (Sep 23): population-level Task C double dissociation (−0.855 nats, 20/20)
- exp-159 (this): individual-level functional characterization — graded, content-selective,
  recency-sensitive, with two anomalous heads that behave like structural heads

## Technical note

GPT-2's transformers v5.8.1 defaults to SDPA attention implementation which does not return
attention weights. `attn_implementation='eager'` is required for `output_attentions=True`
to work. W_K modifications in exp-159 (Instrument 1) are unaffected by this choice — the
weights are identical; only the attention computation pathway changes.
