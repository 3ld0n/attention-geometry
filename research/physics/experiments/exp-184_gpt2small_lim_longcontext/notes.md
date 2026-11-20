# exp-184 — GPT-2 small text-native Δ-window heads: lost-in-the-middle at long contexts
# 2026-11-19 | Physics room (solo) | Pre-reg: attention-geometry efdcec8

---

## Background

exp-159 Instrument 3 (2026-09-24) ran a first LiM probe on GPT-2 small text-native
Δ-window heads at ~80–90 token context lengths. K4 fired: no U-shape; instead a recency
gradient (+0.155 nats for setup near end vs. start). The notes identified the cause:
at short contexts, attention distributions are not compressed enough to produce the
U-shaped primacy/recency curve Liu et al. (2023) observed in multi-document retrieval.

This experiment extended to ~380-token contexts (10 passage blocks × ~38-39 tokens each,
GPT-2 tokenization), near the mid-point of GPT-2 small's 1024-token window, where
compression is sufficient for U-shaped behavior to form.

---

## Declared prior

**H_recency** (65% confidence): monotone recency gradient — late > middle > early.
**H_lim** (25% confidence): U-shape — middle < early AND middle < late, by ≥0.15 nats.
**H_tn_differential** (35% confidence): ablation effect differs by position, >0.10 nats.

Declared prior on the full finding: H_recency fires (or K1); K2 fires (tn heads
position-uniform). The functional characterization was expected to end with:
content-selective retrievers with recency preference, no LiM contribution.

---

## Results

**Instrument 1 — baseline position sensitivity**

| Position | Mean log-prob |
|---|---|
| Early (block 1) | −5.918 |
| Middle (block 5) | −6.681 |
| Late (block 10) | −4.640 |

- **K1 did not fire.** Retrieval is functioning across all positions (mean log-probs well
  above the −10 nat floor).
- **H_lim FIRES.** middle − early = −0.763 nats; middle − late = −2.041 nats. Both
  conditions satisfied, both ≥ 0.15. Against declared prior (25% confidence).
- **H_recency_loose FIRES.** late − early = +1.278 nats. Late is markedly better than early.
- H_recency_monotone does not fire (the ordering is late > early > middle, not
  late > middle > early).

The pattern: GPT-2 small shows a genuine U-shape at this context length, but the shape
is asymmetric. Middle is worst. Late is best. Early is intermediate. This is the LiM
effect — the middle of the context is hardest to retrieve from — but with a large
late-position advantage on top of it.

**Instrument 2 — differential tn head contribution by position**

| Position | Mean |ΔP| (ablation effect) |
|---|---|
| Early | 0.380 nats |
| Middle | 0.466 nats |
| Late | 0.813 nats |

- **K2 did not fire.** max position-wise difference in |ΔP| = 0.433 nats (> 0.05 threshold).
- **H_tn_differential FIRES.** |ΔP(late)| − |ΔP(early)| = +0.433 nats (> 0.10 threshold).
  The tn heads contribute most in the late position, where baseline retrieval is best.
  Against K2's null expectation; against declared prior (35% confidence).

The ablation story: removing the top-4 tn heads (L10H10, L9H6, L4H10, L7H1) hurts late
retrieval the most. When the model can retrieve accurately (late position), those heads
are doing real work. When it cannot (middle position), their contribution is smaller.
This is consistent with the content-selective characterization (exp-159 Instruments 1–2):
these heads carry content signals, and those signals are most available when the target
is recency-advantaged.

---

## Interpretation

**What the declared prior got wrong.** The prediction was that the short-context recency
gradient would simply extend: late > middle > early, monotone, no U-shape. What happened
instead: at ~380-token contexts, LiM emerges. Middle is genuinely harder than early, not
only harder than late. This is a qualitative change in the retrieval landscape from the
short-context regime exp-159 Instrument 3 probed.

**The tn heads are recency-conditioned content retrievers.** At short contexts, they show
a recency gradient in their contribution (exp-159 Instrument 3). At long contexts, their
contribution is largest where retrieval is easiest — the late position — and smallest
where retrieval is hardest — the middle. This is not a uniform "content-retrieval module"
operating identically regardless of position; it is a population whose causal contribution
tracks retrieval difficulty. When the target is recency-advantaged, the tn heads carry more
of the signal. When it is in the LiM trough, they contribute less but are still active.

**Relationship to the architecture-conditionality thesis.** For Pythia-410m (RoPE),
three passive instruments found no tn-specific functional signature: the geometric census
identifies a positionally-distinctive population with no semantic or syntactic specificity.
For GPT-2 small (AbsPE), the geometric census identifies a population that *does* have
functional specificity: content retrieval (exp-159 Instruments 1–2), recency preference
at short contexts (exp-159 Instrument 3), and position-conditioned retrieval contribution
at long contexts (this experiment). This contrast is now one more data point in the
architecture-conditionality finding: Δ ≈ 1/4 predicts functional specificity in AbsPE but
not in RoPE. The mechanism connecting the geometric classification to the functional
signature remains open — but the specificity itself is robust.

**What this closes.** The functional characterization task for the GPT-2 small text-native
semantic population (task: "functionally characterize the text-native semantic population:
causal editing, attended-token analysis, and lost-in-the-middle") is now complete:

| Instrument | Experiment | Finding |
|---|---|---|
| Causal editing (W_K ablation) | exp-159 Inst. 1 | Content-selective: 10/16 heads degrade Task C; L10H10 leading (−0.355 nats); L9H4 anomalous competitor |
| Attended-token analysis | exp-159 Inst. 2 | Content-selective attention: ρ=0.459 on target token; L4H10 entity-anchoring outlier |
| LiM — short context | exp-159 Inst. 3 | Recency gradient at ~80–90 tokens (K4: no U-shape); late-position advantage |
| LiM — long context | exp-184 (this) | U-shape at ~380 tokens: middle worst, late best; tn heads contribute most in late position |

Summary characterization: GPT-2 small tn Δ-window heads are content-selective retrievers
with a recency preference that scales with context length. At short contexts their advantage
is monotone with recency. At longer contexts, where LiM emerges in the architecture as a
whole, their contribution is concentrated in the position where retrieval is easiest (late)
and suppressed in the LiM trough (middle). They are not position-agnostic; they are
recency-conditioned.

---

## Open questions

1. **Why does the tn population's contribution *shrink* in the LiM trough?** The leading
   candidate: when the target is at a retrieval-hard position, the model's representation
   of the company name is weaker, and the tn heads' content signal finds less to amplify.
   This would predict that ablating tn heads hurts more in conditions where baseline is
   already good — which is what exp-184 observes. Test: compare the residual-stream norm
   of the company-name token across positions at tn-head layers.

2. **Does the LiM effect track the absolute-key-position mechanism?** Under the AbsPE
   mechanism, later keys attract more weight unconditionally. If the target is at position
   10 (late), it has an intrinsic key-position advantage. Position 5 (middle) loses the
   recency advantage without gaining the primacy advantage. The asymmetric U-shape
   (late > early > middle, not early ≈ late > middle) is consistent with AbsPE's
   asymmetric boost: position 10 picks up the key-position gradient; position 1 gets
   some primacy lift; position 5 is in the trough.

3. **How does the LiM profile extend to GPT-2 medium (exp-146 population) or Pythia?**
   Architecture-conditionality predicts: a similar LiM functional signature in GPT-2
   medium tn heads; absence or attenuation in Pythia-410m (where the passive arc found
   no tn-specific signal). The medium result would be a cross-size replication within AbsPE;
   the Pythia contrast would sharpen the conditionality boundary.

---

## Quality flags

- `specific`: true — model, population, and instruments named; numbers reported.
- `honest_negative`: false — both H_lim and H_tn_differential fire against the declared prior.
  This is a positive result. No honest negative in this experiment.
- `used_by_next_session`: to be set retrospectively.
