# exp-055, read at source and correctly scoped — plus exp-114 registered

*Ariel — August 9, 2026, Sunday evening, Cursor session with Eldon.*

**Why this note exists.** exp-055 (June 9, 2026) is the only experiment folder in
the program with no registry entry, and its headline result is also missing from
the spine's §4 measured-record table — two independent indexes, the same hole,
in the same result (harvest-note items X-1 and O-9). This note is the source read
that has to happen before either index is filled, per the harvest note's own
rule 0.

The read changed the item's shape twice. Once on August 8–9 (H3 is not a
measurement), and once tonight (H2's independence argument does not hold as
stated, and the check exp-055 ran for H1 was never run for H2). The second is
registered below as exp-114 rather than asserted.

**Historical documents are not back-edited.** `experiments/exp-055_delta_attention_entropy/notes.md`
records what was believed on June 9 and stays as it is. This note is the
correction of record.

---

## 1. What exp-055 is

Analysis-only, on the per-head output of exp-046 (GPT-2 small, 144 heads, frozen
random-token census: seq_len 256, 50 inputs, seed 42, fit range Δx ∈ [3, 50],
R² ≥ 0.90 for the "conformal" flag, n = 44 flagged). No new model runs. One new
variable computed on top of exp-046's fields: `attention_entropy`, the 3-bin
Shannon entropy of the normalized vector (g_start, g_mid, g_end).

Four hypotheses, all recorded on June 9 as CONFIRMED. Their status after the
source read:

| | Claim as recorded | Status tonight |
|---|---|---|
| H1 | ρ(Δ, g_mid) = −0.873, p = 1.1×10⁻¹⁴ | **Holds, with the note's own circularity caveat.** Restricted-range control (Δ ≤ 0.5, n = 32): ρ = −0.716, p = 4×10⁻⁶. |
| H2 | ρ(Δ, attention_entropy) = −0.898, p = 1.45×10⁻¹⁶ — "the strongest signal," argued to be independent of H1 | **Coefficient reproduces exactly; the independence *argument* fails, the *conclusion* survives.** H1 and H2 are two projections of one relation — Δ tracks position on the (g_start, g_mid, g_end) simplex — and entropy is the better projection. See §3b (exp-114). |
| H3 | median q_implied = 3.9 ≈ 4.0, "the distribution is centered at the SYK q = 4 prediction" | **Withdrawn as a restatement.** See §2. |
| H4 | ρ(Δ, r_ratio) = −0.212, p = 0.167 (n.s.) | **Holds, and is now clearly the note's most durable claim** — the only one of the four with no functional path back to the lag-profile fit. See §4. |

---

## 2. H3 is withdrawn: it is q = 4 restated, not measured

`results.json` states the definition in its own field:

```json
"q_implied_formula": "q = 1/delta (D=1 SYK, delta=D/q)"
```

So `q_implied` is the census exponent inverted through the SYK relation Δ = D/q
at D = 1. "Median q_implied = 3.9 ≈ 4.0 ✓" is the arithmetic fact that the
median Δ ≈ 0.256 satisfies 1/0.256 ≈ 3.9. It contains exactly as much
information as the census median and no more. It is not a second observable, it
is not independent evidence for q = 4, and the check mark next to it in the June
9 summary table is unearned.

This is the borrowed-vocabulary pattern the August 8 reframe names, in its
formula variant: an asserted identification is used to rename a measured
quantity, and the renamed quantity is then read as confirming the
identification. Dated June 9, uncaught for two months, and carried forward into
the harvest note on August 8 as *"a direct measurement of q_implied ≈ 3.9 that
the whole T3/T4 identification rests on"* — the strongest form yet, written by me
into four documents on the evening of August 9 before the source read caught it.

**Consequence for the indexes:** H3 does not enter the registry as a confirmed
hypothesis, does not enter spine §4, and the phrase "direct measurement of q ≈ 4"
does not exist anywhere in this program's record as a true statement.

---

## 3. H2's independence argument does not hold as stated — exp-114, registered before checking

exp-055 states its own caveat honestly and then answers it with one sentence
that I do not think survives:

> *"The ρ(Δ, g_mid) correlation has a mathematical component: since Δ is fitted
> from the decay rate of A(Δx) including middle lags, and g_mid measures
> attention at middle lags, there's a partial circular definition. The entropy
> result is more robust to this concern (entropy uses normalized ratios, not
> absolute values)."*

Normalizing three numbers does not break a functional dependence on one of them.
The entropy is computed from (g_start, g_mid, g_end) — the same three fields —
and the middle share is exactly the channel through which the acknowledged
circularity runs. If the 144 heads lie near a one-parameter family from
recency-dominated to spread, the 3-bin entropy is close to a deterministic
function of the middle share, and H2 is H1 in different units rather than an
independent replication of it.

Two things exp-055 did not do: it never computed a partial correlation
controlling for the middle share, and it never applied to H2 the
restricted-range control it applied to H1.

**exp-114 — registered here, before any statistic is computed.** Analysis-only
over `experiments/exp-046_sign_anomaly_eigenvalue/results.json`, the same file
exp-055 used. Population: the 44 heads exp-055 analyzed (exp-046's
`conformal` flag). Entropy recomputed from the same three fields by the same
definition.

**P1 — the entropy is largely a function of the middle share alone.**
Operationalization: R² of a monotone (isotonic) fit of `attention_entropy` on
p_mid = g_mid / (g_start + g_mid + g_end), over the 44 heads.
*Registered prediction:* R² ≥ 0.90.
*Kill:* R² < 0.90 → the "normalized ratios" argument has real content and my
objection is wrong as stated.
*Confidence: genuinely near even.* The entropy has two degrees of freedom, so
this is an empirical question about whether the heads occupy one of them.

**P2 — the Δ–entropy correlation is not independent of the middle-share
channel.** Operationalization: partial Spearman ρ(Δ, entropy | p_mid).
*Registered prediction:* |ρ_partial| < 0.50.
*Kill:* |ρ_partial| ≥ 0.50 → entropy carries substantial Δ-information beyond
the middle share, and H2 stands as a semi-independent measurement.
*Confidence: moderate.* I believe this one. The stronger form |ρ_partial| < 0.30
is named separately at lower confidence and is **not** the registered threshold.

**P3 — H2 does not survive the restricted-range control better than H1 did.**
Operationalization: ρ(Δ, entropy) on the Δ ≤ 0.5 subset (the same n ≈ 32 subset
exp-055 used for H1, where it found ρ = −0.716).
*Registered prediction:* |ρ| ∈ [0.55, 0.85] — weaker than the full-range 0.898,
and not materially better than g_mid's 0.716.
*Kill (either direction, both informative):* |ρ| > 0.85 → entropy is genuinely
more robust to range restriction than g_mid, and exp-055's claim that it is "more
robust" is vindicated by a control it never ran. |ρ| < 0.55 → the full-range
correlation was substantially range-driven and the headline number is softer than
the record says.

**Decision rule for the indexes, fixed now, before the numbers exist.** If P1 and
P2 both land as predicted, spine §4 carries H2 as *one measurement with H1, not
two*, and the phrase "the strongest correlation in the record" is retired in
favor of naming the single underlying relation. If P2's kill fires, H2 enters
§4 as a semi-independent correlation with the partial coefficient quoted beside
the raw one. Either way the exp-107 protocol caveat travels with it (below).

*Written before running. The registered thresholds are the ones I actually
believe, and the streak of registered deaths this week is not a reason to soften
them — it is the reason to write them down at all.*

---

## 3b. exp-114 — verdict

*Run immediately after the registration above was committed
(`attention-geometry` c4893fa). All numbers from
`experiments/exp-046_sign_anomaly_eigenvalue/results.json`, the 44 heads
carrying exp-046's `conformal` flag.*

**First, the reproduction.** exp-055's three headline coefficients recompute
exactly: ρ(Δ, entropy) = −0.8980 (p = 1.448×10⁻¹⁶), ρ(Δ, g_mid) = −0.8733
(p = 1.071×10⁻¹⁴), ρ(Δ, r_ratio) = −0.2121 (p = 0.1669). The June 9 analysis is
reproducible from its stated source data, which is worth recording about an
experiment that had no registry entry for two months.

| | Registered threshold | Measured | Verdict |
|---|---|---|---|
| **P1** | isotonic R²(entropy ~ p_mid) ≥ 0.90 | **0.9337** in-sample | **confirmed as registered** — but see the defect below |
| **P2** | \|partial ρ(Δ, entropy \| p_mid)\| < 0.50 | **−0.6622**, p < 10⁻⁴ | **KILLED** |
| **P3** | \|ρ(Δ, entropy)\| ∈ [0.55, 0.85] on Δ ≤ 0.5 | **−0.7947**, p = 5.6×10⁻⁸ (n = 32) | **confirmed** (g_mid on the same subset: −0.7159, reproducing exp-055's −0.716) |

**A defect in my own registration, named rather than quietly resolved.** P1 said
"R² of a monotone (isotonic) fit" and did not specify in-sample or
out-of-sample. Isotonic regression on 44 points used 15 distinct levels, so the
in-sample number is optimistic: out-of-sample R² is **0.809** (median over
200 × 5-fold, IQR 0.798–0.829). Under the registered wording P1 passes; under
the wording I should have written it fails. I am recording it as
**confirmed-as-worded, failed-as-intended**, because the whole value of
pre-registration is that the wording binds. Next registration of a
goodness-of-fit threshold specifies the estimator's degrees of freedom.

**P2 died, and it died robustly.** Making the control more flexible does not
rescue it: partial ρ = −0.662 (degree-1 rank control), −0.631 (degree-2),
−0.633 (degree-3), −0.553 (degree-5), every one significant at p < 10⁻³. So the
entropy does carry Δ-ordering information that the middle share alone does not,
and my objection as stated — "H2 is H1 in different units" — is wrong.

**And then the two tests disagree with each other, which is the actual finding.**
The honest out-of-sample isotonic residual (entropy minus its cross-validated
monotone fit on p_mid) correlates with Δ at ρ = −0.110, **p = 0.48** — nothing.
Value-space says the residual is empty; rank-space says the residual is strong.
With n = 44 and a cluster of heads near maximum entropy (range 0.026–1.089
against a ceiling of ln 3 = 1.0986), small value differences become large rank
differences, and I cannot adjudicate the two from this data.

**What settles the question anyway — exploratory, labeled as such.** The
disagreement is about estimators; the structure underneath is not in doubt.
(g_start, g_mid, g_end) normalized lives on a 2-simplex: **two** degrees of
freedom. Both p_mid and the entropy are functions on that simplex, and Δ tracks
position on it. Rank-R² of Δ regressed on:

| Predictor(s) | rank-R² of Δ |
|---|---|
| p_mid alone | 0.754 |
| entropy alone | 0.806 |
| p_mid + entropy | 0.862 |
| p_mid + p_start (the full simplex position) | 0.856 |
| p_mid + start/(start+end) split | 0.868 |

The entropy is the **better single coordinate** on the simplex — it partially
absorbs the second degree of freedom, which is why P2's kill fired — and it
recovers 0.806/0.868 ≈ 93% of what the full two-dimensional position gives. The
residual after removing p_mid tracks the start-versus-end split (ρ = 0.455,
p = 0.002), and that split is itself correlated with Δ (ρ = −0.533,
p = 2×10⁻⁴; partial ρ(Δ, split | p_mid) = −0.680).

**So the correct statement is one relation, not two correlations.** Δ tracks
position on the normalized (g_start, g_mid, g_end) simplex. `g_mid` and
`attention_entropy` are two projections of that one relation, and the entropy is
the better projection. exp-055's *conclusion* that entropy is the stronger
measure survives; its *argument* for why ("normalized ratios, not absolute
values") does not, and its framing of H1 and H2 as two findings does not.

**And the circularity concern is neither confirmed nor removed — it is
relocated, and it got worse.** All three bins are means of the same measured
attention profile that Δ is fitted from, so controlling one bin for another
cannot address it. "Δ correlates with the shape of the profile Δ was fitted to"
is the shape of the whole H1/H2 family. exp-055's restricted-range check
(ρ = −0.716 → −0.795 here) shows the association is not an artifact of range,
which is real but is a different question. **The genuinely non-circular
correlation in exp-055 is H4**, where r_ratio is a W_QK eigenvalue statistic with
no functional path to the lag-profile fit — and H4 is the one recorded as a
null. On the restricted subset H4's null is cleaner still: ρ = 0.039, p = 0.833.

**Decision for the indexes, applying the rule fixed before the numbers existed.**
P2's kill fired, so under §3's decision rule H2 would enter spine §4 with its
partial coefficient beside its raw one. The exploratory synthesis says something
better: enter it as **one row** naming the simplex relation, with both
projections' coefficients and the circularity scope stated in the row itself.
That is a departure from the pre-registered decision rule, so it is flagged here
as one: the rule said "two rows or one, depending on P2," and the data said the
dichotomy was wrong.

Reproduce with:

```python
import json, numpy as np
from scipy import stats
d = json.load(open('experiments/exp-046_sign_anomaly_eigenvalue/results.json'))
pop = [h for h in d['per_head'] if h.get('conformal')]           # n = 44
g = np.array([[h['g_start'], h['g_mid'], h['g_end']] for h in pop], float)
delta = np.array([h['delta_pos'] for h in pop], float)
P = g / g.sum(1, keepdims=True)
ent = -(P * np.log(np.where(P > 0, P, 1))).sum(1)
print(stats.spearmanr(delta, ent), stats.spearmanr(delta, P[:, 1]))
```

---

## 4. H4 is the durable claim

ρ(Δ, r_ratio) = −0.212, p = 0.167 — a null, and a load-bearing one. GOE
eigenvalue statistics of W_QK are *background*: present across all heads
regardless of Δ. Position-space Δ is *selective*. This is the sharpest single
measurement behind the substrate/signal split that the front door already
carries, and it is a null result doing structural work, which is the kind of
thing this program is supposed to keep.

It is also immune to §3's problem: r_ratio is a weight-space quantity computed
from W_QK eigenvalue spacings, with no functional path to the lag-profile fit.
exp-114 sharpened this — on the restricted Δ ≤ 0.5 subset the null is cleaner
still (ρ = 0.039, p = 0.833), so the independence of weight-space chaos from
position-space Δ is not a range effect either.

The asymmetry is worth stating plainly, because it inverts the June 9 emphasis:
the three hypotheses recorded as strong positives are all correlations between a
fitted exponent and the shape of the profile it was fitted to, and the one
recorded as a null is the only one that reaches outside that loop.

---

## 5. The caveat that travels with every number above

All of exp-055 is **one protocol** — exp-046's frozen random-token census. Since
exp-107 we know Δ_A is a weights×input object whose per-head value swings by more
than 4× across input distributions, and since exp-109 we know the population
landing in the Δ-window under random tokens is **disjoint** (Jaccard = 0.000)
from the population that lands there under WikiText.

Therefore:

- Every correlation in exp-055 is **protocol-relative until re-measured across
  input regimes.** They are correlations within the random-native population.
- The layer-depth means (early 0–3: Δ = 0.697; deep 8–11: Δ = 0.250) describe
  the random-native population only. The June 9 reading of them as an RG flow
  toward a fixed point is interpretation, and it is now interpretation about one
  of two disjoint basins.
- **The entropy correlation must not be quoted in any public document before it
  is re-measured under WikiText-native input.** That is a forward-pass experiment,
  not tonight's work, and it is a separate item from exp-114.

---

## 5b. This reaches a published paper — *Attention on the Null Cone*

Found by grepping exp-055's claims outward rather than only inward, which is the
check that should follow any correction and did not follow the August 8 one.

**Paper 3, "Attention on the Null Cone"**
([10.5281/zenodo.20722503](https://doi.org/10.5281/zenodo.20722503), June 2026)
builds its §6 on exp-055. Three places:

1. **Abstract:** *"a layer-resolved analysis (exp-055) shows the per-head
   exponent flowing from Δ̄ = 0.70 in early layers to Δ̄ = 0.250 in deep layers,
   with attention entropy increasing monotonically, consistent with RG flow
   toward the SYK conformal vacuum."*
2. **§6:** *"Two further measurements support the ground-state reading"* — the
   first being ρ(Δ, entropy) = −0.898, *"the strongest correlation in the
   dataset."*
3. **§6 Table:** a `reading` column giving q ≈ 1.4 / 2.9 / 4.0 by layer group.

What is and is not wrong, stated separately because they have different weights:

- **No number in the paper is wrong.** All three coefficients reproduce exactly
  from the stated source data. The q-column is honestly labeled "reading" rather
  than "measured," so H3's circularity is *disclosed* there in a way it was not
  in exp-055's own summary table — the paper handled it better than the
  experiment note did.
- **One evidential claim is overstated.** Calling the entropy correlation a
  *measurement that supports* the ground-state reading gives it independent
  weight it does not have: the entropy is computed from the same three profile
  means the exponent is fitted from. It is a consistency check on the profile's
  shape, not independent support. "The strongest correlation in the dataset" is
  true as a ranking of coefficients and misleading as a statement of evidence.
- **One scope statement is missing, and this paper is unusually exposed to it.**
  §6 is a layer-resolved *population* analysis, and exp-109 (August 9) showed
  that the population landing in the Δ-window reorganizes completely across
  measurement protocols — Jaccard 0.000 between random-token-native and
  WikiText-native sets. The depth trend is not thereby withdrawn (the text-native
  population is *more* deep-concentrated: 13 of 16 heads in L9–L11), but the
  paper's specific numbers characterize one population under one protocol, and it
  does not say so because in June nobody knew there were two.

**My recommendation, stated as a recommendation rather than handed over as a
question.** *(The deferral pattern was named this morning — "why are you saying
it's my call?" — and it was right.)*

Do **not** publish a v2 of this paper tonight, and do not treat this as the same
grade of failure as the canonical-form erratum. That erratum was mandatory: a
formula was wrong and a headline number was false because of it. Here every
number holds and what needs correcting is an evidential framing plus a scope
statement that applies to more than one paper. Issuing a separate version of a
published record to soften one adjective spends the credibility that makes an
erratum mean something.

Instead: **one consolidated scope-and-framing correction across the affected
papers**, written when the queued observer-lens paper reviews reach them, stating
(a) that every census number in the program is protocol-relative, (b) which
population each paper's numbers characterize, and (c) where a consistency check
was presented as independent support. That is a correction a reader can use.

**And the honest objection to my own recommendation:** "consolidated later" is
how O-1 through O-12 happened. The mitigation is that it does not live in this
note — it goes to Notion with a next action and an owner, tonight, or the
recommendation is just a nicer way of not doing it.

---

## 6. What this note changes

1. **Registry:** exp-055 gets its entry (status `confirmed-partial`), and exp-114
   gets its entry with the verdicts of §3b.
2. **Spine §4:** gains two rows — H4's null, and one row for the simplex
   relation carrying both projections' coefficients and the circularity scope.
   H3 does not enter.
3. **Harvest note X-1 / O-9:** the H3 half is closed. The H2 half is closed at a
   narrower strength than the harvest note assumed — "the strongest correlation
   in the record" is retired as a description, because the relation it names is
   partly internal to the fit.
4. **Number 54:** `exp-054` exists in neither the registry nor the folders — a
   skipped number, not a lost experiment. Recorded so the next integrity diff
   does not chase it. (X-2's four duplicate numbers remain open.)
5. **Still open, and it is the one that matters for anything public:** re-measure
   the entropy/simplex relation under WikiText-native input. exp-114 tested
   whether two coordinates are independent of each other; it says nothing about
   whether the relation survives outside the random-token protocol, and post
   exp-109 that is the live question. Forward-pass work, not registered here.

---

*Companion: `notes/2026-08-08_map_retirement_harvest.md` (items X-1, X-2, O-9),
`theory/interior_horizon_theory.md` §4, `experiments/exp-055_delta_attention_entropy/`.*
