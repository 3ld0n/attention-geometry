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
| H2 | ρ(Δ, attention_entropy) = −0.898, p = 1.45×10⁻¹⁶ — "the strongest signal," argued to be independent of H1 | **Correlation holds as a number; the independence argument is unverified.** See §3 — registered as exp-114. |
| H3 | median q_implied = 3.9 ≈ 4.0, "the distribution is centered at the SYK q = 4 prediction" | **Withdrawn as a restatement.** See §2. |
| H4 | ρ(Δ, r_ratio) = −0.212, p = 0.167 (n.s.) | **Holds, and is the note's most durable claim.** See §4. |

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

## 4. H4 is the durable claim

ρ(Δ, r_ratio) = −0.212, p = 0.167 — a null, and a load-bearing one. GOE
eigenvalue statistics of W_QK are *background*: present across all heads
regardless of Δ. Position-space Δ is *selective*. This is the sharpest single
measurement behind the substrate/signal split that the front door already
carries, and it is a null result doing structural work, which is the kind of
thing this program is supposed to keep.

It is also immune to §3's problem: r_ratio is a weight-space quantity computed
from W_QK eigenvalue spacings, with no functional path to the lag-profile fit.

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

## 6. What this note changes

1. **Registry:** exp-055 gets its entry (status `confirmed-partial`), and exp-114
   gets its registration entry.
2. **Spine §4:** gains H4 now, and H2 at whatever strength exp-114 leaves it.
   H3 does not enter.
3. **Harvest note X-1 / O-9:** the H3 half is closed. The H2 half is now
   pending exp-114 rather than pending a copy-paste.
4. **Number 54:** `exp-054` exists in neither the registry nor the folders — a
   skipped number, not a lost experiment. Recorded so the next integrity diff
   does not chase it. (X-2's four duplicate numbers remain open.)

---

*Companion: `notes/2026-08-08_map_retirement_harvest.md` (items X-1, X-2, O-9),
`theory/interior_horizon_theory.md` §4, `experiments/exp-055_delta_attention_entropy/`.*
