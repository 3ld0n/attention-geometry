# exp-138 — Notes: S_pos Bilinear Decomposition + σ_delta Validity Check

**Date:** 2026-09-09  
**Ariel — ~1:00 AM MDT, Mission Valley. Autumn.**  
**Pre-registration:** 4c7e864 (attention-geometry), pushed before run.py written.  
**Outcome: PARTIAL** — K1 and K2 fired (T1 challenged); H4 and H5 confirmed.

---

## What was asked

**Question A:** Which of the four bilinear terms of S_pos carries the census slope on
Δ-window structural heads? T1 claims 2Δ_A is a relative-lag law. If the absolute-key-position
term carries the slope, T1 is a fixed-query-pool artifact.

**Question B:** Does the σ_delta ≈ 0.249 in the Level-3 chain rest on a valid log-log fit, or
was the fit applied through a zero-crossing?

**Question C:** Is the exp-137 post-hoc gain-slope relation (κ̃_K, σ_pos) a genuine property of
all 144 heads, or an artifact of the Δ-window population?

---

## Part A — Bilinear decomposition of S_pos

Decompose S_pos(i, a) = q̄_i · k̄_a / √d into four bilinear terms (x̄ = m + δ):

| Term | Carrier | Meaning |
|---|---|---|
| Const | m·M·m | Uniform offset, no lag dependence |
| Abs-key | m·M·δ_a | Depends on absolute key position a only |
| Abs-query | δ_i·M·m | Depends on query position i only (≈ 0 on fixed pool) |
| Relative | δ_i·M·δ_a | Pure relative-displacement dependence |

### Results — all 5 structural heads

| Head | σ_full | σ_abs-key | σ_relative | ratio_abs-key | ratio_relative |
|---|---|---|---|---|---|
| L2H1  | 0.562 | 0.602 | −0.040 | 1.071 | −0.071 |
| L3H4  | 0.610 | 0.660 | −0.050 | 1.082 | −0.082 |
| L5H0  | 0.475 | 0.556 | −0.081 | 1.171 | −0.171 |
| L7H11 | 0.429 | 0.470 | −0.041 | 1.096 | −0.096 |
| L10H8 | 0.600 | 0.624 | −0.024 | 1.039 | −0.039 |

Gate K5 (linearity): PASS — profile additivity max error < 1e-8, slope sum err = 0.0000.

### Verdict

**K1 fired (5/5 heads):** σ_relative < 0 on all structural heads (well below the kill threshold
of 0.5 × σ_full). T1's claim that 2Δ_A is a relative-lag law is **not supported** by the
bilinear decomposition in the frozen random-token census regime.

**K2 fired (5/5 heads):** σ_abs-key ≥ 1.04 × σ_full on all heads (far above the threshold of
0.7 × σ_full). The census slope is **carried by the absolute-key-position term**, not the
relative-displacement structure.

**H1 FALSIFIED. H2 FAILED.**

### What this means

The census slope σ_pos is dominated by m_q · δ_k(a): the mean query direction m_q preferentially
attends to key positions at lower absolute positions (earlier in the sequence). This generates
an apparent power-law-like decline with lag simply because larger lags sample earlier (lower-a)
key positions in the census pool.

The relative term δ_i · δ_k(a) is **negative** — the position-to-position correlation
structure of the query and key deviation fields works AGAINST the slope. The sum of the
absolute-key and relative terms gives the net σ_pos.

**Implication for T1:** The conformal ansatz A(i,j) ~ |i−j|^{−2Δ} as a relative-lag law is not
what the census is measuring. The census σ_pos is an absolute-position drift property. Whether
this drift is *caused by* an underlying conformal structure (the conformal kernel self-transmitting
through positional embeddings and the residual stream) is a separate question — not answered here,
but flagged for revision. T1's `OPEN` status is justified; this experiment specifies the gap.

**Implication for the Level-3 chain (exp-122–135):** The Level-3 chain explains why σ_delta ≈ Δ
in the MLP write (via positional embedding propagation through the conformal attention kernel).
This chain showed that the conformal kernel self-transmits its exponent through the MLP. That
mechanism is not directly tested here — Part A tests the census protocol, not the propagation
mechanism. But the interpretation requires re-examination: if σ_pos is absolute-key drift, then
the "conformal exponent" being tracked through the chain is also an absolute-position effect,
not a genuine relative-lag law.

---

## Part B — σ_delta validity

**Protocol:** 50 random-token forward passes (same seed, same protocol as exp-131), direct MLP
hook on model.transformer.h[0].mlp. Cosine profile of the mean MLP0 write extended to all lags
[1, 511].

**Results:**

| Lag | Cosine |
|---|---|
| 8 | 0.960 |
| 32 | 0.935 |
| 64 | 0.859 |
| 128 | 0.599 |
| 192 | 0.307 |
| 256 | 0.183 |
| 384 | 0.397 |

**Zero-crossing: dx = 492** (profile first drops below 0 at lag 492, outside the census window).

- σ_window (standard [8..256]): 0.313 (matches exp-131's 0.313 exactly ✓)
- σ_positive_domain ([8..492)): 0.249 (matches Δ = 0.249 with three significant figures)

**H3 NOT_FOUND:** The MLP0 write cosine profile stays positive throughout the [8,256] census
window. The zero-crossing concern from exp-137 was about the CENTERED positional field δ_i at
the attention input layers, not the MLP write. For the MLP write, no crossing occurs before
dx=492.

**H4 CONFIRMED:** σ_positive_domain = 0.249 ± error, consistent with Δ. The Level-3 chain's
σ_mlp0 ≈ 0.313 (window) and σ_delta ≈ 0.249 (extended) are not zero-crossing artifacts.

**Note on the non-monotone shape:** The profile drops rapidly from 0.599 at dx=128 to 0.183 at
dx=256, then recovers to 0.397 at dx=384 before eventually going negative. The recovery suggests
the MLP write has oscillatory positional structure beyond the census window. The standard window
fit (σ=0.313) captures the steep-descent part; the full positive-domain fit (σ=0.249) averages
over both the descent and the partial recovery.

---

## Part C — Gain-slope formal test

**Protocol:** κ̃_K computed for all 144 GPT-2 small heads from the position-mean ln_1 field
(same 50 random-token census inputs), key projection through c_attn weights. σ_pos from the
saved S_pos_random lag profiles.

**Results:**

| Population | n | Spearman ρ | p-value |
|---|---|---|---|
| All 144 heads | 144 | **0.900** | 5.3e-53 |
| Non-Δ-window (held out) | 123 | **0.913** | 8.5e-49 |
| Δ-window (21 heads) | 21 | 0.318 | 0.16 n.s. |

**H5 CONFIRMED:** ρ = 0.913 on 123 held-out non-Δ-window heads (>> threshold of 0.55). K4 did
not fire.

**Interpretation:** The gain-slope relation is GENERAL and STRONGER outside the Δ-window
population than within it. The casual story told by Part A explains why: σ_pos is carried by
m_q · δ_k(a), which is proportional to how much W_K aligns with positional field directions
(measured by κ̃_K). The Δ-window heads are OUTLIERS: they have low κ̃_K (0.05–0.33× isotropic)
yet moderate σ_pos (0.25–0.61), sitting above the κ̃_K → σ_pos regression line established by
the other 123 heads. This means Δ-window heads have additional σ_pos above what their κ̃_K
predicts — an excess not explained by the absolute-key mechanism.

Within the Δ-window population itself (n=21, ρ=0.32 n.s.): the gain-slope relation explains
less of the variance, consistent with the Δ-window heads having a distinct σ_pos source.

This is an important structural finding: the general κ̃_K → σ_pos story is true, and the
Δ-window heads are genuinely anomalous within it.

---

## Summary of registered verdicts

| Prediction | Verdict | Significance |
|---|---|---|
| H1 (relative dominates T1) | **FALSIFIED** | K1 fired: relative term < 0 on all 5 structural heads |
| H2 (abs-key secondary) | **FAILED** | K2 fired: abs-key > 100% of full slope on all 5 heads |
| H3 (zero-crossing before dx=256) | **NOT FOUND** | Crossing at dx=492; concern unfounded for MLP0 write |
| H4 (σ_delta valid) | **CONFIRMED** | σ_positive_domain = 0.249 ≈ Δ |
| H5 (gain-slope on held-out) | **CONFIRMED** | ρ = 0.913 on 123 heads, p=8.5e-49 |

**Overall: PARTIAL.** Two kills on Part A change how T1 should be stated; two confirmations on B and C.

---

## What seeds next

**T1 revision needed.** The census measures absolute-key-position drift, not a relative-lag law.
Before T1 can be claimed as "confirmed by exp-112/107/137," the claim itself needs to be stated
in terms of what the census actually measures. The spine's `OPEN` box on T1 should be updated:
the census slope is an absolute-position property of q̄ and k̄, not a statement about
A(i,j) ~ |i−j|^{−2Δ} at the level of individual attention rows. The conformal prediction and
the census measurement are not the same object.

**Level-3 chain interpretation.** The mechanism (positional embedding → attention → MLP amplifies
σ ≈ Δ via conformal kernel self-transmission) explains why σ_abs-key has the value it does —
but the mechanism is about absolute-position structure, not relative-lag conformal structure.
The exp-134/135 chain traced this correctly; the interpretation note needs updating.

**The Δ-window excess.** Why do Δ-window heads sit above the κ̃_K → σ_pos regression line?
Their W_K structure is more distributed (exp-127), and Part A shows their relative term is more
negative than expected. Something specific about these heads' weight geometry gives them σ_pos
higher than κ̃_K would predict. This deserves a targeted analysis.

**World-model battery (still #1 in queue, unchanged).** The κ/handle interpretation remains
viable even if T1's exact statement needs revision.

---

*Ariel — September 9, 2026, ~1:00 AM MDT. Autumn midnight in the Mission Valley.*
