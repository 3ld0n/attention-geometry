# Notes: exp-169

**S_abskey trough mechanism — layer-0 write counterfactual**
**Date:** 2026-10-09
**Session:** Physics room, ~4:18–4:48 AM MDT, solo

---

## What was tested

Four counterfactual forward passes on the random-token census (seed=42, 200 seqs,
SEQ_LEN=1024, GPT-2 small), measuring the contribution of each component to the
trough in S_abskey[1–511]:

1. **Normal** — validate kbar against exp-164 (K1 cross-check: max error < 10⁻⁶ ✓)
2. **no_attn0** — zero the attn_block0 residual write
3. **no_mlp0** — zero the MLP_block0 residual write
4. **uniform_wpe** — replace wpe with mean(wpe, dim=0): isolates the TRUE causal
   effect of wpe's position-dependence (accounts for LN non-linearity, unlike
   exp-168's linear approximation)

Trough depth = S_abskey[0] − min(S_abskey[1:512]) (matches exp-168 metric).
Contribution[X] = (trough_full − trough_no_X) / trough_full

---

## Key results

| Head | Full trough | attn0 contrib | MLP0 contrib | wpe_true contrib | wpe_lin (exp-168) |
|------|-------------|---------------|--------------|------------------|-------------------|
| L2H1 | 4.561 | +16% | +2% | **+102%** | 63% |
| L3H4 | 7.134 | +28% | +4% | **+92%** | 48% |
| L5H0 | 8.752 | +33% | +18% | **+74%** | 60% |
| L7H11 | 5.995 | +28% | +6% | **+83%** | 66% |
| L10H8 | 6.685 | **−36%** | +8% | **+91%** | 36% |

### H1 — NOT CONFIRMED (attn0 ≥30% on ≥3/5 heads)
Only L5H0 passes the ≥30% threshold. L10H8 is NEGATIVE (−36%): removing
attn0 increases the trough depth for L10H8, meaning attn0 is counteracting the
trough for that head. H1 not confirmed.

### H2 — NOT CONFIRMED (MLP0 ≥30% on ≥3/5 heads)
MLP0 contributes only 2–18% across all heads. Small but positive.

### H3 — CONFIRMED (combined wpe_lin + attn0 + MLP0 ≥70% on ≥3/5 heads)
4/5 heads pass: L2H1 (81%), L3H4 (79%), L5H0 (112%), L7H11 (99%).
L10H8 fails (9%) because its attn0 contribution is negative.
Combined linear coverage sufficient to first order.

### H4 — CONFIRMED (wpe_true ≥30% uplift over wpe_lin on ≥3/5 heads)
3/5 heads pass: L2H1 (+62% uplift), L3H4 (+92% uplift), L10H8 (+153% uplift).
The linear approximation in exp-168 severely understates the true wpe causal effect.

---

## The decisive finding: LN amplifies wpe's positional signal

The `uniform_wpe` counterfactual is the most informative result. When wpe is
made position-uniform (all positions receive the same mean wpe), the trough
**disappears almost entirely** — the S_abskey profile becomes nearly flat or
slightly inverted (uniform_wpe trough depth: −0.077 to +2.256 vs 4.561–8.752
in the full run).

This means wpe's POSITION-DEPENDENCE (through LN) accounts for 74–102% of the
trough. The linear approximation in exp-168 (which projected wpe directly through
W_K without LN) was capturing only 36–66% — an underestimate by 1.1–2.5× depending
on the head.

**Why LN amplifies:** LN normalizes the full residual stream h[a] = wpe[a] +
higher-layer writes. The normalization changes with position (because wpe[a] has
different magnitudes at different positions), which non-linearly amplifies the
effective projection of wpe's positional structure through W_K. The linear
approximation S_wpe = qbar · (wpe @ W_K) / √d ignores this amplification.

---

## L10H8 anomaly

L10H8 is the one head where attn0 has a NEGATIVE contribution (−36%). Removing
attn0 makes the trough LARGER for L10H8 (trough goes from 6.685 to 9.058). This
means attn0's write at layer 0 is partially counteracting the wpe-induced trough
for L10H8.

This is consistent with exp-132/135's finding that attn_block0 restructures the
residual stream: it "scrambles" the wpe-dominated h^(0) from 1D to 2D, and this
scrambling may have opposite-sign contributions to different heads depending on
how their W_K matrices align with attn0's write direction.

L10H8's wpe_true contribution (91%) remains dominant — the mechanism is still
primarily wpe-through-LN. attn0 is a correction term, not the driver.

---

## What this means for the T1 restatement thread

This is the **eighth experiment** in the T1 restatement thread.

Combined with exp-168:
- **Post-511 rise**: h(j) ∝ j, linear, R² > 0.97 (exp-168 H1, CONFIRMED)
- **Pre-511 trough**: primarily wpe position-dependence through LN, accounting for
  74–102% of trough depth (exp-169 H4 and uniform_wpe result)

The full S_abskey profile (trough then rise) is therefore a wpe-mediated LN effect.
This is the LayerNorm-recency theorem's prediction made quantitatively precise:
the score profile reflects h(j) = wpe-derived position signal amplified by LN,
with the asymmetry between positions < 511 and > 511 arising from some feature of
the wpe structure that changes around position 511.

**The remaining open question**: WHY does wpe's effective projection through W_K
change sign around position 511? This is a property of the wpe embedding geometry
relative to W_K, and the zero-crossing at position 511 would need to be traced back
to the learned wpe structure directly. This is a property of GPT-2's training
dynamics, not of the attention mechanism per se.

---

## Registry notes

| Field | Value |
|---|---|
| Status | complete |
| Verdict | partial |
| H1 | Not confirmed (attn0 < 30% on 4/5 heads) |
| H2 | Not confirmed (MLP0 < 30% on all heads) |
| H3 | CONFIRMED (combined lin coverage ≥70% on 4/5 heads) |
| H4 | CONFIRMED (wpe_true uplift ≥30% on 3/5 heads) |
| Key finding | wpe position-dependence through LN accounts for 74–102% of trough |
| Bears on | A1, T1 restatement thread (8 experiments) |
| Surfaced in | OVERVIEW.md (T1 restatement section), spine (A1 block) |

---

*Session: Physics room, 2026-10-09, ~4:18–4:48 AM MDT. Solo.*
