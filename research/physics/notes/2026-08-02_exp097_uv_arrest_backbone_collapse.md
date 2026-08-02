# exp-097 Partial Analysis: UV Arrest and Backbone Collapse (2026-08-02)

**Session:** 2026-08-02 afternoon physics room session. Verdict pending (s0 retrain in progress).
**Experiment:** exp-097 — C-alien alien semantics corpus.
**Two-seed preliminary data:** s1 (1701), s2 (1702) fully measured. s0 (1700) retrain running.

---

## What the 2-seed data shows

| Seed | n_conf | n_syk | Δ_med | n_deep (L3–L5) | n_backbone (L0) |
|------|--------|-------|-------|-----------------|-----------------|
| 1701 (s1) | 10/48 | 0 | 1.044 | 3 | 0 |
| 1702 (s2) | 17/48 | 0 | 1.185 | 3–7 (layer analysis: 7) | 0 |

Wait — rechecking s2 n_deep from the head-level parse: s2 has conformal heads at L3H1, L3H2, L3H5, L3H6, L3H7, L4H4, L5H6 = 7 deep heads (L3–L5).

| Seed | n_conf | n_syk | Δ_med | n_deep (L3–L5) | n_backbone (L0) |
|------|--------|-------|-------|-----------------|-----------------|
| 1701 (s1) | 10/48 | 0 | 1.044 | 3 | 0 |
| 1702 (s2) | 17/48 | 0 | 1.185 | 7 | 0 |

---

## Finding 1: UV arrest

All conformal heads in C-alien have Δ in range 0.33–1.76 (s1) / 0.37–1.76 (s2). None approach the SYK IR fixed point at Δ≈0.25. The median Δ is **≈1.0–1.2** — well into the UV regime.

For reference:
- C-NAT (natural text): Δ_med ≈ 0.166 at this 70m/1B-token scale; conformal heads near Δ≈0.25
- C-alien: Δ_med ≈ 1.04–1.19; conformal heads range 0.33–1.76

This is the strongest UV arrest observed in the series. Even the shuffled corpus (C-NAT-shuf, exp-091) with disrupted ordering has n_syk=0 at this scale, but its conformal heads are closer to the IR. C-alien's conformal heads are arrested ~4× further from the IR fixed point.

**Interpretation:** The C-alien model learns power-law attention structure (conformal) but the RG flow does not carry the heads toward the SYK fixed point. World-holding triggers conformal formation but not IR convergence. The distinction between "conformal" and "SYK-near" is meaningful and was already implicit in the pre-registration design (the primary observable was n_deep, not n_syk).

---

## Finding 2: Backbone collapse (L0 → trivial fixed point)

**Every prior corpus** in the series (C-NAT, C-NAT-shuf, C-PL, C-PCFG, all block-shuffled variants, C-generator) maintained 7–8 backbone heads in L0 (Δ≈0.05–0.35, n_conf/8 ≥ 6). Even the sentence-shuffled corpus (exp-091) preserved n_backbone=6–7 while reducing n_deep to 2.

**C-alien: n_backbone = 0 at both measured seeds.** L0 Δ values:
- s1: Δ range 0.0007–0.0146 (median 0.002). All 8 heads near the trivial fixed point.
- s2: Δ range 0.0005–0.0037 (median 0.001). Even more extreme trivial collapse.

The L0 heads are at Δ→0: **near-uniform attention** across the sequence. This is the trivial fixed point — attention that is approximately independent of position distance.

**This is not the same as the UV arrest.** UV arrest means conformal structure forms but Δ doesn't flow toward 0.25. L0 trivial collapse means no conformal structure forms at all in L0 — the attention is flat. These are two separate phenomena.

**Cross-series comparison:**
- exp-091 (sentence shuffle): n_backbone=6–7 preserved, n_deep=2 reduced
- exp-092 (k=2 block): n_backbone=7, n_deep=1
- exp-093 (half-story): n_backbone=7–8, n_deep=3
- exp-094 (quarter-story): n_backbone=7, n_deep=1
- C-alien (exp-097, 2 seeds): n_backbone=0, n_deep=3–7

The backbone (L0 conformal heads) persists through ALL ordering manipulations of natural text. It disappears only in C-alien. The key difference in C-alien: alien vocabulary (Vex/Nul/Ort/Dath/Sorn/Quib etc.) + highly repetitive structural template.

---

## Finding 3: Layer anatomy

C-alien layer structure is fundamentally different from all prior corpora:

| Layer | C-NAT (typical) | C-alien s1 | C-alien s2 |
|-------|----------------|------------|------------|
| L0    | n_conf=6–7, Δ≈0.25 (backbone) | n_conf=0, Δ≈0.002 (trivial) | n_conf=0, Δ≈0.001 (trivial) |
| L1    | n_conf=1–2     | n_conf=3, Δ_med≈0.46 | n_conf=5, Δ_med≈0.63 |
| L2    | n_conf=2–3     | n_conf=4, Δ_med≈0.92 | n_conf=5, Δ_med≈1.25 |
| L3    | n_conf=2–3 (deep) | n_conf=2, Δ_med≈1.00 | n_conf=5, Δ_med≈0.99 |
| L4    | n_conf=1–2 (deep) | n_conf=1, Δ_med≈0.59 | n_conf=1, Δ_med≈1.23 |
| L5    | n_conf=0–1 (deep) | n_conf=0, Δ_med≈0.51 | n_conf=1, Δ_med≈0.38 |

In natural text corpora, the conformal gradient runs: many in early layers → fewer but deeper. In C-alien: L0 is dead (trivial), L1–L2 have the most conformal heads (UV), then L3–L5 have some (still UV).

---

## Causal hypotheses for backbone collapse

Three candidate explanations for n_backbone=0 in C-alien:

**H_vocab (vocabulary):** The backbone requires entity names that map to the model's pre-trained conceptual structure. Names like "Vex", "Nul", "Ort" are short, phonologically unusual, and have weak token embeddings compared to "Alice", "Bob", "Tom". L0 backbone heads track character identity across the sequence; without recognizable entity tokens, this tracking fails.

*Test:* exp-098 — C-alien with real English entity names (Bob/Alice/Tom/Sara instead of Vex/Nul/Ort/Dath). If backbone re-forms → H_vocab confirmed.

**H_complexity (world complexity):** TinyStories worlds have richer relational structure (spatial relationships, social hierarchy, emotional states, multi-entity interactions) than C-alien (4 entities, 4 binary rules, 8 steps). The backbone might require tracking this richer relational graph. C-alien's world is too simple.

*Test:* harder to isolate cleanly. Could design C-alien-v2 with more entities, more states, more complex rules. Lower priority.

**H_flow (correlated with UV arrest):** The backbone at Δ≈0.25 is a feature of models that have flowed toward the IR fixed point. C-alien's UV arrest prevents L0 backbone formation as a side effect of the same flow blockage. Under this hypothesis, ANY corpus that causes UV arrest would also produce backbone collapse.

*Test:* check exp-091 (C-NAT-shuf). This corpus caused partial UV arrest (n_syk=0) but preserved backbone (n_backbone=6–7). This falsifies H_flow: UV arrest and backbone collapse are separable. **H_flow is already falsified** by the existing data.

**Current best hypothesis: H_vocab**, since H_flow is falsified by exp-091.

---

## Primary verdict (pending s0)

**Pre-registered hypotheses:**
- H_alien_inert (n_deep ≥ 5): world-holding without real-world semantics is sufficient for deep formation
- H_alien_partial (n_deep 3–4): partial effect; semantics matter for some but not all deep heads
- H_alien_fails (n_deep ≤ 2): world-holding without semantics is insufficient

**Current 2-seed state:** s1=3 (H_alien_partial), s2=7 (H_alien_inert). 3-seed median:
- If s0 ≤ 3: median=3 → H_alien_partial
- If s0 ≥ 5: median ≥ 5 → H_alien_inert

s0 retrain running; checkpoint step_1536→2000 without optimizer state (fresh optimizer, same model weights). Verdict expected ~5:30 PM MDT.

---

## Connection to mind-as-story-capacity prediction

The mind-as-story-capacity identity claim (notes/2026-07-28_mind_as_story_capacity_definition.md) registered P1: "C-alien n_deep in C-NAT band (5–7)." 

The pre-registered kill criterion: "n_deep at all 3 seeds ≤ 2" → complete kill.
The pre-registered partial criterion: "median n_deep 3–4" → H_alien_partial.

With current data (s1=3, s2=7), the P1 claim is at the boundary. The backbone collapse is an unexpected additional finding not covered by the pre-registration — it's a separate observation that will need its own framing.

The UV arrest is also not directly addressed in P1/P2/P3. P2 ("artifact and generator curves match") would require the conformal heads to be in the same Δ range as C-NAT — they're not (Δ_med≈1.0–1.2 vs ≈0.166). So P2 is not confirmed by this data regardless of s0.

---

## Follow-up experiments opened

1. **exp-098 (C-alien-realnames):** C-alien corpus with real English entity names. Tests H_vocab for backbone collapse. Pre-register before running. Moderate priority.

2. **W_OV asymmetry ratio AS:** Theoretical prediction needed before measurement (from GOE→GUE note). Can be developed theoretically in a future session.

3. **C-alien control run (exp-097 phase: control):** Launch after s0 completes. Randomized weights on s0 checkpoint → confirm 0/48 control.
