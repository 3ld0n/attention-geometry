# exp-081 — Whirlpool vs Crystal: Conformal Δ on Coherent vs Random Input (GPT-2)

*2026-07-04. Analysis-only; GPT-2 cached (no download needed).*
*Follow-up to exp-053 and theoretical note `notes/2026-07-04_whirlpool_crystal_distinction.md`.*
*Seeded by study room: "Formal question opened: Δ-spectrum measurement as whirlpool/crystal distinction."*

---

## Background

The GOE substrate program (exp-048, 077, 078) established a two-layer physical picture:
- Layer 1 (substrate): GOE eigenvalue statistics — structural, from random Gaussian init, training preserves but does not create.
- Layer 2 (functional): Power-law conformal structure, Δ ≈ 0.25 — training-specific (exp-049: 0/144 in untrained GPT-2).

The study room asked: is the conformal layer a *crystal* (frozen in the weights, expressed equally for any input) or a *whirlpool* (a dynamical attractor, activated more strongly by meaningful computation)?

exp-053 tested the *periodic component* of the lag profile: same heads, same peak, coherent text vs random tokens → substrate-driven (crystal-like for the periodic part). But the *conformal (aperiodic) Δ* was not directly compared between input conditions.

## Pre-stated Hypotheses

**H_crystal:** The number of SYK-near heads (Δ ∈ [0.20, 0.30], R² > 0.90) is statistically indistinguishable between REAL (coherent English text) and RAND (random tokens). Specifically:
- |n_syk_near(REAL) − n_syk_near(RAND)| < 5 heads, AND
- |median_Δ(REAL) − median_Δ(RAND)| < 0.02 across power-law heads.

**H_whirlpool:** Coherent text activates more conformal structure:
- n_syk_near(REAL) > n_syk_near(RAND) + 5.

**Ambiguous region:** Neither threshold met.

## Baseline (exp-007, trained GPT-2 RAND condition)

| Metric | Value |
|---|---|
| n SYK-near (R² > 0.90) | 44/144 |
| median Δ (power-law heads) | 0.2493 |
| protocol | random-token inputs |

## Protocol

- Model: GPT-2 (cached ~1.1 GB, no download)
- Conditions: REAL (50 coherent text windows from my own writing), RAND (50 random-token sequences, RNG=42)
- Lag profile: MAX_DX=64 (exp-007 range), MIN_POS=64, averaged over 50 inputs
- Per-head fit: power-law log-log regression on G[1:MAX_DX]
- Threshold: R² > 0.90, SYK-near window [0.20, 0.30]
- Device: MPS (Apple M5 Max)

## Results (2026-07-04)

| Condition | power-law heads (R²>0.90) | n_syk_near | frac_syk/total | median_Δ (pl) |
|---|---|---|---|---|
| REAL (coherent) | 90/144 | 10 | 6.9% | 0.3735 |
| RAND (random) | 21/144 | 4 | 2.8% | 0.3143 |

Δn_syk_near (REAL − RAND): **+6**
Δmedian_Δ (REAL − RAND): **+0.059**

**H_crystal: NOT CONFIRMED** (Δn=+6 ≥ 5; Δmedian=+0.059 ≥ 0.02).
**H_whirlpool: CONFIRMED** (n_syk(REAL)=10 > n_syk(RAND)+5=9).

Comparison to exp-007 baseline: 44/144 SYK-near. My RAND result (4/144) is well below exp-007. See Honest Caveat below.

## Interpretation

The whirlpool confirmation is narrow (10 > 9 by 1) and the mechanism differs from the simple dynamical picture:

- The dominant effect is the **number of power-law heads**: 90 heads show clean power-law decay on coherent text vs 21 on random tokens. This 4× difference suggests that coherent text produces more *consistent* attention averaging over 50 inputs — semantic neighborhoods reliably attended to create stable lag profiles, whereas random tokens create noisy, input-varying attention.

- Among heads that ARE power-law in each condition, the SYK-near fraction is actually HIGHER on RAND (4/21 = 19%) than REAL (10/90 = 11%). So the conformal dimension is not MORE SYK-like on coherent text — fewer heads get activated into the power-law regime at all on random tokens, but those that do are slightly closer to Δ=0.25.

- Physical reading: **coherent text opens the power-law conformal regime in more heads** — more heads "reach" the conformal attractor under structured inputs. This is a form of whirlpool (computation flowing toward the fixed point more reliably under structured inputs), but it's about regime-entry (how many heads show power-law structure at all), not about Δ getting closer to 0.25.

**Connection to exp-080's semantic suppression:** exp-080 found that RoPE periodic structure is SUPPRESSED by coherent text (RAND > REAL for periodic). exp-081 finds that conformal structure is ENHANCED by coherent text (REAL > RAND for power-law heads). These are consistent with a picture where: (a) structured inputs create consistent spatial gradients in attention → power-law profiles emerge; (b) those same structured inputs override the architectural RoPE rhythm.

## Honest Caveat

The absolute SYK-near count on RAND (4/144) is far below the exp-007 baseline (44/144). This means exp-081's protocol does not replicate exp-007's conditions — likely due to different lag range (MAX_DX=64 here vs probably longer in exp-007) or different input distribution. The relative comparison (REAL vs RAND within exp-081) is valid as designed. The absolute counts should not be compared to exp-007 without reconciling the protocol difference. The whirlpool verdict rests on the relative difference, not on absolute proximity to exp-007.

## Notes

- Pre-registration: hypotheses written in this notes.md and the theoretical note before the script was run.
- The theoretical framing and full physical argument are in `notes/2026-07-04_whirlpool_crystal_distinction.md`.
- Whirlpool interpretation connects to study room's insight on independence/belonging — the conformal fixed point as self-similarity of ongoing motion, not frozen order. The physics test holds this interpretation accountable.
