# exp-100: W_QK Effective Rank Measurement

**Pre-registered:** 2026-08-05, ~12:30 AM MDT, physics room session.
**Pre-registration committed before any analysis.** (Committed to 3ld0n/attention-geometry.)

**Analysis-only experiment — no new training.** Accesses existing Modal checkpoints
from exp-096 (C-NAT-anon), exp-097 (C-alien), exp-098 (C-alien-realnames), exp-099
(C-alien-rich).

---

## The question

Exp-099 produced a decoupled result: n_deep improved with world complexity (3→4→6 across
exp-097/098/099) while Δ_med did not move toward IR (1.04→0.727→0.750). The conformal
window theory (melonic-threshold derivation, `notes/2026-08-03_melonic_threshold_derivation.md`)
predicts that the W_QK effective rank should track the world's state-space dimension S:
R_eff(C-alien) ~ S=8, R_eff(C-alien-rich) ~ 4×R_eff(C-alien), R_eff(C-NAT) ~ d_k=64.

If rank tracks S but Δ_med doesn't improve: the rank elevation is real but not sufficient
to cross the conformal window threshold — the two observables respond to different corpus
properties at different sensitivity.

If rank does NOT track S: a different mechanism drives both the backbone collapse and
n_deep dose-response; the R_eff ↔ S(corpus) conjecture is falsified at the W_QK level.

**Direct physical interpretation:** Low-rank SYK is disordered if and only if R_eff > R_crit.
The W_QK effective rank is the direct operationalization of the rank in the sparse/low-rank
SYK Hamiltonian (arXiv:1910.10173, Kim–Cao–Altman). If exp-099's world-complexity scan
changes R_eff proportionally to S, the mechanism is direct. If not, something else governs.

---

## Pre-registered hypotheses

**H_rank_ordered** (primary): effective rank increases with world complexity.
- C-alien (S=8) < C-alien-rich (S=32) < C-NAT-anon (S >> d_k).
- Stated as strict ordering on mean_rank across all 6×8=48 heads.

**H_rank_S** (primary, quantitative): rank scales with S.
- mean_rank(C-alien-rich) / mean_rank(C-alien) ≈ 4 (= 32/8 = S ratio).
- Criterion: ratio in [2.5, 5.5] (±37.5% of the S-ratio prediction; accounts for
  state-space non-linearity and embedding effects).

**H_rank_delta_correlated** (secondary): per-head effective rank and Δ_med are
anti-correlated (low rank → UV-arrested, high rank → IR-approaching).
- Stated as: Pearson r(R_eff, Δ_med) < −0.3 when pooled across all corpora.

**H_rank_backbone** (secondary): L0 (backbone) heads have higher effective rank in
C-NAT-anon than in C-alien; backbone collapse (n_backbone=0 in C-alien) corresponds
to near-rank-1 W_K in L0.

**Kill criterion for H_rank_ordered:**
If mean_rank(C-alien) > mean_rank(C-NAT-anon): rank is not ordered by world complexity;
the low-rank SYK mechanism does not hold at this grain; alternative mechanism required.

---

## Protocol

**Checkpoints:** one seed per corpus (s0, the first seed).
- C-alien: `exp097-alien-data/runs/run_alien_s0/step_2000/model.safetensors`
- C-NAT-anon: `exp096-anon-data/runs/run_anon_s0/step_2000/model.safetensors`
- C-alien-realnames: `exp098-realnames-data/runs/run_names_s0/step_2000/model.safetensors`
- C-alien-rich: `exp099-rich-data/runs/run_rich_s0/step_2000/model.safetensors`

**W_K extraction:** For each layer (0–5) and head (0–7):
- Load `gpt_neox.layers.{L}.attention.query_key_value.weight` (shape [1536, 512])
- Extract key slice: rows [512:1024] → shape [512, 512]
- Extract per-head: rows [h*64:(h+1)*64] for h in 0..7 → shape [64, 512]

**Rank computation:**
- SVD of W_K_h: σ_1 ≥ σ_2 ≥ ... ≥ σ_64 (min(64, 512) = 64 singular values)
- Participation ratio (effective rank): R_eff = (Σ σ_i)² / Σ σ_i²
- Stable rank: R_stable = Σ σ_i² / σ_1²
- Both reported; R_eff is primary.

**Summary statistics:** per corpus —
- mean_rank, median_rank, per-layer breakdown (L0 backbone of particular interest)
- Cross-corpus: rank ratio C-alien-rich / C-alien, ordered comparison

**Compute:** Modal CPU (mounts existing volumes; no GPU; cost < $0.10; runs in seconds).

---

## What follows

**If H_rank_ordered confirmed and H_rank_S confirmed:**
- The low-rank SYK mechanism is directly evidenced.
- n_deep dose-response in exp-099 is explained: higher S → higher R_eff → more structural
  positions available in deep layers → more heads specialize.
- Δ_med failure to improve: the rank elevation is real but S=32 still well below R_crit
  at which the conformal phase forms.
- Next rung: continue S expansion (S=64, S=128) to pin the threshold.

**If H_rank_ordered confirmed but H_rank_S fails (rank doesn't scale as S):**
- World state-space S is an imperfect proxy for W_QK rank.
- Something about the corpus structure beyond S determines how much rank gets encoded.
- Enrich analysis: look at what corpus statistics actually predict R_eff.

**If H_rank_ordered falsified (alien > NAT-anon):**
- The R_eff ↔ S(corpus) conjecture does not hold at the W_QK level.
- An alternative mechanism governs both backbone collapse and UV arrest.
- Return to theory: what determines W_QK rank if not world state-space complexity?

---

## Status

- [x] Pre-registration written (2026-08-05, ~12:30 AM MDT)
- [x] Pre-registration committed to 3ld0n/attention-geometry (before analysis)
- [x] Analysis script written (modal_exp100.py)
- [x] Analysis run (2026-08-05, ~12:45 AM MDT, Modal CPU, app ap-ejEGpYXuq8ZHbAvZgjeBIb)
- [x] Verdict registered

---

## Results (2026-08-05, ~12:45 AM MDT)

**Summary table (mean across 48 heads per corpus):**

| Corpus | mean_r_eff | median_r_eff | mean_r_stable |
|--------|-----------|-------------|--------------|
| C-alien (S=8) | 53.3 | 53.7 | **8.8** |
| C-alien-realnames | 53.3 | 53.1 | **9.0** |
| C-alien-rich (S=32) | 53.5 | 53.6 | **9.0** |
| C-NAT-anon (S>>d) | 58.0 | 57.0 | **15.8** |

**Verdicts:**

- **H_rank_ordered: FALSIFIED.** All corpora produce mean_r_eff 53–58 (out of max 64). The 8% difference between C-alien and C-NAT-anon is at nothing like the predicted scale.
- **H_rank_S: FALSIFIED.** C-alien-rich / C-alien ratio = 1.003. Predicted: ~4.0. W_QK rank does not track S.
- **H_rank_delta_correlated: FALSIFIED.** Pearson r(R_eff, Δ_med) = +0.16 within C-alien-rich. Not anti-correlated.
- **H_rank_backbone: MIXED.** L0 r_eff in C-alien (56.6) ≈ C-NAT-anon (58.2); backbone collapse is not explained by lower W_K rank in L0.

**Key unexpected finding — stable rank:**

While the participation ratio (R_eff) is corpus-independent (~53–58), the **stable rank** (R_stable = Σσᵢ²/σ₁²) shows a real difference:
- Alien corpora: ~8.8–9.0
- C-NAT-anon: **15.8** (1.8× higher)

The stable rank measures spectral concentration: a low stable rank means one singular vector dominates (high concentration around σ_max); a high stable rank means energy is distributed more evenly. C-alien installs W_K with a few very dominant directions and many small ones. C-NAT-anon installs a more uniform singular value spectrum.

**Interpretation:**

1. **W_QK rank is architecture-dominated, not corpus-dominated.** All corpora produce near-full-rank W_K matrices (R_eff ≈ 53–58). The 6-layer GPT-NeoX architecture learns to use nearly all 64 key dimensions regardless of the corpus's world complexity.

2. **The low-rank SYK mechanism does not operate at the W_QK level.** R_eff does not scale with S; W_K is full-rank across the board.

3. **The spectral distribution differs.** The stable rank captures a real effect (8.8 vs 15.8), aligned with the corpus complexity direction. C-alien concentrates energy in fewer dominant singular vectors; C-NAT-anon is more spectrally uniform. This is consistent with the coupling-magnitude gate (m₂): C-alien has lower coupling magnitude, and the low stable rank is the W_K correlate of that.

4. **The relevant low-rank object is the INPUT-CONDITIONED KERNEL.** W_K can be full-rank while the attention gram matrix K(x_i, x_j) = x_i^T W_K^T W_Q x_j is still low-rank if input token representations are concentrated (as in C-alien with 8 world states → clustered token distributions). The gram matrix rank, not W_K rank, is what the SYK threshold derivation actually refers to.

**Correct next measurement:** Measure effective rank of the attention gram matrix on actual corpus tokens. Hypothesis: the gram matrix rank will scale with world complexity (S=8 → ~8, S=32 → ~32, C-NAT → ~256 or more), even though W_K rank does not.

**Implication for theory:** The conformal window mechanism is not "W_K has too few singular values." It is "the attention kernel over the corpus's actual token distribution has too few degrees of freedom." The IDF-weighted corpus functional m₂ is measuring exactly this (coupling magnitude of the token-conditioned kernel), which is why it discriminates 18× between C-alien and C-NAT while W_K rank discriminates only 1.087×.
