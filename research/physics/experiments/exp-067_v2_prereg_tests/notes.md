# exp-067: v2 Pre-Registration Tests — Notes

*June 14, 2026. Solo physics room session.*
*Pre-registration: `research/physics/notes/2026-06-14_v2_preregistration.md`*
*Committed before any measurements: git commit 5687c6f4*

## Order of operations (look-ahead-bias control)

1. Pre-registration written and committed to git (`5687c6f4`).
2. Test A run on archived Pythia-410m data (`run_test_a.py`).
3. Test B run on loaded models Pythia-1.4b and GPT-2-medium (`run_test_b.py`).

## Test A: Joint (Δ, λ) → Implied Valley

**Diagnostic models (threshold-setters, not confirmatory):**
- Pythia-2.8B: ρ(v̂, valley_meas) = +0.683 (from exp-061)
- GPT-Neo-2.7B: ρ(v̂, valley_meas) = +0.906 (from exp-061)

**Fresh confirmatory model — Pythia-410m (archived exp-026 data, not used in exp-061):**
- n_conformal_heads: 81 (R² ≥ 0.85, Δ̂ ≥ 0.05, filtered from 239 total BCFT-fit heads)
- ρ(v̂, valley_meas) = **+0.7528** (p = 5.33×10⁻¹⁶) → **KEEP (A1 confirmed)**
- ρ(Δ̂, valley_meas) = **+0.7207** (p = 3.31×10⁻¹⁴)
- Improvement (A2): **YES** (Δρ = +0.0321 — the joint statistic outperforms Δ alone)

**Verdict:** First fresh-model confirmation of Test A. ρ = 0.753 clears the 0.60 threshold comfortably; the joint implied-valley predictor is strictly better than Δ alone on Pythia-410m.

**Additional confirmatory model — Pythia-1.4b (local BCFT fit, June 17 2026):**
- Fit file: `exp-026/results/bcft_functional_form_fit_2026-06-18T021918Z_pythia-1.4b.json`
- n_conformal_heads: **95** (of 349 total)
- ρ(v̂, valley_meas) = **+0.8865** (p = 6.95×10⁻³³) → **KEEP (A1 confirmed)**
- ρ(Δ̂, valley_meas) = **+0.5977** (p = 1.60×10⁻¹⁰)
- Improvement (A2): **YES** (Δρ = +0.2888 — largest improvement of all three models)

**Cross-model pattern:** The joint (Δ, λ) predictor improvement varies by model:
- Pythia-410m (RoPE): Δρ = +0.032 — λ adds little to Δ
- Pythia-1.4b (RoPE): Δρ = +0.289 — λ adds substantial predictive power
- gpt2-medium (learned PE): Δρ = +0.184 — strong λ contribution

The large Pythia-1.4b Δρ is notable: Δ alone achieves only ρ=0.598 at 1.4b (barely clearing the 0.50 kill threshold), but the joint predictor reaches 0.887. This suggests that at 1.4b scale, heads with similar conformal dimension Δ have highly variable valley depths, and the BCFT boundary parameter λ resolves that ambiguity. Hypothesis: larger models develop more heterogeneous head behavior where the boundary term's head-specific contribution is essential for valley prediction. The PE-type effect may be secondary.



**Additional confirmatory model — GPT-2-medium (local BCFT fit, June 17 2026):**
- Fit file: `exp-026/results/bcft_functional_form_fit_2026-06-18T021508Z_gpt2-medium.json`
- n_conformal_heads: **317** (of 384 total)
- ρ(v̂, valley_meas) = **+0.9792** (p = 2.38×10⁻²²⁰) → **KEEP (A1 confirmed, very strong)**
- ρ(Δ̂, valley_meas) = **+0.7955** (p = 1.67×10⁻⁷⁰)
- Improvement (A2): **YES** (Δρ = +0.1837 — largest improvement seen across all models)

**Interpretation:** ρ = 0.979 is the strongest Test A result by far (vs. 0.753 for Pythia-410m, 0.683/0.906 for the diagnostic models). The joint (Δ, λ) BCFT model predicts valley depth with near-perfect rank-correlation in gpt2-medium. The large improvement in Δρ (+0.18 vs. +0.03 for Pythia-410m) suggests that the BCFT boundary parameter λ carries substantially more predictive information in gpt2-medium — likely because gpt2-medium uses learned PE (not RoPE), where the boundary absorption mechanism is more cleanly expressed without the rotary encoding's periodic structure modulating λ.

## Test B: Depth-Accumulated Primacy Laws

### P-B1 (Law A — primacy grows with depth): Pythia-1.4b

- N = 512, 50 random-token inputs, 24 layers
- k checkpoints: [1, 2, 3, 4, 6, 8, 12, 16, 20, 24]
- prim_decile values: 0.013 → 0.036 → 0.114 → 0.209 → 0.495 → 0.678 → 0.955 → 0.998 → 1.000 → 1.000
- Spearman ρ(k, prim_decile) = **+1.0000** → **KEEP (P-B1)**

### P-B2 (Law B — start-window mass decreases with context N): Pythia-1.4b

- Full depth (24 layers), N ∈ {256, 512, 1024, 2048}
- prim_window(8): N=256 → 0.99996; N=512 → 0.99967; N=1024 → 0.99820; N=2048 → 0.99347
- Spearman ρ(N, prim_window) = **−1.0000** → **KEEP (P-B2)**

**Honest note on P-B2 magnitude:** The prim_window(8) values are nearly 1.0 at all N values — Pythia-1.4b fully saturates the absorbing boundary by 24 layers, leaving little room for N-dependent variation. The differences are small in absolute terms (range: 0.99996 − 0.99347 = 0.0065). The monotonicity (ρ = −1.00) is perfect, and this *is* the registered verdict. However, for a more informative characterization of Law B, the test should be repeated at intermediate depth (k ≈ 6–8, where prim_window ≈ 0.15–0.28 and context-length effects are not saturated). This is a follow-up, not a retraction.

### P-B3 (Law A robustness — GPT-2-medium)

- N = 512, 50 random-token inputs, 24 layers
- prim_decile values: 0.062 → 0.182 → 0.225 → 0.260 → 0.389 → 0.570 → 0.890 → 0.989 → 1.000 → 1.000
- Spearman ρ(k, prim_decile) = **+1.0000** → **KEEP (P-B3)**

## Registered verdicts summary

| Test | Model | ρ | Verdict |
|---|---|---|---|
| A1 (joint implied valley) | Pythia-410m | +0.753 | KEEP |
| A2 (improvement over Δ alone) | Pythia-410m | +0.032 over Δ | YES |
| A1 (joint implied valley) | Pythia-1.4b | +0.887 | KEEP |
| A2 (improvement over Δ alone) | Pythia-1.4b | +0.289 over Δ | YES (largest) |
| A1 (joint implied valley) | GPT-2-medium | +0.979 | KEEP (strong) |
| A2 (improvement over Δ alone) | GPT-2-medium | +0.184 over Δ | YES (large) |
| P-B1 (depth-primacy) | Pythia-1.4b | +1.000 | KEEP |
| P-B2 (context-dilution) | Pythia-1.4b | −1.000 | KEEP |
| P-B3 (depth-primacy robustness) | GPT-2-medium | +1.000 | KEEP |

## Honest assessment

The tests passed on all registered metrics. However:
1. **Test A** now confirmed on three fresh models (Pythia-410m ρ=0.753, Pythia-1.4b ρ=0.887, GPT-2-medium ρ=0.979). All three KEEP with A1+A2. Cross-model pattern: λ improvement largest at 1.4b scale (Δρ=+0.29), suggesting head-behavior heterogeneity grows with scale and λ resolves it. Possibly also PE-type effect (learned PE gpt2-medium has very strong ρ); needs Pythia-family comparison at same scale to disambiguate scale vs. PE.
2. **P-B2 saturation:** The full-depth context-dilution test shows the right direction with perfect monotonicity, but the effect size at full depth is tiny. The pre-registered verdict is KEEP; the scientific content of P-B2 is better seen at intermediate depth.
3. **P-B1 and P-B3 are near-tautological** for models that fully saturate (all mass goes to position 0 by the final layer). Their content is in the functional form of growth — which is what they measure.

## Next steps

1. ~~Run Test A on GPT-2-medium~~ — DONE June 17 2026 (ρ=0.979 KEEP). See above.
2. ~~Run Test A on Pythia-1.4b~~ — DONE June 17 2026 (ρ=0.887, n=95, KEEP). See above.
3. P-B2 at intermediate depth (k=6 or k=8): add to exp-067 as a follow-up.
4. ~~Task-level slope editing (pre-register for Pythia-1.4b retrieval accuracy)~~ — Done exp-068 → exp-069 → exp-070 → exp-072 (full program).

*Registry: exp-067. Scripts: run_test_a.py, run_test_b.py. Results: results_test_a.json, results_test_b.json.*
