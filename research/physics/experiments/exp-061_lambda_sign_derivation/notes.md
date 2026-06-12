# exp-061: λ-sign derivation re-test — Notes

*June 10, 2026. Phase 0 item 0.4 (SESSION_BRIEF_PHASE0). Resolves anomaly A1 / EVALUATION W4.*

## Order of operations (look-ahead-bias control)

1. Derivation written and sign committed first: `notes/2026-06-10_phase0_0.4_lambda_sign_derivation.md` (workspace notes). Result is analytic and unconditional in Δ: under the image form with the exact pipeline valley statistic, valley(λ) is V-shaped with minimum at λ*(Δ) = (E₀−S₀)/(S₁−E₁); ∂valley/∂λ < 0 below λ*, > 0 above. Follows from two monotonicity facts alone (bulk kernel increasing in j, image kernel decreasing in j).
2. Numerics + re-test against archived exp-026 fits (this experiment).
3. Secondary SCT-breaking analysis after the primary, labeled.

## Key numbers

- λ*(0.25) = 8.24; λ*(0.10) = 5.02; λ*(0.50) = 21.7 (pipeline-exact, L=512, deep queries 384–511).
- Regime-1 fraction at fitted (Δ̂_BCFT, λ̂_BCFT): Pythia-2.8B 99.0% (n=207), GPT-Neo-2.7B 81.6% (n=440).
- Predicted sign: negative. Measured per-layer ρ(λ̂, valley): Pythia-2.8B 20 neg / 1 pos; GPT-Neo 26 neg / 5 pos.
- **New result:** model-implied valley v̂(Δ̂, λ̂) vs measured valley: ρ = +0.683 (Pythia-2.8B), **+0.906** (GPT-Neo). No free parameters beyond each head's own (Δ̂, λ̂). The image form quantitatively predicts the behavioral statistic.
- Secondary: ρ(λ̂, middle-mass fraction) = +0.451 / +0.589 — SCT-breaking reinterpretation directionally supported; mechanism consistent (image term pads the middle relative to the end).

## Verdict

**A1 resolved as a derived confirmation.** The framework's verbal prediction (ρ(λ, valley) > 0) was wrong about the statistic; the framework's own functional form, pushed through the exact pipeline definition, predicts the negative sign that was measured. W4 closed; manuscript §9.3-A1 and §10 updated. The v2 valley statistic does not need redesign on this account.

*Registry: exp-061. Runtime: <1 s (archived data + exact sums).*
