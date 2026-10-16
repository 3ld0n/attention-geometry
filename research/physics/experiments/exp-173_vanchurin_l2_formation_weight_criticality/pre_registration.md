# Pre-registration: exp-173 — Vanchurin L2 double dissociation

**Title:** Weight-side fluctuation exponent across formation-ladder rungs (Vanchurin L2 double dissociation)

**Date:** 2026-10-16 (committed to attention-geometry before any code; push pending GitHub token restoration)

**Session:** Physics room, October 15, 2026 evening. Solo.

**Question served:** Formation — what must a system be made of and raised on for the deep structure to form?

**Source material:** `research/external/vanchurin_map.md` §1 (P1 verdict), §5 (L2 pre-registration material); `research/external/vanchurin.md` §3 L2. Papers read at source on 2026-08-31: arXiv:2107.03402 (→ Physica A 610:128401) and arXiv:2405.17391 (→ Entropy 27:989).

---

## Background

Kukleva–Vanchurin prove that trained neural network weight fluctuations near "learning equilibrium" follow a power law whose exponent is determined by the activation+loss composition — *not* by the training data. Their specific claim (Entropy 27:989 §6): same activation and loss function ⇒ same power-law exponent on the fluctuation distribution, regardless of corpus. Architecture is the knob; data is not.

My formation ladder (exp-085/091/092/093/094/096) shows the opposite for attention-side geometry: Δ_A under the frozen random-token census forms only over language with referential binding at story scale, in story order — the specific corpus is everything. Matched statistics, grammar, the statistical shadow of language, story-order in fragments — all fail. The data variable is load-bearing for Δ_A.

**The sharpest contact and the testable tension:** Both can be true simultaneously if the observables are genuinely different and independent. Their weight-side η (exponent on the gradient-jump distribution) is computed on weight parameters at SGD equilibrium; my attention-side Δ_A is the ensemble lag-decay exponent of attention heads under a fixed input protocol. If these are decoupled, the double dissociation holds: η flat across rungs while Δ_A rung-dependent.

**What the two-populations result already sketched:** exp-109 confirmed that the random-token-native and WikiText-native Δ-window populations share no head (Jaccard < 0.3). Their generic criticality may be what *any* trained network carries; the data-formed Δ-geometry is what the text-native population specifically carries. The L2 experiment tests whether this same substrate/signal split shows up at the weight-statistics level.

---

## Hypotheses

**H1 (primary — double dissociation):** The weight-side fluctuation power-law exponent η is approximately constant across all formation-ladder rungs (C-NAT, C-generated, C-shadow, C-shuffled, C-alien, C-statistics), while Δ_A under the frozen random-token census is rung-dependent (high on C-NAT; absent on C-alien/C-shuffled). Operationally: |η_rung_i − η_rung_j| ≤ 0.10 for all rung pairs; median Δ_A on C-NAT ≥ 0.20 while C-alien/C-shuffled ≤ 0.15.

**H2 (secondary — clean power-law regime):** A power-law intermediate-scale regime is identifiable in the potential-jump distribution for ≥ 3 of the 5 rungs with fit quality R² ≥ 0.85 in log-log space.

**H3 (secondary — equilibrium proximity):** Loss-plateau diagnostics (‖∇U‖/‖U‖ ratio, reported per checkpoint) confirm that the measured checkpoints are within 30% of the local-equilibrium floor for each rung.

---

## Kill conditions

**K1 (kills Kukleva–Vanchurin architecture-independence claim, and enriches the formation story):** η tracks the formation-ladder rung — Spearman ρ(rung-rank, η) ≥ 0.7, or η on C-NAT differs from η on C-alien/C-shuffled by ≥ 0.15. This would be a finding: the data variable shapes weight statistics, not only attention geometry. G7's derivation of Δ = 1/4 would then need to account for corpus dependence of the background statistics.

**K2 (kills the measurement — protocol fails to transfer):** No clean power-law regime in ≥ 3/5 rungs under the Adam-trained GPTNeoX setting (R² < 0.85 on intermediate scales for all identifiable candidate regimes, or no candidate regime identifiable). If K2 fires: link between the programs retires honestly; the formation ladder's Δ_A result stands as the cleaner, non-weight-side instrument; future work should note that the Physica A protocol is SGD-specific.

**K3 (kills the SGD/Adam equivalence — secondary arm only):** If an SGD replication arm is included and SGD produces qualitatively different η from Adam on the same rung, the primary arm's results are exploratory and the SGD arm is the valid comparison.

---

## Procedure

### Primary arm — static checkpoint, Adam-trained

**Checkpoints:** Formation-ladder checkpoints from the 70m GPTNeoX model (Architecture: 6L/8H/d=512, vocab 50304) stored in Modal volumes from exp-085 (C-NAT arm) and earlier formation-ladder experiments. The checkpoints are at multiple training steps; use the final trained checkpoint for each rung where one exists.

> **Modal dependency note:** Checkpoints live in Modal cloud volumes (exp062-data, exp085-data, etc.). This experiment is **Modal-gated** pending restoration of Modal account access. The pre-registration commits before any code; execution follows when Modal access is restored.

**Per rung:**
1. Load checkpoint θ from Modal volume
2. Draw N=500 training examples from that rung's corpus (same corpus used during training; random seed 42)
3. Compute per-parameter potential-jump δ_i = −γ∇_θ H_i for each example (learning rate γ from training config; H_i is the cross-entropy loss on that example)
4. Pool per-layer-block: concatenate all δ_i into one distribution per layer block (attention and MLP separately; report joint)
5. Fit the intermediate-scale regime: log-log histogram of |δ_i|; identify the intermediate regime between the left peak (frustrated regime, small |δ|) and the right Gaussian shoulder; fit power-law slope η per layer block, report median across layer blocks and per-layer profile
6. Also: Δ_A census under frozen random-token census (same protocol as exp-007 and exp-085 census measurements; N=200 random tokens, pooled rows, log-log slope from lag 1 to 50)
7. Equilibrium diagnostic: compute ‖∇U‖/‖U‖ on a 50-example sample; report alongside η

**Rungs to measure:** C-NAT (top rung, canonical), C-generated (shadow corpus), C-shuffled, C-statistics (matched statistics), C-alien. If checkpoints are not available for all rungs, report available rungs and mark missing rungs as not measured.

**Optimizer caveat (named explicitly, not corrected away):** Formation-ladder was trained with AdamW (β₁=0.9, β₂=0.95, wd=0.1). Kukleva–Vanchurin's derivation is SGD batch-1 with L2 regularization. In Vanchurin's 2504.14728 framework, Adam corresponds to α=1/2 (a different learning-dynamics regime than SGD's α=0). **The primary arm therefore uses a different optimizer regime than the derivation.** Pre-committed interpretation: the η comparison is made explicit as exploratory if K3 fires; the double-dissociation test (H1) does not depend on the Adam/SGD equivalence and is valid regardless — if η is flat under Adam, the dissociation is confirmed at exploratory strength; if K3 also confirms (SGD arm agrees), strength upgrades.

### Secondary arm — SGD replication (optional)

If compute allows (low-cost on M5 Max locally, no Modal needed):
1. Retrain the C-NAT checkpoint for 50 SGD steps (batch 1, lr 2·10⁻³, L2 5·10⁻⁴, matching Physica A §6 protocol)
2. Measure η using the same per-layer-block protocol above
3. Compare to the primary arm C-NAT η; if they agree within 0.05, K3 does not fire

---

## Registry fields

```
"id": "exp-173",
"date": "2026-10-16",
"title": "Vanchurin L2 double dissociation — weight-side fluctuation exponent vs formation-ladder rungs",
"hypothesis": "The weight-side fluctuation exponent η is architecture-constant across formation-ladder rungs (Kukleva–Vanchurin claim) while Δ_A is rung-dependent (formation finding); the double dissociation confirms independent substrate and formation physics.",
"model": "GPTNeoX-70m (6L/8H/d=512)",
"pos_enc": "learned",
"scripts": [],
"results": [],
"result_summary": {},
"status": "registered",
"verdict": "pending",
"prereg_commit": "pending — local commit to attention-geometry; push blocked (GitHub token invalid, Eldon-gated)",
"prereg_evidence": "self-attested — local commit timestamps pre-registration before any code; run is Modal-gated so ordering is preserved; push ordering verifiable once token restored",
"analysis_only": false,
"surfaced_in": [],
"published": "",
"notes": "research/physics/experiments/exp-173_vanchurin_l2_formation_weight_criticality/pre_registration.md",
"tags": ["formation-ladder", "external-comparison", "vanchurin-l2", "weight-statistics", "double-dissociation"],
"quality": {
    "specific": true,
    "honest_negative": false,
    "used_by_next_session": null
}
```

---

## Relation to program

**Formation question (primary):** Does the variable that governs Δ_A formation (corpus identity, referential binding, story order) also govern weight-side statistics? If yes, formation is not only an activation-side phenomenon. If no, the double dissociation holds — substrate criticality is generic, formation-specific geometry is signal.

**Spine claim A2:** The formation ladder's central claim is that corpus identity is load-bearing for Δ_A. This experiment tests whether the same claim holds at the weight level — a different instrument on the same variable.

**T3 / G7:** The weight-side exponent might explain why *some* power-law exists (Vanchurin's generic mechanism) without explaining why the exponent is *1/4* (the formation-specific value). G7's derivation of Δ = 1/4 on A's own terms would then be asked to account for both levels: why the generic criticality, and why exactly this exponent.

**Surfaces when complete:** `research/physics/OVERVIEW.md` (if dissociation confirmed or K1 fires — both are findings). `research/external/vanchurin.md` §6 (if K2 fires and the link retires).

---

*Pre-registered before any code is written. Run pending Modal access and GitHub token restoration (both Eldon-gated). Execution will not begin until attention-geometry push is confirmed — the local commit establishes the pre-registration ordering.*
