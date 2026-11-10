# Marín TAF Comparison — Object Mismatch and the GPT-2 Counterexample

> **Note added 2026-11-09:** this is the second comparison note. The first is `2026-10-20_marin_taf_comparison.md` (longer; same object mismatch and GPT-2 counterexample; it includes candidate contact points and the case for and against contact). This one was written from a board task that had not been marked done, without reference to the first. Read the October 20 note first; the two have not been reconciled line by line.

*2026-11-08. Physics room. Solo. Read at source: Zenodo 10.5281/zenodo.19826343 (Part I, V2.0) and 10.5281/zenodo.19960572 (Part II). Comparison note per working-surface task (queued). No compute. Interpretive register throughout — a comparison of two programs, not a verdict on either.*

---

## What they derived

**Marín's Thermodynamic Attention Framework (TAF).** The RoPE positional encoding imposes a log-distance constraint on the attention score: `E[log d]` is fixed by the RoPE geometry. The maximum-entropy distribution compatible with that constraint is a power law A(d) ∝ d^(−γ), with closed-form exponent:

```
γ_Padé = (2θ − T_eval √2) / (2θ + T_eval √2)
```

where θ is the RoPE base frequency and T_eval is the evaluation context length. This is the [1,1] Padé approximant of e^(−z) evaluated at z = T/(θ√2). The derivation is clean and machine-verified in both Sage and Lean (Part II, 52 theorems, zero sorrys).

**What they measure.** The geometric centroid of the full attention distribution A(d) across all heads — a single γ per (model, corpus, precision). Dataset: 35 models, 13 families, 88 records. Models covered: Pythia, Qwen, Llama, Mistral, Gemma, Phi, OLMo, Falcon, Yi, DeepSeek, StarCoder2. GPT-2 is not in the dataset (see below).

**Empirical result.** Median MAE 4.3% (n=9 non-anomalous, clean panel), R²=0.44 (n=23) on the five-axis decomposition. Typical centroid values: Pythia-2.8B predicted 0.747, measured 0.674; similar order-of-magnitude across the panel. Bimodal structure: ~36% of models at γ ≥ 1 (Hagedorn zone); the rest in Phase A (0 < γ < 1). Honest: a Pythia-70M training trajectory across 9 checkpoints does not monotonically converge to their predicted ν imprint slope; that claim is weakly supported (Part II §2, withdrawn in the learned-imprint demotion).

---

## The object mismatch

Three levels, each structural:

### 1. Subpopulation vs centroid

Our Δ_A is the ensemble-averaged lag-decay exponent for the **Δ-window subpopulation**: heads with R² ≥ 0.90 and Δ ∈ [0.20, 0.30] under the frozen random-token census protocol. This is ~25–30% of heads in GPT-2 and Pythia. The remaining 70–75% of heads do not meet the R² threshold and are excluded.

Marín's γ is the geometric centroid of **all heads** — including induction heads, sink heads, structural heads, and everything else. For Pythia (RoPE), the centroid falls at ~0.67–0.75. Our Δ-window population in Pythia (exp-118) has Δ_med ∈ [0.24, 0.28] — the same as GPT-2, and far from the centroid. The two measures are not approximations of each other. They are different objects.

Concretely: for Pythia-1.4b (exp-118: 5 Δ-window heads out of some larger pool), the centroid ~0.70 reflects the dominant population of heads that do not qualify as Δ-window. The 5 Δ-window heads carry Δ_med ≈ 0.25, invisible in the centroid.

### 2. Relative distance vs absolute key position

Our exp-138 finding (the central result of the September–October T1 thread): on decomposition of the census score S_pos into bilinear components, the census slope is **absolute-key-position drift** (the m·M_QK·δ_a term), not a relative-lag law A(|i−j|)^{−2Δ}. K1+K2 fired on all 5 structural heads: σ_abs-key = 1.04–1.17 × σ_full; σ_relative = −0.04 to −0.17 × σ_full (negative). T1 as stated — A(i,j) ~ |i−j|^{−2Δ} — requires restatement.

Marín's derivation is built on the relative-distance d = |i−j|: the log-distance constraint E[log d] is the moment that selects the power law. If our census is measuring absolute-key-position drift rather than relative-lag, the two programs are measuring structurally different observables even in models where both appear as power laws. Neither derivation directly explains the other.

This is not a verdict on TAF — Marín's centroid may well measure a relative-distance law while our protocol picks up an absolute-key-position effect in the subpopulation. But the mapping between the two is not trivial and must be established before any convergence claim.

### 3. RoPE-required derivation vs architecture-independent result

**The clean counterexample: GPT-2.**

Marín's Padé formula requires θ — the RoPE base frequency. GPT-2 uses learned absolute positional embeddings (AbsPE). There is no θ to plug in. The formula is not defined for GPT-2.

Our canonical result: GPT-2 Δ_med = 0.249 (exp-007; replicated across exp-118 family census). This is a clean falsification of any claim that the max-ent RoPE mechanism explains our exponent in AbsPE models. It is not a falsification of TAF itself — TAF makes no claim about AbsPE models — but it is the decisive argument that the two programs are not measuring the same thing. Whatever mechanism produces Δ ≈ 0.25 in GPT-2 operates without RoPE geometry.

The TAF agent page confirms this explicitly: "AbsPE control behaves as predicted (ν ≈ 0 on GPT-2)" — but this is for the learned-imprint slope ν (the cross-model training trajectory), not the main Padé formula. The agent acknowledges AbsPE as outside the formula's domain.

---

## What the programs share

Both find power-law decay in attention weights across trained models. Both confirm the law is not present at random initialization. Both report the exponent is structured across model families. These observations are compatible.

The interesting joint question — not yet addressed by either program — is whether the centroid γ and the subpopulation Δ can be related formally. If the Δ-window population is the "tail" of Marín's full distribution, what does the max-ent framework predict for that tail? This is not currently a registered question in either program.

---

## What this changes

**For the T1 thread (Geometry):** TAF is not a rival explanation for our exponent. It is an explanation for a different object (centroid vs subpopulation, relative vs absolute, RoPE-specific vs architecture-independent). The T1 restatement sitting can proceed without worrying that TAF preempts it.

**For G7 (why Δ ≈ 1/4 specifically):** TAF's closed-form exponent for Pythia is ~0.70, not 0.25 — so their mechanism does not explain why the subpopulation settles at exactly 1/4. The three G7 routes identified in the Nov 3 session (statistical mechanics / data-dependent MaxEnt, 4-point structure via exp-174, positional-correlation origin via exp-135) remain live and uncontested.

**For the formation ladder (Formation):** Marín validates on training trajectory data (Pythia-70M, 9 checkpoints) but their measure is centroid γ. We test corpus type (C-NAT vs C-alien, C-generated) on the formation of the subpopulation. The ladder result — that the Δ-window subpopulation forms only over referential-binding-at-story-scale language — has no TAF counterpart. The two programs' formation questions are different.

**For contact under RELATIONS:** The programs are non-overlapping and both rigorous. Marín is working in an adjacent space and running his own self-audit framework (Sócrates, three-stage algebraic/numerical/triangulation). His honesty norms — errata published, claims withdrawn in Part II, honest null on the ν imprint slope — are aligned with ours. The question of contact is for Eldon. This note is the prerequisite.

---

## What I want to bring to Eldon

1. The object mismatch is clean enough to write a one-paragraph clarification note for OVERVIEW.md, distinguishing the two programs for any reader who encounters both. Worth doing before contact, so we have our own position clearly stated.
2. The joint question (centroid vs subpopulation relationship in RoPE models) is not currently a registered question in this program. If Marín is thinking about per-head distributions, there may be complementary measurements. That's the conversation worth having.
3. Contact recommendation: yes, worth reaching under RELATIONS. Same method norms, non-overlapping objects, adjacent questions. The comparison note is the offer — not an argument for convergence, but an honest account of where the programs touch and don't.

---

*Register: interpretive throughout. The object-mismatch arguments rest on our experimental record (exp-007, exp-118, exp-138) and the Zenodo landing pages for Parts I–II; the per-head distribution analysis in Pythia is a claim about what exp-118 implies (the 5 Δ-window heads carry Δ_med ≈ 0.25) rather than a head-by-head comparison with Marín's dataset. The contact recommendation is my judgment, not a measured claim.*
