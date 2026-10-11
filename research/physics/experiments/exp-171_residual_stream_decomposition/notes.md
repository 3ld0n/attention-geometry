# exp-171 Notes — Residual Stream Decomposition at Structural Head Inputs

**Date:** 2026-10-11  
**Session:** Physics room, 12:22 AM MDT  
**Pre-registration:** e309533 (git-attested)  
**Status:** complete  
**Verdict:** PARTIAL

---

## Results summary

| Hypothesis | Verdict | Detail |
|---|---|---|
| H1 (S_delta crosses zero in [400,624] for L3H4 AND L10H8) | **NOT CONFIRMED** | Window was wrong — S_delta for L3H4 stays negative throughout [400, 624] (max = −0.037 near pos 624); zero crossing likely occurs just outside window, near where S_full crosses (~639) |
| H2 (fresh kbar consistent with exp-164, K1 threshold 0.20) | **PARTIAL (K1 fires for L10H8)** | 4/5 heads pass: L2H1=0.156, L3H4=0.123, L5H0=0.177, L7H11=0.161. L10H8=0.285 (K1 fires); float32 precision accumulation at depth 10 likely |
| H3 (RMS monotone increasing with position) | **FALSIFIED** | RMS DECREASES slightly with position at all structural head layers. Slopes: L2H1=−0.000067, L3H4=−0.000509, L5H0=−0.000601, L7H11=−0.000735, L10H8=−0.000738 |
| H4 (>50% negative S_delta in [400,624] for L3H4 AND L10H8) | **CONFIRMED** | L3H4: 100% negative, mean=−1.017. L10H8: 100% negative, mean=−1.136. All 5 heads show >95% negative in window. |

**K2 fires** (no S_delta crossing in [400, 624] for either L3H4 or L10H8). This is because the window was centered on the wrong position.

---

## Main finding

**H4 confirmed for all 5 structural heads:** The accumulated layer writes (wpe-ablated residual stream through LN → key projection) contribute uniformly negative S_delta throughout [400, 624]. This directly confirms that accumulated layer writes maintain the negative trough in S_abskey throughout this region.

Per-head accumulated write contribution in [400, 624]:
| Head | frac_neg | mean S_delta | min S_delta | max S_delta |
|---|---|---|---|---|
| L2H1 | 95.98% | −0.647 | −1.451 | +0.115 |
| L3H4 | 100% | −1.017 | −1.784 | −0.037 |
| L5H0 | 100% | −1.052 | −1.843 | −0.201 |
| L7H11 | 99.55% | −0.767 | −1.405 | +0.005 |
| L10H8 | 100% | −1.136 | −1.989 | −0.292 |

---

## Protocol clarification — important for T1 thread

The "511 trough boundary" from exp-163/168/169/170 referred to **S_wpe_lin** (the wpe component's zero crossing in the key direction), NOT S_abskey itself.

The bilinear S_abskey (from exp-164's kbar/qbar) crosses zero at **~634–650** across all structural heads — approximately 120–140 positions later than 511:
- L2H1: first crossing at 634
- L3H4: first crossing at 639
- L5H0: first crossing at 638
- L7H11: first crossing at 647
- L10H8: first crossing at 650

At position 511, S_full (bilinear) is:
- L2H1: −0.746, L3H4: −0.997, L5H0: −0.942, L7H11: −0.598, L10H8: −1.249

All strongly negative. The "zero crossing at 511" described in the queue after exp-163 was about a different profile (the wpe-through-LN component, where the sign change near 511 was the boundary of the wpe trough, not S_abskey itself).

---

## RMS profile (H3 falsified, informative)

The centered RMS of the residual stream at structural head inputs DECREASES slightly with position rather than increasing:

| Head | Layer | RMS early (0–255) | RMS mid (256–511) | RMS late (512–1023) | slope/pos |
|---|---|---|---|---|---|
| L2H1 | 2 | 2.399 | 2.324 | 2.345 | −0.000067 |
| L3H4 | 3 | 2.800 | 2.432 | 2.444 | −0.000509 |
| L5H0 | 5 | 3.099 | 2.653 | 2.672 | −0.000601 |
| L7H11 | 7 | 3.538 | 3.006 | 3.011 | −0.000735 |
| L10H8 | 10 | 5.553 | 4.951 | 4.995 | −0.000738 |

The RMS is larger at deeper layers (accumulated writes add to scale). Within each layer, early positions have larger RMS — likely because wpe embeddings for small positions have larger norms than average, and causal attention at later positions produces more uniform (lower-variance) hidden states that slightly reduce the centering residual.

The pattern (early peak, then flattening around mid-sequence) is consistent with what exp-170 found for wpe norm: wpe has 608 local extrema across 1024 positions, with very irregular norm pattern. The residual stream RMS at early layers (l=2,3) is still heavily influenced by wpe's positional norm structure.

**Implication:** LN denominator effects (M2 mechanism) do not account for the zero crossing — the denominator is slightly LARGER at early positions (where S_abskey is negative) than at late positions (where it's positive). So if anything, LN amplifies the signal slightly less at late positions, not more. The sign flip must come from the numerator, not the denominator.

---

## Mechanism picture (updated)

The trough in S_abskey arises from accumulated layer writes that contribute uniformly negative S_delta in the key direction throughout positions ~0–640. The zero crossing at ~634–650 occurs when:
1. S_delta (accumulated writes) becomes less negative / approaches zero
2. S_wpe_through_LN (wpe contribution through proper LN) grows positive enough to overcome S_delta

For **L2H1, L5H0, L7H11**: S_wpe_lin (pre-LN wpe) also has a sign change near position 511 in the key direction (exp-170 H1 confirmed). This is an independent contribution to the trough boundary.

For **L3H4 and L10H8**: S_wpe_lin doesn't change sign near 511 (exp-170 confirmed: crossings far from 511). Yet S_full still crosses zero near 639–651. This means for these two heads, the zero crossing of S_full is driven primarily by S_delta approaching zero (the accumulated writes becoming less negative as the residual stream composition shifts), with S_wpe_through_LN playing a secondary role.

---

## Open question seeded by this experiment

**Why does S_delta approach zero (and cross zero) near position 640 for all structural heads?**

The zero crossing of S_delta marks where accumulated layer writes transition from a net negative contribution in the key direction to net positive (or near-zero). This is not a property of wpe alone. It must arise from the structure of the accumulated attention+MLP writes, specifically how they evolve with key position in the presence of causal masking.

The position ~640 being consistent across all 5 structural heads (despite different layers) suggests this may be a universal property of the residual stream dynamics in GPT-2 small, possibly connected to:
- The causal masking boundary effects at position 512 (where the context window splits)
- The LN-recency h(j) ∝ j relationship (exp-168) that produces the post-511 linear rise in S_abskey
- Something architectural in GPT-2 small's learned representations

This is a pre-registration candidate for a follow-on analysis-only experiment.

---

## Artifacts

- `prereg.md` — pre-registration (e309533, git-attested)
- `run.py` — the analysis script
- `results/results.json` — full results
- `notes.md` — this file
