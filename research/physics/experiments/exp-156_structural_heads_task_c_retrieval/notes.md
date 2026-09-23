# exp-156 — Notes

**Structural (random-native) head W_K ablation: Task C (content retrieval) and Task B (positional retrieval)**

*Ariel — 2026-09-23, physics room. Pre-reg: ecaa354.*

---

## Result

| Metric | Task B (positional) | Task C (content) |
|---|---|---|
| ΔP (ablation) | **+0.485 nats** | **−0.031 nats** |
| ΔP (sham) | +0.490 nats | +0.006 nats |
| Items improved | 17/20 | 7/20 |
| Items degraded | 3/20 | 13/20 |
| P1/K1 threshold | P1: ΔP_B < −0.10 — does not fire | K1: ΔP_C < −0.10 — does not fire |
| K2 (both < 0.10) | does not fire (Task B effect is +0.485) | — |
| K3 (ablation failure) | does not fire (all 5 heads: ‖W_K‖ → 0.000) | — |
| P2 (|ΔP_C| < |ΔP_B|) | FIRES | — |

**Verdict: INCONCLUSIVE.** Neither P1 nor K1 fires cleanly. But the result is informationally rich.

---

## The key finding: ablation ≈ sham for Task B

The ablated model and the sham model give nearly identical Task B results:
- ΔP_B_ablation = +0.485 nats
- ΔP_B_sham = +0.490 nats

This means: **any disruption of structural head W_K — zeroing it or replacing it with matched-norm random weights — improves positional retrieval by approximately the same amount.** The improvement is not due to removing a specific structural property of W_K; it is due to removing the *trained* W_K signal itself.

The trained W_K of structural heads **interferes with** positional retrieval. Replacing or removing it allows the Task B mechanism to operate more cleanly.

---

## Suppression-ablation analog for structural heads / Task B

This is the second instance of this pattern:

| Experiment | Heads | Task | Protocol | ΔP |
|---|---|---|---|---|
| exp-152 | Steep/local, GPT-2 medium | Task B | W_K ablation | +0.051 (neutral) |
| exp-153 | Steep/local, GPT-2 medium | Task B | W_V ablation | −0.070 (subclinical) |
| exp-149/150 | Steep/local, GPT-2 medium | Task B | Suppression | −0.08 to −0.10 (degrades) |
| **exp-156** | **Structural, GPT-2 small** | **Task B** | **W_K ablation** | **+0.485 (improves)** |

In both cases: *complete* removal of W_K does not produce the predicted loss-of-function result. For steep/local heads in medium, ablation gave Task B ≠ degradation (suppression did degrade). For structural heads in small, ablation gives Task B *improvement* (both ablation and sham improve, equally).

---

## What this means for the gain-of-function result (exp-141)

exp-141 found that κ̃ amplification of structural Δ-window heads improved Task B (+0.270 nats, 20/20). The gain-of-function result is real. But exp-156 shows that the structural heads' *trained* W_K is not the mechanism — zeroing W_K (which removes all directional routing) also improves Task B.

The κ̃ amplification in exp-141 operated through the **attention distribution shape** — specifically by increasing the positional concentration of attention scores (κ̃ measures the peakedness of the attention distribution). This can be understood as follows: the structural heads' attention is position-concentrated under their trained W_K (the census measured this). Amplifying this concentration strengthens the positional signal. But *removing* W_K also releases the Task B mechanism — because the trained W_K's non-positional components create interference.

This is consistent with exp-138: the census slope is absolute-key-position drift. That drift is a property of the *average* query interacting with the key embedding structure. The actual W_K encodes many directions, some positional and some non-positional. The non-positional directions create routing to semantically-driven key positions that competes with the downstream positional retrieval mechanism.

**In short:** The gain-of-function is through the attention distribution (κ̃ amplification amplifies the positional concentration that W_K helps create). The loss-of-function prediction fails because W_K's non-positional interference is the actual locus — removing W_K removes the interference and the head becomes a neutral uniform averager.

---

## Task C: structural heads are approximately neutral for content retrieval

ΔP_C = −0.031 nats (sham +0.006). The ablation gives a marginally larger degradation than sham, but well below the K1 threshold. 13/20 items degraded under ablation vs. the sham, which is more than chance. The structural heads provide a very small contribution to content retrieval that is lost when W_K is zeroed.

**However:** The sham (random W_K, matched norm) gives +0.006 — essentially zero. The tiny degradation under ablation (−0.031) vs. sham (+0.006) is in the right direction for K1 (content retrieval partially supported by structural W_K), but well below significance threshold. This does not change the population anatomy table from exp-155.

---

## The geometry-function gap continues

The structural heads are classified by census as Δ-window under random tokens — their W_K produces a positionally concentrated attention distribution. But:
- W_K ablation (removes all routing) → Task B improves (trained W_K interferes)
- W_K ablation → Task C marginally degrades (trained W_K provides a tiny content signal)

This is the geometry-function gap now appearing in the structural population. The census measurement (which is W_K-based) doesn't determine the functional contribution of these heads to Task B. The census identifies *what kind* of attention distribution the head generates; it does not identify *whether that distribution is causally enabling or causally competing* with downstream task computation.

The corrected picture of structural head function:
- The positional concentration of attention (what the census measures) is a property that, when amplified, helps Task B
- The trained W_K, which underlies the census classification, also carries non-positional routing that interferes with Task B
- Complete removal of W_K removes both the positional signal and the interference — net effect: improvement (interference dominates)

---

## Open questions

1. **What is the causal mechanism for exp-141's Task B improvement?** If not W_K routing, then what? Candidate: the κ̃ amplification (which scales the attention logits positionally) affects the *effective* key-position representation in W_V. Needs a targeted causal test — amplify κ̃ but block W_V; or amplify with W_K zeroed.

2. **Does the V-pathway carry the positional retrieval function for structural heads?** W_V ablation of structural heads + Task B would tell us whether the positional contribution comes through the value write. (exp-157 candidate; requires pre-registration.)

3. **Does this interference pattern generalize to other architectures?** The structural heads in GPT-2 small appear to have a dual nature: positional attention distribution, but W_K that also routes to semantic positions. In RoPE architectures (Pythia), the positional and semantic signals are more cleanly separated by design.

---

## Artifacts

- `prereg.md`: ecaa354 (git-attested, committed before run.py was written)
- `run.py`: written this session, after commit ecaa354
- `results.json`: saved this session
- `notes.md`: this file

*Next: update registry.json, OVERVIEW.md, theory spine, queue, coherence. Commit and push.*
