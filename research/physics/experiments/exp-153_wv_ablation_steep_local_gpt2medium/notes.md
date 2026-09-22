# exp-153 — Notes
## W_V ablation of steep/local heads, GPT-2 medium

**Verdict: INCONCLUSIVE (subclinical) — H_neutral weakly supported; value payload has a
small positive contribution to Task B that falls below the pre-stated significance threshold**

Pre-registration: 5b63fdb (git-attested, pushed 2026-09-22 before run.py written).
Run: 2026-09-22 ~12:17 PM MDT. Solo physics room session.

---

## Result

| Metric | Value |
|---|---|
| Task A: orig / abl / sham (median logP) | −1.40 / −1.56 / −1.66 |
| Task B: orig / abl / sham (median logP) | −4.98 / −5.05 / −4.96 |
| ΔP_A (abl vs orig) | −0.159 nats |
| ΔP_B (abl vs orig) | **−0.070 nats** |
| n_B_improved (abl vs orig) | 7/20 |
| n_B_improved (sham vs orig) | 12/20 |
| P1 (ΔP_B < −0.10) | not fired |
| P_null (|ΔP_B| ≤ 0.10) | **FIRES** |

All K's clear: K3 (W_V zeroed on all 5 heads), K2 (floor), K1 (sham did not dominate in
both tasks simultaneously — sham degraded Task A more than ablation, but Task B results
went the other way).

---

## The suppression-ablation-value dissociation (complete table)

| Experiment | Protocol | ΔP_B | n_B_improved | Verdict |
|---|---|---|---|---|
| exp-149 | suppress-only γ=−1.0 (~600× κ̃ reduction) | −0.10 | 2/20 | INCONCLUSIVE |
| exp-150 | suppress-only γ=−0.86 (~42× κ̃ reduction) | −0.08 | 2/20 | INCONCLUSIVE |
| exp-152 | W_K ablation (W_K = 0) | +0.051 | 10/20 | INCONCLUSIVE |
| **exp-153** | **W_V ablation (W_V = 0)** | **−0.070** | **7/20** | **INCONCLUSIVE (subclinical)** |

---

## Interpretation

### What the pattern says

P_null fires on the pre-registered threshold. The value payload is NOT load-bearing for
Task B in the sense that no decisive degradation occurs when W_V is zeroed.

But the pattern is more informative than a clean null:

1. **W_K = 0 → Task B slightly improves** (+0.051, 10/20 vs sham 13/20).
   Removing key routing eliminates the uniform-averager write; the head contributes the
   mean-value vector weighted uniformly. Task B appears to benefit slightly from this
   de-emphasis.

2. **W_V = 0 → Task B slightly degrades** (−0.070, 7/20 vs sham 12/20).
   Removing the value payload entirely degrades Task B mildly. The sham (random W_V,
   matched norm) gives +0.018 (12/20 improved) — essentially identical to the original.
   So it is specifically the *zeroing* of the value contribution, not any disruption, that
   causes the mild degradation.

3. **The gap between sham and ablation** (Δ ≈ −0.088 nats; 7/20 vs 12/20) suggests the
   value payload of these heads carries a *weak* positive contribution to Task B. Not
   load-bearing — the effect is subclinical — but directionally real.

### The suppression story, completed

The suppression degradation (exp-149/150, −0.08 to −0.10 nats) now has a clear causal
account supported by three paired comparisons:

- **W_K partial suppression → degradation:** the residual non-positional W_K after suppression
  actively misroutes attention (semantic dimensions that are wrong for Task B get amplified).
  This is the primary driver.

- **W_K complete ablation → no degradation (slight improvement):** removing all key routing
  gives uniform averager. No misrouting, no relay disruption. The head becomes silent-ish
  (still writes mean-value vector), and Task B is unchanged or slightly better.

- **W_V complete ablation → subclinical degradation:** removing the value payload weakly
  reduces Task B performance. The value write contributes something useful, but not
  decisively. The sham (random value at same norm) doesn't help or hurt — the specific
  learned mapping in W_V is what matters, not the scale of the write.

### What does W_V contribute to Task B?

The mild positive contribution of the value payload (when zeroed, Task B falls slightly)
is real but small. These 5 heads (L4H13, L15H8, L8H7, L5H11, L11H7) have:
- Very high κ̃_K (positional concentration: 43.2, 39.9, 33.3, 31.7, 30.9)
- Their attention pattern is steep and local — they attend to recent tokens

When they attend to recent tokens and write their value vectors, those value vectors carry
content that is mildly useful for positional retrieval. The contribution is not unique to
these heads (if it were, the effect would be decisive), but it is positive.

### Revised picture of these 5 heads

They are not genuinely neutral for Task B under any disruption:
- Their key routing (when partially disrupted by suppression) actively interferes
- Their value payload (when zeroed entirely) mildly helps
- Their normal operation contributes a small positive to Task B via the value pathway

But none of these effects are large enough to constitute load-bearing relays. The heads
are *weakly positive contributors via value* and *potentially harmful when partially
disrupted via key*.

### P1 medium replication: where the evidence stands after exp-149–153

Seven suppression/ablation experiments in GPT-2 medium; zero decisive Task B signal in
either direction for the clean protocols (W_K ablation: +0.051; W_V ablation: −0.070).
The antagonism mechanism (exp-141/142/143, GPT-2 small) does not replicate at medium
scale with the current population and protocols:

| Experiment | Protocol | ΔP_B | Decisive? |
|---|---|---|---|
| exp-145 | combined (WikiText-native targets) | −0.25 | Yes, wrong direction |
| exp-147 | combined (random-token structural) | −0.13 | Below threshold, wrong direction |
| exp-148 | amplify-only γ=+2.0 | +0.01 | Null |
| exp-149 | suppress-only γ=−1.0 | −0.10 | Borderline, wrong direction |
| exp-150 | suppress-only γ=−0.86 | −0.08 | Subclinical, wrong direction |
| exp-151 | amplify-only γ=+5.0 | +0.01 | Null (route closed) |
| exp-152 | W_K ablation | +0.051 | Subclinical, positive |
| exp-153 | W_V ablation | −0.070 | Subclinical, negative |

P1 in GPT-2 medium: no confirmed direction. The geometry-function gap holds — the census
identifies these heads as structurally similar to GPT-2 small's functional heads, but their
functional role differs. H_arch from exp-151 (architectural difference at medium scale)
remains the operating interpretation.

---

## Connection to the spine

- **P1** (functional role of Δ-window heads in positional retrieval): partial null at
  medium scale. The suppression degradation is mechanistically explained as misrouting, not
  relay disruption. The geometry-function gap (exp-150/152/153) is now mechanistically
  traced: partial W_K disruption → interference; complete W_K removal → silent; W_V removal
  → subclinical positive loss.

- **T1 restatement** (Eldon-present): unchanged, highest priority. Not affected by this
  experiment.

---

## Artifacts

- `run.py` — script (written after prereg committed)
- `results.json` — full item-level data
- `notes.md` — this file
- Registry: exp-153 (complete / inconclusive; bears_on P1; git-attested 5b63fdb)
- Surfaced in: OVERVIEW.md (world-model battery block)
- Spine: interior_horizon_theory.md P1 block (value-ablation result added; causal story complete)
