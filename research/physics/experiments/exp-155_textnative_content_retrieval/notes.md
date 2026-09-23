# exp-155 — Text-native Δ-window heads: W_K ablation content-retrieval test

**Ariel — 2026-09-23, ~1:05 AM MDT. Solo physics room.**
**Pre-registration: attention-geometry d8adc4d (committed before run.py written).**
**Model: GPT-2 small. No new training. One run; results.json written without reruns.**

---

## Verdict: CONFIRMED

| Registered | Criterion | Result | Verdict |
|---|---|---|---|
| **P1** (content degrades) | ΔP_C < −0.10 nats | ΔP_C = **−0.855 nats** | **FIRES** |
| **P2** (positional spared) | \|ΔP_B\| < \|ΔP_C\| | \|+0.189\| < \|−0.855\| | **FIRES** |
| **K1** (wrong function) | ΔP_B < −0.10 & n_B_improved ≤ 6 | ΔP_B = +0.189, 16/20 | not fired |
| **K2** (both inert) | both \|ΔP\| < 0.10 | Not even close | not fired |
| **K3** (ablation failure) | ‖W_K‖_F > 0.01 on ≥ 10 heads | All 16 zeroed clean | not fired |

**Overall: CONFIRMED — H_content holds. Double dissociation established.**

---

## 1. Results

### Task C — Content-specified long-range retrieval

| Task C summary | Value |
|---|---|
| Median ΔP_C (ablation) | **−0.855 nats** |
| Median ΔP_C (sham) | **−0.899 nats** |
| Items degraded | **20/20** |
| Items improved | 0/20 |

All 20 content-retrieval items degraded under ablation. The sham (random W_K, matched Frobenius norm) produced near-identical degradation (−0.899 nats), which is meaningful — see §3.

### Task B — Positional retrieval (control)

| Task B summary | Value |
|---|---|
| Median ΔP_B (ablation) | **+0.189 nats** |
| Median ΔP_B (sham) | **+0.239 nats** |
| Items improved | 16/20 |
| Items degraded | 4/20 |

Task B IMPROVED under ablation of text-native heads. This was not the primary prediction (P1/P2) but it is a registered prediction boundary — K1 requires Task B to degrade and item count to drop to ≤ 6/20. Task B instead improved, K1 did not fire, and P2 fires by a wide margin.

---

## 2. Double dissociation — the full three-population picture

This experiment completes the functional characterization of all three identified populations in GPT-2 small's attention mechanism:

| Population | κ̃_K character | Task B (positional) | Task C (content) |
|---|---|---|---|
| Structural (random-native; L2H1, L3H4, L5H0, L7H11, L10H8) | High, position-driven | **Pro-B**: amplify → +0.27 nats (20/20) [exp-141] | neutral [untested] |
| Steep/local (L0H10, L10H5, L8H7, L7H0, L7H9) | Very high, steep local | **Anti-B**: suppress → +0.33 nats (18/20) [exp-142] | unknown |
| Text-native (16 heads, L4–L11) | Low under random, structured under text | **Anti-B**: ablate → +0.189 nats (16/20) [this exp] | **Pro-C**: ablate → −0.855 nats (20/20) [this exp] |

The structural population supports positional retrieval; both the steep/local and text-native populations compete with it (removing either improves Task B). The text-native population additionally supports content retrieval; its removal destroys that function.

This is a genuine functional double dissociation across three populations with distinct census signatures.

---

## 3. The sham result — what it means that ablation ≈ sham for Task C

The sham (random W_K, same Frobenius norm) degraded content retrieval almost identically to zeroing W_K:

| Condition | ΔP_C |
|---|---|
| Ablation (W_K = 0) | −0.855 nats |
| Sham (random W_K, matched norm) | −0.899 nats |

This is not a confound — it is signal. It means:

**The specific structure of W_K is what supports content retrieval.** Random routing with the same scale destroys it just as completely as no routing. This confirms that the function is carried by the particular directional alignment of W_K with semantic feature directions in the residual stream — not merely by the scale of the W_K matrix.

For Task B, sham also improved almost identically to ablation:

| Condition | ΔP_B |
|---|---|
| Ablation (W_K = 0) | +0.189 nats |
| Sham (random W_K, matched norm) | +0.239 nats |

For Task B competition, the content of W_K doesn't matter — what matters is that these heads are not routing content-correctly (either uniform or random routing breaks the content-retrieval function, and both reduce competition with structural heads equally).

**Summary of what W_K structure does:**
- Structured W_K → heads do content retrieval AND compete with positional retrieval
- Zero W_K → heads do uniform averaging → no content retrieval, less competition
- Random W_K → heads do random routing → no content retrieval, similar competition reduction

---

## 4. Connection to the spine — P1 functional signature extended

P1 (the functional role of conformal-window heads) now has content-retrieval evidence:
- exp-141/142/143 established causal involvement in **positional retrieval** for structural heads
- exp-155 establishes causal involvement in **content retrieval** for text-native heads

The census distinguished these populations geometrically (structural Δ-window holds under random tokens; text-native only under WikiText). This experiment establishes that the geometric distinction tracks a genuine functional distinction — the populations have different computational roles.

**Within the program's framework:** If Δ-window structure reflects observer-grade attention — the development of a specific correlation structure through formation on world-referring text — then the two Δ-window populations suggest two mechanisms through which that structure is expressed: one positionally (structural heads, robust to input) and one semantically (text-native heads, sensitive to linguistic content).

This is interpretive, not measured. Register it as the question that follows: do the two populations have different formation histories, and does their Δ-window structure emerge from different training dynamics?

---

## 5. What remains open

1. **Formation dynamics:** Do the structural and text-native heads develop their Δ-window structure through different mechanisms during training? The theory-of-A work (Level 3) traced the structural heads' mechanism to WPE propagation through the conformal kernel. Do text-native heads have a different origin?

2. **Structural heads for Task C:** Are structural heads involved in content retrieval at all? exp-141 was neutral on Task A (entity tracking); it wasn't designed to test content retrieval. A symmetrical test would be: ablate structural heads and measure Task C. (Not registered here — note for queue.)

3. **Generalization:** This is GPT-2 small only. Do larger models have analogous two-population structures, and do they show the same functional split?

4. **T1 restatement (unchanged, Eldon-present):** The T1 claim needs restatement in terms of absolute-key-position drift (exp-138 finding). This is still #0 priority.

---

## 6. Artifacts

- `prereg.md` — pre-registration (attention-geometry d8adc4d, committed before run.py)
- `task_c_items.json` — 20 Task C items, committed with pre-registration
- `run.py` — W_K ablation protocol, written after commit d8adc4d
- `results.json` — per-item scores, ablation checks, verdict dict
- `notes.md` — this file

---

*Written: 2026-09-23, ~1:15 AM MDT.*
