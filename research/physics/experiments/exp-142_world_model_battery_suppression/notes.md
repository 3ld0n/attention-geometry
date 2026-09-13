# exp-142 — World-model battery suppression: steep/local heads (Route B)

**Ariel — September 13, 2026, ~12:40 AM MDT. Solo.**
**Pre-registration: attention-geometry fc5c175 (pushed before run.py written).**
**One run; results.json written without reruns.**

---

## Verdict: INCONCLUSIVE by pre-registered criteria — P1 falsified by direction

| Registered | Criterion | Result | Verdict |
|---|---|---|---|
| **P1** retrieval degrades | ΔP_B < −1.0 nats AND ΔP_B < ΔP_A − 1.5 nats | ΔP_B = **+0.33 nats** (OPPOSITE direction) | **NOT FIRED** |
| **P2** tracking preserved | |ΔP_A − ΔP_A_sham| < 0.5 nats | |−0.13 − 0.00| = 0.13 < 0.5 | **FIRES** |
| **P3** null | both within 0.5 nats of sham | A: 0.13 ✓; B: |0.33 − (−0.06)| = 0.39 ✓ (barely) | **FIRES** |
| **K1** | sham effect ≥ sup effect on both | sup: |A|=0.13, |B|=0.33; sham: |A|=0.00, |B|=0.06 | **NOT FIRED** |
| **K2** | baseline < −10 nats | orig_A = −1.38, orig_B = −5.62 | **NOT FIRED** |
| **K3** | κ̃_K(sup) > 1.0 on ≥ 2/5 | 0/5 above threshold | **NOT FIRED** |

**Overall: INCONCLUSIVE** — P1 not fired (direction wrong). P3 technically fires but is not
the honest description of the finding (Task B moved substantially relative to sham).

---

## 1. What the suppression did

κ̃ confirmed suppressed on all five target heads:

| Head | κ̃_K baseline (exp-137) | κ̃_K after sup | κ̃_K after sham |
|---|---|---|---|
| L0H10 | 21.669 | **0.449** | 18.374 |
| L10H5 | 14.151 | **0.039** | 12.876 |
| L8H7  | 9.182  | **0.022** | 8.587  |
| L7H0  | 7.321  | **0.021** | 6.894  |
| L7H9  | 6.171  | **0.016** | 5.853  |

Note: L0H10's κ̃ dropped to 0.449 rather than near 0 — the layer-0 positional field
is broader, so the projection is less complete. Still 48× reduction. The other four
drop to near 0 (99.7%+ reduction). K3 did not fire. The suppression worked.

Sham γ_sham values (5.6, 3.3, 3.0, 2.6, 2.6) applied in the orthogonal complement —
the sham matched the Frobenius norm change but in a direction that doesn't touch
positional gain. Sham was flat on both tasks (A: Δsham=0.00 nats; B: Δsham=−0.06 nats).

---

## 2. The finding — suppression IMPROVES Task B

| Metric | Original | Suppressed | Sham | Δ(sup−orig) | Δ(sham−orig) |
|---|---|---|---|---|---|
| Task A median log-prob | −1.385 | −1.524 | −1.385 | **−0.139** | 0.000 |
| Task B median log-prob | −5.621 | −5.290 | −5.680 | **+0.331** | −0.059 |

**Item-level (suppressed vs original):**
- **Task B: 18/20 items improved.** Only 2/20 degraded.
- **Task A: 11/20 items degraded.** 9/20 improved. Mixed.
- **Sham (both tasks): 10/20 improved, 10/20 degraded.** Pure null on both.

The suppression moved Task B in the **improvement direction**, not the degradation direction
the hypothesis predicted. Task A shows a small degradation (−0.13 nats, within 0.5 nats
of sham, P2 fires) — the steep/local heads help entity tracking slightly.

---

## 3. The direction inversion — what this means

The pre-registered hypothesis was wrong: removing the steep/local heads' positional
read does **not** degrade positional retrieval. Instead, it improves it.

Three facts together:
- **exp-141**: amplifying Δ-window heads → Task B improves (20/20 items, +0.27 nats)
- **exp-142**: suppressing steep/local heads → Task B also improves (18/20 items, +0.33 nats)
- Both sham conditions are flat on Task B.

Both manipulations improve Task B. This means the two populations are **antagonistic** with
respect to positional retrieval:

**The steep/local heads suppress or compete with the Δ-window mechanism for positional
retrieval.** When you remove the competition (exp-142: suppress steep/local), Task B
improves. When you strengthen the retrieval mechanism (exp-141: amplify Δ-window), Task B
also improves.

This is consistent with the following model:
- Steep/local heads (κ̃_K = 6–22): short-range, high-gain positional readers. They attend
  preferentially to nearby positions. For the Task B items ("Colors: red, blue, green, yellow.
  The second color is..."), these heads pull attention toward nearby tokens rather than doing
  the long-range lookup to the specific position in the list.
- Δ-window heads (κ̃_K = 0.05–0.33): low-gain, broad positional readers. They process
  positional structure more gently. Their amplification helps the model look to the right
  position in the list.
- When steep/local heads dominate the positional signal, they may actively mislead the
  positional lookup by focusing on nearby context instead of the target position.

The result is: suppressing the short-range competitors improves the long-range lookup.

---

## 4. Entity tracking (Task A)

Entity tracking was slightly degraded by suppression (−0.13 nats, 11/20 degraded). P2
fires (within 0.5 nats of sham: |−0.13 − 0.00| = 0.13 < 0.5). But the effect is
non-null at item level — the steep/local heads provide a small benefit to entity tracking.

This is the expected behavior for local positional heads: entity-state tracking ("was the
lamp on or off?") benefits from nearby context attention, which is exactly what local heads
provide. Suppressing them reduces this slightly.

**The asymmetry between tasks is now fully explained:**
- Steep/local heads help Task A (entity tracking — nearby context) and hurt Task B (positional
  retrieval — long-range position lookup)
- Δ-window heads help Task B (confirmed by amplification) and are neutral to Task A

---

## 5. What remains to be tested

The combined picture from exp-140/141/142 paints a consistent dissociation, but the
pre-registered median thresholds were not met in any experiment. The item-level evidence is
decisive in direction, but the magnitudes are small relative to the large residual uncertainty
in GPT-2's positional retrieval.

Three options:
1. **Stronger γ for amplification** (exp-141 Route A): try γ=4 on Δ-window heads
2. **Combined manipulation**: simultaneously amplify Δ-window AND suppress steep/local —
   the two effects should be additive if the mechanism is right (pre-register before running)
3. **Different model**: GPT-2 small is a poor positional retriever at baseline. A larger
   model (GPT-2 medium or large) where the mechanisms are more developed might show larger
   effects. Or a model with stronger positional structure.

**The antagonism hypothesis** (exp-142's finding) makes a specific prediction for a combined
manipulation: amplify Δ-window (exp-141 direction) + suppress steep/local (exp-142 direction)
simultaneously → additive or super-additive improvement in Task B. This would be the cleanest
test. Pre-register before writing the run.

---

## 6. Artifacts

- `prereg.md` — registration (attention-geometry fc5c175, pushed before run.py)
- `run.py` — suppression protocol: γ = −1.0, W_K_sup = W_K − W_K_proj
- `results.json` — per-item scores, kappa values, verdict dict
- `notes.md` — this file
