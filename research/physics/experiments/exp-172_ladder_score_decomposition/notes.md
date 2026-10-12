# exp-172 — notes (exploratory run, 2026-10-11)

Status: run complete, all gates pass. Run as an exploration, not a strict confirmatory test (Eldon, 2026-10-11:
"This exploration more than any strong predictions we're trying to test"). The prereg was committed before the run
(f30fb44) and the run followed it; there were no changes to the protocol.

## Result (one line)

In every distance-law head of the six formation-ladder models (3 natural, 3 sentence-shuffled), the fall-off of
the attention score with distance comes entirely from the gap between the two tokens. The part that depends on
absolute position contributes ~0.

## Numbers

- Models: CNAT s0/s1/s2 (natural), CNATshuf s0/s1/s2 (sentence-shuffled), 70M GPT-NeoX, step 2000.
- Heads analysed: 65 conformal heads (the census criterion): 42 in layer 0, 5 in layers 1-2, 18 deep (layers 3-5).
  Per-head table in results.json.
- Gates: K1 (reconstruct attention) max diff 1.2-1.5e-6; K2 (census Δ replicates stored Δ ±0.02) 0 failures in 65 heads;
  K3 (terms add exactly) ~1e-15; K4 (split-half agreement) 100% in all six models.
- Gap share G (fraction of the score slope from the gap term): every head 0.992-1.014.
  Absolute-key share A: every head between -0.011 and +0.006. Mixed and absolute-query terms also ≈ 0.
- Layer-0 positive control passes (G ≥ 0.8 in all heads, all six models).
- Verdicts: H1 (position-driven) false. H2 (gap-driven) true. H3 (control) true. H4 (natural vs shuffled differ in A by ≥0.25) false:
  pooled deep median A = +0.0001 (natural, n=12) vs -0.0055 (shuffled, n=6).

## Sanity checks (diag.py, post-hoc, exploratory)

- Mean-field slope vs real census attention slope (2Δ): median ratio 1.07, range 0.59-1.25 over 65 heads.
  In GPT-2 small this overshot ~2x (exp-165). Here the mean-field description is close to faithful, so the
  decomposition is describing the actual heads.
- Position-dependent part of q,k relative to the constant part: deep layers 10-14%, layer 0 40-54% (N=20 inputs).
  Much of this is sampling noise from averaging a small number of random inputs. Noise has no trend across positions, so
  it does not contribute to the slope; A≈0 is a statement about systematic position-dependence.

## What this does and does not show

1. These models use rotary position encoding. There the score depends on the gap by construction (the rotation
   between query and key is a function of gap). A gap-driven result is what the architecture predicts. The
   informative part is the absence of a hidden position-driven component: nothing in the learned q/k adds
   an absolute-position trend. In GPT-2 (learned absolute positions) that component carried 104-117%.
2. So this says nothing about "the fall-off is physics, not architecture". It says: in rotary models the
   distance-law heads (random-token measurement) are pure gap heads, natural and shuffled alike.
3. It does NOT discriminate natural from shuffled: the one place the prereg expected a difference (H4) is null and
   deep-head counts are small (12 natural vs 6 shuffled), so power is low.
4. The slopes are steep (Δ 0.5-1.2). Nothing here is near Δ=1/4. This does not touch the 1/4 claim.
5. Measured on random tokens (the census protocol). Heads that read real text and their decomposition are not covered.
6. Only heads that pass the census criterion are analysed; heads that fail it are not.

## Files

prereg.md, run.py, diag.py, results/results.json. Checkpoints (step 2000) in /tmp/ladder_ckpt (not committed; Modal volumes
exp062-data, exp091-data).
