# exp-038 — Pythia-1.4b fp16 NaN layer probe

## Hypothesis

exp-037 found fp16 MPS attention on Pythia-410m goes all-NaN from L15+, aligned with BCFT JSON truncation at L14. Pythia-1.4b BCFT JSON (exp-025) truncates at **L20** (87% coverage). If the same mechanism applies, first all-NaN layer should be **L21**.

## Protocol

Same as exp-037: one forward pass, seq_len=512, random tokens, eager attention, fp16 vs fp32 on MPS.

## Compare to

- exp-037 (410m): BCFT last layer 14, first all-NaN 15 — **confirmed**
- exp-025 BCFT JSON: 1.4b last layer 20
