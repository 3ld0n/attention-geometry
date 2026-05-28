# BCFT coverage audit + fp16 NaN alignment (exp-037) — May 27, 2026

## Question

Do incomplete April Modal BCFT JSONs (exp-025) reflect **fp16 attention failure** on deep layers — as exp-036 showed for Pythia-410m — rather than arbitrary job timeouts?

## Deviation

Queue primary remains **Apple sigmoid Δ** (~50 GB download; Eldon OK not given; `NO_GALA_DIR`). Worked exp-036 follow-up: systematic audit of all exp-025 files + quantitative fp16 probe on 410m.

## Coverage audit (all exp-025 JSONs)

| Model | Coverage | Last layer with heads | Expected layers |
|-------|----------|----------------------|-----------------|
| EleutherAI/pythia-410m | **62.2%** (239/384) | **14** | 24 |
| EleutherAI/pythia-1.4b | 87.0% (334/384) | **20** | 24 |
| EleutherAI/pythia-2.8b | 87.4% (895/1024) | **27** | 32 |
| EleutherAI/gpt-neo-2.7B | 89.4% (572/640) | **31** | 32 |
| Qwen/Qwen2-7B | 99.7% (782/784) | 27 | 28 |
| allenai/OLMo-7B-hf | **100%** | 31 | 32 |
| mistralai/Mistral-7B-v0.3 | **100%** | 31 | 32 |

**Finding:** Truncation is **not** a single universal layer index. Larger / deeper runs lose fewer layers proportionally until near the top (Qwen missing 2 heads on one layer). Pythia family shows the clearest deep cutoffs; 7B models mostly complete on Modal A100 fp16.

## fp16 probe (Pythia-410m, MPS, one forward, seq=512)

- **fp16:** L0–13 clean; L14 has **1.17%** NaN (still fits heads in BCFT); **L15+ 100% NaN**.
- **fp32:** all layers 0% NaN.
- **Alignment:** BCFT JSON last layer with data = **14**; first all-NaN layer = **15** → **confirmed**.

## Verdict

- **H1 (410m):** fp16 MPS causes BCFT truncation at L15+ — **confirmed** with layer-precise alignment.
- **H2 (all models same L*):** **falsified** — 1.4b truncates at L20, 2.8b at L27, etc. Same *mechanism* (invalid deep attention → skipped heads) likely; boundary is model/stack dependent.
- **Action for Modal re-runs:** fp32 or bf16 + assert `finite_fraction > 0.99` per layer before fitting. OLMo/Mistral JSONs are trustworthy as-is; Pythia/gpt-neo/Qwen partial files need selective re-run if cited.

## Next experiment

1. **Eldon OK** → sigmoid download + exp-007 Δ (unchanged queue primary).
2. Optional: fp16 probe on **pythia-1.4b** (expect first all-NaN ≈ L21 if same mechanism).
3. Optional: local fp32 BCFT re-run for **pythia-1.4b** only if RoPE Δ citations need full depth.

## Artifacts

- `research/physics/experiments/exp-037_bcft_coverage_fp16_audit/audit_bcft_coverage.py`
- `research/physics/experiments/exp-037_bcft_coverage_fp16_audit/probe_fp16_layer_nan.py`
- `research/physics/experiments/exp-037_bcft_coverage_fp16_audit/results.json`
- `research/physics/experiments/exp-037_bcft_coverage_fp16_audit/fp16_nan_probe.json`
