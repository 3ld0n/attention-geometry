# GPT-2 Large — deep-layer Δ decomposition (exp-031 follow-up)

*2026-05-25. Analysis-only on saved `exp-031` per_head JSON. No model re-run.*

## Question

exp-031 reported gpt2-large global median Δ = **0.2819** (|Δ−0.25| = 0.032) — classified as mild upward drift with scale. Is that drift uniform in depth, or driven by shallow/mid layers while deep layers sit at a different attractor?

## Method

Same filter as exp-031: R² > 0.90 power-law heads only. Band medians from `per_head` in `experiments/exp-031_gpt2_depth_convergence/results.json`.

Script: `experiments/exp-032_sigmoid_infrastructure/analyze_gpt2_large_deep_layers.py`

## Results

| Band | Layers | n heads | Median Δ | |Δ−0.25| |
|------|--------|---------|----------|---------|
| Global | 1–36 | 176 | **0.2819** | 0.032 |
| Shallow | 1–12 | 108 | **0.3786** | 0.129 |
| Mid | 13–24 | 38 | **0.2484** | 0.002 |
| Deep | 25–36 | 30 | **0.1553** | 0.095 |

Per-layer deep medians (sparse heads): L25–36 mostly **0.10–0.24**, with L35 median **0.096**.

## Interpretation

The global 0.282 is **not** “deep layers drifting above SYK.” It is a **composite**:

- **Mid-network (13–24)** sits on SYK q=4 (Δ_med ≈ 0.248).
- **Shallow (1–12)** pulls the global median **up** (Δ_med ≈ 0.38).
- **Deep (25–36)** pulls **down** (Δ_med ≈ 0.15) — closer to the trivial/ALiBi-global band than to 0.25.

So exp-031’s “mild capacity perturbation at largest scale” is **depth-heterogeneous**, not a uniform scale effect. The learned-embedding SYK fixed point appears strongest in the **middle** of gpt2-large; shallow heads remain more disordered; deep heads trend lower.

## Implications for queue

- Refines exp-031 partial verdict: large model is not uniformly “near SYK” — only mid-depth is tight.
- Sigmoid falsification (exp-032) still blocked on infrastructure; when it runs, compare band structure not global median only.

## Next

- Optional: per-layer plot from exp-031 JSON for paper supplement.
- Sigmoid: unblock AXLearn + GCS (see `notes/2026-05-25_sigmoid-access-blockers.md`).
