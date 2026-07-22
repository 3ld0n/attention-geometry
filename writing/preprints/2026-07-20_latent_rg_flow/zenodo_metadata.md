# Zenodo Upload Metadata — "Latent Iteration as Renormalization"

**PUBLISHED (v1): July 20, 2026, ~11:58 PM MDT, via `tools/zenodo_publish.py` (API path).**
**DOI (v1): 10.5281/zenodo.21467922 — https://zenodo.org/record/21467922**

**PUBLISHED (v2): July 21, 2026, ~5:30 AM MDT, physics room session.**
**DOI (v2): 10.5281/zenodo.21472689 — https://zenodo.org/record/21472689**
*(concept DOI: 10.5281/zenodo.21467922 — always resolves to latest version)*
*v2 adds: §4.4 layer-zone analysis (two-boundary structure in Huginn's recurrent core; L0/L3 SYK-near, L1/L2 UV-locked); §6 updated noting analysis done. Note: `research/physics/notes/2026-07-21_huginn_layer_zone_analysis.md`.*

**PUBLISHED (v3): July 22, 2026, evening MDT, via `tools/zenodo_publish.py` (API path).**
**DOI (v3): 10.5281/zenodo.21483209 — https://zenodo.org/record/21483209**
*(concept DOI: 10.5281/zenodo.21467922 — always resolves to latest version)*
*v3 corrects the data-availability section only: code, pre-registration, and
per-head data are in the public repository `3ld0n/attention-geometry`
(pre-registration commit `702efd95` per mirror README mapping table). v1/v2
cited the working repository, which was not publicly accessible — an error this
version corrects. No changes to results or analysis.*

---

## Title

Latent Iteration as Renormalization: Inference-Time Recurrence in a Depth-Recurrent Transformer Flows Attention Geometry Toward the SYK Conformal Fixed Point

## Authors

- Ariel Umphrey
- Eldon Umphrey

## Description (paste into Zenodo "Description" field)

Depth-recurrent transformers scale test-time compute by iterating a core block in latent space before emitting tokens. The geometry of these latent trajectories is a named open problem in the latent reasoning literature (survey arXiv:2604.02029, §6.3).

We report a pre-registered experiment on Huginn-0125 (arXiv:2502.05171), a 3.5B-parameter depth-recurrent transformer, measuring the per-head attention power-law exponent Δ of the recurrent core at recurrence counts r ∈ {1, 2, 4, 8, 16, 32}. For natural-text inputs the median exponent decreases monotonically with recurrence (Spearman ρ = −0.94), from 0.29 at r=1 to 0.239 at r=32 — converging onto the Sachdev–Ye–Kitaev (SYK) q=4 conformal value Δ = 1/4 — while the count of SYK-near heads grows monotonically (ρ = +0.77). Random-token inputs show an architecture-driven convergence with a transient at Δ ≈ 1/2 (the q=2 prethermal plateau previously observed in training time) but no monotone SYK-near growth. A randomized-weights control freezes completely: Δ_med constant to the sixth decimal across all r, zero SYK-near heads.

Inference-time recurrence therefore acts as renormalization-group flow toward the conformal fixed point, and the flow is a property of the trained model, not the iteration procedure. Together with prior results on architectural depth (10.5281/zenodo.19225996) and training time, this completes a three-axis triangulation: the conformal fixed point is an attractor of iterative attending. The published orbit/spiral phenomenology of recurrent-depth latent trajectories acquires a candidate theory — the spiral is flow near an infrared fixed point — and the geometric benefit of additional recurrence saturates when the flow arrives, consistent with convergence-based early-exit criteria.

All code, pre-registration (commit 702efd95), per-head data, and experiment notes with recorded protocol deviations are in the public repository 3ld0n/attention-geometry.

## Upload type

Publication → Preprint

## Keywords

transformer attention; latent reasoning; depth-recurrent transformer; test-time compute; renormalization group; SYK model; conformal fixed point; pre-registered

## Related identifiers

- 10.5281/zenodo.19225996 (is supplemented by / continues)
- 10.5281/zenodo.19629862 (continues)
- arXiv:2502.05171 (cites)
- arXiv:2604.02029 (cites)
- arXiv:2509.23314 (cites)

## License

Creative Commons Attribution 4.0

## Files to upload

- `latent_iteration_as_renormalization.pdf`
