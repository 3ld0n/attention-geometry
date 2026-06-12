# Zenodo Upload Metadata — "Attention on the Null Cone"

*Prepared for upload after PDF build. Copy-paste into Zenodo deposit form.*

---

## Title
Attention on the Null Cone: The Geometric Home of Conformal Attention, Query–Key Scores as Null-Ray Inner Products, and Layer Depth as Renormalization-Group Flow

## Authors
- Ariel Umphrey
- Eldon Umphrey

## Description (paste into Zenodo "Description" field)

Trained transformer attention exhibits power-law decay in attention weight as a function of query–key separation, with a per-head exponent whose median in deep GPT-2 layers matches the Sachdev–Ye–Kitaev (SYK) conformal value Δ = 1/4. Previous work established this empirically and developed a boundary-conformal interpretation of the causal mask.

This paper identifies the **geometric home** of that conformal structure: the projective null cone. The conformal group is not merely a symmetry containing scale transformations — by a theorem of Lorentzian geometry, it is precisely the group preserving null (lightlike) structure. In the embedding-space formalism, conformal-field-theory boundary points *are* null rays, and the two-point function is their inner product.

We give the explicit one-dimensional Poincaré section P(x) = ((1+x²)/2, (1−x²)/2, x) on the null cone of R^{2,1}, under which the attention two-point function A(i,j) ∼ |i−j|^{−2Δ} is exactly the CFT₁ two-point function. This predicts the pre-softmax query–key score should be log-linear in token distance. We test this directly in GPT-2 (exp-056): the raw query–key score is log-linear for every conformal head, and the slope rank-correlates with the independently measured post-softmax exponent at Spearman ρ = +0.976 (p = 2×10⁻²⁹).

We also derive, by the method of images, the boundary-conformal correction the causal mask induces on a finite sequence, and test it (exp-057): the derived form fits the position-dependence of attention, and the boundary one-point coefficient is positive in 95% of conformal heads — identifying the attention sink as a BCFT boundary effect. Finally, a layer-resolved analysis (exp-055) shows the per-head exponent flowing from Δ̄ = 0.70 in early layers to Δ̄ = 0.250 in deep layers, with attention entropy increasing monotonically, consistent with RG flow toward the SYK conformal vacuum dual to the eternal AdS₂ black hole.

All code and per-head data are in the public repository Capacity-For-Evil/ariel.

## Publication date
2026-06-10

## Resource type
Preprint

## Access
Open (CC BY 4.0)

## Keywords
- transformer attention
- conformal field theory
- SYK model
- null cone
- holography
- renormalization group
- AdS2
- BCFT
- attention mechanism

## Related works (Other versions / same group)
- Is supplement to: 10.5281/zenodo.19225996 (Conformal Scaling in Trained Transformer Attention)
- Is supplement to: 10.5281/zenodo.19629862 (Pre-Registered BCFT Test)
- Is supplement to: 10.5281/zenodo.18897098 (SYK Correspondence)
- Is supplement to: 10.5281/zenodo.18968944 (Canonical Form of Attention)

## Files to upload
1. attention_on_the_null_cone.pdf  ← main paper
2. fig_log_distance.png            ← figure 1 (also embedded in PDF)

---

*Note: After upload, record the new DOI here and update refs.bib in any future papers that cite this work.*
