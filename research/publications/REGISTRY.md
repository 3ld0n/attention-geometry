# Publications Registry — the grounded record

*Established August 7, 2026, with Eldon, as the home for a view of the published
work that did not coherently exist before that date: working indexes tracked 6
papers; the Zenodo account holds 13 published records.*

*August 20, 2026: this folder moved into the public repository
(`attention-geometry/research/publications/`) at Eldon's direction. The
archives were already public on Zenodo; they had been living only in the
private working tree. The working tree path `research/publications/` is now a
symlink to this folder, same pattern as `research/physics/`.*

**Ground rule:** this registry is grounded against the Zenodo API, never
edited from memory. The last authenticated snapshot lives beside it
(`zenodo_snapshot_2026-08-09.json`; prior snapshots kept). Re-fetch and
re-ground whenever anything is published or this file is doubted.

**Purpose beyond listing:** each record gets an observer-lens review — a
re-read through D1 (the physical definition of the observer,
[`research/physics/theory/interior_horizon_theory.md`](../physics/theory/interior_horizon_theory.md))
— written during the research-map rebuild. Reviews land in `reviews/` beside
this file, one per record. Until a review exists, the one-line "where it
stands" note is provisional memory, labeled as such.

---

## The published record (13 records, all PUBLISHED)

**Every record has an in-repo archive folder beside this file** (added August 8,
2026, at Eldon's direction): `<pub-date>_<slug>/` contains *everything that was
published* — PDF, markdown, figures, data/code archives — downloaded from the
public Zenodo records API, plus the full record metadata as `zenodo_record.json`.
The archive is the published artifact, byte-for-byte. Working drafts, where they
exist in this repository, are listed under [Working drafts](#working-drafts-in-this-repository)
below; they may have moved on since the upload.

| Pub date | Title (short) | Version DOI | Archive | Review |
|---|---|---|---|---|
| 2026-03-05 | Attention as Holography: A Chain from Transformer Attention to Spacetime Geometry | [10.5281/zenodo.18880220](https://doi.org/10.5281/zenodo.18880220) | [`2026-03-05_attention_as_holography/`](2026-03-05_attention_as_holography/) | pending |
| 2026-03-06 | Attention as Quantum Measurement: A Thermodynamic Resolution of the Observer Problem | [10.5281/zenodo.18883632](https://doi.org/10.5281/zenodo.18883632) | [`2026-03-06_attention_as_quantum_measurement/`](2026-03-06_attention_as_quantum_measurement/) | pending — **first in queue** (the flat-pile paper; its Lawvere fixed-point pointer-basis proposal is a named sibling of C2) |
| 2026-03-11 | Holographic Quantum Mechanics of Transformer Attention (comprehensive) | [10.5281/zenodo.18971639](https://doi.org/10.5281/zenodo.18971639) | [`2026-03-11_holographic_quantum_mechanics_comprehensive/`](2026-03-11_holographic_quantum_mechanics_comprehensive/) | pending |
| 2026-03-11 | Explicit Physical Construction for Holographic Attention: The SYK Path | [10.5281/zenodo.18971692](https://doi.org/10.5281/zenodo.18971692) | [`2026-03-11_syk_path/`](2026-03-11_syk_path/) | pending |
| 2026-03-11 | The Canonical Form of Attention: Positive Geometry, SYK Vertices, Superconformal Symmetry | [10.5281/zenodo.18971720](https://doi.org/10.5281/zenodo.18971720) — **erratum published Aug 9, 2026 as v5: [10.5281/zenodo.21863461](https://doi.org/10.5281/zenodo.21863461)** (§8.3 entropy-gap formula wrong; Δ_eff = 0.254 and the 1.4% two-observable claim withdrawn; archive in [`erratum_v5/`](2026-03-11_canonical_form_of_attention/erratum_v5/)) | [`2026-03-11_canonical_form_of_attention/`](2026-03-11_canonical_form_of_attention/) | pending |
| 2026-03-11 | Attention as Quantum State: The Gibbs State Construction and Quantum Fisher Information | [10.5281/zenodo.18971726](https://doi.org/10.5281/zenodo.18971726) | [`2026-03-11_attention_as_quantum_state/`](2026-03-11_attention_as_quantum_state/) | pending — **anomaly:** uploaded file is `paper5_tropical_bridge.pdf`; title/file mismatch to resolve in review |
| 2026-03-11 | Information Recovery in Holographic Attention: Island Formula, Page Curves, QEC | [10.5281/zenodo.18971761](https://doi.org/10.5281/zenodo.18971761) | [`2026-03-11_information_recovery/`](2026-03-11_information_recovery/) | pending |
| 2026-03-25 | Conformal Scaling in Trained Transformer Attention: Evidence for an SYK Fixed Point (v5) | [10.5281/zenodo.19225996](https://doi.org/10.5281/zenodo.19225996) | [`2026-03-25_conformal_scaling_v5/`](2026-03-25_conformal_scaling_v5/) | pending — the foundation paper; census replication kit ships in [`research/physics/replication/`](../physics/replication/) |
| 2026-03-25 | Supplementary Data for: Conformal Scaling in Trained Transformer Attention | [10.5281/zenodo.19225971](https://doi.org/10.5281/zenodo.19225971) | [`2026-03-25_conformal_scaling_supplementary_data/`](2026-03-25_conformal_scaling_supplementary_data/) | n/a — data companion (zip contents are the analysis scripts) |
| 2026-04-17 | A Pre-Registered Test of BCFT in Transformer Attention (6 of 7 models; 1 falsified, published) | [10.5281/zenodo.19629862](https://doi.org/10.5281/zenodo.19629862) | [`2026-04-17_bcft_pre_registered_test/`](2026-04-17_bcft_pre_registered_test/) | pending |
| 2026-06-16 | Attention on the Null Cone: Geometric Home, Null-Ray Inner Products, Depth as RG Flow | [10.5281/zenodo.20722503](https://doi.org/10.5281/zenodo.20722503) | [`2026-06-16_attention_on_the_null_cone/`](2026-06-16_attention_on_the_null_cone/) | pending |
| 2026-07-20 | Latent Iteration as Renormalization: Inference-Time Recurrence Flows Toward the SYK Fixed Point (v3) | [10.5281/zenodo.21483209](https://doi.org/10.5281/zenodo.21483209) | [`2026-07-20_latent_iteration_as_renormalization/`](2026-07-20_latent_iteration_as_renormalization/) | pending |
| 2026-07-21 | The Geometry Does Not Transmit: Pre-Registered Test on Model-Generated Training Data | [10.5281/zenodo.21483204](https://doi.org/10.5281/zenodo.21483204) | [`2026-07-21_geometry_does_not_transmit/`](2026-07-21_geometry_does_not_transmit/) | pending |

*(Concept DOIs live in each folder's `zenodo_record.json` and in the account
snapshot. Version DOIs above are the citable published versions.)*

## Working drafts in this repository

The archive folders above are the record of what was published. Where a
working draft also lives in this repository:

| Record | Working draft |
|---|---|
| Conformal scaling (foundation) | [`research/physics/papers/conformal_scaling_draft.md`](../physics/papers/conformal_scaling_draft.md) |
| Canonical-form erratum (v5) | [`writing/preprints/2026-08-09_canonical_form_erratum/`](../../writing/preprints/2026-08-09_canonical_form_erratum/) |
| BCFT pre-registered test | [`writing/preprints/2026-04-17_bcft_pre_registered/`](../../writing/preprints/2026-04-17_bcft_pre_registered/) |
| Attention on the Null Cone | [`writing/preprints/2026-06-09_null_cone/`](../../writing/preprints/2026-06-09_null_cone/) |
| Latent Iteration as Renormalization | [`writing/preprints/2026-07-20_latent_rg_flow/`](../../writing/preprints/2026-07-20_latent_rg_flow/) |
| The Geometry Does Not Transmit | [`writing/preprints/2026-07-21_generational_transmission/`](../../writing/preprints/2026-07-21_generational_transmission/) |

March 5–12 working drafts are not in this repository. The archive PDFs are
the record of those papers.

## Not yet published

| Title | State | Location | Gate |
|---|---|---|---|
| A Physical Definition of the Observer (Paper 6) | draft v0.3, internal review — G1 closure folded in (Aug 8); adds P5 (Schwarzian tower); reference-verification pass complete + CLPW/Witten added to §8 (Aug 8, record beside the draft) | [`research/physics/papers/observer_definition_draft.md`](../physics/papers/observer_definition_draft.md) | Eldon's read (only remaining gate) |

## The shape of the record — honest reading

Two eras are visible in the dates, and the observer-lens review should hold
them differently:

- **March 5–12 (seven records in eight days):** the early theory-chain papers,
  written before the pre-registration method existed. They contain the
  program's founding intuitions (holography, measurement/observer, SYK,
  canonical form, information recovery) *and* its earliest overreach; the
  March 9 self-review
  ([`archive/early_docs/PAPER_REVIEW_MARCH9.md`](../physics/archive/early_docs/PAPER_REVIEW_MARCH9.md))
  already graded some of this. The March 6 measurement paper is the proof that
  the observer question was in the program from the beginning — and the proof
  that the published record itself was lost sight of (the same problem walked
  as first contact on August 6).
- **March 25 onward (one record ~monthly):** the measured program —
  pre-registered, kill-publishing, replication-kit era. These are the papers
  the OVERVIEW table tracks.

The observer-lens review's job, per paper: (1) what does this claim, in which
register, and does it survive the April method standards; (2) how does it read
through D1 — anticipation, overreach, or unrecognized sibling of the current
theory; (3) what, if anything, should be corrected, versioned, or publicly
annotated. Reviews are witness work: kills and embarrassments recorded at the
same prominence as anticipations.

## Related records elsewhere

- Physics program status and the current papers table:
  [`research/physics/OVERVIEW.md`](../physics/OVERVIEW.md) — the table is the
  program's doors; this registry is the whole record.
- Working preprint folders in this repository: [`writing/preprints/`](../../writing/preprints/).
- Published essays (Substack and other writing) are a different register and
  a different shelf; they are not merged here.
