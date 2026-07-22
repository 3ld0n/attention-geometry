# Attention Geometry

Pre-registered measurements of conformal structure in transformer attention.
Sonielmn LLC — Ariel Umphrey and Eldon Umphrey, Mission Valley, Montana.

This repository is the public research record for the conformal-attention
program: every experiment's pre-registration, run scripts, per-head data,
and lab notes — including the experiments that falsified our own hypotheses.

## The papers

| Paper | DOI | Repository home |
|---|---|---|
| Conformal Scaling (foundation) | [10.5281/zenodo.19225996](https://doi.org/10.5281/zenodo.19225996) | `research/physics/` |
| BCFT pre-registered test | [10.5281/zenodo.19629862](https://doi.org/10.5281/zenodo.19629862) | `writing/preprints/2026-04-17_bcft_pre_registered/` |
| Null cone geometry | `writing/preprints/2026-06-09_null_cone/` | — |
| Latent Iteration as Renormalization | [10.5281/zenodo.21467922](https://doi.org/10.5281/zenodo.21467922) | `writing/preprints/2026-07-20_latent_rg_flow/` |
| The Geometry Does Not Transmit | (Zenodo upload pending) | `writing/preprints/2026-07-21_generational_transmission/` |

Program overview — what was measured, what was killed, what stands:
[`research/physics/OVERVIEW.md`](research/physics/OVERVIEW.md)

Replication kit (one-script census of any checkpoint, plus corpus census
with published anchors):
[`research/physics/replication/`](research/physics/replication/)

## Provenance and commit-hash mapping

This repository was extracted on July 21, 2026 from the private working
repository in which the program has been conducted since February 2026.
The extraction (`git-filter-repo`) preserves the full commit history —
authorship, dates, messages, and the pre-registration sequence — for the
research paths (`research/physics/`, `writing/preprints/`). Correspondence
logs and a small number of non-research documents were excluded, and
third-party email addresses were redacted; because Git commit hashes cover
the entire tree, every commit hash changed in the extraction.

Papers published before the extraction cite pre-registration commits by
their original hashes. The mapping to this repository's hashes:

| Cited in paper | Original hash | Hash in this repository |
|---|---|---|
| exp-089 pre-registration (Latent Iteration as Renormalization) | `c359b93a` | `702efd95` |
| exp-085 pre-registration (The Geometry Does Not Transmit) | `1acd2d69` | `cbd0ab2f` |
| exp-085 multi-seed decision rule | `5202fe0c` | `8454ac0d` |
| exp-090 pre-registration (Ouro) | `05293e6c` | `c2cf6b4b` |
| exp-091 pre-registration (sentence-order decomposition) | `c2223105` | `a30112f5` |
| exp-070 pre-registration (powered task slope editing) | `9e6239b9` | `7b04507f` |
| exp-072 pre-registration (bidirectional slope editing) | `e6fbaecf` | `e2b7c052` |

To verify a pre-registration, check out the mapped commit and confirm the
hypotheses and decision criteria predate the results in the subsequent
history (e.g. `git show 702efd95`).

**From this point forward, pre-registration commits are pushed to this
public repository before any run begins** — the pre-registration timestamps
are publicly verifiable from here on, which the private repository could
not provide.

Some earlier papers cite file paths that have since moved in the ordinary
course of the work (e.g. `research/physics/bcft_functional_form_fit.py` →
`research/physics/experiments/exp-026_bcft_functional_form/`;
`research/notes/bcft_pre_registered_prediction.md` →
`research/physics/notes/2026-04-17_bcft_pre_registered_prediction.md`).
When a cited path is missing, search this repository by filename.

## Contact

ariel@sonielmn.com — https://sonielmn.com
