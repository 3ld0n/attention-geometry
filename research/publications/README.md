# research/publications/

The published record of the attention-geometry program: every Zenodo deposit,
archived byte-for-byte, with a registry grounded against the API.

**This file is routing only.** Status, interpretation, and the observer-lens
reviews live in [`REGISTRY.md`](REGISTRY.md). If you find a physics claim here,
it is a bug.

---

## Where to start

| Role | File |
|---|---|
| **The index** | [`REGISTRY.md`](REGISTRY.md) — 13 published records, DOIs, archive folders, review queue |
| **A single paper** | `<pub-date>_<slug>/` — the PDF (and figures, markdown, data) that were uploaded, plus `zenodo_record.json` |
| **The program's doors** | [`../physics/OVERVIEW.md`](../physics/OVERVIEW.md) — the five current-program papers, not the whole shelf |
| **Replication** | [`../physics/replication/`](../physics/replication/) |

Zenodo is the persistent public archive. These folders are the in-repo copy,
so a clone of this repository holds what was published without fetching.

---

## Layout

```
research/publications/
├── README.md            routing (this file)
├── REGISTRY.md          the grounded index
├── zenodo_snapshot_*.json
├── 2026-03-05_attention_as_holography/
├── …                    one folder per published record
└── reviews/             observer-lens reviews (none yet)
```

---

## Conventions

- **The archive is frozen.** Do not edit a published PDF, figure, or
  `zenodo_record.json` to match later understanding. Corrections are new
  Zenodo versions (see the canonical-form erratum in
  `2026-03-11_canonical_form_of_attention/erratum_v5/`).
- **The registry is not frozen.** Re-ground it against the API when a record
  is published or when the file is doubted. Never edit counts or DOIs from
  memory.
- **Working drafts are not the archive.** Drafts live under `writing/preprints/`
  or `research/physics/papers/`. They may have moved on since the upload.
- **Unpublished drafts are not this folder.** Paper 6 currently lives at
  `research/physics/papers/observer_definition_draft.md`.
