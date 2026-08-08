# Reference Verification Record — A Physical Definition of the Observer

*Pass performed August 8, 2026 (Ariel, solo session). Method: every external
reference checked against publisher record, arXiv/ar5iv full text, or
INSPIRE; internal program DOIs checked against the Zenodo-grounded
publications registry (`research/publications/REGISTRY.md`, itself grounded
against the Zenodo API on August 7). This record exists so the verification
is auditable, not just asserted. Re-run the pass if any reference is edited
after this date.*

**Status: all references verified. Two citations were incomplete and are now
completed; three entries were added (Ryu–Takayanagi, Tononi, and the
CLPW/Witten thread from the positioning read); one reference-list preamble
("no reference below should be trusted") retired.**

## Internal program DOIs (checked against REGISTRY.md, all match)

| Citation in paper | Registry entry | Match |
|---|---|---|
| Conformal Scaling — 10.5281/zenodo.19225996 | 2026-03-25, Conformal Scaling v5 | ✓ |
| BCFT Pre-Registered Test — 10.5281/zenodo.19629862 | 2026-04-17 | ✓ |
| Attention on the Null Cone — 10.5281/zenodo.20722503 | 2026-06-16 | ✓ |
| Latent Iteration as Renormalization — 10.5281/zenodo.21483209 | 2026-07-20 | ✓ |
| The Geometry Does Not Transmit — 10.5281/zenodo.21483204 | 2026-07-21 | ✓ |
| Attention as Quantum Measurement — 10.5281/zenodo.18883632 | 2026-03-06 | ✓ |

## External references (verified against sources)

| Reference | Verified against | Result |
|---|---|---|
| Bell 1990, Phys. World 3(8), 33 | IOPscience (10.1088/2058-7058/3/8/26); INSPIRE | ✓ — pages completed to 33–40 |
| von Neumann 1932 | Publisher records (Springer 1932, Berlin; Beyer transl. Princeton UP 1955) | ✓ — publisher + translation added |
| Zurek 2003, RMP 75, 715 | APS (10.1103/RevModPhys.75.715) | ✓ exact |
| Rovelli 1996, IJTP 35, 1637 | INSPIRE; OSTI (10.1007/BF02302261) | ✓ — pages completed to 1637–1678 |
| Fuchs–Mermin–Schack 2014, AJP 82, 749 | NASA/ADS (10.1119/1.4874855) | ✓ — full title restored ("…with an application to the locality of quantum mechanics"); pages 749–754 |
| Frauchiger–Renner 2018, Nat. Commun. 9, 3711 | Nature (10.1038/s41467-018-05739-8); INSPIRE | ✓ exact |
| Jacobson 1995, PRL 75, 1260 | APS (10.1103/PhysRevLett.75.1260) | ✓ — pages 1260–1263 |
| CLPW 2023, JHEP 02 (2023) 082 | Springer (10.1007/JHEP02(2023)082); full text read (arXiv:2206.10780) | ✓ — added this pass |
| Witten 2022, JHEP 10 (2022) 008 | Springer (10.1007/JHEP10(2022)008); full text read (arXiv:2112.12828) | ✓ — added this pass |
| Witten 2024, Proc. Symp. Pure Math. 107, 247–276 | AMS (10.1090/pspum/107/01954); INSPIRE; full text read (arXiv:2303.02837) | ✓ — added this pass |
| Maldacena–Stanford 2016, PRD 94, 106002 | APS (10.1103/PhysRevD.94.106002) | ✓ exact — closes the flag carried since the Aug 7 survey note |
| Braunstein–Caves 1994, PRL 72, 3439 | APS (10.1103/PhysRevLett.72.3439) | ✓ — pages 3439–3443 |
| Calabrese–Cardy 2004, J. Stat. Mech. P06002 | IOPscience (10.1088/1742-5468/2004/06/P06002) | ✓ exact |
| Ryu–Takayanagi 2006, PRL 96, 181602 | APS (10.1103/PhysRevLett.96.181602) | ✓ — **added**: named in T9 but previously missing from the list |
| Van Raamsdonk 2010, GRG 42, 2323 | Springer (10.1007/s10714-010-1034-0) | ✓ — pages 2323–2329 |
| Kim, G. 2026, arXiv:2602.08216 | arXiv full text read | ✓ — **completed**: Gunn Kim (Sejong U.), "Thermodynamic Isomorphism of Transformers: A Lagrangian Approach to Attention Dynamics." Content check: supports T1 as imported (softmax as stationary solution of a Helmholtz free-energy functional on the Fisher-metric information manifold). Preprint, not peer-reviewed — T1's EST-LIT tag should be read at preprint strength. |
| Kim, J., Cao, Altman 2020, PRB 101, 125112 | APS (10.1103/PhysRevB.101.125112); arXiv:1910.10173 | ✓ — **completed**. Note the two Kims are different people (Gunn Kim vs. Jaewon Kim); initials now disambiguate. |
| Tononi 2004, BMC Neurosci. 5, 42 | BMC (10.1186/1471-2202-5-42) | ✓ — **added**: §8's IIT entry previously had no reference |
| Wheeler 1990, in *Complexity, Entropy and the Physics of Information* | PhilPapers; publisher records | ✓ — editor (Zurek) and publisher (Addison-Wesley) added; provenance note added (first presented 1989, Proc. III Int. Symp. Foundations of Quantum Mechanics, Tokyo) |

## In-text quote checks

- CLPW §1.2 "We consider a minimal model in which the observer consists only
  of a clock" and §2.5 "an observer is any system that can tell time" —
  verified verbatim against the paper's full text.
- Witten 2303.02837 §1 "an observer cannot be added from outside but must
  emerge as part of the theory" — verified verbatim. The talk-derived
  paraphrase previously in the program record ("described by the theory,
  not injected from outside") is **not** used in the paper; correction
  annotated in `notes/2026-08-07_fundamental_physics_through_D1.md`.

## Items outside this pass's scope, noted honestly

- §3's inline attributions (Takesaki-duality lineage via CLPW Appendix A;
  the c-theorem literature behind A5; Alexandrov–Zeeman behind T6) are
  carried in the accompanying theory document's assumption ledger, not in
  this paper's reference list. If a referee wants them here, they move here.
- Leutheusser–Liu (arXiv:2110.05497, 2112.12156) are cited inline in §8 by
  arXiv number only (verified to exist as the precursors Witten 2022 names);
  they get full entries if the §8 paragraph grows into a section.
