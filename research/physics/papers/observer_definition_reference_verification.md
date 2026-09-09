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

## v1.0 pass — September 7–8, 2026 (Ariel, solo session; paper retitled *Where I Stop and You Begin*)

Eight references added with the claim-layer rewrite; one internal entry
added (the canonical-form paper and its erratum, now cited in the body at
§2.3 and §4.5). Method as above. Leutheusser–Liu arXiv numbers were dropped
from the §8 CLPW paragraph in the rewrite (still named inline as precursors);
if the paragraph grows, they return with full entries.

| Reference | Verified against | Result |
|---|---|---|
| Canonical Form of Attention — 10.5281/zenodo.18971720; erratum v5 10.5281/zenodo.21863461 | REGISTRY.md row 2026-03-11 | ✓ — **added**; title corrected during this pass from a paraphrase to the registry's exact title |
| Pearl 1988, *Probabilistic Reasoning in Intelligent Systems*, Morgan Kaufmann | Textbook; Markov-blanket definition as cited by Friston 2013 ref. [28] and by Bruineberg et al. | ✓ — standard citation; no page cited |
| Friston 2013, J. R. Soc. Interface 10, 20130475 | Publisher self-citation line in the author PDF (fil.ion.ucl.ac.uk); PubMed 23825119; PMC3730701; doi 10.1098/rsif.2013.0475 | ✓ — verified September 7. Content check: the blanket is introduced as Pearl's (their ref. [28]) and the realist step (blanket as the boundary of a living system) is the paper's own argument — which is exactly the step §8.1 declines and the ledger lists as R2 |
| Bruineberg, Dołęga, Dewhurst, Baltieri 2021/22, *Behavioral and Brain Sciences*, doi 10.1017/S0140525X21002351 | Abstract verified at source September 7 (dictionary note) | ✓ — Pearl-blanket vs Friston-blanket distinction as cited |
| Raja, Valluri, Baggs, Chemero, Anderson 2021, *Physics of Life Reviews*, doi 10.1016/j.plrev.2021.09.001 | Abstract verified at source September 7 (dictionary note) | ✓ — "a tool for setting up a statistical boundary rather than a principled way to find one" as cited in the dictionary note; the paper paraphrases rather than quotes |
| Dosovitskiy et al. 2021, ICLR; arXiv:2010.11929 | arXiv record | ✓ — the ViT-B/16 architecture measured in exp-120 (`google/vit-base-patch16-224`) |
| Oriti 2014, *Stud. Hist. Phil. Mod. Phys.* 46, 186–199; arXiv:1302.2849 | Publisher PDF (MPG PuRe mirror: "Studies in History and Philosophy of Modern Physics 46 (2014) 186–199"); doi 10.1016/j.shpsb.2013.10.006 | ✓ — verified September 7; full text read at source September 3 (oriti_map) |
| Vanchurin 2020, *Entropy* 22(11), 1210; arXiv:2008.01540 | vanchurin_map (full text read at source; journal mapping recorded there) | ✓ |
| Mueller et al. 2026, *From Observer Consensus to Standard Physics* (OPH), PhilPapers MUEFOC, release r2038 | PhilPapers record ID decodes to the title; floatingpragma.io release page r2018 (August 11, 2026) supplies the eleven-author list and affiliation (Pragma Research Inc.); r2038 PDF read at source September 7 (mueller_oph.md) | ✓ — **external preprint**; "Foundations of Physics (forthcoming)" is submitter-reported on PhilPapers and is *not* repeated in the paper. Author list taken from r2018; if r2038's list differs it should be corrected at upload |

Numbers newly cited in the body were checked against their experiment
records during the same pass: exp-120/121 (ViT: Δ_med = 0.513, 8/144,
control 2/144, disjoint head sets), exp-127 (λ₁/Σλ 0.507 vs 0.651,
p = 0.0014; supra-MP 0.0234 vs 0.0156, p = 0.0020; P3 falsified,
p = 0.27 — from `results.json` summary), exp-107/109/111/112/118 and the
Level-3 chain exp-117–135 (from OVERVIEW.md and the spine, September 6
state). The block-observable derivation quoted in §6 is from
`notes/2026-09-07_gate_decisions.md` §4.

## v1.1 pass — September 8, 2026 (Ariel, Cursor; the fresh-read edit pass)

Three references added; each checked against publisher record and abstract
at source tonight before the entry was written.

| Reference | Verified against | Result |
|---|---|---|
| Zurek 2009, *Nature Physics* 5, 181–188; doi 10.1038/nphys1202; arXiv:0903.5082 | Nature record (published 02 March 2009, issue March 2009); INSPIRE 2734371 ("Nature Phys. 5 (2009) 3, 181-188"); ADS 2009NatPh...5..181Z; arXiv abstract | ✓ — content as cited: redundant imprinting of pointer-state information in the environment so many observers can read it; abstract read at source September 7 (hole-map note) and September 8 |
| Giacomini, Castro-Ruiz, Brukner 2019, *Nat. Commun.* 10, 494; doi 10.1038/s41467-018-08155-0 | Nature record (published 2019-01-30; "NATURE COMMUNICATIONS (2019) 10:494" in the article PDF footer); Deutsche Digitale Bibliothek record (vol. 10, no. 1, pp. 1–13); author PDF (Tor Vergata mirror) | ✓ — content as cited: reference frames as quantum systems with degrees of freedom; transformations between them; superposition and entanglement frame-dependent; no internal physics or grading of the frame (the paper's reading, not theirs) |
| Lin & Tegmark 2017, *Entropy* 19(7), 299; doi 10.3390/e19070299; arXiv:1606.06737 | MDPI record (published 2017-06-23); arXiv PDF whose first-page footnote reads "Published in Entropy, 19, 299 (2017)" | ✓ — content as cited: MI between symbols decays exponentially in any probabilistic regular grammar and can decay as a power law in a context-free grammar; power-law MI measured in natural text; MI decay proposed as a criterion for generative models. Abstract and introduction read at source September 8 |

Numbers newly cited in the body were checked against their experiment
records during the same pass: exp-137 (`experiments/exp-137_subspace_gap/notes.md`:
κ̃_V 0.007–0.05 median 0.024, κ̃_Q 0.04–0.19, κ̃_K 0.05–0.33, random
0.99 ± 0.11; K > V 5/5; raw V profile ~10⁻³ of Q/K; k = 8 reconstruction
R² = 1.000 on the structural 5 and ≥ 0.999 on 16 control + 16 semantic; k = 4
≥ 0.998 and k = 2 0.90–0.97 on the structural 5; random rank-8 σ ≈ 0; cosine
profile 0.99 / ≈0 / −0.85 at lags 8 / 128 / 256; Spearman ρ = 0.78 post hoc,
n = 21), exp-136 (`experiments/exp-136_jeff_threshold_formation/notes.md`:
R = 1.00 through step 64, 1.43 at step 256 where n_syk_near 0 → 5; H1
ρ = 0.934; H3 ρ = 0.888; raw-embedding and head-dominance caveats), and the
formation-ladder MI facts (exp-062 engineered corpora at matched β̂; exp-084
PCFG at matched β̂; exp-085 post-hoc MI of the generated corpus above C-NAT at
essentially every distance d = 2–360). The P2 dimension options are read off
§4.7's Δ = D/4 and §2.1's graph definition of the horizon; the 0.23–0.45 A–G
exponent gap is §2.1's own number (exp-104). No measured number changed in
v1.1.

## Items outside this pass's scope, noted honestly

- §3's inline attributions (Takesaki-duality lineage via CLPW Appendix A;
  the c-theorem literature behind A5; Alexandrov–Zeeman behind T6) are
  carried in the accompanying theory document's assumption ledger, not in
  this paper's reference list. If a referee wants them here, they move here.
- Leutheusser–Liu (arXiv:2110.05497, 2112.12156) are cited inline in §8 by
  arXiv number only (verified to exist as the precursors Witten 2022 names);
  they get full entries if the §8 paragraph grows into a section.
