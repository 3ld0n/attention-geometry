# Outreach note — DRAFT, DO NOT SEND

*Per SESSION_BRIEF_PHASE1 §3.1.5: drafted June 11, 2026; sending waits until the corpus-statistics test (exp-062) reports, since the ask sharpens decisively either way. Target: 2–3 interpretability groups + 1–2 SYK theorists.*

---

Subject: A falsifiable attention measurement you can run in 10 minutes

We have been measuring a specific quantitative regularity in trained
transformer attention and would value an adversarial replication before we
claim it more broadly.

The measurement: lag-averaged attention profiles of individual heads follow
power laws A(Δx) ∝ |Δx|^(−2Δ), and across every softmax model family we have
censused (GPT-2 family, Pythia, OLMo, Mistral, GALA-7B), the high-quality
subset's median exponent lands near Δ = 0.25, with sharp emergence during
training and zero such structure at initialization. One script reproduces
this on public checkpoints in minutes (attached repo).

Why you might take the time: this program publishes against interest. Our
pre-registered behavioral prediction was falsified on one of seven named
models and published as such; our own field-theoretic interpretation of the
attention sink was tested adversarially at matched complexity, lost, and was
withdrawn. The phenomenology that survived those kills is what we are asking
you to check. We have also shown the exponent is causally addressable: a
low-rank edit to a head's QK positional subspace moves both the exponent and
the head's long-range attention valley in the predicted direction.

The specific ask: run `measure_conformal_heads.py` on a trained model of
your choosing plus its randomized-weight control, and send us the two JSONs.
Prediction committed in the README: trained → conformal sub-population with
median Δ in [0.20, 0.30]; randomized → none. If you find a softmax model
family where this fails, that is exactly what we want to know.

[signature]

---

*Hold until exp-062 verdict. If universality: the ask adds "and we can now say why." If imprint: the ask becomes a measurement-tool pitch and the physics claim is withdrawn — still send, with that stated plainly.*
