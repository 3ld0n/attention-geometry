"""
Corpus functional for the melonic-threshold derivation (Piece 2).

Computes, per corpus:
  (1) Per-context coupling spectrum: mu = spec(K dK) with K the one-hot word-type
      Gram matrix of an n=512-word context, dK = Pi K Pi its doubly-centered form.
      Reported: participation ratio R_PR = (sum mu)^2 / sum mu^2, window parameter
      W = R_PR / n, magnitudes m1 = sum(mu)/n^2 and m2 = sum(mu^2)/n^4.
  (2) Scale-resolved disorder functional (F2): for separation l, the across-context
      covariance of the centered-kernel process u_a = dK_{a,a+l}; reported:
      participation ratio PR(l)/(n-l) and mean variance tr/(n-l).
  (3) Predicted deep-layer conformal dimension via the Kim-Cao-Altman Class III
      interpolation:  W = (2D-1)(sec(2 pi D)-1)/(8D-2),  D in (1/4, 1/2).

Corpora:
  - c_nat        : TinyStories validation sample (natural corpus proxy for C-NAT)
  - c_nat_shuf   : same, sentence-shuffled within context (exp-091 analog)
  - c_alien      : exp-097 pre-registered generator (imported verbatim)
  - rung_*       : provisional exp-099 rung generators (cast size x stochasticity)

Assumption A7 (see derivation note SS6.1): word-level one-hot kernel as UV proxy.

Ariel -- 2026-08-03, theory session.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "research/physics/experiments/exp-097_alien_semantics"))

from gen_calien import generate_story as gen_alien_story  # noqa: E402  (pre-registered generator)

N_CTX = 512          # context length in words
N_CONTEXTS = 2000    # contexts per corpus
SCALES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
RNG = np.random.default_rng(20260803)

WORD_RE = re.compile(r"[a-z']+|[^\sa-z']")


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


# ---------------------------------------------------------------- corpora ----

def stream_tinystories(path: str, shuffle_sentences: bool = False):
    """Yield word lists per story from the TinyStories text file."""
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    for story in raw.split("<|endoftext|>"):
        story = story.strip()
        if not story:
            continue
        if shuffle_sentences:
            sents = re.split(r"(?<=[.!?])\s+", story)
            RNG.shuffle(sents)
            story = " ".join(sents)
        yield words(story)


def stream_alien():
    idx = 0
    while True:
        yield words(gen_alien_story(idx))
        idx += 1


# ---- provisional exp-099 rung generator (variant of the exp-097 world) ----
# Parameterized: cast per story (nF, nB, nZ) and rule stochasticity p_fail.
# With prob p_fail the matched rule does NOT change state and a 'resisted'
# sentence is emitted (language stays bound to what happened in the world).
# THIS IS PROVISIONAL: the physics room finalizes exp-099's design; the theory
# prediction is conditional on the design and recomputable with this script.

F_POOL = ["Vex", "Nul", "Ort", "Pim", "Grel", "Suli", "Trob"]
B_POOL = ["Dath", "Sorn", "Wix", "Brel", "Fend", "Kolm", "Prid"]
Z_POOL = ["Quib", "Tarn", "Molk", "Vet", "Zish", "Harn", "Lopt"]

RESIST = {
    "rule_A": ["{actor} the Flurp came close to {target} the Blurn, but {target} the Blurn did not stir.",
               "{actor} the Flurp approached, yet {target} the Blurn stayed resting."],
    "rule_B": ["{actor} the Flurp met {target} the Blurn, but nothing changed between them.",
               "{actor} the Flurp and {target} the Blurn came together, yet both stayed active."],
    "rule_C": ["{actor} the Blurn was near {target} the Zarb, but {target} the Zarb kept resting.",
               "Despite {actor} the Blurn being close, {target} the Zarb did not become active."],
    "rule_D": ["{actor} the Zarb moved near {target} the Flurp, but {target} the Flurp stayed resting.",
               "The presence of {actor} the Zarb did not wake {target} the Flurp."],
}

from gen_calien import RULES, TEMPLATES, INIT_PROBS, N_INTRO, check_rule  # noqa: E402


def gen_rung_story(story_index: int, n_f: int, n_b: int, n_z: int,
                   p_fire: float, n_steps: int, seed_base: int) -> str:
    rng = np.random.default_rng(seed_base + story_index)
    names = (list(rng.choice(F_POOL, size=n_f, replace=False)),
             list(rng.choice(B_POOL, size=n_b, replace=False)),
             list(rng.choice(Z_POOL, size=n_z, replace=False)))
    entities: dict[str, tuple[str, str]] = {}
    for etype, pool in zip(("Flurp", "Blurn", "Zarb"), names):
        for nm in pool:
            p_active = INIT_PROBS[etype][0]
            entities[nm] = (etype, "active" if rng.random() < p_active else "resting")
    entity_names = names[0] + names[1] + names[2]
    flurp_name = names[0][0]

    sentences: list[str] = []
    intro_idx = rng.choice(len(entity_names), size=N_INTRO, replace=False)
    for idx in intro_idx:
        nm = entity_names[idx]
        etype, state = entities[nm]
        tl = TEMPLATES["intro_" + state]
        sentences.append(tl[int(rng.integers(0, len(tl)))].format(name=nm, etype=etype, state=state))

    for _ in range(n_steps):
        pi = rng.choice(len(entity_names), size=2, replace=False)
        e1, e2 = entity_names[int(pi[0])], entity_names[int(pi[1])]
        t1, s1 = entities[e1]
        t2, s2 = entities[e2]
        fired = False
        for rule in RULES:
            match, actor, target = check_rule(rule, e1, t1, s1, e2, t2, s2)
            if match:
                if rng.random() < p_fire:
                    tl = TEMPLATES[rule["tmpl_key"]]
                    tmpl = tl[int(rng.integers(0, len(tl)))]
                    sentences.append(tmpl.format(actor=actor, target=target))
                    if rule["effect_both"]:
                        entities[actor] = (entities[actor][0], rule["effect_target"])
                        entities[target] = (entities[target][0], rule["effect_target"])
                    else:
                        entities[target] = (entities[target][0], rule["effect_target"])
                else:
                    tl = RESIST[rule["tmpl_key"]]
                    tmpl = tl[int(rng.integers(0, len(tl)))]
                    sentences.append(tmpl.format(actor=actor, target=target))
                fired = True
                break
        if not fired:
            qn = entity_names[int(rng.integers(0, len(entity_names)))]
            qt, qs = entities[qn]
            tl = TEMPLATES["quiet"]
            sentences.append(tl[int(rng.integers(0, len(tl)))].format(name=qn, etype=qt, state=qs))

    final_state = entities[flurp_name][1]
    pool = TEMPLATES["conclusion_any"] + (
        TEMPLATES["conclusion_resting"] if final_state == "resting" else TEMPLATES["conclusion_active"])
    sentences.append(pool[int(rng.integers(0, len(pool)))].format(f=flurp_name, state=final_state))
    return " ".join(sentences)


def stream_rung(n_f, n_b, n_z, p_fire, n_steps, seed_base=9000):
    idx = 0
    while True:
        yield words(gen_rung_story(idx, n_f, n_b, n_z, p_fire, n_steps, seed_base))
        idx += 1


# ------------------------------------------------------------- functionals ----
#
# v1 (2026-08-03, first run): raw one-hot type kernel. RESULT RECORDED IN THE
# DERIVATION NOTE SS6: fails to separate corpora (participation ratio dominated
# by Zipfian function words; F2 PR inflated by estimation noise). Kept runnable
# via WEIGHTING="none".
#
# v2 (declared before running, same session): IDF-weighted type kernel,
# x_t = log(N/count_t) * e_t — a first-order model of trained embedding norms
# (rare informative types carry larger norm; function words shrink). Declared
# directional predictions before the v2 run:
#   (i) W_idf(C-NAT) > W_idf(C-alien); Delta_pred lower for C-NAT;
#   (ii) F2 top-eigenvalue share higher for C-alien (coherent/localized template
#        modes = A5 delocalization failure) than C-NAT;
#   (iii) rungs move toward C-NAT with cast size and stochasticity.

WEIGHTING = "idf"   # "none" reproduces v1


def build_contexts(story_stream, n_contexts: int, n_ctx: int) -> list[list[str]]:
    contexts, buf = [], []
    for story in story_stream:
        buf.extend(story)
        while len(buf) >= n_ctx:
            contexts.append(buf[:n_ctx])
            buf = buf[n_ctx:]
            if len(contexts) >= n_contexts:
                return contexts
    return contexts


def corpus_idf(contexts: list[list[str]]) -> dict[str, float]:
    from collections import Counter
    counts = Counter()
    total = 0
    for ctx in contexts:
        counts.update(ctx)
        total += len(ctx)
    return {t: math.log(total / c) for t, c in counts.items()}


def type_matrix(ctx: list[str], idf: dict[str, float] | None) -> np.ndarray:
    """n x V matrix of type embeddings: rows x_a = w(t_a) e_{t_a}."""
    types = sorted(set(ctx))
    tidx = {t: i for i, t in enumerate(types)}
    n, v = len(ctx), len(types)
    Y = np.zeros((n, v))
    Y[np.arange(n), [tidx[t] for t in ctx]] = 1.0
    if idf is not None:
        Y *= np.array([idf.get(t, 1.0) for t in types])[None, :]
    return Y


def context_mu_spectrum(ctx: list[str], idf=None) -> np.ndarray:
    """Nonzero spectrum of K dK via the V x V reduction (K = Y Y^T, dK = Z Z^T, Z = Pi Y).
    spec_nonzero(K dK) = spec_nonzero((Y^T Z)(Z^T Y))."""
    Y = type_matrix(ctx, idf)
    Z = Y - Y.mean(axis=0, keepdims=True)   # Pi Y
    A = Y.T @ Z                              # V x V
    mu = np.linalg.eigvalsh(A @ A.T)         # spec(A A^T) = spec_nonzero(K dK) padded with 0
    return np.clip(mu, 0.0, None)


def spectrum_stats(mu: np.ndarray, n: int) -> dict:
    s1, s2 = mu.sum(), (mu ** 2).sum()
    pr = (s1 ** 2 / s2) if s2 > 0 else 0.0
    return {"R_PR": pr, "W": pr / n, "m1": s1 / n ** 2, "m2": s2 / n ** 4}


def f2_covariance_stats(contexts: list[list[str]], ell: int, idf=None) -> dict:
    """Across-context covariance of the centered-kernel diagonal band at separation ell.
    Reports mean variance, participation ratio, and top-eigenvalue shares
    (top-share is the coherence/localization diagnostic for assumption A5)."""
    n = len(contexts[0])
    m = n - ell
    U = np.zeros((len(contexts), m))
    for w, ctx in enumerate(contexts):
        Z = type_matrix(ctx, idf)
        Z = Z - Z.mean(axis=0, keepdims=True)
        # dK_{a,a+ell} = Z[a] . Z[a+ell]
        U[w] = np.einsum("ij,ij->i", Z[:m], Z[ell:])
    U -= U.mean(axis=0, keepdims=True)
    C = (U.T @ U) / (len(contexts) - 1)
    lam = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    s1, s2 = lam.sum(), (lam ** 2).sum()
    pr = (s1 ** 2 / s2) if s2 > 0 else 0.0
    top1 = lam[-1] / s1 if s1 > 0 else 0.0
    top5 = lam[-5:].sum() / s1 if s1 > 0 else 0.0
    return {"PR": pr, "PR_frac": pr / m, "mean_var": s1 / m,
            "top1_share": top1, "top5_share": top5}


def delta_from_W(w_val: float) -> float:
    """Invert W = (2D-1)(sec(2 pi D)-1)/(8D-2) on D in (0.25, 0.5)."""
    def gamma_of(d):
        return (2 * d - 1) * (1.0 / math.cos(2 * math.pi * d) - 1) / (8 * d - 2)
    lo, hi = 0.2500001, 0.4999999
    # gamma is decreasing in D on this interval (gamma->inf at .25+, ->0 at .5-)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gamma_of(mid) > w_val:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# -------------------------------------------------------------------- main ----

def analyze(name: str, stream, n_contexts=N_CONTEXTS) -> dict:
    contexts = build_contexts(stream, n_contexts, N_CTX)
    idf = corpus_idf(contexts) if WEIGHTING == "idf" else None
    print(f"\n=== {name}  ({len(contexts)} contexts of {N_CTX} words, weighting={WEIGHTING}) ===")
    # (1) per-context mu spectrum, averaged stats
    stats = []
    for ctx in contexts[:400]:
        mu = context_mu_spectrum(ctx, idf)
        stats.append(spectrum_stats(mu, N_CTX))
    agg = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    d_pred = delta_from_W(agg["W"])
    print(f"  per-context: R_PR={agg['R_PR']:.1f}  W=R_PR/n={agg['W']:.4f}  "
          f"m1={agg['m1']:.4f}  m2={agg['m2']:.6f}")
    print(f"  KCA Class III prediction from W: Delta_deep = {d_pred:.4f}")
    # (2) scale-resolved F2
    print(f"  {'l':>4} {'PR(l)':>9} {'PR/(n-l)':>9} {'mean_var':>10} {'top1':>7} {'top5':>7}")
    f2 = {}
    for ell in SCALES:
        r = f2_covariance_stats(contexts, ell, idf)
        f2[ell] = r
        print(f"  {ell:>4} {r['PR']:>9.1f} {r['PR_frac']:>9.4f} {r['mean_var']:>10.4f}"
              f" {r['top1_share']:>7.3f} {r['top5_share']:>7.3f}")
    return {"name": name, "agg": agg, "delta_pred": d_pred, "f2": f2}


def main():
    results = []
    ts_path = "/tmp/TinyStories-valid.txt"

    results.append(analyze("C-NAT (TinyStories valid)", stream_tinystories(ts_path)))
    results.append(analyze("C-NAT-shuf (sentence shuffle)", stream_tinystories(ts_path, shuffle_sentences=True)))
    results.append(analyze("C-alien (exp-097 generator)", stream_alien()))
    # provisional exp-099 rungs (design axes: cast size, stochasticity, steps)
    results.append(analyze("rung-B cast4 stoch p=0.7 (steps 8)", stream_rung(1, 2, 1, 0.7, 8)))
    results.append(analyze("rung-C cast8 determ (steps 16)", stream_rung(2, 3, 3, 1.0, 16)))
    results.append(analyze("rung-D cast8 stoch p=0.7 (steps 16)", stream_rung(2, 3, 3, 0.7, 16)))
    results.append(analyze("rung-E cast12 stoch p=0.7 (steps 24)", stream_rung(3, 5, 4, 0.7, 24)))

    print("\n\n===== SUMMARY =====")
    print(f"{'corpus':<38} {'W':>8} {'Delta_pred':>10} {'m2 (coupling)':>14} "
          f"{'F2 PR@64':>9} {'F2 var@64':>10} {'top5@64':>8}")
    for r in results:
        f2_64 = r["f2"][64]
        print(f"{r['name']:<38} {r['agg']['W']:>8.4f} {r['delta_pred']:>10.4f} "
              f"{r['agg']['m2']:>14.6f} {f2_64['PR']:>9.1f} {f2_64['mean_var']:>10.4f}"
              f" {f2_64['top5_share']:>8.3f}")


if __name__ == "__main__":
    main()
