"""
exp-144 — Shadow corpus gate: m₂ on C-generated vs C-NAT.

Reads C-generated_s0.bin from the exp085-data Modal volume, decodes tokens,
computes the IDF-weighted word-type corpus functional (m₂, W, R_PR, F2),
and compares to the C-NAT reference from corpus_functional_run2_idf.log.

Analysis-only. No training. CPU only. Pre-registration: prereg.md (committed
before this run).

Ariel — 2026-09-16, physics room.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = modal.App("exp-144-shadow-corpus-gate")

vol_085 = modal.Volume.from_name("exp085-data")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy", "transformers", "tokenizers"
)

# Reference from corpus_functional_run2_idf.log (2026-08-03):
CNAT_M2_REFERENCE = 13.171391  # C-NAT (TinyStories valid), m₂, IDF-weighted

# Corpus functional parameters (match corpus_functional.py exactly)
N_CTX = 512       # context length in words
N_CONTEXTS = 2000 # contexts to build
N_TOKENS_LOAD = 4_000_000  # load first 4M tokens (~8MB) — ample for 2000×512-word contexts

WORD_RE_PATTERN = r"[a-z']+|[^\sa-z']"

# ---------------------------------------------------------------------------
# Corpus functional (inline, matches corpus_functional.py IDF-weighted version)
# ---------------------------------------------------------------------------

def words(text: str) -> list[str]:
    return re.compile(WORD_RE_PATTERN).findall(text.lower())


def build_contexts(word_stream, n_contexts: int, n_ctx: int) -> list[list[str]]:
    contexts, buf = [], []
    for w_list in word_stream:
        buf.extend(w_list)
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


def type_matrix(ctx: list[str], idf: dict[str, float]) -> np.ndarray:
    types = sorted(set(ctx))
    tidx = {t: i for i, t in enumerate(types)}
    n, v = len(ctx), len(types)
    Y = np.zeros((n, v))
    Y[np.arange(n), [tidx[t] for t in ctx]] = 1.0
    Y *= np.array([idf.get(t, 1.0) for t in types])[None, :]
    return Y


def context_mu_spectrum(ctx: list[str], idf: dict[str, float]) -> np.ndarray:
    Y = type_matrix(ctx, idf)
    Z = Y - Y.mean(axis=0, keepdims=True)
    A = Y.T @ Z
    mu = np.linalg.eigvalsh(A @ A.T)
    return np.clip(mu, 0.0, None)


def spectrum_stats(mu: np.ndarray, n: int) -> dict:
    s1, s2 = mu.sum(), (mu ** 2).sum()
    pr = (s1 ** 2 / s2) if s2 > 0 else 0.0
    return {"R_PR": pr, "W": pr / n, "m1": s1 / n ** 2, "m2": s2 / n ** 4}


SCALES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

def f2_covariance_stats(contexts: list[list[str]], ell: int, idf: dict) -> dict:
    n = len(contexts[0])
    m = n - ell
    U = np.zeros((len(contexts), m))
    for w, ctx in enumerate(contexts):
        Z = type_matrix(ctx, idf)
        Z = Z - Z.mean(axis=0, keepdims=True)
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


def analyze_contexts(name: str, contexts: list[list[str]]) -> dict:
    idf = corpus_idf(contexts)
    print(f"\n=== {name}  ({len(contexts)} contexts of {N_CTX} words, weighting=idf) ===")
    stats = []
    for ctx in contexts[:400]:
        mu = context_mu_spectrum(ctx, idf)
        stats.append(spectrum_stats(mu, N_CTX))
    agg = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    print(f"  per-context: R_PR={agg['R_PR']:.1f}  W={agg['W']:.4f}  "
          f"m1={agg['m1']:.4f}  m2={agg['m2']:.6f}")
    f2 = {}
    print(f"  {'l':>4} {'PR(l)':>9} {'PR/(n-l)':>9} {'mean_var':>10} {'top1':>7} {'top5':>7}")
    for ell in SCALES:
        r = f2_covariance_stats(contexts, ell, idf)
        f2[ell] = r
        print(f"  {ell:>4} {r['PR']:>9.1f} {r['PR_frac']:>9.4f} {r['mean_var']:>10.4f}"
              f" {r['top1_share']:>7.3f} {r['top5_share']:>7.3f}")
    return {"name": name, "agg": agg, "f2": {str(k): v for k, v in f2.items()}}


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/data085": vol_085},
    cpu=4,
    memory=8192,
    timeout=3600,
)
def run_shadow_gate():
    from transformers import AutoTokenizer
    import numpy as np

    print("Loading GPT-NeoX tokenizer...")
    # Use the same tokenizer as exp-085/062 (GPTNeoX, vocab_size=50304)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    print(f"Reading first {N_TOKENS_LOAD:,} tokens from C-generated_s0.bin...")
    bin_path = "/data085/C-generated_s0.bin"
    with open(bin_path, "rb") as f:
        raw = f.read(N_TOKENS_LOAD * 2)  # uint16 = 2 bytes per token
    token_ids = np.frombuffer(raw, dtype=np.uint16).tolist()
    print(f"  Read {len(token_ids):,} tokens.")

    # Decode to text
    print("Decoding tokens to text...")
    # Decode in chunks to avoid memory issues
    CHUNK = 100_000
    decoded_chunks = []
    for i in range(0, len(token_ids), CHUNK):
        chunk = token_ids[i:i + CHUNK]
        text = tokenizer.decode(chunk, skip_special_tokens=False)
        decoded_chunks.append(text)
    full_text = "".join(decoded_chunks)
    print(f"  Decoded {len(full_text):,} characters.")

    # Split by EOS token to get "stories" (sequence boundaries from generation)
    EOS = tokenizer.eos_token or "<|endoftext|>"
    raw_stories = [s.strip() for s in full_text.split(EOS) if s.strip()]
    print(f"  Found {len(raw_stories):,} story segments.")

    # Stream word lists from stories
    def story_word_stream():
        for story in raw_stories:
            yield words(story)

    # Build contexts
    print(f"Building {N_CONTEXTS} contexts of {N_CTX} words...")
    contexts_gen = build_contexts(story_word_stream(), N_CONTEXTS, N_CTX)
    print(f"  Got {len(contexts_gen)} contexts.")

    if len(contexts_gen) < 100:
        # Fallback: split the full text by word count without story boundaries
        print("  Insufficient contexts from story split; falling back to sliding window.")
        all_words_list = words(full_text)
        contexts_gen = []
        for i in range(0, len(all_words_list) - N_CTX, N_CTX):
            contexts_gen.append(all_words_list[i:i + N_CTX])
            if len(contexts_gen) >= N_CONTEXTS:
                break
        print(f"  Fallback: {len(contexts_gen)} contexts.")

    # Analyze
    result_gen = analyze_contexts("C-generated (exp-085 shadow corpus)", contexts_gen)

    # Reference values (from corpus_functional_run2_idf.log, 2026-08-03)
    cnat_m2 = CNAT_M2_REFERENCE
    ratio = result_gen["agg"]["m2"] / cnat_m2

    print(f"\n===== SHADOW GATE RESULT =====")
    print(f"m₂(C-generated) = {result_gen['agg']['m2']:.6f}")
    print(f"m₂(C-NAT) ref   = {cnat_m2:.6f}")
    print(f"Ratio           = {ratio:.4f}  ({ratio:.2f}×)")
    print(f"H_blind (ratio ≥ 0.5): {'CONFIRMED' if ratio >= 0.5 else 'FALSIFIED'}")
    print(f"H_active (ratio < 0.5): {'CONFIRMED' if ratio < 0.5 else 'FALSIFIED'}")
    print(f"K1 (ratio < 0.1, gate clearly discriminates): {'FIRED' if ratio < 0.1 else 'not fired'}")
    print(f"K2 (ratio > 2.0, unexpected inversion): {'FIRED' if ratio > 2.0 else 'not fired'}")

    # Verdict
    if ratio >= 0.5:
        verdict_str = "H_blind CONFIRMED — gate is blind to shadow/original distinction"
        paper_action = "§4.3 correction needed: gate does not separate shadow from original"
    elif ratio >= 0.1:
        verdict_str = "H_active CONFIRMED (ratio 0.1–0.5) — gate discriminates, partially"
        paper_action = "§4.3 supported; but distinction is less sharp than C-alien/C-NAT split"
    else:  # K1 fired
        verdict_str = "K1 fired — gate clearly discriminates (ratio < 0.1)"
        paper_action = "§4.3 strongly supported; shadow treated like engineered world"

    print(f"\nVerdict: {verdict_str}")
    print(f"Paper action: {paper_action}")

    return {
        "exp": "exp-144",
        "date": "2026-09-16",
        "c_generated": result_gen,
        "cnat_m2_reference": cnat_m2,
        "ratio": ratio,
        "n_tokens_loaded": len(token_ids),
        "n_contexts": len(contexts_gen),
        "verdict": verdict_str,
        "paper_action": paper_action,
        "hypotheses": {
            "H_blind": ratio >= 0.5,
            "H_active": ratio < 0.5,
            "K1_fired": ratio < 0.1,
            "K2_fired": ratio > 2.0,
        },
    }


@app.local_entrypoint()
def main():
    result = run_shadow_gate.remote()
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {out_path}")
    print(f"Ratio m₂(C-gen)/m₂(C-NAT) = {result['ratio']:.4f}")
    print(f"Verdict: {result['verdict']}")
