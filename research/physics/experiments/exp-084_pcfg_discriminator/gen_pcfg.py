"""
exp-084 — PCFG corpus generator + β̂ calibration.

Pre-registration of record: notes.md (committed 2026-07-09).
Do NOT change GRAMMAR_PARAMS without a dated calibration log entry in notes.md.

Grammar (pre-registered design; terminal slots use the exp-062 256-symbol alphabet):

  S  → NP VP            (prob 1 - P_S2)
  S  → S CC S           (prob P_S2)               ← calibration knob A

  NP → Det N            (prob 0.55)
  NP → Det Adj N        (prob 0.25)
  NP → NP PP            (prob 0.20)

  VP → V NP             (prob (1 - P_VP3) * 0.60)
  VP → V NP PP          (prob (1 - P_VP3) * 0.40)
  VP → V CP             (prob P_VP3)               ← calibration knob B

  PP → P NP             (fixed)
  CP → Comp S           (fixed, recursive)

Role assignment (pre-registered, by position in alphabet.json sorted by token id):
  Det:   alphabet[  0 ..  14]  15 symbols
  N:     alphabet[ 15 ..  74]  60 symbols
  V:     alphabet[ 75 .. 124]  50 symbols
  Adj:   alphabet[125 .. 164]  40 symbols
  P:     alphabet[165 .. 184]  20 symbols
  Comp:  alphabet[185 .. 189]   5 symbols
  CC:    alphabet[190 .. 199]  10 symbols
  (symbols 200..255 are not used by the grammar)

Calibration log (all rounds must be written here before the next run):
  Round 0 (design, pre-reg 2026-07-09):
    P_S2=0.35, P_VP3=0.45
    Branching factor m_S = (1-P_S2)*P_VP3 + 2*P_S2 = 0.65*0.45 + 0.70 = 0.9925
    Expected mean sentence length ≈ 716 tokens (heavy-tailed distribution expected).
    Target β̂ = 0.79 ± 0.05 (matching exp-062 C-PL40).

Usage:
  python gen_pcfg.py pilot                  # 100M-token pilot + β̂ measurement
  python gen_pcfg.py full                   # 1.06B-token full corpus (after calibration)
  python gen_pcfg.py mi <path.bin>          # measure β̂ on an existing corpus
  python gen_pcfg.py stats                  # sentence-length statistics (10M pilot)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# ─── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
EXP062_DIR = SCRIPT_DIR.parent / "exp-062_corpus_statistics"
CORPUS_DIR = SCRIPT_DIR / "corpora"
CORPUS_DIR.mkdir(exist_ok=True)

# Import MI estimator from exp-062 (shared infrastructure; identical protocol).
sys.path.insert(0, str(EXP062_DIR))
from gen_corpora import mutual_information_profile  # noqa: E402

# ─── corpus sizes (identical to exp-062) ──────────────────────────────────────
TRAIN_TOKENS = 1_050_000_000
HELD_TOKENS  =    10_000_000
PILOT_TOKENS =   100_000_000
STATS_TOKENS =    10_000_000   # small corpus for sentence-length statistics

# ─── CALIBRATION PARAMETERS (adjust only via logged calibration rounds) ───────
GRAMMAR_PARAMS: dict[str, float] = {
    "P_S2":  0.35,   # Round 0 design value
    "P_VP3": 0.45,   # Round 0 design value
}

# ─── fixed grammar probabilities (pre-registered, not calibration knobs) ──────
NP_P1 = 0.55   # NP → Det N
NP_P2 = 0.25   # NP → Det Adj N
NP_P3 = 0.20   # NP → NP PP   (= 1 - NP_P1 - NP_P2)

# VP non-CP relative split (of the 1-P_VP3 probability mass):
VP_NONCAP_V_NP     = 0.60   # V NP
VP_NONCAP_V_NP_PP  = 0.40   # V NP PP

# Maximum terminals per sentence (safety cap; prevents infinite loops near critical).
MAX_SENT_TOKENS = 100_000

# Corpus seed (frozen after pre-registration, before any corpus generation).
CORPUS_SEED = 4001

# ─── role assignment ──────────────────────────────────────────────────────────
ROLE_SLICES: dict[str, tuple[int, int]] = {
    "Det":   (0,   15),
    "N":     (15,  75),
    "V":     (75,  125),
    "Adj":   (125, 165),
    "P":     (165, 185),
    "Comp":  (185, 190),
    "CC":    (190, 200),
}
TERMINAL_ROLES = set(ROLE_SLICES)


def load_roles() -> dict[str, np.ndarray]:
    """Load alphabet.json and partition into grammatical roles by pre-registered slices."""
    path = EXP062_DIR / "alphabet.json"
    if not path.exists():
        raise FileNotFoundError(
            f"alphabet.json not found at {path}\n"
            "Run: python exp-062_corpus_statistics/gen_corpora.py alphabet"
        )
    ids = np.array(json.loads(path.read_text())["ids"], dtype=np.uint16)
    assert len(ids) == 256, f"alphabet has {len(ids)} symbols, expected 256"
    return {role: ids[lo:hi] for role, (lo, hi) in ROLE_SLICES.items()}


# ─── PCFG generator ───────────────────────────────────────────────────────────

def _make_expander(params: dict[str, float]):
    """Return a closure that expands one non-terminal symbol given a random float u ∈ [0,1)."""
    p_s2  = params["P_S2"]
    p_vp3 = params["P_VP3"]
    p_vp_noncap = 1.0 - p_vp3
    p_vp1 = p_vp_noncap * VP_NONCAP_V_NP
    p_vp2 = p_vp_noncap * VP_NONCAP_V_NP_PP
    # cumulative thresholds for VP
    vp_t1 = p_vp1
    vp_t2 = p_vp1 + p_vp2

    def expand(sym: str, u: float) -> list[str]:
        if sym == "S":
            if u < p_s2:
                return ["S", "CC", "S"]
            else:
                return ["NP", "VP"]
        elif sym == "NP":
            if u < NP_P1:
                return ["Det", "N"]
            elif u < NP_P1 + NP_P2:
                return ["Det", "Adj", "N"]
            else:
                return ["NP", "PP"]
        elif sym == "VP":
            if u < vp_t1:
                return ["V", "NP"]
            elif u < vp_t2:
                return ["V", "NP", "PP"]
            else:
                return ["V", "CP"]
        elif sym == "PP":
            return ["P", "NP"]
        elif sym == "CP":
            return ["Comp", "S"]
        else:
            raise ValueError(f"Unknown non-terminal: {sym!r}")

    return expand


def gen_sentence(expand_fn, roles: dict[str, np.ndarray],
                 randoms: np.ndarray, r_start: int, rng
                 ) -> tuple[list[int], int]:
    """
    Expand one sentence from the root symbol 'S'.

    Pre-sampled `randoms` array is used for grammar rule choices (fast path).
    Terminal token IDs are sampled directly from rng (simple, always correct).
    Returns (token_list, next_r_start).  If randoms is exhausted mid-sentence,
    the caller should pass a fresh block and retry from r_start.
    """
    stack: list[str] = ["S"]
    tokens: list[int] = []
    r_pos = r_start

    while stack and len(tokens) < MAX_SENT_TOKENS:
        sym = stack.pop()
        if sym in TERMINAL_ROLES:
            pool = roles[sym]
            tokens.append(int(pool[int(rng.integers(len(pool)))]))
        else:
            if r_pos >= len(randoms):
                # Signal randoms exhaustion — caller refills and retries this sentence
                # by returning the original r_start (not r_pos).
                return [], r_start
            expansion = expand_fn(sym, float(randoms[r_pos]))
            r_pos += 1
            stack.extend(reversed(expansion))

    return tokens, r_pos


def gen_corpus(n_tokens: int, params: dict[str, float], seed: int,
               roles: dict[str, np.ndarray]) -> np.ndarray:
    """
    Generate n_tokens token IDs using the pre-registered PCFG.

    Returns a uint16 array of token IDs ready to save as .bin.
    """
    expand_fn = _make_expander(params)
    rng = np.random.default_rng(seed)

    output = np.empty(n_tokens, dtype=np.uint16)
    written = 0

    # Pre-sampled rule-choice randoms (refilled as needed).
    # A sentence with MAX_SENT_TOKENS terminals can use at most ~3x as many
    # rule applications (each terminal requires ≤3 non-terminal expansions).
    RAND_BLOCK = max(5_000_000, MAX_SENT_TOKENS * 5)
    randoms = rng.random(RAND_BLOCK)
    r_pos = 0

    n_sents = 0
    while written < n_tokens:
        # Refill randoms if too close to the end to fit a maximal sentence
        if r_pos > RAND_BLOCK - MAX_SENT_TOKENS * 5:
            randoms = rng.random(RAND_BLOCK)
            r_pos = 0

        tokens, r_pos_new = gen_sentence(expand_fn, roles, randoms, r_pos, rng)
        if not tokens:
            # randoms block was exhausted mid-sentence; refill and retry
            randoms = rng.random(RAND_BLOCK)
            r_pos = 0
            continue
        r_pos = r_pos_new

        take = min(len(tokens), n_tokens - written)
        output[written:written + take] = tokens[:take]
        written += take
        n_sents += 1

        if written % 50_000_000 < len(tokens):
            print(f"  pcfg: {written/1e6:.0f}M/{n_tokens/1e6:.0f}M "
                  f"({n_sents} sentences, mean len {written/n_sents:.0f})", flush=True)

    print(f"generated {written/1e6:.0f}M tokens in {n_sents} sentences "
          f"(mean {written/n_sents:.1f} tok/sent)")
    return output


# ─── sentence-length statistics ───────────────────────────────────────────────

def sentence_length_stats(params: dict[str, float], roles: dict[str, np.ndarray],
                           n_sents: int = 50_000) -> dict:
    """Sample n_sents sentences and report length statistics."""
    expand_fn = _make_expander(params)
    rng = np.random.default_rng(CORPUS_SEED + 99)

    RAND_BLOCK = max(2_000_000, MAX_SENT_TOKENS * 5)
    randoms = rng.random(RAND_BLOCK)
    r_pos = 0

    lengths = []
    for _ in range(n_sents):
        if r_pos > RAND_BLOCK - MAX_SENT_TOKENS * 5:
            randoms = rng.random(RAND_BLOCK)
            r_pos = 0
        tokens, r_pos_new = gen_sentence(expand_fn, roles, randoms, r_pos, rng)
        if not tokens:
            randoms = rng.random(RAND_BLOCK)
            r_pos = 0
            tokens, r_pos_new = gen_sentence(expand_fn, roles, randoms, r_pos, rng)
        r_pos = r_pos_new
        lengths.append(len(tokens))

    lens = np.array(lengths)
    stats = {
        "n_sampled": int(n_sents),
        "mean": float(lens.mean()),
        "median": float(np.median(lens)),
        "p90": float(np.percentile(lens, 90)),
        "p99": float(np.percentile(lens, 99)),
        "max": int(lens.max()),
        "frac_gt_512": float((lens > 512).mean()),
        "frac_gt_128": float((lens > 128).mean()),
        "frac_gt_64": float((lens > 64).mean()),
        "frac_capped": float((lens >= MAX_SENT_TOKENS).mean()),
        "grammar_params": params,
    }
    return stats


# ─── MI measurement (uses exp-062 estimator verbatim) ─────────────────────────

def measure_mi(corpus_path: Path) -> dict:
    """Measure β̂ on a flat uint16 token corpus using the exp-062 MI protocol."""
    ids = np.fromfile(corpus_path, dtype=np.uint16)
    # PCFG corpus: only 200 of 256 role symbols appear; treat all as symbols 0..255
    # mapped by the role assignment. For MI estimation we use symbol ranks 0..199.
    # Map each token id to its symbol rank (position in roles).
    all_roles = load_roles()
    # Build a token-id → rank lookup table
    lut = np.full(65536, -1, dtype=np.int32)
    rank = 0
    for role in ["Det", "N", "V", "Adj", "P", "Comp", "CC"]:
        for tok in all_roles[role]:
            lut[tok] = rank
            rank += 1
    sym = lut[ids]
    # Drop any tokens not in the grammar (shouldn't happen, but guard)
    sym = sym[sym >= 0].astype(np.uint16)
    n_sym = rank  # 200 distinct symbols used
    print(f"MI measurement: {len(sym)/1e6:.1f}M symbols, {n_sym} distinct symbols")
    out = mutual_information_profile(sym, np.random.default_rng(7))
    res_path = corpus_path.with_suffix(".mi.json")
    res_path.write_text(json.dumps(out, indent=1))
    print(f"{corpus_path.name}: β̂ = {out['beta_hat']:.4f}  "
          f"(R² {out['fit_r2']:.3f})  →  {res_path.name}")
    return out


# ─── save / load ──────────────────────────────────────────────────────────────

def save_corpus(name: str, tokens: np.ndarray, tag: str) -> Path:
    path = CORPUS_DIR / f"{name}_{tag}.bin"
    tokens.astype(np.uint16).tofile(path)
    print(f"saved {path}  ({len(tokens)/1e6:.0f}M tokens)")
    return path


# ─── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    roles = load_roles()

    if cmd == "stats":
        stats = sentence_length_stats(GRAMMAR_PARAMS, roles, n_sents=100_000)
        print(json.dumps(stats, indent=2))
        out = CORPUS_DIR / "sentence_length_stats.json"
        out.write_text(json.dumps(stats, indent=1))
        print(f"saved → {out}")

    elif cmd == "pilot":
        print(f"Generating PILOT corpus ({PILOT_TOKENS/1e6:.0f}M tokens) ...")
        print(f"Grammar params: {GRAMMAR_PARAMS}")
        tokens = gen_corpus(PILOT_TOKENS, GRAMMAR_PARAMS, CORPUS_SEED, roles)
        path = save_corpus("C-PCFG", tokens, "pilot")
        print("Measuring β̂ ...")
        measure_mi(path)

    elif cmd == "full":
        print(f"Generating FULL corpus ({(TRAIN_TOKENS+HELD_TOKENS)/1e6:.0f}M tokens) ...")
        print(f"Grammar params: {GRAMMAR_PARAMS}")
        tokens = gen_corpus(TRAIN_TOKENS + HELD_TOKENS, GRAMMAR_PARAMS, CORPUS_SEED, roles)
        path = save_corpus("C-PCFG", tokens, "full")
        print("Measuring β̂ on full corpus ...")
        measure_mi(path)

    elif cmd == "mi":
        if len(sys.argv) < 3:
            print("Usage: gen_pcfg.py mi <path.bin>")
            sys.exit(1)
        measure_mi(Path(sys.argv[2]))

    else:
        print(f"Unknown command: {cmd!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
