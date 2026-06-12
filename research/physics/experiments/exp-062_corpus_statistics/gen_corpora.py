"""
exp-062 — Corpus generators + MI estimator (Phase 1.1, pre-registered).

Pre-registration of record: notes/2026-06-11_corpus_statistics_preregistration.md
(committed 2026-06-11 BEFORE this script was run). Do not change targets,
seeds, estimator, or fit windows without a dated log entry in notes.md.

Corpora (token-id sequences over the Pythia tokenizer space):
  C-SR    order-1 Markov chain, Dirichlet(0.1) rows, seed 3001
  C-PL15  quantized fGn, target MI exponent beta = 0.30, seed 3002
  C-PL25  quantized fGn, target beta = 0.50, seed 3003
  C-PL40  quantized fGn, target beta = 0.80, seed 3004
  C-NAT   TinyStories (Pythia tokenizer), doc-shuffle seed 3005

Synthetic alphabet: the 256 most frequent TinyStories token ids (recorded in
alphabet.json). Each corpus: TRAIN_TOKENS train + HELD_TOKENS held-out.

Usage:
  python gen_corpora.py alphabet                # build + save the 256-id alphabet
  python gen_corpora.py pilot <corpus>          # 100M-token pilot + beta-hat (calibration)
  python gen_corpora.py full <corpus>           # full corpus -> .bin (uint16 memmap)
  python gen_corpora.py mi <path.bin>           # measure beta-hat on an existing corpus
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
CORPUS_DIR = OUT / "corpora"
CORPUS_DIR.mkdir(exist_ok=True)

TRAIN_TOKENS = 1_050_000_000   # 2000 steps x 524288 tokens
HELD_TOKENS = 10_000_000
PILOT_TOKENS = 100_000_000
N_SYMBOLS = 256
SEG = 1 << 20                  # fGn synthesis segment length (Davies-Harte)

# Hurst design values (pre-registered): beta = 4 - 4H. Calibration loop may
# adjust H pre-training only; every adjustment must be logged in notes.md.
# Round 1 (design): H = {0.925, 0.875, 0.800} -> measured beta_hat {0.435, 0.590, 0.851}
#   (consistent overshoot from residual floor-fit bias; see notes.md calibration log).
# Round 2: linear correction beta_hat ~ 0.185 + 0.832*(4-4H) inverted for targets.
SPECS = {
    "C-SR":   {"kind": "markov", "seed": 3001},
    "C-PL15": {"kind": "fgn", "seed": 3002, "H": 0.9655, "beta_target": 0.30},
    "C-PL25": {"kind": "fgn", "seed": 3003, "H": 0.9053, "beta_target": 0.50},
    "C-PL40": {"kind": "fgn", "seed": 3004, "H": 0.8152, "beta_target": 0.80},
    "C-NAT":  {"kind": "natural", "seed": 3005},
}

# MI estimator (pre-registered, prereg §5.3)
MI_DISTANCES = np.unique(np.round(np.logspace(0, np.log2(512), 24, base=2)).astype(int))
MI_PAIRS = 10_000_000
MI_SHUFFLE_REPS = 20
MI_FIT_LO, MI_FIT_HI = 8, 256


# ── alphabet ─────────────────────────────────────────────────────────────────

def build_alphabet() -> np.ndarray:
    """256 most frequent TinyStories token ids under the Pythia tokenizer."""
    path = OUT / "alphabet.json"
    if path.exists():
        return np.array(json.loads(path.read_text())["ids"], dtype=np.uint16)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    counts = np.zeros(tok.vocab_size + 1024, dtype=np.int64)
    n_docs = 0
    for ex in ds:
        ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
        np.add.at(counts, ids, 1)
        n_docs += 1
        if n_docs >= 20000:   # ~10M tokens; ranks are stable well before this
            break
    ids = np.argsort(counts)[::-1][:N_SYMBOLS].astype(np.uint16)
    ids.sort()
    path.write_text(json.dumps({
        "ids": ids.tolist(), "n_docs_counted": n_docs,
        "tokenizer": "EleutherAI/pythia-70m",
    }, indent=1))
    print(f"alphabet: {N_SYMBOLS} ids saved ({n_docs} docs counted)")
    return ids


# ── generators (emit SYMBOLS 0..255; mapping to token ids happens at save) ───

def _markov_walk(cdf: np.ndarray, u: np.ndarray, s0: int) -> np.ndarray:
    out = np.empty(len(u), dtype=np.uint16)
    s = s0
    for i in range(len(u)):
        s = int(np.searchsorted(cdf[s], u[i]))
        out[i] = s
    return out


try:  # numba gives ~100x on the sequential walk; pure-numpy fallback kept identical
    from numba import njit

    @njit(cache=True)
    def _markov_walk_jit(cdf, u, s0):  # pragma: no cover
        out = np.empty(len(u), dtype=np.uint16)
        s = s0
        for i in range(len(u)):
            row = cdf[s]
            lo, hi = 0, len(row)
            x = u[i]
            while lo < hi:
                mid = (lo + hi) // 2
                if row[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            s = lo
            out[i] = s
        return out

    _MARKOV_IMPL = _markov_walk_jit
except ImportError:
    _MARKOV_IMPL = _markov_walk


def gen_markov(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T = rng.dirichlet(np.full(N_SYMBOLS, 0.1), size=N_SYMBOLS)  # rows
    cdf = np.cumsum(T, axis=1)
    s0 = int(rng.integers(N_SYMBOLS))
    out = np.empty(n, dtype=np.uint16)
    done = 0
    while done < n:  # chunked so the uniform buffer stays bounded
        take = min(1 << 26, n - done)
        u = rng.random(take)
        out[done:done + take] = _MARKOV_IMPL(cdf, u, s0)
        s0 = int(out[done + take - 1])
        done += take
        print(f"  markov: {done/1e6:.0f}M/{n/1e6:.0f}M", flush=True)
    return out


def _fgn_segment(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """Exact fGn via Davies-Harte circulant embedding."""
    k = np.arange(n + 1)
    gamma = 0.5 * ((k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))
    row = np.concatenate([gamma, gamma[-2:0:-1]])
    lam = np.fft.rfft(row).real
    lam = np.clip(lam, 0, None)
    m = len(row)
    z = rng.standard_normal(m // 2 + 1) + 1j * rng.standard_normal(m // 2 + 1)
    z[0] = z[0].real * np.sqrt(2)
    z[-1] = z[-1].real * np.sqrt(2)
    f = np.fft.irfft(z * np.sqrt(lam * m / 2), n=m)
    return f[:n]


def gen_fgn_symbols(n: int, H: float, seed: int) -> np.ndarray:
    """Quantized fGn: per-segment exact synthesis, 256 quantile bins.

    Quantile edges are fixed from the standard normal (fGn segments are
    stationary unit-variance Gaussian), so bins are equal-probability.
    """
    from scipy.stats import norm
    edges = norm.ppf(np.linspace(0, 1, N_SYMBOLS + 1)[1:-1])
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.uint16)
    done = 0
    while done < n:
        take = min(SEG, n - done)
        x = _fgn_segment(SEG, H, rng)[:take]
        out[done:done + take] = np.searchsorted(edges, x).astype(np.uint16)
        done += take
        if done % (SEG * 64) < SEG:
            print(f"  fgn H={H}: {done/1e6:.0f}M/{n/1e6:.0f}M", flush=True)
    return out


def gen_natural(n: int, seed: int, alphabet: np.ndarray) -> np.ndarray:
    """TinyStories -> Pythia token ids, doc-shuffled, packed (returns token ids)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.uint16)
    done = 0
    epoch = 0
    eos = tok.eos_token_id
    # TinyStories-train holds ~0.45B Pythia tokens < the 1.06B budget; corpus
    # wraps in re-shuffled epochs (generator amendment, logged in notes.md).
    while done < n:
        epoch += 1
        for idx in rng.permutation(len(ds)):
            ids = tok(ds[int(idx)]["text"], add_special_tokens=False)["input_ids"]
            ids = ids + [eos]
            take = min(len(ids), n - done)
            out[done:done + take] = np.asarray(ids[:take], dtype=np.uint16)
            done += take
            if done >= n:
                break
            if done % 50_000_000 < len(ids):
                print(f"  natural: {done/1e6:.0f}M/{n/1e6:.0f}M (epoch {epoch})", flush=True)
    return out


# ── MI estimator (pre-registered) ────────────────────────────────────────────

def mutual_information_profile(sym: np.ndarray, rng: np.random.Generator) -> dict:
    """Plug-in MI at each distance, shuffle-corrected; symbols must be 0..255."""
    n = len(sym)
    res = {}
    for d in MI_DISTANCES:
        m = min(MI_PAIRS, n - d)
        starts = rng.integers(0, n - d, size=m) if m < n - d else np.arange(n - d)
        a, b = sym[starts].astype(np.int64), sym[starts + d].astype(np.int64)
        joint = np.bincount(a * N_SYMBOLS + b, minlength=N_SYMBOLS ** 2
                            ).reshape(N_SYMBOLS, N_SYMBOLS).astype(np.float64)
        mi_obs = _mi_from_joint(joint)
        null = []
        for _ in range(MI_SHUFFLE_REPS):
            bp = b[rng.permutation(len(b))]
            jn = np.bincount(a * N_SYMBOLS + bp, minlength=N_SYMBOLS ** 2
                             ).reshape(N_SYMBOLS, N_SYMBOLS).astype(np.float64)
            null.append(_mi_from_joint(jn))
        res[int(d)] = {"mi": mi_obs, "mi_null": float(np.mean(null)),
                       "mi_corr": mi_obs - float(np.mean(null))}
    # beta-hat OF RECORD: free-floor fit MI_corr(d) = A·d^(−β) + c over the full
    # measured range (amendment 2026-06-11, validated on analytic ground truth:
    # plug-in MI under long memory has a d-independent excess floor that the
    # plain log-log OLS conflates with the decay; see prereg addendum).
    ds_all = np.array([d for d in sorted(res) if res[d]["mi_corr"] > 0])
    y_all = np.array([res[d]["mi_corr"] for d in ds_all])
    from scipy.optimize import least_squares

    def resid(p):
        log_amp, beta, log_c = p
        return np.log(np.exp(log_amp) * ds_all ** (-beta) + np.exp(log_c)) - np.log(y_all)

    best = None
    for b0 in (0.3, 0.6, 1.0):
        r = least_squares(
            resid, [np.log(y_all[0] * ds_all[0] ** b0), b0,
                    np.log(max(y_all[-1] * 0.5, 1e-9))],
            bounds=([-25, 0.01, -25], [5, 3, 0]))
        if best is None or r.cost < best.cost:
            best = r
    pred = np.log(np.exp(best.x[0]) * ds_all ** (-best.x[1]) + np.exp(best.x[2]))
    ly = np.log(y_all)
    r2_floor = 1 - np.sum((ly - pred) ** 2) / max(np.sum((ly - ly.mean()) ** 2), 1e-30)

    # plain windowed OLS retained as a secondary diagnostic
    m = (ds_all >= MI_FIT_LO) & (ds_all <= MI_FIT_HI)
    A = np.column_stack([np.ones(m.sum()), np.log(ds_all[m])])
    coef, *_ = np.linalg.lstsq(A, np.log(y_all[m]), rcond=None)

    return {"profile": res,
            "beta_hat": float(best.x[1]), "floor": float(np.exp(best.x[2])),
            "fit_r2": float(r2_floor), "n_fit_points": int(len(ds_all)),
            "beta_hat_plain_ols": float(-coef[1])}


def _mi_from_joint(joint: np.ndarray) -> float:
    n = joint.sum()
    p = joint / n
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    mask = p > 0
    return float((p[mask] * np.log(p[mask] / (px @ py)[mask])).sum())


# ── symbol<->token mapping and io ────────────────────────────────────────────

def save_corpus(name: str, sym_or_ids: np.ndarray, alphabet: np.ndarray,
                is_symbols: bool, tag: str) -> Path:
    ids = alphabet[sym_or_ids] if is_symbols else sym_or_ids
    path = CORPUS_DIR / f"{name}_{tag}.bin"
    ids.astype(np.uint16).tofile(path)
    print(f"saved {path} ({len(ids)/1e6:.0f}M tokens)")
    return path


def generate(name: str, n_tokens: int, tag: str) -> Path:
    spec = SPECS[name]
    alphabet = build_alphabet()
    if spec["kind"] == "markov":
        sym = gen_markov(n_tokens, spec["seed"])
        return save_corpus(name, sym, alphabet, True, tag)
    if spec["kind"] == "fgn":
        sym = gen_fgn_symbols(n_tokens, spec["H"], spec["seed"])
        return save_corpus(name, sym, alphabet, True, tag)
    ids = gen_natural(n_tokens, spec["seed"], alphabet)
    return save_corpus(name, ids, alphabet, False, tag)


def measure_mi(path: Path) -> None:
    ids = np.fromfile(path, dtype=np.uint16)
    alphabet = build_alphabet()
    # map token ids -> symbol ranks; ids outside alphabet -> rank by frequency cap
    lut = np.full(65536, -1, dtype=np.int32)
    lut[alphabet] = np.arange(len(alphabet))
    sym = lut[ids]
    if (sym < 0).any():   # natural corpus: bucket all non-alphabet ids into one symbol? No -
        # prereg §5.3: top-256-rank mapping for C-NAT. Recompute ranks on THIS corpus.
        vals, counts = np.unique(ids, return_counts=True)
        top = vals[np.argsort(counts)[::-1][:N_SYMBOLS]]
        lut = np.full(65536, N_SYMBOLS - 1, dtype=np.int32)
        lut[top] = np.arange(N_SYMBOLS)
        sym = lut[ids]
    out = mutual_information_profile(sym.astype(np.uint16), np.random.default_rng(7))
    res_path = path.with_suffix(".mi.json")
    res_path.write_text(json.dumps(out, indent=1))
    print(f"{path.name}: beta_hat = {out['beta_hat']:.4f} (R² {out['fit_r2']:.3f}) -> {res_path.name}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "alphabet":
        build_alphabet()
    elif cmd == "pilot":
        p = generate(sys.argv[2], PILOT_TOKENS, "pilot")
        measure_mi(p)
    elif cmd == "full":
        generate(sys.argv[2], TRAIN_TOKENS + HELD_TOKENS, "full")
    elif cmd == "mi":
        measure_mi(Path(sys.argv[2]))
    else:
        raise SystemExit(f"unknown command {cmd}")
