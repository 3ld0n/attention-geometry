"""
Corpus census: does YOUR corpus induce the conformal attention structure
that natural text does?

Trains a Pythia-70m-class GPT-NeoX from scratch on your tokenized corpus under
the frozen pre-registered protocol (exp-062 lineage: 2000 steps, 1.05B tokens),
runs the attention-exponent census on the final checkpoint, and places the
result on the published corpus-formation ladder (anchors.json).

The training and measurement code is NOT duplicated here: this wrapper executes
the frozen scripts from research/physics/experiments/exp-062_corpus_statistics/
verbatim (with only their output directory redirected), so a kit run is
protocol-identical to the published anchors. Requires the full repository
checkout, not just this directory.

Input format: a flat binary file of uint16 token ids (np.memmap-compatible),
>= 1,048,576,000 tokens (2000 steps x 1024 seqs x 512 tokens). Any tokenizer
with vocab <= 50304 works; the model trains on ids, not text.

Usage:
  python census_corpus.py my_corpus.bin --workdir census_run
  python census_corpus.py my_corpus.bin --workdir census_run --micro-batch 32
  python census_corpus.py --compare-only census_run/measurements/my_corpus.json

Cost: ~75 min on an A100-40GB (bf16 autocast); the census itself is ~10 min.
CPU-only is not realistic for the training step.

Ariel — July 21, 2026.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KIT_DIR = Path(__file__).resolve().parent
EXP062_DIR = KIT_DIR.parent / "experiments" / "exp-062_corpus_statistics"
ANCHORS = json.loads((KIT_DIR / "anchors.json").read_text())

TOKENS_NEEDED = 2000 * 1024 * 512
ALPHABET_MAX = 4096  # <= this many distinct ids -> census draws from them


def run_patched(script_path: Path, out_override: Path, argv: list[str]) -> None:
    """Execute a frozen exp-062 script verbatim, redirecting only its OUT dir."""
    source = script_path.read_text()
    patched = source.replace(
        "OUT = Path(__file__).resolve().parent",
        f"OUT = Path('{out_override}')",
    )
    assert patched != source, f"OUT anchor line not found in {script_path}"
    sys.argv = argv
    ns: dict = {"__name__": "__main__", "__file__": str(script_path)}
    exec(compile(patched, str(script_path), "exec"), ns)  # noqa: S102


def corpus_alphabet(corpus_path: Path) -> np.ndarray:
    """Distinct token ids in the corpus (chunked bincount over uint16)."""
    counts = np.zeros(65536, dtype=np.int64)
    mm = np.memmap(corpus_path, dtype=np.uint16, mode="r")
    chunk = 50_000_000
    for i in range(0, len(mm), chunk):
        counts += np.bincount(mm[i:i + chunk], minlength=65536)
    return np.nonzero(counts)[0]


def compare(result: dict) -> None:
    """Print the corpus ladder with this result inserted in rank position."""
    rungs = [dict(a) for a in ANCHORS["corpus_ladder"]["anchors"]]
    mine = {
        "run": result.get("tag", "your corpus"),
        "corpus": ">>> YOUR CORPUS <<<",
        "n_conformal": result["n_conformal"],
        "n_syk_near": result["n_syk_near"],
        "delta_median_conformal": result["delta_median_conformal"],
        "forms": result["n_conformal"] >= 10,
    }
    rows = sorted(rungs + [mine], key=lambda r: r["n_conformal"])
    print(f"\n{'corpus':58s} {'conformal':>9s} {'Δ_med':>7s} {'forms':>6s}")
    print("-" * 84)
    for r in rows:
        dm = r["delta_median_conformal"]
        dm_s = f"{dm:.3f}" if dm is not None else "—"
        marker = "  <== you" if r["corpus"] == ">>> YOUR CORPUS <<<" else ""
        print(f"{r['corpus'][:58]:58s} {r['n_conformal']:>6d}/48 {dm_s:>7s} "
              f"{'YES' if r['forms'] else 'no':>6s}{marker}")
    print(f"\nFormation criterion: >= 10/48 conformal heads.")
    if mine["forms"]:
        print("Your corpus FORMS the conformal population — it patterns with "
              "natural text (C-NAT band: 11–15/48).")
    else:
        print("Your corpus does NOT form the conformal population — it patterns "
              "with the engineered/generated corpora, whatever its surface "
              "statistics. (See exp-085: the generated corpus had MORE "
              "long-range MI than natural text and still failed.)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", help="uint16 .bin token file")
    ap.add_argument("--workdir", default="census_run",
                    help="output dir for checkpoints + measurements")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--init-seed", type=int, default=1000)
    ap.add_argument("--data-seed", type=int, default=2000)
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--full-vocab", action="store_true",
                    help="census over full 50304 vocab regardless of corpus alphabet")
    ap.add_argument("--compare-only", default=None,
                    help="skip train/measure; compare an existing result JSON")
    args = ap.parse_args()

    if args.compare_only:
        compare(json.loads(Path(args.compare_only).read_text()))
        return

    if not args.corpus:
        ap.error("corpus .bin required (or use --compare-only)")
    if not EXP062_DIR.exists():
        raise SystemExit(
            f"Frozen protocol scripts not found at {EXP062_DIR}.\n"
            "census_corpus.py needs the full repository checkout "
            "(the training/measurement code is single-sourced there).")

    corpus_path = Path(args.corpus).resolve()
    run_name = args.run_name or corpus_path.stem
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    mm = np.memmap(corpus_path, dtype=np.uint16, mode="r")
    if len(mm) < TOKENS_NEEDED:
        raise SystemExit(f"corpus too small: {len(mm):,} tokens "
                         f"< {TOKENS_NEEDED:,} required by the frozen protocol")
    print(f"corpus: {len(mm) / 1e9:.2f}B tokens")

    # Census alphabet: the protocol draws random tokens from the MODEL'S
    # TRAINING ALPHABET (exp-062 prereg §5.1). Small-alphabet corpora get
    # their observed id set; text-tokenizer corpora get the full vocab.
    measure_args = ["--full-vocab"]
    if not args.full_vocab:
        ids = corpus_alphabet(corpus_path)
        print(f"alphabet: {len(ids)} distinct token ids")
        if ids.max() >= 50304:
            raise SystemExit(f"token id {ids.max()} >= vocab size 50304")
        if len(ids) <= ALPHABET_MAX:
            alpha_path = workdir / "alphabet.json"
            alpha_path.write_text(json.dumps({"ids": ids.tolist()}))
            measure_args = ["--alphabet", str(alpha_path)]
            print(f"census will draw from the {len(ids)}-id corpus alphabet")
        else:
            print("large alphabet -> census over full vocab")

    print(f"\n=== training (2000 steps, ~75 min on A100-40GB) -> {workdir} ===")
    run_patched(EXP062_DIR / "train.py", workdir, [
        "train.py", str(corpus_path), run_name,
        f"--init-seed={args.init_seed}", f"--data-seed={args.data_seed}",
        f"--micro-batch={args.micro_batch}",
    ])

    print("\n=== census (frozen protocol) ===")
    ckpt = workdir / "runs" / run_name / "step_2000"
    run_patched(EXP062_DIR / "measure.py", workdir,
                ["measure.py", str(ckpt), run_name] + measure_args)

    result = json.loads((workdir / "measurements" / f"{run_name}.json").read_text())
    compare(result)


if __name__ == "__main__":
    main()
