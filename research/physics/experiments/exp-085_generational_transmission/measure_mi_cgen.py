"""
Post-hoc MI measurement of the C-generated corpus (exp-085), using the
pre-registered exp-062 estimator (gen_corpora.py: mutual_information_profile,
free-floor fit of record + plain OLS diagnostic).

Post-hoc status is explicit: the exp-085 pre-registration noted the generated
corpus was not measured before training; this fills that gap for the paper.
The estimator itself is byte-identical to exp-062's (imported from the file).

    .venv/bin/python3 -m modal run research/physics/experiments/exp-085_generational_transmission/measure_mi_cgen.py

Ariel — July 21, 2026.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).resolve().parent
EXP062_DIR = SCRIPT_DIR.parent / "exp-062_corpus_statistics"

app = modal.App("exp085-mi")
vol_085 = modal.Volume.from_name("exp085-data", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.4.6", "scipy==1.17.1")
    .add_local_file(EXP062_DIR / "gen_corpora.py", "/root/gen_corpora.py")
    .add_local_file(EXP062_DIR / "alphabet.json", "/root/alphabet.json")
)

CORPUS = "/data085/C-generated_s0.bin"


@app.function(timeout=3600, volumes={"/data085": vol_085}, memory=16384, cpu=8, image=image)
def measure() -> dict:
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    import gen_corpora as gc

    ids = np.fromfile(CORPUS, dtype=np.uint16)
    print(f"corpus: {len(ids)/1e6:.0f}M tokens")

    # Same mapping rule as exp-062 measure_mi for natural corpora (prereg §5.3):
    # top-256 token ids by frequency ON THIS CORPUS -> symbol ranks.
    vals, counts = np.unique(ids, return_counts=True)
    top = vals[np.argsort(counts)[::-1][: gc.N_SYMBOLS]]
    lut = np.full(65536, gc.N_SYMBOLS - 1, dtype=np.int32)
    lut[top] = np.arange(gc.N_SYMBOLS)
    sym = lut[ids].astype(np.uint16)

    out = gc.mutual_information_profile(sym, np.random.default_rng(7))
    print(f"beta_hat = {out['beta_hat']:.4f} (R² {out['fit_r2']:.3f}), "
          f"plain OLS = {out['beta_hat_plain_ols']:.4f}")
    return out


@app.local_entrypoint()
def main():
    out = measure.remote()
    path = SCRIPT_DIR / "C-generated_s0.mi.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"Wrote {path}")
    print(f"beta_hat of record: {out['beta_hat']:.4f} (fit R² {out['fit_r2']:.3f})")
    print(f"plain OLS diagnostic: {out['beta_hat_plain_ols']:.4f}")
