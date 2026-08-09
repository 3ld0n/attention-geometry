"""Exact sign constraints on the connected bilocal, checked before exp-107 is designed.

Three algebraic claims, each verified on random matrices to machine precision.
They fix what exp-107 can and cannot find, which is why they are checked first:

C1  mean(K) = ||v_bar||^2  for K = V V^T.
C2  sum_{a != b} K_tilde_{ab} = - sum_a ||v_a - v_bar||^2,  K_tilde = K - mean(K) 11^T.
    The off-diagonal mass of the centered value Gram is negative ALWAYS, with
    magnitude exactly the total value-vector variance. Not a random-token artifact.
C3  sum_{i,j} (A K_tilde A^T)_{ij} = ||sum_a m_a v_a||^2 - ||sum_a v_a||^2,
    where m = A^T 1 is the incoming-attention mass per key.
    NOT sign-definite. So a positive connected G-profile is not forbidden.

C2 and C3 together are the design constraint: natural text cannot remove the
negative mass (C2), so if it produces a positive connected profile over the census
window it must do so by SHAPING K_tilde -- positive near-diagonal, negative far --
and that shape is measurable in the same forward pass.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
RNG = np.random.default_rng(20260808)


def row_stochastic(n: int, sink: float = 0.0) -> np.ndarray:
    """Causal row-stochastic A, optionally with an attention sink on position 0."""
    logits = RNG.normal(size=(n, n))
    logits[np.triu_indices(n, k=1)] = -np.inf
    logits[:, 0] += sink
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(axis=1, keepdims=True)


def check(n: int, d: int, sink: float) -> dict:
    V = RNG.normal(size=(n, d))
    K = V @ V.T
    vbar = V.mean(axis=0)

    c1_lhs, c1_rhs = K.mean(), vbar @ vbar

    Kt = K - K.mean()
    off = Kt.sum() - np.trace(Kt)
    variance = ((V - vbar) ** 2).sum()

    A = row_stochastic(n, sink=sink)
    m = A.sum(axis=0)
    total = (A @ Kt @ A.T).sum()
    c3_rhs = (m @ V) @ (m @ V) - (V.sum(axis=0) @ V.sum(axis=0))

    scale = max(abs(c1_rhs), 1.0)
    return {
        "n": n, "d": d, "sink": sink,
        "C1_rel_err": abs(c1_lhs - c1_rhs) / scale,
        "C2_rel_err": abs(off + variance) / max(variance, 1.0),
        "C2_off_diagonal_sum": off,
        "C2_is_negative": bool(off < 0),
        "C3_rel_err": abs(total - c3_rhs) / max(abs(c3_rhs), 1.0),
        "C3_total_connected": total,
        "C3_sign": int(np.sign(total)),
    }


def main() -> None:
    cells = [check(n, d, sink)
             for n in (64, 256)
             for d in (8, 64)
             for sink in (0.0, 6.0)]

    worst = {k: max(c[k] for c in cells)
             for k in ("C1_rel_err", "C2_rel_err", "C3_rel_err")}
    signs = sorted({c["C3_sign"] for c in cells})

    print(f"cells: {len(cells)}")
    for k, v in worst.items():
        print(f"  worst {k}: {v:.3e}")
    print(f"  C2 negative in all cells: {all(c['C2_is_negative'] for c in cells)}")
    print(f"  C3 signs observed: {signs}  "
          f"({'both signs -> not sign-definite' if len(signs) > 1 else 'ONE SIGN ONLY'})")

    for c in cells:
        if c["sink"] in (0.0, 6.0) and c["n"] == 256 and c["d"] == 64:
            print(f"  n=256 d=64 sink={c['sink']}: total connected = "
                  f"{c['C3_total_connected']:+.4f}")

    (HERE / "sign_constraints.json").write_text(json.dumps(
        {"cells": cells, "worst": worst,
         "C2_negative_everywhere": all(c["C2_is_negative"] for c in cells),
         "C3_signs_observed": signs}, indent=1, default=float))


if __name__ == "__main__":
    main()
