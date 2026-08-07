"""Supplement to g1_fixed_point.py run 5: mode identification at beta*J=30.
Which reparameterization mode number n is each top eigenvector pair?
Saves the fixed point + top eigenvectors for future P6a instrument design."""

import numpy as np
from g1_fixed_point import (BilocalMap, solve_syk_ti, ti_to_matrix,
                            fit_delta, reparam_mode, dense_jacobian_spectrum)

def main():
    beta, bJ, Nsm = 1.0, 30.0, 96
    J = bJ / beta
    bm = BilocalMap(beta, J, 0.0, N=Nsm)
    tt, gg, _, _ = solve_syk_ti(beta, J, 0.0)
    Gm = ti_to_matrix(beta, tt, gg, Nsm)
    Gs, _, _ = bm.converge_matrix_fixed_point(Gm, max_iter=8000)
    evals_c, evecs, pairs, sym_err, max_imag = dense_jacobian_spectrum(bm, Gs)
    evals = evals_c.real
    Dl, _ = fit_delta(beta, tt, gg, J)
    if not np.isfinite(Dl):
        Dl = 0.25

    iu, ju = np.triu_indices(Nsm, k=1)
    print(f"beta*J={bJ}, Delta_fit={Dl:.4f}")
    print("per-mode overlap of top eigenvector pairs (rows: eig; cols: n):")
    header = "   eig     | " + " | ".join(f"n={n}" for n in range(1, 7))
    print(header)
    for r in range(1, 11):
        vec = np.real(evecs[:, -r])
        dmat = np.zeros((Nsm, Nsm))
        dmat[iu, ju] = vec / np.sqrt(2)
        dmat -= dmat.T
        nrm2 = bm.inner(dmat, dmat)
        row = []
        for n_mode in range(1, 7):
            tot = 0.0
            # project onto the 2D (sin, cos) subspace for this n
            f1 = reparam_mode(bm, Gs, Dl, n_mode, phase=0.0)
            f2 = reparam_mode(bm, Gs, Dl, n_mode, phase=np.pi / 2)
            # orthonormalize the pair
            f1 = f1 / np.sqrt(bm.inner(f1, f1))
            f2 = f2 - bm.inner(f1, f2) * f1
            f2 = f2 / np.sqrt(bm.inner(f2, f2))
            tot = (bm.inner(f1, dmat) ** 2 + bm.inner(f2, dmat) ** 2) / nrm2
            row.append(tot)
        print(f"  {evals[-r]:+.4f} | "
              + " | ".join(f"{v:.2f}" for v in row))

    # save for instrument design
    top = {f"evec{r}": np.real(evecs[:, -r]) for r in range(1, 5)}
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "logs", "g1_top_modes_bJ30.npz")
    np.savez(out, evals=evals, Gstar=Gs, tau=bm.tau, Delta=Dl,
             pairs_iu=iu, pairs_ju=ju, **top)
    print(f"saved {out}")

if __name__ == "__main__":
    main()
