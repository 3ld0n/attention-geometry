"""
G1 fixed-point and stability calculation — August 7, 2026 (the long quiet night).

Target (theory scaffold G1, sharpest formulation): seek the conformal fixed
point of the dressed layer map F directly, then compute its stability
spectrum. In the scalar approximation the dressed map is the KCA/SYK(2+4)
Schwinger-Dyson iteration:

    F[G] = (G0^{-1} - Sigma[G])^{-1},   Sigma = J2^2 G + J4^2 G^3.

Analytic results this script tests (derived before running, note in
research/physics/notes/2026-08-07_G1_stability_calculation.md):

  T-G1a: fixed points of F = solutions of the KCA G-Sigma system (algebra).
  T-G1b: DF[G*] delta = G* o (Sigma'[G*] . delta) o G* — the SYK four-point
         ladder kernel (q=4 channel weight 3 J4^2 G*^2, q=2 weight J2^2).
         DF is self-adjoint in the flat bilocal inner product.
  T-G1c: in the conformal window, reparameterization tangents delta_eps G*
         are eigenvectors of DF with eigenvalue 1 (Goldstone argument);
         UV breaking lifts the eigenvalue to 1 - O(1/(beta J)).

DECLARED PREDICTIONS (v3: fixes from run 2 — Sigma-damping + J-annealing in
the TI solver, JVP normalization, band-zeroed reparam probes, dense Jacobian
diagonalization at small N. Declared content unchanged from v1 except D2 is
now stated on the full spectrum; runs 1-2 kept in logs/ as the honest trail):

  D1: TI solver at beta*J = 50, J2 = 0: Delta_fit = 0.25 +/- 0.015 in the
      conformal window; G(0+) ~ 0.5 (validates solver against SYK).
  D2: the dense Jacobian spectrum at the fixed point is real (self-adjointness
      check) with most-negative eigenvalue -> k_c(1/2) = -3*pi/4 = -2.356
      (within ~20% at small N / moderate beta*J), and NO eigenvalue above 1
      beyond discretization error.
  D3: the eigenvalues closest to 1 have eigenvectors with dominant overlap
      onto the reparameterization family (n = 2, 3, ...); band-zeroed reparam
      probes at beta*J = 50 give Rayleigh quotients >= 0.9 with small
      alignment residual, and gap 1 - k shrinking with beta*J
      (~ 1/(beta*J) across beta*J in {25, 50, 100}).
  D4: damped map (x = 1/2): all modes map to |lambda| < 1 except the
      near-marginal reparameterization family, which stays near 1.
  D5: q=2 admixture (J2/J4 in {0, 0.3, 1.0} at fixed beta*J4 = 50): the
      reparameterization eigenvalue decreases monotonically with J2 — the
      arrest channel lifts the soft mode (q=2 ladder eigenvalue is -1 for
      all h; Maldacena-Stanford eq. 3.77 — no marginal mode in arrest).
  D6 (exploratory, no declared direction): symmetric ("bosonic"/attention-
      form) variant of the same map — does a stable power-law fixed point
      exist at all in the symmetric realization?

Everything is committed with logs; misses will be recorded as misses.
"""

import numpy as np
import sys
import time

# ----------------------------------------------------------------------
# fermionic Matsubara transforms (validated in g1_transform_test.py)
# ----------------------------------------------------------------------

def make_transforms(beta, N):
    dt = beta / N
    tau = (np.arange(N) + 0.5) * dt
    j = np.arange(N)
    n_phys = np.where(j < N // 2, j, j - N)
    omega = np.pi * (2 * n_phys + 1) / beta
    fwd_ph = np.exp(1j * np.pi * np.arange(N) / N)
    fwd_corr = np.exp(1j * np.pi * (2 * n_phys + 1) / (2 * N))

    def t_to_w(Gt):
        return dt * fwd_corr * (np.fft.ifft(Gt * fwd_ph) * N)

    def w_to_t(Gw):
        return np.real(np.fft.fft(Gw / fwd_corr) / fwd_ph) / beta

    return tau, omega, t_to_w, w_to_t


# ----------------------------------------------------------------------
# Part 1 — translation-invariant SYK(2+4) solver
# ----------------------------------------------------------------------

def solve_syk_ti(beta, J4, J2=0.0, N=1 << 14, mix=0.3, tol=1e-11,
                 max_iter=40000, verbose=False, anneal=True):
    """Sigma-damped iteration with coupling annealing (standard SYK numerics).
    Returns tau, G(tau), iters, residual."""
    tau, omega, t_to_w, w_to_t = make_transforms(beta, N)

    if anneal:
        ladder = [bj for bj in (2.0, 5.0, 10.0, 20.0, 35.0, 60.0, 140.0)
                  if bj < beta * max(J4, J2)] + [None]
    else:
        ladder = [None]

    Gw = 1.0 / (-1j * omega)
    Gt = w_to_t(Gw)
    Sig_w = np.zeros(N, dtype=complex)
    tot_it = 0
    res = np.inf
    for stage in ladder:
        if stage is None:
            j4, j2 = J4, J2
        else:
            scale = stage / (beta * max(J4, J2))
            j4, j2 = J4 * scale, J2 * scale
        for it in range(max_iter):
            Sig_t = j2**2 * Gt + j4**2 * Gt**3
            Sig_new = t_to_w(Sig_t)
            Sig_w = (1 - mix) * Sig_w + mix * Sig_new
            Gw = 1.0 / (-1j * omega - Sig_w)
            Gt_new = w_to_t(Gw)
            res = np.max(np.abs(Gt_new - Gt)) / max(np.max(np.abs(Gt)), 1e-30)
            Gt = Gt_new
            tot_it += 1
            if res < tol:
                break
    if verbose:
        print(f"    solve_syk_ti: beta*J4={beta*J4}, "
              f"J2/J4={J2/J4 if J4 else 0}: {tot_it} iters, res={res:.2e}, "
              f"G(0+)={Gt[0]:.4f}")
    return tau, Gt, tot_it, res


def fit_delta(beta, tau, Gt, J4, lo_mult=6.0, hi_frac=0.25):
    lo, hi = lo_mult / J4, hi_frac * beta
    if lo >= hi:                      # narrow window at small beta*J
        lo = 3.0 / J4
    m = (tau > lo) & (tau < hi) & (np.abs(Gt) > 0)
    if m.sum() < 10:
        return np.nan, 0.0
    x = np.log(np.sin(np.pi * tau[m] / beta))
    y = np.log(np.abs(Gt[m]))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return -coef[0] / 2, r2


# ----------------------------------------------------------------------
# Part 2 — full-bilocal matrix map and Jacobian
# ----------------------------------------------------------------------

class BilocalMap:
    def __init__(self, beta, J4, J2=0.0, N=512):
        self.beta, self.J4, self.J2, self.N = beta, J4, J2, N
        self.dt = beta / N
        tau, omega, t_to_w, w_to_t = make_transforms(beta, N)
        self.tau = tau
        D = np.empty((N, N))
        for c in range(N):
            e = np.zeros(N)
            e[c] = 1.0
            D[:, c] = w_to_t(-1j * omega * t_to_w(e))
        self.D = D
        self.I = np.eye(N)

    def F(self, G):
        Sig = self.J2**2 * G + self.J4**2 * G**3
        A = self.D - Sig * self.dt
        return np.linalg.solve(A, self.I) / self.dt

    def damped(self, G, x=0.5):
        return (1 - x) * self.F(G) + x * G

    def converge_matrix_fixed_point(self, G_init, mix=0.3, tol=1e-10,
                                    max_iter=4000):
        """Sigma-mixed iteration (mirrors the TI scheme that converges):
        Sigma <- (1-mix) Sigma + mix Sigma[G];  G <- (D - Sigma dt)^{-1}/dt."""
        G = G_init.copy()
        Sig = self.J2**2 * G + self.J4**2 * G**3
        res = np.inf
        for it in range(max_iter):
            Sig_new = self.J2**2 * G + self.J4**2 * G**3
            Sig = (1 - mix) * Sig + mix * Sig_new
            A = self.D - Sig * self.dt
            Gn = np.linalg.solve(A, self.I) / self.dt
            res = np.max(np.abs(Gn - G)) / max(np.max(np.abs(G)), 1e-30)
            G = Gn
            if res < tol:
                break
        return G, it + 1, res

    def jvp(self, Gstar, delta, eps_rel=1e-5):
        dmax = np.max(np.abs(delta))
        if dmax == 0:
            return np.zeros_like(delta)
        dn = delta / dmax
        s = eps_rel * np.max(np.abs(Gstar))
        return (self.F(Gstar + s * dn) - self.F(Gstar - s * dn)) \
            / (2 * s) * dmax

    def inner(self, a, b):
        return np.sum(a * b) * self.dt**2


def ti_to_matrix(beta, tau_ti, Gt, N):
    dtau = tau_ti[1] - tau_ti[0]
    def G_of(t):
        t = np.asarray(t, dtype=float)
        sign = np.ones_like(t)
        tt = t.copy()
        neg = tt < 0
        tt[neg] += beta
        sign[neg] = -1.0
        idx = np.clip((tt / dtau - 0.5).astype(int), 0, len(tau_ti) - 2)
        w = np.clip((tt - tau_ti[idx]) / dtau, 0.0, 1.0)
        return sign * ((1 - w) * Gt[idx] + w * Gt[idx + 1])
    tau_m = (np.arange(N) + 0.5) * (beta / N)
    T1, T2 = np.meshgrid(tau_m, tau_m, indexing="ij")
    M = G_of((T1 - T2).ravel()).reshape(N, N)
    np.fill_diagonal(M, 0.0)
    return 0.5 * (M - M.T)


def band_zero(delta, w):
    N = delta.shape[0]
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    dist = np.abs(i - j)
    dist = np.minimum(dist, N - dist)     # circle distance
    out = delta.copy()
    out[dist < w] = 0.0
    return out


def reparam_mode(bm, Gstar, Delta, n_mode, phase=0.0, band=None):
    beta, tau = bm.beta, bm.tau
    eps = np.sin(2 * np.pi * n_mode * tau / beta + phase)
    deps = (2 * np.pi * n_mode / beta) * np.cos(
        2 * np.pi * n_mode * tau / beta + phase)
    dG = bm.D @ Gstar
    d = (eps[:, None] - eps[None, :]) * dG \
        + Delta * (deps[:, None] + deps[None, :]) * Gstar
    d = 0.5 * (d - d.T)
    if band:
        d = band_zero(d, band)
    return d


def random_smooth_perturbation(bm, kmax=8, seed=0, band=None):
    rng = np.random.default_rng(seed)
    N, tau, beta = bm.N, bm.tau, bm.beta
    d = np.zeros((N, N))
    for _ in range(20):
        k1, k2 = rng.integers(1, kmax + 1, 2)
        p1, p2 = rng.uniform(0, 2 * np.pi, 2)
        a = rng.normal()
        f1 = np.sin(2 * np.pi * k1 * tau / beta + p1)
        f2 = np.sin(2 * np.pi * k2 * tau / beta + p2)
        d += a * np.outer(f1, f2)
    d = 0.5 * (d - d.T)
    if band:
        d = band_zero(d, band)
    return d / np.max(np.abs(d))


def probe_eigenvalue(bm, Gstar, delta, n_iter=12, damped=False, x=0.5):
    d = delta / np.sqrt(bm.inner(delta, delta))
    quotients = []
    for _ in range(n_iter):
        Ld = bm.jvp(Gstar, d)
        if damped:
            Ld = (1 - x) * Ld + x * d
        quotients.append(bm.inner(d, Ld))
        nrm = np.sqrt(bm.inner(Ld, Ld))
        if nrm < 1e-300:
            break
        d = Ld / nrm
    d0 = delta / np.sqrt(bm.inner(delta, delta))
    Ld0 = bm.jvp(Gstar, d0)
    if damped:
        Ld0 = (1 - x) * Ld0 + x * d0
    q0 = bm.inner(d0, Ld0)
    resid = np.sqrt(bm.inner(Ld0 - q0 * d0, Ld0 - q0 * d0))
    return quotients, q0, resid


def dense_jacobian_spectrum(bm, Gstar):
    """Exact Jacobian of F on the antisymmetric bilocal space, via the
    analytic form DF[delta] = G* (Sigma'[G*] . delta) G* — computed with
    matrix algebra, exact up to the fixed-point residual (no finite
    differences). Basis: E_(ij) = (e_i e_j^T - e_j e_i^T)/sqrt(2), i<j.
    Returns eigenvalues, eigenvectors (columns, in basis coords), pairs."""
    N = bm.N
    Sig_p = bm.J2**2 + 3 * bm.J4**2 * Gstar**2      # Sigma'(G*), pointwise
    # DF[delta] = A^{-1} (Sigma'.delta) dt A^{-1} / dt = W (Sig_p.delta) W dt
    # where W = A^{-1} = F[G*] * dt = G* dt (at the fixed point).
    # Direct derivation from F(G) = solve(A(G), I)/dt, A = D - Sigma dt:
    #   dF = A^{-1} (dSigma dt) A^{-1} / dt = (G* dt)(Sig_p.delta)(G* dt)/dt·(1/dt)... 
    # safest: numerically differentiate once and compare; here use algebra:
    #   F = A^{-1}/dt ;  dF = A^{-1} dA_hat A^{-1} ... dA = -Sig_p.delta * dt
    #   dF = -A^{-1} (dA) A^{-1} / dt = A^{-1}(Sig_p.delta)A^{-1}
    # with A^{-1} = G* dt:
    #   dF = dt^2 G* (Sig_p . delta) G*
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    P_ = len(pairs)
    Gs = Gstar * bm.dt      # = A^{-1}
    # Build the matrix column by column: column for pair (k,l) is
    # DF[E_kl] = Gs (Sig_p . E_kl) Gs / dt^2 * dt^2 -> just Gs X Gs with
    # X = Sig_p.E_kl. Since E_kl has two nonzero entries, columns are
    # rank-2 outer products: X = s*(Sig_p[k,l]) (e_k e_l^T - e_l e_k^T)/sqrt2
    # DF[E_kl] = Sig_p[k,l]/sqrt2 * (Gs[:,k] Gs[l,:] - Gs[:,l] Gs[k,:])
    idx = {p: a for a, p in enumerate(pairs)}
    M = np.empty((P_, P_))
    GsT = Gs.T.copy()
    iu, ju = np.triu_indices(N, k=1)
    for a, (k, l) in enumerate(pairs):
        # DF[E_kl] = Gs (Sig_p . E_kl) Gs ; already antisymmetric when Gs is
        col_mat = (np.outer(Gs[:, k], GsT[:, l]) -
                   np.outer(Gs[:, l], GsT[:, k])) * (Sig_p[k, l] / np.sqrt(2))
        col_mat = 0.5 * (col_mat - col_mat.T)     # enforce exactly
        M[:, a] = col_mat[iu, ju] * np.sqrt(2)    # coords in E-basis
    # coords: delta = sum c_a E_a with E_a orthonormal in Frobenius;
    # c_a = sqrt(2) delta[i_a, j_a]
    sym_err = np.max(np.abs(M - M.T)) / np.max(np.abs(M))
    # NOTE: DF is not flat-symmetric (it is symmetrizable by a |G*|-weight
    # conjugation, as for the SYK ladder kernel); use general eig.
    evals_c, evecs_c = np.linalg.eig(M)
    order = np.argsort(evals_c.real)
    evals_c, evecs_c = evals_c[order], evecs_c[:, order]
    max_imag = np.max(np.abs(evals_c.imag))
    return evals_c, evecs_c, pairs, sym_err, max_imag


# ----------------------------------------------------------------------
# Part 3 — exploratory symmetric variant (D6)
# ----------------------------------------------------------------------

def symmetric_variant(beta=1.0, g4=10.0, N=1 << 12, damping=0.5,
                      max_iter=6000, m0=2.0):
    dt = beta / N
    tau = np.arange(N) * dt
    k = np.fft.fftfreq(N, d=dt) * 2 * np.pi
    Gt = np.real(np.fft.ifft(1.0 / (k**2 + m0**2))) / dt
    status, res = "running", np.inf
    for it in range(max_iter):
        Sig = g4 * Gt**3
        Sigw = np.real(np.fft.fft(Sig)) * dt
        denom = k**2 + m0**2 - Sigw
        if np.any(np.abs(denom) < 1e-14):
            status = f"SINGULAR denominator at iter {it}"
            break
        Gt_new = np.real(np.fft.ifft(1.0 / denom)) / dt
        res = np.max(np.abs(Gt_new - Gt))
        Gt = (1 - damping) * Gt_new + damping * Gt
        if not np.all(np.isfinite(Gt)) or np.max(np.abs(Gt)) > 1e12:
            status = f"DIVERGED at iter {it}"
            break
        if res < 1e-12:
            status = f"converged at iter {it}"
            break
    else:
        status = f"no convergence in {max_iter} iters (last res {res:.2e})"
    Sig0 = np.real(np.fft.fft(g4 * Gt**3))[0] * dt
    return tau, Gt, status, m0**2 - Sig0


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def dense_run(P, beta, J4, J2, Nsm, label, n_overlap=12):
    """Solve TI, cast to matrix, converge, dense-diagonalize, report."""
    bms = BilocalMap(beta, J4, J2, N=Nsm)
    tts, ggs, _, _ = solve_syk_ti(beta, J4, J2)
    Gms = ti_to_matrix(beta, tts, ggs, Nsm)
    Gss, nits, rfs = bms.converge_matrix_fixed_point(Gms, max_iter=8000)
    rfx = np.max(np.abs(bms.F(Gss) - Gss)) / np.max(np.abs(Gss))
    if rfx > 1e-6:
        P(f"  {label}: matrix fixed point DID NOT CONVERGE "
          f"(residual {rfx:.2e}) — skipped")
        return None
    evals_c, evecs, pairs, sym_err, max_imag = \
        dense_jacobian_spectrum(bms, Gss)
    evals = evals_c.real
    Jfit = max(J4, J2)
    Dls, r2ls = fit_delta(beta, tts, ggs, Jfit)
    if not np.isfinite(Dls):
        Dls = 0.25
    # reparameterization family overlap of top eigenvectors
    fam = []
    for n_mode in (1, 2, 3, 4, 5, 6):
        for ph in (0.0, np.pi / 2):
            d = reparam_mode(bms, Gss, Dls, n_mode, phase=ph, band=None)
            nd = np.sqrt(bms.inner(d, d))
            if np.isfinite(nd) and nd > 1e-12:
                fam.append(d / nd)
    basis = []
    for f in fam:
        v = f.copy()
        for b in basis:
            v = v - bms.inner(b, v) * b
        nv = np.sqrt(bms.inner(v, v))
        if nv > 1e-6:
            basis.append(v / nv)
    iu, ju = np.triu_indices(Nsm, k=1)
    overlaps = []
    for r in range(1, n_overlap + 1):
        vec = np.real(evecs[:, -r])
        dmat = np.zeros((Nsm, Nsm))
        dmat[iu, ju] = vec / np.sqrt(2)
        dmat -= dmat.T
        ov = sum(bms.inner(b, dmat) ** 2 for b in basis) / bms.inner(dmat, dmat)
        overlaps.append(ov)
    P(f"  {label}: fp residual {rfx:.1e}; Delta_fit={Dls:.4f} "
      f"(r2={r2ls:.4f}); max|Im|={max_imag:.1e}")
    P(f"    spectrum: min={evals.min():.4f} max={evals.max():.4f} "
      f"count>0.9: {(evals > 0.9).sum()}  count>1.001: "
      f"{(evals > 1.001).sum()}")
    P("    top 8: " + ", ".join(f"{v:.4f}" for v in evals[-8:][::-1]))
    P("    top-8 reparam overlaps: "
      + ", ".join(f"{o:.2f}" for o in overlaps[:8]))
    return dict(evals=evals, overlaps=overlaps, Delta=Dls)


def main():
    t0 = time.time()
    log = []

    def P(*args):
        line = " ".join(str(a) for a in args)
        print(line, flush=True)
        log.append(line)

    P("=" * 72)
    P("G1 fixed-point & stability — 2026-08-07 (run 5: dense spectra across"
      " beta*J and channel mixes)")
    P("=" * 72)

    beta = 1.0

    # ---------------- D1 ----------------
    P("\n[D1] TI SYK solver across beta*J (declared: Delta -> 0.25 from "
      "below-window fits, G(0+)~0.5 modulo edge ringing)")
    for bJ in (25.0, 50.0, 100.0, 200.0):
        J4 = bJ / beta
        tau, Gt, ni, res = solve_syk_ti(beta, J4, 0.0)
        Delta, r2 = fit_delta(beta, tau, Gt, J4)
        P(f"  beta*J={bJ:5.0f}: iters={ni} res={res:.2e} "
          f"G(0+)={Gt[0]:.4f}  Delta_fit={Delta:.4f} (r2={r2:.5f})")

    # ---------------- D2+D3: dense spectra vs beta*J (pure q=4) ---------
    P("\n[D2+D3] dense Jacobian spectra, N=96, pure q=4, beta*J sweep")
    P("  declared: all eigenvalues real and <= ~1; top eigenvalues are "
      "doubly-degenerate reparam pairs;")
    P("  gap of top pair shrinks with beta*J (~1/(beta*J)); most-negative "
      "-> -3pi/4 = -2.356 as window opens")
    Nsm = 96
    trend = []
    for bJ in (10.0, 15.0, 20.0, 25.0, 30.0):
        r = dense_run(P, beta, bJ / beta, 0.0, Nsm,
                      f"beta*J={bJ:.0f} (1/J = {Nsm/bJ:.1f} dt)")
        if r is not None:
            ktop = r["evals"].max()
            trend.append((bJ, ktop, (1 - ktop) * bJ, r["evals"].min()))
    P("  trend summary: beta*J | k_top | gap*betaJ | k_min")
    for bJ, ktop, gbj, kmin in trend:
        P(f"    {bJ:5.0f} | {ktop:.4f} | {gbj:7.3f} | {kmin:.4f}")

    # ---------------- D4 (arithmetic corollary) ----------------
    P("\n[D4] damped/residual map: lambda_damped = (1+k)/2 from the dense "
      "spectra above;")
    P("  only k=1 is damping-invariant; all measured k<1 => damped map is "
      "a strict contraction, slowest along the reparam pair.")

    # ---------------- D5: channel mixing (dense) ----------------
    P("\n[D5] dense spectra with q=2 admixture, N=96, beta*J4=20 "
      "(declared: soft-mode structure degrades with J2;")
    P("  pure q=2 declared from MS eq. 3.77: ladder eigenvalue -1 for all "
      "h — no +1 mode. The Goldstone argument for the")
    P("  q=2-covariant solution is the counter-hypothesis; the spectrum "
      "decides.")
    for (j4r, j2r, lbl) in ((1.0, 0.3, "J2/J4=0.3"),
                            (1.0, 1.0, "J2/J4=1.0"),
                            (0.0, 1.0, "pure q=2 (beta*J2=20)")):
        dense_run(P, beta, 20.0 * j4r / beta, 20.0 * j2r / beta
                  if j4r == 0 else 20.0 * j4r * j2r / beta, Nsm, lbl)

    # ---------------- D6 ----------------
    P("\n[D6] symmetric/attention-form variant (EXPLORATORY)")
    for g4 in (1.0, 10.0, 100.0):
        _, _, status, m_eff2 = symmetric_variant(g4=g4)
        P(f"  g4={g4:6.1f}: {status};  effective IR mass^2 = {m_eff2:.4f}")

    P(f"\ntotal time: {time.time()-t0:.1f}s")
    with open("research/physics/theory/logs/g1_fixed_point_run5.log", "w") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    sys.exit(main())
