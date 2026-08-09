"""exp-105 — floor-aware estimators for the bilocal exponent.

Two independent methods, specified in notes.md before use:

  M1  conditioned 3-parameter fit of P(dx) = c + b*dx^(-2*Delta)
  M2  double centering G_cen = Pi H Pi, then the census's 2-parameter estimator

M1 shares no machinery with M2 and they rest on different assumptions. Neither
may be reported as Delta_G unless it passes the synthetic gate in validate.py.

Ariel — August 8, 2026.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
KIT = HERE.parent.parent / "replication" / "measure_conformal_heads.py"

_spec = importlib.util.spec_from_file_location("census_kit", KIT)
kit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kit)

FIT_LO, FIT_HI = kit.FIT_LO, kit.FIT_HI
TAIL_LO, TAIL_HI = 400, 511
DELTA_BOUNDS = (0.005, 2.0)
CORR_MAX = 0.99          # c-Delta correlation above this -> non-identifiable

# Validated envelope. M1 recovers Delta to within V1_TOL only inside this
# region (validate.py, first pass). Outside it the fit is refused rather than
# reported, which is what pre-registered criterion V3 demands: an estimator that
# is wrong quietly is worse than one that is wrong loudly.
ENVELOPE_MAX_RATIO = 10.0     # fitted floor / Delta-bearing variation
ENVELOPE_MAX_NOISE = 8.0e-3   # std of log-space fit residuals (6e-3 + margin,
                              # so data whose true noise is 6e-3 is accepted)
# Operating range, established by calibration (validate.py) rather than chosen.
# Accuracy is controlled by the PRODUCT Delta * ratio, which is the physically
# natural combination: a larger Delta decays faster, so less of the
# Delta-bearing variation survives above the floor inside the fit window. At
# Delta <= 0.375 every in-regime cell recovers to within 0.012 even at ratio 10;
# every failure has Delta >= 0.5 together with ratio >= 4.
ENVELOPE_MAX_DELTA_RATIO = 3.5


@dataclass
class Fit:
    delta: float | None
    r2: float | None
    ok: bool                    # method reports its own result as usable
    reason: str = ""
    extra: dict | None = None


def _r2(y: np.ndarray, pred: np.ndarray) -> float | None:
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return None
    return 1.0 - float(((y - pred) ** 2).sum()) / ss_tot


def m1_three_param(profile: np.ndarray,
                   tail: tuple[int, int] = (TAIL_LO, TAIL_HI)) -> Fit:
    """Conditioned 3-parameter fit. Fits log P (relative residuals)."""
    p = np.asarray(profile, dtype=float)
    lags = np.arange(FIT_LO, FIT_HI + 1, dtype=float)
    y = p[FIT_LO:FIT_HI + 1]

    if not np.all(np.isfinite(y)) or y.min() <= 0:
        return Fit(None, None, False, "non-positive or non-finite profile")

    scale = float(y[0])
    yn = y / scale                                   # O(1) amplitudes
    tail_med = float(np.median(p[tail[0]:tail[1] + 1])) / scale
    c0 = max(tail_med, 1e-12)
    b0 = max(yn[0] - c0, 1e-12)

    # Delta init from the 2-parameter fit of the tail-subtracted profile.
    d_init, _ = kit.fit_head(p - float(np.median(p[tail[0]:tail[1] + 1])))
    if d_init is None or not np.isfinite(d_init):
        d_init = 0.25
    d0 = float(np.clip(d_init, *DELTA_BOUNDS))

    log_yn = np.log(yn)

    def resid(theta):
        gamma, beta, delta = theta
        model = np.exp(gamma) + np.exp(beta) * lags ** (-2.0 * delta)
        return np.log(np.clip(model, 1e-300, None)) - log_yn

    try:
        sol = least_squares(
            resid, x0=[np.log(c0), np.log(b0), d0],
            bounds=([-50.0, -50.0, DELTA_BOUNDS[0]], [50.0, 50.0, DELTA_BOUNDS[1]]),
            max_nfev=20000)
    except Exception as exc:                                    # noqa: BLE001
        return Fit(None, None, False, f"least_squares raised: {exc}")

    if not sol.success:
        return Fit(None, None, False, f"no convergence: {sol.status}")

    gamma, beta, delta = sol.x
    model = np.exp(gamma) + np.exp(beta) * lags ** (-2.0 * delta)
    r2 = _r2(log_yn, np.log(model))

    # Identifiability from the Jacobian at the solution.
    corr = None
    try:
        J = sol.jac
        cov = np.linalg.inv(J.T @ J)
        sd = np.sqrt(np.diag(cov))
        if np.all(sd > 0):
            corr = float(abs(cov[0, 2] / (sd[0] * sd[2])))
    except np.linalg.LinAlgError:
        corr = None

    # Where the fit places itself relative to the validated envelope.
    c, b = float(np.exp(gamma)), float(np.exp(beta))
    span = FIT_LO ** (-2.0 * delta) - FIT_HI ** (-2.0 * delta)
    ratio_est = abs(c / (b * span)) if b * span != 0 else np.inf
    noise_est = float(np.std(sol.fun))
    extra = {"c_over_b": c / b, "corr_c_delta": corr,
             "ratio_est": float(ratio_est), "noise_est": noise_est}

    if delta <= DELTA_BOUNDS[0] * 1.001 or delta >= DELTA_BOUNDS[1] * 0.999:
        return Fit(float(delta), r2, False, "Delta at bound", extra)
    if corr is not None and corr > CORR_MAX:
        return Fit(float(delta), r2, False, f"non-identifiable (corr={corr:.4f})", extra)
    if ratio_est > ENVELOPE_MAX_RATIO:
        return Fit(float(delta), r2, False,
                   f"floor ratio {ratio_est:.1f} outside validated envelope", extra)
    if noise_est > ENVELOPE_MAX_NOISE:
        return Fit(float(delta), r2, False,
                   f"residual scatter {noise_est:.2e} outside validated envelope", extra)
    if delta * ratio_est > ENVELOPE_MAX_DELTA_RATIO:
        return Fit(float(delta), r2, False,
                   f"Delta*ratio = {delta * ratio_est:.2f} outside calibrated "
                   f"operating range (Delta={delta:.3f}, ratio={ratio_est:.1f})", extra)

    return Fit(float(delta), r2, True, "", extra)


def double_center(M: np.ndarray, lo: int = 0, hi: int | None = None) -> np.ndarray:
    """Pi M Pi over the index block [lo, hi), zero elsewhere.

    Annihilates a constant and any term depending on one index alone, which is
    exactly terms 1-3 of melonic eq. (2.1).
    """
    A = np.asarray(M, dtype=float)
    n = A.shape[-1]
    hi = n if hi is None else hi
    out = np.zeros_like(A)
    blk = A[..., lo:hi, lo:hi]
    row = blk.mean(axis=-1, keepdims=True)
    col = blk.mean(axis=-2, keepdims=True)
    grand = blk.mean(axis=(-2, -1), keepdims=True)
    out[..., lo:hi, lo:hi] = blk - row - col + grand
    return out


def m2_centered(profile_cen: np.ndarray) -> Fit:
    """Census 2-parameter estimator on a double-centered profile.

    The centered profile changes sign by construction. Fit the positive region
    only and report the sign-crossing lag; a crossing inside the fit window is a
    reported limitation, not something to smooth over.
    """
    p = np.asarray(profile_cen, dtype=float)
    seg = p[FIT_LO:FIT_HI + 1]
    if not np.all(np.isfinite(seg)):
        return Fit(None, None, False, "non-finite centered profile")

    pos = seg > 0
    if pos[0] and (~pos).any():
        cross = int(FIT_LO + np.argmax(~pos))
    else:
        cross = None

    if pos.sum() < 5:
        return Fit(None, None, False, "fewer than 5 positive lags",
                   {"cross_lag": cross})

    hi = cross - 1 if cross is not None else FIT_HI
    if hi - FIT_LO + 1 < 5:
        return Fit(None, None, False, f"positive region too short (cross={cross})",
                   {"cross_lag": cross})

    lags = np.arange(FIT_LO, hi + 1, dtype=float)
    y = p[FIT_LO:hi + 1]
    ok = y > 0
    if ok.sum() < 5:
        return Fit(None, None, False, "too few positive points", {"cross_lag": cross})

    lx, ly = np.log(lags[ok]), np.log(y[ok])
    X = np.column_stack([np.ones_like(lx), lx])
    coef, *_ = np.linalg.lstsq(X, ly, rcond=None)
    pred = X @ coef
    r2 = _r2(ly, pred)
    delta = float(-coef[1] / 2.0)

    inside = cross is not None and FIT_LO < cross <= FIT_HI
    return Fit(delta, r2, not inside,
               "sign crossing inside fit window" if inside else "",
               {"cross_lag": cross, "n_used": int(ok.sum())})


def m0_census(profile: np.ndarray) -> Fit:
    """The published 2-parameter estimator, for V4 (strict-generalization check)."""
    d, r2 = kit.fit_head(np.asarray(profile, dtype=float))
    if d is None:
        return Fit(None, None, False, "census fit returned None")
    return Fit(float(d), None if r2 is None else float(r2), True)
