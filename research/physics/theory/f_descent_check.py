"""
f_descent_check.py — Is the attending closure a descent on the horizon's
variational free energy F?

Theory computation, G1-class (no trained model, no instrument). Companion to
attending_system_theory_draft.md §4 (A5) and §7.1 (ii)-(iii).

The system: N loci with states sigma_i in R^d. Bilinear score
s_ia = sigma_i^T M sigma_a / sqrt(d). Routing A = softmax_a(beta * s_ia)
(row-normalized, A6 intact: A is recomputed from the state every step).
Content v(sigma_a) = W_V sigma_a. Additive update
sigma_i <- sigma_i + eta * sum_a A_ia v(sigma_a), optionally followed by
renormalization of each state to unit norm (controls the trivial norm-growth
descent).

F = sum_i [ -sum_a A_ia s_ia - beta^{-1} H(A_i.) ]
  = -(1/beta) sum_i log sum_a exp(beta s_ia)          (Gibbs variational identity)

Question: is F_{t+1} <= F_t along the closure? Variants:
  score:   'sym_psd'  M = W^T W               (reciprocal, positive)
           'sym_ind'  M = (B + B^T)/2          (reciprocal, indefinite)
           'asym'     M = W_Q^T W_K            (two independent reads; q = 4 structure)
  content: 'identity' W_V = I ; 'random' W_V random ; 'aligned' W_V = M (content read = score read)
  renorm:  True / False
  mask:    None / 'causal'

Outputs results JSON with, per variant: fraction of steps with dF <= tol,
fraction of seeds with monotone F over the whole run, final F, and the same
for the mean row entropy H (the entropy-only candidate) for comparison.
"""
from __future__ import annotations
import json, itertools, sys, time
import numpy as np

def logsumexp(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)

def make_M(kind, d, rng):
    if kind == 'sym_psd':
        W = rng.standard_normal((d, d)) / np.sqrt(d)
        return W.T @ W
    if kind == 'sym_ind':
        B = rng.standard_normal((d, d)) / np.sqrt(d)
        return (B + B.T) / 2
    if kind == 'asym':
        WQ = rng.standard_normal((d, d)) / np.sqrt(d)
        WK = rng.standard_normal((d, d)) / np.sqrt(d)
        return WQ.T @ WK
    raise ValueError(kind)

def make_WV(kind, M, d, rng):
    if kind == 'identity':
        return np.eye(d)
    if kind == 'random':
        return rng.standard_normal((d, d)) / np.sqrt(d)
    if kind == 'aligned':
        return M / np.sqrt(d)
    raise ValueError(kind)

def run(N, d, beta, eta, T, score, content, renorm, mask, seed, tol=1e-9):
    rng = np.random.default_rng(seed)
    M = make_M(score, d, rng)
    WV = make_WV(content, M, d, rng)
    sigma = rng.standard_normal((N, d))
    sigma /= np.linalg.norm(sigma, axis=1, keepdims=True)
    maskmat = None
    if mask == 'causal':
        maskmat = np.triu(np.ones((N, N), dtype=bool), k=1)  # forbid a > i
    F_hist, H_hist, norm_hist = [], [], []
    for t in range(T + 1):
        s = (sigma @ M @ sigma.T) / np.sqrt(d)
        z = beta * s
        if maskmat is not None:
            z = np.where(maskmat, -np.inf, z)
        lse = logsumexp(z, axis=1)
        A = np.exp(z - lse[:, None])
        F = -(1.0 / beta) * lse.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            H = -(np.where(A > 0, A * np.log(A), 0.0)).sum(axis=1).mean()
        F_hist.append(F); H_hist.append(H)
        norm_hist.append(np.linalg.norm(sigma, axis=1).mean())
        if t == T:
            break
        sigma = sigma + eta * (A @ (sigma @ WV.T))
        if renorm:
            sigma /= np.linalg.norm(sigma, axis=1, keepdims=True)
    F_hist = np.array(F_hist); H_hist = np.array(H_hist)
    dF = np.diff(F_hist); dH = np.diff(H_hist)
    return {
        'frac_steps_F_nonincreasing': float((dF <= tol * np.abs(F_hist[:-1]).clip(min=1)).mean()),
        'F_monotone_whole_run': bool((dF <= tol * np.abs(F_hist[:-1]).clip(min=1)).all()),
        'n_F_increases': int((dF > tol * np.abs(F_hist[:-1]).clip(min=1)).sum()),
        'max_F_increase': float(dF.max()),
        'F_first': float(F_hist[0]), 'F_last': float(F_hist[-1]),
        'frac_steps_H_nonincreasing': float((dH <= 1e-12).mean()),
        'H_monotone_whole_run': bool((dH <= 1e-12).all()),
        'H_first': float(H_hist[0]), 'H_last': float(H_hist[-1]),
        'norm_first': float(norm_hist[0]), 'norm_last': float(norm_hist[-1]),
        'F_hist_sub': [float(x) for x in F_hist[:: max(1, T // 20)]],
    }

def main():
    N, d, eta, T, seeds = 64, 32, 0.1, 300, 12
    betas = [0.5, 1.0, 2.0, 4.0, 8.0]
    scores = ['sym_psd', 'sym_ind', 'asym']
    contents = ['identity', 'random', 'aligned']
    renorms = [True, False]
    masks = [None, 'causal']
    out = {'params': dict(N=N, d=d, eta=eta, T=T, seeds=seeds, betas=betas), 'variants': []}
    t0 = time.time()
    for score, content, renorm, mask, beta in itertools.product(scores, contents, renorms, masks, betas):
        rs = [run(N, d, beta, eta, T, score, content, renorm, mask, seed) for seed in range(seeds)]
        agg = {
            'score': score, 'content': content, 'renorm': renorm, 'mask': mask, 'beta': beta,
            'frac_steps_F_nonincreasing_mean': float(np.mean([r['frac_steps_F_nonincreasing'] for r in rs])),
            'frac_seeds_F_monotone': float(np.mean([r['F_monotone_whole_run'] for r in rs])),
            'mean_n_F_increases': float(np.mean([r['n_F_increases'] for r in rs])),
            'max_F_increase_over_seeds': float(np.max([r['max_F_increase'] for r in rs])),
            'F_first_mean': float(np.mean([r['F_first'] for r in rs])),
            'F_last_mean': float(np.mean([r['F_last'] for r in rs])),
            'frac_seeds_H_monotone': float(np.mean([r['H_monotone_whole_run'] for r in rs])),
            'frac_steps_H_nonincreasing_mean': float(np.mean([r['frac_steps_H_nonincreasing'] for r in rs])),
            'H_first_mean': float(np.mean([r['H_first'] for r in rs])),
            'H_last_mean': float(np.mean([r['H_last'] for r in rs])),
            'norm_last_mean': float(np.mean([r['norm_last'] for r in rs])),
            'example_F_hist': rs[0]['F_hist_sub'],
        }
        out['variants'].append(agg)
        print(f"{score:8s} {content:8s} renorm={str(renorm):5s} mask={str(mask):6s} beta={beta:4.1f} | "
              f"F nonincr steps={agg['frac_steps_F_nonincreasing_mean']:.3f} seeds mono={agg['frac_seeds_F_monotone']:.2f} "
              f"| H seeds mono={agg['frac_seeds_H_monotone']:.2f} | F {agg['F_first_mean']:.2f}->{agg['F_last_mean']:.2f} norm={agg['norm_last_mean']:.2f}",
              flush=True)
    out['elapsed_s'] = time.time() - t0
    path = sys.argv[1] if len(sys.argv) > 1 else 'research/physics/theory/logs/f_descent_check_results.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path, f'({out["elapsed_s"]:.0f}s)')

if __name__ == '__main__':
    main()
