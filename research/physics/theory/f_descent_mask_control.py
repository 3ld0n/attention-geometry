import numpy as np, sys
sys.path.insert(0, '.')
from f_descent_check import make_M, make_WV, logsumexp

def run(mask_kind, score, content, beta=2.0, N=64, d=32, eta=0.1, T=300, seed=0, tol=1e-9):
    rng = np.random.default_rng(seed); M = make_M(score, d, rng); WV = make_WV(content, M, d, rng)
    sigma = rng.standard_normal((N, d)); sigma /= np.linalg.norm(sigma, axis=1, keepdims=True)
    if mask_kind == 'causal':
        mask = np.triu(np.ones((N, N), bool), 1)
    elif mask_kind == 'sym_sparse':
        U = rng.random((N, N)) < 0.5; mask = np.triu(U, 1); mask = mask | mask.T
    elif mask_kind == 'asym_sparse':
        mask = rng.random((N, N)) < 0.5; np.fill_diagonal(mask, False)
    else:
        mask = None
    Fh = []
    for t in range(T + 1):
        z = beta * (sigma @ M @ sigma.T) / np.sqrt(d)
        if mask is not None: z = np.where(mask, -np.inf, z)
        lse = logsumexp(z, 1); A = np.exp(z - lse[:, None]); Fh.append(-(1 / beta) * lse.sum())
        if t == T: break
        sigma = sigma + eta * (A @ (sigma @ WV.T)); sigma /= np.linalg.norm(sigma, axis=1, keepdims=True)
    dF = np.diff(Fh); ok = (dF <= tol * np.abs(np.array(Fh[:-1])))
    return ok.mean(), Fh[0], Fh[-1]

for score, content in [('sym_psd', 'identity'), ('sym_psd', 'aligned'), ('asym', 'aligned')]:
    for mk in [None, 'sym_sparse', 'asym_sparse', 'causal']:
        r = [run(mk, score, content, seed=s) for s in range(12)]
        print(f"{score:8s} {content:8s} mask={str(mk):11s} frac nonincr steps={np.mean([x[0] for x in r]):.3f}  "
              f"seeds mono={np.mean([x[0] == 1.0 for x in r]):.2f}  F {np.mean([x[1] for x in r]):.1f}->{np.mean([x[2] for x in r]):.1f}")
