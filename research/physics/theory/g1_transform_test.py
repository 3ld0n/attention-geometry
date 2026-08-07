"""Debug harness: fermionic Matsubara transforms + spectral derivative.
Validates against explicit sums at small N before the main run."""

import numpy as np

def make_transforms(beta, N):
    """Fermionic transforms on midpoint grid tau_k = (k+1/2) dt.

    Conventions:
      G(iw_n) = int_0^beta dtau e^{+i w_n tau} G(tau),  w_n = pi(2n+1)/beta
      G(tau)  = (1/beta) sum_n e^{-i w_n tau} G(iw_n)

    Index layout: j = 0..N-1 with n_phys[j] = j for j < N/2 else j - N.
    """
    dt = beta / N
    tau = (np.arange(N) + 0.5) * dt
    j = np.arange(N)
    n_phys = np.where(j < N // 2, j, j - N)
    omega = np.pi * (2 * n_phys + 1) / beta

    ph_k = np.exp(1j * np.pi * (np.arange(N) + 0.5) / N)     # e^{i pi tau_k / beta}
    ph_n = np.exp(1j * np.pi * n_phys / N)                    # e^{i pi n / N} residual

    def t_to_w(Gt):
        # Gw[j] = dt * sum_k e^{i w_n tau_k} G[k]
        #       = dt * e^{i pi (2n+1)/(2N)} * sum_k G[k] e^{i pi k/N} e^{2pi i nk/N}
        x = Gt * np.exp(1j * np.pi * np.arange(N) / N)
        s = np.fft.ifft(x) * N          # sum_k x_k e^{+2pi i jk/N}
        return dt * np.exp(1j * np.pi * (2 * n_phys + 1) / (2 * N)) * s

    def w_to_t(Gw):
        # G[k] = (1/beta) sum_j e^{-i w_n tau_k} Gw[j]
        x = Gw * np.exp(-1j * np.pi * (2 * n_phys + 1) / (2 * N))
        s = np.fft.fft(x)               # sum_j x_j e^{-2pi i jk/N}
        return np.real(s * np.exp(-1j * np.pi * np.arange(N) / N)) / beta

    return tau, omega, t_to_w, w_to_t


def test_roundtrip():
    beta, N = 1.7, 64
    tau, omega, t_to_w, w_to_t = make_transforms(beta, N)
    rng = np.random.default_rng(0)
    G = rng.normal(size=N)
    G2 = w_to_t(t_to_w(G))
    print("roundtrip max err:", np.max(np.abs(G - G2)))

def test_explicit():
    beta, N = 1.3, 32
    tau, omega, t_to_w, w_to_t = make_transforms(beta, N)
    rng = np.random.default_rng(1)
    G = rng.normal(size=N)
    dt = beta / N
    Gw_exp = np.array([dt * np.sum(np.exp(1j * w * tau) * G) for w in omega])
    Gw = t_to_w(G)
    print("explicit forward max err:", np.max(np.abs(Gw - Gw_exp)))
    Gt_exp = np.real(np.array(
        [np.sum(np.exp(-1j * omega * t) * Gw_exp) for t in tau])) / beta
    Gt = w_to_t(Gw)
    print("explicit inverse max err:", np.max(np.abs(Gt - Gt_exp)))

def test_free():
    # G_free(iw) = 1/(-iw)  ->  G(tau) = 1/2 on (0, beta)
    beta, N = 1.0, 1 << 12
    tau, omega, t_to_w, w_to_t = make_transforms(beta, N)
    Gt = w_to_t(1.0 / (-1j * omega))
    print("free G(tau) sample [should all be ~0.5]:",
          Gt[0], Gt[N // 4], Gt[N // 2], Gt[-1])

def test_derivative_matrix():
    # antiperiodic spectral derivative on midpoint grid
    beta, N = 1.0, 128
    dt = beta / N
    tau = (np.arange(N) + 0.5) * dt
    j = np.arange(N)
    n_phys = np.where(j < N // 2, j, j - N)
    omega = np.pi * (2 * n_phys + 1) / beta
    tau_, om_, t_to_w, w_to_t = make_transforms(beta, N)
    # D f = w_to_t( -i w * t_to_w(f) )?   d/dtau e^{-i w tau} = -i w e^{-i w tau}
    def D(f):
        return w_to_t(-1j * omega * t_to_w(f))
    # antiperiodic test function: f = sin(pi tau / beta) is periodic-odd...
    # antiperiodic on [0,beta]: f(tau+beta) = -f(tau): e.g. cos(pi tau/beta)
    f = np.cos(np.pi * tau / beta)
    fp = -np.pi / beta * np.sin(np.pi * tau / beta)
    print("derivative max err (cos):", np.max(np.abs(D(f) - fp)))
    f2 = np.sin(3 * np.pi * tau / beta)
    fp2 = 3 * np.pi / beta * np.cos(3 * np.pi * tau / beta)
    print("derivative max err (sin3):", np.max(np.abs(D(f2) - fp2)))

if __name__ == "__main__":
    test_roundtrip()
    test_explicit()
    test_free()
    test_derivative_matrix()
