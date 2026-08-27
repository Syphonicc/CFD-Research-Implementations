r"""
Kuramoto-Sivashinsky solver, ETDRK4 (Kassam & Trefethen 2005).

    u_t + u*u_x + u_xx + u_xxxx = 0,   periodic on [0, Lx)

Fourier form (u_hat = FFT(u)):

    d(u_hat)/dt = (k^2 - k^4)*u_hat  -  (i*k/2)*FFT(u^2)
                   \_________/          \______________/
                    L (linear)            N (nonlinear)

The linear part is violently stiff: the k^4 term gives the highest mode an
eigenvalue ~ -k_max^4, so explicit RK4 would need dt ~ k_max^-4. ETDRK4
integrates L exactly via the matrix exponential and treats only N explicitly,
so dt is set by accuracy rather than stability. The phi-functions are evaluated
by contour integration to avoid cancellation as h*L -> 0.

IMPLEMENTATION NOTE (this cost an hour to find):
Using the complex fft, roundoff breaks the Hermitian symmetry of u_hat at the
1e-16 level. That asymmetry is then amplified by the *physical* linear
instability at rate sigma_max = 0.25, so the spurious imaginary part of u grows
like exp(0.25*t) and swamps the real solution around t ~ 350 -- long after any
short validation run would notice. Using rfft/irfft makes a real field
structurally guaranteed, so the mode cannot exist. Do not "fix" this by
discarding the imaginary part each step; that hides the growth without removing
the forcing.
"""

import numpy as np


def ks_solve(Lx=22.0, N=64, dt=0.25, T=10000.0, T_transient=1000.0,
             u0=None, seed=0, M_contour=32, verbose=True):
    """
    Integrate KS and return snapshots after transient removal.

    Lx          domain length; number of unstable modes ~ Lx/(2*pi)
    N           grid points (must be even)
    dt          timestep
    T           integration time recorded, after the transient
    T_transient time discarded so the trajectory lands on the attractor

    Returns x (N,), t (n_save,), U (n_save, N)
    """
    assert N % 2 == 0, "N must be even"

    x = Lx * np.arange(N) / N
    k = 2.0 * np.pi * np.fft.rfftfreq(N, d=Lx / N)      # (N//2+1,)

    if u0 is None:
        rng = np.random.default_rng(seed)
        u0 = (np.cos(2 * np.pi * x / Lx) * (1.0 + np.sin(2 * np.pi * x / Lx))
              + 0.01 * rng.standard_normal(N))
    u0 = np.asarray(u0, dtype=float)

    L_op = k**2 - k**4                  # sigma(k), positive for |k| < 1
    g = -0.5j * k                       # u*u_x = 0.5*d(u^2)/dx
    g[-1] = 0.0                         # kill Nyquist for odd derivatives

    # --- ETDRK4 coefficients by contour integration ---
    E = np.exp(dt * L_op)
    E2 = np.exp(dt * L_op / 2.0)
    r = np.exp(1j * np.pi * (np.arange(1, M_contour + 1) - 0.5) / M_contour)
    LR = dt * L_op[:, None] + r[None, :]

    Q  = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean(
        (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean(
        (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean(
        (-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1))

    def nl(vhat):
        u = np.fft.irfft(vhat, n=N)     # real by construction
        return g * np.fft.rfft(u * u)

    def step(v):
        Nv = nl(v)
        a  = E2 * v + Q * Nv
        Na = nl(a)
        b  = E2 * v + Q * Na
        Nb = nl(b)
        c  = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nl(c)
        return E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3

    v = np.fft.rfft(u0)
    n_tr = int(round(T_transient / dt))
    n_save = int(round(T / dt))

    if verbose:
        print(f"KS: Lx={Lx}, N={N}, dt={dt}", flush=True)
        print(f"  unstable modes ~ Lx/(2pi) = {Lx/(2*np.pi):.2f}", flush=True)
        print(f"  transient: {n_tr} steps", flush=True)

    for i in range(n_tr):
        v = step(v)
        if not np.all(np.isfinite(v)):
            raise RuntimeError(f"blew up in transient at step {i}")

    U = np.empty((n_save, N))
    t = np.arange(n_save) * dt
    for i in range(n_save):
        U[i] = np.fft.irfft(v, n=N)
        v = step(v)
        if not np.all(np.isfinite(v)):
            raise RuntimeError(f"blew up in recording at step {i}")

    if verbose:
        print(f"  done: U{U.shape}, range [{U.min():.3f}, {U.max():.3f}]",
              flush=True)
    return x, t, U


if __name__ == "__main__":
    x, t, U = ks_solve(Lx=22.0, N=64, dt=0.25, T=5000.0, T_transient=1000.0)
    np.savez_compressed("ks_L22_N64.npz", x=x, t=t, U=U)
    print("saved ks_L22_N64.npz", flush=True)
