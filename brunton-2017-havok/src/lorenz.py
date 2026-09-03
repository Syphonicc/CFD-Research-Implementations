r"""
Lorenz system, the canonical chaotic testbed used in HAVOK.

    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = x y - beta z

Standard chaotic parameters: sigma = 10, rho = 28, beta = 8/3.

HAVOK uses only the x(t) component -- a single scalar measurement. The point of
the method is that delay embedding reconstructs the full attractor from that one
channel, which is Takens' theorem in action. Nothing here knows the equations.

Known properties used for validation:
  leading Lyapunov exponent   lambda_1 ~ 0.906
  Kaplan-Yorke dimension      D_KY     ~ 2.06
  fixed points                (+/- sqrt(beta(rho-1)), same, rho-1) = (+/-8.485, +/-8.485, 27)
  lobe switching              irregular, no fixed period -- this is the thing
                              HAVOK's intermittent forcing is meant to predict
"""

import numpy as np
from scipy.integrate import solve_ivp

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0


def rhs(t, s, sigma=SIGMA, rho=RHO, beta=BETA):
    x, y, z = s
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def lorenz(T=100.0, dt=0.001, x0=(-8.0, 8.0, 27.0), T_transient=20.0,
           rtol=1e-12, atol=1e-12, sigma=SIGMA, rho=RHO, beta=BETA):
    """
    Integrate the Lorenz system and return the trajectory after transient removal.

    T           recorded duration
    dt          output spacing. HAVOK is sensitive to this: the delay window
                is q*dt, so dt and q trade off against each other.
    T_transient discarded so the trajectory is on the attractor
    rtol, atol  tight by default. Chaotic trajectories diverge, so loose
                tolerances give a *different* trajectory, not merely a less
                accurate one.

    Returns t (n,), S (n, 3)
    """
    if T_transient > 0:
        pre = solve_ivp(rhs, [0, T_transient], list(x0), args=(sigma, rho, beta),
                        rtol=rtol, atol=atol, dense_output=False)
        x0 = pre.y[:, -1]

    t_eval = np.arange(0, T, dt)
    sol = solve_ivp(rhs, [0, T], list(x0), args=(sigma, rho, beta),
                    t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    return sol.t, sol.y.T


def fixed_points(rho=RHO, beta=BETA):
    """The two nontrivial fixed points, i.e. the centres of the two lobes."""
    c = np.sqrt(beta * (rho - 1.0))
    return np.array([[c, c, rho - 1.0], [-c, -c, rho - 1.0]])


def lobe_index(S):
    """
    Which lobe the trajectory is on, by the sign of x. Returns +1 / -1.
    Crude but sufficient: the two lobes are separated by x = 0.
    """
    return np.sign(S[:, 0])


def switching_times(S, t):
    """Times at which the trajectory crosses between lobes."""
    lobe = lobe_index(S)
    idx = np.where(np.diff(lobe) != 0)[0]
    return t[idx]


def lyapunov(T=2000.0, dt=0.01, d0=1e-9, renorm_every=10, seed=0,
             sigma=SIGMA, rho=RHO, beta=BETA):
    """
    Leading Lyapunov exponent by trajectory separation with renormalisation.
    Same method as the KS solver's; literature value for Lorenz is ~0.906.
    """
    rng = np.random.default_rng(seed)
    _, S = lorenz(T=50.0, dt=dt, T_transient=20.0)
    s1 = S[-1].copy()
    pert = rng.standard_normal(3)
    s2 = s1 + d0 * pert / np.linalg.norm(pert)

    n = int(T / (dt * renorm_every))
    span = dt * renorm_every
    total = 0.0
    for _ in range(n):
        a = solve_ivp(rhs, [0, span], s1, args=(sigma, rho, beta),
                      rtol=1e-12, atol=1e-12).y[:, -1]
        b = solve_ivp(rhs, [0, span], s2, args=(sigma, rho, beta),
                      rtol=1e-12, atol=1e-12).y[:, -1]
        d = np.linalg.norm(b - a)
        total += np.log(d / d0)
        s1, s2 = a, a + (b - a) * d0 / d
    return total / (n * span)


if __name__ == "__main__":
    print("Lorenz validation", flush=True)
    t, S = lorenz(T=100.0, dt=0.001)
    print(f"  trajectory: {S.shape}, x range "
          f"[{S[:,0].min():.2f}, {S[:,0].max():.2f}]", flush=True)

    fp = fixed_points()
    print(f"  fixed points: {np.round(fp[0], 3)}, {np.round(fp[1], 3)}", flush=True)

    sw = switching_times(S, t)
    gaps = np.diff(sw)
    print(f"  lobe switches in T=100: {len(sw)}", flush=True)
    print(f"  interval between switches: mean {gaps.mean():.3f}, "
          f"std {gaps.std():.3f}  <- irregular, as expected", flush=True)

    lam = lyapunov(T=500.0)
    print(f"  leading Lyapunov exponent: {lam:.4f}  (literature ~0.906)", flush=True)
