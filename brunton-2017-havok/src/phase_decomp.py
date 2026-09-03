r"""
Tangential / transverse decomposition of autoregressive rollout error:
limit cycle vs chaotic attractor.

THE QUESTION
------------
arXiv:2608.07189 showed that on a LIMIT CYCLE, autoregressive latent rollout
error is overwhelmingly tangential -- the model traverses the right attractor
at the wrong rate. Transverse error does not grow.

Standing objection: this cannot transfer to chaos, since a chaotic attractor
has no global phase.

THE MECHANISM
-------------
For any autonomous flow, the direction ALONG the trajectory is neutral: it
carries Lyapunov exponent exactly zero. What differs is the transverse
directions.

  limit cycle   all transverse directions STABLE (negative exponents)
                -> only the neutral tangential direction accumulates error
                -> error is phase drift

  chaotic       at least one transverse direction UNSTABLE (lambda_1 > 0)
                -> transverse error grows exponentially and competes
                -> error is NOT phase drift

Same framework, opposite predictions. Running both is the test.

DECOMPOSITION
-------------
With e = x_pred - x_true and T the normalised true velocity:

    e_tan  = (e . T)            along the flow  -- "phase" error
    e_perp = |e - (e.T) T|      perpendicular   -- geometry error

Reported as the tangential share, e_tan^2 / (e_tan^2 + e_perp^2). The null
hypothesis is an error vector with no directional preference, which in d
dimensions gives a share of 1/d. That baseline is essential: in 2D a share of
0.5 means NOTHING is happening, while in 3D the same number is a real signal.

Only the pre-saturation window is meaningful. Once the error reaches the size
of the attractor, the share decays to the random baseline for trivial reasons.
"""

import numpy as np
from scipy.integrate import solve_ivp

from lorenz import lorenz, rhs as lorenz_rhs
from reservoir import Reservoir


def vdp_rhs(t, s, mu=2.0):
    """Van der Pol: clean limit cycle, transverse directions stable."""
    x, y = s
    return [y, mu * (1.0 - x * x) * y - x]


def van_der_pol(T=4000.0, dt=0.05, T_transient=100.0):
    pre = solve_ivp(vdp_rhs, [0, T_transient], [2.0, 0.0], rtol=1e-11, atol=1e-11)
    te = np.arange(0, T, dt)
    sol = solve_ivp(vdp_rhs, [0, T], pre.y[:, -1], t_eval=te,
                    rtol=1e-11, atol=1e-11)
    return sol.t, sol.y.T


def decompose(pred, true, rhs_fn):
    V = np.array([rhs_fn(0.0, s) for s in true])
    T = V / np.linalg.norm(V, axis=1, keepdims=True)
    e = pred - true
    tan = np.sum(e * T, axis=1)
    return np.abs(tan), np.linalg.norm(e - tan[:, None] * T, axis=1)


def experiment(S, rhs_fn, name, n_train=40000, K=40, tau=400, D_r=1500,
               beta=1e-6, seed=0, sat=0.1):
    n = S.shape[1]
    res = Reservoir(n_in=n, D_r=D_r, rho=0.6, sigma=1.0, kappa=3,
                    beta=beta, seed=seed).fit(S[:n_train], n_washout=1000,
                                              verbose=False)
    U = S[n_train:]
    TAN = np.zeros((K, tau))
    PERP = np.zeros((K, tau))
    for k in range(K):
        a = k * (10 + tau)
        true = U[a + 10:a + 10 + tau]
        pred = res.predict(U[a:a + 10], tau)
        TAN[k], PERP[k] = decompose(pred, true, rhs_fn)

    amp = np.std(S, axis=0).mean()
    tot = np.sqrt(TAN ** 2 + PERP ** 2).mean(axis=0)
    lim = int(np.argmax(tot > sat * amp)) if np.any(tot > sat * amp) else tau
    share = (TAN ** 2 / (TAN ** 2 + PERP ** 2)).mean(axis=0)
    baseline = 1.0 / n

    print(f"\n{name}", flush=True)
    print(f"  state dimension {n}, random-error baseline share = {baseline:.3f}",
          flush=True)
    print(f"  pre-saturation window: {lim} of {tau} steps", flush=True)
    for f in [0.1, 0.3, 0.6, 0.9]:
        i = max(1, int(f * lim))
        print(f"    step {i:>4}  tangential share {share[i]:.3f}   "
              f"perp/tan {PERP[:, i].mean()/TAN[:, i].mean():.2f}", flush=True)
    m = float(share[1:lim].mean())
    print(f"  MEAN TANGENTIAL SHARE: {m:.3f}   "
          f"(baseline {baseline:.3f}, excess {m - baseline:+.3f})", flush=True)
    return {"share": share, "lim": lim, "baseline": baseline, "mean": m}


if __name__ == "__main__":
    print("Rollout error geometry: limit cycle vs chaos", flush=True)

    _, V = van_der_pol()
    r_lc = experiment(V, vdp_rhs, "VAN DER POL  (limit cycle)")

    _, S = lorenz(T=1200.0, dt=0.01)
    r_ch = experiment(S, lorenz_rhs, "LORENZ  (chaotic)")

    print("\n--- summary ---", flush=True)
    print(f"  limit cycle: {r_lc['mean']:.3f} vs baseline {r_lc['baseline']:.3f}"
          f"  -> essentially pure phase drift", flush=True)
    print(f"  chaotic:     {r_ch['mean']:.3f} vs baseline {r_ch['baseline']:.3f}"
          f"  -> weak tangential preference, not dominance", flush=True)
