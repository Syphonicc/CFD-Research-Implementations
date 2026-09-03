r"""
HAVOK: Hankel Alternative View Of Koopman.
Brunton et al., Nature Communications 8, 19 (2017).

THE IDEA
--------
Take a single scalar measurement x(t) from a chaotic system. Build the Hankel
matrix of delays and take its SVD:

    H = U S V*

The columns of V are the "eigen-time-delay coordinates" v_1 ... v_q. HAVOK's
claim is that the leading r-1 of them obey an approximately LINEAR system,
driven by the r-th:

    d/dt [v_1 ... v_{r-1}]^T  =  A [v_1 ... v_{r-1}]^T  +  B v_r

So chaos is not modelled as a linear system (impossible -- continuous spectrum)
but as a linear system plus an intermittent forcing term. The forcing v_r is
near zero most of the time and bursts before lobe switches.

WHY IT SIDESTEPS THE CONTINUOUS-SPECTRUM PROBLEM
------------------------------------------------
Plain DMD tries to represent the whole attractor with finitely many eigenvalues,
which cannot work. HAVOK admits that up front: it linearises only the part of
the dynamics that IS linear, and quarantines the rest into a forcing signal that
it measures rather than models. The non-Gaussian statistics of that forcing are
the paper's main empirical result.

WHAT THE DELAY WINDOW CONTROLS
------------------------------
The window is q*dt. The Lorenz attractor switches lobes at irregular intervals
averaging ~1.7 time units. A window shorter than that cannot contain a switching
event, so no operator fitted on it can represent switching -- regardless of rank.

This is directly testable and `sweep_window()` below does it.
"""

import numpy as np
from dmd import hankel


def havok(x, q, dt, r, deriv="central"):
    """
    Fit the HAVOK model to a scalar series.

    x  (m,)  scalar measurement
    q        number of delays; the window is q*dt
    dt       sample spacing
    r        truncation rank. v_1..v_{r-1} are the linear coordinates,
             v_r is the forcing. MUST be given explicitly -- see notes.

    Returns dict with V (delay coordinates), A, B, forcing, and the
    reconstruction residual.
    """
    H = hankel(x, q)
    U, S, Vh = np.linalg.svd(H, full_matrices=False)
    V = Vh.conj().T                      # (cols, q), columns are v_1..v_q
    V = V[:, :r]

    # central differences for dV/dt, dropping the endpoints
    if deriv == "central":
        dV = (V[2:] - V[:-2]) / (2 * dt)
        Vm = V[1:-1]
    else:
        dV = (V[1:] - V[:-1]) / dt
        Vm = V[:-1]

    # regress d/dt of the first r-1 coordinates on all r coordinates
    Xi = np.linalg.lstsq(Vm, dV[:, :r - 1], rcond=None)[0]   # (r, r-1)
    A = Xi[:r - 1, :].T                                       # (r-1, r-1)
    B = Xi[r - 1, :].reshape(-1, 1)                           # (r-1, 1)

    pred = Vm @ Xi
    resid = np.linalg.norm(pred - dV[:, :r - 1]) / np.linalg.norm(dV[:, :r - 1])

    return {
        "U": U, "S": S, "V": V, "A": A, "B": B,
        "forcing": V[:, r - 1], "Vm": Vm, "dV": dV,
        "residual": float(resid), "q": q, "r": r, "dt": dt,
    }


def antisymmetry(A):
    """
    HAVOK predicts A is nearly antisymmetric (energy-preserving apart from the
    forcing). Returns ||A + A^T|| / ||A||; small means strongly antisymmetric.
    """
    return float(np.linalg.norm(A + A.T) / np.linalg.norm(A))


def forcing_kurtosis(f):
    """
    Excess kurtosis of the forcing. Gaussian is 0. The paper's central
    empirical claim is heavy tails, i.e. strongly positive.
    """
    f = (f - f.mean()) / f.std()
    return float(np.mean(f ** 4) - 3.0)


def sweep_window(x, dt, q_list, r=15, switch_interval=None, verbose=True):
    """
    The experiment this folder exists for: vary the delay window and watch
    what changes.

    Reports, per window length:
      residual     how well the linear-plus-forcing model fits
      antisym      how antisymmetric A is (HAVOK's structural prediction)
      kurtosis     heavy-tailedness of the forcing (the empirical claim)
    """
    rows = []
    if verbose:
        print(f"{'q':>6} {'window':>8} {'resid':>9} {'antisym':>9} {'kurtosis':>10}",
              flush=True)
        if switch_interval:
            print(f"       (mean lobe-switch interval = {switch_interval:.2f} t.u.)",
                  flush=True)
    for q in q_list:
        try:
            h = havok(x, q, dt, r)
        except Exception as e:
            if verbose:
                print(f"{q:>6} {'--':>8}  failed: {e}", flush=True)
            continue
        row = {
            "q": q, "window": q * dt, "residual": h["residual"],
            "antisym": antisymmetry(h["A"]),
            "kurtosis": forcing_kurtosis(h["forcing"]),
        }
        rows.append(row)
        if verbose:
            print(f"{q:>6} {row['window']:>8.2f} {row['residual']:>9.4f} "
                  f"{row['antisym']:>9.4f} {row['kurtosis']:>10.2f}", flush=True)
    return rows
