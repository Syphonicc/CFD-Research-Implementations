r"""
Evaluation for chaotic forecasting.

On a chaotic attractor absolute error is meaningless without a timescale: every
model fails eventually, by construction. Results are therefore reported as
valid prediction time measured in Lyapunov times, t * lambda_1.

Pathak et al. protocol:
  - K = 30 non-overlapping prediction intervals of length tau = 1000
  - before each, reservoir state reset to r = 0 and driven with true data for
    eps = 10 steps
  - RMSE averaged over the K intervals
  - repeated for 10 random reservoir realisations

The RMSE threshold defining "valid" is NOT specified in the paper. The
convention used here is normalised RMSE = 0.5, which is standard in the
follow-on literature (Vlachas et al. 2020 among others). Any reported valid
prediction time must state its threshold.
"""

import numpy as np


def nrmse_curve(pred, true, norm=None):
    """
    Normalised RMSE as a function of lead time.

    pred, true : (n_steps, Q)
    norm       : scalar normalisation. Defaults to the RMS of `true` over the
                 whole evaluation window, i.e. error relative to the natural
                 variability of the attractor.
    """
    if norm is None:
        norm = np.sqrt(np.mean(true ** 2))
    err = np.sqrt(np.mean((pred - true) ** 2, axis=1))
    return err / norm


def valid_time(pred, true, dt, lam, threshold=0.5, norm=None):
    """
    Lead time at which the NRMSE first exceeds `threshold`, in Lyapunov times.
    Returns the full window length if the threshold is never crossed.
    """
    e = nrmse_curve(pred, true, norm=norm)
    idx = np.argmax(e > threshold) if np.any(e > threshold) else len(e)
    return idx * dt * lam


def evaluate(res, U_test, dt, lam, K=30, tau=1000, eps=10,
             threshold=0.5, verbose=True):
    """
    Run the K-interval protocol on a trained reservoir.

    Returns dict with per-interval valid times and the mean NRMSE curve.
    """
    need = K * (eps + tau)
    if len(U_test) < need:
        raise ValueError(f"need {need} test steps, have {len(U_test)}")

    norm = np.sqrt(np.mean(U_test ** 2))
    vts, curves = [], []

    for k in range(K):
        a = k * (eps + tau)
        seed = U_test[a:a + eps]
        true = U_test[a + eps:a + eps + tau]
        pred = res.predict(seed, tau)

        vts.append(valid_time(pred, true, dt, lam,
                              threshold=threshold, norm=norm))
        curves.append(nrmse_curve(pred, true, norm=norm))

        if verbose and (k + 1) % 10 == 0:
            print(f"    interval {k+1}/{K}: "
                  f"running mean VPT = {np.mean(vts):.2f} Lyap", flush=True)

    vts = np.asarray(vts)
    return {
        "valid_times": vts,
        "vpt_mean": float(vts.mean()),
        "vpt_std": float(vts.std()),
        "nrmse_mean": np.mean(curves, axis=0),
        "lyapunov_axis": np.arange(tau) * dt * lam,
        "threshold": threshold,
    }
