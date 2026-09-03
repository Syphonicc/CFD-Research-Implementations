r"""
Dynamic Mode Decomposition.

THE SETUP
---------
Given snapshots x_1 ... x_m (columns, each of dimension n), form

    X  = [x_1 ... x_{m-1}]        X' = [x_2 ... x_m]

and *assume* a single linear operator advances every snapshot:

    X' = A X

A is n x n, which for a flow field is enormous, so it is never formed. Instead
project onto the leading POD modes and eigendecompose there.

RELATION TO POD
---------------
Step 1 of DMD *is* POD. The SVD X = U S V* gives:
    U  columns: POD spatial modes, orthonormal in space
    V  columns: POD temporal coefficients, orthonormal in time
    S           singular values, energy ranking

POD stops there. It is a static decomposition: shuffling the snapshot order
leaves U and S unchanged. It knows nothing about dynamics.

DMD continues: it regresses X' onto X in that reduced basis, producing a small
operator whose eigenvalues carry growth rate and frequency. Time ordering is
essential. Each DMD mode obeys the separable ansatz

    x(t) ~ sum_j  phi_j * exp(omega_j t) * b_j

i.e. one spatial structure multiplied by ONE complex exponential. A POD mode's
temporal coefficient is an arbitrary function of time and generally mixes many
frequencies; a DMD mode's is a pure exponential by construction. That is the
whole difference, and it is why DMD modes are not orthogonal.

EXACT vs PROJECTED
------------------
Projected DMD (Schmid 2010):  Phi = U W
Exact DMD    (Tu et al 2014): Phi = X' V S^-1 W

Exact modes are true eigenvectors of A; projected modes are their projection
onto the POD subspace. They agree when the data are exactly rank-r. Exact is
the default here.
"""

import numpy as np


def dmd(X, dt=1.0, r=None, exact=True, tol=1e-10):
    """
    Standard DMD.

    X      (n, m) snapshot matrix, columns are snapshots in time order
    dt     time between snapshots
    r      truncation rank. None keeps all modes above `tol`.
           NOTE: for Hankel/chaotic data an explicit rank is required --
           energy-based truncation silently keeps spurious modes.
    exact  exact DMD modes (Tu 2014) vs projected (Schmid 2010)

    Returns dict with:
      lam     discrete eigenvalues of A
      omega   continuous eigenvalues, log(lam)/dt  (real = growth, imag = freq)
      Phi     (n, r) DMD modes
      b       mode amplitudes fit to the first snapshot
      freq    ordinary frequency, imag(omega)/(2 pi)
      growth  real(omega)
      U,S,V   the POD factors, returned so the POD/DMD relation is inspectable
    """
    X1 = X[:, :-1]
    X2 = X[:, 1:]

    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    if r is None:
        r = int(np.sum(S > tol * S[0]))
    r = min(r, len(S))
    U, S, V = U[:, :r], S[:r], Vh[:r].conj().T

    # A projected onto the POD basis
    At = U.conj().T @ X2 @ V @ np.diag(1.0 / S)

    lam, W = np.linalg.eig(At)

    if exact:
        Phi = X2 @ V @ np.diag(1.0 / S) @ W
    else:
        Phi = U @ W

    omega = np.log(lam.astype(complex)) / dt
    b = np.linalg.lstsq(Phi, X[:, 0], rcond=None)[0]

    return {
        "lam": lam, "omega": omega, "Phi": Phi, "b": b,
        "freq": omega.imag / (2 * np.pi), "growth": omega.real,
        "U": U, "S": S, "V": V, "r": r,
    }


def dmd_reconstruct(res, n_steps, dt):
    """Rebuild the field from modes: x(t) = sum_j phi_j exp(omega_j t) b_j."""
    t = np.arange(n_steps) * dt
    T = np.exp(np.outer(res["omega"], t))          # (r, n_steps)
    return res["Phi"] @ (res["b"][:, None] * T)


def hankel(x, q):
    """
    Hankel (delay-embedding) matrix with q delays.

    x : (m,) scalar series or (n, m) multivariate series
    q : number of delays (rows of blocks)

    Returns (n*q, m-q+1). Column k stacks q consecutive snapshots, so the
    augmented state carries a WINDOW of history rather than an instant.

    This is Takens' embedding in practical form. Plain DMD assumes x(t)
    determines x(t+dt); when the true system has dynamics the snapshot cannot
    see -- a slow modulation, a hidden variable -- that assumption fails and
    the fitted operator chases a moving target. Stacking delays puts the
    hidden structure INSIDE the state.

    The window length q*dt must span the slowest timescale you need the
    operator to see. Below that threshold the augmented state still cannot
    resolve the modulation; above it, the operator becomes genuinely
    time-invariant over the embedding. The transition is a threshold, not a
    trend -- which is why delay sweeps show reversals rather than smooth
    improvement.
    """
    x = np.atleast_2d(x)
    if x.shape[0] > x.shape[1]:
        x = x.T
    n, m = x.shape
    cols = m - q + 1
    if cols < 1:
        raise ValueError(f"q={q} too large for series of length {m}")
    H = np.empty((n * q, cols))
    for i in range(q):
        H[i * n:(i + 1) * n, :] = x[:, i:i + cols]
    return H


def hankel_dmd(x, q, dt=1.0, r=None, exact=True):
    """DMD applied to the delay-embedded series. `r` should be given explicitly."""
    H = hankel(x, q)
    out = dmd(H, dt=dt, r=r, exact=exact)
    out["H"] = H
    out["q"] = q
    return out
