r"""
Echo-state reservoir computer, following Pathak et al., PRL 120, 024102 (2018).

Reservoir update (paper Eq. 1 form; no leak rate, no bias):

    r(t+dt) = tanh( A @ r(t) + W_in @ u(t) )

Output layer:

    v(t) = P1 @ r(t) + P2 @ r(t)**2

The quadratic term is not optional. tanh is odd, so with P2 = 0 the reservoir
has a symmetry: if r(t) is an attracting orbit with output v(t), then -r(t) is
also attracting with output -v(t). KS is not invariant under y -> -y, so that
symmetry is wrong for this system and the linear-only readout fails. The paper
states this explicitly (their ref. [16]).

Implementation: the readout is fit as a single ridge regression on the stacked
feature vector s = [r ; r**2], of dimension 2*D_r.

    P = (S Y^T) (S S^T + beta I)^-1

S S^T is (2 D_r) x (2 D_r) and does NOT depend on training length, so it is
accumulated in chunks. At the paper's D_r = 5000 that is a 10000 x 10000 matrix
(~800 MB float64); the full feature matrix S at T = 70,000 would be 5.6 GB and
is never formed.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs


class Reservoir:
    def __init__(self, n_in, D_r=5000, rho=0.6, sigma=1.0, kappa=3,
                 beta=1e-6, seed=0):
        """
        n_in   input dimension (Q for the single-reservoir case)
        D_r    reservoir nodes
        rho    spectral radius of A
        sigma  W_in entries drawn uniform on [-sigma, sigma]
        kappa  average degree of A (directed Erdos-Renyi)
        beta   ridge regularisation. NOT given in the paper's main text.
        """
        self.n_in, self.D_r, self.beta = n_in, D_r, beta
        self.rho, self.sigma, self.kappa, self.seed = rho, sigma, kappa, seed
        rng = np.random.default_rng(seed)

        # --- adjacency: sparse directed Erdos-Renyi, average degree kappa ---
        density = kappa / D_r
        A = sparse.random(D_r, D_r, density=density, random_state=seed,
                          data_rvs=lambda n: rng.uniform(-1.0, 1.0, n),
                          format='csr')
        # rescale to the requested spectral radius
        k_eig = min(6, D_r - 2)
        try:
            lam = np.abs(eigs(A, k=k_eig, which='LM',
                              return_eigenvectors=False, maxiter=10000)).max()
        except Exception:
            lam = np.abs(np.linalg.eigvals(A.toarray())).max()
        if lam <= 0:
            raise RuntimeError("adjacency matrix has zero spectral radius")
        self.A = (A * (rho / lam)).tocsr()

        # --- input matrix: each node reads exactly one input component ------
        # (Pathak's construction: W_in is sparse, one nonzero per row)
        W_in = np.zeros((D_r, n_in))
        which = rng.integers(0, n_in, size=D_r)
        W_in[np.arange(D_r), which] = rng.uniform(-sigma, sigma, size=D_r)
        self.W_in = W_in

        self.P = None   # (n_in, 2*D_r), set by fit()

    # ------------------------------------------------------------------ #
    def _features(self, r):
        """s = [r ; r^2]."""
        return np.concatenate([r, r * r])

    def _advance(self, r, u):
        return np.tanh(self.A @ r + self.W_in @ u)

    def drive(self, U, r0=None):
        """
        Run the reservoir open-loop through the data U (n_steps, n_in).
        Returns the final state and, optionally, all states.
        """
        r = np.zeros(self.D_r) if r0 is None else r0.copy()
        for u in U:
            r = self._advance(r, u)
        return r

    # ------------------------------------------------------------------ #
    def fit(self, U, n_washout=1000, chunk=5000, verbose=True):
        """
        Train the readout to map r(t) -> u(t+dt).

        U          (n_steps, n_in) training trajectory
        n_washout  initial steps discarded so the reservoir state becomes
                   independent of r=0 (echo state property). NOT specified
                   in the paper.
        """
        n_steps = len(U) - 1
        Ds = 2 * self.D_r

        SS = np.zeros((Ds, Ds))          # S S^T
        SY = np.zeros((self.n_in, Ds))   # Y S^T

        r = np.zeros(self.D_r)
        for i in range(n_washout):
            r = self._advance(r, U[i])

        if verbose:
            print(f"  fit: D_r={self.D_r}, feature dim={Ds}, "
                  f"{n_steps - n_washout} training pairs", flush=True)

        # Alignment: absorb u[i] FIRST, then record. The resulting state is
        # r(t+dt) in the paper's notation and its readout targets u[i+1].
        # Recording before advancing trains a two-step map, which is still
        # learnable open-loop (one-step NRMSE looks fine) but is inconsistent
        # with the closed loop in predict() and destroys the rollout.
        buf_s, buf_y = [], []
        for i in range(n_washout, n_steps):
            r = self._advance(r, U[i])
            buf_s.append(self._features(r))
            buf_y.append(U[i + 1])

            if len(buf_s) >= chunk:
                S = np.asarray(buf_s).T           # (Ds, chunk)
                Y = np.asarray(buf_y).T           # (n_in, chunk)
                SS += S @ S.T
                SY += Y @ S.T
                buf_s, buf_y = [], []

        if buf_s:
            S = np.asarray(buf_s).T
            Y = np.asarray(buf_y).T
            SS += S @ S.T
            SY += Y @ S.T

        SS[np.diag_indices_from(SS)] += self.beta
        self.P = np.linalg.solve(SS, SY.T).T       # (n_in, Ds)
        self.r_end = r
        return self

    # ------------------------------------------------------------------ #
    def predict(self, U_seed, n_steps):
        """
        Teacher-force on U_seed (eps steps), then run closed-loop for n_steps.
        Reservoir state is reset to zero first, as in the paper's protocol.
        """
        if self.P is None:
            raise RuntimeError("call fit() first")
        r = np.zeros(self.D_r)
        for u in U_seed:
            r = self._advance(r, u)

        out = np.empty((n_steps, self.n_in))
        v = self.P @ self._features(r)
        for i in range(n_steps):
            out[i] = v
            r = self._advance(r, v)          # <- closed loop: own output back in
            v = self.P @ self._features(r)
        return out
