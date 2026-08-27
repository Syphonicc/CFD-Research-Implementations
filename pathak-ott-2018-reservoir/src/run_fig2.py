r"""
Pathak et al. (2018), Fig. 2 configuration:
Kuramoto-Sivashinsky, L = 22, Q = 64, mu = 0, single reservoir.

Usage:
    python run_fig2.py              # paper parameters, ~5 min
    python run_fig2.py --quick      # D_r=1000, short training, ~1 min
    python run_fig2.py --beta-sweep # the unspecified ridge parameter
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ks_solver import ks_solve
from reservoir import Reservoir
from evaluate import evaluate, nrmse_curve

# measured with lyapunov.py on this solver at L=22; see notes/ks_solver.md
LAMBDA_1 = 0.0466
DT = 0.25


def get_data(n_train, n_test, Lx=22.0, N=64):
    T = (n_train + n_test) * DT
    x, t, U = ks_solve(Lx=Lx, N=N, dt=DT, T=T, T_transient=1000.0,
                       verbose=False)
    return x, U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--beta-sweep", action="store_true")
    ap.add_argument("--D_r", type=int, default=5000)
    ap.add_argument("--n_train", type=int, default=70000)
    ap.add_argument("--beta", type=float, default=1e-4)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--tau", type=int, default=800)
    a = ap.parse_args()

    if a.quick:
        a.D_r, a.n_train, a.K, a.tau = 1000, 20000, 6, 400

    n_test = a.K * (10 + a.tau) + 1000
    print(f"generating KS data: {a.n_train + n_test} steps", flush=True)
    x, U = get_data(a.n_train, n_test)

    if a.beta_sweep:
        print("\nridge parameter sweep (not specified in the paper)", flush=True)
        for beta in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
            res = Reservoir(64, D_r=a.D_r, beta=beta, seed=0).fit(
                U[:a.n_train], n_washout=1000, verbose=False)
            r = evaluate(res, U[a.n_train:], DT, LAMBDA_1,
                         K=a.K, tau=a.tau, verbose=False)
            print(f"  beta={beta:.0e}  VPT = {r['vpt_mean']:.2f} "
                  f"+/- {r['vpt_std']:.2f}", flush=True)
        return

    print(f"training: D_r={a.D_r}, T={a.n_train}, beta={a.beta:.0e}", flush=True)
    t0 = time.time()
    res = Reservoir(64, D_r=a.D_r, rho=0.6, sigma=1.0, kappa=3,
                    beta=a.beta, seed=0).fit(U[:a.n_train], n_washout=1000)
    print(f"  fit took {time.time()-t0:.0f}s", flush=True)

    r = evaluate(res, U[a.n_train:], DT, LAMBDA_1, K=a.K, tau=a.tau)
    print(f"\nVPT = {r['vpt_mean']:.2f} +/- {r['vpt_std']:.2f} Lyapunov times "
          f"(NRMSE threshold {r['threshold']})", flush=True)

    # ---- Fig 2 style spacetime comparison ----
    seed = U[a.n_train:a.n_train + 10]
    true = U[a.n_train + 10:a.n_train + 10 + a.tau]
    pred = res.predict(seed, a.tau)
    lyap = np.arange(a.tau) * DT * LAMBDA_1

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    kw = dict(aspect="auto", origin="lower", cmap="RdBu_r",
              vmin=-3, vmax=3, extent=[0, lyap[-1], 0, 22])
    ax[0].imshow(true.T, **kw); ax[0].set_ylabel("x"); ax[0].set_title("truth")
    ax[1].imshow(pred.T, **kw); ax[1].set_ylabel("x"); ax[1].set_title("reservoir prediction")
    im = ax[2].imshow((pred - true).T, aspect="auto", origin="lower",
                      cmap="RdBu_r", vmin=-3, vmax=3,
                      extent=[0, lyap[-1], 0, 22])
    ax[2].set_ylabel("x"); ax[2].set_xlabel(r"$\Lambda_{max} t$")
    ax[2].set_title("error")
    fig.colorbar(im, ax=ax, shrink=0.6)
    plt.savefig("../figures/fig2_spacetime.png", dpi=120, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r["lyapunov_axis"], r["nrmse_mean"])
    ax.axhline(r["threshold"], color="r", ls="--",
               label=f"threshold {r['threshold']}")
    ax.axvline(r["vpt_mean"], color="g", ls=":",
               label=f"VPT = {r['vpt_mean']:.2f}")
    ax.set_xlabel(r"$\Lambda_{max} t$"); ax.set_ylabel("NRMSE")
    ax.legend(); ax.set_title(f"D_r={a.D_r}, T={a.n_train}, beta={a.beta:.0e}")
    plt.tight_layout()
    plt.savefig("../figures/fig2_nrmse.png", dpi=120)
    print("figures written to ../figures/", flush=True)


if __name__ == "__main__":
    main()
