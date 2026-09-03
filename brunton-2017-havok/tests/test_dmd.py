r"""
Analytic tests for the DMD implementation.

Every test uses a signal whose eigenvalues are known in closed form, so a
failure localises to the code rather than to the data.

A SUBTLETY THAT BITES FIRST-TIME IMPLEMENTATIONS
------------------------------------------------
A single spatial vector modulated by cos(wt) is RANK ONE. Writing
cos = (e^{iwt} + e^{-iwt})/2 gives two exponentials sharing the *same* spatial
structure, so the two DMD modes are linearly dependent and cannot be separated.
Truncating such data to r=2 manufactures a spurious mode with a large negative
growth rate.

An oscillation needs TWO independent spatial structures -- a cos part and a sin
part -- to be rank 2 and DMD-representable. Physically this is a travelling or
rotating structure rather than a purely standing one. Check the singular value
spectrum before choosing r.
"""

import numpy as np
from dmd import dmd, dmd_reconstruct, hankel, hankel_dmd


def check(name, got, want, tol):
    ok = np.abs(got - want) < tol
    flag = "PASS" if np.all(ok) else "FAIL"
    print(f"  [{flag}] {name}: got {np.round(got, 6)}, want {want}", flush=True)
    return bool(np.all(ok))


def test_rank_deficiency():
    print("\n0. rank of a standing oscillation (the trap)")
    dt = 0.01
    t = np.arange(0, 10, dt)
    a = np.array([1.0, 0.5, -0.2])
    X = np.outer(a, np.cos(2 * np.pi * 1.3 * t))
    S = np.linalg.svd(X, compute_uv=False)
    r = int(np.sum(S > 1e-10 * S[0]))
    return check("numerical rank of a*cos(wt)", r, 1, 0.5)


def test_pure_oscillation():
    print("\n1. undamped oscillation: freq known, growth zero")
    dt, f = 0.01, 1.3
    t = np.arange(0, 10, dt)
    a1 = np.array([1.0, 0.5, -0.2])
    a2 = np.array([0.1, -0.8, 0.3])
    X = np.outer(a1, np.cos(2 * np.pi * f * t)) + \
        np.outer(a2, np.sin(2 * np.pi * f * t))
    res = dmd(X, dt=dt, r=2)
    ok = check("frequency", np.abs(res["freq"]), f, 1e-8)
    ok &= check("growth rate", res["growth"], 0.0, 1e-8)
    return ok


def test_damped_oscillation():
    print("\n2. damped oscillation: decay rate known")
    dt, f, sig = 0.01, 1.3, -0.35
    t = np.arange(0, 10, dt)
    env = np.exp(sig * t)
    a1 = np.array([1.0, 0.5, -0.2])
    a2 = np.array([0.1, -0.8, 0.3])
    X = np.outer(a1, env * np.cos(2 * np.pi * f * t)) + \
        np.outer(a2, env * np.sin(2 * np.pi * f * t))
    res = dmd(X, dt=dt, r=2)
    ok = check("growth rate", res["growth"], sig, 1e-8)
    ok &= check("frequency", np.abs(res["freq"]), f, 1e-8)
    return ok


def test_two_frequencies():
    print("\n3. two frequencies, distinct spatial modes, plus reconstruction")
    dt = 0.01
    t = np.arange(0, 10, dt)
    xs = np.linspace(0, 2 * np.pi, 64)
    f1, f2 = 0.7, 2.1
    X = (np.outer(np.sin(xs), np.cos(2 * np.pi * f1 * t))
         + np.outer(np.cos(xs), np.sin(2 * np.pi * f1 * t))
         + np.outer(np.sin(3 * xs), np.cos(2 * np.pi * f2 * t))
         + np.outer(np.cos(3 * xs), np.sin(2 * np.pi * f2 * t)))
    res = dmd(X, dt=dt, r=4)
    got = np.sort(np.unique(np.round(np.abs(res["freq"]), 6)))
    ok = check("frequencies", got, np.array([f1, f2]), 1e-5)
    rec = dmd_reconstruct(res, len(t), dt).real
    err = np.linalg.norm(rec - X) / np.linalg.norm(X)
    print(f"  [{'PASS' if err < 1e-9 else 'FAIL'}] reconstruction rel err: {err:.2e}",
          flush=True)
    return ok and err < 1e-9


def test_hankel_shape():
    print("\n4. Hankel matrix construction")
    x = np.arange(10.0)
    H = hankel(x, 4)
    ok = check("shape", np.array(H.shape), np.array([4, 7]), 0.5)
    ok &= check("first column", H[:, 0], np.array([0., 1., 2., 3.]), 1e-12)
    return ok


def test_hankel_hidden_frequency():
    print("\n5. delay embedding recovers a frequency a scalar series hides")
    dt = 0.01
    t = np.arange(0, 20, dt)
    f1, f2 = 0.7, 1.9
    x = np.cos(2 * np.pi * f1 * t) + 0.6 * np.cos(2 * np.pi * f2 * t)

    # a scalar series is rank 1: plain DMD has nothing to work with
    res_plain = dmd(np.atleast_2d(x), dt=dt, r=1)
    print(f"       plain DMD on the scalar series -> "
          f"{len(res_plain['freq'])} mode(s), useless", flush=True)

    res = hankel_dmd(x, q=200, dt=dt, r=4)
    got = np.sort(np.unique(np.round(np.abs(res["freq"]), 4)))
    return check("frequencies from delays", got, np.array([f1, f2]), 1e-3)


if __name__ == "__main__":
    tests = [test_rank_deficiency, test_pure_oscillation,
             test_damped_oscillation, test_two_frequencies,
             test_hankel_shape, test_hankel_hidden_frequency]
    results = [t() for t in tests]
    print(f"\n{sum(results)}/{len(results)} passed", flush=True)
