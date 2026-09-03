# Brunton et al. (2017) — HAVOK: chaos as an intermittently forced linear system

> S. L. Brunton, B. W. Brunton, J. L. Proctor, E. Kaiser, J. N. Kutz,
> *Chaos as an intermittently forced linear system*,
> Nature Communications **8**, 19 (2017).
> [DOI](https://doi.org/10.1038/s41467-017-00030-8) ·
> [arXiv:1608.05306](https://arxiv.org/abs/1608.05306) ·
> open access · [author code](http://faculty.washington.edu/sbrunton/HAVOK.zip)

**Claim being tested:** delay-embedding a chaotic time series and applying DMD
yields a linear model in the leading delay coordinates, *forced* by the lowest-
energy delay coordinate. The forcing is intermittent and non-Gaussian, and its
bursts predict lobe switching in the Lorenz attractor.

---

## Why this paper

It is the reference for the delay-window argument used in
[arXiv:2608.07189](https://arxiv.org/abs/2608.07189) but never derived there.
The specific gap being closed: *why* Hankel-DMD performance changes as a
threshold in the delay length rather than as a smooth trend. The theory is
written up in [`notes/pod_vs_dmd.md`](notes/pod_vs_dmd.md).

Also chosen because it is fully reproducible: open access, Lorenz system
specified in closed form, author code published, laptop-scale.

## Status

| Target | Published | Reproduced | Status |
|--------|-----------|------------|--------|
| DMD on analytic signals | — | 6/6 analytic tests pass | done — `tests/test_dmd.py` |
| Fig. 1/2 — Lorenz delay embedding, eigen-time-delay coordinates | qualitative | — | not started |
| Fig. 3 — forcing statistics are non-Gaussian, long-tailed | qualitative + histogram | — | not started |
| Forcing bursts predict lobe switching | qualitative | — | not started |
| Delay-length sweep — the threshold behaviour | *not in the paper* | — | own extension |

The last row is not a reproduction. It is the question this folder exists to
answer, run on a system where the ground truth is known.

## What the paper does not specify

| Missing | Value used | Status |
|---------|-----------|--------|
| number of delays `q` | — | open |
| truncation rank `r` | — | open |
| Lorenz integration tolerance / timestep | — | open |
| threshold defining a forcing "burst" | — | open |

Fill in as resolved, with the evidence that resolved it.

## Layout

```
src/
  dmd.py          DMD (exact + projected), Hankel embedding, Hankel-DMD
  lorenz.py       Lorenz system integrator
  run_havok.py    the reproduction
tests/
  test_dmd.py     analytic signals with closed-form eigenvalues
notes/
  pod_vs_dmd.md   POD vs DMD mathematics, and where the delay window comes from
figures/
```

## Reproducing

```bash
python tests/test_dmd.py     # must be 6/6 before anything else is trusted
python src/run_havok.py
```

## Note on the tests

`tests/test_dmd.py` checks the implementation against signals whose eigenvalues
are known in closed form — a pure oscillation (growth exactly zero), a damped
oscillation (known decay rate), a two-frequency field, and a delay-embedding
case where a scalar series hides a frequency that Hankel-DMD recovers.

Test 0 records a trap worth knowing: a single spatial structure modulated by
`cos(wt)` is rank **one**, not two, because both exponentials share the same
spatial vector. Truncating it to `r=2` fabricates a spurious mode with a large
false growth rate. This was hit while writing the suite.
