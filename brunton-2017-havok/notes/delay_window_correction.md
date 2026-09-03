# The delay window: a correction

An earlier version of `pod_vs_dmd.md` argued that the Hankel delay window must
span the slowest timescale in the system, and that performance therefore shows a
threshold as `q*dt` crosses that period. Run on Lorenz, HAVOK contradicts this.

## What the sweep shows

Lorenz, `r=15`, mean lobe-switch interval 1.73 time units:

| window (t.u.) | residual | antisymmetry | forcing kurtosis |
|---|---|---|---|
| 0.10 | 0.020 | 0.0001 | 75.5 |
| 0.20 | 0.061 | 0.0000 | 51.8 |
| 0.50 | 0.094 | 0.0001 | 9.9 |
| 1.00 | 0.143 | 0.0007 | 0.9 |
| 1.75 | 0.190 | 0.0007 | −0.2 |
| 3.00 | 0.238 | 0.0008 | −0.1 |
| 9.00 | 0.251 | 0.0030 | −0.1 |

Both the fit quality and the heavy-tailed forcing statistics are **best at short
windows and degrade monotonically as the window lengthens**. By the time the
window reaches the switching interval, the forcing kurtosis has collapsed to
zero — the forcing has become Gaussian and lost the property the paper is about.

The optimum is around 0.1 t.u., which is exactly the paper's choice
(`q=100`, `dt=0.001`).

## Why the earlier argument was wrong here

It assumed the linear operator is being asked to *represent* the slow timescale,
in which case that timescale must be inside the state.

HAVOK does the opposite. It deliberately excludes the switching from the linear
part and quarantines it in the forcing term `v_r`. Ask the linear operator to
span a switching event and you are asking it to do the thing the method was
designed not to attempt — so the fit degrades and the forcing loses its burst
structure.

## The reconciliation

Both statements can hold, for different objectives:

- **Operator must capture the slow dynamics** (a modulated flow, Hankel-DMD used
  for prediction): the window must span the modulation period. Threshold behaviour.
- **Operator captures only the locally linear part, with the rest as measured
  forcing** (HAVOK): the window should be short enough to *exclude* the
  intermittent event. Optimum, not threshold.

The window length is set by what you are asking the operator to do, not by the
system's slowest timescale alone. That is a sharper statement than the original
and it is the one supported by evidence.

**This affects the argument in arXiv:2608.07189.** The delay-sweep reversal
reported there is consistent with the first regime, but the paper states the
mechanism in general terms. It should specify the objective, or the claim is
falsified by HAVOK's own sweep.

## What did reproduce

At the paper's parameters (`q=100`, `dt=0.001`, `r=15`):

| Claim | Result |
|---|---|
| A is nearly antisymmetric | ‖A+Aᵀ‖/‖A‖ = 7e-5 — confirmed, and it holds across every window tested |
| forcing is non-Gaussian, long-tailed | excess kurtosis 69.4 (Gaussian = 0) — confirmed |
| forcing bursts precede lobe switching | mean max\|v_r\| before a switch = 2.16 vs 0.81 in random windows, ratio 2.7 — confirmed |
| model residual | 0.020 |
