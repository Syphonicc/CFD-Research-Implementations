# Implementation failures worth recording

Two bugs in this reproduction shared a shape: **a validation that passed while
the thing actually being measured was broken.**

## 1. KS solver — Hermitian symmetry amplified by the physical instability

Full write-up in `ks_solver.md`. Summary: the complex-FFT implementation blew up
at t ≈ 355, independent of `N` (32→128) and `dt` (0.25→0.025). Resolution- and
timestep-independent failure is not a stability problem. Integrating the
identical semi-discrete system with scipy's BDF was stable, isolating the fault
to the stepper. Tracking `max|Im(u)|` showed growth from 2e-16 to 5e+15 at a
rate of 0.215 per time unit — which is `σ_max = 0.25`, the physical linear
instability. Roundoff breaks the Hermitian symmetry of `û`; the instability then
amplifies it exponentially.

Fix: `rfft`/`irfft`, making a real field structurally guaranteed.

Kassam & Trefethen's reference `kursiv.m` shares this property. It is invisible
there because the textbook example only integrates to t = 150. This port passed
at t = 150 and failed at t = 600.

## 2. Reservoir — step misalignment between training and rollout

First working version gave VPT = 0.10 Lyapunov times: the rollout died within
eight steps. But the teacher-forced one-step NRMSE was 4.2e-3, i.e. the readout
was accurate.

That split located it. `fit()` recorded the reservoir state *before* absorbing
`u[i]` while targeting `u[i+1]` — training a two-step map. `predict()` advanced
then read, a one-step map. A two-step map is perfectly learnable, so every
open-loop metric looked healthy; only the closed loop exposed the inconsistency.

Fix: advance first, then record. See the comment in `reservoir.py:fit`.

## The transferable point

In both cases the diagnostic was the same: find a measurement that *separates*
the components. For the solver, BDF on the same spatial discretisation separated
integrator from equation. For the reservoir, teacher-forced one-step error
separated readout quality from closed-loop stability. Neither bug was findable
from the aggregate metric alone.
