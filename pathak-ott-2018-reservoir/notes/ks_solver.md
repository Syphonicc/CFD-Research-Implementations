# KS solver: ETDRK4, validation, and one failure mode

## The equation

```
u_t + u·u_x + u_xx + u_xxxx = 0,    periodic on [0, Lx)
```

In Fourier space with `û = FFT(u)`:

```
dû/dt = (k² - k⁴)·û  -  (ik/2)·FFT(u²)
         \_______/       \___________/
          L linear         N nonlinear
```

Term by term:

- `u·u_x` — Burgers nonlinearity. Transfers energy between scales.
- `u_xx` — moved to the RHS this is diffusion with a **negative** coefficient.
  Anti-diffusion: it amplifies gradients rather than smoothing them. Energy source.
- `u_xxxx` — fourth-order hyperdiffusion, scales as `k⁴`. Energy sink at small scales.

Energy in at large scales, out at small scales, nonlinearity moving it between.
That is the turbulent cascade in one dimension.

## Linear stability

Substituting `u ~ exp(ikx)exp(σt)` into the linearised equation:

```
σ(k) = k² - k⁴
```

Positive for `k < 1`, maximised at `dσ/dk = 2k - 4k³ = 0` → `k = 1/√2 ≈ 0.707`,
giving `σ_max = 0.25` and a characteristic cell width `2π√2 ≈ 8.9`.

Periodic boundaries quantise `k_n = 2πn/Lx`, so the number of linearly unstable
modes is approximately `Lx/2π`. **`Lx` sets the attractor dimension.** It does
not change the physics, only how many unstable directions exist.

## Why ETDRK4

The `k⁴` term makes the linear operator violently stiff: the highest resolved
mode has eigenvalue `~ -k_max⁴`. At `Lx=22, N=64` that is `-6.9e3`. Explicit RK4
would need `dt ~ k_max⁻⁴`.

ETDRK4 (Kassam & Trefethen, SIAM J. Sci. Comput. **26**, 1214, 2005) integrates
the linear part exactly via the matrix exponential and treats only the nonlinear
part explicitly, so `dt` is set by accuracy rather than stability. The
φ-functions are evaluated by contour integration in the complex plane to avoid
catastrophic cancellation as `hL → 0`.

## The failure mode — worth reading before porting kursiv.m

The first implementation used the complex FFT and blew up around `t ≈ 355`.

The diagnostic sequence:

1. **Blow-up time was independent of `N` (32→128) and `dt` (0.25→0.025).**
   Resolution- and timestep-independent failure is not a stability problem.
2. **Integrating the identical semi-discrete system with scipy's BDF was stable
   to t=500.** That isolates the fault to the stepper, not the equation or the
   spatial discretisation.
3. **Tracking `max|Im(u)|` over time:** 2e-16 at t=0, growing to 5e+15 by t=337.
   Fitted growth rate ≈ 0.215 per time unit.

That rate is the physical linear instability, `σ_max = 0.25`. Roundoff breaks the
Hermitian symmetry of `û` at the 1e-16 level; the instability then amplifies the
spurious imaginary part exponentially until it swamps the real solution.

**Fix:** use `rfft`/`irfft`. A real field becomes structurally guaranteed and the
mode cannot exist.

**Non-fix:** discarding the imaginary part each step. That hides the growth
without removing the forcing.

Note that Kassam & Trefethen's reference `kursiv.m` shares this property. It is
not visible there because the textbook example only integrates to `t = 150`.
This port passed at t=150 and failed at t=600.

> **A validation run shorter than the production run proves nothing.**

## Validation

| Check | Result | Expected |
|-------|--------|----------|
| Long-time stability | t = 30,000, max\|u\| = 3.096 | bounded attractor |
| Resolution independence | ⟨u²⟩ = 1.394 / 1.404 / 1.416 (N=32/64/128) | convergent |
| Timestep independence | ⟨u²⟩ = 1.404 / 1.397 / 1.389 (dt=0.25/0.1/0.05) | convergent |
| Leading Lyapunov exponent | **λ₁ = 0.0466** | ~0.043 at L=22 |
| Spectral peak | k = 0.571 | see below |
| Spectral tail | E(k_max)/E(peak) = 8e-34 | fully resolved |

Two results that need interpretation rather than acceptance:

**λ₁ = 0.0466 is ~8% above the commonly quoted 0.043.** Conventions differ
across the literature (some sources use `L = 22`, others `L = 22π`). Pin this
against whichever paper is being reproduced rather than inheriting a number.

**The spectral peak sits at k = 0.571, below the linear prediction of 0.707.**
Expected. Linear theory predicts which mode grows fastest *from rest*; the
saturated nonlinear state redistributes energy toward lower wavenumbers. The gap
is the nonlinearity, not an error.
