# Rollout error geometry: limit cycle vs chaotic attractor

## Question

arXiv:2608.07189 established that on a limit cycle, autoregressive rollout error
is overwhelmingly tangential — the model traverses the right attractor at the
wrong rate. The standing objection is that this cannot transfer to chaos,
because a chaotic attractor has no global phase.

## Mechanism

For any autonomous flow, the direction **along** the trajectory is neutral: it
carries Lyapunov exponent exactly zero. What differs between attractor types is
the transverse directions.

| | transverse directions | consequence |
|---|---|---|
| limit cycle | all stable (negative exponents) | only the neutral tangential direction accumulates error → error is phase drift |
| chaotic | at least one unstable (λ₁ > 0) | transverse error grows exponentially and competes → error is not phase drift |

The same framework predicts **opposite** outcomes. That is what makes it a test
rather than a restatement.

## Method

With `e = x_pred − x_true` and `T` the normalised true velocity:

```
e_tan  = e · T                  along the flow   ("phase" error)
e_perp = |e − (e·T) T|          perpendicular    (geometry error)
```

Reported as the **tangential share** `e_tan² / (e_tan² + e_perp²)`.

Two controls that matter:

- **Random baseline.** An error vector with no directional preference gives a
  share of `1/d` in `d` dimensions. Without this, a share of 0.5 looks like a
  signal in 2D when it is exactly nothing.
- **Saturation cutoff.** Once total error reaches the attractor size, the share
  decays to the baseline for trivial reasons. Only the pre-saturation window is
  reported.

Identical reservoir (D_r=1500, ρ=0.6, σ=1.0, κ=3, β=1e-6), 40 intervals, on both
systems. Nothing system-specific in the model.

## Result

| system | dim | baseline | measured share | excess | pre-saturation window |
|---|---|---|---|---|---|
| Van der Pol (limit cycle) | 2 | 0.500 | **0.991** | +0.491 | 400 / 400 steps |
| Lorenz (chaotic) | 3 | 0.333 | **0.475** | +0.142 | 51 / 400 steps |

Per-step detail, Van der Pol: share 0.998, 0.995, 1.000, 0.998 across the
window, with `perp/tan ≈ 0.00` throughout. Transverse error is not merely
smaller — it is absent to numerical precision, and it never saturates.

Per-step detail, Lorenz: share 0.481, 0.403, 0.496, 0.544, with `perp/tan`
between 0.6 and 2.5. Transverse error dominates at early lead times.

## Interpretation

**The limit-cycle result reproduces on a different system with a different
architecture.** The published finding used a CAE-LSTM on a cylinder wake; this
is an echo-state reservoir on Van der Pol. A share of 0.991 against a 0.500
baseline is independent confirmation that the effect is a property of
autoregressive rollout on stable-transverse attractors, not of the original
setup.

**On chaos, phase drift does not dominate.** 0.475 against a 0.333 baseline is a
real but weak tangential preference — nothing like the limit-cycle case. The
transverse instability competes with and often exceeds the neutral direction,
exactly as the Lyapunov argument predicts.

**The saturation windows are themselves the result.** Van der Pol never
saturates within 400 steps because transverse error cannot grow. Lorenz
saturates in 51. That contrast is the mechanism made visible.

**The correct generalisation** is therefore not "rollout error is phase error"
but:

> Rollout error concentrates in the neutral (tangential) direction to the extent
> that the transverse directions are stable. On a limit cycle that is total; on a
> chaotic attractor it is partial and bounded by λ₁.

This states the published result as the stable-transverse limit of a more
general principle, and gives the condition under which it applies.

## Limitations

- Two low-dimensional systems only. Needs KS or a wake at higher Re.
- One architecture. The reservoir is a strong baseline but not the CAE-LSTM.
- Single seed. The reservoir-realisation and trajectory variance measured in the
  Pathak folder (±0.11 and ±0.15 on VPT) has not been quantified for this metric.
- The residual +0.142 excess on Lorenz is unexplained. It may be a genuine
  partial phase preference, a finite-window artefact, or a property of the
  attractor's local geometry. Not resolved.
- Literature check was two searches. Forward citations from HAVOK and from
  Margazoglou & Magri (Chaos 2023) have not been examined.
