# POD and DMD: what is actually different

Both start from the same object and the same SVD. The difference is one
assumption, and everything else follows from it.

## The data

Stack $m$ snapshots of an $n$-dimensional field as columns:

$$\mathbf{X} = \begin{bmatrix} \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_m \end{bmatrix} \in \mathbb{R}^{n \times m}$$

## POD

Take the SVD:

$$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^*$$

- $\mathbf{U}$ — columns are the POD **spatial** modes, orthonormal in space
- $\mathbf{V}$ — columns are the POD **temporal** coefficients, orthonormal in time
- $\boldsymbol{\Sigma}$ — singular values, the energy ranking

So POD already has both a spatial and a temporal factor. Reconstruction is

$$\mathbf{x}(t_k) = \sum_j \sigma_j \, \mathbf{u}_j \, v_j(t_k)$$

and the temporal coefficient $v_j(t_k)$ is **an arbitrary sequence of numbers**.
Nothing constrains it. A single POD mode's coefficient generally contains many
frequencies mixed together.

**The decisive property:** POD is invariant to the ordering of the snapshots.
Shuffle the columns of $\mathbf{X}$ and $\mathbf{U}$ and $\boldsymbol{\Sigma}$
are unchanged — only $\mathbf{V}$ permutes. POD is a *static* decomposition. It
optimally captures variance and knows nothing about dynamics.

This is the honest answer to "isn't POD already spatiotemporal?" It is, but its
temporal part is descriptive, not dynamical.

## DMD

Split the snapshots into two staggered matrices:

$$\mathbf{X}_1 = \begin{bmatrix} \mathbf{x}_1 & \cdots & \mathbf{x}_{m-1}\end{bmatrix}, \qquad
\mathbf{X}_2 = \begin{bmatrix} \mathbf{x}_2 & \cdots & \mathbf{x}_{m}\end{bmatrix}$$

and impose the **one assumption**: a single linear operator advances every
snapshot by one step.

$$\mathbf{X}_2 \approx \mathbf{A}\mathbf{X}_1, \qquad \mathbf{A} \in \mathbb{R}^{n\times n}$$

$\mathbf{A}$ is the best-fit linear map in a least-squares sense,
$\mathbf{A} = \mathbf{X}_2\mathbf{X}_1^{+}$. It is $n \times n$ — for a flow
field with $10^6$ grid points, unusable. So project it.

### The algorithm

1. **SVD of $\mathbf{X}_1$, truncated to rank $r$** — this step *is* POD:
   $\mathbf{X}_1 \approx \mathbf{U}_r\boldsymbol{\Sigma}_r\mathbf{V}_r^*$

2. **Project $\mathbf{A}$ onto the POD basis:**
   $$\tilde{\mathbf{A}} = \mathbf{U}_r^*\mathbf{A}\mathbf{U}_r = \mathbf{U}_r^*\mathbf{X}_2\mathbf{V}_r\boldsymbol{\Sigma}_r^{-1}$$
   Now $r \times r$ — small.

3. **Eigendecompose:** $\tilde{\mathbf{A}}\mathbf{W} = \mathbf{W}\boldsymbol{\Lambda}$

4. **Lift back to modes.**
   - Projected DMD (Schmid 2010): $\boldsymbol{\Phi} = \mathbf{U}_r\mathbf{W}$
   - Exact DMD (Tu et al. 2014): $\boldsymbol{\Phi} = \mathbf{X}_2\mathbf{V}_r\boldsymbol{\Sigma}_r^{-1}\mathbf{W}$

   Exact modes are true eigenvectors of $\mathbf{A}$; projected modes are their
   projection onto the POD subspace. They coincide when the data are exactly
   rank $r$.

5. **Convert to continuous time:** $\omega_j = \dfrac{\ln \lambda_j}{\Delta t}$

   $\mathrm{Re}(\omega_j)$ is a growth rate, $\mathrm{Im}(\omega_j)$ an angular
   frequency. This is where the physics appears.

6. **Amplitudes:** $\mathbf{b} = \boldsymbol{\Phi}^{+}\mathbf{x}_1$

### The resulting ansatz

$$\boxed{\;\mathbf{x}(t) \approx \sum_{j=1}^{r} \boldsymbol{\phi}_j \, e^{\omega_j t} \, b_j\;}$$

**This is the whole difference.** Each DMD mode is a spatial structure
multiplied by exactly ONE complex exponential in time. A POD mode's temporal
behaviour is whatever the data gave; a DMD mode's is a pure exponential by
construction.

## Side by side

| | POD | DMD |
|---|---|---|
| Objective | maximise captured variance | best-fit linear time-advance operator |
| Spatial modes | orthonormal | **not** orthogonal |
| Temporal behaviour | arbitrary, multi-frequency | single $e^{\omega t}$ per mode |
| Ranking | by energy | by eigenvalue (growth/frequency) |
| Snapshot order | irrelevant | **essential** |
| Output per mode | structure + energy | structure + frequency + growth rate |
| Built on | the SVD | the SVD, plus a regression in that basis |

DMD is not an alternative to POD. **DMD is POD plus a linear regression in the
reduced basis.** Step 1 of every DMD implementation is a POD.

For a cylinder wake, POD returns mode *pairs* whose correspondence to vortex
shedding you have to infer; DMD returns the shedding frequency and its
harmonics as separately labelled modes, each with its own spatial structure.
That is why DMD spread through experimental fluid dynamics: it makes a
multi-frequency flow legible mode by mode.

## Two failure modes worth knowing before using it

### Rank deficiency of standing oscillations

A single spatial vector modulated by $\cos(\omega t)$ is **rank one**. Writing
$\cos\omega t = \tfrac{1}{2}(e^{i\omega t} + e^{-i\omega t})$ gives two
exponentials sharing the *same* spatial structure, so the corresponding DMD
modes are linearly dependent. Truncating such data to $r=2$ manufactures a
spurious mode with a large false growth rate.

An oscillation needs a cos part **and** a sin part with distinct spatial
structures to be rank 2 and DMD-representable — physically, a travelling or
convecting structure rather than a purely standing one. Always look at the
singular value spectrum before choosing $r$.

(Verified in `test_dmd.py`, test 0.)

### Continuous spectrum — why DMD must fail on chaos

DMD approximates the Koopman operator restricted to a finite-dimensional
subspace. That works when the system has a **discrete** spectrum: a limit cycle
has a countable set of frequencies, so finitely many eigenvalues represent it
exactly.

A chaotic attractor has a **continuous** spectrum. No finite set of Koopman
eigenfunctions spans the dynamics. DMD will still return $r$ eigenvalues — they
just do not mean anything, and they look perfectly plausible. This is a failure
that produces confident wrong answers rather than a crash.

## Delay embedding, and where the window length comes from

Plain DMD assumes $\mathbf{x}(t)$ determines $\mathbf{x}(t+\Delta t)$. When the
true system has dynamics the snapshot cannot see — a slow modulation, an
unmeasured variable — that assumption is false and the fitted operator is
chasing a moving target.

Takens' embedding theorem says a delay vector

$$\mathbf{h}(t) = \begin{bmatrix}\mathbf{x}(t) \\ \mathbf{x}(t+\Delta t) \\ \vdots \\ \mathbf{x}(t+(q-1)\Delta t)\end{bmatrix}$$

reconstructs a space diffeomorphic to the attractor, for $q$ large enough.
Stacking snapshots into a Hankel matrix and running DMD on it is Hankel-DMD.

**The window length argument.** The augmented state spans $q\Delta t$ of
history. If $q\Delta t$ is *shorter* than the slowest timescale in the system,
that timescale is still invisible to the state — the operator is fitting
something that changes underneath it and performs worse than useless. Once
$q\Delta t$ *exceeds* that timescale, the modulation lives inside the state and
the operator becomes genuinely time-invariant over the embedding.

That is why a delay sweep shows a **threshold, not a trend**: performance can
reverse from worse-than-baseline to better-than-baseline as $q$ crosses the
slowest period, rather than improving smoothly.

Practical note: on Hankel data, always pass an **explicit rank**. Energy-based
truncation silently retains spurious modes because delay embedding redistributes
the singular value spectrum.

## References

- Schmid, *J. Fluid Mech.* **656**, 5 (2010) — original DMD
- Tu et al., *J. Comput. Dyn.* **1**, 391 (2014) — exact DMD, the theory paper
- Brunton et al., *Nat. Commun.* **8**, 19 (2017) — HAVOK, delay embedding + Koopman
- Arbabi & Mezić, *SIAM J. Appl. Dyn. Syst.* **16**, 2096 (2017) — Hankel-DMD convergence
