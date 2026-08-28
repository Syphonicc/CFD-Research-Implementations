# CFD-Research-Implementations

Reproductions of published results in flow physics and data-driven modelling of
fluid systems. Each folder targets one paper, reports a specific number against
the published value, and documents the choices the paper left unspecified.

Failed and partial reproductions are kept, not deleted. The gap between a paper
and a working implementation is usually the interesting part.

## Reproductions

| # | Paper | Target metric | Published | Reproduced | Status |
|---|-------|---------------|-----------|------------|--------|
| 1 | Fukagata & Fukami — CNN-AE | ** | ** | ** | — |
| 2 | [Pathak et al., PRL **120**, 024102 (2018)](pathak-ott-2018-reservoir/) — reservoir computing on Kuramoto–Sivashinsky | valid prediction time | ~8 Lyapunov times | — | in progress |

## Conventions

Each reproduction folder follows the same layout:

```
<author>-<year>-<method>/
├── README.md      paper reference, parameter table, results, deviations
├── src/           implementation
├── figures/       generated output
└── notes/         theory worked through, derivations, dead ends
```

Every folder README states three things explicitly:

1. **What the paper specifies** — the parameters lifted directly from the text
2. **What it doesn't** — hyperparameters absent from the paper, and the value chosen
3. **What matched and what didn't** — with numbers, not adjectives

## Environment

Python ≥ 3.10, `numpy`, `scipy`, `matplotlib`. PyTorch only where a reproduction
requires it. Generated data (`*.npz`, `*.npy`) is gitignored — every dataset here
is reproducible from the scripts with a fixed seed.
