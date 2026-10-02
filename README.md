# Reproducibility audit — Chemom. Intell. Lab. Syst. 277 (2026) 105833

Supporting material for the corrigendum to *Prior-guided spectral gating for
interpretable deep learning in near-infrared spectroscopy*.

Every finding was produced twice, independently, from the article's own public
repository. Both reproductions are kept here side by side, unmerged, because
where they differ the difference is itself part of the finding.

```
reproductions/
  audit/         first reproduction — the audit that identified the errors
    verificar_pipeline_tecator.py     Table 1, both target columns, 10 replicates
    verificar_varredura_treino.py     training sweep: epochs, learning rate, KL weight
  independent/   second reproduction — written without access to the first
    verify_all.py                     column identity, PLS, gates, Gasoline
    verify_table1.py                  full Table 1, 10 replicates, both columns
outputs/         captured output of each script, as run on one machine
```

## Running them

Both sets read the article's repository through one environment variable:

```bash
export PGSG_REPO=/path/to/prior-guided-spectral-gating
python reproductions/audit/verificar_pipeline_tecator.py
python reproductions/independent/verify_all.py
```

Requires `numpy`, `pandas`, `scikit-learn`, `torch` and `scipy`. Nothing is
written to `PGSG_REPO`; scratch files go to `priors_tmp/` in the working
directory.

The only modification made to the audit scripts was replacing a hard-coded
absolute path with `PGSG_REPO`. They are otherwise exactly as written.

## What reproduces exactly

These are deterministic given the published split (`random_state=42`), and both
reproductions agree to every digit:

| quantity | published | reproduced |
|---|---|---|
| PLS single test, R² / RMSE | 0.919 / 0.296 | 0.9191 / 0.2957 |
| RMSECV at H minimum | 0.2093 ± 0.0269 | 0.2093 ± 0.0269 |
| H at minimum / H\* | 14 / 10 | 14 / 10 |
| Gasoline PLS, R² / RMSE | 0.975 / 0.145 | 0.9752 / 0.1453 |
| Gasoline H\* | 5 | 5 |
| parameter counts | 1,765 / 6,882 / 109,221 | identical |
| gate values at initialization | all eight of Table 2 | identical |

The column labelled `Fat` in `data/raw/tecator/targets.csv` is water: identical
in 215 of 215 rows to the water column of two independently labelled
distributions, maximum difference zero. On the correct target, H\* becomes 12
and PLS gives R² 0.946 / RMSE 0.243.

## What does not reproduce across machines

The network arms. The published pipeline sets no random seed, and even with
seeds fixed the medians move with library versions. Running the *same* script on
two machines:

| arm (col 0 / col 1) | machine A | machine B |
|---|---|---|
| Ours (ANOVA), single test | 0.922 / 0.922 | 0.932 / 0.900 |
| Ours (ANOVA), 5-fold | 0.869 / 0.868 | 0.864 / 0.859 |
| Plain CNN, single test | −0.019 / 0.005 | −0.016 / 0.001 |

The replicate spread within a single machine is wider than the difference
between machines — the single-test ANOVA arm alone ranges from 0.82 to 0.95.
This is why the corrigendum reports the network rows as medians with ranges, to
two decimals, and states the PGSG-to-PLS gap as approximately 6% on the
published target and approximately 9% on the correct one, rather than as point
values.

## One trap worth recording

The published pipeline uses a different recipe for each arm: the compact model
trains with KL weight 1e-3, weight decay 1e-4 and patience 30; the heavy CNN
with KL weight 1e-4, no weight decay and patience 20; the randomly initialized
arm runs all 200 epochs with no early stopping. Applying one recipe to every arm
shifts the heavy-CNN median by several tenths of an R² unit. The first
divergence between the two reproductions here came from exactly that, and
disappeared once each arm used its own recipe.

## Criterion for the gate measurements

Fixed before measuring: a training condition *reproduces* a published pair when
the top-gate weight **and** the gate–prior correlation both match the published
values at three decimal places. Untrained reproduces the pair in 10 of 10 seeds
on both datasets; the article's training protocol reproduces it in none.
Individual quantities taken in isolation behave differently, which is why the
joint criterion was declared in advance.

Across the twenty conditions of the sweep (training length, learning rate, KL
weight), zero training is the only one that reproduces the pair in 10 of 10.
Every other condition reaches 3 of 10 or fewer, and those partial hits occur
only at one to five epochs or at a learning rate of 1e-4 — conditions in which
training has barely moved the model.

The KL weight grid does not behave as Section 3.4 reports either. Over the range
the article tested, 1e-4 to 1e-1, the Tecator gate-prior correlation stays
between 0.333 and 0.350, against a claimed collapse above r = 0.85 for
lambda >= 0.01. Pushing two orders of magnitude past the article's grid, to
lambda = 1 and 10, reaches only 0.42 and 0.49 on Tecator and 0.55 and 0.61 on
Gasoline. At lambda = 0, with the regularizer off entirely, the correlation is
0.337 — the same value as at the article's lambda = 0.001.
