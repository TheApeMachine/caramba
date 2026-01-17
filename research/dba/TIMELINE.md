# Timeline

This file shows the exact timeline how things ran.

## Exploratory Runs

These runs were done to select the best decoupled model.

It turned out that the sem16 geo32 model underperformed when benchmarked, and the sem8 geo32 v40 model was the best.

GPU: A100 80GB
LAYERS: 22
STEPS: 10k

manifest: ./config/presets/dba_paper_rerun.yml

## Full Runs

These runs directly compare the sem8 geo32 v40 model with the baseline.

GPU: A100 80GB
LAYERS: 22
STEPS: 100k

manifest: ./config/presets/dba_paper_rerun.yml

## Ablations

These runs were to satisfy the reviewer request to show that the results were not due to other factors.
These also revealed a pretty big win for the gated DBA variant.

GPU: A100 40GB
LAYERS: 12
STEPS: 10k

manifest: ./config/presets/dba_paper_local.yml

## Gated Full Run

Given the success of the gated DBA variant in the ablations, we ran a full-length run to compare it with the baseline.

GPU: A100 80GB
LAYERS: 22
STEPS: 100k

manifest: ./config/presets/dba_paper_gated.yml
