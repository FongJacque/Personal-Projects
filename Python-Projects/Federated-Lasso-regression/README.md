# Federated Lasso Regression

A from-scratch implementation of federated (distributed) Lasso regression,
built to explore how a sparse linear model can be trained across multiple
data-holding nodes that never share their raw data with each other or a
central server — only model coefficients ever leave a node.

## What it does

Three simulated nodes, each holding a different amount of local data,
collaboratively train a single shared Lasso model:

- Each node independently tunes its own regularization parameter (lambda)
  via grid search on its own validation set
- Each communication round, every node runs local coordinate-descent
  epochs, then a central aggregator combines the results via a
  **weighted average** (weighted by each node's sample size)
- The aggregator never sees raw data — only coefficient vectors
- Convergence is tracked via the change in the global coefficient vector
  round to round

The federated result is then compared against a **centralized pooled
baseline** (all data combined in one place) to evaluate how much accuracy,
if any, is lost by keeping data decentralized — including a full
confusion matrix comparing which features each approach identifies as
relevant.

## Why this is interesting

Federated learning matters anywhere raw data can't leave its source —
healthcare records across hospitals, financial data across institutions,
or any setting where privacy or regulation prevents centralizing data.
This project explores the core mechanics of that setting: local
optimization, coordinate descent with soft-thresholding for the L1
penalty, weighted aggregation, and convergence behavior under different
numbers of local epochs per round (E = 1, 5, 10).

## A note on the data

This repository ships with a **synthetic data generator**
(`generate_synthetic_data.py`) rather than the original dataset used
during development, out of respect for the source the assignment-style
data came from. The generator produces data with the same structure —
three nodes of different sizes, 600 features, a sparse true coefficient
vector — so the algorithm runs identically and produces comparable
results.

## How to run

```bash
pip install numpy pandas matplotlib
python final_project.py
```

## What I'd improve with more time

- Add asynchronous aggregation (nodes update on their own schedule
  rather than waiting for every node each round)
- Explore adaptive per-round communication (skip a round if a node's
  local update is negligible)
- Add cross-validation for lambda selection instead of a single
  train/val split per node
