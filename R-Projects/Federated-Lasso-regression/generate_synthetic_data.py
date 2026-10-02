"""
generate_synthetic_data.py
---------------------------
Generates synthetic data matching the shape and statistical structure of
the original three-node federated Lasso setup, WITHOUT using any of the
original assignment data. Safe to publish and safe for anyone who clones
the repo to run immediately.

Structure matched:
    - p = 600 features
    - 3 nodes with different training sizes (400 / 240 / 160), mirroring
      the original node size imbalance that motivates weighted aggregation
    - A sparse true coefficient vector (most features irrelevant), since
      that's the whole point of demonstrating Lasso's variable selection
    - Gaussian noise added to the response, so recovery isn't trivial

Run this once before run_project.py / final_project.py if the data/
folder is empty. It's deterministic (fixed seed) so results are
reproducible run to run.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
P = 600                       # number of features, matches original
N_ACTIVE = 15                 # number of truly non-zero coefficients
NOISE_STD = 1.0

# (train_size, val_size) per node, matching the original node imbalance
NODE_SIZES = {
    1: (400, 100),   # becomes "A" (largest) after sorting in final_project.py
    2: (240, 60),    # becomes "B"
    3: (160, 40),    # becomes "C" (smallest)
}

OUTPUT_DIR = Path(__file__).parent / "data"


def make_true_beta(p=P, n_active=N_ACTIVE, rng=None):
    """
    Build a sparse ground-truth coefficient vector.

    Most entries are exactly zero; a small random subset of indices get
    non-zero coefficients drawn from a moderate range, so recovering the
    correct active set is a real (not trivial) test for the Lasso.
    """
    beta = np.zeros(p)
    active_idx = rng.choice(p, size=n_active, replace=False)
    beta[active_idx] = rng.uniform(-3, 3, size=n_active)
    return beta


def make_node_data(m_train, m_val, beta_true, p=P, noise_std=NOISE_STD, rng=None):
    """
    Generate one node's train/val split from a shared true beta.

    X is standard normal (independent features); y = X @ beta_true + noise.
    Using the SAME beta_true across all nodes simulates the realistic
    federated setting where clients share an underlying model but hold
    different local samples of it.
    """
    X_train = rng.normal(size=(m_train, p))
    X_val = rng.normal(size=(m_val, p))

    y_train = X_train @ beta_true + rng.normal(scale=noise_std, size=m_train)
    y_val = X_val @ beta_true + rng.normal(scale=noise_std, size=m_val)

    return X_train, y_train, X_val, y_val


def main():
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    beta_true = make_true_beta(rng=rng)
    np.save(OUTPUT_DIR / "beta_true.npy", beta_true)  # kept for optional recovery-accuracy checks

    for node_id, (m_train, m_val) in NODE_SIZES.items():
        X_train, y_train, X_val, y_val = make_node_data(
            m_train, m_val, beta_true, rng=rng
        )

        pd.DataFrame(X_train).to_csv(OUTPUT_DIR / f"node{node_id}_X_train.csv", index=False)
        pd.DataFrame(y_train, columns=["0"]).to_csv(OUTPUT_DIR / f"node{node_id}_y_train.csv", index=False)
        pd.DataFrame(X_val).to_csv(OUTPUT_DIR / f"node{node_id}_X_val.csv", index=False)
        pd.DataFrame(y_val, columns=["0"]).to_csv(OUTPUT_DIR / f"node{node_id}_y_val.csv", index=False)

        print(f"Node {node_id}: wrote {m_train} train / {m_val} val rows (p={P})")

    print(f"\nDone. Synthetic data written to {OUTPUT_DIR}/")
    print(f"{N_ACTIVE} of {P} coefficients are truly non-zero (see beta_true.npy).")


if __name__ == "__main__":
    main()
