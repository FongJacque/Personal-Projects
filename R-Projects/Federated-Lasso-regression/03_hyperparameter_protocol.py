# =============================================================================
# STAT 102B – Final Project
# FILE 3: HYPERPARAMETER PROTOCOL
# Each node independently selects its own λ_j by searching a log grid
# and minimising its own validation MSE. λ_j is frozen for all rounds.
# Grid: 20 values log-spaced from 1.0 down to 0.001.
# Each candidate is evaluated after 100 CD epochs (near-converged).
# The pooled baseline λ is tuned here too for convenience.
# =============================================================================

import numpy as np
from 01_data_loading import nodes, WEIGHTS, P
from 02_core_optimization_operators import lasso_cd_epoch, lasso_val_mse

LAMBDA_GRID = np.logspace(0, -3, 20)   # [1.0, 0.786, ..., 0.001]

def tune_lambda(node_data, lam_grid=LAMBDA_GRID, n_epochs=100):
    """
    Select λ for one node via grid search on validation MSE.

    For each candidate λ in lam_grid:
      1. Start from β = 0.
      2. Run n_epochs CD epochs on the node's training data.
      3. Evaluate validation MSE (no penalty).
    Return the λ achieving the lowest validation MSE.

    Parameters
    ----------
    node_data : dict with keys X_train, y_train, X_val, y_val, m
    lam_grid  : 1-D array of candidate regularization values
    n_epochs  : int number of CD epochs per candidate (default 100)

    Returns
    -------
    best_lam : float
    """
    X_tr, y_tr = node_data["X_train"], node_data["y_train"]
    X_v,  y_v  = node_data["X_val"],   node_data["y_val"]
    p = X_tr.shape[1]

    best_lam, best_loss = None, np.inf
    for lam in lam_grid:
        beta = lasso_cd_epoch(X_tr, y_tr, np.zeros(p), lam, n_epochs=n_epochs)
        val  = lasso_val_mse(X_v, y_v, beta)
        if val < best_loss:
            best_loss = val
            best_lam  = lam
    return best_lam

if __name__ == "__main__":
    import numpy as np
    from 01_data_loading import nodes

    lambda_j = {}
    for label in ["A", "B", "C"]:
        lambda_j[label] = tune_lambda(nodes[label])
        print(f"  Node {label}: λ = {lambda_j[label]:.6f}")

    # Pooled node for baseline
    import numpy as np
    X_pool_tr = np.vstack([nodes[l]["X_train"] for l in ["A","B","C"]])
    y_pool_tr = np.concatenate([nodes[l]["y_train"] for l in ["A","B","C"]])
    X_pool_v  = np.vstack([nodes[l]["X_val"]   for l in ["A","B","C"]])
    y_pool_v  = np.concatenate([nodes[l]["y_val"]   for l in ["A","B","C"]])
    pool_node = {"X_train": X_pool_tr, "y_train": y_pool_tr,
                 "X_val": X_pool_v, "y_val": y_pool_v, "m": X_pool_tr.shape[0]}
    lambda_pool = tune_lambda(pool_node)
    print(f"  Pooled λ = {lambda_pool:.6f}")
