# =============================================================================
# STAT 102B – Final Project
# FILE 5: THE POOLED BASELINE
# Combines all three nodes' training data into one unified dataset.
# Tunes λ on the pooled validation set using the same log grid.
# Runs coordinate descent to near-convergence (200 epochs).
# The resulting active set serves as ground truth for confusion matrices.
# This pipeline is kept strictly separate from the federated algorithm.
# =============================================================================

import numpy as np
from 01_data_loading import nodes, P
from 02_core_optimization_operators import lasso_cd_epoch
from 03_hyperparameter_protocol import tune_lambda

# --- Assemble pooled dataset (allowed only for the baseline) ---
X_pool_tr = np.vstack([nodes[l]["X_train"] for l in ["A", "B", "C"]])
y_pool_tr = np.concatenate([nodes[l]["y_train"] for l in ["A", "B", "C"]])
X_pool_v  = np.vstack([nodes[l]["X_val"]   for l in ["A", "B", "C"]])
y_pool_v  = np.concatenate([nodes[l]["y_val"]   for l in ["A", "B", "C"]])

pool_node = {
    "X_train": X_pool_tr, "y_train": y_pool_tr,
    "X_val":   X_pool_v,  "y_val":   y_pool_v,
    "m":       X_pool_tr.shape[0]
}

def fit_pooled_baseline(pool_node, n_epochs=200):
    """
    Fit the centralized pooled Lasso.
    Tunes λ on the pooled validation set, then runs CD to near-convergence.

    Parameters
    ----------
    pool_node : dict with X_train, y_train, X_val, y_val, m
    n_epochs  : int CD epochs for the final fit (default 200)

    Returns
    -------
    beta_pool   : (p,) fitted coefficient vector
    lambda_pool : float selected regularization parameter
    """
    lambda_pool = tune_lambda(pool_node)
    beta_pool   = lasso_cd_epoch(
        pool_node["X_train"], pool_node["y_train"],
        np.zeros(P), lambda_pool, n_epochs=n_epochs)
    return beta_pool, lambda_pool

if __name__ == "__main__":
    print("Fitting pooled baseline ...")
    beta_pool, lambda_pool = fit_pooled_baseline(pool_node)
    n_active = int(np.sum(np.abs(beta_pool) > 1e-4))
    print(f"  λ_pool   = {lambda_pool:.6f}")
    print(f"  Active coefficients: {n_active}")
