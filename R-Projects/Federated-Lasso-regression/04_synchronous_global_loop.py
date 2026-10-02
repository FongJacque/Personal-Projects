# =============================================================================
# STAT 102B – Final Project
# FILE 4: THE SYNCHRONOUS GLOBAL LOOP
# Implements the central-aggregator federated Lasso algorithm.
#
# Each communication round t:
#   1. Server broadcasts current global consensus β_t to all nodes.
#   2. Each node initialises from β_t and runs E local CD epochs with its λ_j.
#   3. Server aggregates: β_{t+1} = Σ w_j · β_{j,t}
#   4. Server logs weighted training loss and checks stopping criterion.
#
# Stopping: ||β_{t+1} − β_t||₂ < 1e-5
# Privacy rule: the server never accesses any node's raw X or y data.
# =============================================================================

import numpy as np
from 01_data_loading import nodes, WEIGHTS, P
from 02_core_optimization_operators import lasso_cd_epoch, lasso_train_loss

TOLERANCE  = 1e-5
MAX_ROUNDS = 2000

def weighted_train_loss(nodes, beta_dict, lambda_j, weights):
    """
    Global diagnostic loss: weighted sum of each node's Lasso training loss.
    Nodes pass back scalar losses so the server never touches raw data.
    """
    return sum(
        weights[l] * lasso_train_loss(
            nodes[l]["X_train"], nodes[l]["y_train"], beta_dict[l], lambda_j[l])
        for l in nodes
    )

def run_federated(nodes, lambda_j, weights, E=1, tol=TOLERANCE, max_rounds=MAX_ROUNDS):
    """
    Synchronous central-aggregator federated Lasso.

    Parameters
    ----------
    nodes     : dict  label → node data dict (X_train, y_train, X_val, y_val, m)
    lambda_j  : dict  label → frozen local regularization parameter
    weights   : dict  label → aggregation weight (proportional to m_j)
    E         : int   number of local CD epochs per communication round
    tol       : float convergence tolerance on ||β_{t+1} − β_t||₂
    max_rounds: int   hard cap on communication rounds (for safety)

    Returns
    -------
    beta_global  : (p,) final converged global coefficient vector
    loss_history : list of weighted training losses, one per round
    n_rounds     : int  number of rounds until convergence
    """
    beta_global  = np.zeros(P)
    loss_history = []

    for t in range(max_rounds):
        # --- Local update: each node runs E epochs from current global β ---
        beta_local = {
            l: lasso_cd_epoch(
                nodes[l]["X_train"], nodes[l]["y_train"],
                beta_global, lambda_j[l], n_epochs=E)
            for l in nodes
        }

        # --- Server aggregation: weighted average ---
        beta_new = sum(weights[l] * beta_local[l] for l in nodes)

        # --- Log global weighted training loss ---
        loss_history.append(
            weighted_train_loss(nodes, beta_local, lambda_j, weights))

        # --- Check stopping criterion ---
        delta       = np.linalg.norm(beta_new - beta_global)
        beta_global = beta_new

        if delta < tol:
            print(f"  E={E}: converged at round {t+1}  (Δ={delta:.2e})")
            return beta_global, loss_history, t + 1

    print(f"  E={E}: reached max_rounds={max_rounds} without convergence.")
    return beta_global, loss_history, max_rounds

if __name__ == "__main__":
    from 03_hyperparameter_protocol import tune_lambda

    print("Tuning λ_j ...")
    lambda_j = {l: tune_lambda(nodes[l]) for l in ["A","B","C"]}
    for l in ["A","B","C"]:
        print(f"  Node {l}: λ = {lambda_j[l]:.6f}")

    results = {}
    for E in [1, 5, 10]:
        beta_g, loss_hist, n_rounds = run_federated(nodes, lambda_j, WEIGHTS, E=E)
        results[E] = {"beta": beta_g, "loss": loss_hist, "rounds": n_rounds}
