import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path

np.random.seed(1)

# =============================================================================
# SECTION 1: DATA LOADING
# Load each node's train/validation data into a dictionary.
# Raw data never leaves node scope — the aggregator only sees beta vectors.
# Node sizes: 400 train rows (largest), 240, and 160. We relabel so
# Node A = largest = weight 0.50, etc.
#
# NOTE: this repo ships with a synthetic data generator (generate_synthetic_
# data.py) rather than the original coursework dataset. Run that first if
# data/ is empty — this script will also auto-generate it for you below.
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"

if not (DATA_DIR / "node1_X_train.csv").exists():
    print("No data found — generating synthetic data (see generate_synthetic_data.py)...")
    import generate_synthetic_data
    generate_synthetic_data.main()

def load_node_data(node_id):
    """
    Load train and validation arrays for one node from CSV files.

    Parameters
    ----------
    node_id : int
        File index of the node to load (1, 2, or 3).

    Returns
    -------
    dict
        Dictionary with keys 'X_train' (m, p), 'y_train' (m,),
        'X_val' (v, p), 'y_val' (v,), and 'm' (int training sample size).
    """
    X_train = pd.read_csv(DATA_DIR / f"node{node_id}_X_train.csv").values.astype(float)
    y_train = pd.read_csv(DATA_DIR / f"node{node_id}_y_train.csv").values.astype(float).ravel()
    X_val   = pd.read_csv(DATA_DIR / f"node{node_id}_X_val.csv").values.astype(float)
    y_val   = pd.read_csv(DATA_DIR / f"node{node_id}_y_val.csv").values.astype(float).ravel()
    return {"X_train": X_train, "y_train": y_train,
            "X_val":   X_val,   "y_val":   y_val,
            "m": X_train.shape[0]}

raw_data = {j: load_node_data(j) for j in [1, 2, 3]}

# Sort by descending training size → A=400, B=240, C=160
sorted_keys = sorted(raw_data, key=lambda j: raw_data[j]["m"], reverse=True)
nodes = {}
for label, key in zip(["A", "B", "C"], sorted_keys):
    nodes[label] = raw_data[key]

for label, d in nodes.items():
    print(f"Node {label}: m_train={d['m']}, m_val={d['X_val'].shape[0]}, p={d['X_train'].shape[1]}")

total_m = sum(d["m"] for d in nodes.values())
WEIGHTS = {label: nodes[label]["m"] / total_m for label in nodes}
print(f"Weights: { {k: round(v,2) for k,v in WEIGHTS.items()} }")

P = 600

# =============================================================================
# SECTION 2: CORE OPTIMIZATION OPERATORS
# Soft-thresholding and one coordinate-descent epoch for the Lasso objective:
#   (1/2m)||y - Xbeta||^2 + lambda||beta||1
# One epoch = one full sequential pass over all p=600 coordinates.
# Vectorized: precompute column norms z_k = ||x_k||^2 once per epoch call.
# =============================================================================

def soft_threshold(rho, lam):
    """
    Apply the soft-thresholding operator element-wise.

    Computes sign(rho) * max(|rho| - lam, 0), the closed-form solution
    to the proximal operator of the L1 penalty. Entries with |rho| <= lam
    are shrunk exactly to zero; entries beyond lam are shrunk toward zero
    by lam.

    Parameters
    ----------
    rho : float or ndarray
        Input value(s) to threshold. In coordinate descent this is the
        univariate OLS numerator x_k^T r_k for coordinate k.
    lam : float
        Threshold level. Must be non-negative and already scaled to match
        the units of rho (multiply by m when using the 1/2m loss convention).

    Returns
    -------
    float or ndarray
        Soft-thresholded value(s), same shape as rho.
    """
    return np.sign(rho) * np.maximum(np.abs(rho) - lam, 0.0)

def lasso_cd_epoch(X, y, beta, lam, n_epochs=1):
    """
    Run one or more full coordinate-descent epochs on the Lasso objective.

    Minimises (1/2m)||y - X*beta||^2 + lam*||beta||_1 by cycling through
    all p coordinates sequentially. The residual is updated incrementally
    after each coordinate step (O(m) per step) rather than recomputing
    X*beta from scratch (O(m*p)). Column squared norms z_k = ||x_k||^2
    are precomputed once before the epoch loop.

    Per coordinate k the update rule is:
        r_k   = y - X*beta + x_k * beta_k   (partial residual)
        rho_k = x_k^T r_k                   (univariate OLS numerator)
        z_k   = ||x_k||^2                   (precomputed column norm)
        beta_k <- soft(rho_k, lam * m) / z_k

    Parameters
    ----------
    X : ndarray of shape (m, p)
        Design matrix for this node's training data.
    y : ndarray of shape (m,)
        Response vector for this node's training data.
    beta : ndarray of shape (p,)
        Starting coefficient vector. Copied internally; not modified in place.
    lam : float
        Regularization parameter. Multiplied by m internally to match the
        scale of rho_k under the 1/2m loss convention.
    n_epochs : int, optional
        Number of full passes over all p coordinates. Default is 1.

    Returns
    -------
    beta : ndarray of shape (p,)
        Updated coefficient vector after n_epochs passes.
    """
    m, p  = X.shape
    beta  = beta.copy()
    lam_s = lam * m                        # absorb 1/m scaling into lambda
    z     = np.einsum('ij,ij->j', X, X)   # shape (p,) — column squared norms
    resid = y - X @ beta                   # initialise residual

    for _ in range(n_epochs):
        for k in range(p):
            beta_k_old = beta[k]
            rho_k      = X[:, k] @ resid + z[k] * beta_k_old  # partial resid trick
            beta_k_new = soft_threshold(rho_k, lam_s) / z[k] if z[k] != 0 else 0.0
            beta[k]    = beta_k_new
            # update residual incrementally (avoids full X@beta each step)
            resid     += X[:, k] * (beta_k_old - beta_k_new)

    return beta

def lasso_train_loss(X, y, beta, lam):
    """
    Compute the Lasso training objective at a given coefficient vector.

    Evaluates (1/2m)||y - X*beta||^2 + lam*||beta||_1, the penalised
    least-squares loss used both as the local node objective during
    coordinate descent and as a server-side diagnostic after aggregation.

    Parameters
    ----------
    X : ndarray of shape (m, p)
        Design matrix.
    y : ndarray of shape (m,)
        Response vector.
    beta : ndarray of shape (p,)
        Coefficient vector at which to evaluate the loss.
    lam : float
        Regularization parameter, same scale as used during fitting.

    Returns
    -------
    float
        Scalar value of the Lasso objective.
    """
    m     = len(y)
    resid = y - X @ beta
    return (resid @ resid) / (2 * m) + lam * np.sum(np.abs(beta))

def lasso_val_mse(X, y, beta):
    """
    Compute the mean squared error on a validation set without penalty.

    Used exclusively during the hyperparameter grid search to select lambda_j
    for each node. The penalty is excluded so the criterion measures pure
    predictive accuracy rather than the regularised training objective.

    Parameters
    ----------
    X : ndarray of shape (v, p)
        Validation design matrix.
    y : ndarray of shape (v,)
        Validation response vector.
    beta : ndarray of shape (p,)
        Coefficient vector to evaluate.

    Returns
    -------
    float
        Mean squared prediction error (1/v) * ||y - X*beta||^2.
    """
    resid = y - X @ beta
    return (resid @ resid) / len(y)

# =============================================================================
# SECTION 3: HYPERPARAMETER PROTOCOL
# Each node independently selects its own lambda_j by searching a log grid
# and minimising its own validation MSE. lambda_j is frozen for all rounds.
# Grid: 20 values log-spaced from 1.0 down to 0.001.
# We run 100 CD epochs per candidate to get a near-converged estimate.
# =============================================================================

LAMBDA_GRID = np.logspace(0, -3, 20)

def tune_lambda(node_data, lam_grid=LAMBDA_GRID, n_epochs=100):
    """
    Select the regularization parameter lambda for one node via grid search.

    For each candidate lambda in lam_grid, runs n_epochs of coordinate
    descent from beta=0 on the node's training data, then evaluates the
    resulting model on the node's validation set by MSE. Returns the lambda
    with the lowest validation MSE. The selected lambda is frozen for all
    subsequent communication rounds.

    Parameters
    ----------
    node_data : dict
        Node data dictionary with keys 'X_train', 'y_train', 'X_val',
        'y_val', and 'm', as returned by load_node_data.
    lam_grid : ndarray, optional
        1-D array of candidate lambda values. Default is LAMBDA_GRID
        (20 values log-spaced from 1.0 to 0.001).
    n_epochs : int, optional
        Number of CD epochs per candidate to approximate convergence.
        Default is 100.

    Returns
    -------
    best_lam : float
        The lambda value achieving the lowest validation MSE for this node.
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

print("\n--- Tuning local lambda_j for each node ---")
lambda_j = {}
for label in ["A", "B", "C"]:
    lambda_j[label] = tune_lambda(nodes[label])
    print(f"  Node {label}: lambda = {lambda_j[label]:.6f}")

# Pooled lambda tuned here too (used in Section 5)
X_pool_tr = np.vstack([nodes[l]["X_train"] for l in ["A","B","C"]])
y_pool_tr = np.concatenate([nodes[l]["y_train"] for l in ["A","B","C"]])
X_pool_v  = np.vstack([nodes[l]["X_val"]   for l in ["A","B","C"]])
y_pool_v  = np.concatenate([nodes[l]["y_val"]   for l in ["A","B","C"]])
pool_node = {"X_train": X_pool_tr, "y_train": y_pool_tr,
             "X_val":   X_pool_v,  "y_val":   y_pool_v, "m": X_pool_tr.shape[0]}

lambda_pool = tune_lambda(pool_node)
print(f"  Pooled lambda = {lambda_pool:.6f}")

print(f"\nOptimal (lambda_A, lambda_B, lambda_C) = ({lambda_j['A']:.6f}, {lambda_j['B']:.6f}, {lambda_j['C']:.6f})")

# =============================================================================
# SECTION 4: THE SYNCHRONOUS GLOBAL LOOP
# Round t:
#   1. Broadcast current global beta_t to all nodes.
#   2. Each node runs E local CD epochs starting from beta_t using its own lambda_j.
#   3. Server computes weighted average: beta_{t+1} = sum w_j · beta_{j,t}
#   4. Log the weighted training loss; check ||beta_{t+1} - beta_t||2 < 1e-5.
# The server never accesses raw X or y data.
# =============================================================================

TOLERANCE  = 1e-5
MAX_ROUNDS = 2000

def weighted_train_loss(nodes, beta_dict, lambda_j, weights):
    """
    Compute the global weighted training loss across all nodes.

    Aggregates each node's local Lasso training loss into a single scalar
    used to track convergence across communication rounds. The server uses
    beta vectors returned by each node — raw data is never accessed here.

    Parameters
    ----------
    nodes : dict
        Maps node labels ('A', 'B', 'C') to node data dicts containing
        'X_train' and 'y_train'.
    beta_dict : dict
        Maps node labels to each node's updated local coefficient vector
        of shape (p,) after the current round's E epochs.
    lambda_j : dict
        Maps node labels to each node's frozen lambda value.
    weights : dict
        Maps node labels to aggregation weights w_j, proportional to
        training sample sizes and summing to 1.

    Returns
    -------
    float
        Weighted sum of local Lasso objectives: sum_j w_j * L_j(beta_j).
    """
    return sum(weights[l] * lasso_train_loss(
        nodes[l]["X_train"], nodes[l]["y_train"], beta_dict[l], lambda_j[l])
               for l in nodes)

def run_federated(nodes, lambda_j, weights, E=1, tol=TOLERANCE, max_rounds=MAX_ROUNDS):
    """
    Run the synchronous central-aggregator federated Lasso algorithm.

    Each communication round t proceeds as follows: the server broadcasts
    the current global beta_t to all nodes; each node runs E local CD epochs
    from beta_t using its own frozen lambda_j; the server aggregates via
    weighted average to produce beta_{t+1}; the server logs the weighted
    training loss and checks ||beta_{t+1} - beta_t||_2 < tol. The server
    never accesses raw node data — only coefficient vectors are exchanged.

    Parameters
    ----------
    nodes : dict
        Maps node labels ('A', 'B', 'C') to node data dicts containing
        'X_train' and 'y_train'.
    lambda_j : dict
        Maps node labels to each node's frozen lambda value, as selected
        by tune_lambda prior to calling this function.
    weights : dict
        Maps node labels to aggregation weights w_j, proportional to
        training sample sizes and summing to 1.
    E : int, optional
        Number of local CD epochs each node runs per communication round.
        Default is 1.
    tol : float, optional
        Convergence tolerance on ||beta_{t+1} - beta_t||_2. Default is 1e-5.
    max_rounds : int, optional
        Hard cap on communication rounds. Default is 2000.

    Returns
    -------
    beta_global : ndarray of shape (p,)
        Final converged global coefficient vector.
    loss_history : list of float
        Weighted training loss recorded at the end of each round.
    n_rounds : int
        Number of rounds until convergence, or max_rounds if not reached.
    """
    beta_global  = np.zeros(P)
    loss_history = []

    for t in range(max_rounds):
        beta_local = {l: lasso_cd_epoch(
                            nodes[l]["X_train"], nodes[l]["y_train"],
                            beta_global, lambda_j[l], n_epochs=E)
                      for l in nodes}

        # Server aggregation (weighted average)
        beta_new = sum(weights[l] * beta_local[l] for l in nodes)

        # Log and check stopping
        loss_history.append(weighted_train_loss(nodes, beta_local, lambda_j, weights))
        delta       = np.linalg.norm(beta_new - beta_global)
        beta_global = beta_new

        if delta < tol:
            print(f"  E={E}: converged at round {t+1}  (delta={delta:.2e})")
            return beta_global, loss_history, t + 1

    print(f"  E={E}: reached max_rounds={max_rounds}")
    return beta_global, loss_history, max_rounds

print("\n--- Running federated algorithm for E = 1, 5, 10 ---")
results = {}
for E in [1, 5, 10]:
    beta_g, loss_hist, n_rounds = run_federated(nodes, lambda_j, WEIGHTS, E=E)
    results[E] = {"beta": beta_g, "loss": loss_hist, "rounds": n_rounds}

# =============================================================================
# SECTION 5: THE POOLED BASELINE
# Combine all three training sets into one pooled dataset.
# Tune lambda on the pooled validation set (same grid).
# Run CD to convergence (200 epochs). Treat its active set as ground truth.
# This code is kept strictly separate from the federated pipeline.
# =============================================================================

print("\n--- Fitting centralized pooled Lasso baseline ---")
beta_pool = lasso_cd_epoch(X_pool_tr, y_pool_tr, np.zeros(P), lambda_pool, n_epochs=200)
print(f"  lambda_pool={lambda_pool:.6f}, active={np.sum(np.abs(beta_pool) > 1e-4)}")

# =============================================================================
# SECTION 6: EVALUATION
# Active set threshold ε = 1e-4 (used consistently across all models).
# Confusion matrix: federated active set vs. pooled baseline active set.
# Sensitivity analysis table for E = 1, 5, 10.
# =============================================================================

EPSILON = 1e-4

def active_set(beta, eps=EPSILON):
    """
    Return a boolean mask of active (non-zero) coefficients.

    A coefficient beta_k is considered active if |beta_k| > eps. The
    threshold accounts for floating-point near-zeros produced by
    soft-thresholding and is applied consistently across all models.

    Parameters
    ----------
    beta : ndarray of shape (p,)
        Coefficient vector to threshold.
    eps : float, optional
        Activity threshold. Default is EPSILON (1e-4).

    Returns
    -------
    ndarray of shape (p,), dtype bool
        Boolean mask where True indicates an active coefficient.
    """
    return np.abs(beta) > eps

def conf_matrix(pred_active, true_active):
    """
    Compute a confusion matrix comparing two binary active sets.

    Treats the pooled baseline's active set as ground truth (positive class)
    and the federated model's active set as predictions. Each of the p=600
    coefficients is classified into one of four cells.

    Parameters
    ----------
    pred_active : ndarray of shape (p,), dtype bool
        Boolean mask of active coefficients from the federated model.
    true_active : ndarray of shape (p,), dtype bool
        Boolean mask of active coefficients from the pooled baseline,
        treated as ground truth.

    Returns
    -------
    TP : int
        True positives — active in both federated and pooled.
    FP : int
        False positives — active in federated but inactive in pooled.
    FN : int
        False negatives — inactive in federated but active in pooled.
    TN : int
        True negatives — inactive in both federated and pooled.
    """
    TP = int(np.sum( pred_active &  true_active))
    FP = int(np.sum( pred_active & ~true_active))
    FN = int(np.sum(~pred_active &  true_active))
    TN = int(np.sum(~pred_active & ~true_active))
    return TP, FP, FN, TN

true_active = active_set(beta_pool)
print(f"\nPooled baseline: {true_active.sum()} active coefficients")

print("\n--- Sensitivity Analysis Table ---")
print(f"{'E':>4} | {'Rounds':>7} | {'Final Loss':>12} | {'#Active':>7} | {'TN':>5} | {'FP':>5} | {'FN':>5} | {'TP':>5}")
print("-" * 70)

summary = {}
for E in [1, 5, 10]:
    beta_g    = results[E]["beta"]
    loss_hist = results[E]["loss"]
    final_loss = loss_hist[-1]
    pred_act  = active_set(beta_g)
    n_active  = int(pred_act.sum())
    TP, FP, FN, TN = conf_matrix(pred_act, true_active)
    summary[E] = dict(rounds=results[E]["rounds"], final_loss=final_loss,
                      n_active=n_active, TP=TP, FP=FP, FN=FN, TN=TN)
    print(f"{E:>4} | {results[E]['rounds']:>7} | {final_loss:>12.6f} | {n_active:>7} | {TN:>5} | {FP:>5} | {FN:>5} | {TP:>5}")

# =============================================================================
# SECTION 7: PLOTS
# (a) Overlay loss trajectories for E = 1, 5, 10
# (b) Confusion-matrix heatmaps for each E
# =============================================================================

COLORS = {1: "#1f77b4", 5: "#ff7f0e", 10: "#2ca02c"}

# --- Plot 1: Loss trajectories ---
fig, ax = plt.subplots(figsize=(8, 5))
for E in [1, 5, 10]:
    loss = results[E]["loss"]
    ax.plot(range(1, len(loss)+1), loss,
            label=f"E={E}  ({results[E]['rounds']} rounds)",
            color=COLORS[E], linewidth=2)
ax.set_xlabel("Communication Round", fontsize=12)
ax.set_ylabel("Weighted Training Loss", fontsize=12)
ax.set_title("Global Training Loss Trajectory  (E = 1, 5, 10)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Plot 2: Confusion-matrix heatmaps ---
fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
for idx, E in enumerate([1, 5, 10]):
    s  = summary[E]
    cm = np.array([[s["TP"], s["FP"]], [s["FN"], s["TN"]]])
    ax = axes2[idx]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred +", "Pred -"], fontsize=11)
    ax.set_yticklabels(["True +", "True -"], fontsize=11)
    ax.set_title(f"E={E}  ({s['rounds']} rounds)\n{s['n_active']} active coefs", fontsize=11)
    for i, j in product(range(2), range(2)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=14, color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.colorbar(im, ax=ax, shrink=0.7)
fig2.suptitle("Confusion Matrices: Federated vs. Pooled Lasso (Ground Truth)", fontsize=13)
plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 8: FINAL SUMMARY PRINTOUT
# =============================================================================

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"\n1. Optimal local regularization parameters:")
print(f"   lambda_A = {lambda_j['A']:.6f}  (Node A, m=400, w=0.50)")
print(f"   lambda_B = {lambda_j['B']:.6f}  (Node B, m=240, w=0.30)")
print(f"   lambda_C = {lambda_j['C']:.6f}  (Node C, m=160, w=0.20)")
print(f"   lambda_pool = {lambda_pool:.6f}  (Pooled baseline)")
print(f"\n2. Loss trajectories: see plots above")
for E in [1, 5, 10]:
    s = summary[E]
    print(f"\nE={E}:")
    print(f"   Rounds to converge : {s['rounds']}")
    print(f"   Final training loss: {s['final_loss']:.6f}")
    print(f"   Active coefficients: {s['n_active']}")
    print(f"   TP={s['TP']}, FP={s['FP']}, FN={s['FN']}, TN={s['TN']}")
print(f"\nPooled baseline active: {true_active.sum()}")
print("\nDone.")