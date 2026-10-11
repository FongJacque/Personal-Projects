# =============================================================================
# STAT 102B – Final Project
# FILE 2: CORE OPTIMIZATION OPERATORS
# Soft-thresholding and coordinate-descent epochs for the Lasso objective:
#   (1/2m)||y - Xβ||² + λ||β||₁
# One epoch = one full sequential pass over all p=600 coordinates.
# Residual is updated incrementally per coordinate to avoid O(mp) matrix
# multiplications inside the inner loop.
# =============================================================================

import numpy as np

def soft_threshold(rho, lam):
    """
    Soft-thresholding operator: sign(ρ) · max(|ρ| − λ, 0).
    Applied element-wise; λ must already be scaled to match ρ's units.
    """
    return np.sign(rho) * np.maximum(np.abs(rho) - lam, 0.0)

def lasso_cd_epoch(X, y, beta, lam, n_epochs=1):
    """
    Run n_epochs full coordinate-descent passes on the Lasso objective.

    For each coordinate k within an epoch:
        r_k  = y - Xβ + x_k · β_k     (partial residual, O(n) via incremental update)
        ρ_k  = x_k^T r_k               (univariate OLS numerator)
        z_k  = ||x_k||²                (column squared norm, precomputed)
        β_k ← soft(ρ_k, λ · m) / z_k  (soft-threshold then rescale)

    The factor of m absorbs the 1/2m scaling so that ρ_k and the penalty
    threshold live on the same scale.

    Parameters
    ----------
    X        : (m, p) design matrix
    y        : (m,)  response vector
    beta     : (p,)  starting coefficient vector (not modified in place)
    lam      : float regularization parameter
    n_epochs : int   number of full passes over all p coordinates

    Returns
    -------
    beta : (p,) updated coefficient vector
    """
    m, p  = X.shape
    beta  = beta.copy()
    lam_s = lam * m                         # scale λ to match x^T r units
    z     = np.einsum('ij,ij->j', X, X)    # (p,) column squared norms
    resid = y - X @ beta                    # initialise residual once

    for _ in range(n_epochs):
        for k in range(p):
            beta_k_old = beta[k]
            rho_k      = X[:, k] @ resid + z[k] * beta_k_old   # partial resid trick
            beta_k_new = soft_threshold(rho_k, lam_s) / z[k] if z[k] != 0 else 0.0
            beta[k]    = beta_k_new
            resid     += X[:, k] * (beta_k_old - beta_k_new)    # incremental update

    return beta

def lasso_train_loss(X, y, beta, lam):
    """Lasso training objective: (1/2m)||y − Xβ||² + λ||β||₁."""
    m     = len(y)
    resid = y - X @ beta
    return (resid @ resid) / (2 * m) + lam * np.sum(np.abs(beta))

def lasso_val_mse(X, y, beta):
    """Validation MSE (no penalty) — used only for λ selection."""
    resid = y - X @ beta
    return (resid @ resid) / len(y)

if __name__ == "__main__":
    np.random.seed(0)
    X_t = np.random.randn(50, 10)
    y_t = X_t[:, 0] * 2 + np.random.randn(50) * 0.1
    b   = lasso_cd_epoch(X_t, y_t, np.zeros(10), lam=0.01, n_epochs=50)
    print("Smoke-test β (first 3 coords, expect ~[2,0,0]):", b[:3].round(3))
