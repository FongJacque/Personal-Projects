# =============================================================================
# STAT 102B – Final Project
# MAIN RUNNER — imports from all five module files and produces all results.
# Run this file to reproduce every number, plot, and table in the report.
# =============================================================================

import sys, os
sys.path.insert(0, "/home/claude")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product as iproduct

# --- Import from the five structural modules ---
from data_loading             import nodes, WEIGHTS, P
from core_optimization_operators import lasso_cd_epoch, lasso_train_loss, lasso_val_mse
from hyperparameter_protocol  import tune_lambda, LAMBDA_GRID
from synchronous_global_loop  import run_federated, TOLERANCE
from pooled_baseline          import fit_pooled_baseline, pool_node

np.random.seed(42)
OUT = "/mnt/user-data/outputs/"
os.makedirs(OUT, exist_ok=True)

# ── 1. Tune local λ_j ──────────────────────────────────────────────────────
print("Tuning local λ_j ...")
lambda_j = {l: tune_lambda(nodes[l]) for l in ["A","B","C"]}
for l in ["A","B","C"]:
    print(f"  Node {l}: λ = {lambda_j[l]:.6f}")

# ── 2. Fit pooled baseline ──────────────────────────────────────────────────
print("\nFitting pooled baseline ...")
beta_pool, lambda_pool = fit_pooled_baseline(pool_node)
print(f"  λ_pool = {lambda_pool:.6f}")

# ── 3. Run federated algorithm for E = 1, 5, 10 ────────────────────────────
print("\nRunning federated algorithm ...")
results = {}
for E in [1, 5, 10]:
    beta_g, loss_hist, n_rounds = run_federated(nodes, lambda_j, WEIGHTS, E=E)
    results[E] = {"beta": beta_g, "loss": loss_hist, "rounds": n_rounds}

# ── 4. Evaluation ──────────────────────────────────────────────────────────
EPSILON     = 1e-4
true_active = np.abs(beta_pool) > EPSILON
print(f"\nPooled baseline active coefficients: {true_active.sum()}")

def conf_matrix(pred, truth):
    TP = int(np.sum( pred &  truth))
    FP = int(np.sum( pred & ~truth))
    FN = int(np.sum(~pred &  truth))
    TN = int(np.sum(~pred & ~truth))
    return TP, FP, FN, TN

print(f"\n{'E':>4} | {'Rounds':>7} | {'Final Loss':>12} | {'#Active':>7} | {'TN':>5} | {'FP':>5} | {'FN':>5} | {'TP':>5}")
print("-" * 70)

summary = {}
for E in [1, 5, 10]:
    beta_g     = results[E]["beta"]
    pred_act   = np.abs(beta_g) > EPSILON
    n_active   = int(pred_act.sum())
    final_loss = results[E]["loss"][-1]
    TP, FP, FN, TN = conf_matrix(pred_act, true_active)
    summary[E] = dict(rounds=results[E]["rounds"], final_loss=final_loss,
                      n_active=n_active, TP=TP, FP=FP, FN=FN, TN=TN)
    print(f"{E:>4} | {results[E]['rounds']:>7} | {final_loss:>12.6f} | {n_active:>7} | {TN:>5} | {FP:>5} | {FN:>5} | {TP:>5}")

# ── 5. Plots ───────────────────────────────────────────────────────────────
COLORS = {1: "#1f77b4", 5: "#ff7f0e", 10: "#2ca02c"}

# Loss trajectory
fig, ax = plt.subplots(figsize=(8, 5))
for E in [1, 5, 10]:
    loss = results[E]["loss"]
    ax.plot(range(1, len(loss)+1), loss,
            label=f"E={E}  ({results[E]['rounds']} rounds)",
            color=COLORS[E], linewidth=2)
ax.set_xlabel("Communication Round", fontsize=12)
ax.set_ylabel("Weighted Training Loss", fontsize=12)
ax.set_title("Global Training Loss Trajectory  (E = 1, 5, 10)", fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + "loss_trajectory.png", dpi=150, bbox_inches="tight")
plt.close()

# Confusion-matrix heatmaps
fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
for idx, E in enumerate([1, 5, 10]):
    s  = summary[E]
    cm = np.array([[s["TP"], s["FP"]], [s["FN"], s["TN"]]])
    ax = axes2[idx]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred +", "Pred −"], fontsize=11)
    ax.set_yticklabels(["True +", "True −"], fontsize=11)
    ax.set_title(f"E={E}  ({s['rounds']} rounds)\n{s['n_active']} active coefs", fontsize=11)
    for i, j in iproduct(range(2), range(2)):
        ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                fontsize=14, color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.colorbar(im, ax=ax, shrink=0.7)
fig2.suptitle("Confusion Matrices: Federated vs. Pooled Lasso (Ground Truth)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT + "confusion_heatmaps.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved plots to outputs/")

# ── 6. Final summary ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"\n1. Optimal local regularization parameters:")
for l in ["A","B","C"]:
    print(f"   λ_{l} = {lambda_j[l]:.6f}  (Node {l}, m={nodes[l]['m']}, w={WEIGHTS[l]:.2f})")
print(f"   λ_pool = {lambda_pool:.6f}")
print(f"\n2. Loss trajectories: see loss_trajectory.png")
for E in [1, 5, 10]:
    s = summary[E]
    print(f"\nE={E}: rounds={s['rounds']}, final_loss={s['final_loss']:.6f}, "
          f"active={s['n_active']}, TP={s['TP']}, FP={s['FP']}, FN={s['FN']}, TN={s['TN']}")
print(f"\nPooled baseline active: {true_active.sum()}")
print("\nDone.")
