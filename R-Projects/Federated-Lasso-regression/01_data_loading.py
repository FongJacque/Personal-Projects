# =============================================================================
# STAT 102B – Final Project
# FILE 1: DATA LOADING
# Load each node's train/validation data into a dictionary.
# Raw data never leaves node scope — the aggregator only sees β vectors.
# File node3 has 400 train rows (largest), node2 has 240, node1 has 160.
# We relabel so Node A = largest = weight 0.50, etc.
# =============================================================================

import numpy as np
import pandas as pd

DATA_DIR = "/mnt/user-data/uploads/"

def load_node_data(node_id):
    """Load train and validation arrays for a given node file index (1,2,3)."""
    X_train = pd.read_csv(f"{DATA_DIR}node{node_id}_X_train.csv").values.astype(float)
    y_train = pd.read_csv(f"{DATA_DIR}node{node_id}_y_train.csv").values.astype(float).ravel()
    X_val   = pd.read_csv(f"{DATA_DIR}node{node_id}_X_val.csv").values.astype(float)
    y_val   = pd.read_csv(f"{DATA_DIR}node{node_id}_y_val.csv").values.astype(float).ravel()
    return {"X_train": X_train, "y_train": y_train,
            "X_val":   X_val,   "y_val":   y_val,
            "m":       X_train.shape[0]}

raw_data = {j: load_node_data(j) for j in [1, 2, 3]}

# Sort by descending training size → A=400, B=240, C=160
sorted_keys = sorted(raw_data, key=lambda j: raw_data[j]["m"], reverse=True)
nodes = {}
for label, key in zip(["A", "B", "C"], sorted_keys):
    nodes[label] = raw_data[key]

# Weights proportional to training sizes (0.50, 0.30, 0.20)
total_m = sum(d["m"] for d in nodes.values())
WEIGHTS = {label: nodes[label]["m"] / total_m for label in nodes}

P = 600  # number of predictors

if __name__ == "__main__":
    for label, d in nodes.items():
        print(f"Node {label}: m_train={d['m']}, m_val={d['X_val'].shape[0]}, p={d['X_train'].shape[1]}")
    print(f"Weights: { {k: round(v, 2) for k, v in WEIGHTS.items()} }")
