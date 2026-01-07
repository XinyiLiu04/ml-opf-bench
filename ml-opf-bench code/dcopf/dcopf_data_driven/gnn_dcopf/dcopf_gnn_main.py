# -*- coding: utf-8 -*-
"""
GNN for DCOPF - Main Experiment Script
Graph Neural Network Supervised Learning Approach

Version: v4.1 - API_TEST mode with dynamic test graph

Features:
- Graph structure utilizing power system topology
- Message passing on all nodes (including Slack)
- Predicts only non-Slack generators
- Slack determined by power balance
- Auto-identify and handle Slack Bus
- Support API_TEST mode with dynamic test graph and constraints

Key Improvements in v4.1:
-------------------------
- API_TEST mode builds separate test graph from test_params
- Model forward accepts dynamic params for node features
- Evaluation uses test graph structure and test constraints
- True generalization: GNN adapts to different topologies and constraints

Evaluation Metrics (6 metrics aligned with DNN):
- MAE Pg (%) - Non-Slack, Slack
- Pg Violation (p.u., Mean of Max) - Non-Slack, Slack
- Branch Violation (p.u., Mean of Max)
- Cost Gap (%)
- Training Time (s)
- Inference Time (ms)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys

sys.path.append(os.path.dirname(__file__))

from dcopf_data_setup import (
    load_parameters_from_csv,
    DataSplitMode,
    split_data_by_mode
)

from dcopf_slack_utils import (
    identify_slack_bus_and_gens,
    update_params_with_slack_info
)

from gnn_model_dcopf import DCOPF_GNN
from gnn_utils_dcopf import build_graph_from_real_topology, evaluate_split


# =====================================================================
# Data Preparation Function
# =====================================================================
def load_and_prepare_gnn_data(file_path, params, column_names):
    """
    Load and prepare GNN training data

    Returns:
        x_data_raw: Load data [n_samples, n_buses] p.u.
        y_pg_raw_non_slack: Non-Slack generation [n_samples, n_g_non_slack] p.u.
        y_pg_raw_all: Full generation [n_samples, n_g] p.u.
    """
    import pandas as pd

    full_df = pd.read_csv(file_path)
    n_samples = len(full_df)
    n_buses = params['general']['n_buses']

    load_prefix = column_names['load_prefix']
    load_cols = [col for col in full_df.columns if col.startswith(load_prefix)]

    load_bus_ids = []
    for col in load_cols:
        bus_id = int(col[len(load_prefix):])
        load_bus_ids.append(bus_id)

    x_data_raw_loads = full_df[load_cols].values.astype('float32')

    x_data_raw = np.zeros((n_samples, n_buses), dtype='float32')
    for i, bus_id in enumerate(load_bus_ids):
        if bus_id <= n_buses:
            x_data_raw[:, bus_id - 1] = x_data_raw_loads[:, i]

    pg_cols = [col for col in full_df.columns if col.startswith(column_names['gen_prefix'])]
    y_pg_raw_all = full_df[pg_cols].values.astype('float32')

    non_slack_indices = params['general']['non_slack_gen_indices']
    y_pg_raw_non_slack = y_pg_raw_all[:, non_slack_indices]

    return x_data_raw, y_pg_raw_non_slack, y_pg_raw_all


# =====================================================================
# Main Experiment Function
# =====================================================================
def gnn_dcopf_experiment(
        case_name,
        params_path,
        dataset_path,
        column_names,
        n_train_use=10000,
        seed=42,
        n_epochs=100,
        learning_rate=0.001,
        hidden_dim=64,
        num_gnn_layers=4,
        batch_size=256,
        device='cuda',
        split_mode=DataSplitMode.RANDOM_SPLIT,
        test_data_path=None,
        test_params_path=None,
        n_test_samples=1000
):
    """GNN for DCOPF main experiment function (v4.1 - API_TEST support)"""

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ========== 1. Load training parameters and identify Slack ==========
    print("\n" + "=" * 70)
    print("Loading Training Parameters")
    print("=" * 70)

    params = load_parameters_from_csv(case_name, params_path, is_api=False)
    slack_info = identify_slack_bus_and_gens(params)
    params = update_params_with_slack_info(params, slack_info)

    # ========== 2. Load test parameters (API_TEST mode) ==========
    test_params = None
    if split_mode == DataSplitMode.API_TEST:
        if test_params_path is None:
            raise ValueError('API_TEST mode requires test_params_path')

        print("\n" + "=" * 70)
        print("API_TEST Mode: Loading Test Parameters")
        print("=" * 70)

        # Load API params
        test_params = load_parameters_from_csv(case_name, test_params_path, is_api=True)
        test_slack_info = identify_slack_bus_and_gens(test_params)
        test_params = update_params_with_slack_info(test_params, test_slack_info)

        # Print constraint changes
        print(f"\nConstraint Comparison (Training vs API):")
        pg_min_train = params['constraints']['Pg_min'].ravel()
        pg_max_train = params['constraints']['Pg_max'].ravel()
        pg_min_api = test_params['constraints']['Pg_min'].ravel()
        pg_max_api = test_params['constraints']['Pg_max'].ravel()

        pg_min_change = np.mean(np.abs(pg_min_api - pg_min_train) / (np.abs(pg_min_train) + 1e-8)) * 100
        pg_max_change = np.mean(np.abs(pg_max_api - pg_max_train) / (np.abs(pg_max_train) + 1e-8)) * 100

        print(f"  Pg_min avg change: {pg_min_change:.2f}%")
        print(f"  Pg_max avg change: {pg_max_change:.2f}%")
    else:
        test_params = params

    # ========== 3. Build training graph structure ==========
    print("\n" + "=" * 70)
    print("Building Training Graph Structure")
    print("=" * 70)

    edges, edge_weights, node_types = build_graph_from_real_topology(
        params, params_path, case_name, is_api=False
    )

    print(f"  Training graph edges: {edges.shape[1]}")

    # ========== 4. Build test graph structure (API_TEST mode) ==========
    test_edges = None
    test_edge_weights = None

    if split_mode == DataSplitMode.API_TEST:
        print("\n" + "=" * 70)
        print("Building Test Graph Structure (API)")
        print("=" * 70)

        test_edges, test_edge_weights, _ = build_graph_from_real_topology(
            test_params, test_params_path, case_name, is_api=True
        )

        print(f"  Test graph edges: {test_edges.shape[1]}")
        print(f"  Graph structure change: {abs(test_edges.shape[1] - edges.shape[1])} edges")

        # Compare edge weights
        if test_edges.shape[1] == edges.shape[1]:
            # Only compare if same number of edges (same topology)
            weight_diff = torch.abs(test_edge_weights - edge_weights).mean()
            print(f"  Avg edge weight change: {weight_diff.item():.6f}")
        else:
            print(f"  Warning: Topology changed! Different number of edges")

    # ========== 5. Load training data ==========
    print("\n" + "=" * 70)
    print("Loading Training Data")
    print("=" * 70)

    x_data_raw, y_pg_raw_non_slack, y_pg_raw_all = load_and_prepare_gnn_data(
        dataset_path, params, column_names
    )

    n_gen = params['general']['n_g']
    n_g_non_slack = params['general']['n_g_non_slack']
    n_buses = params['general']['n_buses']

    print(f"  Total samples: {len(x_data_raw)}")
    print(f"  Input dim: {n_buses} (buses)")
    print(f"  Output dim: {n_g_non_slack} (non-Slack generators)")

    raw_data = {
        'x': x_data_raw,
        'y_pg_non_slack': y_pg_raw_non_slack,
        'y_pg_all': y_pg_raw_all
    }

    # ========== 6. Data split ==========
    train_idx, val_idx, test_idx, x_test_external, y_test_external = split_data_by_mode(
        x_data_raw=x_data_raw,
        y_pg_raw=y_pg_raw_all,
        mode=split_mode,
        n_train_use=n_train_use,
        seed=seed,
        test_data_path=test_data_path,
        params=params,
        column_names=column_names,
        n_test_samples=n_test_samples
    )

    # ========== 7. Data normalization ==========
    from sklearn.preprocessing import MinMaxScaler

    x_scaler = MinMaxScaler().fit(x_data_raw[train_idx])
    y_pg_non_slack_scaler = MinMaxScaler().fit(y_pg_raw_non_slack[train_idx])
    scalers = {
        'x': x_scaler,
        'y_pg_non_slack': y_pg_non_slack_scaler
    }

    x_train_scaled = x_scaler.transform(x_data_raw[train_idx])
    y_train_scaled = y_pg_non_slack_scaler.transform(y_pg_raw_non_slack[train_idx])
    x_val_scaled = x_scaler.transform(x_data_raw[val_idx])
    y_val_scaled = y_pg_non_slack_scaler.transform(y_pg_raw_non_slack[val_idx])

    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        X_test = None
    else:
        x_test_scaled = x_scaler.transform(x_data_raw[test_idx])
        X_test = torch.tensor(x_test_scaled, dtype=torch.float32)

    X_train = torch.tensor(x_train_scaled, dtype=torch.float32)
    Y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_val = torch.tensor(x_val_scaled, dtype=torch.float32)
    Y_val = torch.tensor(y_val_scaled, dtype=torch.float32)

    # ========== 8. Create model ==========
    print("\n" + "=" * 70)
    print("Creating GNN Model")
    print("=" * 70)

    model = DCOPF_GNN(
        n_buses=n_buses,
        n_gen_non_slack=n_g_non_slack,
        hidden_dim=hidden_dim,
        num_gnn_layers=num_gnn_layers,
        edge_dim=2,
        params=params
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params:,}")

    # ========== 9. Training ==========
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    t0 = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        n_train = len(X_train)
        n_batches = (n_train + batch_size - 1) // batch_size
        indices = torch.randperm(n_train)

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            batch_indices = indices[start_idx:end_idx]

            X_batch = X_train[batch_indices].to(device)
            Y_batch = Y_train[batch_indices].to(device)

            optimizer.zero_grad()
            pred = model(X_batch, edges.to(device), edge_weights.to(device))
            loss = criterion(pred, Y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(X_batch)

        train_loss = epoch_loss / n_train

        # Validation
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val.to(device), edges.to(device), edge_weights.to(device))
            val_loss = float(criterion(pred_val, Y_val.to(device)).item())

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print every epoch
        print(f"Epoch {epoch}/{n_epochs} - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

    train_time = time.perf_counter() - t0

    print(f"\nTraining completed in {train_time:.2f}s")

    # ========== 10. Evaluation (test set only) ==========
    print("\n" + "=" * 70)
    print("Evaluation Phase")
    print("=" * 70)

    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        test_data_external_dict = {
            'x': x_test_external,
            'y_pg_all': y_test_external
        }

        # Pass test graph structure and test params
        test_metrics = evaluate_split(
            model, None, None, raw_data,
            params, scalers,
            edges, edge_weights, device,
            test_data_external=test_data_external_dict,
            test_params=test_params,
            test_edges=test_edges,  # API graph
            test_edge_weights=test_edge_weights  # API edge weights
        )

        if split_mode == DataSplitMode.API_TEST:
            print("\nAPI_TEST mode active:")
            print("  - Using API graph structure")
            print("  - Using API constraint parameters")
            print("  - GNN message passing adapts to new topology!")
    else:
        test_metrics = evaluate_split(
            model, X_test, test_idx, raw_data,
            params, scalers,
            edges, edge_weights, device,
            test_params=test_params
        )

    # ========== 11. Speed test ==========
    model.eval()

    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        test_sample = torch.tensor(
            x_scaler.transform(x_test_external[:1]),
            dtype=torch.float32,
            device=device
        )
        # Use test graph for speed test in API mode
        speed_edges = test_edges if test_edges is not None else edges
        speed_weights = test_edge_weights if test_edge_weights is not None else edge_weights
    else:
        test_sample = X_test[:1].to(device)
        speed_edges = edges
        speed_weights = edge_weights

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_sample, speed_edges.to(device), speed_weights.to(device))
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Measure inference time
    n_repeats = 100
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            t_start = time.perf_counter()
            _ = model(test_sample, speed_edges.to(device), speed_weights.to(device))
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t_start)

    inference_time_ms = np.mean(times) * 1000

    # ========== 12. Print results ==========
    print("\n" + "=" * 70)
    print("Test Set Results")
    print("=" * 70)
    print(f"\nNon-Slack Generators:")
    print(f"  MAE:        {test_metrics['mae_pg_non_slack']:.4f}%")
    print(f"  Violation:  {test_metrics['viol_pg_non_slack']:.4f} p.u.")
    print(f"\nSlack-Only Generators:")
    print(f"  MAE:        {test_metrics['mae_pg_slack']:.4f}%")
    print(f"  Violation:  {test_metrics['viol_pg_slack']:.4f} p.u.")
    print(f"\nBranch:")
    print(f"  Violation:  {test_metrics['viol_branch']:.4f} p.u.")
    print(f"\nCost Gap:     {test_metrics['cost_gap_percent']:.4f}%")
    print(f"\nTraining Time:   {train_time:.2f} s")
    print(f"Inference Time:  {inference_time_ms:.4f} ms")
    print("\n" + "=" * 70 + "\n")


# =====================================================================
# Main Program
# =====================================================================
if __name__ == "__main__":
    # ===================================================================
    # Experiment Configuration
    # ===================================================================

    # --- 1. Case Configuration ---
    CASE_NAME = 'pglib_opf_case118_ieee'
    CASE_SHORT_NAME = 'case118'

    # --- 2. Data Split Mode ---
    SPLIT_MODE = DataSplitMode.API_TEST

    # --- 3. Training & Test Sample Counts ---
    N_TRAIN_USE = 10000
    N_TEST_SAMPLES = 2483

    # --- 4. GNN Hyperparameters ---
    N_EPOCHS = 100
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    HIDDEN_DIM = 128
    NUM_GNN_LAYERS = 2
    SEED = 42

    # --- 5. Path Configuration ---
    ROOT_DIR = r"C:\Users\Aloha\Desktop\dataset"
    TRAIN_VARIANCE = "v=0.12"
    TEST_VARIANCE = "v=0.25"

    COLUMN_NAMES = {
        'load_prefix': 'pd',
        'gen_prefix': 'pg',
        'lambda': 'lambda',
        'mu_g_min_prefix': 'mu_g_min_',
        'mu_g_max_prefix': 'mu_g_max_',
        'mu_line_pos_prefix': 'mu_line_max_',
        'mu_line_neg_prefix': 'mu_line_min_',
    }

    # ===================================================================
    # Path Generation
    # ===================================================================

    params_path = os.path.join(ROOT_DIR, "DCOPF Constraints", CASE_SHORT_NAME)
    train_data_path = os.path.join(
        ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}({TRAIN_VARIANCE})",
        f"{CASE_NAME}_dataset_with_duals.csv"
    )

    if SPLIT_MODE == DataSplitMode.GENERALIZATION:
        test_data_path = os.path.join(
            ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}({TEST_VARIANCE})",
            f"{CASE_NAME}_dataset_with_duals.csv"
        )
        test_params_path = None
    elif SPLIT_MODE == DataSplitMode.API_TEST:
        test_data_path = os.path.join(
            ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}(v=api)",
            f"{CASE_NAME}__api_dataset_with_duals.csv"
        )
        test_params_path = os.path.join(
            ROOT_DIR, "DCOPF Constraints", f"{CASE_SHORT_NAME}(api)"
        )
    else:
        test_data_path = None
        test_params_path = None

    # ===================================================================
    # Device Detection
    # ===================================================================
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    # ===================================================================
    # Run Experiment
    # ===================================================================
    gnn_dcopf_experiment(
        case_name=CASE_NAME,
        params_path=params_path,
        dataset_path=train_data_path,
        column_names=COLUMN_NAMES,
        n_train_use=N_TRAIN_USE,
        seed=SEED,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        hidden_dim=HIDDEN_DIM,
        num_gnn_layers=NUM_GNN_LAYERS,
        batch_size=BATCH_SIZE,
        device=device_name,
        split_mode=SPLIT_MODE,
        test_data_path=test_data_path,
        test_params_path=test_params_path,
        n_test_samples=N_TEST_SAMPLES
    )