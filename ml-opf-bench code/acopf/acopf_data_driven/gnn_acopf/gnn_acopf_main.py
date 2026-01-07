# -*- coding: utf-8 -*-
"""
GNN for ACOPF - Main Experiment Script (V8 - Simplified Output)

Graph neural network-based supervised learning approach

Modifications (V8):
- Only predicts non-slack pg and generator vm
- Simplified output: only test set metrics
- All comments and outputs in English
- Removed all file saving (JSON/CSV/model)
- All violations in p.u. units
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys

# Import configuration
try:
    import acopf_config
except ImportError:
    print("Error: Unable to import acopf_config.py")
    print("Please ensure acopf_config.py is in the same directory")
    sys.exit(1)

# Import data modules
try:
    from acopf_data_setup import (
        load_parameters_from_csv,
        load_and_scale_acopf_data,
        DataMode,
        prepare_data_splits,
        load_generalization_test_data,
        load_api_test_data
    )
except ImportError:
    print("Error: Unable to import 'acopf_data_setup' module.")
    sys.exit(1)

# Import GNN modules
try:
    import gnn_utils
    from gnn_model import ACOPF_GNN
    from gnn_utils import (
        build_graph_structure,
        init_pypower_options,
        load_case_from_csv,
        evaluate_split
    )
except ImportError:
    print("Error: Unable to import GNN-related modules.")
    sys.exit(1)


# =====================================================================
# Main Experiment Function
# =====================================================================
def gnn_acopf_experiment(
        case_name,
        params_path,
        data_path,
        log_path,
        results_path,
        # Data mode parameters
        data_mode='random_split',
        n_train_use=None,
        test_data_path=None,
        test_params_path=None,
        n_test_samples=None,
        # Training parameters
        seed=42,
        n_epochs=1000,
        learning_rate=0.001,
        hidden_dim=64,
        num_gnn_layers=4,
        batch_size=256,
        device='cuda',
        **kwargs  # Accept other unused parameters
):
    """
    GNN-based ACOPF main experiment function

    Supports four data modes:
    - RANDOM_SPLIT: Random split
    - FIXED_VALTEST: Fixed validation/test sets
    - GENERALIZATION: Cross-distribution generalization test
    - API_TEST: API data test (different topology)
    """
    # ========== 1. Initialization ==========
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print(f"GNN-based ACOPF Experiment")
    print(f"{'=' * 70}")
    print(f"Case: {case_name}")
    print(f"Data Mode: {data_mode}")
    print(f"Device: {device}")
    print(f"{'=' * 70}")

    # ========================================================================
    # 2. Load training parameters and PyPower case data
    # ========================================================================
    print(f"\n[Step 1] Loading training parameters and case data...")
    params = load_parameters_from_csv(case_name, params_path)

    # Verify bus numbering
    print(f"\n[Verification] Bus numbering info:")
    bus_ids = params['general']['bus_ids']
    print(f"  Bus ID range: [{bus_ids.min()}, {bus_ids.max()}]")
    print(f"  Bus count: {len(bus_ids)}")

    # Build graph structure (using training params)
    edges, edge_weights, node_types = build_graph_structure(params)

    # Initialize PyPower
    init_pypower_options()
    gnn_utils.GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)
    print(f"  ✓ Training params and PyPower case data loaded")

    # ========================================================================
    # 3. Load training data and fit scalers
    # ========================================================================
    print(f"\n[Step 2] Loading training data...")
    x_data_scaled, y_data_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True)

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    baseMVA = params['general']['BASE_MVA']

    print(f"\n[Training Data Info]")
    print(
        f"  Buses: {n_buses}, Generators: {n_gen} (Non-Slack: {n_gen_non_slack}), Loads: {n_loads}, Base MVA: {baseMVA}")
    print(f"  Graph edges: {edges.shape[1]}")
    if cost_baseline:
        print(f"  Cost Baseline: {cost_baseline:.2f} $/h")

    # ========================================================================
    # 4. Load and split data based on data mode
    # ========================================================================
    if data_mode == DataMode.API_TEST:
        print(f"\n{'=' * 60}")
        print(f"Data Mode: API_TEST")
        print(f"{'=' * 60}")

        if test_data_path is None or test_params_path is None:
            raise ValueError("API_TEST mode requires test_data_path and test_params_path")

        train_idx, val_idx, _ = prepare_data_splits(
            x_data_scaled, y_data_scaled,
            mode=DataMode.API_TEST,
            n_train_use=n_train_use,
            seed=seed
        )

        test_params, test_x_scaled, test_y_scaled, test_raw_data, _ = \
            load_api_test_data(
                test_data_path,
                test_params_path,
                scalers,
                n_test_samples=n_test_samples or 1000,
                seed=seed
            )

        test_idx = np.arange(len(test_x_scaled))

        # Build API test graph structure (using API params)
        test_edges, test_edge_weights, test_node_types = build_graph_structure(test_params)

        # Load API PyPower case data
        test_case_name = os.path.basename(test_data_path)
        if test_case_name.endswith('_pd.csv'):
            test_case_name = test_case_name[:-7]

        GLOBAL_CASE_DATA_TEST = load_case_from_csv(test_case_name, test_params_path)
        print(f"  ✓ API test PyPower case data loaded")

        print(f"\n[API Test Data Info]")
        print(f"  Buses: {test_params['general']['n_buses']}")
        print(
            f"  Generators: {test_params['general']['n_gen']} (Non-Slack: {test_params['general']['n_gen_non_slack']})")
        print(f"  Loads: {test_params['general']['n_loads']}")
        print(f"  Base MVA: {test_params['general']['BASE_MVA']}")
        print(f"  Graph edges: {test_edges.shape[1]}")

    elif data_mode == DataMode.GENERALIZATION:
        print(f"\n{'=' * 60}")
        print(f"Data Mode: GENERALIZATION")
        print(f"{'=' * 60}")

        if test_data_path is None:
            raise ValueError("GENERALIZATION mode requires test_data_path")

        train_idx, val_idx, _ = prepare_data_splits(
            x_data_scaled, y_data_scaled,
            mode=DataMode.GENERALIZATION,
            n_train_use=n_train_use,
            seed=seed
        )

        test_x_scaled, test_y_scaled, test_raw_data, _ = \
            load_generalization_test_data(
                test_data_path,
                params,
                scalers,
                n_test_samples=n_test_samples or 1000,
                seed=seed
            )

        test_idx = np.arange(len(test_x_scaled))
        test_params = params
        test_edges = edges
        test_edge_weights = edge_weights
        GLOBAL_CASE_DATA_TEST = gnn_utils.GLOBAL_CASE_DATA

    else:
        print(f"\n{'=' * 60}")
        print(f"Data Mode: {data_mode}")
        print(f"{'=' * 60}")

        train_idx, val_idx, test_idx = prepare_data_splits(
            x_data_scaled, y_data_scaled,
            mode=data_mode,
            n_train_use=n_train_use,
            seed=seed
        )

        test_x_scaled, test_y_scaled, test_raw_data = x_data_scaled, y_data_scaled, raw_data
        test_params = params
        test_edges = edges
        test_edge_weights = edge_weights
        GLOBAL_CASE_DATA_TEST = gnn_utils.GLOBAL_CASE_DATA

    # ========================================================================
    # 5. Prepare training data
    # ========================================================================
    print(f"\n[Dataset Sizes]")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")

    X_train = torch.tensor(x_data_scaled[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(y_data_scaled[train_idx], dtype=torch.float32)
    X_val = torch.tensor(x_data_scaled[val_idx], dtype=torch.float32)
    Y_val = torch.tensor(y_data_scaled[val_idx], dtype=torch.float32)
    X_test = torch.tensor(test_x_scaled[test_idx], dtype=torch.float32)

    # ========================================================================
    # 6. Create model and train
    # ========================================================================
    print(f"\n[Step 3] Creating GNN model...")

    model = ACOPF_GNN(
        n_buses=n_buses,
        n_gen=n_gen,
        n_gen_non_slack=n_gen_non_slack,
        n_loads=n_loads,
        hidden_dim=hidden_dim,
        num_gnn_layers=num_gnn_layers,
        edge_dim=2,
        params=params
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Network: Input({2 * n_loads}) -> GNN({hidden_dim}x{num_gnn_layers}) -> Output({n_gen_non_slack + n_gen})")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    print(f"\n[Step 4] Training...")
    print(f"  Epochs: {n_epochs}, Learning rate: {learning_rate}, Batch size: {batch_size}")

    n_train = len(X_train)
    n_batches = (n_train + batch_size - 1) // batch_size
    train_losses, val_losses = [], []

    t0 = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
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

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val.to(device), edges.to(device), edge_weights.to(device))
            val_loss = float(criterion(pred_val, Y_val.to(device)).item())

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 1 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"Epoch {epoch:4d}/{n_epochs} - train: {train_loss:.6f} - val: {val_loss:.6f}")

    train_time = time.perf_counter() - t0
    print(f"\n✓ Training completed in {train_time:.2f} seconds")

    # ========================================================================
    # 7. Model evaluation (using test params)
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"Test Set Evaluation")
    print(f"{'=' * 70}")

    GLOBAL_CASE_DATA_BACKUP = gnn_utils.GLOBAL_CASE_DATA
    gnn_utils.GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_TEST

    if data_mode == DataMode.API_TEST:
        test_split_name = "API Test"
    elif data_mode == DataMode.GENERALIZATION:
        test_split_name = "Generalization Test"
    else:
        test_split_name = "Test"

    test_metrics = evaluate_split(
        model, X_test, test_idx, test_raw_data,
        test_params,
        scalers,
        test_edges,
        test_edge_weights,
        device, test_split_name, verbose=True
    )

    gnn_utils.GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_BACKUP

    # ========================================================================
    # 8. Inference speed test
    # ========================================================================
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(100):
            t_start = time.perf_counter()
            _ = model(X_test[:1].to(device), test_edges.to(device), test_edge_weights.to(device))
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t_start)

    latency_ms = np.mean(times) * 1000

    # ========================================================================
    # 9. Print final results (simplified)
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"Final Results Summary")
    print(f"{'=' * 70}")

    print(f"\nData Mode: {data_mode}")
    print(f"Test Case: {case_name}")

    print(f"\n--- Accuracy Metrics ---")
    print(f"MAE_Pg (Non-Slack): {test_metrics['mae_pg_non_slack_percent']:.4f}%")
    print(f"MAE_Vm (Generator): {test_metrics['mae_vm_percent']:.4f}%")
    print(f"MAE_Qg (All Gens):  {test_metrics['mae_qg_percent']:.4f}%")
    print(f"MAE_Va (All Buses): {test_metrics['mae_va_deg']:.4f} degrees")

    print(f"\n--- Violations (p.u.) ---")
    print(f"Pg_viol (Non-Slack): {test_metrics['mean_pg_viol_non_slack_pu']:.6f} p.u.")
    print(f"Pg_viol (Slack):     {test_metrics['mean_pg_viol_slack_pu']:.6f} p.u.")
    print(f"Qg_viol (All Gens):  {test_metrics['mean_max_qg_viol_pu']:.6f} p.u.")
    print(f"Vm_viol (All Buses): {test_metrics['mean_max_vm_viol_pu']:.6f} p.u.")
    print(f"Branch_viol:         {test_metrics['mean_max_branch_viol_pu']:.6f} p.u. (1.0 = 100% overload)")

    print(f"\n--- Cost Metrics ---")
    print(f"Cost Gap: {test_metrics['cost_optimality_gap_percent']:.4f}%")

    print(f"\n--- Performance ---")
    print(f"Inference Time: {latency_ms:.4f} ms/sample")
    print(f"Training Time:  {train_time:.2f} s")
    print(f"Convergence Rate: {test_metrics['convergence_rate_percent']:.2f}%")

    print(f"{'=' * 70}")

    # No file saving - removed all JSON/CSV/model saving

    return test_metrics


# =====================================================================
# Main Function
# =====================================================================
if __name__ == "__main__":
    # =========================================================================
    # Read configuration from acopf_config.py and run experiment
    # =========================================================================

    # Print configuration
    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)

    # Get all paths
    paths = acopf_config.get_all_paths()

    # Get all training parameters
    params = acopf_config.get_all_params()

    # Add GNN-specific parameters
    GNN_HIDDEN_DIM = 256
    GNN_NUM_LAYERS = 2

    params['hidden_dim'] = GNN_HIDDEN_DIM
    params['num_gnn_layers'] = GNN_NUM_LAYERS

    print(f"\n[GNN Specific Configuration]")
    print(f"  GNN Hidden Dimension: {GNN_HIDDEN_DIM}")
    print(f"  GNN Layers: {GNN_NUM_LAYERS}")
    print("=" * 70)

    # Execute experiment
    results = gnn_acopf_experiment(**paths, **params)

    print("\n✓ Experiment completed successfully!")