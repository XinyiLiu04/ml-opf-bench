# -*- coding: utf-8 -*-
"""
Traditional DNN DCOPF Main Experiment (PyTorch) - Streamlined Benchmark Version

Strict evaluation metrics as defined in the paper:
- MAE Pg (%) - Non-Slack & Slack-Only
- Pg Violation (p.u.) - Non-Slack & Slack-Only, Mean of Max
- Branch Violation (p.u.) - Mean of Max, as multiple of capacity
- Cost Gap (%)
- Training Time (s)
- Inference Time (ms)

Data split: 10:1:1 (consistent with PINN)
"""

import numpy as np
import pandas as pd
import torch
import time
import os
import sys

import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dcopf_data_setup import load_parameters_from_csv, DataSplitMode, split_data_by_mode

# Import DCOPF evaluation functions
from dcopf_violation_metrics import (
    feasibility as dc_feasibility,
    compute_cost,
    compute_cost_gap_percentage,
    compute_branch_violation_pu,
    compute_mae_percentage
)

sys.path.append('/home/claude')
from dcopf_slack_utils import (
    identify_slack_bus_and_gens,
    update_params_with_slack_info,
    reconstruct_full_pg,
    compute_detailed_mae,
    compute_detailed_pg_violations_pu
)


# =====================================================================
# PyTorch Neural Network Model
# =====================================================================
class TraditionalNN_DCOPF(nn.Module):
    """PyTorch version of traditional neural network for DCOPF"""

    def __init__(self, input_size, output_size, hidden_layers=[128, 128]):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_and_prepare_data_trad(full_dataset_path, params):
    """
    Load and prepare data
    Use dynamic column detection to adapt to Julia-generated dataset_with_duals.csv
    """
    full_df = pd.read_csv(full_dataset_path)
    n_samples = len(full_df)
    n_buses = params['general']['n_buses']

    # Dynamically detect Load columns: pd1, pd2, pd3, ...
    pd_cols_available = sorted([col for col in full_df.columns if col.startswith('pd')])

    # Extract load bus IDs
    load_bus_ids = []
    for col in pd_cols_available:
        bus_id = int(col[2:])  # Extract 10 from 'pd10'
        load_bus_ids.append(bus_id)

    # Read load data
    x_data_raw_loads = full_df[pd_cols_available].values.astype('float32')

    # Expand to all buses (buses without load are filled with 0)
    x_data_raw_full = np.zeros((n_samples, n_buses), dtype='float32')
    for i, bus_id in enumerate(load_bus_ids):
        if bus_id <= n_buses:
            x_data_raw_full[:, bus_id - 1] = x_data_raw_loads[:, i]

    # Dynamically detect Generation columns: pg1, pg2, pg3, ...
    pg_cols_available = sorted([col for col in full_df.columns if col.startswith('pg')])

    # Use correct key 'n_g' instead of 'n_gen'
    expected_n_gen = params['general']['n_g']
    if len(pg_cols_available) != expected_n_gen:
        print(f"[WARNING] Number of generators in CSV ({len(pg_cols_available)}) "
              f"does not match parameters file ({expected_n_gen})!")

    y_pg_raw = full_df[pg_cols_available].values.astype('float32')

    # Extract non-slack generator data
    if 'non_slack_gen_indices' in params['general']:
        non_slack_indices = params['general']['non_slack_gen_indices']
        y_pg_raw_non_slack = y_pg_raw[:, non_slack_indices]

        return x_data_raw_full, y_pg_raw_non_slack, y_pg_raw
    else:
        print("[WARNING] Slack information not found, returning all generator data")
        return x_data_raw_full, y_pg_raw, y_pg_raw


# =====================================================================
# Evaluation Function (strict metrics as per paper)
# =====================================================================
def evaluate_split(model, X_tensor, indices, raw_data, scalers, params, device,
                   test_data_external=None, test_params=None):
    """
    Evaluation function - calculate detailed metrics

    Returns metrics:
    - MAE: non_slack, slack
    - Viol_pg (%): non_slack, slack
    - Branch_viol (%)
    - Cost metrics
    """
    model.eval()
    eval_params = test_params if test_params is not None else params

    # Determine if using external test data
    if test_data_external is not None:
        x_raw_eval = test_data_external['x']
        y_true_pg_all = test_data_external['y_pg_all']

        # Scale test data
        x_scaled = scalers['x'].transform(x_raw_eval)
        X_tensor_eval = torch.tensor(x_scaled, dtype=torch.float32, device=device)

        with torch.no_grad():
            y_pred_non_slack_scaled = model(X_tensor_eval).cpu().numpy()
    else:
        # Normal mode: use indices
        with torch.no_grad():
            y_pred_non_slack_scaled = model(X_tensor).cpu().numpy()

        x_raw_eval = raw_data['x'][indices]
        y_true_pg_all = raw_data['y_pg_all'][indices]

    baseMVA = eval_params['general']['BASE_MVA']

    # Inverse transform non-slack predictions
    y_pred_non_slack = scalers['y_pg_non_slack'].inverse_transform(y_pred_non_slack_scaled)

    # Reconstruct full Pg vector (including slack)
    pd_total = x_raw_eval.sum(axis=1)
    y_pred_pg_all = reconstruct_full_pg(y_pred_non_slack, pd_total, eval_params)

    # --- 1. Calculate detailed MAE (%) ---
    mae_dict = compute_detailed_mae(y_true_pg_all, y_pred_non_slack, y_pred_pg_all, eval_params)

    # --- 2. Calculate DCOPF violations (using full Pg) ---
    gen_up_viol_pu, gen_lo_viol_pu, line_viol_pu, _ = dc_feasibility(
        y_pred_pg_all, x_raw_eval, eval_params
    )

    # --- 3. Calculate detailed Pg violations (p.u.) ---
    viol_dict = compute_detailed_pg_violations_pu(
        gen_up_viol_pu, gen_lo_viol_pu, eval_params
    )

    # --- 4. Calculate Branch violations (p.u.) ---
    branch_violation_pu = compute_branch_violation_pu(
        line_viol_pu, eval_params['constraints']['Pl_max']
    )

    # --- 5. Calculate Cost metrics ---
    cost_coeffs = {
        'C2': eval_params['constraints'].get('C_Pg_c2', np.zeros(y_true_pg_all.shape[1])),
        'C1': eval_params['constraints']['C_Pg'],
        'C0': eval_params['constraints'].get('C_Pg_c0', np.zeros(y_true_pg_all.shape[1]))
    }

    cost_true = compute_cost(y_true_pg_all, cost_coeffs)
    cost_pred = compute_cost(y_pred_pg_all, cost_coeffs)
    cost_gap_pct = compute_cost_gap_percentage(cost_true, cost_pred)

    return {
        # MAE metrics
        'mae_pg_non_slack': mae_dict['mae_non_slack'],
        'mae_pg_slack': mae_dict['mae_slack'],
        # Violation metrics (p.u.)
        'viol_pg_non_slack': viol_dict['viol_non_slack'],
        'viol_pg_slack': viol_dict['viol_slack'],
        'viol_branch': branch_violation_pu,
        # Cost metrics
        'cost_gap_percent': cost_gap_pct,
    }


# =====================================================================
# Main Experiment Function
# =====================================================================
def traditional_nn_experiment_pytorch(
        case_name,
        params_path,
        dataset_path,
        n_train_use=10000,
        seed=42,
        n_epochs=1000,
        learning_rate=0.001,
        batch_size=128,
        hidden_layers=[128, 128],
        device='cuda',
        split_mode=DataSplitMode.RANDOM_SPLIT,
        test_data_path=None,
        test_params_path=None,
        column_names=None,
        n_test_samples=1000
):
    """
    Run traditional neural network DCOPF experiment using PyTorch
    Strict evaluation metrics as per paper definition
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\nRunning: {split_mode.value} - {case_name}\n")

    # Load training constraint parameters
    params = load_parameters_from_csv(case_name, params_path, is_api=False)

    # Automatically identify slack bus and generators
    slack_info = identify_slack_bus_and_gens(params)
    params = update_params_with_slack_info(params, slack_info)

    # If API_TEST mode, additionally load test constraint parameters
    test_params = None
    if split_mode == DataSplitMode.API_TEST:
        if test_params_path is None:
            raise ValueError("API_TEST mode requires test_params_path")
        test_params = load_parameters_from_csv(case_name, test_params_path, is_api=True)
        # Identify and add slack information for test parameters
        test_slack_info = identify_slack_bus_and_gens(test_params)
        test_params = update_params_with_slack_info(test_params, test_slack_info)
    else:
        test_params = params

    # If column_names not provided, use default values
    if column_names is None:
        column_names = {
            'load_prefix': 'pd',
            'gen_prefix': 'pg',
            'lambda': 'lambda',
            'mu_g_min_prefix': 'mu_g_min_',
            'mu_g_max_prefix': 'mu_g_max_',
            'mu_line_pos_prefix': 'mu_line_max_',
            'mu_line_neg_prefix': 'mu_line_min_',
        }

    x_data_raw, y_pg_raw_non_slack, y_pg_raw_all = load_and_prepare_data_trad(dataset_path, params)

    raw_data = {
        'x': x_data_raw,
        'y_pg_non_slack': y_pg_raw_non_slack,  # For training
        'y_pg_all': y_pg_raw_all  # For evaluation
    }

    # Data splitting
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

    # Data standardization
    x_scaler = MinMaxScaler().fit(x_data_raw[train_idx])
    y_pg_non_slack_scaler = MinMaxScaler().fit(y_pg_raw_non_slack[train_idx])
    scalers = {
        'x': x_scaler,
        'y_pg_non_slack': y_pg_non_slack_scaler,
    }

    x_train_scaled = x_scaler.transform(x_data_raw[train_idx])
    y_train_scaled = y_pg_non_slack_scaler.transform(y_pg_raw_non_slack[train_idx])
    x_val_scaled = x_scaler.transform(x_data_raw[val_idx])
    y_val_scaled = y_pg_non_slack_scaler.transform(y_pg_raw_non_slack[val_idx])

    # Handle test set
    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        X_test = None  # Will be created in evaluate
    else:
        x_test_scaled = x_scaler.transform(x_data_raw[test_idx])
        X_test = torch.tensor(x_test_scaled, dtype=torch.float32, device=device)

    # Convert to PyTorch Tensor
    X_train = torch.tensor(x_train_scaled, dtype=torch.float32, device=device)
    Y_train = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)
    X_val = torch.tensor(x_val_scaled, dtype=torch.float32, device=device)
    Y_val = torch.tensor(y_val_scaled, dtype=torch.float32, device=device)

    # Create model
    model = TraditionalNN_DCOPF(
        input_size=x_data_raw.shape[1],
        output_size=params['general']['n_g_non_slack'],
        hidden_layers=hidden_layers
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    n_train = len(X_train)
    n_batches = (n_train + batch_size - 1) // batch_size

    # ============ Training loop ============
    t0 = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        model.train()

        epoch_loss = 0.0
        indices = torch.randperm(n_train, device=device)

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            batch_indices = indices[start_idx:end_idx]

            X_batch = X_train[batch_indices]
            Y_batch = Y_train[batch_indices]

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(X_batch)

        train_loss = epoch_loss / n_train

        # Calculate validation loss
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            val_loss = float(criterion(pred_val, Y_val).item())

        # Print every epoch
        print(f"Epoch {epoch}/{n_epochs} - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

    train_time = time.perf_counter() - t0

    # === Evaluation (test set only) ===
    if split_mode == DataSplitMode.GENERALIZATION:
        test_data_external_dict = {
            'x': x_test_external,
            'y_pg_all': y_test_external
        }
        test_metrics = evaluate_split(
            model, None, None, raw_data, scalers, params, device,
            test_data_external=test_data_external_dict,
            test_params=test_params
        )
    elif split_mode == DataSplitMode.API_TEST:
        test_data_external_dict = {
            'x': x_test_external,
            'y_pg_all': y_test_external
        }
        test_metrics = evaluate_split(
            model, None, None, raw_data, scalers, params, device,
            test_data_external=test_data_external_dict,
            test_params=test_params
        )
    else:
        test_metrics = evaluate_split(
            model, X_test, test_idx, raw_data, scalers, params, device,
            test_params=test_params
        )

    # === Speed evaluation ===
    model.eval()

    # Prepare test sample
    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        test_sample = torch.tensor(
            x_scaler.transform(x_test_external[:1]),
            dtype=torch.float32,
            device=device
        )
    else:
        test_sample = X_test[:1]

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_sample)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Measure
    n_repeats = 100
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            t_start = time.perf_counter()
            _ = model(test_sample)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t_start)

    latency_ms = np.mean(times) * 1000

    # === Print final results ===
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
    print(f"Inference Time:  {latency_ms:.4f} ms")
    print("\n" + "=" * 70 + "\n")

    return test_metrics


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

    # --- 4. Training Hyperparameters ---
    N_EPOCHS = 100
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    HIDDEN_LAYERS = [128, 128]
    SEED = 42

    # --- 5. Path Configuration (Manual) ---
    ROOT_DIR = r"C:\Users\Aloha\Desktop\dataset"
    TRAIN_VARIANCE = "v=0.12"
    TEST_VARIANCE = "v=0.25"

    # Column name mapping
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
    # Path Generation (based on split mode)
    # ===================================================================

    # Training constraints and data paths
    params_path = os.path.join(ROOT_DIR, "DCOPF Constraints", CASE_SHORT_NAME)
    train_data_path = os.path.join(
        ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}({TRAIN_VARIANCE})",
        f"{CASE_NAME}_dataset_with_duals.csv"
    )

    # Test data path (based on mode)
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
    results = traditional_nn_experiment_pytorch(
        case_name=CASE_NAME,
        params_path=params_path,
        dataset_path=train_data_path,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        hidden_layers=HIDDEN_LAYERS,
        n_train_use=N_TRAIN_USE,
        seed=SEED,
        device=device_name,
        split_mode=SPLIT_MODE,
        test_data_path=test_data_path,
        test_params_path=test_params_path,
        column_names=COLUMN_NAMES,
        n_test_samples=N_TEST_SAMPLES
    )