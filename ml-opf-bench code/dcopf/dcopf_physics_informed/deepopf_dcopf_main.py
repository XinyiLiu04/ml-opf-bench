# -*- coding: utf-8 -*-
"""
DeepOPF (PINN) for DCOPF - Main Experiment Script
Physics-Informed Neural Network Method

Version: v5.0 - Aligned with DNN Version

Features:
- Auto-identify and handle Slack Bus
- Predict only non-Slack generators
- Detailed metrics (Non-Slack, Slack)
- Zero-order gradient estimation (core unchanged)
- Support API_TEST mode with dual constraint parameters
- Complete Cost evaluation

Evaluation Metrics:
- Training Time (s)
- Inference Time (ms)
- MAE Pg (%) - Non-Slack, Slack
- Pg Violation (p.u., Mean of Max) - Non-Slack, Slack
- Branch Violation (p.u., Mean of Max)
- Cost Gap (%)
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Function
from sklearn.preprocessing import MinMaxScaler

# Import custom modules
sys.path.append(os.path.dirname(__file__))

from dcopf_data_setup import (
    load_parameters_from_csv,
    DataSplitMode,
    split_data_by_mode
)
from dcopf_violation_metrics import (
    feasibility,
    compute_mae_percentage,
    compute_branch_violation_pu,
    compute_cost,
    compute_cost_gap_percentage
)
from dcopf_slack_utils import (
    identify_slack_bus_and_gens,
    update_params_with_slack_info,
    reconstruct_full_pg,
    compute_detailed_mae,
    compute_detailed_pg_violations_pu
)

# Global parameters (for physics constraint layer)
GLOBAL_PARAMS = {}
GLOBAL_SCALERS = {}


# ============================================================================
# Physics Constraint Penalty Calculation
# ============================================================================

def compute_dcopf_penalty(y_pred_pg_non_slack, x_pd, params):
    """
    Calculate DCOPF constraint violation penalty (with Slack Bus handling)

    Parameters:
    -----------
    y_pred_pg_non_slack : np.ndarray
        Predicted non-Slack generator outputs, shape (n_samples, n_g_non_slack), p.u.
    x_pd : np.ndarray
        Load demand, shape (n_samples, n_buses), p.u.
    params : dict
        System parameters, must include slack info

    Returns:
    --------
    penalties : np.ndarray
        Total penalty for each sample, shape (n_samples,)
    """
    n_samples = y_pred_pg_non_slack.shape[0]

    # Step 1: Reconstruct full Pg vector (including Slack generators)
    pd_total = x_pd.sum(axis=1)
    y_pred_pg_all = reconstruct_full_pg(
        pg_non_slack=y_pred_pg_non_slack,
        pd_total=pd_total,
        params=params
    )

    # Step 2: Calculate constraint violations using full Pg
    gen_up_viol, gen_lo_viol, line_viol, balance_err = feasibility(
        y_pred_pg=y_pred_pg_all,
        x_pd=x_pd,
        params=params
    )

    # Step 3: Calculate total penalty for each sample
    ctol = 1e-4
    penalties = np.zeros(n_samples)

    for i in range(n_samples):
        # 1. Generator constraint penalty
        pg_viol = gen_up_viol[i, :] + gen_lo_viol[i, :]
        pg_viol[pg_viol < ctol] = 0
        pg_penalty = np.sum(np.abs(pg_viol))

        # 2. Line constraint penalty
        line_v = line_viol[i, :]
        line_v[line_v < ctol] = 0
        line_penalty = np.sum(np.abs(line_v))

        # 3. Power balance penalty
        balance_penalty = np.abs(balance_err[i])
        if balance_penalty < ctol:
            balance_penalty = 0

        # Total penalty
        penalties[i] = pg_penalty + line_penalty + balance_penalty

    return penalties


# ============================================================================
# PINN Physics Constraint Layer
# ============================================================================

class Penalty_DCOPF(Function):
    """
    DCOPF physics constraint penalty layer

    Forward: Calculate constraint violation penalty
    Backward: Use zero-order gradient estimation
    """

    @staticmethod
    def forward(ctx, nn_output_scaled, x_input_scaled):
        """
        Forward pass: Calculate penalty term

        Parameters:
        -----------
        nn_output_scaled : torch.Tensor
            Network output (scaled non-Slack pg), shape (batch_size, n_g_non_slack)
        x_input_scaled : torch.Tensor
            Network input (scaled pd), shape (batch_size, n_buses)

        Returns:
        --------
        total_penalty : torch.Tensor
            Scalar, average penalty
        """
        ctx.save_for_backward(nn_output_scaled, x_input_scaled)

        nn_output_np = nn_output_scaled.cpu().detach().numpy()
        x_input_np = x_input_scaled.cpu().detach().numpy()

        params = GLOBAL_PARAMS
        scalers = GLOBAL_SCALERS

        # Inverse transform
        y_pred_pg_non_slack = scalers['y_pg_non_slack'].inverse_transform(nn_output_np)
        x_raw = scalers['x'].inverse_transform(x_input_np)

        # Calculate penalty
        penalty_list = compute_dcopf_penalty(y_pred_pg_non_slack, x_raw, params)

        # Return average penalty
        total_penalty = np.mean(penalty_list)

        return torch.tensor(total_penalty, dtype=torch.float32, device=nn_output_scaled.device)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Zero-order gradient estimation

        Method:
        -------
        1. Generate random unit vector v ∈ R^{n_g_non_slack}
        2. Calculate f(x + h*v) and f(x - h*v)
        3. Estimate directional derivative: (f(x+h*v) - f(x-h*v)) / (2h)
        4. Estimate gradient: directional_derivative * v * output_dim
        """
        nn_output_scaled, x_input_scaled = ctx.saved_tensors

        nn_output_np = nn_output_scaled.cpu().detach().numpy()
        x_input_np = x_input_scaled.cpu().detach().numpy()

        batch_size, output_dim = nn_output_np.shape
        params = GLOBAL_PARAMS
        scalers = GLOBAL_SCALERS

        # Generate random unit vector
        vec = np.random.randn(batch_size, output_dim)
        vec_norm = np.linalg.norm(vec, axis=1).reshape(-1, 1)
        vector_h = vec / (vec_norm + 1e-10)

        h = 1e-4

        # Calculate perturbed outputs
        nn_output_plus_h = np.clip(nn_output_np + vector_h * h, 0, 1)
        nn_output_minus_h = np.clip(nn_output_np - vector_h * h, 0, 1)

        # Inverse transform
        x_raw = scalers['x'].inverse_transform(x_input_np)
        y_pred_pg_non_slack_plus = scalers['y_pg_non_slack'].inverse_transform(nn_output_plus_h)
        y_pred_pg_non_slack_minus = scalers['y_pg_non_slack'].inverse_transform(nn_output_minus_h)

        # Calculate penalties at +h and -h
        penalty_plus = compute_dcopf_penalty(y_pred_pg_non_slack_plus, x_raw, params)
        penalty_minus = compute_dcopf_penalty(y_pred_pg_non_slack_minus, x_raw, params)

        # Estimate gradient
        gradient_estimate = np.zeros((batch_size, output_dim), dtype='float32')

        for i in range(batch_size):
            directional_derivative = (penalty_plus[i] - penalty_minus[i]) / (2 * h)
            gradient_estimate[i] = directional_derivative * vector_h[i] * output_dim

        # Gradient based on average loss
        final_gradient = gradient_estimate * (1.0 / batch_size)

        return torch.from_numpy(final_gradient).to(nn_output_scaled.device) * grad_output, None


# ============================================================================
# PINN Model
# ============================================================================

class PINN_DCOPF(nn.Module):
    """
    Physics-Informed Neural Network for DCOPF

    Structure:
    ----------
    - Standard DNN (input pd → output non-Slack pg)
    - Physics constraint layer (calculate penalty and provide gradient)
    """

    def __init__(self, input_dim, output_dim, hidden_sizes=[256, 256]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        self.penalty_layer = Penalty_DCOPF.apply

    def forward(self, x):
        """
        Forward pass

        Returns:
        --------
        x_sol : torch.Tensor
            Network prediction (scaled non-Slack pg), shape (batch_size, n_g_non_slack)
        x_penalty : torch.Tensor
            Physics constraint penalty, scalar
        """
        x_sol = self.net(x)
        x_penalty = self.penalty_layer(x_sol, x)
        return x_sol, x_penalty.to(x_sol.device)


# ============================================================================
# Data Loading Function
# ============================================================================

def load_and_prepare_deepopf_data(file_path, params, column_names):
    """
    Load and prepare DeepOPF training data

    Parameters:
    -----------
    file_path : str
        Dataset CSV path
    params : dict
        Network parameter dictionary (must include slack info)
    column_names : dict
        Column name mapping

    Returns:
    --------
    x_data_raw : np.ndarray
        Raw input data, shape (n_samples, n_buses), p.u.
    y_pg_raw_non_slack : np.ndarray
        Raw non-Slack generator data, shape (n_samples, n_g_non_slack), p.u.
    y_pg_raw_all : np.ndarray
        Raw all generator data, shape (n_samples, n_g), p.u.
    """
    import pandas as pd

    full_df = pd.read_csv(file_path)
    n_samples = len(full_df)
    n_buses = params['general']['n_buses']

    # Load demand data (handle sparse buses)
    load_prefix = column_names['load_prefix']
    load_cols = [col for col in full_df.columns if col.startswith(load_prefix)]

    load_bus_ids = []
    for col in load_cols:
        bus_id = int(col[len(load_prefix):])
        load_bus_ids.append(bus_id)

    x_data_raw_loads = full_df[load_cols].values.astype('float32')

    # Expand to all buses (handle sparse buses)
    x_data_raw = np.zeros((n_samples, n_buses), dtype='float32')
    for i, bus_id in enumerate(load_bus_ids):
        if bus_id <= n_buses:
            x_data_raw[:, bus_id - 1] = x_data_raw_loads[:, i]

    # Load generation data (all generators)
    pg_cols = [col for col in full_df.columns if col.startswith(column_names['gen_prefix'])]
    y_pg_raw_all = full_df[pg_cols].values.astype('float32')

    # Extract non-Slack generator data
    non_slack_indices = params['general']['non_slack_gen_indices']
    y_pg_raw_non_slack = y_pg_raw_all[:, non_slack_indices]

    return x_data_raw, y_pg_raw_non_slack, y_pg_raw_all


# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(
        model,
        X,
        indices,
        raw_data,
        scalers,
        params,
        device,
        test_data_external=None,
        test_params=None
):
    """
    Evaluate model performance (aligned with DNN evaluation logic)

    Parameters:
    -----------
    model : nn.Module
        PINN model
    X : torch.Tensor or None
        Input tensor (normal mode), shape (n_samples, n_buses)
    indices : np.ndarray or None
        Sample indices (normal mode)
    raw_data : dict
        Raw data dictionary, contains 'x', 'y_pg_non_slack', 'y_pg_all'
    scalers : dict
        Scaler dictionary
    params : dict
        Training system parameters
    device : torch.device
        Compute device
    test_data_external : dict or None
        External test data, contains 'x', 'y_pg_all'
    test_params : dict or None
        Test system parameters (only needed for API_TEST mode)

    Returns:
    --------
    result : dict
        Evaluation metrics (6 metrics aligned with DNN)
    """
    # Determine which parameters to use for evaluation
    eval_params = test_params if test_params is not None else params

    # IMPORTANT: Temporarily switch global parameters
    global GLOBAL_PARAMS
    original_global_params = GLOBAL_PARAMS
    GLOBAL_PARAMS = eval_params

    try:
        model.eval()

        # Determine if using external test data
        if test_data_external is not None:
            # Generalization or API_TEST mode
            x_raw_eval = test_data_external['x']
            y_true_pg_all = test_data_external['y_pg_all']

            # Scale test data
            x_scaled = scalers['x'].transform(x_raw_eval)
            X_eval = torch.tensor(x_scaled, dtype=torch.float32, device=device)

            with torch.no_grad():
                y_pred_non_slack_scaled, _ = model(X_eval)
        else:
            # Normal mode
            with torch.no_grad():
                y_pred_non_slack_scaled, _ = model(X.to(device))

            x_raw_eval = raw_data['x'][indices]
            y_true_pg_all = raw_data['y_pg_all'][indices]

        y_pred_non_slack_scaled_np = y_pred_non_slack_scaled.cpu().numpy()

        # Inverse transform non-Slack predictions
        y_pred_non_slack = scalers['y_pg_non_slack'].inverse_transform(y_pred_non_slack_scaled_np)

        # Reconstruct full Pg vector
        pd_total = x_raw_eval.sum(axis=1)
        y_pred_pg_all = reconstruct_full_pg(
            pg_non_slack=y_pred_non_slack,
            pd_total=pd_total,
            params=eval_params
        )

        # --- 1. Calculate detailed MAE (%) ---
        mae_dict = compute_detailed_mae(
            y_true_all=y_true_pg_all,
            y_pred_non_slack=y_pred_non_slack,
            y_pred_all=y_pred_pg_all,
            params=eval_params
        )

        # --- 2. Calculate DCOPF violations (p.u.) ---
        gen_up_viol, gen_lo_viol, line_viol, balance_err = feasibility(
            y_pred_pg=y_pred_pg_all,
            x_pd=x_raw_eval,
            params=eval_params
        )

        # --- 3. Calculate detailed Pg violations (p.u.) ---
        viol_dict = compute_detailed_pg_violations_pu(
            gen_up_viol=gen_up_viol,
            gen_lo_viol=gen_lo_viol,
            params=eval_params
        )

        # --- 4. Calculate Branch violations (p.u.) ---
        viol_branch_pu = compute_branch_violation_pu(
            line_viol=line_viol,
            Pl_max=eval_params['constraints']['Pl_max']
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

        result = {
            # MAE metrics
            'mae_pg_non_slack': mae_dict['mae_non_slack'],
            'mae_pg_slack': mae_dict['mae_slack'],
            # Violation metrics (p.u.)
            'viol_pg_non_slack': viol_dict['viol_non_slack'],
            'viol_pg_slack': viol_dict['viol_slack'],
            'viol_branch': viol_branch_pu,
            # Cost metrics
            'cost_gap_percent': cost_gap_pct,
        }

    finally:
        # Restore global parameters
        GLOBAL_PARAMS = original_global_params

    return result


# ============================================================================
# Main Training Function
# ============================================================================

def train_pinn_dcopf(
        case_name,
        params_path,
        dataset_path,
        column_names,
        n_train_use=10000,
        hidden_sizes=[256, 256],
        n_epochs=100,
        batch_size=256,
        learning_rate=1e-3,
        penalty_weight=0.1,
        seed=42,
        device='cuda',
        split_mode=DataSplitMode.RANDOM_SPLIT,
        test_data_path=None,
        test_params_path=None,
        n_test_samples=1000
):
    """
    PINN for DCOPF training function (aligned with DNN version)

    Returns:
    --------
    test_metrics : dict
        6 evaluation metrics (aligned with DNN)
    """

    global GLOBAL_PARAMS, GLOBAL_SCALERS

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)

    # Load training constraint parameters
    params = load_parameters_from_csv(case_name, params_path, is_api=False)

    # Auto-identify Slack Bus
    slack_info = identify_slack_bus_and_gens(params)
    params = update_params_with_slack_info(params, slack_info)

    # If API_TEST mode, load test constraint parameters
    test_params = None
    if split_mode == DataSplitMode.API_TEST:
        if test_params_path is None:
            raise ValueError('API_TEST mode requires test_params_path')
        test_params = load_parameters_from_csv(case_name, test_params_path, is_api=True)
        test_slack_info = identify_slack_bus_and_gens(test_params)
        test_params = update_params_with_slack_info(test_params, test_slack_info)
    else:
        test_params = params

    # Use training constraints during training
    GLOBAL_PARAMS = params

    n_buses = params['general']['n_buses']
    n_gen_non_slack = params['general']['n_g_non_slack']

    # Load data
    x_data_raw, y_pg_raw_non_slack, y_pg_raw_all = load_and_prepare_deepopf_data(
        dataset_path, params, column_names
    )

    raw_data = {
        'x': x_data_raw,
        'y_pg_non_slack': y_pg_raw_non_slack,
        'y_pg_all': y_pg_raw_all
    }

    # Data split
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

    # Data normalization
    x_scaler = MinMaxScaler().fit(x_data_raw[train_idx])
    y_pg_non_slack_scaler = MinMaxScaler().fit(y_pg_raw_non_slack[train_idx])

    scalers = {
        'x': x_scaler,
        'y_pg_non_slack': y_pg_non_slack_scaler
    }
    GLOBAL_SCALERS = scalers

    x_train_scaled = x_scaler.transform(x_data_raw[train_idx])
    y_train_scaled = y_pg_non_slack_scaler.transform(y_pg_raw_non_slack[train_idx])
    x_val_scaled = x_scaler.transform(x_data_raw[val_idx])
    y_val_scaled = y_pg_non_slack_scaler.transform(y_pg_raw_non_slack[val_idx])

    # Handle test set
    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        X_test = None
    else:
        x_test_scaled = x_scaler.transform(x_data_raw[test_idx])
        X_test = torch.tensor(x_test_scaled, dtype=torch.float32)

    # Prepare training/validation tensors
    X_train = torch.from_numpy(x_train_scaled).float().to(device)
    Y_train = torch.from_numpy(y_train_scaled).float().to(device)
    X_val = torch.from_numpy(x_val_scaled).float().to(device)

    train_dataset = Data.TensorDataset(X_train, Y_train)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Build model
    model = PINN_DCOPF(
        input_dim=x_data_raw.shape[1],
        output_dim=n_gen_non_slack,
        hidden_sizes=hidden_sizes
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    # Training loop
    training_start = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()

        epoch_total = 0.0
        epoch_mse = 0.0
        epoch_penalty = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            pred, penalty = model(batch_x)

            mse_loss = criterion(pred, batch_y)
            total_loss = 1.0 * mse_loss + penalty_weight * penalty

            total_loss.backward()
            optimizer.step()

            epoch_total += total_loss.item() * len(batch_x)
            epoch_mse += mse_loss.item() * len(batch_x)
            epoch_penalty += penalty.item() * len(batch_x)

        avg_total = epoch_total / len(X_train)
        avg_mse = epoch_mse / len(X_train)
        avg_penalty = epoch_penalty / len(X_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, val_penalty = model(X_val)
            val_mse = criterion(
                val_pred,
                torch.from_numpy(y_val_scaled).float().to(device)
            )
            val_loss = 1.0 * val_mse.item() + penalty_weight * val_penalty.item()

        # Print every epoch (aligned with DNN)
        print(f"Epoch {epoch}/{n_epochs} - train_loss: {avg_total:.6f} - val_loss: {val_loss:.6f}")

    train_time = time.time() - training_start

    # Evaluation (test set only)
    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        test_data_external_dict = {
            'x': x_test_external,
            'y_pg_all': y_test_external
        }
        test_metrics = evaluate_model(
            model=model,
            X=None,
            indices=None,
            raw_data=raw_data,
            scalers=scalers,
            params=params,
            device=device,
            test_data_external=test_data_external_dict,
            test_params=test_params
        )
    else:
        test_metrics = evaluate_model(
            model=model,
            X=X_test,
            indices=test_idx,
            raw_data=raw_data,
            scalers=scalers,
            params=params,
            device=device,
            test_params=test_params
        )

    # Speed evaluation
    model.eval()

    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        test_sample = torch.tensor(
            x_scaler.transform(x_test_external[:1]),
            dtype=torch.float32,
            device=device
        )
    else:
        test_sample = X_test[:1].to(device)

    times = []
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(test_sample)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Measure
        for _ in range(100):
            t_start = time.time()
            _ = model(test_sample)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - t_start)

    latency_ms = np.mean(times) * 1000

    # Print final results (aligned with DNN format)
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


# ============================================================================
# Main Program
# ============================================================================

if __name__ == '__main__':
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

    # --- 4. PINN Hyperparameters ---
    N_EPOCHS = 100
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    HIDDEN_SIZES = [128, 128]
    PENALTY_WEIGHT = 0.005
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
    # Path Generation
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
    results = train_pinn_dcopf(
        case_name=CASE_NAME,
        params_path=params_path,
        dataset_path=train_data_path,
        column_names=COLUMN_NAMES,
        n_train_use=N_TRAIN_USE,
        hidden_sizes=HIDDEN_SIZES,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        penalty_weight=PENALTY_WEIGHT,
        seed=SEED,
        device=device_name,
        split_mode=SPLIT_MODE,
        test_data_path=test_data_path,
        test_params_path=test_params_path,
        n_test_samples=N_TEST_SAMPLES
    )