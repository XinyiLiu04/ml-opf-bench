# -*- coding: utf-8 -*-
"""
DeepOPF (PINN) + Post-processing for DCOPF - Main Experiment Script
Physics-Informed Neural Network with Iterative Optimization

Version: v5.0 - Aligned with DNN Version

Features:
- Physics-informed training with zero-order gradient estimation
- Post-processing with iterative optimization (adjusts only non-Slack generators)
- Auto-identify and handle Slack Bus
- Support API_TEST mode with dual constraint parameters

Evaluation Metrics (6 metrics aligned with DNN):
- MAE Pg (%) - Non-Slack, Slack
- Pg Violation (p.u., Mean of Max) - Non-Slack, Slack
- Branch Violation (p.u., Mean of Max)
- Cost Gap (%)
- Training Time (s)
- Inference Time (ms) - Two stages: NN + Post-processing
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

# Global parameters
GLOBAL_PARAMS = {}
GLOBAL_SCALERS = {}


# ============================================================================
# Post-processing: Iterative Optimization (Non-Slack Only)
# ============================================================================

def post_process_solution(Pg_pred_non_slack, x_pd, params, max_iter=100):
    """
    Post-processing: Iterative optimization of non-Slack generators

    Core Idea:
    ----------
    1. Adjust only non-Slack generators (satisfy upper/lower limits)
    2. Slack generators always determined by power balance: Pg_slack = Σ Pd - Σ Pg_non_slack
    3. If Slack exceeds limits, proportionally adjust non-Slack to compensate
    4. Iterate until convergence

    Parameters:
    -----------
    Pg_pred_non_slack : np.ndarray
        Predicted non-Slack generator outputs, shape (n_samples, n_g_non_slack), p.u.
    x_pd : np.ndarray
        Load demand, shape (n_samples, n_buses), p.u.
    params : dict
        System parameters (must include slack info)
    max_iter : int
        Maximum iterations

    Returns:
    --------
    Pg_corrected_non_slack : np.ndarray
        Corrected non-Slack generator outputs, shape (n_samples, n_g_non_slack), p.u.
    """
    n_samples = Pg_pred_non_slack.shape[0]

    slack_indices = params['general']['slack_gen_indices']
    non_slack_indices = params['general']['non_slack_gen_indices']

    Pg_min = params['constraints']['Pg_min'].ravel()
    Pg_max = params['constraints']['Pg_max'].ravel()

    Pg_min_non_slack = Pg_min[non_slack_indices]
    Pg_max_non_slack = Pg_max[non_slack_indices]
    Pg_min_slack = Pg_min[slack_indices]
    Pg_max_slack = Pg_max[slack_indices]

    Pg_non_slack = Pg_pred_non_slack.copy()
    pd_total = x_pd.sum(axis=1)

    convergence_tol = 1e-6

    for iteration in range(max_iter):
        Pg_non_slack = np.clip(Pg_non_slack, Pg_min_non_slack[np.newaxis, :],
                               Pg_max_non_slack[np.newaxis, :])

        Pg_full = reconstruct_full_pg(
            pg_non_slack=Pg_non_slack,
            pd_total=pd_total,
            params=params
        )

        Pg_slack = Pg_full[:, slack_indices]

        slack_upper_viol = np.maximum(0, Pg_slack - Pg_max_slack[np.newaxis, :])
        slack_lower_viol = np.maximum(0, Pg_min_slack[np.newaxis, :] - Pg_slack)

        max_slack_viol_per_sample = np.maximum(
            np.max(slack_upper_viol, axis=1),
            np.max(slack_lower_viol, axis=1)
        )

        max_viol = np.max(max_slack_viol_per_sample)

        if max_viol < convergence_tol:
            break

        for i in range(n_samples):
            if max_slack_viol_per_sample[i] < convergence_tol:
                continue

            Pg_slack_feasible = np.clip(Pg_slack[i], Pg_min_slack, Pg_max_slack)
            slack_total_adjustment = np.sum(Pg_slack_feasible) - np.sum(Pg_slack[i])
            target_adjustment = -slack_total_adjustment

            if abs(target_adjustment) < 1e-8:
                continue

            Pg_non_slack_current = Pg_non_slack[i]

            if target_adjustment > 0:
                capacity_up = Pg_max_non_slack - Pg_non_slack_current
                total_capacity_up = np.sum(capacity_up)

                if total_capacity_up > 1e-8:
                    adjustment_ratio = np.minimum(1.0, target_adjustment / total_capacity_up)
                    Pg_non_slack[i] += capacity_up * adjustment_ratio
                else:
                    Pg_non_slack[i] = Pg_max_non_slack

            else:
                capacity_down = Pg_non_slack_current - Pg_min_non_slack
                total_capacity_down = np.sum(capacity_down)

                if total_capacity_down > 1e-8:
                    adjustment_ratio = np.minimum(1.0, abs(target_adjustment) / total_capacity_down)
                    Pg_non_slack[i] -= capacity_down * adjustment_ratio
                else:
                    Pg_non_slack[i] = Pg_min_non_slack

        Pg_non_slack = np.clip(Pg_non_slack, Pg_min_non_slack[np.newaxis, :],
                               Pg_max_non_slack[np.newaxis, :])

    return Pg_non_slack


# ============================================================================
# Physics Constraint Penalty Calculation
# ============================================================================

def compute_dcopf_penalty(y_pred_pg_non_slack, x_pd, params):
    """Calculate DCOPF constraint violation penalty"""
    n_samples = y_pred_pg_non_slack.shape[0]

    pd_total = x_pd.sum(axis=1)
    y_pred_pg_full = reconstruct_full_pg(
        pg_non_slack=y_pred_pg_non_slack,
        pd_total=pd_total,
        params=params
    )

    gen_up_viol, gen_lo_viol, line_viol, balance_err = feasibility(
        y_pred_pg=y_pred_pg_full,
        x_pd=x_pd,
        params=params
    )

    ctol = 1e-4
    penalties = np.zeros(n_samples)

    for i in range(n_samples):
        pg_viol = gen_up_viol[i, :] + gen_lo_viol[i, :]
        pg_viol[pg_viol < ctol] = 0
        pg_penalty = np.sum(np.abs(pg_viol))

        line_v = line_viol[i, :]
        line_v[line_v < ctol] = 0
        line_penalty = np.sum(np.abs(line_v))

        balance_penalty = np.abs(balance_err[i])
        if balance_penalty < ctol:
            balance_penalty = 0

        penalties[i] = pg_penalty + line_penalty + balance_penalty

    return penalties


# ============================================================================
# PINN Physics Constraint Layer
# ============================================================================

class Penalty_DCOPF(Function):
    """DCOPF physics constraint penalty layer with zero-order gradient estimation"""

    @staticmethod
    def forward(ctx, nn_output_scaled, x_input_scaled):
        ctx.save_for_backward(nn_output_scaled, x_input_scaled)

        nn_output_np = nn_output_scaled.cpu().detach().numpy()
        x_input_np = x_input_scaled.cpu().detach().numpy()

        params = GLOBAL_PARAMS
        scalers = GLOBAL_SCALERS

        y_pred_pg_non_slack = scalers['y_pg_non_slack'].inverse_transform(nn_output_np)
        x_raw = scalers['x'].inverse_transform(x_input_np)

        penalty_list = compute_dcopf_penalty(y_pred_pg_non_slack, x_raw, params)
        total_penalty = np.mean(penalty_list)

        return torch.tensor(total_penalty, dtype=torch.float32, device=nn_output_scaled.device)

    @staticmethod
    def backward(ctx, grad_output):
        nn_output_scaled, x_input_scaled = ctx.saved_tensors

        nn_output_np = nn_output_scaled.cpu().detach().numpy()
        x_input_np = x_input_scaled.cpu().detach().numpy()

        batch_size, output_dim = nn_output_np.shape
        params = GLOBAL_PARAMS
        scalers = GLOBAL_SCALERS

        vec = np.random.randn(batch_size, output_dim)
        vec_norm = np.linalg.norm(vec, axis=1).reshape(-1, 1)
        vector_h = vec / (vec_norm + 1e-10)

        h = 1e-4

        nn_output_plus_h = np.clip(nn_output_np + vector_h * h, 0, 1)
        nn_output_minus_h = np.clip(nn_output_np - vector_h * h, 0, 1)

        x_raw = scalers['x'].inverse_transform(x_input_np)
        y_pred_pg_plus = scalers['y_pg_non_slack'].inverse_transform(nn_output_plus_h)
        y_pred_pg_minus = scalers['y_pg_non_slack'].inverse_transform(nn_output_minus_h)

        penalty_plus = compute_dcopf_penalty(y_pred_pg_plus, x_raw, params)
        penalty_minus = compute_dcopf_penalty(y_pred_pg_minus, x_raw, params)

        gradient_estimate = np.zeros((batch_size, output_dim), dtype='float32')

        for i in range(batch_size):
            directional_derivative = (penalty_plus[i] - penalty_minus[i]) / (2 * h)
            gradient_estimate[i] = directional_derivative * vector_h[i] * output_dim

        final_gradient = gradient_estimate * (1.0 / batch_size)

        return torch.from_numpy(final_gradient).to(nn_output_scaled.device) * grad_output, None


# ============================================================================
# PINN Model
# ============================================================================

class PINN_DCOPF(nn.Module):
    """Physics-Informed Neural Network for DCOPF"""

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
        x_sol = self.net(x)
        x_penalty = self.penalty_layer(x_sol, x)
        return x_sol, x_penalty.to(x_sol.device)


# ============================================================================
# Data Loading Function
# ============================================================================

def load_and_prepare_deepopf_data(file_path, params, column_names):
    """Load and prepare DeepOPF training data"""
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


# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(
        model,
        X,
        indices,
        raw_data,
        params,
        scalers,
        device,
        test_data_external=None,
        test_params=None
):
    """Evaluate model performance with post-processing (aligned with DNN)"""

    eval_params = test_params if test_params is not None else params

    global GLOBAL_PARAMS
    original_global_params = GLOBAL_PARAMS
    GLOBAL_PARAMS = eval_params

    try:
        model.eval()

        if test_data_external is not None:
            x_raw_eval = test_data_external['x']
            y_true_pg_all = test_data_external['y_pg_all']
            x_scaled = scalers['x'].transform(x_raw_eval)
            X_eval = torch.tensor(x_scaled, dtype=torch.float32, device=device)

            with torch.no_grad():
                y_pred_non_slack_scaled, _ = model(X_eval)
        else:
            with torch.no_grad():
                y_pred_non_slack_scaled, _ = model(X.to(device))
            x_raw_eval = raw_data['x'][indices]
            y_true_pg_all = raw_data['y_pg_all'][indices]

        y_pred_non_slack_scaled_np = y_pred_non_slack_scaled.cpu().numpy()
        y_pred_non_slack = scalers['y_pg_non_slack'].inverse_transform(y_pred_non_slack_scaled_np)

        # Post-processing
        pd_total = x_raw_eval.sum(axis=1)
        y_pred_non_slack_corrected = post_process_solution(
            Pg_pred_non_slack=y_pred_non_slack,
            x_pd=x_raw_eval,
            params=eval_params,
            max_iter=100
        )

        # Reconstruct full Pg
        y_pred_pg_all = reconstruct_full_pg(
            pg_non_slack=y_pred_non_slack_corrected,
            pd_total=pd_total,
            params=eval_params
        )

        # Calculate MAE
        mae_dict = compute_detailed_mae(
            y_true_all=y_true_pg_all,
            y_pred_non_slack=y_pred_non_slack_corrected,
            y_pred_all=y_pred_pg_all,
            params=eval_params
        )

        # Calculate violations (p.u.)
        gen_up_viol, gen_lo_viol, line_viol, balance_err = feasibility(
            y_pred_pg=y_pred_pg_all,
            x_pd=x_raw_eval,
            params=eval_params
        )

        viol_dict = compute_detailed_pg_violations_pu(
            gen_up_viol=gen_up_viol,
            gen_lo_viol=gen_lo_viol,
            params=eval_params
        )

        viol_branch_pu = compute_branch_violation_pu(
            line_viol=line_viol,
            Pl_max=eval_params['constraints']['Pl_max']
        )

        # Calculate Cost
        cost_coeffs = {
            'C2': eval_params['constraints'].get('C_Pg_c2', np.zeros(y_true_pg_all.shape[1])),
            'C1': eval_params['constraints']['C_Pg'],
            'C0': eval_params['constraints'].get('C_Pg_c0', np.zeros(y_true_pg_all.shape[1]))
        }

        cost_true = compute_cost(y_true_pg_all, cost_coeffs)
        cost_pred = compute_cost(y_pred_pg_all, cost_coeffs)
        cost_gap_pct = compute_cost_gap_percentage(cost_true, cost_pred)

        result = {
            'mae_pg_non_slack': mae_dict['mae_non_slack'],
            'mae_pg_slack': mae_dict['mae_slack'],
            'viol_pg_non_slack': viol_dict['viol_non_slack'],
            'viol_pg_slack': viol_dict['viol_slack'],
            'viol_branch': viol_branch_pu,
            'cost_gap_percent': cost_gap_pct,
        }

    finally:
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
    """PINN for DCOPF training function with post-processing (aligned with DNN version)"""

    global GLOBAL_PARAMS, GLOBAL_SCALERS

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)

    # Load parameters
    params = load_parameters_from_csv(case_name, params_path, is_api=False)
    slack_info = identify_slack_bus_and_gens(params)
    params = update_params_with_slack_info(params, slack_info)
    GLOBAL_PARAMS = params

    test_params = None
    if split_mode == DataSplitMode.API_TEST:
        if test_params_path is None:
            raise ValueError('API_TEST mode requires test_params_path')
        test_params = load_parameters_from_csv(case_name, test_params_path, is_api=True)
        test_slack_info = identify_slack_bus_and_gens(test_params)
        test_params = update_params_with_slack_info(test_params, test_slack_info)
    else:
        test_params = params

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
    scalers = {'x': x_scaler, 'y_pg_non_slack': y_pg_non_slack_scaler}
    GLOBAL_SCALERS = scalers

    x_train_scaled = x_scaler.transform(x_data_raw[train_idx])
    y_train_scaled = y_pg_non_slack_scaler.transform(y_pg_raw_non_slack[train_idx])
    x_val_scaled = x_scaler.transform(x_data_raw[val_idx])
    y_val_scaled = y_pg_non_slack_scaler.transform(y_pg_raw_non_slack[val_idx])

    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        X_test = None
    else:
        x_test_scaled = x_scaler.transform(x_data_raw[test_idx])
        X_test = torch.tensor(x_test_scaled, dtype=torch.float32)

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

    # Training
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

    # Evaluation
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
            params=params,
            scalers=scalers,
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
            params=params,
            scalers=scalers,
            device=device,
            test_params=test_params
        )

    # Speed test - Stage 1: Neural Network
    model.eval()

    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        test_sample = torch.tensor(
            x_scaler.transform(x_test_external[:1]),
            dtype=torch.float32,
            device=device
        )
        test_sample_raw = x_test_external[:1]
    else:
        test_sample = X_test[:1].to(device)
        test_sample_raw = x_data_raw[test_idx[:1]]

    times_nn = []
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(test_sample)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Measure
        for _ in range(100):
            t_start = time.time()
            pred_scaled, _ = model(test_sample)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times_nn.append(time.time() - t_start)

    nn_time_ms = np.mean(times_nn) * 1000

    # Speed test - Stage 2: Post-processing
    # Get prediction for post-processing timing
    with torch.no_grad():
        pred_scaled_single, _ = model(test_sample)
    pred_np = pred_scaled_single.cpu().numpy()
    pred_raw = scalers['y_pg_non_slack'].inverse_transform(pred_np)

    times_pp = []
    for _ in range(100):
        t_start = time.time()
        _ = post_process_solution(
            Pg_pred_non_slack=pred_raw,
            x_pd=test_sample_raw,
            params=test_params,
            max_iter=100
        )
        times_pp.append(time.time() - t_start)

    pp_time_ms = np.mean(times_pp) * 1000

    # Print results (aligned with DNN format)
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
    print("\n" + "=" * 70)

    # Print two-stage time summary
    print("\nOverall Performance Summary")
    print("=" * 70)
    print(f"\nTraining Time:   {train_time:.2f} s")
    print(f"\nInference Time:")
    print(f"  Stage 1 (Neural Network):   {nn_time_ms:.4f} ms")
    print(f"  Stage 2 (Post-processing):  {pp_time_ms:.4f} ms")
    print(f"  Total:                      {nn_time_ms + pp_time_ms:.4f} ms")
    print("\n" + "=" * 70 + "\n")


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
    train_pinn_dcopf(
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