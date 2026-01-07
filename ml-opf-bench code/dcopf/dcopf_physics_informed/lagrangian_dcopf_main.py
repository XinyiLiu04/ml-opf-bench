# -*- coding: utf-8 -*-
"""
Lagrangian Dual DCOPF - Streamlined Version

Implements Lagrangian dual method for DC Optimal Power Flow:
- Predicts only non-slack generator outputs
- Uses PTDF matrix for constraint evaluation (no power flow needed)
- Supports 4 data modes: random_split, valid_fixed, generalization, api_test
- Prints only test set results
- All comments and outputs in English
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys

# Import DCOPF modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dcopf_data_setup import (
    load_parameters_from_csv,
    DataSplitMode,
    split_data_by_mode,
    load_and_prepare_data_generalization
)
from dcopf_violation_metrics import (
    feasibility as dc_feasibility,
    compute_cost,
    compute_cost_gap_percentage,
    compute_branch_violation_pu,
    compute_mae_percentage
)
from dcopf_config import PathConfig

sys.path.append('/home/claude')
from dcopf_slack_utils import (
    identify_slack_bus_and_gens,
    update_params_with_slack_info,
    reconstruct_full_pg,
    compute_detailed_mae,
    compute_detailed_pg_violations_pu
)

from sklearn.preprocessing import MinMaxScaler


# =====================================================================
# Neural Network
# =====================================================================
class LagrangianDNN_DCOPF(nn.Module):
    """
    Lagrangian Dual Neural Network for DCOPF

    Input: Load demand (n_buses)
    Output: Non-slack generator outputs (n_g_non_slack)
    """

    def __init__(self, input_dim, n_g_non_slack, hidden_dims=[128, 128]):
        super().__init__()
        self.n_g_non_slack = n_g_non_slack
        output_dim = n_g_non_slack

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# =====================================================================
# Lagrangian Constraint Violation Computation
# =====================================================================
def compute_constraint_violations(Y_pred_scaled, X_scaled, scalers, params):
    """
    Compute constraint violations for Lagrangian dual method

    Constraints:
    1. Generator limits: Pg_min ≤ Pg ≤ Pg_max
    2. Branch flow limits: |Pl| ≤ Pl_max (via PTDF matrix)

    Args:
        Y_pred_scaled: Scaled predictions, shape (batch, n_g_non_slack)
        X_scaled: Scaled load, shape (batch, n_buses)
        scalers: Scalers dictionary
        params: Parameters dictionary

    Returns:
        violations: Dictionary of violation values (mean across samples)
    """
    # Denormalize predictions
    pg_pred_non_slack = scalers['y_pg_non_slack'].inverse_transform(
        Y_pred_scaled.detach().cpu().numpy()
    )
    x_pd = scalers['x'].inverse_transform(X_scaled.detach().cpu().numpy())

    # Reconstruct full Pg (including slack)
    pd_total = x_pd.sum(axis=1)
    pg_pred_full = reconstruct_full_pg(pg_pred_non_slack, pd_total, params)

    # Calculate constraint violations
    gen_up_viol, gen_lo_viol, line_viol, _ = dc_feasibility(
        pg_pred_full, x_pd, params
    )

    violations = {}

    # Generator constraint violations (all generators)
    violations['nu_pg'] = np.mean(gen_up_viol + gen_lo_viol)

    # Branch flow violations
    violations['nu_branch'] = np.mean(line_viol)

    return violations


# =====================================================================
# Training Function
# =====================================================================
def train_lagrangian_dual(
        model, train_loader, val_loader, scalers, params,
        n_epochs=100, lr=1e-3, rho=1e-2, device='cpu'
):
    """
    Lagrangian Dual training for DCOPF

    Algorithm:
    1. Minimize MSE loss + Lagrangian constraint penalty
    2. Update Lagrangian multipliers using gradient ascent

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        scalers: Data scalers
        params: System parameters
        n_epochs: Number of training epochs
        lr: Learning rate for model parameters (α)
        rho: Step size for Lagrangian multipliers (ρ)
        device: Computing device
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Initialize Lagrangian multipliers
    lambda_multipliers = {
        'lambda_pg': 0.0,
        'lambda_branch': 0.0,
    }

    history = {'train_loss': [], 'val_loss': []}

    print(f"\n{'=' * 70}")
    print(f"Lagrangian Dual Training")
    print(f"{'=' * 70}")
    print(f"Optimizer learning rate α: {lr}, Lagrangian step size ρ: {rho}")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            Y_pred = model(X_batch)

            # MSE loss
            mse_loss = criterion(Y_pred, Y_batch)

            # Constraint violations
            violations = compute_constraint_violations(
                Y_pred, X_batch, scalers, params
            )

            # Lagrangian weighted constraint loss
            constraint_loss = 0.0
            for key, nu_val in violations.items():
                lambda_key = f"lambda_{key.replace('nu_', '')}"
                if lambda_key in lambda_multipliers:
                    constraint_loss += lambda_multipliers[lambda_key] * nu_val

            total_loss = mse_loss + constraint_loss
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        # Update Lagrangian multipliers
        for key, nu_val in violations.items():
            lambda_key = f"lambda_{key.replace('nu_', '')}"
            if lambda_key in lambda_multipliers:
                lambda_multipliers[lambda_key] = max(
                    0.0, lambda_multipliers[lambda_key] + rho * nu_val
                )

        avg_train_loss = epoch_loss / n_batches
        history['train_loss'].append(avg_train_loss)

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val = X_val.to(device)
                Y_val = Y_val.to(device)
                Y_pred_val = model(X_val)
                val_loss += criterion(Y_pred_val, Y_val).item()
                n_val += 1

        avg_val_loss = val_loss / n_val if n_val > 0 else 0.0
        history['val_loss'].append(avg_val_loss)

        # Print every epoch
        print(f"Epoch {epoch + 1:4d}/{n_epochs} - train_loss: {avg_train_loss:.6f} - val_loss: {avg_val_loss:.6f}")

    return history, lambda_multipliers


# =====================================================================
# Evaluation Function
# =====================================================================
def evaluate_model(
        model, X_tensor, indices, raw_data, scalers, params, device,
        test_data_external=None, test_params=None
):
    """
    Evaluate model on test set

    Returns metrics:
    - MAE: non_slack, slack
    - Violations (p.u.): non_slack, slack, branch
    - Cost gap (%)
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

    # Inverse transform non-slack predictions
    y_pred_non_slack = scalers['y_pg_non_slack'].inverse_transform(
        y_pred_non_slack_scaled
    )

    # Reconstruct full Pg vector (including slack)
    pd_total = x_raw_eval.sum(axis=1)
    y_pred_pg_all = reconstruct_full_pg(y_pred_non_slack, pd_total, eval_params)

    # --- 1. Calculate detailed MAE (%) ---
    mae_dict = compute_detailed_mae(
        y_true_pg_all, y_pred_non_slack, y_pred_pg_all, eval_params
    )

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
def lagrangian_dcopf_experiment(
        case_name,
        params_path,
        dataset_path,
        # Data mode parameters
        split_mode=DataSplitMode.RANDOM_SPLIT,
        n_train_use=10000,
        test_data_path=None,
        test_params_path=None,
        n_test_samples=1000,
        # Training parameters
        n_epochs=100,
        learning_rate=0.001,
        lagrangian_lr=0.01,
        hidden_layers=[128, 128],
        batch_size=128,
        seed=42,
        device='cuda',
        column_names=None
):
    """
    Lagrangian Dual DCOPF main experiment function

    Supports four data modes:
    - RANDOM_SPLIT: Random split (10:1:1)
    - VALID_FIXED: Fixed validation/test sets
    - GENERALIZATION: Cross-distribution test
    - API_TEST: Different topology test
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\nRunning: {split_mode.value} - {case_name}\n")

    # ========================================================================
    # 1. Load training parameters
    # ========================================================================
    params = load_parameters_from_csv(case_name, params_path, is_api=False)

    # Automatically identify slack bus and generators
    slack_info = identify_slack_bus_and_gens(params)
    params = update_params_with_slack_info(params, slack_info)

    # If API_TEST mode, load test parameters
    test_params = None
    if split_mode == DataSplitMode.API_TEST:
        if test_params_path is None:
            raise ValueError("API_TEST mode requires test_params_path")
        test_params = load_parameters_from_csv(case_name, test_params_path, is_api=True)
        test_slack_info = identify_slack_bus_and_gens(test_params)
        test_params = update_params_with_slack_info(test_params, test_slack_info)
    else:
        test_params = params

    # Default column names
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

    # ========================================================================
    # 2. Load and prepare training data
    # ========================================================================
    from dnn_dcopf_main import load_and_prepare_data_trad

    x_data_raw, y_pg_raw_non_slack, y_pg_raw_all = load_and_prepare_data_trad(
        dataset_path, params
    )

    raw_data = {
        'x': x_data_raw,
        'y_pg_non_slack': y_pg_raw_non_slack,
        'y_pg_all': y_pg_raw_all
    }

    # ========================================================================
    # 3. Data splitting
    # ========================================================================
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

    # ========================================================================
    # 4. Data normalization
    # ========================================================================
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

    # Convert to PyTorch tensors
    X_train = torch.tensor(x_train_scaled, dtype=torch.float32, device=device)
    Y_train = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)
    X_val = torch.tensor(x_val_scaled, dtype=torch.float32, device=device)
    Y_val = torch.tensor(y_val_scaled, dtype=torch.float32, device=device)

    # Create data loaders
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))

    # ========================================================================
    # 5. Create model and train
    # ========================================================================
    model = LagrangianDNN_DCOPF(
        input_dim=x_data_raw.shape[1],
        n_g_non_slack=params['general']['n_g_non_slack'],
        hidden_dims=hidden_layers
    ).to(device)

    # Training
    t0 = time.perf_counter()
    history, final_lambdas = train_lagrangian_dual(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        scalers=scalers,
        params=params,
        n_epochs=n_epochs,
        lr=learning_rate,
        rho=lagrangian_lr,
        device=device
    )
    train_time = time.perf_counter() - t0

    # ========================================================================
    # 6. Model evaluation (test set only)
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"Test Set Results")
    print(f"{'=' * 70}")

    if split_mode == DataSplitMode.GENERALIZATION:
        test_data_external_dict = {
            'x': x_test_external,
            'y_pg_all': y_test_external
        }
        test_metrics = evaluate_model(
            model, None, None, raw_data, scalers, params, device,
            test_data_external=test_data_external_dict,
            test_params=test_params
        )
    elif split_mode == DataSplitMode.API_TEST:
        test_data_external_dict = {
            'x': x_test_external,
            'y_pg_all': y_test_external
        }
        test_metrics = evaluate_model(
            model, None, None, raw_data, scalers, params, device,
            test_data_external=test_data_external_dict,
            test_params=test_params
        )
    else:
        test_metrics = evaluate_model(
            model, X_test, test_idx, raw_data, scalers, params, device,
            test_params=test_params
        )

    # ========================================================================
    # 7. Inference speed test
    # ========================================================================
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

    # ========================================================================
    # 8. Print final results
    # ========================================================================
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


# =====================================================================
# Main Function
# =====================================================================
if __name__ == "__main__":
    # ===================================================================
    # Experiment Configuration
    # ===================================================================

    # --- 1. Case Configuration ---
    CASE_NAME = 'pglib_opf_case30_ieee'
    CASE_SHORT_NAME = 'case30'

    # --- 2. Data Split Mode ---
    SPLIT_MODE = DataSplitMode.RANDOM_SPLIT

    # --- 3. Training & Test Sample Counts ---
    N_TRAIN_USE = 10000
    N_TEST_SAMPLES = 2483

    # --- 4. Training Hyperparameters ---
    N_EPOCHS = 100
    LEARNING_RATE = 0.01
    LAGRANGIAN_LR = 0.1  # Lagrangian step size ρ
    BATCH_SIZE = 128
    HIDDEN_LAYERS = [128, 128]
    SEED = 42

    # --- 5. Path Configuration ---
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
    results = lagrangian_dcopf_experiment(
        case_name=CASE_NAME,
        params_path=params_path,
        dataset_path=train_data_path,
        split_mode=SPLIT_MODE,
        n_train_use=N_TRAIN_USE,
        test_data_path=test_data_path,
        test_params_path=test_params_path,
        n_test_samples=N_TEST_SAMPLES,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        lagrangian_lr=LAGRANGIAN_LR,
        hidden_layers=HIDDEN_LAYERS,
        batch_size=BATCH_SIZE,
        seed=SEED,
        device=device_name,
        column_names=COLUMN_NAMES
    )