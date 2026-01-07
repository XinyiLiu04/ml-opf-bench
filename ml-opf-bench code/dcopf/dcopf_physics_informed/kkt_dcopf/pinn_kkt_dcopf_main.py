# -*- coding: utf-8 -*-
"""
PINN-KKT for DCOPF - Main Experiment Script
Physics-Informed Neural Network with Explicit KKT Conditions

Version: v5.0 - Aligned with DNN Version

Features:
- Explicit modeling of KKT conditions
- Multi-task learning (Pg_non_slack + all dual variables)
- Weighted MAE loss
- Auto-identify and handle Slack Bus
- Support API_TEST mode with dual constraint parameters

Evaluation Metrics (6 metrics aligned with DNN):
- MAE Pg (%) - Non-Slack, Slack
- Pg Violation (p.u., Mean of Max) - Non-Slack, Slack
- Branch Violation (p.u., Mean of Max)
- Cost Gap (%)
- Training Time (s)
- Inference Time (ms)

PINN-KKT Specifics:
- Pg output dimension: n_g_non_slack (predict only non-Slack)
- Dual variable dimension: n_g (all generators, because Slack also has constraints)
- Forward reconstructs full Pg for KKT error
- Evaluation reconstructs full Pg for violation calculation
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

# Import Slack-aware PINN-KKT model
from PinnModel import PinnModel


# =====================================================================
# Data Loading Function
# =====================================================================

def load_and_prepare_pinn_kkt_data(file_path, params, column_names):
    """
    Load and prepare PINN-KKT training data

    PINN-KKT requires all outputs: Pg + dual variables
    Returns 3 Pg values: y_pg_raw_non_slack (for training), y_pg_raw_all (for evaluation)
    Dual variable dimensions keep n_g (all generators)
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

    # Load dual variables (PINN-KKT specific) - keep n_g dimension
    y_lambda_raw = full_df[column_names['lambda']].values.reshape(-1, 1).astype('float32')

    mu_g_min_cols = [f"{column_names['mu_g_min_prefix']}{i}" for i in params['general']['g_bus']]
    y_mu_g_min_raw = full_df[mu_g_min_cols].values.astype('float32')

    mu_g_max_cols = [f"{column_names['mu_g_max_prefix']}{i}" for i in params['general']['g_bus']]
    y_mu_g_max_raw = full_df[mu_g_max_cols].values.astype('float32')

    valid_branch_indices = np.where(params['constraints']['Pl_max'] < 1e10)[0]
    valid_branch_ids = params['general']['branch_ids'][valid_branch_indices]

    mu_line_pos_cols = [f"{column_names['mu_line_pos_prefix']}{i}" for i in valid_branch_ids]
    y_mu_line_pos_raw = full_df[mu_line_pos_cols].values.astype('float32')

    mu_line_neg_cols = [f"{column_names['mu_line_neg_prefix']}{i}" for i in valid_branch_ids]
    y_mu_line_neg_raw = full_df[mu_line_neg_cols].values.astype('float32')

    return (x_data_raw, y_pg_raw_non_slack, y_pg_raw_all,
            y_lambda_raw, y_mu_g_min_raw, y_mu_g_max_raw,
            y_mu_line_pos_raw, y_mu_line_neg_raw)


# =====================================================================
# Evaluation Function
# =====================================================================

def evaluate_model(
        model,
        X_tensor,
        indices,
        raw_data_dict,
        scalers,
        params,
        device,
        test_data_external=None,
        test_params=None
):
    """Evaluate PINN-KKT model (aligned with DNN)"""

    eval_params = test_params if test_params is not None else params

    model.eval()

    if test_data_external is not None:
        x_raw = test_data_external['x']
        y_true_pg_all = test_data_external['y_pg_all']

        x_scaled = scalers['x'].transform(x_raw)
        X_eval = torch.tensor(x_scaled, dtype=torch.float32, device=device)

        with torch.no_grad():
            outputs = model(X_eval)
            y_pred_pg_non_slack_scaled = outputs[0]
    else:
        y_true_pg_all = raw_data_dict['y_pg_all'][indices]
        x_raw = raw_data_dict['x'][indices]

        with torch.no_grad():
            outputs = model(X_tensor)
            y_pred_pg_non_slack_scaled = outputs[0]

    y_pred_pg_non_slack_scaled_np = y_pred_pg_non_slack_scaled.cpu().numpy()
    y_pred_pg_non_slack = scalers['pg_non_slack'].inverse_transform(y_pred_pg_non_slack_scaled_np)

    # Reconstruct full Pg
    pd_total = x_raw.sum(axis=1)
    y_pred_pg_all = reconstruct_full_pg(
        pg_non_slack=y_pred_pg_non_slack,
        pd_total=pd_total,
        params=eval_params
    )

    # Calculate MAE
    mae_dict = compute_detailed_mae(
        y_true_all=y_true_pg_all,
        y_pred_non_slack=y_pred_pg_non_slack,
        y_pred_all=y_pred_pg_all,
        params=eval_params
    )

    # Calculate violations (p.u.)
    gen_up_viol, gen_lo_viol, line_viol, balance_err = feasibility(
        y_pred_pg=y_pred_pg_all,
        x_pd=x_raw,
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

    return {
        'mae_pg_non_slack': mae_dict['mae_non_slack'],
        'mae_pg_slack': mae_dict['mae_slack'],
        'viol_pg_non_slack': viol_dict['viol_non_slack'],
        'viol_pg_slack': viol_dict['viol_slack'],
        'viol_branch': viol_branch_pu,
        'cost_gap_percent': cost_gap_pct,
    }


# =====================================================================
# Main Training Function
# =====================================================================

def train_pinn_kkt_dcopf(
        case_name,
        params_path,
        dataset_path,
        column_names,
        n_train_use=10000,
        neurons_pg=[128, 128],
        neurons_lm=[128, 128],
        n_epochs=100,
        batch_size=128,
        learning_rate=1e-3,
        weight1=0.005,
        weight2=0.005,
        seed=42,
        device='cuda',
        split_mode=DataSplitMode.RANDOM_SPLIT,
        test_data_path=None,
        scale_type='minmax',
        test_params_path=None,
        n_test_samples=1000
):
    """PINN-KKT for DCOPF training function (aligned with DNN version)"""

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)

    # Load parameters
    params = load_parameters_from_csv(case_name, params_path, is_api=False)
    slack_info = identify_slack_bus_and_gens(params)
    params = update_params_with_slack_info(params, slack_info)

    test_params = None
    if split_mode == DataSplitMode.API_TEST:
        if test_params_path is None:
            raise ValueError('API_TEST mode requires test_params_path')
        test_params = load_parameters_from_csv(case_name, test_params_path, is_api=True)
        test_slack_info = identify_slack_bus_and_gens(test_params)
        test_params = update_params_with_slack_info(test_params, test_slack_info)
    else:
        test_params = params

    n_buses = params['general']['n_buses']
    n_gen = params['general']['n_g']
    n_gen_non_slack = params['general']['n_g_non_slack']
    n_line = params['general']['n_line']

    # Load data
    (x_data_raw, y_pg_raw_non_slack, y_pg_raw_all,
     y_lambda_raw, y_mu_g_min_raw, y_mu_g_max_raw,
     y_mu_line_pos_raw, y_mu_line_neg_raw) = load_and_prepare_pinn_kkt_data(
        dataset_path, params, column_names
    )

    raw_data_dict = {
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
    if scale_type == 'minmax':
        x_scaler = MinMaxScaler().fit(x_data_raw[train_idx])
    elif scale_type == 'standard':
        x_scaler = StandardScaler().fit(x_data_raw[train_idx])
    else:
        x_scaler = MinMaxScaler().fit(x_data_raw[train_idx])

    pg_non_slack_scaler = MinMaxScaler().fit(y_pg_raw_non_slack[train_idx])
    lambda_scaler = MinMaxScaler().fit(y_lambda_raw[train_idx])
    mu_g_min_scaler = MinMaxScaler().fit(y_mu_g_min_raw[train_idx])
    mu_g_max_scaler = MinMaxScaler().fit(y_mu_g_max_raw[train_idx])
    mu_line_pos_scaler = MinMaxScaler().fit(y_mu_line_pos_raw[train_idx])
    mu_line_neg_scaler = MinMaxScaler().fit(y_mu_line_neg_raw[train_idx])

    scalers = {
        'x': x_scaler,
        'pg_non_slack': pg_non_slack_scaler,
        'lambda': lambda_scaler,
        'mu_g_min': mu_g_min_scaler,
        'mu_g_max': mu_g_max_scaler,
        'mu_line_pos': mu_line_pos_scaler,
        'mu_line_neg': mu_line_neg_scaler
    }

    x_train_scaled = x_scaler.transform(x_data_raw[train_idx])
    y_pg_train_scaled = pg_non_slack_scaler.transform(y_pg_raw_non_slack[train_idx])
    y_lambda_train_scaled = lambda_scaler.transform(y_lambda_raw[train_idx])
    y_mu_g_min_train_scaled = mu_g_min_scaler.transform(y_mu_g_min_raw[train_idx])
    y_mu_g_max_train_scaled = mu_g_max_scaler.transform(y_mu_g_max_raw[train_idx])
    y_mu_line_pos_train_scaled = mu_line_pos_scaler.transform(y_mu_line_pos_raw[train_idx])
    y_mu_line_neg_train_scaled = mu_line_neg_scaler.transform(y_mu_line_neg_raw[train_idx])

    x_val_scaled = x_scaler.transform(x_data_raw[val_idx])
    y_pg_val_scaled = pg_non_slack_scaler.transform(y_pg_raw_non_slack[val_idx])
    y_lambda_val_scaled = lambda_scaler.transform(y_lambda_raw[val_idx])
    y_mu_g_min_val_scaled = mu_g_min_scaler.transform(y_mu_g_min_raw[val_idx])
    y_mu_g_max_val_scaled = mu_g_max_scaler.transform(y_mu_g_max_raw[val_idx])
    y_mu_line_pos_val_scaled = mu_line_pos_scaler.transform(y_mu_line_pos_raw[val_idx])
    y_mu_line_neg_val_scaled = mu_line_neg_scaler.transform(y_mu_line_neg_raw[val_idx])

    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        X_test = None
    else:
        x_test_scaled = x_scaler.transform(x_data_raw[test_idx])
        X_test = torch.tensor(x_test_scaled, dtype=torch.float32)

    X_train = torch.from_numpy(x_train_scaled).float().to(device)
    Y_train_pg = torch.from_numpy(y_pg_train_scaled).float().to(device)
    Y_train_lambda = torch.from_numpy(y_lambda_train_scaled).float().to(device)
    Y_train_mu_g_min = torch.from_numpy(y_mu_g_min_train_scaled).float().to(device)
    Y_train_mu_g_max = torch.from_numpy(y_mu_g_max_train_scaled).float().to(device)
    Y_train_mu_line_pos = torch.from_numpy(y_mu_line_pos_train_scaled).float().to(device)
    Y_train_mu_line_neg = torch.from_numpy(y_mu_line_neg_train_scaled).float().to(device)
    Y_train_physics = torch.zeros((len(X_train), 1), dtype=torch.float32, device=device)

    X_val = torch.from_numpy(x_val_scaled).float().to(device)
    Y_val_pg = torch.from_numpy(y_pg_val_scaled).float().to(device)
    Y_val_lambda = torch.from_numpy(y_lambda_val_scaled).float().to(device)
    Y_val_mu_g_min = torch.from_numpy(y_mu_g_min_val_scaled).float().to(device)
    Y_val_mu_g_max = torch.from_numpy(y_mu_g_max_val_scaled).float().to(device)
    Y_val_mu_line_pos = torch.from_numpy(y_mu_line_pos_val_scaled).float().to(device)
    Y_val_mu_line_neg = torch.from_numpy(y_mu_line_neg_val_scaled).float().to(device)
    Y_val_physics = torch.zeros((len(X_val), 1), dtype=torch.float32, device=device)

    train_dataset = Data.TensorDataset(
        X_train, Y_train_pg, Y_train_lambda, Y_train_mu_g_min,
        Y_train_mu_g_max, Y_train_mu_line_pos, Y_train_mu_line_neg, Y_train_physics
    )
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Prepare simulation parameters for PINN model
    simulation_parameters = params.copy()
    simulation_parameters['training'] = {
        'neurons_in_hidden_layers_Pg': neurons_pg,
        'neurons_in_hidden_layers_Lm': neurons_lm
    }

    if scale_type == 'minmax':
        simulation_parameters['pd_scale_type'] = 'minmax'
        simulation_parameters['pd_min'] = x_scaler.data_min_
        simulation_parameters['pd_max'] = x_scaler.data_max_
    elif scale_type == 'standard':
        simulation_parameters['pd_scale_type'] = 'standard'
        simulation_parameters['pd_mean'] = x_scaler.mean_
        simulation_parameters['pd_std'] = x_scaler.scale_
    else:
        simulation_parameters['pd_scale_type'] = None

    # Calculate Lg_Max
    Lg_Max = [
        np.max(np.abs(y_lambda_raw)),
        np.max(np.abs(y_mu_g_max_raw)),
        np.max(np.abs(y_mu_g_min_raw)),
        np.max(np.abs(y_mu_line_pos_raw)),
        np.max(np.abs(y_mu_line_neg_raw))
    ]
    simulation_parameters['Lg_Max'] = Lg_Max

    # Build model
    model = PinnModel(
        weight1=weight1,
        weight2=weight2,
        simulation_parameters=simulation_parameters,
        learning_rate=learning_rate,
        device=device
    )

    # Training
    train_losses = []
    val_losses = []

    training_start = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_data in train_loader:
            X_batch = batch_data[0]
            Y_batch = batch_data[1:]

            model.optimizer.zero_grad()
            outputs = model(X_batch)
            loss, _ = model.compute_loss(outputs, Y_batch)
            loss.backward()
            model.optimizer.step()

            epoch_loss += loss.item() * len(X_batch)

        train_loss = epoch_loss / len(X_train)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_targets = (Y_val_pg, Y_val_lambda, Y_val_mu_g_min,
                           Y_val_mu_g_max, Y_val_mu_line_pos, Y_val_mu_line_neg, Y_val_physics)
            val_loss_tensor, _ = model.compute_loss(val_outputs, val_targets)
            val_loss = val_loss_tensor.item()
            val_losses.append(val_loss)

        # Print every epoch (aligned with DNN)
        print(f"Epoch {epoch}/{n_epochs} - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

    train_time = time.time() - training_start

    # Evaluation (test set only)
    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        test_data_external_dict = {
            'x': x_test_external,
            'y_pg_all': y_test_external
        }
        test_metrics = evaluate_model(
            model=model,
            X_tensor=None,
            indices=None,
            raw_data_dict=raw_data_dict,
            scalers=scalers,
            params=params,
            device=device,
            test_data_external=test_data_external_dict,
            test_params=test_params
        )
    else:
        test_metrics = evaluate_model(
            model=model,
            X_tensor=X_test.to(device),
            indices=test_idx,
            raw_data_dict=raw_data_dict,
            scalers=scalers,
            params=params,
            device=device,
            test_params=test_params
        )

    # Speed test
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

    inference_time_ms = np.mean(times) * 1000

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
    print(f"\nTraining Time:   {train_time:.2f} s")
    print(f"Inference Time:  {inference_time_ms:.4f} ms")
    print("\n" + "=" * 70 + "\n")


# =====================================================================
# Main Program
# =====================================================================

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

    # --- 4. PINN-KKT Hyperparameters ---
    N_EPOCHS = 100
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    NEURONS_PG = [128, 128]
    NEURONS_LM = [128, 128]
    WEIGHT1 = 0.005
    WEIGHT2 = 0.005
    SCALE_TYPE = 'minmax'
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
    train_pinn_kkt_dcopf(
        case_name=CASE_NAME,
        params_path=params_path,
        dataset_path=train_data_path,
        column_names=COLUMN_NAMES,
        n_train_use=N_TRAIN_USE,
        neurons_pg=NEURONS_PG,
        neurons_lm=NEURONS_LM,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight1=WEIGHT1,
        weight2=WEIGHT2,
        seed=SEED,
        device=device_name,
        split_mode=SPLIT_MODE,
        test_data_path=test_data_path,
        scale_type=SCALE_TYPE,
        test_params_path=test_params_path,
        n_test_samples=N_TEST_SAMPLES
    )