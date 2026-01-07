# -*- coding: utf-8 -*-
"""
Lagrangian Dual ACOPF (V8 - Simplified Output)
Based on paper: Fioretto et al. 2019

Modifications (V8):
- DNN only predicts non-slack pg and generator vm
- Simplified output: only test set metrics
- All comments and outputs in English
- Removed all file saving (JSON/CSV/model)
- All violations in p.u. units
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import sys
from pathlib import Path

# PyPower
from pypower.api import runpf, ppoption

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
        load_api_test_data,
        reconstruct_full_pg  # Import reconstruction function
    )
except ImportError:
    print("Error: Unable to import 'acopf_data_setup' module.")
    sys.exit(1)

# Import evaluation modules
try:
    from acopf_violation_metrics import evaluate_acopf_predictions
except ImportError:
    print("Error: Unable to import 'acopf_violation_metrics' module.")
    sys.exit(1)

# =====================================================================
# PyPower Interface
# =====================================================================
GLOBAL_CASE_DATA = None
PPOPT = None


def init_pypower_options():
    global PPOPT
    ppopt = ppoption()
    PPOPT = ppoption(ppopt, OUT_ALL=0, VERBOSE=0, ENFORCE_Q_LIMS=0)


def load_case_from_csv(case_name, constraints_path):
    """
    Load PyPower case data from CSV files

    Note: case_name should be the full case name (including __api suffix)
    """
    base_path = Path(constraints_path)

    base_mva_df = pd.read_csv(base_path / f"{case_name}_base_mva.csv")
    bus_df = pd.read_csv(base_path / f"{case_name}_bus_data.csv")
    gen_df = pd.read_csv(base_path / f"{case_name}_gen_data.csv")
    branch_df = pd.read_csv(base_path / f"{case_name}_branch_data.csv")

    baseMVA = base_mva_df['value'].iloc[0]

    # BUS Matrix
    bus = np.zeros((len(bus_df), 13))
    bus[:, 0] = bus_df['bus_id'].values
    bus[:, 1] = bus_df['type'].values
    bus[:, 2] = bus_df['pd_pu'].values * baseMVA
    bus[:, 3] = bus_df['qd_pu'].values * baseMVA
    bus[:, 6] = 1
    bus[:, 7] = bus_df['vm_pu'].values
    bus[:, 8] = bus_df['va_deg'].values
    bus[:, 9] = bus_df['base_kv'].values if 'base_kv' in bus_df.columns else 1.0
    bus[:, 10] = 1
    bus[:, 11] = bus_df['vmax_pu'].values
    bus[:, 12] = bus_df['vmin_pu'].values

    # GEN Matrix
    gen = np.zeros((len(gen_df), 21))
    gen[:, 0] = gen_df['bus_id'].values
    gen[:, 3] = gen_df['qg_max_pu'].values * baseMVA
    gen[:, 4] = gen_df['qg_min_pu'].values * baseMVA
    gen[:, 5] = gen_df['vg_pu'].values
    gen[:, 6] = baseMVA
    gen[:, 7] = 1
    gen[:, 8] = gen_df['pg_max_pu'].values * baseMVA
    gen[:, 9] = gen_df['pg_min_pu'].values * baseMVA

    # BRANCH Matrix
    branch = np.zeros((len(branch_df), 13))
    branch[:, 0] = branch_df['f_bus'].values
    branch[:, 1] = branch_df['t_bus'].values
    branch[:, 2] = branch_df['r_pu'].values
    branch[:, 3] = branch_df['x_pu'].values
    branch[:, 4] = branch_df['b_pu'].values
    branch[:, 5] = branch_df['rate_a_pu'].values * baseMVA
    branch[:, 6] = branch[:, 5]
    branch[:, 7] = branch[:, 5]
    branch[:, 8] = branch_df['tap_ratio'].values
    branch[:, 9] = branch_df['shift_deg'].values
    branch[:, 10] = 1
    branch[:, 11] = -360
    branch[:, 12] = 360

    rate_a_values = branch_df['rate_a_pu'].values
    branch[:, 5:8][np.isnan(rate_a_values) | np.isinf(rate_a_values), :] = 9900.0

    # GENCOST Matrix
    gencost = np.zeros((len(gen_df), 7))
    gencost[:, 0] = 2
    gencost[:, 3] = 3
    gencost[:, 4] = gen_df['cost_c2'].values
    gencost[:, 5] = gen_df['cost_c1'].values
    gencost[:, 6] = gen_df['cost_c0'].values

    ppc = {
        'version': '2',
        'baseMVA': baseMVA,
        'bus': bus,
        'gen': gen,
        'branch': branch,
        'gencost': gencost
    }

    return ppc


def solve_pf_for_evaluation(pd, qd, pg_non_slack, vm_gen, params):
    """
    Run power flow calculation

    Args:
        pd: Load active power (p.u.)
        qd: Load reactive power (p.u.)
        pg_non_slack: Non-slack generator active power (p.u.)
        vm_gen: Generator bus voltage (p.u.)
        params: Parameters dictionary
    """
    global GLOBAL_CASE_DATA, PPOPT

    BASE_MVA = params['general']['BASE_MVA']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    n_gen = params['general']['n_gen']
    load_bus_ids = params['general']['load_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    mpc = {
        'version': GLOBAL_CASE_DATA['version'],
        'baseMVA': GLOBAL_CASE_DATA['baseMVA'],
        'bus': GLOBAL_CASE_DATA['bus'].copy(),
        'gen': GLOBAL_CASE_DATA['gen'].copy(),
        'branch': GLOBAL_CASE_DATA['branch'].copy(),
        'gencost': GLOBAL_CASE_DATA['gencost']
    }

    # Set loads
    for i, bus_id in enumerate(load_bus_ids):
        bus_idx = bus_id_to_idx.get(int(bus_id))
        if bus_idx is not None:
            mpc["bus"][bus_idx, 2] = pd[i] * BASE_MVA
            mpc["bus"][bus_idx, 3] = qd[i] * BASE_MVA

    # Set generator active power (only non-slack generators)
    for i, gen_idx in enumerate(non_slack_gen_idx):
        mpc["gen"][gen_idx, 1] = pg_non_slack[i] * BASE_MVA

    # Set generator voltage
    for g in range(n_gen):
        mpc["gen"][g, 5] = vm_gen[g]

    result = runpf(mpc, PPOPT)
    return result


# =====================================================================
# Neural Network
# =====================================================================
class LagrangianDNN_ACOPF(nn.Module):
    def __init__(self, input_dim, n_gen_non_slack, n_gen, hidden_dims=[256, 256]):
        super().__init__()
        self.n_gen_non_slack = n_gen_non_slack
        self.n_gen = n_gen
        output_dim = n_gen_non_slack + n_gen  # Pg_non_slack + Vm_gen

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
def compute_constraint_violations(Y_pred_scaled, scalers, params):
    """
    Compute constraint violations (Lagrangian paper's violation degrees)

    Args:
        Y_pred_scaled: Scaled predictions, shape (batch, n_gen_non_slack + n_gen)
        scalers: Scalers dictionary
        params: Parameters dictionary
    """
    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    # Denormalize
    pg_pred_non_slack = scalers['pg'].inverse_transform(Y_pred_scaled[:, :n_gen_non_slack])
    vm_pred_gen = scalers['vm'].inverse_transform(Y_pred_scaled[:, n_gen_non_slack:])

    # Reconstruct full arrays for constraint checking
    pg_pred_full = np.zeros((Y_pred_scaled.shape[0], n_gen))
    for i in range(len(pg_pred_non_slack)):
        pg_pred_full[i] = reconstruct_full_pg(pg_pred_non_slack[i], params)

    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    vm_pred_all = np.zeros((Y_pred_scaled.shape[0], n_buses))
    vm_pred_all[:, gen_bus_indices] = vm_pred_gen
    non_gen_mask = np.ones(n_buses, dtype=bool)
    non_gen_mask[gen_bus_indices] = False
    vm_pred_all[:, non_gen_mask] = 1.0

    # Get constraints
    pg_min = params['generator']['pg_min'].flatten()
    pg_max = params['generator']['pg_max'].flatten()
    vm_min = params['bus']['vm_min']
    vm_max = params['bus']['vm_max']

    violations = {}

    # Voltage magnitude constraints (all buses)
    vm_lower = np.maximum(0, vm_min - vm_pred_all)
    vm_upper = np.maximum(0, vm_pred_all - vm_max)
    violations['nu_2a_vm'] = np.mean(vm_lower + vm_upper)

    # Active power constraints (all generators including slack)
    pg_lower = np.maximum(0, pg_min - pg_pred_full)
    pg_upper = np.maximum(0, pg_pred_full - pg_max)
    violations['nu_3a_pg'] = np.mean(pg_lower + pg_upper)

    # Other constraints (simplified)
    violations['nu_3b_qg'] = 0.0
    violations['nu_6a_active'] = 0.0
    violations['nu_6b_reactive'] = 0.0

    return violations


# =====================================================================
# Training Function
# =====================================================================
def train_lagrangian_dual(model, train_loader, val_loader, scalers, params,
                          n_epochs=100, lr=1e-3, rho=1e-2, device='cpu'):
    """
    Lagrangian Dual training (Paper Algorithm 1)

    Note: The Lagrangian algorithm itself is NOT modified, only the input/output dimensions
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Initialize Lagrangian multipliers
    lambda_multipliers = {
        'lambda_2a_vm': 0.0,
        'lambda_3a_pg': 0.0,
        'lambda_3b_qg': 0.0,
        'lambda_6a_active': 0.0,
        'lambda_6b_reactive': 0.0,
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
                Y_pred.detach().cpu().numpy(), scalers, params
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

        # Update Lagrangian multipliers (Algorithm 1 - unchanged)
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

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch + 1:4d}/{n_epochs} - Train Loss={avg_train_loss:.6f} - Val Loss={avg_val_loss:.6f}")

    return history, lambda_multipliers


# =====================================================================
# Evaluation Function
# =====================================================================
def evaluate_model(model, X_test, test_idx, raw_data, scalers, params, device, split_name, verbose=True):
    """Evaluate model using unified evaluation module"""
    if verbose:
        print(f"\n{split_name} Evaluation:")

    model.eval()

    with torch.no_grad():
        Y_pred_scaled = model(X_test.to(device)).cpu().numpy()

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    # Denormalize
    y_pred_pg_non_slack = scalers['pg'].inverse_transform(Y_pred_scaled[:, :n_gen_non_slack])
    y_pred_vm_gen = scalers['vm'].inverse_transform(Y_pred_scaled[:, n_gen_non_slack:])

    # Reconstruct full arrays
    y_pred_pg_full = reconstruct_full_pg(y_pred_pg_non_slack, params)

    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    y_pred_vm_all = np.zeros((len(X_test), n_buses), dtype=y_pred_vm_gen.dtype)
    y_pred_vm_all[:, gen_bus_indices] = y_pred_vm_gen
    non_gen_mask = np.ones(n_buses, dtype=bool)
    non_gen_mask[gen_bus_indices] = False
    y_pred_vm_all[:, non_gen_mask] = 1.0

    # Extract loads
    x_raw = scalers['x'].inverse_transform(X_test.cpu().numpy())
    pd_pu = x_raw[:, :n_loads]
    qd_pu = x_raw[:, n_loads:]

    # True values
    y_true_pg = raw_data['pg'][test_idx]
    y_true_vm = raw_data['vm'][test_idx]
    y_true_qg = raw_data['qg'][test_idx]
    y_true_va_rad = raw_data['va'][test_idx]

    # Run power flow
    n_samples = len(X_test)
    pf_results_list = []
    converge_flags = []

    if verbose:
        print(f"  Computing power flow for {n_samples} samples...")

    for i in range(n_samples):
        try:
            r1_pf = solve_pf_for_evaluation(pd_pu[i], qd_pu[i], y_pred_pg_non_slack[i], y_pred_vm_gen[i], params)
            pf_results_list.append(r1_pf)
            converge_flags.append(r1_pf[0]['success'])
        except:
            pf_results_list.append(({'success': False, 'gen': np.zeros((n_gen, 21)),
                                     'bus': np.zeros((n_buses, 13)), 'branch': np.zeros((1, 17))},))
            converge_flags.append(False)

    if verbose:
        print(f"    ✓ Converged: {sum(converge_flags)}/{n_samples}")

    # Call unified evaluation module
    metrics = evaluate_acopf_predictions(
        y_pred_pg=y_pred_pg_full,
        y_pred_vm=y_pred_vm_all,
        y_true_pg=y_true_pg,
        y_true_vm=y_true_vm,
        y_true_qg=y_true_qg,
        y_true_va_rad=y_true_va_rad,
        pf_results_list=pf_results_list,
        converge_flags=converge_flags,
        params=params,
        verbose=verbose
    )

    return metrics


# =====================================================================
# Main Experiment Function
# =====================================================================
def lagrangian_acopf_experiment(
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
        n_epochs=500,
        learning_rate=0.001,
        lagrangian_lr=0.01,  # Lagrangian specific parameter
        hidden_sizes=[256, 256],
        batch_size=256,
        seed=42,
        device='cuda'
):
    """
    Lagrangian Dual ACOPF main experiment function

    Supports four data modes:
    - RANDOM_SPLIT: Random split
    - FIXED_VALTEST: Fixed validation/test sets
    - GENERALIZATION: Cross-distribution generalization test
    - API_TEST: API data test (different topology)
    """
    global GLOBAL_CASE_DATA

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print(f"Lagrangian Dual ACOPF Experiment")
    print(f"{'=' * 70}")
    print(f"Case: {case_name}")
    print(f"Data Mode: {data_mode}")
    print(f"Device: {device}")
    print(f"{'=' * 70}")

    # ========================================================================
    # 1. Load training parameters and PyPower case data
    # ========================================================================
    print(f"\n[Step 1] Loading training parameters and case data...")
    init_pypower_options()
    params = load_parameters_from_csv(case_name, params_path)
    GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)
    print(f"  ✓ Training params and PyPower case data loaded")

    # ========================================================================
    # 2. Load training data and fit scalers
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
    if cost_baseline:
        print(f"  Cost Baseline: {cost_baseline:.2f} $/h")

    # ========================================================================
    # 3. Load and split data based on data mode
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
        GLOBAL_CASE_DATA_TEST = GLOBAL_CASE_DATA

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
        GLOBAL_CASE_DATA_TEST = GLOBAL_CASE_DATA

    # ========================================================================
    # 4. Create DataLoader
    # ========================================================================
    print(f"\n[Dataset Sizes]")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")

    train_dataset = TensorDataset(
        torch.tensor(x_data_scaled[train_idx], dtype=torch.float32),
        torch.tensor(y_data_scaled[train_idx], dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(x_data_scaled[val_idx], dtype=torch.float32),
        torch.tensor(y_data_scaled[val_idx], dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))

    # ========================================================================
    # 5. Create model and train
    # ========================================================================
    print(f"\n[Step 3] Creating model...")
    model = LagrangianDNN_ACOPF(
        input_dim=x_data_scaled.shape[1],
        n_gen_non_slack=n_gen_non_slack,
        n_gen=n_gen,
        hidden_dims=hidden_sizes
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Network: {x_data_scaled.shape[1]} -> {' -> '.join(map(str, hidden_sizes))} -> {n_gen_non_slack + n_gen}")
    print(f"  Total params: {total_params:,}")

    # Training
    print(f"\n[Step 4] Training...")
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

    print(f"\n✓ Training completed in {train_time:.2f} seconds")

    # ========================================================================
    # 6. Model evaluation (using test params)
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"Test Set Evaluation")
    print(f"{'=' * 70}")

    GLOBAL_CASE_DATA_BACKUP = GLOBAL_CASE_DATA
    GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_TEST

    if data_mode == DataMode.API_TEST:
        test_split_name = "API Test"
    elif data_mode == DataMode.GENERALIZATION:
        test_split_name = "Generalization Test"
    else:
        test_split_name = "Test"

    X_test = torch.tensor(test_x_scaled[test_idx], dtype=torch.float32)

    test_metrics = evaluate_model(
        model, X_test, test_idx, test_raw_data, scalers,
        test_params,
        device, test_split_name, verbose=True
    )

    GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_BACKUP

    # ========================================================================
    # 7. Inference speed test
    # ========================================================================
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(100):
            t_start = time.perf_counter()
            _ = model(X_test[:1].to(device))
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t_start)

    latency_ms = np.mean(times) * 1000

    # ========================================================================
    # 8. Print final results (simplified)
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

    # No file saving - removed all JSON/CSV saving

    return test_metrics


# =====================================================================
# Main Function
# =====================================================================
if __name__ == "__main__":
    # =========================================================================
    # Read configuration from acopf_config.py and run experiment
    # =========================================================================

    # Lagrangian specific parameters
    LAGRANGIAN_LR = 0.1  # Lagrangian step size ρ

    # Print configuration
    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)

    # Get all paths
    paths = acopf_config.get_all_paths()

    # Get all training parameters
    params = acopf_config.get_all_params()

    # Add Lagrangian specific parameters
    params['lagrangian_lr'] = LAGRANGIAN_LR

    print(f"\n[Lagrangian Specific Configuration]")
    print(f"  Lagrangian step size ρ: {LAGRANGIAN_LR}")
    print("=" * 70)

    # Execute experiment
    results = lagrangian_acopf_experiment(**paths, **params)

    print("\n✓ Experiment completed successfully!")