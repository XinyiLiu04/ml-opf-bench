# -*- coding: utf-8 -*-
"""
ACOPF PINN Main Experiment File (PyTorch Version)

Key Features:
1. Integrated with acopf_config.py configuration system
2. Supports 4 data modes: random_split, fixed_valtest, generalization, api_test
3. Only predicts non-slack generator Pg and generator bus Vm
4. Uses PyPower for power flow validation
5. Uses acopf_violation_metrics.py for evaluation
6. Simplified output: test set metrics only
7. All computations in p.u. units
8. No file saving (JSON/CSV/model)

Author: Auto-generated
Date: 2025-01-06
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import os
import sys
from pathlib import Path

from pypower.runpf import runpf
from pypower.ppoption import ppoption

# Import configuration
try:
    import acopf_config
except ImportError:
    print("Error: Unable to import acopf_config.py")
    sys.exit(1)

try:
    from acopf_data_setup import (
        load_parameters_from_csv,
        load_and_scale_acopf_data,
        DataMode,
        prepare_data_splits,
        load_generalization_test_data,
        load_api_test_data,
        reconstruct_full_pg
    )
except ImportError:
    print("Warning: Unable to import 'acopf_data_setup' module.")
    sys.exit(1)

try:
    from acopf_violation_metrics import evaluate_acopf_predictions
except ImportError:
    print("Warning: Unable to import 'acopf_violation_metrics' module.")
    sys.exit(1)

try:
    from acopf_PinnModel import PinnModel
except ImportError:
    print("Warning: Unable to import 'acopf_PinnModel' module.")
    sys.exit(1)

GLOBAL_CASE_DATA = None
PPOPT = None


# =====================================================================
# PyPower Helper Functions (identical to DNN version)
# =====================================================================
def init_pypower_options():
    """Initialize PyPower options"""
    global PPOPT
    ppopt = ppoption()
    PPOPT = ppoption(ppopt, OUT_ALL=0, VERBOSE=0, ENFORCE_Q_LIMS=0)


def load_case_from_csv(case_name, constraints_path):
    """Load PyPower case data from CSV files (identical to DNN version)"""
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
    bus[:, 2] = bus_df['pd_pu'].values
    bus[:, 3] = bus_df['qd_pu'].values
    bus[:, 6] = 1
    bus[:, 7] = bus_df['vm_pu'].values
    bus[:, 8] = bus_df['va_deg'].values
    bus[:, 9] = bus_df['base_kv'].values
    bus[:, 10] = 1
    bus[:, 11] = bus_df['vmax_pu'].values
    bus[:, 12] = bus_df['vmin_pu'].values

    # GEN Matrix
    gen = np.zeros((len(gen_df), 21))
    gen[:, 0] = gen_df['bus_id'].values
    gen[:, 3] = gen_df['qg_max_pu'].values
    gen[:, 4] = gen_df['qg_min_pu'].values
    gen[:, 5] = gen_df['vg_pu'].values
    gen[:, 6] = baseMVA
    gen[:, 7] = 1
    gen[:, 8] = gen_df['pg_max_pu'].values
    gen[:, 9] = gen_df['pg_min_pu'].values

    # BRANCH Matrix
    branch = np.zeros((len(branch_df), 13))
    branch[:, 0] = branch_df['f_bus'].values
    branch[:, 1] = branch_df['t_bus'].values
    branch[:, 2] = branch_df['r_pu'].values
    branch[:, 3] = branch_df['x_pu'].values
    branch[:, 4] = branch_df['b_pu'].values
    branch[:, 5] = branch_df['rate_a_pu'].values
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

    ppc = {'version': '2', 'baseMVA': baseMVA, 'bus': bus, 'gen': gen, 'branch': branch, 'gencost': gencost}
    ppc['bus'][:, 2] *= baseMVA
    ppc['bus'][:, 3] *= baseMVA
    ppc['gen'][:, 3] *= baseMVA
    ppc['gen'][:, 4] *= baseMVA
    ppc['gen'][:, 8] *= baseMVA
    ppc['gen'][:, 9] *= baseMVA
    mask = (ppc['branch'][:, 5] != 0) & (ppc['branch'][:, 5] < 9000)
    ppc['branch'][mask, 5:8] *= baseMVA
    return ppc


def solve_pf_custom_optimized(pd, qd, pg_non_slack, vm_gen, params):
    """
    Run power flow calculation (modified: accepts non-slack generator Pg and generator Vm)

    Args:
        pd: Load active power (p.u.), shape (n_loads,)
        qd: Load reactive power (p.u.), shape (n_loads,)
        pg_non_slack: Non-slack generator active power (p.u.), shape (n_gen_non_slack,)
        vm_gen: Generator bus voltage (p.u.), shape (n_gen,)
        params: Parameters dictionary

    Returns:
        Power flow calculation results
    """
    global GLOBAL_CASE_DATA, PPOPT
    BASE_MVA = params['general']['BASE_MVA']

    mpc_pf = {
        'version': GLOBAL_CASE_DATA['version'],
        'baseMVA': GLOBAL_CASE_DATA['baseMVA'],
        'bus': GLOBAL_CASE_DATA['bus'].copy(),
        'gen': GLOBAL_CASE_DATA['gen'].copy(),
        'branch': GLOBAL_CASE_DATA['branch'],
        'gencost': GLOBAL_CASE_DATA['gencost']
    }

    # Set loads
    load_bus_ids = params['general']['load_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    for i, bus_id in enumerate(load_bus_ids):
        bus_idx = bus_id_to_idx.get(int(bus_id))
        if bus_idx is not None:
            mpc_pf["bus"][bus_idx, 2] = pd[i] * BASE_MVA
            mpc_pf["bus"][bus_idx, 3] = qd[i] * BASE_MVA

    # Set generator active power (only non-slack generators)
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    n_gen = params['general']['n_gen']

    # For non-slack generators: set predicted pg
    for i, gen_idx in enumerate(non_slack_gen_idx):
        mpc_pf["gen"][gen_idx, 1] = pg_non_slack[i] * BASE_MVA

    # For slack generators: don't set pg (or set to 0), let power flow algorithm auto-balance
    # PyPower will automatically adjust slack bus pg to satisfy power balance

    # Set generator voltage
    for i in range(n_gen):
        mpc_pf["gen"][i, 5] = vm_gen[i]

    return runpf(mpc_pf, PPOPT)


def compute_admittance_matrix(params):
    """
    Compute admittance matrix Y from branch parameters

    Returns:
        Y_real: Real part of admittance matrix
        Y_imag: Imaginary part of admittance matrix
    """
    n_buses = params['general']['n_buses']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    # Branch parameters
    f_bus = params['branch']['f_bus']
    t_bus = params['branch']['t_bus']
    r_pu = params['branch']['r_pu']
    x_pu = params['branch']['x_pu']
    b_pu = params['branch']['b_pu']
    tap_ratio = params['branch']['tap_ratio']
    shift_deg = params['branch']['shift_deg']

    # Initialize Y matrix
    Y = np.zeros((n_buses, n_buses), dtype=complex)

    for k in range(len(f_bus)):
        i = bus_id_to_idx[int(f_bus[k])]
        j = bus_id_to_idx[int(t_bus[k])]

        # Series impedance
        z = complex(r_pu[k], x_pu[k])
        if abs(z) < 1e-10:
            y_series = 0
        else:
            y_series = 1.0 / z

        # Shunt admittance
        y_shunt = complex(0, b_pu[k])

        # Transformer tap
        tap = tap_ratio[k] if tap_ratio[k] != 0 else 1.0
        shift = shift_deg[k] * np.pi / 180.0
        tap_complex = tap * np.exp(1j * shift)

        # Build Y matrix
        Y[i, i] += y_series / (tap * np.conj(tap_complex)) + y_shunt / 2
        Y[j, j] += y_series + y_shunt / 2
        Y[i, j] -= y_series / np.conj(tap_complex)
        Y[j, i] -= y_series / tap_complex

    return Y.real, Y.imag


# =====================================================================
# KEY MODIFICATION: Handle model output dimension change
# =====================================================================
def evaluate_split(model, X, indices, raw_data, params, scalers, device, split_name, verbose=True):
    """
    Evaluate model performance on specified dataset (modified: handle non-slack pg and generator Vm)
    """
    if verbose:
        print(f"\n{split_name} Evaluation:")

    model.eval()
    with torch.no_grad():
        # Model outputs: (vm_gen, pg_non_slack, ...)
        outputs = model(X.to(device))
        vm_gen_scaled = outputs[0].cpu().numpy()
        pg_non_slack_scaled = outputs[1].cpu().numpy()

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    # Denormalize
    y_pred_pg_non_slack = scalers['pg'].inverse_transform(pg_non_slack_scaled)
    y_pred_vm_gen = scalers['vm'].inverse_transform(vm_gen_scaled)

    # Reconstruct full Pg array (for power flow calculation)
    y_pred_pg_full = reconstruct_full_pg(y_pred_pg_non_slack, params)

    # Reconstruct full Vm array (all buses)
    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    y_pred_vm_all = np.zeros((len(X), n_buses), dtype=y_pred_vm_gen.dtype)
    y_pred_vm_all[:, gen_bus_indices] = y_pred_vm_gen

    # For non-generator buses, use nominal voltage (1.0 p.u.)
    non_gen_mask = np.ones(n_buses, dtype=bool)
    non_gen_mask[gen_bus_indices] = False
    y_pred_vm_all[:, non_gen_mask] = 1.0

    # Extract true values
    y_true_pg = raw_data['pg'][indices]
    y_true_vm = raw_data['vm'][indices]
    y_true_qg = raw_data['qg'][indices]
    y_true_va_rad = raw_data['va'][indices]

    # Extract input data
    x_raw_data = scalers['x'].inverse_transform(X.cpu().numpy())
    pd_pu = x_raw_data[:, :n_loads]
    qd_pu = x_raw_data[:, n_loads:]

    # Run power flow calculation (use non-slack pg)
    n_samples = len(X)
    pf_results_list = []
    converge_flags = []

    if verbose:
        print(f"  Computing power flow for {n_samples} samples...")

    for i in range(n_samples):
        try:
            # Pass non-slack generator pg and generator vm
            r1_pf = solve_pf_custom_optimized(
                pd_pu[i],
                qd_pu[i],
                y_pred_pg_non_slack[i],
                y_pred_vm_gen[i],
                params
            )
            pf_results_list.append(r1_pf)
            converge_flags.append(r1_pf[0]['success'])
        except:
            pf_results_list.append((
                {'success': False,
                 'gen': np.zeros((n_gen, 21)),
                 'bus': np.zeros((n_buses, 13)),
                 'branch': np.zeros((1, 17))},
            ))
            converge_flags.append(False)

    if verbose:
        print(f"    ✓ Converged: {sum(converge_flags)}/{n_samples}")

    # Evaluate (using full Pg)
    return evaluate_acopf_predictions(
        y_pred_pg_full,
        y_pred_vm_all,
        y_true_pg,
        y_true_vm,
        y_true_qg,
        y_true_va_rad,
        pf_results_list,
        converge_flags,
        params,
        verbose=verbose
    )


def acopf_pinn_experiment(
        case_name,
        params_path,
        data_path,
        log_path,
        results_path,
        data_mode='random_split',
        n_train_use=None,
        test_data_path=None,
        test_params_path=None,
        n_test_samples=None,
        seed=42,
        n_epochs=1000,
        learning_rate=0.001,
        hidden_sizes_V=[256, 256],
        hidden_sizes_G=[256, 256],
        hidden_sizes_Lg=[128, 128],
        w_dual=0.001,
        w_physics=0.001,
        batch_size=None,
        device='cuda',
        tolerances=None
):
    """
    ACOPF PINN experiment main function

    PINN-specific parameters:
        hidden_sizes_V: List of hidden layer sizes for V network (voltage prediction)
                        Example: [256, 256] means two hidden layers with 256 neurons each
        hidden_sizes_G: List of hidden layer sizes for G network (power prediction)
                        Example: [256, 256]
        hidden_sizes_Lg: List of hidden layer sizes for Lg network (dual variables)
                         Example: [128, 128] (typically smaller than V/G networks)
        w_dual: Weight for dual variable loss in total loss function
                Typical range: 0.001 - 0.01
        w_physics: Weight for physics constraint (KKT) loss in total loss function
                   Typical range: 0.01 - 0.1

    Other parameters: Same as DNN version (see acopf_dnn_main.py)
    """
    global GLOBAL_CASE_DATA, PPOPT
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print(f"ACOPF PINN Experiment")
    print(f"{'=' * 70}")
    print(f"Device: {device_obj}")
    print(f"Case: {case_name}")
    print(f"Data Mode: {data_mode}")
    print(f"{'=' * 70}")

    # ========================================================================
    # 1. Load training data params and PyPower case data
    # ========================================================================
    params = load_parameters_from_csv(case_name, params_path)
    init_pypower_options()
    GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)
    print(f"✓ Training params and PyPower case data loaded")

    # ========================================================================
    # 2. Compute admittance matrix for physics loss
    # ========================================================================
    Y_real, Y_imag = compute_admittance_matrix(params)
    params['Y_real'] = Y_real
    params['Y_imag'] = Y_imag
    print(f"✓ Admittance matrix computed: {Y_real.shape}")

    # ========================================================================
    # 3. Add network structure to params
    # ========================================================================
    params['training'] = {
        'neurons_in_hidden_layers_V': hidden_sizes_V,
        'neurons_in_hidden_layers_G': hidden_sizes_G,
        'neurons_in_hidden_layers_Lg': hidden_sizes_Lg,
    }

    # ========================================================================
    # 4. Load training data and fit scalers
    # ========================================================================
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
    # 5. Load and split data based on data mode
    # ========================================================================
    if data_mode == DataMode.API_TEST:
        print(f"\n{'=' * 70}")
        print(f"Data Mode: API_TEST")
        print(f"{'=' * 70}")

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

        # Add admittance matrix for test params
        Y_real_test, Y_imag_test = compute_admittance_matrix(test_params)
        test_params['Y_real'] = Y_real_test
        test_params['Y_imag'] = Y_imag_test

        GLOBAL_CASE_DATA_TEST = load_case_from_csv(test_case_name, test_params_path)
        print(f"✓ API test PyPower case data loaded")

        print(f"\n[API Test Data Info]")
        print(f"  Buses: {test_params['general']['n_buses']}")
        print(
            f"  Generators: {test_params['general']['n_gen']} (Non-Slack: {test_params['general']['n_gen_non_slack']})")
        print(f"  Loads: {test_params['general']['n_loads']}")
        print(f"  Base MVA: {test_params['general']['BASE_MVA']}")

    elif data_mode == DataMode.GENERALIZATION:
        print(f"\n{'=' * 70}")
        print(f"Data Mode: GENERALIZATION")
        print(f"{'=' * 70}")

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
        print(f"\n{'=' * 70}")
        print(f"Data Mode: {data_mode}")
        print(f"{'=' * 70}")

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
    # 6. Prepare training data (including auxiliary Qg and Va for physics loss)
    # ========================================================================
    X_train = torch.tensor(x_data_scaled[train_idx], dtype=torch.float32, device=device_obj)
    Y_train = torch.tensor(y_data_scaled[train_idx], dtype=torch.float32, device=device_obj)
    X_val = torch.tensor(x_data_scaled[val_idx], dtype=torch.float32, device=device_obj)
    Y_val = torch.tensor(y_data_scaled[val_idx], dtype=torch.float32, device=device_obj)
    X_test = torch.tensor(test_x_scaled[test_idx], dtype=torch.float32, device=device_obj)

    # Prepare auxiliary Qg and Va for training (from dataset)
    qg_train_raw = raw_data['qg'][train_idx]
    va_train_raw = raw_data['va'][train_idx]

    # Scale Qg and Va using their scalers
    qg_train_scaled = scalers['qg'].transform(qg_train_raw)
    va_train_scaled = scalers['va'].transform(va_train_raw)

    Qg_train = torch.tensor(qg_train_scaled, dtype=torch.float32, device=device_obj)
    Va_train = torch.tensor(va_train_scaled, dtype=torch.float32, device=device_obj)

    print(f"\n[Dataset Sizes]")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    # ========================================================================
    # 7. Create PINN model
    # ========================================================================
    input_dim = x_data_scaled.shape[1]
    output_dim = y_data_scaled.shape[1]  # n_gen_non_slack + n_gen

    print(f"\n{'=' * 70}")
    print(f"Model Configuration")
    print(f"{'=' * 70}")
    print(f"Input dim: {input_dim} (Load: pd + qd)")
    print(f"Output dim: {output_dim} (pg_non_slack: {n_gen_non_slack} + vm_gen: {n_gen})")
    print(f"V Network: {input_dim} -> {' -> '.join(map(str, hidden_sizes_V))} -> {n_gen}")
    print(f"G Network: {input_dim} -> {' -> '.join(map(str, hidden_sizes_G))} -> {n_gen_non_slack}")
    print(f"Lg Network: {input_dim} -> {' -> '.join(map(str, hidden_sizes_Lg))} -> (dual vars)")
    print(f"Training params: epochs={n_epochs}, lr={learning_rate}, batch_size={batch_size or 'full batch'}")
    print(f"Loss weights: w_dual={w_dual}, w_physics={w_physics}")
    print(f"{'=' * 70}")

    model = PinnModel(
        W_dual=w_dual,
        W_PINN=w_physics,
        simulation_parameters=params,
        learning_rate=learning_rate,
        device=device_obj
    ).to(device_obj)

    # ========================================================================
    # 8. Training loop
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"Training Progress")
    print(f"{'=' * 70}")

    n_train = len(X_train)
    batch_size = batch_size or n_train
    n_batches = (n_train + batch_size - 1) // batch_size
    train_losses = []
    t0 = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_kkt = 0.0

        indices = torch.randperm(n_train, device=device_obj)

        for i in range(n_batches):
            batch_idx = indices[i * batch_size:min((i + 1) * batch_size, n_train)]

            X_batch = X_train[batch_idx]
            Y_batch = Y_train[batch_idx]
            Qg_batch = Qg_train[batch_idx]
            Va_batch = Va_train[batch_idx]

            # Forward pass with auxiliary Qg and Va
            model.optimizer.zero_grad()
            outputs = model(X_batch, qg_aux=Qg_batch, va_aux=Va_batch)

            # Prepare targets (split Y_batch into components)
            # Y_batch contains: [pg_non_slack, vm_gen]
            pg_target = Y_batch[:, :n_gen_non_slack]
            vm_target = Y_batch[:, n_gen_non_slack:]

            # Dummy targets for dual variables (not supervised)
            mu_pg_min_target = torch.zeros_like(outputs[2])
            mu_pg_max_target = torch.zeros_like(outputs[3])
            mu_vm_min_target = torch.zeros_like(outputs[4])
            mu_vm_max_target = torch.zeros_like(outputs[5])
            physics_dummy = torch.zeros(len(X_batch), device=device_obj)

            targets = (vm_target, pg_target, mu_pg_min_target, mu_pg_max_target,
                       mu_vm_min_target, mu_vm_max_target, physics_dummy)

            # Compute loss
            total_loss, losses = model.compute_loss(outputs, targets)

            # Backward pass
            total_loss.backward()
            model.optimizer.step()

            epoch_loss += total_loss.item() * len(X_batch)
            epoch_kkt += losses[6] * len(X_batch)  # KKT loss is the 7th component

        avg_loss = epoch_loss / n_train
        avg_kkt = epoch_kkt / n_train
        train_losses.append(avg_loss)

        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"Epoch {epoch:4d}/{n_epochs} - Train Loss: {avg_loss:.6f} - KKT Error: {avg_kkt:.6f}")

    train_time = time.perf_counter() - t0
    print(f"\n✓ Training completed in {train_time:.2f} seconds")

    # ========================================================================
    # 9. Model evaluation (use test params)
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

    test_metrics = evaluate_split(
        model, X_test, test_idx, test_raw_data,
        test_params,
        scalers, device_obj, test_split_name,
        verbose=True
    )

    GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_BACKUP

    # ========================================================================
    # 10. Inference speed test
    # ========================================================================
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model.predict(X_test[:1])

    times = [time.perf_counter() for _ in range(101)]
    with torch.no_grad():
        for i in range(100):
            _ = model.predict(X_test[:1])
            if device_obj.type == 'cuda':
                torch.cuda.synchronize()
            times[i + 1] = time.perf_counter()

    latency_ms = np.mean(np.diff(times)) * 1000

    # ========================================================================
    # 11. Simplified result summary (only specified metrics)
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

    # No file saving - removed all JSON/CSV/model saving code

    return test_metrics


if __name__ == "__main__":
    # =========================================================================
    # Read configuration from acopf_config.py and run experiment
    # =========================================================================

    # Print configuration
    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)

    paths = acopf_config.get_all_paths()
    config_params = acopf_config.get_all_params()

    # Extract hidden_sizes and remove from config_params
    # For PINN, we use the same structure for V, G, and a smaller one for Lg
    hidden_sizes = config_params.pop('hidden_sizes', [256, 256])

    # PINN-specific network structures
    # V network (voltage): same as DNN hidden layers
    # G network (power): same as DNN hidden layers
    # Lg network (dual variables): smaller network (half the neurons)
    hidden_sizes_V = hidden_sizes
    hidden_sizes_G = hidden_sizes
    hidden_sizes_Lg = [h // 2 for h in hidden_sizes]  # Half size for dual network

    print(f"\nPINN Network Structures:")
    print(f"  V Network (Voltage):      {hidden_sizes_V}")
    print(f"  G Network (Power):        {hidden_sizes_G}")
    print(f"  Lg Network (Dual Vars):   {hidden_sizes_Lg}")

    # PINN-specific hyperparameters (can be adjusted)
    W_DUAL = 0.005  # Weight for dual variable loss
    W_PHYSICS = 0.005  # Weight for physics constraint (KKT) loss

    print(f"\nPINN Loss Weights:")
    print(f"  Dual Variables:   {W_DUAL}")
    print(f"  Physics (KKT):    {W_PHYSICS}")

    # Run experiment
    results = acopf_pinn_experiment(
        **paths,
        **config_params,
        # PINN-specific parameters
        hidden_sizes_V=hidden_sizes_V,
        hidden_sizes_G=hidden_sizes_G,
        hidden_sizes_Lg=hidden_sizes_Lg,
        w_dual=W_DUAL,
        w_physics=W_PHYSICS
    )

    print("\n✓ Experiment completed successfully!")