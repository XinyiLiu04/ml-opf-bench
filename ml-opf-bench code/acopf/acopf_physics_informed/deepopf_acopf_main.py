# -*- coding: utf-8 -*-
"""
DeepOPF (PINN) ACOPF Main Experiment Script (V8 - Simplified Output)

ACOPF solving based on Physics-Informed Neural Networks
Uses zero-order optimization for gradient estimation of constraint penalties

Modifications (V8):
- DNN only predicts non-slack pg and generator vm
- Simplified output: only test set metrics
- All comments and outputs in English
- Removed all file saving (JSON/CSV/model)
- All violations in p.u. units
"""

import os
import sys
import time
import copy
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Function
from sklearn.preprocessing import MinMaxScaler
import multiprocessing as mp
from pathlib import Path
import traceback

from pypower.runpf import runpf
from pypower.ppoption import ppoption

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
# Global Variables
# =====================================================================
GLOBAL_PARAMS = {}
GLOBAL_SCALERS = {}
GLOBAL_CASE_DATA = None
GLOBAL_POOL = None
PPOPT = None

# Worker process specific global variables
WORKER_PARAMS = None
WORKER_CASE_DATA = None
WORKER_PPOPT = None


# =====================================================================
# Multiprocessing Worker Initialization
# =====================================================================
def init_worker_globals(final_params, case_name, params_path, ppopt):
    """Initialize worker process global variables"""
    global WORKER_PARAMS, WORKER_CASE_DATA, WORKER_PPOPT

    worker_pid = os.getpid()
    # print(f"[Worker {worker_pid}] Initializing...")

    WORKER_PPOPT = ppopt
    WORKER_PARAMS = final_params
    # print(f"[Worker {worker_pid}] ✓ Parameters received.")

    # print(f"[Worker {worker_pid}] Re-loading case data from disk...")
    WORKER_CASE_DATA = load_case_from_csv(case_name, params_path)

    # if WORKER_PARAMS:
    #     n_ids = len(WORKER_PARAMS['general']['load_bus_ids'])
    #     n_loads = WORKER_PARAMS['general']['n_loads']
    #     print(f"[Worker {worker_pid}] ✓ Params check: n_loads = {n_loads}, len(load_bus_ids) = {n_ids}")


def init_global_pool(n_cores, final_params, case_name, params_path, ppopt):
    """Initialize global process pool"""
    global GLOBAL_POOL
    if GLOBAL_POOL is None:
        try:
            ctx = mp.get_context('spawn')
        except RuntimeError:
            ctx = mp.get_context()

        init_args = (final_params, case_name, params_path, ppopt)
        GLOBAL_POOL = ctx.Pool(n_cores, initializer=init_worker_globals, initargs=init_args)
        print(f"✓ Initialized process pool with {n_cores} workers")


def close_global_pool():
    """Close global process pool"""
    global GLOBAL_POOL
    if GLOBAL_POOL is not None:
        GLOBAL_POOL.close()
        GLOBAL_POOL.join()
        GLOBAL_POOL = None
        # print("✓ Closed process pool.")


# =====================================================================
# PyPower Interface
# =====================================================================
def init_pypower_options():
    """Initialize PyPower solver options"""
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

    ppc = {
        'version': '2',
        'baseMVA': baseMVA,
        'bus': bus,
        'gen': gen,
        'branch': branch,
        'gencost': gencost
    }

    # PyPower needs MW/Mvar units (convert from p.u.)
    ppc['bus'][:, 2] *= baseMVA
    ppc['bus'][:, 3] *= baseMVA
    ppc['gen'][:, 3] *= baseMVA
    ppc['gen'][:, 4] *= baseMVA
    ppc['gen'][:, 8] *= baseMVA
    ppc['gen'][:, 9] *= baseMVA
    mask = (ppc['branch'][:, 5] != 0) & (ppc['branch'][:, 5] < 9000)
    ppc['branch'][mask, 5:8] *= baseMVA

    return ppc


def solve_pf_custom_optimized_worker(pd, qd, pg_non_slack, vm_gen):
    """
    Worker process power flow calculation

    Args:
        pd: Load active power (p.u.)
        qd: Load reactive power (p.u.)
        pg_non_slack: Non-slack generator active power (p.u.)
        vm_gen: Generator bus voltage (p.u.)
    """
    global WORKER_CASE_DATA, WORKER_PPOPT, WORKER_PARAMS

    if WORKER_CASE_DATA is None or WORKER_PARAMS is None:
        return None

    BASE_MVA = WORKER_PARAMS['general']['BASE_MVA']
    non_slack_gen_idx = WORKER_PARAMS['general']['non_slack_gen_idx']
    n_gen = WORKER_PARAMS['general']['n_gen']

    mpc_pf = {
        'version': WORKER_CASE_DATA['version'],
        'baseMVA': WORKER_CASE_DATA['baseMVA'],
        'bus': WORKER_CASE_DATA['bus'].copy(),
        'gen': WORKER_CASE_DATA['gen'].copy(),
        'branch': WORKER_CASE_DATA['branch'],
        'gencost': WORKER_CASE_DATA['gencost']
    }

    # Set loads
    load_bus_ids = WORKER_PARAMS['general']['load_bus_ids']
    bus_id_to_idx = WORKER_PARAMS['general']['bus_id_to_idx']

    for i, bus_id in enumerate(load_bus_ids):
        bus_idx = bus_id_to_idx.get(int(bus_id))
        if bus_idx is not None:
            mpc_pf["bus"][bus_idx, 2] = pd[i] * BASE_MVA
            mpc_pf["bus"][bus_idx, 3] = qd[i] * BASE_MVA

    # Set generator active power (only non-slack generators)
    for i, gen_idx in enumerate(non_slack_gen_idx):
        mpc_pf["gen"][gen_idx, 1] = pg_non_slack[i] * BASE_MVA

    # Set generator voltage
    for i in range(n_gen):
        mpc_pf["gen"][i, 5] = vm_gen[i]

    return runpf(mpc_pf, WORKER_PPOPT)


def solve_pf_custom_optimized(pd, qd, pg_non_slack, vm_gen, params):
    """
    Main process power flow calculation

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
    for i, gen_idx in enumerate(non_slack_gen_idx):
        mpc_pf["gen"][gen_idx, 1] = pg_non_slack[i] * BASE_MVA

    # Set generator voltage
    for i in range(n_gen):
        mpc_pf["gen"][i, 5] = vm_gen[i]

    return runpf(mpc_pf, PPOPT)


# =====================================================================
# Penalty Calculation (DO NOT MODIFY ALGORITHM)
# =====================================================================
def zero_order_penalty_abs(pf_results):
    """Calculate zero-order penalty value"""
    standval = pf_results[0]["baseMVA"]
    ctol = 1e-4

    # 1. Generator active power constraints
    On_index = np.where(pf_results[0]["gen"][:, 7] == 1)[0]
    Pg_temp1 = (pf_results[0]["gen"][On_index, 9] - pf_results[0]["gen"][On_index, 1]) / standval
    Pg_temp1[Pg_temp1 < ctol] = 0
    Pg_temp2 = (pf_results[0]["gen"][On_index, 1] - pf_results[0]["gen"][On_index, 8]) / standval
    Pg_temp2[Pg_temp2 < ctol] = 0
    PG_penalty = np.abs(Pg_temp1) + np.abs(Pg_temp2)

    # 2. Generator reactive power constraints
    Qg_temp1 = (pf_results[0]["gen"][On_index, 4] - pf_results[0]["gen"][On_index, 2]) / standval
    Qg_temp1[Qg_temp1 < ctol] = 0
    Qg_temp2 = (pf_results[0]["gen"][On_index, 2] - pf_results[0]["gen"][On_index, 3]) / standval
    Qg_temp2[Qg_temp2 < ctol] = 0
    QG_penalty = np.abs(Qg_temp1) + np.abs(Qg_temp2)

    # 3. PQ node voltage constraints
    PQ_index = np.where(pf_results[0]["bus"][:, 1] == 1)[0]
    V_temp1 = pf_results[0]["bus"][PQ_index, 12] - pf_results[0]["bus"][PQ_index, 7]
    V_temp1[V_temp1 < ctol] = 0
    V_temp2 = pf_results[0]["bus"][PQ_index, 7] - pf_results[0]["bus"][PQ_index, 11]
    V_temp2[V_temp2 < ctol] = 0
    V_penalty = np.abs(V_temp1) + np.abs(V_temp2)

    # 4. Branch flow constraints
    Ff = abs(pf_results[0]["branch"][:, 13] + 1j * pf_results[0]["branch"][:, 14])
    Ft = abs(pf_results[0]["branch"][:, 15] + 1j * pf_results[0]["branch"][:, 16])

    Branch_index = np.where(pf_results[0]["branch"][:, 5] != 0)[0]
    Branch_bound = pf_results[0]["branch"][Branch_index, 5]
    Ff_temp = (Ff[Branch_index] - Branch_bound) / standval
    Ff_temp[Ff_temp < ctol] = 0
    Ft_temp = (Ft[Branch_index] - Branch_bound) / standval
    Ft_temp[Ft_temp < ctol] = 0
    Ff_penalty = np.abs(Ff_temp)
    Ft_penalty = np.abs(Ft_temp)

    return np.sum(PG_penalty) + np.sum(QG_penalty) + np.sum(V_penalty) + np.sum(Ff_penalty) + np.sum(Ft_penalty)


def compute_penalty_worker_mp(args):
    """Multiprocessing worker penalty calculation"""
    pd, qd, pg_non_slack, vm_gen = args

    try:
        r1_pf = solve_pf_custom_optimized_worker(pd, qd, pg_non_slack, vm_gen)
        if r1_pf is None:
            penalty = 1.0
        elif r1_pf[0]['success']:
            penalty = zero_order_penalty_abs(r1_pf)
        else:
            penalty = 1.0
    except Exception:
        penalty = 1.0
    return penalty


# =====================================================================
# Custom Backpropagation Layer
# =====================================================================
class Penalty_ACPF_Optimized(Function):
    """Physics constraint penalty layer (zero-order optimization)"""

    @staticmethod
    def forward(ctx, nn_output_scaled, x_input_scaled):
        ctx.save_for_backward(nn_output_scaled, x_input_scaled)

        nn_output_np = nn_output_scaled.cpu().detach().numpy()
        x_input_np = x_input_scaled.cpu().detach().numpy()

        batch_size = nn_output_np.shape[0]
        params = GLOBAL_PARAMS
        scalers = GLOBAL_SCALERS

        n_gen = params['general']['n_gen']
        n_gen_non_slack = params['general']['n_gen_non_slack']
        n_buses = params['general']['n_buses']
        n_loads = params['general']['n_loads']

        # Denormalize (NEW: handle new dimensions)
        y_pred_pg_non_slack = scalers['pg'].inverse_transform(nn_output_np[:, :n_gen_non_slack])
        y_pred_vm_gen = scalers['vm'].inverse_transform(nn_output_np[:, n_gen_non_slack:])

        x_raw = scalers['x'].inverse_transform(x_input_np)
        pd = x_raw[:, :n_loads]
        qd = x_raw[:, n_loads:]

        global GLOBAL_POOL
        if GLOBAL_POOL is None:
            # Serial calculation
            penalty_list = np.zeros(batch_size)
            for i in range(batch_size):
                r1_pf = solve_pf_custom_optimized(pd[i], qd[i], y_pred_pg_non_slack[i], y_pred_vm_gen[i], params)
                if r1_pf[0]['success']:
                    penalty_list[i] = zero_order_penalty_abs(r1_pf)
                else:
                    penalty_list[i] = 1.0
        else:
            # Parallel calculation
            args_list = [(pd[i], qd[i], y_pred_pg_non_slack[i], y_pred_vm_gen[i]) for i in range(batch_size)]
            results = GLOBAL_POOL.map(compute_penalty_worker_mp, args_list)
            penalty_list = np.array(results)

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

        n_gen = params['general']['n_gen']
        n_gen_non_slack = params['general']['n_gen_non_slack']
        n_buses = params['general']['n_buses']
        n_loads = params['general']['n_loads']

        # Random direction
        vec = np.random.randn(batch_size, output_dim)
        vec_norm = np.linalg.norm(vec, axis=1).reshape(-1, 1)
        vector_h = vec / (vec_norm + 1e-10)

        h = 1e-4

        nn_output_plus_h = np.clip(nn_output_np + vector_h * h, 0, 1)
        nn_output_minus_h = np.clip(nn_output_np - vector_h * h, 0, 1)

        x_raw = scalers['x'].inverse_transform(x_input_np)
        pd = x_raw[:, :n_loads]
        qd = x_raw[:, n_loads:]

        # Plus h (NEW: handle new dimensions)
        y_pred_pg_plus = scalers['pg'].inverse_transform(nn_output_plus_h[:, :n_gen_non_slack])
        y_pred_vm_plus = scalers['vm'].inverse_transform(nn_output_plus_h[:, n_gen_non_slack:])

        # Minus h (NEW: handle new dimensions)
        y_pred_pg_minus = scalers['pg'].inverse_transform(nn_output_minus_h[:, :n_gen_non_slack])
        y_pred_vm_minus = scalers['vm'].inverse_transform(nn_output_minus_h[:, n_gen_non_slack:])

        gradient_estimate = np.zeros((batch_size, output_dim), dtype='float32')

        global GLOBAL_POOL
        if GLOBAL_POOL is None:
            penalty_plus = np.zeros(batch_size)
            penalty_minus = np.zeros(batch_size)
            for i in range(batch_size):
                r1_plus = solve_pf_custom_optimized(pd[i], qd[i], y_pred_pg_plus[i], y_pred_vm_plus[i], params)
                penalty_plus[i] = zero_order_penalty_abs(r1_plus) if r1_plus[0]['success'] else 1.0

                r1_minus = solve_pf_custom_optimized(pd[i], qd[i], y_pred_pg_minus[i], y_pred_vm_minus[i], params)
                penalty_minus[i] = zero_order_penalty_abs(r1_minus) if r1_minus[0]['success'] else 1.0
        else:
            args_plus = [(pd[i], qd[i], y_pred_pg_plus[i], y_pred_vm_plus[i]) for i in range(batch_size)]
            args_minus = [(pd[i], qd[i], y_pred_pg_minus[i], y_pred_vm_minus[i]) for i in range(batch_size)]

            results_plus = GLOBAL_POOL.map(compute_penalty_worker_mp, args_plus)
            results_minus = GLOBAL_POOL.map(compute_penalty_worker_mp, args_minus)

            penalty_plus = np.array(results_plus)
            penalty_minus = np.array(results_minus)

        for i in range(batch_size):
            directional_derivative = (penalty_plus[i] - penalty_minus[i]) / (2 * h)
            gradient_estimate[i] = directional_derivative * vector_h[i] * output_dim

        final_gradient = gradient_estimate * (1.0 / batch_size)

        return torch.from_numpy(final_gradient).to(nn_output_scaled.device) * grad_output, None


# =====================================================================
# Model Definition
# =====================================================================
class PINN_ACOPF(nn.Module):
    """PINN ACOPF Model"""

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
        self.penalty_layer = Penalty_ACPF_Optimized.apply

    def forward(self, x):
        x_sol = self.net(x)
        x_penalty = self.penalty_layer(x_sol, x)
        return x_sol, x_penalty.to(x_sol.device)


# =====================================================================
# Evaluation Function
# =====================================================================
def evaluate_model(model, X, indices, raw_data, params, scalers, device, split_name, verbose=True):
    """
    Evaluate model - calls acopf_violation_metrics.evaluate_acopf_predictions
    """
    if verbose:
        print(f"\n{split_name} Evaluation:")

    model.eval()
    with torch.no_grad():
        y_pred_scaled, _ = model(X.to(device))

    y_pred_scaled_np = y_pred_scaled.cpu().numpy()

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    # Denormalize (NEW: handle new dimensions)
    y_pred_pg_non_slack = scalers['pg'].inverse_transform(y_pred_scaled_np[:, :n_gen_non_slack])
    y_pred_vm_gen = scalers['vm'].inverse_transform(y_pred_scaled_np[:, n_gen_non_slack:])

    # Reconstruct full Pg array
    y_pred_pg_full = reconstruct_full_pg(y_pred_pg_non_slack, params)

    # Reconstruct full Vm array (all buses)
    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    y_pred_vm_all = np.zeros((len(X), n_buses), dtype=y_pred_vm_gen.dtype)
    y_pred_vm_all[:, gen_bus_indices] = y_pred_vm_gen

    # Fill non-generator buses with 1.0 p.u.
    non_gen_mask = np.ones(n_buses, dtype=bool)
    non_gen_mask[gen_bus_indices] = False
    y_pred_vm_all[:, non_gen_mask] = 1.0

    # True values
    y_true_pg = raw_data['pg'][indices]
    y_true_vm = raw_data['vm'][indices]
    y_true_qg = raw_data['qg'][indices]
    y_true_va_rad = raw_data['va'][indices]

    # Loads
    x_raw_data = scalers['x'].inverse_transform(X.cpu().numpy())
    pd_pu = x_raw_data[:, :n_loads]
    qd_pu = x_raw_data[:, n_loads:]

    # Run power flow calculation
    n_samples = len(X)
    pf_results_list = []
    converge_flags = []

    if verbose:
        print(f"  Computing power flow for {n_samples} samples...")

    for i in range(n_samples):
        try:
            r1_pf = solve_pf_custom_optimized(pd_pu[i], qd_pu[i], y_pred_pg_non_slack[i], y_pred_vm_gen[i], params)
            pf_results_list.append(r1_pf)
            converge_flags.append(r1_pf[0]['success'])
        except:
            pf_results_list.append(({'success': False, 'gen': np.zeros((n_gen, 21)),
                                     'bus': np.zeros((n_buses, 13)), 'branch': np.zeros((1, 17))},))
            converge_flags.append(False)

    if verbose:
        print(f"    ✓ Converged: {sum(converge_flags)}/{n_samples}")

    # Call unified evaluation function
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
# Main Training Function
# =====================================================================
def train_pinn_acopf(
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
        hidden_sizes=[256, 256],
        n_epochs=100,
        batch_size=256,
        learning_rate=1e-3,
        penalty_weight=0.1,
        seed=42,
        device='cuda',
        n_cores=8
):
    """
    PINN ACOPF training main function

    Supports four data modes:
    - RANDOM_SPLIT: Random split
    - FIXED_VALTEST: Fixed validation/test sets
    - GENERALIZATION: Cross-distribution generalization test
    - API_TEST: API data test (different topology)
    """
    global GLOBAL_PARAMS, GLOBAL_SCALERS, GLOBAL_CASE_DATA, PPOPT

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU.")
        device = 'cpu'
    device_obj = torch.device(device)

    print(f"\n{'=' * 70}")
    print(f"DeepOPF (PINN) ACOPF Training")
    print(f"{'=' * 70}")
    print(f"Case: {case_name}")
    print(f"Data Mode: {data_mode}")
    print(f"Device: {device_obj}")
    print(f"Penalty weight: {penalty_weight}")
    print(f"Process pool cores: {n_cores}")

    # ========================================================================
    # 1. Load training parameters and PyPower case data
    # ========================================================================
    print(f"\n[Step 1] Loading training parameters and case data...")
    init_pypower_options()
    params = load_parameters_from_csv(case_name, params_path)
    GLOBAL_PARAMS = params
    GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)
    print(f"  ✓ Training params and case data loaded")

    # ========================================================================
    # 2. Load training data and fit scalers
    # ========================================================================
    print(f"\n[Step 2] Loading training data...")
    x_data_scaled, y_data_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True)
    GLOBAL_SCALERS = scalers

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
    # 3. Initialize process pool (using training params)
    # ========================================================================
    print(f"\n[Step 3] Initializing process pool...")
    init_global_pool(n_cores, GLOBAL_PARAMS, case_name, params_path, PPOPT)

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

        test_case_name = os.path.basename(test_data_path)
        if test_case_name.endswith('_pd.csv'):
            test_case_name = test_case_name[:-7]

        GLOBAL_CASE_DATA_TEST = load_case_from_csv(test_case_name, test_params_path)
        print(f"✓ API test PyPower case data loaded")

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
    # 5. Prepare training data
    # ========================================================================
    X_train = torch.from_numpy(x_data_scaled[train_idx]).float().to(device_obj)
    Y_train = torch.from_numpy(y_data_scaled[train_idx]).float().to(device_obj)
    X_val = torch.from_numpy(x_data_scaled[val_idx]).float().to(device_obj)
    Y_val = torch.from_numpy(y_data_scaled[val_idx]).float().to(device_obj)
    X_test = torch.from_numpy(test_x_scaled[test_idx]).float().to(device_obj)

    print(f"\n[Dataset Sizes]")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    train_dataset = Data.TensorDataset(X_train, Y_train)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # ========================================================================
    # 6. Create model and train
    # ========================================================================
    print(f"\n[Step 4] Building model...")
    model = PINN_ACOPF(x_data_scaled.shape[1], y_data_scaled.shape[1], hidden_sizes).to(device_obj)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Network: {x_data_scaled.shape[1]} -> {' -> '.join(map(str, hidden_sizes))} -> {y_data_scaled.shape[1]}")
    print(f"  Total params: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    # Training
    print(f"\n{'=' * 70}")
    print(f"Training Progress")
    print(f"{'=' * 70}")
    print(f"Loss = 1.0×MSE + {penalty_weight}×Penalty")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_epoch = 0

    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_total = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred, penalty = model(batch_x)
            mse_loss = criterion(pred, batch_y)
            total_loss = mse_loss + penalty_weight * penalty
            total_loss.backward()
            optimizer.step()

            epoch_total += total_loss.item() * len(batch_x)

        avg_total = epoch_total / len(X_train)
        train_losses.append(avg_total)

        model.eval()
        with torch.no_grad():
            val_pred, val_penalty = model(X_val)
            val_mse = criterion(val_pred, Y_val)
            val_loss = val_mse.item() + penalty_weight * val_penalty.item()
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"Epoch {epoch:4d}/{n_epochs} - Train Loss: {avg_total:.6f} - Val Loss: {val_loss:.6f}")

    train_time = time.time() - t0
    print(f"\n✓ Training completed in {train_time:.2f} seconds")

    # ========================================================================
    # 7. Model evaluation (using test params)
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"Test Set Evaluation")
    print(f"{'=' * 70}")

    # Temporarily switch GLOBAL_CASE_DATA for test evaluation
    GLOBAL_CASE_DATA_BACKUP = GLOBAL_CASE_DATA
    GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_TEST

    if data_mode == DataMode.API_TEST:
        test_split_name = "API Test"
    elif data_mode == DataMode.GENERALIZATION:
        test_split_name = "Generalization Test"
    else:
        test_split_name = "Test"

    test_metrics = evaluate_model(
        model, X_test, test_idx, test_raw_data,
        test_params,
        scalers, device_obj, test_split_name,
        verbose=True
    )

    GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_BACKUP  # Restore

    # ========================================================================
    # 8. Inference speed test
    # ========================================================================
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(100):
            t_start = time.time()
            _ = model(X_test[:1])
            if device_obj.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - t_start)

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

    # Close process pool
    close_global_pool()

    # No file saving - removed all JSON/CSV/model saving

    return test_metrics


# =====================================================================
# Main Function
# =====================================================================
if __name__ == '__main__':
    # =========================================================================
    # Read configuration from acopf_config.py and run experiment
    # =========================================================================

    # PINN specific parameters
    PENALTY_WEIGHT = 0.005
    N_CORES = 8

    # Print configuration
    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)

    # Get all paths
    paths = acopf_config.get_all_paths()

    # Get all training parameters
    params = acopf_config.get_all_params()

    # Add PINN specific parameters
    params['penalty_weight'] = PENALTY_WEIGHT
    params['n_cores'] = N_CORES

    print(f"\n[PINN Specific Configuration]")
    print(f"  Penalty weight: {PENALTY_WEIGHT}")
    print(f"  Process cores: {N_CORES}")
    print("=" * 70)

    # Execute experiment
    results = train_pinn_acopf(**paths, **params)

    print("\n✓ Experiment completed successfully!")