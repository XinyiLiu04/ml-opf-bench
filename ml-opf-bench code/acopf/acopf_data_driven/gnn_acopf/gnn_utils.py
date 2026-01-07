# -*- coding: utf-8 -*-
"""
GNN Utility Functions (V8 - Non-Slack Pg + Generator Vm)

Responsibilities:
- Graph structure construction (build_graph_structure)
- PyPower interface (load_case_from_csv, solve_pf_custom_optimized)
- GNN evaluation function (evaluate_split) - calls acopf_violation_metrics

Reused modules:
- acopf_violation_metrics.evaluate_acopf_predictions
- acopf_data_setup.reconstruct_full_pg
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from pypower.runpf import runpf
from pypower.ppoption import ppoption

# Import evaluation module
from acopf_violation_metrics import evaluate_acopf_predictions
from acopf_data_setup import reconstruct_full_pg


# =====================================================================
# Global Variables
# =====================================================================
GLOBAL_CASE_DATA = None
PPOPT = None


# =====================================================================
# Graph Structure Construction
# =====================================================================
def build_graph_structure(params):
    """
    Build graph structure from network parameters (edges, edge features, node types)

    Correctly handles sparse bus numbering

    Args:
        params: Parameter dictionary returned by load_parameters_from_csv()

    Returns:
        edges: [2, num_edges] - Edge indices (PyG format)
        edge_weights: [num_edges, 2] - Edge features [r_pu, x_pu]
        node_types: [n_buses, 3] - Node types [is_gen, is_load, is_slack]
    """
    n_buses = params['general']['n_buses']
    bus_ids = params['general']['bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    # Extract branch information
    f_bus = params['branch']['f_bus']
    t_bus = params['branch']['t_bus']
    r_pu = params['branch']['r_pu']
    x_pu = params['branch']['x_pu']

    # Build bidirectional edges (using indices not bus_id)
    edges_list = []
    edge_weights_list = []

    for i in range(len(f_bus)):
        # Convert bus_id to array index
        f_idx = bus_id_to_idx[int(f_bus[i])]
        t_idx = bus_id_to_idx[int(t_bus[i])]

        # Bidirectional edges
        edges_list.append([f_idx, t_idx])
        edges_list.append([t_idx, f_idx])

        # Edge features (same for both directions)
        edge_feat = [r_pu[i], x_pu[i]]
        edge_weights_list.append(edge_feat)
        edge_weights_list.append(edge_feat)

    edges = torch.tensor(edges_list, dtype=torch.long).t()  # [2, num_edges]
    edge_weights = torch.tensor(edge_weights_list, dtype=torch.float32)  # [num_edges, 2]

    # Build node type markers
    node_types = torch.zeros(n_buses, 3, dtype=torch.float32)

    gen_bus_ids = params['general']['gen_bus_ids']
    load_bus_ids = params['general']['load_bus_ids']

    # Use mapping to fill node types
    for gid in gen_bus_ids:
        idx = bus_id_to_idx[int(gid)]
        node_types[idx, 0] = 1.0  # is_gen

    for lid in load_bus_ids:
        idx = bus_id_to_idx[int(lid)]
        node_types[idx, 1] = 1.0  # is_load

    print(f"\n✓ Graph structure constructed:")
    print(f"  Nodes: {n_buses}")
    print(f"  Edges: {edges.shape[1]} (bidirectional)")
    print(f"  Generator nodes: {int(node_types[:, 0].sum())}")
    print(f"  Load nodes: {int(node_types[:, 1].sum())}")
    print(f"  Edge feature dim: {edge_weights.shape[1]}")

    return edges, edge_weights, node_types


# =====================================================================
# PyPower Interface
# =====================================================================
def init_pypower_options():
    """Initialize PyPower solver options"""
    global PPOPT
    ppopt = ppoption()
    PPOPT = ppoption(ppopt, OUT_ALL=0, VERBOSE=0, ENFORCE_Q_LIMS=0)


def load_case_from_csv(case_name, constraints_path):
    """Load PyPower case data from CSV"""
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

    # PyPower needs MW/Mvar units
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
    Run power flow calculation (using predicted pg_non_slack and vm_gen)

    Args:
        pd: Load active power (p.u.)
        qd: Load reactive power (p.u.)
        pg_non_slack: Non-slack generator active power (p.u.)
        vm_gen: Generator bus voltage (p.u.)
        params: Parameters dictionary

    Correctly handles sparse bus numbering
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

    load_bus_ids = params['general']['load_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    # Set loads
    for i, bus_id in enumerate(load_bus_ids):
        bus_idx = bus_id_to_idx.get(int(bus_id))
        if bus_idx is not None:
            mpc_pf["bus"][bus_idx, 2] = pd[i] * BASE_MVA
            mpc_pf["bus"][bus_idx, 3] = qd[i] * BASE_MVA

    # Set generator active power (only non-slack generators)
    for i, gen_idx in enumerate(non_slack_gen_idx):
        mpc_pf["gen"][gen_idx, 1] = pg_non_slack[i] * BASE_MVA

    # Set generator voltage (all generators)
    for i in range(n_gen):
        mpc_pf["gen"][i, 5] = vm_gen[i]

    return runpf(mpc_pf, PPOPT)


# =====================================================================
# Evaluation Function (calls acopf_violation_metrics)
# =====================================================================
def evaluate_split(model, X, indices, raw_data, params, scalers, edges, edge_weights, device, split_name, verbose=True):
    """
    Evaluate GNN model - calls acopf_violation_metrics.evaluate_acopf_predictions

    Args:
        ...
        params: Network parameters (can be train_params or test_params)
        ...
    """
    if verbose:
        print(f"\n{split_name} Evaluation:")

    model.eval()
    with torch.no_grad():
        # ✅ Key change: pass params to model for dynamic constraint injection
        y_pred_scaled = model(
            X.to(device),
            edges.to(device),
            edge_weights.to(device),
            params=params  # ← 新增：传入当前评估使用的 params
        )

    y_pred_scaled_np = y_pred_scaled.cpu().numpy()

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    # Denormalize
    y_pred_pg_non_slack = scalers['pg'].inverse_transform(y_pred_scaled_np[:, :n_gen_non_slack])
    y_pred_vm_gen = scalers['vm'].inverse_transform(y_pred_scaled_np[:, n_gen_non_slack:])

    # Reconstruct full arrays
    y_pred_pg_full = reconstruct_full_pg(y_pred_pg_non_slack, params)

    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    y_pred_vm_all = np.zeros((len(X), n_buses), dtype=y_pred_vm_gen.dtype)
    y_pred_vm_all[:, gen_bus_indices] = y_pred_vm_gen
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