# gnn_utils_dcopf.py
"""
GNN Utility Functions - DCOPF Version
Build graph structure using real power system topology

Version: v4.1 - API_TEST mode with dynamic test graph support
"""

import numpy as np
import pandas as pd
import torch
import os
import sys

sys.path.append(os.path.dirname(__file__))

from dcopf_violation_metrics import (
    feasibility,
    compute_branch_violation_pu,
    compute_cost,
    compute_cost_gap_percentage
)

from dcopf_slack_utils import (
    reconstruct_full_pg,
    compute_detailed_mae,
    compute_detailed_pg_violations_pu
)


# =====================================================================
# Graph Structure Building
# =====================================================================
def build_graph_from_real_topology(params, params_path, case_name, is_api=False):
    """
    Build graph structure from real branch information

    Requires enhanced Julia script to generate *_branch_info.csv

    Args:
        params: Parameter dictionary from load_parameters_from_csv()
        params_path: Parameter file directory
        case_name: Case name (e.g., 'pglib_opf_case14_ieee')
        is_api: Whether this is for API test data (affects file naming)

    Returns:
        edges: [2, num_edges] - Edge indices (PyG format)
        edge_weights: [num_edges, 2] - Edge features [r_pu, x_pu]
        node_types: [n_buses, 2] - Node types [is_gen, is_load]
    """
    n_buses = params['general']['n_buses']
    bus_ids = params['general']['l_bus']

    bus_id_to_idx = {int(bid): i for i, bid in enumerate(bus_ids)}

    # Handle API file naming convention (double underscore)
    suffix = "__api" if is_api else ""
    branch_info_path = os.path.join(params_path, f"{case_name}{suffix}_branch_info.csv")

    if not os.path.exists(branch_info_path):
        print(f"\nWarning: {case_name}_branch_info.csv not found")
        print(f"   Using fallback (infer from PTDF)")
        print(f"   Recommend running enhanced Julia script")
        return build_graph_from_ptdf_fallback(params)

    branch_df = pd.read_csv(branch_info_path)

    print(f"\nBuilding graph from real branch info:")
    print(f"  File: {branch_info_path}")
    print(f"  Branches: {len(branch_df)}")

    edges_list = []
    edge_weights_list = []

    for _, row in branch_df.iterrows():
        f_bus_id = int(row['f_bus'])
        t_bus_id = int(row['t_bus'])
        r_pu = float(row['r_pu'])
        x_pu = float(row['x_pu'])

        if f_bus_id not in bus_id_to_idx or t_bus_id not in bus_id_to_idx:
            print(f"  [Skip] Branch {row['branch_id']}: "
                  f"Bus {f_bus_id}->{t_bus_id} not in network")
            continue

        f_idx = bus_id_to_idx[f_bus_id]
        t_idx = bus_id_to_idx[t_bus_id]

        # Bidirectional edges
        edges_list.append([f_idx, t_idx])
        edges_list.append([t_idx, f_idx])

        edge_feat = [r_pu, x_pu]
        edge_weights_list.append(edge_feat)
        edge_weights_list.append(edge_feat)

    if len(edges_list) == 0:
        raise ValueError("Cannot build graph: no valid branch connections!")

    edges = torch.tensor(edges_list, dtype=torch.long).t()
    edge_weights = torch.tensor(edge_weights_list, dtype=torch.float32)

    node_types = torch.zeros(n_buses, 2, dtype=torch.float32)
    gen_bus_ids = params['general']['g_bus']

    for gid in gen_bus_ids:
        idx = bus_id_to_idx[int(gid)]
        node_types[idx, 0] = 1.0  # is_gen

    node_types[:, 1] = 1.0  # is_load (all buses in DCOPF)

    print(f"  Nodes: {n_buses}")
    print(f"  Edges: {edges.shape[1]} (bidirectional)")
    print(f"  Gen nodes: {int(node_types[:, 0].sum())}")
    print(f"  Edge features: r_pu, x_pu")

    return edges, edge_weights, node_types


def build_graph_from_ptdf_fallback(params):
    """
    Fallback: infer graph from PTDF matrix

    Used when branch_info.csv is not available
    """
    print("\nUsing PTDF fallback to build graph...")

    n_buses = params['general']['n_buses']
    PTDF = params['constraints']['PTDF']

    bus_ids = params['general']['l_bus']
    bus_id_to_idx = {int(bid): i for i, bid in enumerate(bus_ids)}

    edges_set = set()
    edge_features = {}

    for branch_idx in range(PTDF.shape[1]):
        ptdf_col = PTDF[:, branch_idx]
        abs_ptdf = np.abs(ptdf_col)

        if abs_ptdf.max() > 1e-6:
            top_2_indices = np.argsort(abs_ptdf)[-2:]

            if len(top_2_indices) >= 2 and abs_ptdf[top_2_indices[0]] > 1e-6:
                i, j = int(top_2_indices[0]), int(top_2_indices[1])

                if i != j:
                    edges_set.add((i, j))
                    edges_set.add((j, i))

                    weight = float(abs_ptdf[i] + abs_ptdf[j])
                    edge_features[(i, j)] = [0.01, weight]
                    edge_features[(j, i)] = [0.01, weight]

    if len(edges_set) < n_buses - 1:
        print("  Warning: Too few edges from PTDF, using simple chain graph")
        return build_simple_chain_graph(params)

    edges_list = list(edges_set)
    edge_weights_list = [edge_features[e] for e in edges_list]

    edges = torch.tensor(edges_list, dtype=torch.long).t()
    edge_weights = torch.tensor(edge_weights_list, dtype=torch.float32)

    node_types = torch.zeros(n_buses, 2, dtype=torch.float32)
    gen_bus_ids = params['general']['g_bus']

    for gid in gen_bus_ids:
        idx = bus_id_to_idx[int(gid)]
        node_types[idx, 0] = 1.0

    node_types[:, 1] = 1.0

    print(f"  Completed from PTDF:")
    print(f"    Nodes: {n_buses}")
    print(f"    Edges: {edges.shape[1]}")

    return edges, edge_weights, node_types


def build_simple_chain_graph(params):
    """
    Simple chain graph (last fallback)

    Warning: Inaccurate structure, for testing only!
    """
    print("\nWarning: Using simple chain graph (testing only)")

    n_buses = params['general']['n_buses']
    bus_ids = params['general']['l_bus']
    bus_id_to_idx = {int(bid): i for i, bid in enumerate(bus_ids)}

    edges_list = []
    edge_weights_list = []

    for i in range(n_buses - 1):
        edges_list.append([i, i + 1])
        edges_list.append([i + 1, i])
        edge_weights_list.append([0.01, 0.05])
        edge_weights_list.append([0.01, 0.05])

    for i in range(0, n_buses - 3, 3):
        edges_list.append([i, i + 3])
        edges_list.append([i + 3, i])
        edge_weights_list.append([0.02, 0.10])
        edge_weights_list.append([0.02, 0.10])

    edges = torch.tensor(edges_list, dtype=torch.long).t()
    edge_weights = torch.tensor(edge_weights_list, dtype=torch.float32)

    node_types = torch.zeros(n_buses, 2, dtype=torch.float32)
    gen_bus_ids = params['general']['g_bus']

    for gid in gen_bus_ids:
        idx = bus_id_to_idx[int(gid)]
        node_types[idx, 0] = 1.0

    node_types[:, :, 4] = 1.0

    print(f"  Nodes: {n_buses}")
    print(f"  Edges: {edges.shape[1]}")

    return edges, edge_weights, node_types


# =====================================================================
# Evaluation Function (with dynamic test graph support)
# =====================================================================
def evaluate_split(
        model,
        X,
        indices,
        raw_data,
        params,
        scalers,
        edges,
        edge_weights,
        device,
        test_data_external=None,
        test_params=None,
        test_edges=None,
        test_edge_weights=None
):
    """
    Evaluate GNN model (v4.1 - with test graph support)

    Key Feature: API_TEST mode support
    ----------------------------------
    - Accepts optional test_edges and test_edge_weights
    - Uses test graph structure for API testing
    - Passes test_params to model forward for dynamic node features

    This enables GNN to truly leverage:
    1. API's graph topology (different edges)
    2. API's edge parameters (different r_pu, x_pu)
    3. API's constraint parameters (different Pg_min, Pg_max in node features)

    Args:
        model: GNN model
        X: Input features (train/val set)
        indices: Data indices (train/val set)
        raw_data: Raw data dict
        params: Training params
        scalers: Data scalers
        edges: Training graph edges
        edge_weights: Training graph edge weights
        device: torch.device
        test_data_external: Optional external test data
        test_params: Optional test params (API params)
        test_edges: Optional test graph edges (API graph)
        test_edge_weights: Optional test graph edge weights (API graph)

    Returns:
        metrics: Dict of evaluation metrics
    """

    eval_params = test_params if test_params is not None else params

    # Decide which graph structure to use
    if test_edges is not None and test_edge_weights is not None:
        eval_edges = test_edges
        eval_edge_weights = test_edge_weights
        print("\n[Eval] Using TEST graph structure (API mode)")
        print(f"  Test edges: {eval_edges.shape[1]}")
        print(f"  Node features will use API constraints")
    else:
        eval_edges = edges
        eval_edge_weights = edge_weights
        print("\n[Eval] Using TRAIN graph structure")

    model.eval()

    if test_data_external is not None:
        # External test data (GENERALIZATION or API_TEST mode)
        x_raw_eval = test_data_external['x']
        y_true_pg_all = test_data_external['y_pg_all']

        x_scaled = scalers['x'].transform(x_raw_eval)
        X_eval = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            # Pass eval_params to model forward (API constraints)
            y_pred_non_slack_scaled = model(
                X_eval.to(device),
                eval_edges.to(device),
                eval_edge_weights.to(device),
                params=eval_params  # Dynamic params for node features
            )
    else:
        # Internal test data (RANDOM_SPLIT or VALID_FIXED mode)
        with torch.no_grad():
            y_pred_non_slack_scaled = model(
                X.to(device),
                eval_edges.to(device),
                eval_edge_weights.to(device),
                params=eval_params
            )

        x_raw_eval = raw_data['x'][indices]
        y_true_pg_all = raw_data['y_pg_all'][indices]

    # Inverse transform predictions
    y_pred_non_slack_scaled_np = y_pred_non_slack_scaled.cpu().numpy()
    y_pred_non_slack = scalers['y_pg_non_slack'].inverse_transform(y_pred_non_slack_scaled_np)

    # Get raw input data
    if test_data_external is None:
        x_raw = scalers['x'].inverse_transform(X.cpu().numpy())
    else:
        x_raw = x_raw_eval

    # Reconstruct full Pg (including Slack)
    pd_total = x_raw.sum(axis=1)
    y_pred_all = reconstruct_full_pg(
        pg_non_slack=y_pred_non_slack,
        pd_total=pd_total,
        params=eval_params
    )

    # Calculate MAE
    mae_dict = compute_detailed_mae(
        y_true_all=y_true_pg_all,
        y_pred_non_slack=y_pred_non_slack,
        y_pred_all=y_pred_all,
        params=eval_params
    )

    # Calculate violations (p.u.)
    gen_up_viol, gen_lo_viol, line_viol, balance_err = feasibility(
        y_pred_pg=y_pred_all,
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
    cost_pred = compute_cost(y_pred_all, cost_coeffs)
    cost_gap_pct = compute_cost_gap_percentage(cost_true, cost_pred)

    return {
        'mae_pg_non_slack': mae_dict['mae_non_slack'],
        'mae_pg_slack': mae_dict['mae_slack'],
        'viol_pg_non_slack': viol_dict['viol_non_slack'],
        'viol_pg_slack': viol_dict['viol_slack'],
        'viol_branch': viol_branch_pu,
        'cost_gap_percent': cost_gap_pct,
    }