# -*- coding: utf-8 -*-
"""
ACOPF Evaluation Metrics Module (V10-PU-Units)

Modifications:
1. evaluate_acopf_predictions: Separate statistics for non-slack, all, and slack MAE_PG
2. All violations in p.u. units
3. Branch violation in p.u. (1.0 = 100% overload)
4. Print separately for three metrics

Includes:
1. Violation calculation: PG/QG/VM/Branch violations
2. MAE calculation: PG/VM/QG/VA accuracy metrics (separate non-slack and all)
3. Cost calculation: Generation cost evaluation
"""
import numpy as np


# =====================================================================
# Violation Calculation Functions - ALL IN P.U. UNITS
# =====================================================================
def calculate_single_sample_violations(r1_pf, is_converged, base_mva):
    """
    Calculate 4 types of maximum violations for single sample (PG/QG/VM/Branch)

    Returns: All violations in p.u. units
    """
    if not is_converged:
        return 1000.0, 1000.0, 1000.0, 1000.0

    gen = r1_pf[0]['gen']
    bus = r1_pf[0]['bus']
    branch = r1_pf[0]['branch']

    # PG violation (convert to p.u.)
    pg_runpf_mw = gen[:, 1]
    pg_min_mw = gen[:, 9]
    pg_max_mw = gen[:, 8]
    pg_viol_mw = np.maximum(0, pg_min_mw - pg_runpf_mw) + np.maximum(0, pg_runpf_mw - pg_max_mw)
    pg_viol_pu = pg_viol_mw / base_mva
    max_pg_viol_pu = np.max(pg_viol_pu) if pg_viol_pu.size > 0 else 0.0

    # QG violation (convert to p.u.)
    qg_runpf_mvar = gen[:, 2]
    qg_min_mvar = gen[:, 4]
    qg_max_mvar = gen[:, 3]
    qg_viol_mvar = np.maximum(0, qg_min_mvar - qg_runpf_mvar) + np.maximum(0, qg_runpf_mvar - qg_max_mvar)
    qg_viol_pu = qg_viol_mvar / base_mva
    max_qg_viol_pu = np.max(qg_viol_pu) if qg_viol_pu.size > 0 else 0.0

    # VM violation (already in p.u.)
    vm_runpf_pu = bus[:, 7]
    vm_min_pu = bus[:, 12]
    vm_max_pu = bus[:, 11]
    vm_viol_pu = np.maximum(0, vm_min_pu - vm_runpf_pu) + np.maximum(0, vm_runpf_pu - vm_max_pu)
    max_vm_viol_pu = np.max(vm_viol_pu) if vm_viol_pu.size > 0 else 0.0

    # Branch violation (in p.u., 1.0 = 100% overload)
    rate_a_mva = branch[:, 5]
    limit_idx = (rate_a_mva > 0) & (rate_a_mva < 9000)
    max_branch_viol_pu = 0.0

    if np.any(limit_idx):
        Ff_MVA = np.abs(branch[limit_idx, 13] + 1j * branch[limit_idx, 14])
        Ft_MVA = np.abs(branch[limit_idx, 15] + 1j * branch[limit_idx, 16])
        rate_a_MVA = rate_a_mva[limit_idx]

        # Violation in p.u. (1.0 means 100% overload)
        ff_viol_pu = np.maximum(0, (Ff_MVA / rate_a_MVA) - 1)
        ft_viol_pu = np.maximum(0, (Ft_MVA / rate_a_MVA) - 1)

        all_branch_viols = np.concatenate([ff_viol_pu, ft_viol_pu])
        max_branch_viol_pu = np.max(all_branch_viols) if all_branch_viols.size > 0 else 0.0

    return max_pg_viol_pu, max_qg_viol_pu, max_vm_viol_pu, max_branch_viol_pu


def extract_pf_results(r1_pf, is_converged, base_mva, n_gen, n_buses):
    """Extract QG and VA values from power flow results"""
    if not is_converged:
        return np.zeros(n_gen), np.zeros(n_buses)

    gen = r1_pf[0]['gen']
    bus = r1_pf[0]['bus']

    qg_pu = gen[:, 2] / base_mva
    va_deg = bus[:, 8]

    return qg_pu, va_deg


# =====================================================================
# MAE Calculation Functions
# =====================================================================
def compute_mae_percentage(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE)

    Normalized by mean of absolute true values
    """
    epsilon = 1e-8
    mae = np.mean(np.abs(y_true - y_pred))
    mean_true = np.mean(np.abs(y_true)) + epsilon
    return 100.0 * mae / mean_true


def compute_mae_absolute(y_true, y_pred):
    """Calculate Mean Absolute Error (MAE)"""
    return np.mean(np.abs(y_true - y_pred))


# =====================================================================
# Cost Calculation Functions
# =====================================================================
def compute_cost_from_pg(pg_pu, cost_coeffs):
    """Calculate generation cost from pg (p.u.)"""
    cost_c2 = cost_coeffs['cost_c2']
    cost_c1 = cost_coeffs['cost_c1']
    cost_c0 = cost_coeffs['cost_c0']

    if pg_pu.ndim == 1:
        cost = np.sum(cost_c2 * pg_pu ** 2 + cost_c1 * pg_pu + cost_c0)
    else:
        cost_per_gen = (cost_c2.reshape(1, -1) * pg_pu ** 2 +
                        cost_c1.reshape(1, -1) * pg_pu +
                        cost_c0.reshape(1, -1))
        cost = np.sum(cost_per_gen, axis=1)

    return cost


def compute_cost_metrics(pg_true, pg_pred, cost_coeffs):
    """Calculate cost-related metrics"""
    cost_true = compute_cost_from_pg(pg_true, cost_coeffs)
    cost_pred = compute_cost_from_pg(pg_pred, cost_coeffs)

    cost_true_mean = np.mean(cost_true)
    cost_pred_mean = np.mean(cost_pred)
    cost_optimality_gap = np.mean((cost_pred - cost_true) / (cost_true + 1e-8)) * 100

    return {
        'cost_true_mean': cost_true_mean,
        'cost_pred_mean': cost_pred_mean,
        'cost_optimality_gap_percent': cost_optimality_gap
    }


# =====================================================================
# KEY MODIFICATION: Comprehensive evaluation with slack classification
# =====================================================================
def evaluate_acopf_predictions(
        y_pred_pg,  # Predicted pg (p.u.), may be full or only non-slack
        y_pred_vm,  # Predicted vm (p.u.)
        y_true_pg,  # True pg (p.u.), must be full (including slack)
        y_true_vm,  # True vm (p.u.)
        y_true_qg,  # True qg (p.u.)
        y_true_va_rad,  # True va (radians)
        pf_results_list,  # Power flow calculation results list
        converge_flags,  # Convergence flags list
        params,  # Parameters dictionary
        verbose=True
):
    """
    Comprehensive evaluation of ACOPF predictions (V10: Support slack classification)

    New features:
        - Separately calculate non-slack, all, and slack MAE_PG
        - Include all three metrics in printing and return
        - All violations in p.u. units
    """
    n_samples = len(y_pred_pg)
    n_gen = params['general']['n_gen']
    n_buses = params['general']['n_buses']
    base_mva = params['general']['BASE_MVA']

    # Check if slack_gen_mask exists (backward compatibility)
    has_slack_info = 'slack_gen_mask' in params['general']

    # ==================== 1. Extract QG and VA from power flow results ====================
    y_pred_qg_pf = np.zeros((n_samples, n_gen))
    y_pred_va_pf = np.zeros((n_samples, n_buses))

    max_pg_viol_per_sample = np.zeros(n_samples)
    max_qg_viol_per_sample = np.zeros(n_samples)
    max_vm_viol_per_sample = np.zeros(n_samples)
    max_branch_viol_per_sample = np.zeros(n_samples)

    n_converged = 0

    for i in range(n_samples):
        is_converged = converge_flags[i]
        if is_converged:
            n_converged += 1

        qg_pu, va_deg = extract_pf_results(
            pf_results_list[i], is_converged, base_mva, n_gen, n_buses
        )
        y_pred_qg_pf[i, :] = qg_pu
        y_pred_va_pf[i, :] = va_deg

        pg_vio, qg_vio, vm_vio, branch_vio = calculate_single_sample_violations(
            pf_results_list[i], is_converged, base_mva
        )
        max_pg_viol_per_sample[i] = pg_vio
        max_qg_viol_per_sample[i] = qg_vio
        max_vm_viol_per_sample[i] = vm_vio
        max_branch_viol_per_sample[i] = branch_vio

    # ==================== 2. Extract full Pg from power flow results ====================
    y_pred_pg_pf_full = np.zeros((n_samples, n_gen))
    for i in range(n_samples):
        if converge_flags[i]:
            pg_mw = pf_results_list[i][0]['gen'][:, 1]
            y_pred_pg_pf_full[i, :] = pg_mw / base_mva
        else:
            if y_pred_pg.shape[1] == n_gen:
                y_pred_pg_pf_full[i, :] = y_pred_pg[i, :]
            else:
                y_pred_pg_pf_full[i, :] = 0

    # ==================== 3. Calculate MAE_PG separately (key modification) ====================
    if has_slack_info:
        slack_gen_mask = params['general']['slack_gen_mask']

        mae_pg_non_slack = compute_mae_percentage(
            y_true_pg[:, ~slack_gen_mask],
            y_pred_pg_pf_full[:, ~slack_gen_mask]
        )

        mae_pg_slack = compute_mae_percentage(
            y_true_pg[:, slack_gen_mask],
            y_pred_pg_pf_full[:, slack_gen_mask]
        )

        mae_pg_all = compute_mae_percentage(
            y_true_pg,
            y_pred_pg_pf_full
        )
    else:
        # Backward compatibility: no slack information, only calculate overall
        mae_pg_all = compute_mae_percentage(y_true_pg, y_pred_pg_pf_full)
        mae_pg_non_slack = mae_pg_all
        mae_pg_slack = 0.0

    # ==================== 4. Other MAE metrics ====================
    mae_vm = compute_mae_percentage(y_true_vm, y_pred_vm)
    mae_qg = compute_mae_percentage(y_true_qg, y_pred_qg_pf)

    y_true_va_deg = y_true_va_rad * (180.0 / np.pi)
    mae_va_deg = compute_mae_absolute(y_true_va_deg, y_pred_va_pf)

    # ==================== 5. Cost calculation ====================
    cost_metrics = compute_cost_metrics(y_true_pg, y_pred_pg_pf_full, params['generator'])

    # ==================== 6. Violation metrics summary (with slack classification) ====================
    mean_max_pg_viol_pu = np.mean(max_pg_viol_per_sample)
    mean_max_qg_viol_pu = np.mean(max_qg_viol_per_sample)
    mean_max_vm_viol_pu = np.mean(max_vm_viol_per_sample)
    mean_max_branch_viol_pu = np.mean(max_branch_viol_per_sample)
    convergence_rate = (n_converged / n_samples) * 100

    # Separately calculate Slack and Non-Slack PG violations
    if has_slack_info:
        slack_pg_viols = []
        non_slack_pg_viols = []
        slack_gen_idx = np.where(slack_gen_mask)[0]

        for i in range(n_samples):
            if converge_flags[i]:
                gen = pf_results_list[i][0]['gen']
                pg_mw = gen[:, 1]
                pg_min = gen[:, 9]
                pg_max = gen[:, 8]
                viols_mw = np.maximum(0, pg_min - pg_mw) + np.maximum(0, pg_mw - pg_max)
                viols_pu = viols_mw / base_mva

                # Slack violation
                slack_pg_viols.append(viols_pu[slack_gen_idx[0]] if len(slack_gen_idx) > 0 else 0.0)

                # Non-Slack maximum violation
                non_slack_pg_viols.append(np.max(viols_pu[~slack_gen_mask]) if np.any(~slack_gen_mask) else 0.0)
            else:
                slack_pg_viols.append(1000.0)
                non_slack_pg_viols.append(1000.0)

        mean_slack_pg_viol = np.mean(slack_pg_viols)
        mean_non_slack_pg_viol = np.mean(non_slack_pg_viols)
    else:
        mean_slack_pg_viol = 0.0
        mean_non_slack_pg_viol = mean_max_pg_viol_pu

    # ==================== 7. Return all metrics (with classification) ====================
    metrics = {
        # MAE metrics (classified)
        'mae_pg_non_slack_percent': mae_pg_non_slack,  # Main metric
        'mae_pg_all_percent': mae_pg_all,  # Auxiliary metric
        'mae_pg_slack_percent': mae_pg_slack,  # For analysis

        # Other MAE metrics
        'mae_vm_percent': mae_vm,
        'mae_qg_percent': mae_qg,
        'mae_va_deg': mae_va_deg,

        # Cost metrics
        'cost_true_mean': cost_metrics['cost_true_mean'],
        'cost_pred_mean': cost_metrics['cost_pred_mean'],
        'cost_optimality_gap_percent': cost_metrics['cost_optimality_gap_percent'],

        # Violation metrics (classified, all in p.u.)
        'mean_max_pg_viol_pu': mean_max_pg_viol_pu,  # System-level
        'mean_pg_viol_non_slack_pu': mean_non_slack_pg_viol,  # Non-Slack
        'mean_pg_viol_slack_pu': mean_slack_pg_viol,  # Slack

        'mean_max_qg_viol_pu': mean_max_qg_viol_pu,
        'mean_max_vm_viol_pu': mean_max_vm_viol_pu,
        'mean_max_branch_viol_pu': mean_max_branch_viol_pu,
        'convergence_rate_percent': convergence_rate,
    }

    return metrics