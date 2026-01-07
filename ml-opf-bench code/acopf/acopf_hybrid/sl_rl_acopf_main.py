# -*- coding: utf-8 -*-
"""
SL + RL OPF Solver (V8 - Simplified Output)

Phase 1: Supervised Learning Pretraining
Phase 2: Reinforcement Learning (PPO) Fine-tuning - Using Delta Learning

Modifications (V8):
- DNN only predicts non-slack pg and generator vm
- Simplified output: only test set metrics
- All comments and outputs in English
- Removed all file saving (JSON/CSV/model)
- All violations in p.u. units
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import gymnasium as gym
from gymnasium import spaces

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
    from acopf_violation_metrics import (
        calculate_single_sample_violations,
        evaluate_acopf_predictions
    )
except ImportError:
    print("Error: Unable to import 'acopf_violation_metrics' module.")
    sys.exit(1)

# PyPower
from pypower.runpf import runpf
from pypower.ppoption import ppoption

# Try to import SB3
from stable_baselines3 import PPO

HAS_SB3 = True

torch.set_default_dtype(torch.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"{'=' * 70}")
print(f"SL+RL OPF System (V8)")
print(f"Device: {DEVICE} | SB3: {HAS_SB3}")
print(f"{'=' * 70}\n")

# =====================================================================
# Global Variables
# =====================================================================
GLOBAL_CASE_DATA = None
PPOPT = None


# =====================================================================
# PyPower Interface
# =====================================================================
def init_pypower_options():
    """Initialize PyPower options"""
    global PPOPT
    ppopt = ppoption()
    PPOPT = ppoption(ppopt, OUT_ALL=0, VERBOSE=0, ENFORCE_Q_LIMS=0)


def load_case_from_csv(case_name, constraints_path):
    """
    Load PyPower case data from CSV files

    Note: case_name should be the full case name (including __api suffix)
    """
    import pandas as pd

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
    gencost[:, 4] = gen_df['cost_c2'].values / (baseMVA ** 2)
    gencost[:, 5] = gen_df['cost_c1'].values / baseMVA
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


def solve_pf_custom_optimized(pd, qd, pg_non_slack, vm_gen, params):
    """
    Power flow calculation

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
# Model Definition
# =====================================================================
class UnifiedOPFPolicy(nn.Module):
    """Unified OPF policy network"""

    def __init__(self, input_size, output_size, hidden_layers=[256, 256]):
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =====================================================================
# Supervised Learning Trainer
# =====================================================================
class SupervisedTrainer:
    """Supervised learning trainer"""

    def __init__(self, model: nn.Module, params: Dict, scalers: Dict,
                 learning_rate: float = 1e-3, device: torch.device = DEVICE):
        self.model = model.to(device)
        self.params = params
        self.scalers = scalers
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train_epoch(self, X_train, Y_train, batch_size) -> float:
        self.model.train()
        n_train = len(X_train)
        n_batches = (n_train + batch_size - 1) // batch_size

        epoch_loss = 0.0
        indices = torch.randperm(n_train, device=self.device)

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            batch_indices = indices[start_idx:end_idx]

            X_batch = X_train[batch_indices]
            Y_batch = Y_train[batch_indices]

            self.optimizer.zero_grad()
            Y_pred = self.model(X_batch)
            loss = self.criterion(Y_pred, Y_batch)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * len(X_batch)

        return epoch_loss / n_train

    def validate(self, X_val, Y_val) -> float:
        self.model.eval()
        with torch.no_grad():
            pred_val = self.model(X_val)
            val_loss = float(self.criterion(pred_val, Y_val).item())
        return val_loss

    def train(self, X_train, Y_train, X_val, Y_val, n_epochs: int = 1000,
              batch_size: int = 256):
        history = {'train_loss': [], 'val_loss': []}

        print(f"\n{'=' * 70}")
        print(f"Phase 1: Supervised Learning Pretraining")
        print(f"{'=' * 70}")

        t0 = time.time()
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(X_train, Y_train, batch_size)
            val_loss = self.validate(X_val, Y_val)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == n_epochs - 1:
                print(f"Epoch {epoch + 1:4d}/{n_epochs}: "
                      f"Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        train_time = time.time() - t0
        print(f"\n✓ SL training completed in {train_time:.2f} seconds")
        return history, train_time


# =====================================================================
# Evaluation Functions
# =====================================================================
def evaluate_model(model, X, indices, raw_data, params, scalers, device, split_name="Test", verbose=True):
    """
    Evaluate model - calls acopf_violation_metrics.evaluate_acopf_predictions
    """
    if verbose:
        print(f"\n{split_name} Evaluation:")

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X.to(device))

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


def evaluate_rl_model(agent, X_test, test_idx, pretrained_model, params, scalers,
                      raw_data, device, adjust_vm=True, split_name="Test (RL)", verbose=True):
    """Evaluate RL fine-tuned model"""
    if verbose:
        print(f"\n{split_name} Evaluation:")

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    n_samples = len(X_test)

    y_pred_pg_non_slack_list = []
    y_pred_vm_gen_list = []

    pretrained_model.eval()
    for i in range(n_samples):
        X_single = X_test[i:i + 1].to(device)

        with torch.no_grad():
            sl_pred = pretrained_model(X_single)
            pg_sl_scaled = sl_pred[:, :n_gen_non_slack].cpu().numpy()
            vm_sl_scaled = sl_pred[:, n_gen_non_slack:].cpu().numpy()

        if adjust_vm:
            obs = torch.cat([X_single, sl_pred[:, :n_gen_non_slack], sl_pred[:, n_gen_non_slack:]], dim=1)
        else:
            obs = torch.cat([X_single, sl_pred[:, :n_gen_non_slack]], dim=1)
        obs_np = obs.cpu().numpy().astype(np.float32).flatten()

        action, _ = agent.predict(obs_np, deterministic=True)
        action = np.clip(action, -1.0, 1.0)

        if adjust_vm:
            action_pg = action[:n_gen_non_slack]
            action_vm = action[n_gen_non_slack:]
        else:
            action_pg = action
            action_vm = None

        # Apply delta to non-slack pg
        pg_sl = scalers['pg'].inverse_transform(pg_sl_scaled)
        pg_min = params['generator']['pg_min'].flatten()[non_slack_gen_idx]
        pg_max = params['generator']['pg_max'].flatten()[non_slack_gen_idx]
        pg_range = (pg_max - pg_min)
        delta_scale_pg = 0.1
        delta_pg = action_pg * delta_scale_pg * pg_range
        pg_non_slack_new = np.clip(pg_sl + delta_pg, pg_min, pg_max)

        if adjust_vm:
            vm_sl = scalers['vm'].inverse_transform(vm_sl_scaled)
            vm_min = params['bus']['vm_min'][gen_bus_ids]
            vm_max = params['bus']['vm_max'][gen_bus_ids]
            vm_range = (vm_max - vm_min)
            delta_scale_vm = 0.05
            delta_vm = action_vm * delta_scale_vm * vm_range
            vm_gen_new = np.clip(vm_sl + delta_vm, vm_min, vm_max)
        else:
            vm_gen_new = scalers['vm'].inverse_transform(vm_sl_scaled)

        y_pred_pg_non_slack_list.append(pg_non_slack_new.flatten())
        y_pred_vm_gen_list.append(vm_gen_new.flatten())

    y_pred_pg_non_slack = np.array(y_pred_pg_non_slack_list)
    y_pred_vm_gen = np.array(y_pred_vm_gen_list)

    # Reconstruct full arrays
    y_pred_pg_full = reconstruct_full_pg(y_pred_pg_non_slack, params)

    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    y_pred_vm_all = np.zeros((n_samples, n_buses), dtype=y_pred_vm_gen.dtype)
    y_pred_vm_all[:, gen_bus_indices] = y_pred_vm_gen
    non_gen_mask = np.ones(n_buses, dtype=bool)
    non_gen_mask[gen_bus_indices] = False
    y_pred_vm_all[:, non_gen_mask] = 1.0

    # True values
    y_true_pg = raw_data['pg'][test_idx]
    y_true_vm = raw_data['vm'][test_idx]
    y_true_qg = raw_data['qg'][test_idx]
    y_true_va_rad = raw_data['va'][test_idx]

    # Loads
    x_raw_data = scalers['x'].inverse_transform(X_test.cpu().numpy())
    pd_pu = x_raw_data[:, :n_loads]
    qd_pu = x_raw_data[:, n_loads:]

    # Run power flow calculation
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
# RL Environment
# =====================================================================
class OPFGymEnv(gym.Env):
    """RL Environment: Learn fine-tuning of SL predictions (Delta Learning)"""
    metadata = {'render_modes': ['human']}

    def __init__(self, X_data, pretrained_model, params, scalers, raw_data, indices, adjust_vm=True):
        super().__init__()
        self.X_data = X_data
        self.pretrained_model = pretrained_model.eval()
        self.params = params
        self.scalers = scalers
        self.raw_data = raw_data
        self.indices = indices
        self.device = next(pretrained_model.parameters()).device
        self.delta_scale_pg = 0.1
        self.delta_scale_vm = 0.05
        self.adjust_vm = adjust_vm

        n_gen = params['general']['n_gen']
        n_gen_non_slack = params['general']['n_gen_non_slack']
        n_buses = params['general']['n_buses']
        gen_bus_ids = params['general']['gen_bus_ids']

        # Action space: delta adjustments for non-slack pg (and optionally generator vm)
        if adjust_vm:
            self.action_space = spaces.Box(low=-1.0, high=1.0,
                                           shape=(n_gen_non_slack + n_gen,), dtype=np.float32)
            obs_dim = X_data.shape[1] + n_gen_non_slack + n_gen
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0,
                                           shape=(n_gen_non_slack,), dtype=np.float32)
            obs_dim = X_data.shape[1] + n_gen_non_slack

        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self.current_X = None
        self.current_idx = None
        self.sl_pred = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        idx = np.random.randint(0, len(self.indices))
        self.current_idx = self.indices[idx]
        self.current_X = self.X_data[idx:idx + 1].to(self.device)

        with torch.no_grad():
            self.sl_pred = self.pretrained_model(self.current_X)

        n_gen_non_slack = self.params['general']['n_gen_non_slack']

        pg_sl_scaled = self.sl_pred[:, :n_gen_non_slack]

        if self.adjust_vm:
            vm_sl_scaled = self.sl_pred[:, n_gen_non_slack:]
            obs = torch.cat([self.current_X, pg_sl_scaled, vm_sl_scaled], dim=1)
        else:
            obs = torch.cat([self.current_X, pg_sl_scaled], dim=1)

        return obs.cpu().numpy().astype(np.float32).flatten(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        n_gen = self.params['general']['n_gen']
        n_gen_non_slack = self.params['general']['n_gen_non_slack']
        n_buses = self.params['general']['n_buses']
        n_loads = self.params['general']['n_loads']
        BASE_MVA = self.params['general']['BASE_MVA']
        non_slack_gen_idx = self.params['general']['non_slack_gen_idx']
        gen_bus_ids = self.params['general']['gen_bus_ids']
        bus_id_to_idx = self.params['general']['bus_id_to_idx']

        with torch.no_grad():
            pg_sl_scaled = self.sl_pred[:, :n_gen_non_slack].cpu().numpy()
            vm_sl_scaled = self.sl_pred[:, n_gen_non_slack:].cpu().numpy()

            if self.adjust_vm:
                action_pg = action[:n_gen_non_slack]
                action_vm = action[n_gen_non_slack:]
            else:
                action_pg = action
                action_vm = None

            # Apply delta to non-slack pg
            pg_sl = self.scalers['pg'].inverse_transform(pg_sl_scaled)
            pg_min = self.params['generator']['pg_min'].flatten()[non_slack_gen_idx]
            pg_max = self.params['generator']['pg_max'].flatten()[non_slack_gen_idx]
            pg_range = (pg_max - pg_min)
            delta_pg = action_pg * self.delta_scale_pg * pg_range
            pg_non_slack_new = np.clip(pg_sl + delta_pg, pg_min, pg_max)

            if self.adjust_vm:
                vm_sl = self.scalers['vm'].inverse_transform(vm_sl_scaled)
                gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
                vm_min = self.params['bus']['vm_min'][gen_bus_indices]
                vm_max = self.params['bus']['vm_max'][gen_bus_indices]
                vm_range = (vm_max - vm_min)
                delta_vm = action_vm * self.delta_scale_vm * vm_range
                vm_gen_new = np.clip(vm_sl + delta_vm, vm_min, vm_max)
            else:
                vm_gen_new = self.scalers['vm'].inverse_transform(vm_sl_scaled)

            pg_new_scaled = self.scalers['pg'].transform(pg_non_slack_new)
            vm_new_scaled = self.scalers['vm'].transform(vm_gen_new)

            Y_new_scaled = np.hstack([pg_new_scaled, vm_new_scaled])
            Y_new_tensor = torch.tensor(Y_new_scaled, dtype=torch.float32, device=self.device)

            Y_true_scaled = torch.tensor(
                np.hstack([
                    self.scalers['pg'].transform(self.raw_data['pg_non_slack'][self.current_idx:self.current_idx + 1]),
                    self.scalers['vm'].transform(self.raw_data['vm_gen'][self.current_idx:self.current_idx + 1])
                ]),
                dtype=torch.float32,
                device=self.device
            )

            mse_new = torch.mean((Y_new_tensor - Y_true_scaled) ** 2).item()
            mse_sl = torch.mean((self.sl_pred - Y_true_scaled) ** 2).item()
            mse_improvement = (mse_sl - mse_new) / (mse_sl + 1e-6)

            x_raw = self.scalers['x'].inverse_transform(self.current_X.cpu().numpy())
            pd_pu = x_raw[0, :n_loads]
            qd_pu = x_raw[0, n_loads:]

            try:
                r1_pf = solve_pf_custom_optimized(pd_pu, qd_pu, pg_non_slack_new[0], vm_gen_new[0], self.params)
                is_converged = r1_pf[0]['success']

                pg_vio, qg_vio, vm_vio, branch_vio = calculate_single_sample_violations(
                    r1_pf=r1_pf, is_converged=is_converged, base_mva=BASE_MVA
                )

                # Normalize violations (already in p.u. from new metrics module)
                pg_vio_norm = min(pg_vio / 0.5, 2.0)
                qg_vio_norm = min(qg_vio / 5.0, 2.0)
                vm_vio_norm = min(vm_vio / 0.1, 2.0)
                branch_vio_norm = min(branch_vio / 0.1, 2.0)

                violation_penalty = pg_vio_norm + qg_vio_norm + vm_vio_norm + branch_vio_norm
                convergence_bonus = 20.0 if is_converged else -50.0

            except Exception:
                violation_penalty = 8.0
                convergence_bonus = -50.0
                pg_vio = qg_vio = vm_vio = branch_vio = 1000.0
                is_converged = False

            reward = 100.0 * mse_improvement + convergence_bonus - 50.0 * violation_penalty

            if self.adjust_vm:
                obs = torch.cat([self.current_X, Y_new_tensor[:, :n_gen_non_slack], Y_new_tensor[:, n_gen_non_slack:]],
                                dim=1)
            else:
                obs = torch.cat([self.current_X, Y_new_tensor[:, :n_gen_non_slack]], dim=1)

        done = True
        obs_np = obs.cpu().numpy().astype(np.float32).flatten()

        info = {
            'mse': float(mse_new),
            'mse_sl': float(mse_sl),
            'mse_improvement': float(mse_improvement * 100),
            'pg_viol': float(pg_vio),
            'qg_viol': float(qg_vio),
            'vm_viol': float(vm_vio),
            'branch_viol': float(branch_vio),
            'converged': is_converged,
        }

        return obs_np, float(reward), done, False, info


def train_rl_phase(X_train, pretrained_model, params, scalers, raw_data, train_idx,
                   n_steps: int = 10000, learning_rate: float = 3e-4,
                   adjust_vm: bool = True) -> tuple:
    """RL fine-tuning phase"""
    print(f"\n{'=' * 70}")
    print(f"Phase 2: Reinforcement Learning Fine-tuning (PPO + Delta Learning)")
    print(f"{'=' * 70}")

    rl_env = OPFGymEnv(X_train, pretrained_model, params, scalers, raw_data, train_idx, adjust_vm=adjust_vm)

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']

    if adjust_vm:
        policy_net_arch = dict(pi=[512, 512], vf=[512, 512])
    else:
        policy_net_arch = dict(pi=[256, 256], vf=[256, 256])

    agent = PPO(
        "MlpPolicy",
        rl_env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,  # Suppress PPO training output
        device='cpu',
        policy_kwargs=dict(net_arch=policy_net_arch)
    )

    t0 = time.time()
    agent.learn(total_timesteps=n_steps)
    rl_train_time = time.time() - t0
    print(f"\n✓ RL training completed in {rl_train_time:.2f} seconds")

    return agent, rl_train_time


# =====================================================================
# Main Experiment Function
# =====================================================================
def run_sl_rl_experiment(
        case_name: str,
        params_path: str,
        data_path: str,
        log_path: str,
        results_path: str,
        # Data mode parameters
        data_mode: str = 'random_split',
        n_train_use: Optional[int] = None,
        test_data_path: Optional[str] = None,
        test_params_path: Optional[str] = None,
        n_test_samples: Optional[int] = None,
        # SL parameters
        hidden_sizes: list = [256, 256],
        n_epochs: int = 500,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        # RL parameters
        enable_rl: bool = False,
        rl_steps: int = 10000,
        rl_learning_rate: float = 3e-4,
        adjust_vm: bool = True,
        # Other
        seed: int = 42,
        device: str = 'cuda',
) -> Dict:
    """
    Complete SL+RL experiment

    Supports four data modes:
    - RANDOM_SPLIT: Random split
    - FIXED_VALTEST: Fixed validation/test sets
    - GENERALIZATION: Cross-distribution generalization test
    - API_TEST: API data test (different topology)
    """
    global GLOBAL_CASE_DATA, PPOPT

    torch.manual_seed(seed)
    np.random.seed(seed)
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print(f"SL+RL ACOPF Experiment")
    print(f"{'=' * 70}")
    print(f"Case: {case_name}")
    print(f"Data Mode: {data_mode}")
    print(f"Device: {device_obj}")
    print(f"SL Epochs: {n_epochs} | RL Steps: {rl_steps if enable_rl else 0}")
    print(f"{'=' * 70}")

    # ========================================================================
    # 1. Load training parameters and PyPower case data
    # ========================================================================
    print(f"\n[Step 1] Loading training parameters and case data...")
    params = load_parameters_from_csv(case_name, params_path)
    init_pypower_options()
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
    # 4. Prepare tensors
    # ========================================================================
    print(f"\n[Dataset Sizes]")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")

    X_train = torch.tensor(x_data_scaled[train_idx], dtype=torch.float32, device=device_obj)
    Y_train = torch.tensor(y_data_scaled[train_idx], dtype=torch.float32, device=device_obj)
    X_val = torch.tensor(x_data_scaled[val_idx], dtype=torch.float32, device=device_obj)
    Y_val = torch.tensor(y_data_scaled[val_idx], dtype=torch.float32, device=device_obj)
    X_test = torch.tensor(test_x_scaled[test_idx], dtype=torch.float32)

    # ========================================================================
    # 5. Phase 1: Supervised Learning
    # ========================================================================
    print(f"\n[Step 3] Creating model...")
    model = UnifiedOPFPolicy(
        input_size=x_data_scaled.shape[1],
        output_size=y_data_scaled.shape[1],
        hidden_layers=hidden_sizes,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Network: {x_data_scaled.shape[1]} -> {' -> '.join(map(str, hidden_sizes))} -> {y_data_scaled.shape[1]}")
    print(f"  Total params: {total_params:,}")

    print(f"\n[Step 4] SL pretraining...")
    trainer = SupervisedTrainer(model, params, scalers, learning_rate, device_obj)

    sl_history, sl_train_time = trainer.train(
        X_train, Y_train, X_val, Y_val,
        n_epochs=n_epochs,
        batch_size=batch_size
    )

    # ========================================================================
    # 6. SL phase evaluation (using test params)
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"SL Phase - Test Set Evaluation")
    print(f"{'=' * 70}")

    GLOBAL_CASE_DATA_BACKUP = GLOBAL_CASE_DATA
    GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_TEST

    if data_mode == DataMode.API_TEST:
        split_name = "API Test (SL)"
    elif data_mode == DataMode.GENERALIZATION:
        split_name = "Generalization Test (SL)"
    else:
        split_name = "Test (SL)"

    sl_test_metrics = evaluate_model(
        model, X_test, test_idx, test_raw_data,
        test_params,
        scalers, device_obj, split_name, verbose=True
    )

    GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_BACKUP

    # SL inference speed test
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(100):
            t_start = time.perf_counter()
            _ = model(X_test[:1].to(device_obj))
            if device_obj.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t_start)

    sl_latency_ms = np.mean(times) * 1000

    # Print SL results
    print(f"\n{'=' * 70}")
    print(f"SL Results Summary")
    print(f"{'=' * 70}")
    print(f"\nData Mode: {data_mode}")
    print(f"Test Case: {case_name}")

    print(f"\n--- Accuracy Metrics ---")
    print(f"MAE_Pg (Non-Slack): {sl_test_metrics['mae_pg_non_slack_percent']:.4f}%")
    print(f"MAE_Vm (Generator): {sl_test_metrics['mae_vm_percent']:.4f}%")
    print(f"MAE_Qg (All Gens):  {sl_test_metrics['mae_qg_percent']:.4f}%")
    print(f"MAE_Va (All Buses): {sl_test_metrics['mae_va_deg']:.4f} degrees")

    print(f"\n--- Violations (p.u.) ---")
    print(f"Pg_viol (Non-Slack): {sl_test_metrics['mean_pg_viol_non_slack_pu']:.6f} p.u.")
    print(f"Pg_viol (Slack):     {sl_test_metrics['mean_pg_viol_slack_pu']:.6f} p.u.")
    print(f"Qg_viol (All Gens):  {sl_test_metrics['mean_max_qg_viol_pu']:.6f} p.u.")
    print(f"Vm_viol (All Buses): {sl_test_metrics['mean_max_vm_viol_pu']:.6f} p.u.")
    print(f"Branch_viol:         {sl_test_metrics['mean_max_branch_viol_pu']:.6f} p.u. (1.0 = 100% overload)")

    print(f"\n--- Cost Metrics ---")
    print(f"Cost Gap: {sl_test_metrics['cost_optimality_gap_percent']:.4f}%")

    print(f"\n--- Performance ---")
    print(f"Inference Time: {sl_latency_ms:.4f} ms/sample")
    print(f"Training Time:  {sl_train_time:.2f} s")
    print(f"Convergence Rate: {sl_test_metrics['convergence_rate_percent']:.2f}%")
    print(f"{'=' * 70}")

    # ========================================================================
    # 7. Phase 2: Reinforcement Learning (optional)
    # ========================================================================
    rl_results = None

    if enable_rl and HAS_SB3:
        try:
            print(f"\n[Step 5] RL fine-tuning...")
            rl_agent, rl_train_time = train_rl_phase(
                X_train, model, params, scalers, raw_data, train_idx,
                rl_steps, rl_learning_rate, adjust_vm=adjust_vm
            )

            print(f"\n{'=' * 70}")
            print(f"RL Phase - Test Set Evaluation")
            print(f"{'=' * 70}")

            GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_TEST

            if data_mode == DataMode.API_TEST:
                split_name = "API Test (RL)"
            elif data_mode == DataMode.GENERALIZATION:
                split_name = "Generalization Test (RL)"
            else:
                split_name = "Test (RL)"

            rl_test_metrics = evaluate_rl_model(
                rl_agent, X_test, test_idx, model,
                test_params,
                scalers, test_raw_data, device_obj,
                adjust_vm=adjust_vm, split_name=split_name, verbose=True
            )

            GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_BACKUP

            # RL inference speed test
            times = []
            for _ in range(100):
                X_single = X_test[:1].to(device_obj)
                with torch.no_grad():
                    sl_pred = model(X_single)
                if adjust_vm:
                    obs = torch.cat([X_single, sl_pred[:, :n_gen_non_slack], sl_pred[:, n_gen_non_slack:]], dim=1)
                else:
                    obs = torch.cat([X_single, sl_pred[:, :n_gen_non_slack]], dim=1)
                obs_np = obs.cpu().numpy().astype(np.float32).flatten()

                t_start = time.perf_counter()
                _, _ = rl_agent.predict(obs_np, deterministic=True)
                times.append(time.perf_counter() - t_start)

            rl_latency_ms = np.mean(times) * 1000

            # Print RL results
            print(f"\n{'=' * 70}")
            print(f"RL Results Summary")
            print(f"{'=' * 70}")
            print(f"\nData Mode: {data_mode}")
            print(f"Test Case: {case_name}")

            print(f"\n--- Accuracy Metrics ---")
            print(f"MAE_Pg (Non-Slack): {rl_test_metrics['mae_pg_non_slack_percent']:.4f}%")
            print(f"MAE_Vm (Generator): {rl_test_metrics['mae_vm_percent']:.4f}%")
            print(f"MAE_Qg (All Gens):  {rl_test_metrics['mae_qg_percent']:.4f}%")
            print(f"MAE_Va (All Buses): {rl_test_metrics['mae_va_deg']:.4f} degrees")

            print(f"\n--- Violations (p.u.) ---")
            print(f"Pg_viol (Non-Slack): {rl_test_metrics['mean_pg_viol_non_slack_pu']:.6f} p.u.")
            print(f"Pg_viol (Slack):     {rl_test_metrics['mean_pg_viol_slack_pu']:.6f} p.u.")
            print(f"Qg_viol (All Gens):  {rl_test_metrics['mean_max_qg_viol_pu']:.6f} p.u.")
            print(f"Vm_viol (All Buses): {rl_test_metrics['mean_max_vm_viol_pu']:.6f} p.u.")
            print(f"Branch_viol:         {rl_test_metrics['mean_max_branch_viol_pu']:.6f} p.u. (1.0 = 100% overload)")

            print(f"\n--- Cost Metrics ---")
            print(f"Cost Gap: {rl_test_metrics['cost_optimality_gap_percent']:.4f}%")

            print(f"\n--- Performance ---")
            print(f"Inference Time: {rl_latency_ms:.4f} ms/sample")
            print(f"Training Time:  {rl_train_time:.2f} s")
            print(f"Convergence Rate: {rl_test_metrics['convergence_rate_percent']:.2f}%")
            print(f"{'=' * 70}")

            rl_results = {
                'enabled': True,
                'metrics': rl_test_metrics,
                'training_time_s': rl_train_time,
                'inference_time_ms': rl_latency_ms,
            }

        except Exception as e:
            print(f"\n❌ RL phase failed: {e}")
            import traceback
            traceback.print_exc()
            rl_results = {'enabled': False, 'error': str(e)}

    # No file saving - removed all JSON/CSV/model saving

    print("\n✓ Experiment completed successfully!")

    return {
        'sl_metrics': sl_test_metrics,
        'sl_train_time': sl_train_time,
        'sl_inference_time': sl_latency_ms,
        'rl_results': rl_results
    }


# =====================================================================
# Main Function
# =====================================================================
if __name__ == "__main__":
    # =========================================================================
    # Read configuration from acopf_config.py and run experiment
    # =========================================================================

    # SL+RL specific parameters
    ENABLE_RL = True  # Whether to enable RL fine-tuning
    RL_STEPS = 10000  # RL training steps
    RL_LEARNING_RATE = 3e-4  # RL learning rate
    ADJUST_VM = True  # Whether to adjust Vm (False means only adjust Pg)

    # Print configuration
    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)

    # Get all paths
    paths = acopf_config.get_all_paths()

    # Get all training parameters
    params = acopf_config.get_all_params()

    # Add SL+RL specific parameters
    params['enable_rl'] = ENABLE_RL
    params['rl_steps'] = RL_STEPS
    params['rl_learning_rate'] = RL_LEARNING_RATE
    params['adjust_vm'] = ADJUST_VM

    print(f"\n[SL+RL Specific Configuration]")
    print(f"  Enable RL: {ENABLE_RL}")
    if ENABLE_RL:
        print(f"  RL Steps: {RL_STEPS}")
        print(f"  RL Learning Rate: {RL_LEARNING_RATE}")
        print(f"  Adjust Vm: {ADJUST_VM}")
    print("=" * 70)

    # Execute experiment
    results = run_sl_rl_experiment(**paths, **params)

    print("\n✓ All experiments completed successfully!")