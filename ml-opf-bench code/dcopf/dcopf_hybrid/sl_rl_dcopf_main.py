# -*- coding: utf-8 -*-
"""
DeepOPF + RL for DCOPF - Main Experiment Script
Two-Stage Training: Physics-Informed Neural Network + Reinforcement Learning

Version: v3.0 - Aligned with DNN Version

Features:
- Stage 1: DeepOPF (PINN) - Predict non-Slack generators with physics constraints
- Stage 2: RL (PPO) - Fine-tune with delta learning
- Auto-identify and handle Slack Bus
- Support API_TEST mode with dual constraint parameters

Evaluation Metrics (6 metrics aligned with DNN):
- MAE Pg (%) - Non-Slack, Slack
- Pg Violation (p.u., Mean of Max) - Non-Slack, Slack
- Branch Violation (p.u., Mean of Max)
- Cost Gap (%)
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

# Gymnasium and SB3
import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3 import PPO

    HAS_SB3 = True
except ImportError:
    print("WARNING: stable_baselines3 not installed, RL stage will be skipped")
    HAS_SB3 = False

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


# =====================================================================
# Stage 1: DeepOPF Components
# =====================================================================

def compute_dcopf_penalty(y_pred_pg_non_slack, x_pd, params):
    """
    Calculate DCOPF constraint violation penalty

    Parameters:
    -----------
    y_pred_pg_non_slack : np.ndarray
        Predicted non-Slack generator outputs, shape (n_samples, n_g_non_slack), p.u.
    x_pd : np.ndarray
        Load demand, shape (n_samples, n_buses), p.u.
    params : dict
        System parameters

    Returns:
    --------
    penalties : np.ndarray
        Total penalty for each sample, shape (n_samples,), p.u.
    """
    n_samples = y_pred_pg_non_slack.shape[0]

    # Reconstruct full Pg vector
    pd_total = x_pd.sum(axis=1)
    y_pred_pg_all = reconstruct_full_pg(
        pg_non_slack=y_pred_pg_non_slack,
        pd_total=pd_total,
        params=params
    )

    # Calculate constraint violations
    gen_up_viol, gen_lo_viol, line_viol, balance_err = feasibility(
        y_pred_pg=y_pred_pg_all,
        x_pd=x_pd,
        params=params
    )

    # Calculate total penalty
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


class Penalty_DCOPF(Function):
    """DCOPF physics constraint penalty layer"""

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
        y_pred_pg_non_slack_plus = scalers['y_pg_non_slack'].inverse_transform(nn_output_plus_h)
        y_pred_pg_non_slack_minus = scalers['y_pg_non_slack'].inverse_transform(nn_output_minus_h)

        penalty_plus = compute_dcopf_penalty(y_pred_pg_non_slack_plus, x_raw, params)
        penalty_minus = compute_dcopf_penalty(y_pred_pg_non_slack_minus, x_raw, params)

        gradient_estimate = np.zeros((batch_size, output_dim), dtype='float32')

        for i in range(batch_size):
            directional_derivative = (penalty_plus[i] - penalty_minus[i]) / (2 * h)
            gradient_estimate[i] = directional_derivative * vector_h[i] * output_dim

        final_gradient = gradient_estimate * (1.0 / batch_size)

        return torch.from_numpy(final_gradient).to(nn_output_scaled.device) * grad_output, None


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


def load_and_prepare_deepopf_data(file_path, params, column_names):
    """Load and prepare DeepOPF training data"""
    import pandas as pd

    full_df = pd.read_csv(file_path)
    n_samples = len(full_df)
    n_buses = params['general']['n_buses']

    load_prefix = column_names['load_prefix']
    load_cols = [col for col in full_df.columns if col.startswith(load_prefix)]

    x_data_raw = np.zeros((n_samples, n_buses), dtype='float32')

    for col in load_cols:
        bus_id = int(col[len(load_prefix):])
        if bus_id <= n_buses:
            x_data_raw[:, bus_id - 1] = full_df[col].values

    pg_cols = [col for col in full_df.columns if col.startswith(column_names['gen_prefix'])]
    y_pg_raw_all = full_df[pg_cols].values.astype('float32')

    non_slack_indices = params['general']['non_slack_gen_indices']
    y_pg_raw_non_slack = y_pg_raw_all[:, non_slack_indices]

    return x_data_raw, y_pg_raw_non_slack, y_pg_raw_all


# =====================================================================
# Stage 2: RL Components - Gym Environment
# =====================================================================

class DCOPFDeltaEnv(gym.Env):
    """DCOPF Delta Learning Environment"""

    metadata = {'render_modes': ['human']}

    def __init__(self, X_data, pretrained_model, params, scalers,
                 raw_data, indices, delta_scale=0.1):
        super().__init__()

        self.X_data = X_data
        self.pretrained_model = pretrained_model.eval()
        self.params = params
        self.scalers = scalers
        self.raw_data = raw_data
        self.indices = indices
        self.delta_scale = delta_scale
        self.device = next(pretrained_model.parameters()).device

        n_gen_non_slack = params['general']['n_g_non_slack']
        n_buses = params['general']['n_buses']

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_gen_non_slack,), dtype=np.float32
        )

        obs_dim = n_buses + n_gen_non_slack
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        self.current_X = None
        self.current_idx = None
        self.sl_pred_pg = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        idx = np.random.randint(0, len(self.indices))
        self.current_idx = self.indices[idx]
        self.current_X = self.X_data[idx:idx + 1].to(self.device)

        with torch.no_grad():
            sl_pred, _ = self.pretrained_model(self.current_X)
            self.sl_pred_pg = sl_pred.cpu().numpy()

        obs = np.hstack([
            self.current_X.cpu().numpy().flatten(),
            self.sl_pred_pg.flatten()
        ])

        return obs.astype(np.float32), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        non_slack_indices = self.params['general']['non_slack_gen_indices']
        n_buses = self.params['general']['n_buses']

        # Calculate adjusted Pg_non_slack (p.u.)
        pg_sl_non_slack = self.scalers['y_pg_non_slack'].inverse_transform(self.sl_pred_pg)

        pg_min_all = self.params['constraints']['Pg_min'].flatten()
        pg_max_all = self.params['constraints']['Pg_max'].flatten()

        pg_min = pg_min_all[non_slack_indices]
        pg_max = pg_max_all[non_slack_indices]
        pg_range = pg_max - pg_min

        delta_pg_non_slack = action * self.delta_scale * pg_range
        pg_new_non_slack = np.clip(
            pg_sl_non_slack + delta_pg_non_slack,
            pg_min,
            pg_max
        )

        # Reconstruct full Pg
        x_raw = self.scalers['x'].inverse_transform(self.current_X.cpu().numpy())
        pd_total = x_raw.sum(axis=1)

        pg_new_all = reconstruct_full_pg(
            pg_non_slack=pg_new_non_slack,
            pd_total=pd_total,
            params=self.params
        )

        # Calculate violations (p.u.)
        gen_up_viol, gen_lo_viol, line_viol, balance_err = feasibility(
            y_pred_pg=pg_new_all,
            x_pd=x_raw,
            params=self.params
        )

        # Calculate reward components
        pg_true_all = self.raw_data['y_pg_all'][self.current_idx:self.current_idx + 1]
        pg_true_non_slack = pg_true_all[:, non_slack_indices]

        pg_new_non_slack_scaled = self.scalers['y_pg_non_slack'].transform(pg_new_non_slack)
        pg_true_non_slack_scaled = self.scalers['y_pg_non_slack'].transform(pg_true_non_slack)

        mse_new_non_slack = np.mean((pg_new_non_slack_scaled - pg_true_non_slack_scaled) ** 2)
        mse_sl_non_slack = np.mean((self.sl_pred_pg - pg_true_non_slack_scaled) ** 2)

        mse_improvement = (mse_sl_non_slack - mse_new_non_slack) / (mse_sl_non_slack + 1e-6)

        # Violation penalty (p.u. normalized to 0-2 range)
        viol_dict = compute_detailed_pg_violations_pu(
            gen_up_viol=gen_up_viol,
            gen_lo_viol=gen_lo_viol,
            params=self.params
        )

        viol_branch_pu = compute_branch_violation_pu(
            line_viol=line_viol,
            Pl_max=self.params['constraints']['Pl_max']
        )

        # Normalize violations
        pg_viol_norm = min(viol_dict['viol_non_slack'] / 0.5, 2.0)  # 0.5 p.u. → normalized
        branch_viol_norm = min(viol_branch_pu / 0.1, 2.0)  # 0.1 p.u. → normalized
        balance_norm = min(balance_err[0] / 0.01, 2.0)  # 0.01 p.u. → normalized

        violation_penalty = pg_viol_norm + branch_viol_norm + balance_norm

        # Feasibility check
        is_feasible = (
                np.max(gen_up_viol) < 1e-4 and
                np.max(gen_lo_viol) < 1e-4 and
                np.max(line_viol) < 1e-4 and
                balance_err[0] < 1e-4
        )
        convergence_bonus = 20.0 if is_feasible else -50.0

        # Total reward
        reward = (
                100.0 * mse_improvement +
                convergence_bonus -
                50.0 * violation_penalty
        )

        # New observation
        pg_new_non_slack_scaled = self.scalers['y_pg_non_slack'].transform(pg_new_non_slack)
        obs = np.hstack([
            self.current_X.cpu().numpy().flatten(),
            pg_new_non_slack_scaled.flatten()
        ])

        done = True

        info = {
            'mse_new_non_slack': float(mse_new_non_slack),
            'mse_sl_non_slack': float(mse_sl_non_slack),
            'mse_improvement': float(mse_improvement * 100),
            'pg_viol_pu': float(viol_dict['viol_non_slack']),
            'branch_viol_pu': float(viol_branch_pu),
            'balance_err_pu': float(balance_err[0]),
            'is_feasible': is_feasible,
        }

        return obs.astype(np.float32), float(reward), done, False, info


# =====================================================================
# Stage 2: RL Training
# =====================================================================

def train_rl_dcopf_phase(
        X_train, pretrained_model, params, scalers, raw_data, train_idx,
        n_steps=5000, learning_rate=3e-4, delta_scale=0.1
):
    """RL fine-tuning stage"""

    rl_env = DCOPFDeltaEnv(
        X_train, pretrained_model, params, scalers,
        raw_data, train_idx, delta_scale=delta_scale
    )

    policy_net_arch = dict(pi=[256, 256], vf=[256, 256])

    agent = PPO(
        "MlpPolicy",
        rl_env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device='cpu',
        policy_kwargs=dict(net_arch=policy_net_arch)
    )

    t0 = time.time()
    agent.learn(total_timesteps=n_steps)
    rl_train_time = time.time() - t0

    return agent, rl_train_time


# =====================================================================
# Evaluation Functions
# =====================================================================

def evaluate_deepopf_model(
        model, X, indices, raw_data, params, scalers, device,
        test_data_external=None, test_params=None
):
    """Evaluate DeepOPF model (aligned with DNN)"""

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

        pd_total = x_raw_eval.sum(axis=1)
        y_pred_pg_all = reconstruct_full_pg(
            pg_non_slack=y_pred_non_slack,
            pd_total=pd_total,
            params=eval_params
        )

        # Calculate MAE
        mae_dict = compute_detailed_mae(
            y_true_all=y_true_pg_all,
            y_pred_non_slack=y_pred_non_slack,
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


def evaluate_rl_dcopf(
        agent, X_test, test_idx, pretrained_model, params, scalers,
        raw_data, device, delta_scale=0.1, test_params=None
):
    """Evaluate RL fine-tuned model (aligned with DNN)"""

    eval_params = test_params if test_params is not None else params

    non_slack_indices = params['general']['non_slack_gen_indices']
    n_samples = len(X_test)

    y_pred_pg_non_slack_list = []
    y_pred_pg_all_list = []

    pretrained_model.eval()

    for i in range(n_samples):
        X_single = X_test[i:i + 1].to(device)

        with torch.no_grad():
            sl_pred, _ = pretrained_model(X_single)
            pg_sl_non_slack_scaled = sl_pred.cpu().numpy()

        obs = np.hstack([
            X_single.cpu().numpy().flatten(),
            pg_sl_non_slack_scaled.flatten()
        ])
        obs_np = obs.astype(np.float32)

        action, _ = agent.predict(obs_np, deterministic=True)
        action = np.clip(action, -1.0, 1.0)

        pg_sl_non_slack = scalers['y_pg_non_slack'].inverse_transform(pg_sl_non_slack_scaled)

        pg_min_all = eval_params['constraints']['Pg_min'].flatten()
        pg_max_all = eval_params['constraints']['Pg_max'].flatten()
        pg_min = pg_min_all[non_slack_indices]
        pg_max = pg_max_all[non_slack_indices]
        pg_range = pg_max - pg_min

        delta_pg = action * delta_scale * pg_range
        pg_new_non_slack = np.clip(pg_sl_non_slack + delta_pg, pg_min, pg_max)

        x_raw_single = scalers['x'].inverse_transform(X_single.cpu().numpy())
        pd_total = x_raw_single.sum(axis=1)

        pg_new_all = reconstruct_full_pg(
            pg_non_slack=pg_new_non_slack,
            pd_total=pd_total,
            params=eval_params
        )

        y_pred_pg_non_slack_list.append(pg_new_non_slack.flatten())
        y_pred_pg_all_list.append(pg_new_all.flatten())

    y_pred_pg_non_slack = np.array(y_pred_pg_non_slack_list)
    y_pred_pg_all = np.array(y_pred_pg_all_list)

    y_true_pg_all = raw_data['y_pg_all'][test_idx]
    x_raw_data = scalers['x'].inverse_transform(X_test.cpu().numpy())

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
        x_pd=x_raw_data,
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

def run_deepopf_rl_experiment(
        case_name,
        params_path,
        dataset_path,
        column_names,
        split_mode=DataSplitMode.RANDOM_SPLIT,
        test_data_path=None,
        test_params_path=None,
        n_test_samples=1000,
        n_train_use=10000,
        sl_hidden_sizes=[256, 256],
        sl_epochs=100,
        sl_batch_size=128,
        sl_learning_rate=1e-3,
        penalty_weight=0.005,
        enable_rl=True,
        rl_steps=5000,
        rl_learning_rate=3e-4,
        delta_scale=0.1,
        seed=42,
        device='cuda'
):
    """Complete DeepOPF + RL experiment (aligned with DNN version)"""

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

    scalers = {
        'x': x_scaler,
        'y_pg_non_slack': y_pg_non_slack_scaler
    }
    GLOBAL_SCALERS = scalers

    x_train_scaled = x_scaler.transform(x_data_raw[train_idx])
    y_train_scaled = y_pg_non_slack_scaler.transform(y_pg_raw_non_slack[train_idx])
    x_val_scaled = x_scaler.transform(x_data_raw[val_idx])
    y_val_scaled = y_pg_non_slack_scaler.transform(y_pg_raw_non_slack[val_idx])

    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        x_test_scaled = x_scaler.transform(x_test_external)
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
        batch_size=sl_batch_size,
        shuffle=True
    )

    # Build model
    model = PINN_DCOPF(
        input_dim=x_data_raw.shape[1],
        output_dim=n_gen_non_slack,
        hidden_sizes=sl_hidden_sizes
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=sl_learning_rate, betas=(0.9, 0.99))

    # Stage 1: DeepOPF Training
    training_start = time.time()

    for epoch in range(1, sl_epochs + 1):
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
        avg_mse = epoch_mse / len(X_train)
        avg_penalty = epoch_penalty / len(X_train)

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
        print(f"Epoch {epoch}/{sl_epochs} - train_loss: {avg_total:.6f} - val_loss: {val_loss:.6f}")

    sl_train_time = time.time() - training_start

    # Stage 1: DeepOPF Evaluation
    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        test_data_external_dict = {
            'x': x_test_external,
            'y_pg_all': y_test_external
        }
        sl_test_metrics = evaluate_deepopf_model(
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
        sl_test_metrics = evaluate_deepopf_model(
            model=model,
            X=X_test,
            indices=test_idx,
            raw_data=raw_data,
            params=params,
            scalers=scalers,
            device=device,
            test_params=test_params
        )

    # DeepOPF speed test
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
        for _ in range(10):
            _ = model(test_sample)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        for _ in range(100):
            t_start = time.time()
            _ = model(test_sample)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - t_start)

    sl_inference_time_ms = np.mean(times) * 1000

    # Print DeepOPF results
    print("\n" + "=" * 70)
    print("Test Set Results (DeepOPF - Stage 1)")
    print("=" * 70)
    print(f"\nNon-Slack Generators:")
    print(f"  MAE:        {sl_test_metrics['mae_pg_non_slack']:.4f}%")
    print(f"  Violation:  {sl_test_metrics['viol_pg_non_slack']:.4f} p.u.")
    print(f"\nSlack-Only Generators:")
    print(f"  MAE:        {sl_test_metrics['mae_pg_slack']:.4f}%")
    print(f"  Violation:  {sl_test_metrics['viol_pg_slack']:.4f} p.u.")
    print(f"\nBranch:")
    print(f"  Violation:  {sl_test_metrics['viol_branch']:.4f} p.u.")
    print(f"\nCost Gap:     {sl_test_metrics['cost_gap_percent']:.4f}%")
    print("\n" + "=" * 70 + "\n")

    # Stage 2: RL (optional)
    rl_train_time = 0.0
    rl_latency_ms = 0.0

    if enable_rl and HAS_SB3:
        try:
            rl_agent, rl_train_time = train_rl_dcopf_phase(
                X_train, model, params, scalers, raw_data, train_idx,
                n_steps=rl_steps,
                learning_rate=rl_learning_rate,
                delta_scale=delta_scale
            )

            # RL Evaluation
            if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
                X_test_rl = torch.tensor(x_test_scaled, dtype=torch.float32)
                rl_test_metrics = evaluate_rl_dcopf(
                    rl_agent, X_test_rl, np.arange(len(x_test_external)), model,
                    params, scalers, {'x': x_test_external, 'y_pg_all': y_test_external},
                    device, delta_scale, test_params
                )
            else:
                rl_test_metrics = evaluate_rl_dcopf(
                    rl_agent, X_test, test_idx, model,
                    params, scalers, raw_data, device,
                    delta_scale, test_params
                )

            # RL speed test
            times = []
            for _ in range(100):
                if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
                    X_single = torch.tensor(
                        x_scaler.transform(x_test_external[:1]),
                        dtype=torch.float32,
                        device=device
                    )
                else:
                    X_single = X_test[:1].to(device)

                with torch.no_grad():
                    sl_pred, _ = model(X_single)

                obs = np.hstack([
                    X_single.cpu().numpy().flatten(),
                    sl_pred.cpu().numpy().flatten()
                ])
                obs_np = obs.astype(np.float32)

                t_start = time.time()
                _, _ = rl_agent.predict(obs_np, deterministic=True)
                times.append(time.time() - t_start)

            rl_latency_ms = np.mean(times) * 1000

            # Print RL results
            print("\n" + "=" * 70)
            print("Test Set Results (RL - Stage 2)")
            print("=" * 70)
            print(f"\nNon-Slack Generators:")
            print(f"  MAE:        {rl_test_metrics['mae_pg_non_slack']:.4f}%")
            print(f"  Violation:  {rl_test_metrics['viol_pg_non_slack']:.4f} p.u.")
            print(f"\nSlack-Only Generators:")
            print(f"  MAE:        {rl_test_metrics['mae_pg_slack']:.4f}%")
            print(f"  Violation:  {rl_test_metrics['viol_pg_slack']:.4f} p.u.")
            print(f"\nBranch:")
            print(f"  Violation:  {rl_test_metrics['viol_branch']:.4f} p.u.")
            print(f"\nCost Gap:     {rl_test_metrics['cost_gap_percent']:.4f}%")
            print("\n" + "=" * 70 + "\n")

        except Exception as e:
            print(f"\nRL stage failed: {e}")

    # Print overall performance summary
    print("\n" + "=" * 70)
    print("Overall Performance Summary")
    print("=" * 70)
    print(f"\nTraining Time:")
    print(f"  Stage 1 (DeepOPF): {sl_train_time:.2f} s")
    if enable_rl and HAS_SB3 and rl_train_time > 0:
        print(f"  Stage 2 (RL):      {rl_train_time:.2f} s")
        print(f"  Total:             {sl_train_time + rl_train_time:.2f} s")
    else:
        print(f"  Total:             {sl_train_time:.2f} s")

    print(f"\nInference Time:")
    print(f"  Stage 1 (DeepOPF): {sl_inference_time_ms:.4f} ms")
    if enable_rl and HAS_SB3 and rl_latency_ms > 0:
        print(f"  Stage 2 (RL):      {rl_latency_ms:.4f} ms")
        print(f"  Total:             {sl_inference_time_ms + rl_latency_ms:.4f} ms")
    else:
        print(f"  Total:             {sl_inference_time_ms:.4f} ms")
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

    # --- 4. DeepOPF Hyperparameters ---
    SL_EPOCHS = 100
    SL_LEARNING_RATE = 1e-3
    SL_BATCH_SIZE = 128
    SL_HIDDEN_SIZES = [128, 128]
    PENALTY_WEIGHT = 0.005

    # --- 5. RL Hyperparameters ---
    ENABLE_RL = True
    RL_STEPS = 10000
    RL_LEARNING_RATE = 3e-4
    DELTA_SCALE = 0.1

    # --- 6. Other ---
    SEED = 42

    # --- 7. Path Configuration (Manual) ---
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
    run_deepopf_rl_experiment(
        case_name=CASE_NAME,
        params_path=params_path,
        dataset_path=train_data_path,
        column_names=COLUMN_NAMES,
        split_mode=SPLIT_MODE,
        test_data_path=test_data_path,
        test_params_path=test_params_path,
        n_test_samples=N_TEST_SAMPLES,
        n_train_use=N_TRAIN_USE,
        sl_hidden_sizes=SL_HIDDEN_SIZES,
        sl_epochs=SL_EPOCHS,
        sl_batch_size=SL_BATCH_SIZE,
        sl_learning_rate=SL_LEARNING_RATE,
        penalty_weight=PENALTY_WEIGHT,
        enable_rl=ENABLE_RL,
        rl_steps=RL_STEPS,
        rl_learning_rate=RL_LEARNING_RATE,
        delta_scale=DELTA_SCALE,
        seed=SEED,
        device=device_name,
    )