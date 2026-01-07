"""
ACOPF PINN Layer - PyTorch Version
===================================
Core Functions:
1. Encapsulates DenseCoreNetwork
2. Computes KKT error (physics constraints)
3. Handles sparse bus numbering
4. Adapts to non-slack generator Pg and generator bus Vm

KKT Condition Checks:
1. Power flow equations (nonlinear)
2. Generator constraints (pg, qg)
3. Voltage constraints (vm)
4. Complementary slackness conditions
5. Dual feasibility
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from acopf_DenseCoreNetwork import DenseCoreNetwork
except ImportError:
    print("Warning: Unable to import DenseCoreNetwork, ensure file is in same directory")


class PinnLayer(nn.Module):
    """
    ACOPF PINN Layer - PyTorch Version
    Adapted for non-slack Pg and generator Vm prediction
    """

    def __init__(self, simulation_parameters, device='cuda'):
        super(PinnLayer, self).__init__()

        self.device = device

        # ============ Network topology parameters ============
        self.n_bus = simulation_parameters['general']['n_buses']
        self.n_gen = simulation_parameters['general']['n_gen']
        self.n_gen_non_slack = simulation_parameters['general']['n_gen_non_slack']
        self.n_branches = simulation_parameters['general']['n_branches']
        self.BASE_MVA = simulation_parameters['general']['BASE_MVA']

        # Sparse bus mapping
        self.bus_ids = simulation_parameters['general']['bus_ids']
        self.bus_id_to_idx = simulation_parameters['general']['bus_id_to_idx']

        # Load bus mapping (for pd, qd injection)
        load_bus_ids = simulation_parameters['general']['load_bus_ids']
        self.load_to_bus_idx = torch.tensor(
            [self.bus_id_to_idx[int(bid)] for bid in load_bus_ids],
            dtype=torch.long, device=device
        )
        self.n_loads = len(load_bus_ids)

        # Generator to bus mapping
        gen_bus_ids = simulation_parameters['general']['gen_bus_ids']
        self.gen_to_bus_idx = torch.tensor(
            [self.bus_id_to_idx[int(bid)] for bid in gen_bus_ids],
            dtype=torch.long, device=device
        )

        # Non-slack generator indices
        self.non_slack_gen_idx = torch.tensor(
            simulation_parameters['general']['non_slack_gen_idx'],
            dtype=torch.long, device=device
        )

        # ============ Network structure parameters ============
        neurons_V = simulation_parameters['training']['neurons_in_hidden_layers_V']
        neurons_G = simulation_parameters['training']['neurons_in_hidden_layers_G']
        neurons_Lg = simulation_parameters['training']['neurons_in_hidden_layers_Lg']

        self.core_network = DenseCoreNetwork(
            n_gen=self.n_gen,
            n_gen_non_slack=self.n_gen_non_slack,
            neurons_in_hidden_layers_V=neurons_V,
            neurons_in_hidden_layers_G=neurons_G,
            neurons_in_hidden_layers_Lg=neurons_Lg
        ).to(device)

        # ============ Constraint parameters ============
        # Generator constraints (only non-slack for Pg)
        self.pg_min = torch.tensor(
            simulation_parameters['generator']['pg_min'],
            dtype=torch.float32, device=device
        )
        self.pg_max = torch.tensor(
            simulation_parameters['generator']['pg_max'],
            dtype=torch.float32, device=device
        )
        self.qg_min = torch.tensor(
            simulation_parameters['generator']['qg_min'],
            dtype=torch.float32, device=device
        )
        self.qg_max = torch.tensor(
            simulation_parameters['generator']['qg_max'],
            dtype=torch.float32, device=device
        )

        # Extract non-slack generator constraints
        self.pg_min_non_slack = self.pg_min[:, self.non_slack_gen_idx]
        self.pg_max_non_slack = self.pg_max[:, self.non_slack_gen_idx]

        # Generation cost
        self.cost_c1 = torch.tensor(
            simulation_parameters['generator']['cost_c1'],
            dtype=torch.float32, device=device
        )

        # Voltage constraints (only generator buses)
        vm_min_all = torch.tensor(
            simulation_parameters['bus']['vm_min'],
            dtype=torch.float32, device=device
        )
        vm_max_all = torch.tensor(
            simulation_parameters['bus']['vm_max'],
            dtype=torch.float32, device=device
        )

        # Extract generator bus voltage limits
        self.vm_min_gen = vm_min_all[self.gen_to_bus_idx]
        self.vm_max_gen = vm_max_all[self.gen_to_bus_idx]

        # ============ Admittance matrix ============
        if 'Y_real' in simulation_parameters:
            self.Y_real = torch.tensor(
                simulation_parameters['Y_real'],
                dtype=torch.float32, device=device
            )
            self.Y_imag = torch.tensor(
                simulation_parameters['Y_imag'],
                dtype=torch.float32, device=device
            )
        else:
            # If not pre-computed, will error on call
            self.Y_real = None
            self.Y_imag = None

    def reconstruct_full_pg(self, pg_non_slack):
        """
        Reconstruct full Pg array from non-slack generator Pg
        Slack positions filled with 0 (will be adjusted by power flow)

        Args:
            pg_non_slack: (batch_size, n_gen_non_slack) or (n_gen_non_slack,)

        Returns:
            pg_full: (batch_size, n_gen) or (n_gen,)
        """
        if pg_non_slack.ndim == 1:
            # Single sample
            pg_full = torch.zeros(self.n_gen, dtype=pg_non_slack.dtype, device=self.device)
            pg_full[self.non_slack_gen_idx] = pg_non_slack
        else:
            # Batch samples
            batch_size = pg_non_slack.shape[0]
            pg_full = torch.zeros(batch_size, self.n_gen, dtype=pg_non_slack.dtype, device=self.device)
            pg_full[:, self.non_slack_gen_idx] = pg_non_slack

        return pg_full

    def compute_power_flow_error(self, vm_gen, va, pg_full, qg, pd, qd):
        """
        Compute power flow equation error

        Power flow equations:
            P_i = V_i * Σ(V_j * |Y_ij| * cos(θ_i - θ_j - φ_ij))
            Q_i = V_i * Σ(V_j * |Y_ij| * sin(θ_i - θ_j - φ_ij))

        Using matrix form:
            S = V * conj(Y @ V)

        Args:
            vm_gen: (batch, n_gen) - Generator bus voltages
            va: (batch, n_bus) - All bus voltage angles
            pg_full: (batch, n_gen) - All generator active power (including slack)
            qg: (batch, n_gen) - All generator reactive power
            pd: (batch, n_loads) - Load active power (only load buses)
            qd: (batch, n_loads) - Load reactive power (only load buses)
        """
        if self.Y_real is None or self.Y_imag is None:
            raise ValueError("Admittance matrix not initialized! Provide Y_real and Y_imag in simulation_parameters")

        batch_size = vm_gen.shape[0]

        # ============ Construct voltage vector for all buses ============
        # Expand generator Vm to all buses (non-generator buses use nominal 1.0 p.u.)
        vm_all = torch.ones(batch_size, self.n_bus, device=self.device)
        vm_all[:, self.gen_to_bus_idx] = vm_gen

        # ============ Build complex voltage vector ============
        # V = vm * exp(j * va) = vm * (cos(va) + j * sin(va))
        v_real = vm_all * torch.cos(va)  # (batch, n_bus)
        v_imag = vm_all * torch.sin(va)  # (batch, n_bus)

        # ============ Compute Y @ V (complex multiplication) ============
        # (Y_real + j*Y_imag) @ (V_real + j*V_imag)
        # = (Y_real @ V_real - Y_imag @ V_imag) + j*(Y_real @ V_imag + Y_imag @ V_real)

        yv_real = torch.matmul(v_real, self.Y_real.t()) - torch.matmul(v_imag, self.Y_imag.t())
        yv_imag = torch.matmul(v_real, self.Y_imag.t()) + torch.matmul(v_imag, self.Y_real.t())

        # ============ Compute V * conj(Y @ V) ============
        # V * conj(YV) = (v_real + j*v_imag) * (yv_real - j*yv_imag)
        # = v_real*yv_real + v_imag*yv_imag + j*(v_imag*yv_real - v_real*yv_imag)

        S_real = v_real * yv_real + v_imag * yv_imag  # Active power P
        S_imag = v_imag * yv_real - v_real * yv_imag  # Reactive power Q

        # ============ Construct injection power ============
        # P_inj = Pg - Pd (aggregated by bus)
        # Q_inj = Qg - Qd (aggregated by bus)

        P_inj = torch.zeros(batch_size, self.n_bus, device=self.device)
        Q_inj = torch.zeros(batch_size, self.n_bus, device=self.device)

        # Load injection (negative) - map from load buses to all buses
        # pd and qd are (batch, n_loads), need to scatter to (batch, n_bus)
        P_inj.scatter_add_(1, self.load_to_bus_idx.unsqueeze(0).expand(batch_size, -1), -pd)
        Q_inj.scatter_add_(1, self.load_to_bus_idx.unsqueeze(0).expand(batch_size, -1), -qd)

        # Generator injection (positive), map to corresponding buses
        # Use scatter_add to aggregate to buses
        P_inj.scatter_add_(1, self.gen_to_bus_idx.unsqueeze(0).expand(batch_size, -1), pg_full)
        Q_inj.scatter_add_(1, self.gen_to_bus_idx.unsqueeze(0).expand(batch_size, -1), qg)

        # ============ Compute power balance error ============
        P_error = torch.abs(S_real - P_inj)  # (batch, n_bus)
        Q_error = torch.abs(S_imag - Q_inj)  # (batch, n_bus)

        # Average error
        power_flow_error = torch.mean(P_error, dim=1) + torch.mean(Q_error, dim=1)

        return power_flow_error

    def get_kkt_error(self, vm_gen, pg_non_slack, qg, va, pd, qd,
                      mu_pg_min, mu_pg_max, mu_vm_min, mu_vm_max):
        """
        Compute KKT error (physics constraint loss)

        Args:
            vm_gen: Generator bus voltage (predicted)
            pg_non_slack: Non-slack generator active power (predicted)
            qg, va: Auxiliary variables (from dataset during training)
            pd, qd: Input loads
            mu_*: Dual variables

        Returns:
            kkt_error: (batch_size,) KKT error
        """
        batch_size = vm_gen.shape[0]

        # Reconstruct full Pg array for power flow calculation
        pg_full = self.reconstruct_full_pg(pg_non_slack)

        # ============ 1. Power flow equation error ============
        kkt_error = self.compute_power_flow_error(vm_gen, va, pg_full, qg, pd, qd)

        # ============ 2. Generator constraint violations ============
        # Pg constraints (only non-slack generators)
        pg_viol_upper = torch.relu(pg_non_slack - self.pg_max_non_slack)
        pg_viol_lower = torch.relu(self.pg_min_non_slack - pg_non_slack)
        kkt_error = kkt_error + torch.sum(pg_viol_upper + pg_viol_lower, dim=1) / self.n_gen_non_slack

        # Qg constraints (all generators, from auxiliary data)
        qg_viol_upper = torch.relu(qg - self.qg_max)
        qg_viol_lower = torch.relu(self.qg_min - qg)
        kkt_error = kkt_error + torch.sum(qg_viol_upper + qg_viol_lower, dim=1) / self.n_gen

        # ============ 3. Voltage constraint violations ============
        # Vm constraints (only generator buses)
        vm_viol_upper = torch.relu(vm_gen - self.vm_max_gen)
        vm_viol_lower = torch.relu(self.vm_min_gen - vm_gen)
        kkt_error = kkt_error + torch.sum(vm_viol_upper + vm_viol_lower, dim=1) / self.n_gen

        # ============ 4. Complementary slackness conditions ============
        # mu_pg * (pg - pg_max) ≈ 0
        # mu_pg * (pg_min - pg) ≈ 0
        comp_pg_upper = torch.abs(mu_pg_max * (pg_non_slack - self.pg_max_non_slack))
        comp_pg_lower = torch.abs(mu_pg_min * (self.pg_min_non_slack - pg_non_slack))
        kkt_error = kkt_error + torch.sum(comp_pg_upper + comp_pg_lower, dim=1) / self.n_gen_non_slack

        comp_vm_upper = torch.abs(mu_vm_max * (vm_gen - self.vm_max_gen))
        comp_vm_lower = torch.abs(mu_vm_min * (self.vm_min_gen - vm_gen))
        kkt_error = kkt_error + torch.sum(comp_vm_upper + comp_vm_lower, dim=1) / self.n_gen

        # ============ 5. Dual feasibility ============
        # Dual variables should be non-negative
        kkt_error = kkt_error + torch.sum(torch.relu(-mu_pg_min), dim=1) / self.n_gen_non_slack
        kkt_error = kkt_error + torch.sum(torch.relu(-mu_pg_max), dim=1) / self.n_gen_non_slack
        kkt_error = kkt_error + torch.sum(torch.relu(-mu_vm_min), dim=1) / self.n_gen
        kkt_error = kkt_error + torch.sum(torch.relu(-mu_vm_max), dim=1) / self.n_gen

        return kkt_error

    def forward(self, inputs, qg_aux=None, va_aux=None):
        """
        Forward propagation

        Args:
            inputs: (batch, 2*n_loads) - [pd, qd] normalized input
            qg_aux: (batch, n_gen) - qg provided from dataset during training (normalized)
            va_aux: (batch, n_bus) - va provided from dataset during training (normalized)

        Returns:
            vm_gen, pg_non_slack: Predicted variables
            mu_pg_min, mu_pg_max, mu_vm_min, mu_vm_max: Dual variables
            kkt_error: KKT error
        """
        # Call core network
        vm_gen, pg_non_slack, mu_pg_min, mu_pg_max, mu_vm_min, mu_vm_max = self.core_network(inputs)

        # Separate pd and qd from inputs
        # inputs contains [pd, qd], each of size n_loads
        pd = inputs[:, :self.n_loads]
        qd = inputs[:, self.n_loads:]

        # Note: For now assume inputs are already in p.u. (normalized by scalers)
        # In actual training, scalers handle normalization/denormalization

        # Compute KKT error
        # Note: qg and va are provided externally during training
        if qg_aux is not None and va_aux is not None:
            kkt_error = self.get_kkt_error(
                vm_gen=vm_gen, pg_non_slack=pg_non_slack, qg=qg_aux, va=va_aux,
                pd=pd, qd=qd,
                mu_pg_min=mu_pg_min, mu_pg_max=mu_pg_max,
                mu_vm_min=mu_vm_min, mu_vm_max=mu_vm_max
            )
        else:
            # Inference mode, don't compute KKT error
            kkt_error = torch.zeros(inputs.shape[0], device=self.device)

        return vm_gen, pg_non_slack, mu_pg_min, mu_pg_max, mu_vm_min, mu_vm_max, kkt_error