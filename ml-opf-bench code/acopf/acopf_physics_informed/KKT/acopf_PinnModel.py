"""
ACOPF PINN Model - PyTorch Version
===================================
Encapsulates complete PINN model, including:
1. PinnLayer
2. Loss function computation
3. Training optimizer

Adapted for non-slack Pg and generator Vm prediction
"""

import os
import sys
import torch
import torch.nn as nn

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from acopf_PinnLayer import PinnLayer
except ImportError:
    print("Warning: Unable to import PinnLayer, ensure file is in same directory")


class PinnModel(nn.Module):
    """
    ACOPF PINN Complete Model - PyTorch Version
    Adapted for non-slack Pg and generator Vm
    """

    def __init__(self, W_dual, W_PINN, simulation_parameters, learning_rate=0.001, device='cuda'):
        """
        Initialize PINN model

        Args:
            W_dual: Dual variable loss weight (recommended: 1e-3)
            W_PINN: Physics constraint loss weight (recommended: 0.05)
            simulation_parameters: System parameters dictionary
            learning_rate: Learning rate
            device: Computing device
        """
        super(PinnModel, self).__init__()

        self.device = device
        self.pinn_layer = PinnLayer(
            simulation_parameters=simulation_parameters,
            device=device
        )

        # ============ Loss weights ============
        # Format: [vm_gen, pg_non_slack, mu_pg_min, mu_pg_max, mu_vm_min, mu_vm_max, kkt_error]
        self.loss_weights = [
            1.0,  # vm_gen prediction loss
            1.0,  # pg_non_slack prediction loss
            W_dual,  # mu_pg_min loss
            W_dual,  # mu_pg_max loss
            W_dual,  # mu_vm_min loss
            W_dual,  # mu_vm_max loss
            W_PINN  # KKT physics loss
        ]

        print(f"Loss weights:")
        print(f"  vm_gen/pg_non_slack: {self.loss_weights[0]:.4f}")
        print(f"  Dual variables: {W_dual:.4f}")
        print(f"  Physics constraints: {W_PINN:.4f}")

        # ============ Optimizer ============
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # ============ Loss function ============
        self.criterion = nn.L1Loss()  # MAE

    def forward(self, inputs, qg_aux=None, va_aux=None):
        """
        Forward propagation

        Args:
            inputs: (batch, 2*n_loads) - [pd, qd]
            qg_aux: (batch, n_gen) - Auxiliary qg (used during training)
            va_aux: (batch, n_bus) - Auxiliary va (used during training)

        Returns:
            All outputs (including KKT error)
        """
        return self.pinn_layer(inputs, qg_aux=qg_aux, va_aux=va_aux)

    def compute_loss(self, outputs, targets):
        """
        Compute weighted loss

        Args:
            outputs: Model outputs (vm_gen, pg_non_slack, mu_pg_min, mu_pg_max, mu_vm_min, mu_vm_max, kkt_error)
            targets: Target values (vm_gen, pg_non_slack, mu_pg_min, mu_pg_max, mu_vm_min, mu_vm_max, physics_dummy)

        Returns:
            total_loss: Total loss
            losses: List of individual losses
        """
        total_loss = 0.0
        losses = []

        # First 6 items are supervised learning losses (vm_gen, pg_non_slack, dual variables)
        for i in range(6):
            loss = self.criterion(outputs[i], targets[i])
            weighted_loss = self.loss_weights[i] * loss
            total_loss += weighted_loss
            losses.append(loss.item())

        # 7th item is KKT physics loss (no need to compare with target, just minimize)
        kkt_error = outputs[6]
        physics_loss = torch.mean(kkt_error)
        weighted_physics = self.loss_weights[6] * physics_loss
        total_loss += weighted_physics
        losses.append(physics_loss.item())

        return total_loss, losses

    def predict(self, x, qg_aux=None, va_aux=None):
        """
        Prediction (inference mode)

        Returns:
            vm_gen, pg_non_slack: Predicted primary variables
        """
        self.eval()
        with torch.no_grad():
            vm_gen, pg_non_slack, _, _, _, _, _ = self.forward(x, qg_aux=qg_aux, va_aux=va_aux)
        return vm_gen, pg_non_slack

    def train_step(self, x_batch, y_batch, qg_batch, va_batch):
        """
        Single training step

        Args:
            x_batch: Input [pd, qd]
            y_batch: Target (vm_gen, pg_non_slack, mu_pg_min, mu_pg_max, mu_vm_min, mu_vm_max, dummy)
            qg_batch: Auxiliary qg
            va_batch: Auxiliary va

        Returns:
            loss: Total loss
            losses: Individual losses
        """
        self.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.forward(x_batch, qg_aux=qg_batch, va_aux=va_batch)

        # Compute loss
        loss, losses = self.compute_loss(outputs, y_batch)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item(), losses