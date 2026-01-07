"""
ACOPF Dense Core Network - PyTorch Version
==========================================
Network Structure:
1. V Network: Input → vm_gen (only generator bus voltage magnitudes)
2. G Network: Input → pg_non_slack (only non-slack generator active power)
3. Lg Network: Input → dual variables (mu_pg_min/max, mu_vm_min/max)

Key Modifications:
- V network outputs only generator bus Vm (not all buses)
- G network outputs only non-slack generator Pg (excludes slack)
- Dual variables adapted accordingly
"""

import torch
import torch.nn as nn


class DenseCoreNetwork(nn.Module):
    """
    ACOPF PINN Core Neural Network - PyTorch Version
    Supports dynamic layer configuration
    """

    def __init__(
            self,
            n_gen,
            n_gen_non_slack,
            neurons_in_hidden_layers_V,
            neurons_in_hidden_layers_G,
            neurons_in_hidden_layers_Lg
    ):
        super(DenseCoreNetwork, self).__init__()

        self.n_gen = n_gen
        self.n_gen_non_slack = n_gen_non_slack

        # ========== V Network (predict generator bus Vm) ==========
        v_layers = []
        prev_size = None  # Will be determined at first forward
        for i, n_units in enumerate(neurons_in_hidden_layers_V):
            if i > 0:
                v_layers.append(nn.Linear(prev_size, n_units))
                v_layers.append(nn.ReLU())
            prev_size = n_units

        self.v_hidden_sizes = neurons_in_hidden_layers_V
        self.v_hidden = None  # Lazy initialization

        # V output layer: only output vm_gen (generator buses)
        self.v_output = nn.Linear(prev_size, n_gen)

        # ========== G Network (predict non-slack generator Pg) ==========
        g_layers = []
        prev_size = None
        for i, n_units in enumerate(neurons_in_hidden_layers_G):
            if i > 0:
                g_layers.append(nn.Linear(prev_size, n_units))
                g_layers.append(nn.ReLU())
            prev_size = n_units

        self.g_hidden_sizes = neurons_in_hidden_layers_G
        self.g_hidden = None  # Lazy initialization

        # G output layer: only output pg_non_slack (exclude slack generators)
        self.g_output = nn.Linear(prev_size, n_gen_non_slack)

        # ========== Lg Network (dual variables) ==========
        lg_layers = []
        prev_size = None
        for i, n_units in enumerate(neurons_in_hidden_layers_Lg):
            if i > 0:
                lg_layers.append(nn.Linear(prev_size, n_units))
                lg_layers.append(nn.ReLU())
            prev_size = n_units

        self.lg_hidden_sizes = neurons_in_hidden_layers_Lg
        self.lg_hidden = None  # Lazy initialization

        # Lg output layers: dual variables for constraints
        self.mu_pg_min_output = nn.Linear(prev_size, n_gen_non_slack)  # Only non-slack
        self.mu_pg_max_output = nn.Linear(prev_size, n_gen_non_slack)
        self.mu_vm_min_output = nn.Linear(prev_size, n_gen)  # Generator buses
        self.mu_vm_max_output = nn.Linear(prev_size, n_gen)

        self._initialized = False

    def _initialize_layers(self, input_size):
        """Initialize network layers at first forward pass"""
        if self._initialized:
            return

        device = next(self.parameters()).device

        # Initialize V network
        v_layers = []
        prev_size = input_size
        for n_units in self.v_hidden_sizes:
            v_layers.append(nn.Linear(prev_size, n_units))
            v_layers.append(nn.ReLU())
            prev_size = n_units
        self.v_hidden = nn.Sequential(*v_layers).to(device)

        # Initialize G network
        g_layers = []
        prev_size = input_size
        for n_units in self.g_hidden_sizes:
            g_layers.append(nn.Linear(prev_size, n_units))
            g_layers.append(nn.ReLU())
            prev_size = n_units
        self.g_hidden = nn.Sequential(*g_layers).to(device)

        # Initialize Lg network
        lg_layers = []
        prev_size = input_size
        for n_units in self.lg_hidden_sizes:
            lg_layers.append(nn.Linear(prev_size, n_units))
            lg_layers.append(nn.ReLU())
            prev_size = n_units
        self.lg_hidden = nn.Sequential(*lg_layers).to(device)

        self._initialized = True

    def forward(self, inputs):
        """
        Forward propagation

        Args:
            inputs: (batch_size, n_features) - [pd, qd]

        Returns:
            vm_gen: (batch_size, n_gen) - Generator bus voltage magnitudes
            pg_non_slack: (batch_size, n_gen_non_slack) - Non-slack generator active power
            mu_pg_min: (batch_size, n_gen_non_slack)
            mu_pg_max: (batch_size, n_gen_non_slack)
            mu_vm_min: (batch_size, n_gen)
            mu_vm_max: (batch_size, n_gen)
        """
        # Initialize on first call
        if not self._initialized:
            self._initialize_layers(inputs.shape[1])

        # V network forward pass
        x_v = self.v_hidden(inputs)
        vm_gen = self.v_output(x_v)  # Only generator bus Vm

        # G network forward pass
        x_g = self.g_hidden(inputs)
        pg_non_slack = self.g_output(x_g)  # Only non-slack generator Pg

        # Lg network forward pass (dual variables)
        x_lg = self.lg_hidden(inputs)
        mu_pg_min = self.mu_pg_min_output(x_lg)
        mu_pg_max = self.mu_pg_max_output(x_lg)
        mu_vm_min = self.mu_vm_min_output(x_lg)
        mu_vm_max = self.mu_vm_max_output(x_lg)

        return vm_gen, pg_non_slack, mu_pg_min, mu_pg_max, mu_vm_min, mu_vm_max


if __name__ == "__main__":
    print("=" * 70)
    print("Testing ACOPF DenseCoreNetwork (PyTorch)")
    print("=" * 70)

    # Create test model
    n_gen = 5
    n_gen_non_slack = 4  # Exclude 1 slack generator
    batch_size = 32
    n_loads = 14

    model = DenseCoreNetwork(
        n_gen=n_gen,
        n_gen_non_slack=n_gen_non_slack,
        neurons_in_hidden_layers_V=[128, 128, 128],
        neurons_in_hidden_layers_G=[128, 128, 128],
        neurons_in_hidden_layers_Lg=[128, 128, 128]
    )

    # Test forward pass
    x_test = torch.randn(batch_size, 2 * n_loads)  # [pd, qd]

    vm_gen, pg_non_slack, mu_pg_min, mu_pg_max, mu_vm_min, mu_vm_max = model(x_test)

    print(f"\nInput dimension: {x_test.shape}")
    print(f"\nOutput dimensions:")
    print(f"  vm_gen: {vm_gen.shape} (generator buses only)")
    print(f"  pg_non_slack: {pg_non_slack.shape} (non-slack generators only)")
    print(f"  mu_pg_min: {mu_pg_min.shape}")
    print(f"  mu_pg_max: {mu_pg_max.shape}")
    print(f"  mu_vm_min: {mu_vm_min.shape}")
    print(f"  mu_vm_max: {mu_vm_max.shape}")

    # Test parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")

    print("\n✓ Test passed!")