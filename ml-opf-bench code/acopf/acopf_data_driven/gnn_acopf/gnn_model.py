# gnn_model.py
"""
GNN Model for ACOPF (V9 - Dynamic Params Support)

Modifications (V9):
- forward() accepts optional params for dynamic constraint injection
- build_node_features() uses passed params instead of self.params
- Supports same topology with different constraints (e.g., API test)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class GNNLayer(MessagePassing):
    """
    Single GNN layer - Message passing layer
    (No changes from V8)
    """

    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__(aggr='mean')

        self.message_nn = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels),
            nn.Tanh()
        )

        self.update_nn = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Tanh()
        )

    def forward(self, x, edge_index, edge_attr):
        x_original = x
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.update_nn(torch.cat([x_original, out], dim=-1))
        return out

    def message(self, x_j, edge_attr):
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_nn(msg_input)


class ACOPF_GNN(nn.Module):
    """
    GNN-based ACOPF solver (V9 - Dynamic Params)

    Key modifications from V8:
        - forward() accepts optional `params` argument
        - build_node_features() uses dynamic params for constraints
        - Enables testing with different constraint parameters (same topology)
    """

    def __init__(
            self,
            n_buses,
            n_gen,
            n_gen_non_slack,
            n_loads,
            hidden_dim=64,
            num_gnn_layers=4,
            edge_dim=2,
            params=None
    ):
        super().__init__()

        self.n_buses = n_buses
        self.n_gen = n_gen
        self.n_gen_non_slack = n_gen_non_slack
        self.n_loads = n_loads
        self.params = params  # Default params (training)

        # ========== Handle sparse bus numbering ==========
        bus_id_to_idx = params['general']['bus_id_to_idx']
        gen_bus_ids = params['general']['gen_bus_ids']
        non_slack_gen_idx = params['general']['non_slack_gen_idx']

        # These indices are topology-dependent, fixed at init (same for train/test)
        self.gen_indices = torch.tensor(
            [bus_id_to_idx[int(gid)] for gid in gen_bus_ids],
            dtype=torch.long
        )

        non_slack_gen_bus_ids = gen_bus_ids[non_slack_gen_idx]
        self.non_slack_gen_indices = torch.tensor(
            [bus_id_to_idx[int(gid)] for gid in non_slack_gen_bus_ids],
            dtype=torch.long
        )

        # Node feature dimension
        node_feature_dim = 9

        # Input embedding layer
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.Tanh()
        )

        # GNN message passing layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim, edge_dim)
            for _ in range(num_gnn_layers)
        ])

        # Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_gnn_layers)
        ])

        # Output head 1: Non-slack generator active power
        self.pg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Output head 2: Generator bus voltage
        self.vm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def build_node_features(self, pd_qd_batch, device, params):
        """
        Build node feature matrix for each batch sample

        Args:
            pd_qd_batch: [batch_size, 2*n_loads] - Normalized load data
            device: torch device
            params: Parameters dictionary (can be train or test params)

        Returns:
            node_features: [batch_size, n_buses, 9]
        """
        batch_size = pd_qd_batch.shape[0]

        # ========== Use passed params (dynamic) ==========
        bus_id_to_idx = params['general']['bus_id_to_idx']
        load_bus_ids = params['general']['load_bus_ids']
        gen_bus_ids = params['general']['gen_bus_ids']

        # Extract constraint parameters from passed params (key change!)
        pg_min_all = torch.tensor(
            params['generator']['pg_min'],
            dtype=torch.float32,
            device=device
        ).squeeze()
        pg_max_all = torch.tensor(
            params['generator']['pg_max'],
            dtype=torch.float32,
            device=device
        ).squeeze()

        vm_min = torch.tensor(
            params['bus']['vm_min'],
            dtype=torch.float32,
            device=device
        )
        vm_max = torch.tensor(
            params['bus']['vm_max'],
            dtype=torch.float32,
            device=device
        )

        # Initialize node features [batch, n_buses, 9]
        node_features = torch.zeros(batch_size, self.n_buses, 9, device=device)

        # 1. Fill load data (pd, qd) - dimensions 0 and 1
        pd_batch = pd_qd_batch[:, :self.n_loads]
        qd_batch = pd_qd_batch[:, self.n_loads:]

        for i, lid in enumerate(load_bus_ids):
            idx = bus_id_to_idx[int(lid)]
            node_features[:, idx, 0] = pd_batch[:, i]
            node_features[:, idx, 1] = qd_batch[:, i]

        # 2. Fill generator constraints (pg_min, pg_max) - dimensions 2 and 3
        for i, gid in enumerate(gen_bus_ids):
            idx = bus_id_to_idx[int(gid)]
            node_features[:, idx, 2] = pg_min_all[i]
            node_features[:, idx, 3] = pg_max_all[i]

        # 3. Fill voltage constraints (vm_min, vm_max) - dimensions 4 and 5
        node_features[:, :, 4] = vm_min.unsqueeze(0)
        node_features[:, :, 5] = vm_max.unsqueeze(0)

        # 4. Fill node type markers - dimensions 6, 7, 8
        for gid in gen_bus_ids:
            idx = bus_id_to_idx[int(gid)]
            node_features[:, idx, 6] = 1.0  # is_gen

        for lid in load_bus_ids:
            idx = bus_id_to_idx[int(lid)]
            node_features[:, idx, 7] = 1.0  # is_load

        return node_features

    def forward(self, x, edge_index, edge_attr, params=None):
        """
        Forward propagation

        Args:
            x: [batch_size, 2*n_loads] - Normalized load data
            edge_index: [2, num_edges] - Graph edges (PyG format)
            edge_attr: [num_edges, edge_dim] - Edge features
            params: Optional parameters dict (default: self.params)
                    Pass test_params during evaluation for correct constraints

        Returns:
            output: [batch_size, n_gen_non_slack + n_gen]
        """
        # Use passed params or default to training params
        if params is None:
            params = self.params

        batch_size = x.shape[0]
        device = x.device

        # 1. Build node features with dynamic params
        node_features = self.build_node_features(x, device, params)

        # 2. Batch processing
        all_pg = []
        all_vm = []

        for b in range(batch_size):
            h = node_features[b]

            # 3. Node embedding
            h = self.node_embedding(h)

            # 4. GNN message passing
            for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms):
                h_prev = h
                h = gnn_layer(h, edge_index, edge_attr)
                h = layer_norm(h)
                h = h + h_prev  # Residual connection

            # 5. Extract features (indices are topology-based, same for train/test)
            h_non_slack_gen = h[self.non_slack_gen_indices]
            h_gen = h[self.gen_indices]

            # 6. Output heads
            pg_non_slack = self.pg_head(h_non_slack_gen).squeeze(-1)
            vm_gen = self.vm_head(h_gen).squeeze(-1)

            all_pg.append(pg_non_slack)
            all_vm.append(vm_gen)

        # 7. Stack batch results
        pg_batch = torch.stack(all_pg, dim=0)
        vm_batch = torch.stack(all_vm, dim=0)

        # 8. Concatenate output
        output = torch.cat([pg_batch, vm_batch], dim=-1)

        return output