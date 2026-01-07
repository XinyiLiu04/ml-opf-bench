# gnn_model_dcopf.py
"""
GNN Model for DCOPF (with Dynamic Params Support)
Version: v4.1 - API_TEST mode with dynamic graph and constraints

Key Features:
- Input: Active load pd
- Output: Non-slack generator active power
- Slack nodes participate in message passing but no output
- Slack generation determined by power balance
- Handles sparse bus numbering
- Accepts dynamic params for API testing
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class GNNLayer(MessagePassing):
    """
    Single GNN Layer - Message Passing

    Features:
        - Aggregation: mean
        - Edge features: (r_pu, x_pu)
        - Activation: Tanh
    """

    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__(aggr='mean')

        # Message function: neighbor features + edge features
        self.message_nn = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels),
            nn.Tanh()
        )

        # Update function: aggregated message + self features
        self.update_nn = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Tanh()
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass

        Args:
            x: [N, in_channels] - Node features
            edge_index: [2, E] - Edge connectivity
            edge_attr: [E, edge_dim] - Edge features

        Returns:
            [N, out_channels] - Updated node features
        """
        x_original = x
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.update_nn(torch.cat([x_original, out], dim=-1))
        return out

    def message(self, x_j, edge_attr):
        """
        Construct message

        Args:
            x_j: [E, in_channels] - Source node features (neighbors)
            edge_attr: [E, edge_dim] - Edge features

        Returns:
            [E, out_channels] - Messages
        """
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_nn(msg_input)


class DCOPF_GNN(nn.Module):
    """
    GNN-based DCOPF Solver (with Slack Bus support and Dynamic Params)

    Architecture:
        Input(loads) → Node Features → Embedding → GNN Layers×N → Output Head(non-slack pg)

    Slack Bus Handling:
    -------------------
    1. Message passing: All nodes (including Slack) participate
       - Slack node constraints (pg_min, pg_max) encoded in node features
       - Slack nodes influence neighbor representations via message passing
       - Preserves complete network topology information

    2. Output extraction: Only non-Slack generator nodes
       - Use self.gen_indices to locate non-Slack generator nodes
       - Output dimension = n_gen_non_slack

    3. Power balance reconstruction: Done in evaluation via reconstruct_full_pg()
       - Pg_slack = Σ Pd - Σ Pg_non_slack
       - Ensures physical consistency

    Dynamic Params for API Testing:
    -------------------------------
    - build_node_features() accepts optional params argument
    - forward() accepts optional params argument
    - Training: params=None, uses self.params (v=0.12 constraints)
    - API Testing: params=test_params, uses API constraints
    - Enables true generalization to different constraint sets

    Advantages:
    - Fully utilizes graph structure and topology
    - Maintains physical constraint correctness
    - Aligns evaluation metrics with DNN/LR methods
    - Can adapt to different constraint parameters at test time

    Note:
        - Handles sparse bus numbering (using bus_id_to_idx mapping)
        - Node features include all generator constraints (including Slack)
        - Output: Only non-Slack generator active power
    """

    def __init__(
            self,
            n_buses,
            n_gen_non_slack,
            hidden_dim=64,
            num_gnn_layers=4,
            edge_dim=2,
            params=None
    ):
        super().__init__()

        if params is None:
            raise ValueError("params cannot be None! Must contain Slack info")

        self.n_buses = n_buses
        self.n_gen = n_gen_non_slack
        self.params = params  # Default params (training params)

        # Build bus ID to index mapping
        bus_ids = params['general']['l_bus']
        self.bus_id_to_idx = {int(bid): i for i, bid in enumerate(bus_ids)}

        # Get non-Slack generator node indices
        non_slack_gen_indices = params['general']['non_slack_gen_indices']
        all_gen_bus_ids = params['general']['g_bus']
        non_slack_gen_bus_ids = all_gen_bus_ids[non_slack_gen_indices]

        # Map non-Slack generator bus IDs to graph node indices
        self.gen_indices = torch.tensor(
            [self.bus_id_to_idx[int(gid)] for gid in non_slack_gen_bus_ids],
            dtype=torch.long
        )

        print(f"\n[GNN Model] Initialization:")
        print(f"  Total generators: {params['general']['n_g']}")
        print(f"  Non-Slack generators: {n_gen_non_slack}")
        print(f"  Slack generators: {params['general']['n_slack_gens']}")
        print(f"  Output dimension: {n_gen_non_slack} (non-Slack only)")
        print(f"  Dynamic params support: Enabled")

        # Node feature dimension: [pd, pg_min, pg_max, is_gen, is_load]
        node_feature_dim = 5

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

        # Output head: Generator active power
        self.pg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def build_node_features(self, pd_batch, device, params=None):
        """
        Build node feature matrix for each batch (with dynamic params)

        Key: Handles sparse bus numbering correctly

        Dynamic Params Support:
        ----------------------
        Although model only predicts non-Slack generators, node features still
        contain constraint information for ALL generators (including Slack).
        This is because:
        1. Slack nodes participate in graph message passing
        2. Slack node constraints help neighbor node representation learning
        3. Maintains complete network topology

        Args:
            pd_batch: [batch_size, n_buses] - Normalized load data
            device: torch.device
            params: Optional, params dict for building node features
                   - None: Use self.params (training params)
                   - dict: Use specified params (API test params)

        Returns:
            node_features: [batch_size, n_buses, 5]
        """
        # Decide which params to use
        if params is None:
            params = self.params

        batch_size = pd_batch.shape[0]

        # Extract constraint parameters from the (potentially dynamic) params
        pg_min = torch.tensor(
            params['constraints']['Pg_min'],
            dtype=torch.float32,
            device=device
        ).squeeze()  # [n_gen_total]

        pg_max = torch.tensor(
            params['constraints']['Pg_max'],
            dtype=torch.float32,
            device=device
        ).squeeze()  # [n_gen_total]

        # Use ALL generator bus IDs (including Slack)
        gen_bus_ids = params['general']['g_bus']

        # Initialize node features [batch, n_buses, 5]
        node_features = torch.zeros(batch_size, self.n_buses, 5, device=device)

        # 1. Fill load data (pd) - dimension 0
        node_features[:, :, 0] = pd_batch

        # 2. Fill generator constraints (pg_min, pg_max) - dimensions 1 and 2
        # For ALL generator nodes (including Slack)
        for i, gid in enumerate(gen_bus_ids):
            idx = self.bus_id_to_idx[int(gid)]
            node_features[:, idx, 1] = pg_min[i]  # pg_min (may be from API params)
            node_features[:, idx, 2] = pg_max[i]  # pg_max (may be from API params)

        # 3. Fill node type markers - dimensions 3, 4
        # Mark all generator nodes (including Slack)
        for gid in gen_bus_ids:
            idx = self.bus_id_to_idx[int(gid)]
            node_features[:, idx, 3] = 1.0  # is_gen

        # All buses are load nodes (in DCOPF)
        node_features[:, :, 4] = 1.0  # is_load

        return node_features

    def forward(self, x, edge_index, edge_attr, params=None):
        """
        Forward pass (with Slack Bus and dynamic params support)

        Process:
        1. Build node features (all nodes, including Slack)
        2. Node embedding
        3. GNN message passing (all nodes participate, including Slack)
        4. Extract only non-Slack generator node outputs
        5. Return non-Slack Pg predictions

        Dynamic Params Support:
        ----------------------
        - Training: params=None, uses self.params (v=0.12 constraints)
        - API Testing: params=test_params, node features reflect API constraints
        - Enables GNN to adapt to different constraint sets at test time

        Args:
            x: [batch_size, n_buses] - Normalized load data
            edge_index: [2, num_edges] - Graph edges (PyG format)
            edge_attr: [num_edges, edge_dim] - Edge features
            params: Optional, params dict for building node features
                   - None: Use self.params (training mode)
                   - dict: Use specified params (API_TEST mode)

        Returns:
            output: [batch_size, n_gen_non_slack] - Non-Slack generator active power
        """
        batch_size = x.shape[0]
        device = x.device

        # 1. Build node features (may use API params)
        node_features = self.build_node_features(x, device, params)  # [batch, n_buses, 5]

        # 2. Process each sample in batch
        all_pg = []

        for b in range(batch_size):
            # Single sample node features [n_buses, 5]
            h = node_features[b]

            # 3. Node embedding
            h = self.node_embedding(h)  # [n_buses, hidden_dim]

            # 4. GNN message passing (multiple layers)
            # All nodes participate (including Slack)
            for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms):
                h_prev = h
                h = gnn_layer(h, edge_index, edge_attr)
                h = layer_norm(h)
                # Residual connection
                h = h + h_prev

            # 5. Extract only non-Slack generator node features
            h_gen = h[self.gen_indices]  # [n_gen_non_slack, hidden_dim]

            # 6. Output head
            pg = self.pg_head(h_gen).squeeze(-1)  # [n_gen_non_slack]

            all_pg.append(pg)

        # 7. Stack batch results
        pg_batch = torch.stack(all_pg, dim=0)  # [batch, n_gen_non_slack]

        return pg_batch  # Return non-Slack Pg