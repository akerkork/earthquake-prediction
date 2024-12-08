import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

class BipartiteReadIn(MessagePassing):
    """
    Bipartite Read-in implementation using MessagePassing.
    Aggregates features from U-set to V-set nodes.
    """
    def __init__(self, aggr='mean'):
        super().__init__(aggr=aggr)  # Aggregation can be 'mean', 'sum', or 'max'

    def forward(self, x_u, x_v, edge_index):
        return self.propagate(edge_index, x=(x_u, x_v))

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class BipartiteReadOut(MessagePassing):
    """
    Bipartite Read-out implementation using MessagePassing.
    Aggregates features from V-set back to U-set nodes.
    """
    def __init__(self, aggr='mean'):
        super().__init__(aggr=aggr)  # Aggregation can be 'mean', 'sum', or 'max'

    def forward(self, x_v, x_u, edge_index):
        return self.propagate(edge_index.flip(0), x=(x_v, x_u))  # Flip edge_index for reverse aggregation

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


def generate_synthetic_bipartite_graph_with_readin_readout(num_nodes_u=10, num_nodes_v=15, feature_dim=5, num_edges=20):
    """
    Generate synthetic bipartite graph data with random features, perform Read-in and Read-out, and analyze results.

    Parameters:
    - num_nodes_u (int): Number of nodes in set U.
    - num_nodes_v (int): Number of nodes in set V.
    - feature_dim (int): Dimensionality of node features.
    - num_edges (int): Number of edges between U and V.

    Returns:
    - Data object representing the bipartite graph with aggregated features after Read-in and Read-out.
    """
    # Generate random features for nodes in U and V
    features_u = torch.rand((num_nodes_u, feature_dim))
    features_v = torch.rand((num_nodes_v, feature_dim))

    # Generate random edges between U and V
    edge_index_u = torch.randint(0, num_nodes_u, (num_edges,))
    edge_index_v = torch.randint(0, num_nodes_v, (num_edges,))
    edge_index = torch.stack([edge_index_u, edge_index_v], dim=0)

    # Create a PyTorch Geometric Data object
    data = Data(
        x_u=features_u,
        x_v=features_v,
        edge_index=edge_index
    )

    # Perform Bipartite Read-in
    read_in = BipartiteReadIn(aggr='mean')
    aggregated_features_v = read_in(data.x_u, data.x_v, data.edge_index)
    data.aggregated_x_v = aggregated_features_v

    # Perform Bipartite Read-out
    read_out = BipartiteReadOut(aggr='mean')
    aggregated_features_u = read_out(data.aggregated_x_v, data.x_u, data.edge_index)
    data.aggregated_x_u = aggregated_features_u

    return data

# Example usage
synthetic_data = generate_synthetic_bipartite_graph_with_readin_readout()
print("Features of U nodes (Initial):\n", synthetic_data.x_u)
print("Features of V nodes (Initial):\n", synthetic_data.x_v)
print("Edge index:\n", synthetic_data.edge_index)
print("Aggregated Features of V nodes (After Read-in):\n", synthetic_data.aggregated_x_v)
print("Aggregated Features of U nodes (After Read-out):\n", synthetic_data.aggregated_x_u)
