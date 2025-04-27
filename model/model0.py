import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing
from torch_geometric.data import Data
from torch_scatter import scatter_mean

class GeoInvariantConv(MessagePassing):
    """
    Graph convolutional layer that maintains geometric invariance/equivariance
    by using relative distances and angles
    """
    def __init__(self, in_channels, out_channels):
        super(GeoInvariantConv, self).__init__(aggr="mean")
        self.lin = nn.Linear(in_channels * 2 + 1, out_channels)  # Node features + edge features
        
    def forward(self, x, edge_index, edge_attr):
        # x: Node features [N, in_channels]
        # edge_index: Graph connectivity [2, E]
        # edge_attr: Edge features (distances) [E, 1]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
    def message(self, x_i, x_j, edge_attr):
        # Create message using features from source and target nodes and their edge features
        message = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.lin(message)

class BindingSiteGNN(nn.Module):
    """
    Graph neural network for processing binding site atoms
    with geometric invariance considerations
    """
    def __init__(self, node_feature_dim=10, hidden_dim=64, output_dim=256):
        super(BindingSiteGNN, self).__init__()
        
        # Initial node feature processing
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Graph convolution layers
        self.conv1 = GeoInvariantConv(hidden_dim, hidden_dim)
        self.conv2 = GeoInvariantConv(hidden_dim, hidden_dim)
        self.conv3 = GeoInvariantConv(hidden_dim, hidden_dim)
        
        # Final output transformation
        self.output_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Normalization and activation
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        
    def forward(self, data):
        # Get graph components
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Initial node feature encoding
        x = self.node_encoder(x)
        
        # Apply graph convolutions with residual connections
        x1 = self.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = x + x1  # Residual connection
        
        x2 = self.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = x + x2  # Residual connection
        
        x3 = self.conv3(x, edge_index, edge_attr)
        x = x + x3  # Residual connection
        
        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Final output projection
        out = self.output_nn(x)
        
        return out

class SubstrateEncoder(nn.Module):
    def __init__(self, input_dim=384, output_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.ReLU()
        )
    
    def forward(self, substrate_features):
        return self.encoder(substrate_features)

class EnzymeKineticsPredictionModel(nn.Module):
    """
    End-to-end model for predicting enzyme kinetics parameters
    using binding site GNN and substrate embeddings
    """
    def __init__(self, node_feature_dim=10, substrate_dim=384, hidden_dim=256, output_dim=2):
        super(EnzymeKineticsPredictionModel, self).__init__()
        
        # Binding site encoder using GNN
        self.binding_site_encoder = BindingSiteGNN(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        # Substrate encoder 
        self.substrate_encoder = SubstrateEncoder(
            input_dim=substrate_dim,
            output_dim=hidden_dim
        )
        
        # Fusion and prediction layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.criterion = nn.MSELoss()
        
    def forward(self, binding_site_data, substrate_features):
        # Process binding site using GNN
        binding_site_embedding = self.binding_site_encoder(binding_site_data)
        
        # Process substrate features
        substrate_embedding = self.substrate_encoder(substrate_features)
        
        # Concatenate and predict
        combined = torch.cat([binding_site_embedding, substrate_embedding], dim=1)
        output = self.fusion(combined)
        
        return output