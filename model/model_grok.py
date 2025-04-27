import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing

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

class ImprovedEnzymePredictionModel(nn.Module):
    def __init__(self, node_feature_dim=26, substrate_dim=384, hidden_dim=128):
        super().__init__()
        # GNN for binding site graph
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Substrate processing
        self.substrate_fc = nn.Linear(substrate_dim, hidden_dim)
        
        # Fusion and output
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, 2)  # Km and kcat
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.criterion = nn.MSELoss()

    def forward(self, binding_site, substrate):
        # binding_site: PyG Data object (x, edge_index, edge_attr, batch)
        x, edge_index, batch = binding_site.x, binding_site.edge_index, binding_site.batch
        
        # GNN layers
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        pocket_embedding = global_mean_pool(x, batch)  # Pool to fixed size
        
        # Substrate embedding
        substrate_emb = self.relu(self.substrate_fc(substrate))
        substrate_emb = self.dropout(substrate_emb)
        
        # Fusion
        fused = torch.cat([pocket_embedding, substrate_emb], dim=-1)
        fused = self.relu(self.fusion_fc(fused))
        fused = self.dropout(fused)
        
        # Output
        return self.output_fc(fused)