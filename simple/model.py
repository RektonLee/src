import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class ImprovedEnzymePredictionModel(nn.Module):
    def __init__(self, node_feature_dim=30, substrate_dim=384, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.substrate_fc = nn.Linear(substrate_dim, hidden_dim)
        self.fusion_fc = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.criterion = nn.MSELoss()

    def forward(self, binding_site, substrate):
        x = self.relu(self.conv1(binding_site.x, binding_site.edge_index))
        x = self.conv2(self.dropout(x), binding_site.edge_index)
        x = global_mean_pool(x, binding_site.batch)

        substrate = self.dropout(self.relu(self.substrate_fc(substrate)))
        fused = self.dropout(self.relu(self.fusion_fc(torch.cat([x, substrate], dim=-1))))
        return self.output_fc(fused)