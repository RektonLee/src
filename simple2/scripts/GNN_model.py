import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch_geometric.utils as utils

class PocketGNN(nn.Module):
    """
    Graph Neural Network for Enzyme Kinetic Parameter Prediction (Km, kcat).
    - Node features: One-hot encoded atom types
    - Edge index: Distance-based neighboring within pocket
    - Output: [kcat, Km] (regression)
    """

    def __init__(self, num_atom_types, hidden_dim=128):
        super(PocketGNN, self).__init__()

        # Atom type embedding
        self.linear = nn.Linear(num_atom_types, hidden_dim)


        # GCN layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Readout
        self.readout = global_mean_pool

        # MLP for final regression
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Predict kcat and Km simultaneously
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 确保没有孤立节点
        edge_index, _ = utils.add_self_loops(edge_index, num_nodes=x.size(0))
        
        # 特征变换
        x = self.linear(x)
        
        # 消息传递
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # 全局池化 - 这里确保每个图会被池化为一个向量
        x = self.readout(x, batch)  # [batch_size, hidden_dim]
        
        # MLP回归
        out = self.mlp(x)  # [batch_size, 2]
        return out
