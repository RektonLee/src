import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool,MessagePassing, GatedGraphConv
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
        
class EdgeEnhancedGNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr='add')
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [N, node_dim], edge_attr: [E, edge_dim]
        edge_attr = self.edge_mlp(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: neighbor node features
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.node_mlp(msg_input)

class PocketGNN1(nn.Module):
    """
    Edge-enhanced Graph Neural Network for Enzyme Kinetics (kcat, Km)
    """
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.node_encoder = nn.Linear(node_input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EdgeEnhancedGNNLayer(hidden_dim, edge_input_dim)
            for _ in range(num_layers)
        ])
        self.readout = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Predict [kcat, Km]
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)

        for layer in self.layers:
            x = x + layer(x, edge_index, edge_attr)  # residual connection

        x = self.readout(x, batch)
        out = self.mlp(x)
        return out


class PocketGNN_Gated(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList([GatedGraphConv(out_channels=hidden_dim, num_layers=1) for _ in range(num_layers)])
        self.readout = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 输出 kcat, Km
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_proj(x)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.readout(x, batch)
        return self.mlp(x)
    



'''
级别 | 内容 | 技术细节
节点特征 | 元素onehot + 残基onehot + 是否配体 + 最近邻距离 + PQR电荷 + 电子结构 + Atom-level手工特征 | get_atom_features（RDKit提取芳香性、杂化轨道、价态、电荷等）
分子全局特征 | 分子指纹（Morgan Fingerprint） + SMILES的CLS向量（可选） | 用Attention融合到节点特征上（dynamic weight attention）
边特征 | 单位方向向量（vec/distance）+ 距离标量 + 是否共价键（推测） | 更细粒度边建模
GNN结构 | GatedGCNConv升级版（支持边特征）或 EdgeGatedGCNConv | 保持E(3)等变性/不变性
Loss优化 | 损失改为HuberLoss（生物实验数据噪声大，更稳定） | 
额外 | 保留pdb_id、原子索引信息，便于后续可解释性分析 | 

'''