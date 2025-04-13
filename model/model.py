import torch
from torch import nn
from torch_cluster import radius_graph
from torch_scatter import scatter_mean

class EquivariantGNN(nn.Module):
    """等变图神经网络"""
    def __init__(self):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(3, 64),  # 输入特征维度
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        self.conv_layers = nn.ModuleList([
            EquivariantBlock(128, 128),
            EquivariantBlock(128, 256)
        ])
    
    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        edge_index = radius_graph(pos, r=5.0, batch=batch)
        
        # 节点特征编码
        h = self.node_encoder(x)
        
        # 等变卷积
        for conv in self.conv_layers:
            h = conv(h, pos, edge_index)
        
        # 全局池化
        return scatter_mean(h, batch, dim=0)

class EquivariantBlock(nn.Module):
    """等变网络块"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, in_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2*in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, h, pos, edge_index):
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)
        edge_feat = self.distance_encoder(dist)
        
        # 消息传递
        messages = torch.cat([h[row], edge_feat], dim=-1)
        aggregated = scatter_mean(messages, col, dim=0, dim_size=h.size(0))
        
        # 节点更新
        return self.update_mlp(torch.cat([h, aggregated], dim=-1))