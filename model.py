import torch
import torch.nn as nn
import torch.nn.functional as F
# Define the SubstrateAttention class that was referenced but not implemented
from torch_geometric.nn import GCNConv, global_mean_pool

class BindingSiteEncoder(nn.Module):
    def __init__(self, input_channels=7, output_dim=256):
        super().__init__()
        
        # 处理活性位点原子特征的多层感知机
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # 自注意力层处理原子间关系
        self.self_attention = nn.MultiheadAttention(128, 4, batch_first=True)
        
        # 最终处理
        self.final = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
    
    def forward(self, binding_site_features):
        # binding_site_features: [batch_size, max_atoms, features]
        batch_size, max_atoms, _ = binding_site_features.shape
        
        # 对每个原子应用MLP
        atom_features = self.mlp(binding_site_features)
        
        # 应用自注意力机制处理原子间关系
        attn_output, _ = self.self_attention(atom_features, atom_features, atom_features)
        
        # 池化获得全局表示
        global_repr = attn_output.mean(dim=1)
        
        # 最终转换
        output = self.final(global_repr)
        return output


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
    def __init__(self, node_feature_dim=30, substrate_dim=768, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.substrate_fc = nn.Linear(substrate_dim, hidden_dim)
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, 2)  # 输出 Km 和 kcat
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.criterion = nn.MSELoss()

    def forward(self, binding_site, substrate):
        # binding_site 是 Data 对象，包含 x, edge_index, edge_attr, batch
        x, edge_index, edge_attr, batch = binding_site.x, binding_site.edge_index, binding_site.edge_attr, binding_site.batch

        # GNN 处理口袋图
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        pocket_embedding = global_mean_pool(x, batch)  # 池化成固定维度

        # 处理底物嵌入
        substrate_emb = self.relu(self.substrate_fc(substrate))
        substrate_emb = self.dropout(substrate_emb)

        # 融合
        fused = torch.cat([pocket_embedding, substrate_emb], dim=-1)
        fused = self.relu(self.fusion_fc(fused))
        fused = self.dropout(fused)

        # 输出
        output = self.output_fc(fused)
        return output