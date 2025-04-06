import torch
import torch.nn as nn
import torch.nn.functional as F
# Define the SubstrateAttention class that was referenced but not implemented
ddd

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
    def __init__(self, binding_site_dim=7, substrate_dim=384):
        super().__init__()
        
        # 使用BindingSiteEncoder处理活性位点特征
        self.binding_site_encoder = BindingSiteEncoder(
            input_channels=binding_site_dim,
            output_dim=256
        )
        
        # 处理底物特征
        self.substrate_encoder = SubstrateEncoder(
            input_dim=substrate_dim,
            output_dim=256
        )
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(512, 256),  # 256(binding_site) + 256(substrate) = 512
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 预测Km和kcat
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
    
    def forward(self, binding_site_features, substrate_features):
        # 编码活性位点特征
        binding_site_encoded = self.binding_site_encoder(binding_site_features)  # [batch, 256]
        
        # 编码底物特征
        substrate_encoded = self.substrate_encoder(substrate_features)  # [batch, 256]
        
        # 特征融合
        combined = torch.cat([binding_site_encoded, substrate_encoded], dim=1)
        
        # 预测
        output = self.predictor(combined)
        return output