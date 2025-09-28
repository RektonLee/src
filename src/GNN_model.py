import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool,MessagePassing, GatedGraphConv,GATConv
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)

        for layer in self.layers:
            x = x + layer(x, edge_index, edge_attr)  # residual connection

        x = self.readout(x, batch)
        out = self.mlp(x)
        
        # 检查模型输出是否包含 NaN
        if torch.isnan(out).any():
            raise ValueError("模型输出包含 NaN 值")
        
        return out
    def get_graph_embedding(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)
        for layer in self.layers:
            x = x + layer(x, edge_index, edge_attr)
        x = self.readout(x, batch)  # 🔹 关键：池化为图表示
        return x
    # def get_graph_embedding(self, data):
    #     """获取图嵌入表示(在池化层之前)"""
    #     x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    #     x = self.node_encoder(x)
        
    #     for layer in self.layers:
    #         x = x + layer(x, edge_index, edge_attr)
            
    #     return x  # 返回节点级嵌入，不进行池化


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

class PocketGNNWithAttentionNoTemp(nn.Module):
    """
    GNN with Graph Attention Layers for Enzyme Kinetics (kcat, Km), WITHOUT temperature input.
    """
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim=256, num_layers=6, heads=8, dropout=0.1, concat_heads=True):
        super().__init__()
        self.node_encoder = nn.Linear(node_input_dim, hidden_dim)

        self.att_layers = nn.ModuleList()
        current_dim = hidden_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            if concat_heads and not is_last:
                self.att_layers.append(
                    GATConv(current_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True, edge_dim=edge_input_dim if i==0 else None)
                )
                current_dim = hidden_dim
            else:
                self.att_layers.append(
                    GATConv(current_dim, hidden_dim, heads=heads, dropout=dropout, concat=False, edge_dim=edge_input_dim if i==0 else None)
                )
                current_dim = hidden_dim

        self.readout = global_mean_pool

        # MLP for regression (no temperature fusion)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # 输出 kcat, Km
        )

    def forward(self, data, return_attention_weights=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)
        x = F.relu(x)

        all_attention_weights = []
        for i, layer in enumerate(self.att_layers):
            if hasattr(layer, 'edge_dim') and layer.edge_dim is not None and i == 0:
                x_new, attention_info = layer(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            else:
                x_new, attention_info = layer(x, edge_index, return_attention_weights=True)
            x = F.elu(x_new)
            if return_attention_weights:
                all_attention_weights.append(attention_info[1])

        graph_x = self.readout(x, batch)  # [batch_size, hidden_dim]

        # 直接使用图表示，不融合温度特征
        out = self.mlp(graph_x)

        if torch.isnan(out).any():
            raise ValueError("模型输出包含 NaN 值")

        if return_attention_weights:
            return out, all_attention_weights
        return out

    def get_graph_embedding(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)
        x = F.relu(x)

        for i, layer in enumerate(self.att_layers):
            if hasattr(layer, 'edge_dim') and layer.edge_dim is not None and i == 0:
                x = layer(x, edge_index, edge_attr=edge_attr)
            else:
                x = layer(x, edge_index)
            x = F.elu(x)

        graph_x = self.readout(x, batch)

        # 直接返回图表示，不融合温度特征
        return graph_x


class PocketGNNWithAttention(nn.Module):
    """
    GNN with Graph Attention Layers for Enzyme Kinetics (kcat, Km), with temperature as global input.
    """
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim=256, num_layers=6, heads=8, dropout=0.1, concat_heads=True):
        super().__init__()
        self.node_encoder = nn.Linear(node_input_dim, hidden_dim)

        self.att_layers = nn.ModuleList()
        current_dim = hidden_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            if concat_heads and not is_last:
                self.att_layers.append(
                    GATConv(current_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True, edge_dim=edge_input_dim if i==0 else None)
                )
                current_dim = hidden_dim
            else:
                self.att_layers.append(
                    GATConv(current_dim, hidden_dim, heads=heads, dropout=dropout, concat=False, edge_dim=edge_input_dim if i==0 else None)
                )
                current_dim = hidden_dim

        self.readout = global_mean_pool

        # 🔹 将 temperature 映射到 hidden_dim 再融合
        self.temp_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Sigmoid()  # 可换成 ReLU / Tanh 等
        )

        # MLP for regression
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # 输出 kcat, Km
        )

    def forward(self, data, return_attention_weights=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)
        x = F.relu(x)

        all_attention_weights = []
        for i, layer in enumerate(self.att_layers):
            if hasattr(layer, 'edge_dim') and layer.edge_dim is not None and i == 0:
                x_new, attention_info = layer(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            else:
                x_new, attention_info = layer(x, edge_index, return_attention_weights=True)
            x = F.elu(x_new)
            if return_attention_weights:
                all_attention_weights.append(attention_info[1])

        graph_x = self.readout(x, batch)  # [batch_size, hidden_dim]

        # 🔹 temperature 融合部分
        if hasattr(data, 'temperature'):
            ##
            # temp_QJD=torch.full((13,), 333.15, dtype=torch.float) ##临时使用！！！
            # temp = temp_QJD.view(-1, 1).to(graph_x.device)  #
##
            temp = data.temperature.view(-1, 1).to(graph_x.device)  # shape: [batch, 1]临时注释
            temp_embed = self.temp_mlp(temp)                        # shape: [batch, hidden_dim]
            graph_x = graph_x + temp_embed                          # 融合温度特征
        else:
            raise ValueError("❌ Data object 中缺少 temperature 字段")

        out = self.mlp(graph_x)

        if torch.isnan(out).any():
            raise ValueError("模型输出包含 NaN 值")

        if return_attention_weights:
            return out, all_attention_weights
        return out

    def get_graph_embedding(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)
        x = F.relu(x)

        for i, layer in enumerate(self.att_layers):
            if hasattr(layer, 'edge_dim') and layer.edge_dim is not None and i == 0:
                x = layer(x, edge_index, edge_attr=edge_attr)
            else:
                x = layer(x, edge_index)
            x = F.elu(x)

        graph_x = self.readout(x, batch)

        # ➕ 若需要返回融合后的图表示，也可加上 temp_embed
        if hasattr(data, 'temperature'):
            temp = data.temperature.view(-1, 1).to(graph_x.device)
            temp_embed = self.temp_mlp(temp)
            graph_x = graph_x + temp_embed

        return graph_x


class PocketGNNWithAttentionNoTemp(nn.Module):
    """
    GNN with Graph Attention Layers for Enzyme Kinetics (kcat, Km), WITHOUT temperature input.
    """
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim=256, num_layers=6, heads=8, dropout=0.1, concat_heads=True):
        super().__init__()
        self.node_encoder = nn.Linear(node_input_dim, hidden_dim)

        self.att_layers = nn.ModuleList()
        current_dim = hidden_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            if concat_heads and not is_last:
                self.att_layers.append(
                    GATConv(current_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True, edge_dim=edge_input_dim if i==0 else None)
                )
                current_dim = hidden_dim
            else:
                self.att_layers.append(
                    GATConv(current_dim, hidden_dim, heads=heads, dropout=dropout, concat=False, edge_dim=edge_input_dim if i==0 else None)
                )
                current_dim = hidden_dim

        self.readout = global_mean_pool

        # MLP for regression (no temperature module)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # 输出 kcat, Km
        )

    def forward(self, data, return_attention_weights=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)
        x = F.relu(x)

        all_attention_weights = []
        for i, layer in enumerate(self.att_layers):
            if hasattr(layer, 'edge_dim') and layer.edge_dim is not None and i == 0:
                x_new, attention_info = layer(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            else:
                x_new, attention_info = layer(x, edge_index, return_attention_weights=True)
            x = F.elu(x_new)
            if return_attention_weights:
                all_attention_weights.append(attention_info[1])

        graph_x = self.readout(x, batch)  # [batch_size, hidden_dim]

        # 直接进行回归，不融合温度特征
        out = self.mlp(graph_x)

        if torch.isnan(out).any():
            raise ValueError("模型输出包含 NaN 值")

        if return_attention_weights:
            return out, all_attention_weights
        return out

    def get_graph_embedding(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)
        x = F.relu(x)

        for i, layer in enumerate(self.att_layers):
            if hasattr(layer, 'edge_dim') and layer.edge_dim is not None and i == 0:
                x = layer(x, edge_index, edge_attr=edge_attr)
            else:
                x = layer(x, edge_index)
            x = F.elu(x)

        graph_x = self.readout(x, batch)
        return graph_x


class PocketGNNKcatOnly(nn.Module):
    """
    专门用于kcat预测的GNN模型，支持增强的边特征（24维）
    不包含温度特征，只预测kcat值
    """
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim=256, num_layers=6, heads=8, dropout=0.1, concat_heads=True):
        super().__init__()
        self.node_encoder = nn.Linear(node_input_dim, hidden_dim)

        self.att_layers = nn.ModuleList()
        current_dim = hidden_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            if concat_heads and not is_last:
                self.att_layers.append(
                    GATConv(current_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True, edge_dim=edge_input_dim if i==0 else None)
                )
                current_dim = hidden_dim
            else:
                self.att_layers.append(
                    GATConv(current_dim, hidden_dim, heads=heads, dropout=dropout, concat=False, edge_dim=edge_input_dim if i==0 else None)
                )
                current_dim = hidden_dim

        self.readout = global_mean_pool

        # MLP for kcat-only regression
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # 只输出 kcat
        )

    def forward(self, data, return_attention_weights=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)
        x = F.relu(x)

        all_attention_weights = []
        for i, layer in enumerate(self.att_layers):
            if hasattr(layer, 'edge_dim') and layer.edge_dim is not None and i == 0:
                x_new, attention_info = layer(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            else:
                x_new, attention_info = layer(x, edge_index, return_attention_weights=True)
            x = F.elu(x_new)
            if return_attention_weights:
                all_attention_weights.append(attention_info[1])

        graph_x = self.readout(x, batch)  # [batch_size, hidden_dim]

        # 直接进行kcat回归，不融合温度特征
        out = self.mlp(graph_x)  # [batch_size, 1]

        if torch.isnan(out).any():
            raise ValueError("模型输出包含 NaN 值")

        if return_attention_weights:
            return out, all_attention_weights
        return out

    def get_graph_embedding(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)
        x = F.relu(x)

        for i, layer in enumerate(self.att_layers):
            if hasattr(layer, 'edge_dim') and layer.edge_dim is not None and i == 0:
                x = layer(x, edge_index, edge_attr=edge_attr)
            else:
                x = layer(x, edge_index)
            x = F.elu(x)

        graph_x = self.readout(x, batch)
        return graph_x

