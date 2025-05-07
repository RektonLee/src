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


class PocketGNNWithAttention(nn.Module):
    """
    GNN with Graph Attention Layers for Enzyme Kinetics (kcat, Km)
    """
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim=256, num_layers=6, heads=8, dropout=0.1, concat_heads=True):
        super().__init__()
        self.node_encoder = nn.Linear(node_input_dim, hidden_dim)
        # 如果GATConv也需要边特征，可以自定义GATConv或寻找支持边特征的变体
        # PyG的GATConv默认不直接使用edge_attr进行注意力计算，但可以在消息传递中加入
        # 这里我们先用标准的GATConv，如果需要严格的边特征参与注意力，需要进一步定制

        self.att_layers = nn.ModuleList()
        current_dim = hidden_dim
        for i in range(num_layers):
            # 如果 concat_heads 为 True，输出维度是 hidden_dim * heads
            # 如果为 False (通常在最后一层或中间层后接线性变换)，输出维度是 hidden_dim
            # 这里我们让每一层的输出维度保持为 hidden_dim (如果是多头拼接，之后需要一个线性层降维)
            # 或者，我们可以让GATConv的out_channels = hidden_dim // heads，然后concat后维度还是hidden_dim

            # 方案1: 输出维度 hidden_dim * heads，然后用线性层降维 (更灵活)
            # self.att_layers.append(
            #     GATConv(current_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
            # )
            # current_dim = hidden_dim * heads # 更新下一层的输入维度
            # # 如果需要保持hidden_dim，可以在每层GATConv后加一个线性层
            # self.add_module(f"lin_after_gat_{i}", nn.Linear(hidden_dim * heads, hidden_dim))

            # 方案2: GATConv直接输出hidden_dim (通过调整out_channels和concat)
            # 最后一层GAT通常concat=False 或者 heads=1 (或平均)
            is_last_gat_layer = (i == num_layers - 1)
            if concat_heads and not is_last_gat_layer: # 中间层多头拼接
                self.att_layers.append(
                    GATConv(current_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True, edge_dim=edge_input_dim if i==0 else None) # 假设第一层可以接受edge_dim
                )
                current_dim = hidden_dim # 因为 hidden_dim // heads * heads == hidden_dim
            else: # 最后一层GAT，或者不拼接头 (此时通常会对头的结果取平均)
                 self.att_layers.append(
                    GATConv(current_dim, hidden_dim, heads=heads, dropout=dropout, concat=False, edge_dim=edge_input_dim if i==0 else None) # concat=False，输出维度是hidden_dim
                )
                 current_dim = hidden_dim


        self.readout = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(current_dim, hidden_dim), #确保这里的current_dim与GAT层输出一致
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Predict [kcat, Km]
        )

        # 权重初始化 (可选，但通常有益)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, GATConv):
                # GATConv内部有自己的初始化，但也可以覆盖
                pass


    def forward(self, data, return_attention_weights=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.node_encoder(x)
        x = F.relu(x) # 通常在编码后加激活

        all_attention_weights = []

        for i, layer in enumerate(self.att_layers):
            # GATConv的forward方法可以返回注意力权重
            # x_new, attention_weights = layer(x, edge_index, return_attention_weights=True)
            # 如果GATConv层支持edge_dim, 需要传递进去
            if hasattr(layer, 'edge_dim') and layer.edge_dim is not None and i==0 : # 假设只有第一层用原始edge_attr
                 x_new, attention_info = layer(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            else:
                 x_new, attention_info = layer(x, edge_index, return_attention_weights=True)

            x = x_new # 更新节点表示
            x = F.elu(x) # GAT论文中常用elu作为激活函数
            # x = F.dropout(x, p=self.mlp[2].p if hasattr(self.mlp[2], 'p') else 0.1, training=self.training) # 在GAT层之间也可以加dropout

            if return_attention_weights:
                # attention_info 是一个元组 (edge_index_with_attention, attention_weights_per_head)
                # attention_weights_per_head 的形状是 [num_edges, num_heads]
                all_attention_weights.append(attention_info[1])

            # 如果之前GATConv的输出是 current_dim * heads (concat=True)，且需要保持hidden_dim
            # lin_layer = getattr(self, f"lin_after_gat_{i}", None)
            # if lin_layer:
            #     x = lin_layer(x)
            #     x = F.elu(x) # 再次激活

        # 图级别表示
        graph_x = self.readout(x, batch)

        # MLP回归
        out = self.mlp(graph_x)

        if torch.isnan(out).any():
            raise ValueError("模型输出包含 NaN 值")

        if return_attention_weights:
            return out, all_attention_weights
        else:
            return out

    def get_graph_embedding(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)
        x = F.relu(x)

        for i, layer in enumerate(self.att_layers):
            if hasattr(layer, 'edge_dim') and layer.edge_dim is not None and i==0:
                 x = layer(x, edge_index, edge_attr=edge_attr)
            else:
                 x = layer(x, edge_index)
            x = F.elu(x)
            # lin_layer = getattr(self, f"lin_after_gat_{i}", None)
            # if lin_layer:
            #     x = lin_layer(x)
            #     x = F.elu(x)

        graph_x = self.readout(x, batch)
        return graph_x

    # 如果需要获取节点级别的嵌入（池化前）
    def get_node_embeddings_before_pool(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)
        x = F.relu(x)

        for i, layer in enumerate(self.att_layers):
            if hasattr(layer, 'edge_dim') and layer.edge_dim is not None and i==0:
                 x = layer(x, edge_index, edge_attr=edge_attr)
            else:
                 x = layer(x, edge_index)
            x = F.elu(x)
            # lin_layer = getattr(self, f"lin_after_gat_{i}", None)
            # if lin_layer:
            #     x = lin_layer(x)
            #     x = F.elu(x)
        return x # 返回节点级嵌入，不进行池化