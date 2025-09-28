import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, MessagePassing, GatedGraphConv, GATConv
import torch_geometric.utils as utils

class PocketGNNKcatEnhanced(nn.Module):
    """
    增强版kcat预测GNN模型
    优化点：
    1. 残差连接
    2. 层归一化
    3. 多尺度池化
    4. 改进的MLP
    5. 边特征在多层中使用
    """
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim=256, num_layers=6, heads=8, dropout=0.1, concat_heads=True):
        super().__init__()
        
        # 节点编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 边特征编码器（用于在多层中使用边特征）
        self.edge_encoder = nn.Linear(edge_input_dim, hidden_dim // 4)
        
        # GAT层 + 残差连接
        self.att_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        current_dim = hidden_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            
            if concat_heads and not is_last:
                self.att_layers.append(
                    GATConv(current_dim, hidden_dim // heads, heads=heads, 
                           dropout=dropout, concat=True, edge_dim=hidden_dim // 4)
                )
                current_dim = hidden_dim
            else:
                self.att_layers.append(
                    GATConv(current_dim, hidden_dim, heads=heads, 
                           dropout=dropout, concat=False, edge_dim=hidden_dim // 4)
                )
                current_dim = hidden_dim
            
            # 层归一化
            self.layer_norms.append(nn.LayerNorm(current_dim))

        # 多尺度池化
        self.readout_mean = global_mean_pool
        self.readout_max = global_max_pool
        self.readout_sum = global_add_pool
        
        # 池化特征融合
        self.pool_fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # 改进的MLP - 更深且有跳跃连接
        self.mlp = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Linear(hidden_dim // 4, 1)
        ])
        
        self.mlp_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim),
            nn.LayerNorm(hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 4)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # 跳跃连接的投影层
        self.skip_proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.skip_proj2 = nn.Linear(hidden_dim, hidden_dim // 4)

    def forward(self, data, return_attention_weights=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 节点编码
        x = self.node_encoder(x)
        
        # 边特征编码（在所有层中使用）
        edge_attr_encoded = self.edge_encoder(edge_attr)
        
        # GAT层 + 残差连接
        all_attention_weights = []
        
        for i, (layer, norm) in enumerate(zip(self.att_layers, self.layer_norms)):
            residual = x
            
            # GAT层
            if return_attention_weights:
                x_new, attention_info = layer(x, edge_index, edge_attr=edge_attr_encoded, return_attention_weights=True)
                all_attention_weights.append(attention_info[1])
            else:
                x_new = layer(x, edge_index, edge_attr=edge_attr_encoded)
            
            # 残差连接 + 层归一化
            if residual.shape == x_new.shape:
                x = norm(x_new + residual)
            else:
                x = norm(x_new)
            
            x = F.elu(x)

        # 多尺度池化
        graph_mean = self.readout_mean(x, batch)
        graph_max = self.readout_max(x, batch)
        graph_sum = self.readout_sum(x, batch)
        
        # 融合多尺度特征
        graph_x = torch.cat([graph_mean, graph_max, graph_sum], dim=1)
        graph_x = self.pool_fusion(graph_x)
        graph_x = F.elu(graph_x)

        # 改进的MLP with跳跃连接
        mlp_input = graph_x
        
        # 第一层
        out = self.mlp[0](mlp_input)
        out = self.mlp_norms[0](out)
        out = F.elu(out)
        out = self.dropout(out)
        
        # 第二层 + 跳跃连接
        skip1 = self.skip_proj1(mlp_input)
        out = self.mlp[1](out) + skip1
        out = self.mlp_norms[1](out)
        out = F.elu(out)
        out = self.dropout(out)
        
        # 第三层 + 跳跃连接
        skip2 = self.skip_proj2(mlp_input)
        out = self.mlp[2](out) + skip2
        out = self.mlp_norms[2](out)
        out = F.elu(out)
        out = self.dropout(out)
        
        # 输出层
        out = self.mlp[3](out)

        if torch.isnan(out).any():
            raise ValueError("模型输出包含 NaN 值")

        if return_attention_weights:
            return out, all_attention_weights
        return out

    def get_graph_embedding(self, data):
        """获取图嵌入表示"""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 节点编码
        x = self.node_encoder(x)
        
        # 边特征编码
        edge_attr_encoded = self.edge_encoder(edge_attr)
        
        # GAT层
        for i, (layer, norm) in enumerate(zip(self.att_layers, self.layer_norms)):
            residual = x
            x_new = layer(x, edge_index, edge_attr=edge_attr_encoded)
            
            if residual.shape == x_new.shape:
                x = norm(x_new + residual)
            else:
                x = norm(x_new)
            x = F.elu(x)

        # 多尺度池化
        graph_mean = self.readout_mean(x, batch)
        graph_max = self.readout_max(x, batch)
        graph_sum = self.readout_sum(x, batch)
        
        # 融合多尺度特征
        graph_x = torch.cat([graph_mean, graph_max, graph_sum], dim=1)
        graph_x = self.pool_fusion(graph_x)
        
        return graph_x


class PocketGNNKcatUltra(nn.Module):
    """
    Ultra版本：集成更多先进技术
    1. 多头自注意力
    2. 图Transformer
    3. 虚拟节点
    4. 边更新机制
    """
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim=256, num_layers=6, heads=8, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        
        # 节点和边编码器
        self.node_encoder = nn.Linear(node_input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_input_dim, hidden_dim)
        
        # 虚拟节点（全局信息聚合）
        self.virtual_node_emb = nn.Parameter(torch.randn(hidden_dim))
        self.virtual_node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph Transformer层
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 编码
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # 虚拟节点初始化
        batch_size = batch.max().item() + 1
        virtual_nodes = self.virtual_node_emb.unsqueeze(0).repeat(batch_size, 1)
        
        # Transformer层
        for layer in self.transformer_layers:
            x, virtual_nodes = layer(x, edge_index, edge_attr, batch, virtual_nodes)
        
        # 使用虚拟节点作为图表示
        graph_x = virtual_nodes
        
        # 预测
        out = self.predictor(graph_x)
        
        return out


class GraphTransformerLayer(nn.Module):
    """Graph Transformer层"""
    def __init__(self, hidden_dim, heads, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        
        # Feed Forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch, virtual_nodes):
        # 简化的实现，实际中需要更复杂的图注意力机制
        residual = x
        x = self.norm1(x + self.dropout(x))  # 残差连接
        x = self.norm2(x + self.dropout(self.ffn(x)))  # FFN
        
        # 虚拟节点更新（简化）
        batch_size = virtual_nodes.shape[0]
        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() > 0:
                node_features = x[mask]
                # 聚合节点特征到虚拟节点
                virtual_nodes[i] = virtual_nodes[i] + node_features.mean(dim=0)
        
        return x, virtual_nodes
