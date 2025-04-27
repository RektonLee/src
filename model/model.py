import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricBindingSiteEncoder(nn.Module):
    """
    Geometric binding site encoder that uses 3D coordinates and atom features
    to create a representation that is invariant to rotations and translations.
    """
    def __init__(self, node_feature_dim=7, hidden_dim=128, output_dim=256, num_layers=3):
        super().__init__()
        
        # Node feature embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.message_layers.append(EdgeConvLayer(hidden_dim, hidden_dim))
            
        # Final node embedding projection
        self.node_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Graph-level readout
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x, pos, edge_index, batch=None):
        """
        Args:
            x: Node features [num_nodes, node_feature_dim]
            pos: Node positions [num_nodes, 3]
            edge_index: Graph connectivity [2, num_edges]
            batch: Node batch indices [num_nodes]
        """
        # Initial node embedding
        node_feats = self.node_embedding(x)
        
        # Message passing
        for layer in self.message_layers:
            node_feats = layer(node_feats, pos, edge_index)
        
        # Final node-level projection
        node_feats = self.node_projection(node_feats)
        
        # Graph-level pooling
        if batch is None:
            # Single graph mode
            graph_feats = node_feats.mean(dim=0, keepdim=True)
        else:
            # Batch mode - aggregate features for each graph
            graph_feats = torch_scatter.scatter_mean(node_feats, batch, dim=0)
        
        # Final graph-level projection
        graph_feats = self.graph_projection(graph_feats)
        
        return graph_feats


class EdgeConvLayer(nn.Module):
    """
    Edge convolution layer that creates messages based on node features
    and their relative positions, making it invariant to rotations and translations.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Edge feature network
        self.edge_nn = nn.Sequential(
            nn.Linear(in_channels * 2 + 1, out_channels),
            nn.ReLU(),
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.LayerNorm(out_channels)
        )
        
        # Node update network
        self.node_nn = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.LayerNorm(out_channels)
        )
        
    def forward(self, x, pos, edge_index):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            pos: Node positions [num_nodes, 3]
            edge_index: Graph connectivity [2, num_edges]
        """
        src, dst = edge_index
        
        # Compute relative distances (invariant to translations)
        rel_pos = pos[dst] - pos[src]
        dist = torch.norm(rel_pos, dim=1, keepdim=True)  # [num_edges, 1]
        
        # Create edge features from source and destination nodes, plus distance
        edge_feat_src = x[src]
        edge_feat_dst = x[dst]
        edge_feats = torch.cat([edge_feat_src, edge_feat_dst, dist], dim=1)
        
        # Apply edge network
        messages = self.edge_nn(edge_feats)
        
        # Aggregate messages at destination nodes
        aggr_messages = torch_scatter.scatter_mean(messages, dst, dim=0, dim_size=x.size(0))
        
        # Update node features
        node_inputs = torch.cat([x, aggr_messages], dim=1)
        updated_x = self.node_nn(node_inputs)
        
        return updated_x


class SubstrateEncoder(nn.Module):
    """
    Encoder for substrate SMILES embeddings
    """
    def __init__(self, input_dim=384, output_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, substrate_features):
        return self.encoder(substrate_features)


class EnzymeKineticsModel(nn.Module):
    """
    Full model for enzyme kinetics prediction combining binding site and substrate information
    """
    def __init__(self, 
                 node_feature_dim=7, 
                 substrate_dim=384, 
                 hidden_dim=128, 
                 output_dim=256):
        super().__init__()
        
        self.binding_site_encoder = GeometricBindingSiteEncoder(
            node_feature_dim=node_feature_dim, 
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        self.substrate_encoder = SubstrateEncoder(
            input_dim=substrate_dim,
            output_dim=output_dim
        )
        
        # Fusion of binding site and substrate representations
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(output_dim // 2)
        )
        
        # Output heads for Km and kcat
        self.output_head = nn.Linear(output_dim // 2, 2)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def forward(self, binding_site_data, substrate_features):
        """
        Args:
            binding_site_data: Data object containing binding site information
                - x: Node features
                - pos: Node positions
                - edge_index: Graph connectivity
                - batch: Node batch indices
            substrate_features: Substrate embeddings [batch_size, substrate_dim]
        """
        # Extract components from binding site data
        x, pos, edge_index, batch = binding_site_data.x, binding_site_data.pos, binding_site_data.edge_index, binding_site_data.batch
        
        # Encode binding site
        binding_site_repr = self.binding_site_encoder(x, pos, edge_index, batch)
        
        # Encode substrate
        substrate_repr = self.substrate_encoder(substrate_features)
        
        # Fusion
        joint_repr = torch.cat([binding_site_repr, substrate_repr], dim=1)
        fused = self.fusion(joint_repr)
        
        # Final prediction
        output = self.output_head(fused)
        
        return output