"""
基于PDB文件的GNN训练流程
包含活性口袋处理、等变图网络、数据关联等功能
"""
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model import EquivariantGNN

# 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_pdb(uniprot_id):
    """处理单个PDB文件，返回图数据结构"""
    pdb_path = f"/home/lizihao/Work/enzyme_prediction/src/output/pdb_files/{uniprot_id}.pdb"
    if not os.path.exists(pdb_path):
        return None
    
    # 活性位点识别（假设已实现）
    atoms = identify_binding_site(pdb_path)  # <mcsymbol name="identify_binding_site" filename="pdb_utils.py" path="src/utils/pdb_utils.py" startline="42" type="function"></mcsymbol>
    
    # 构建图数据
    node_features = []
    positions = []
    for atom in atoms:
        # 特征包含原子类型、氨基酸类型、电荷等
        features = [
            atom['type_idx'],
            atom['residue_idx'],
            atom['charge']
        ]
        node_features.append(features)
        positions.append(atom['position'])
    
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        pos=torch.tensor(positions, dtype=torch.float),
        uniprot=uniprot_id
    )

def load_data():
    """加载所有数据"""
    # 加载CSV数据
    df = pd.read_csv('cleaned_data.csv')
    
    # 加载预处理的底物嵌入
    data = np.load('pro_data.npz')
    substrate_embeds = torch.tensor(data['substrate_embeddings'])
    
    # 处理蛋白质结构
    graphs = []
    labels = []
    substrate_indices = []
    for idx, row in df.iterrows():
        graph = process_pdb(row['uniprot_id'])
        if graph is None:
            continue
            
        # 添加底物嵌入索引
        graph.substrate_idx = idx
        graphs.append(graph)
        labels.append(data['y'][idx])
        substrate_indices.append(idx)
    
    return graphs, torch.tensor(labels), substrate_embeds, substrate_indices

class FusionModel(nn.Module):
    """融合蛋白质GNN和底物嵌入的模型"""
    def __init__(self):
        super().__init__()
        self.gnn = EquivariantGNN()  # <mcsymbol name="EquivariantGNN" filename="model.py" path="src/model/model.py" startline="15" type="class"></mcsymbol>
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 512),  # 假设GNN输出256维，底物嵌入128维
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    
    def forward(self, protein_data, substrate_embeds):
        protein_feat = self.gnn(protein_data)
        substrate_feat = substrate_embeds[protein_data.substrate_idx]
        combined = torch.cat([protein_feat, substrate_feat], dim=-1)
        return self.fusion(combined)

def train():
    """训练主流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    logger.info("Loading data...")
    graphs, labels, substrate_embeds, indices = load_data()
    
    # 保存处理后的图数据
    torch.save({
        'graphs': graphs,
        'indices': indices
    }, 'processed_graphs.pt')
    
    # 创建数据加载器
    loader = DataLoader(graphs, batch_size=32, shuffle=True)
    
    # 初始化模型
    model = FusionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 训练循环
    logger.info("Starting training...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            outputs = model(batch, substrate_embeds.to(device))
            loss = criterion(outputs, labels[batch.substrate_idx].unsqueeze(1).to(device))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        logger.info(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'fusion_model.pt')

if __name__ == '__main__':
    train()