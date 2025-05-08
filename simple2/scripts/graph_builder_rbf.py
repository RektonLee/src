# graph_builder2.py
# =============================
# enzyme_kinetics_gnn_v1.3
# 融合方向信息，加入温度，Embedding原子和残基类型
# 2025-05
# =============================

import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from Bio.PDB import PDBParser
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from tqdm import tqdm
import torch.nn as nn

# ==== 配置参数 ====
CSV_PATH = '/home/lizihao/Work/enzyme_prediction/data/cleaned_data.csv'
POCKET_DIR = 'data/processed/pockets'
SAVE_PATH = 'data/processed/dataset_rbf.pt' # 修改了保存路径
DIST_CUTOFF = 4.0  # 距离阈值（Å）
NUM_RBF = 10 # RBF函数的数量
ATOM_EMBED_DIM = 32  # 原子类型embedding维度
RESIDUE_EMBED_DIM = 32  # 残基类型embedding维度

# ==== 编码器 ====
element_list = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']
residue_list = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','LIG']

# ==== RBF 函数 ====
def sinc_expansion(edge_dist, num_rbf, cutoff):
    """
    Calculate sinc radial basis function.
    """
    n = torch.arange(1, num_rbf + 1, device=edge_dist.device).float()
    r_scaled = edge_dist.unsqueeze(-1) / cutoff
    return torch.sin(n * torch.pi * r_scaled) / r_scaled

# ==== 解析pdb ====
def parse_pocket(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pocket', pdb_path)
    atoms = []
    for atom in structure.get_atoms():
        if atom.element != 'H':
            resname = atom.get_parent().get_resname()
            chain_id = atom.get_parent().get_full_id()[2]
            is_ligand = 1 if resname == 'UNL' or chain_id == ' ' else 0
            atoms.append({
                'coord': atom.coord,
                'element': atom.element,
                'residue': resname if resname in residue_list else 'LIG',
                'is_ligand': is_ligand
            })
    return atoms

# ==== 构图 ====
def build_graph(atoms, temperature):
    coords = np.array([atom['coord'] for atom in atoms])
    elements = np.array([atom['element'] for atom in atoms])
    residues = np.array([atom['residue'] for atom in atoms])
    is_ligand = np.array([[atom['is_ligand']] for atom in atoms])

    # 生成原子和残基的索引
    element_indices = [element_list.index(e) if e in element_list else 0 for e in elements]
    residue_indices = [residue_list.index(r) if r in residue_list else 0 for r in residues]
    
    element_indices = torch.tensor(element_indices, dtype=torch.long)
    residue_indices = torch.tensor(residue_indices, dtype=torch.long)
    is_ligand = torch.tensor(is_ligand, dtype=torch.float32)
    
    # 计算其他特征
    dmat = np.linalg.norm(coords[:,None,:]-coords[None,:,:], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    min_dists = torch.tensor(np.min(dmat, axis=1, keepdims=True), dtype=torch.float32)
    
    # 温度特征
    temperature_tensor = torch.tensor([temperature] * len(atoms), dtype=torch.float).unsqueeze(1)
    
    # 构建节点特征 (不包括原子和残基类型)
    node_features = torch.cat([
        is_ligand,
        min_dists,
        temperature_tensor
    ], dim=-1)
    
    # 检查节点特征是否包含 NaN
    if torch.isnan(node_features).any():
        raise ValueError("节点特征包含 NaN 值")
    
    pos = torch.tensor(coords, dtype=torch.float)
    edge_index = radius_graph(pos, r=DIST_CUTOFF)
    row, col = edge_index
    distances = torch.norm(pos[row] - pos[col], dim=1)
    directions = (pos[col] - pos[row]) / (distances.unsqueeze(-1) + 1e-6)
    rbf_features = sinc_expansion(distances, NUM_RBF, DIST_CUTOFF)
    edge_attr = torch.cat([rbf_features, directions], dim=-1)

    # 构建图数据
    data = Data(
        x=node_features,  # 基础节点特征
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        element_indices=element_indices,  # 原子类型索引
        residue_indices=residue_indices,  # 残基类型索引
        temperature=temperature_tensor
    )
    return data



# ==== 主函数 ====
if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)
    dataset = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        uniprot = row['uniprot']
        smiles = row['substrate_smiles'].split(';')[0]
        import hashlib
        pid_hash = int(hashlib.sha256(smiles.encode()).hexdigest(),16) & 0xffff
        pdb_file = f"{uniprot}_{pid_hash}_10A.pdb"
        pdb_path = os.path.join(POCKET_DIR, pdb_file)

        if not os.path.exists(pdb_path):
            continue

        atoms = parse_pocket(pdb_path)
        if len(atoms)<3:
            continue

        kcat = row.get('kcat Wildtype', np.nan)
        km = row.get('Km Wildtype', np.nan)
        temperature = row.get('temperature', 298.0)  # 默认温度 298K

        # 过滤无效或非法值
        if not np.isfinite(kcat) or not np.isfinite(km):
            continue
        if kcat <= 0 or km <= 0:
            continue

        data = build_graph(atoms, temperature) # 加入 temperature
        label = torch.log10(torch.tensor([kcat, km], dtype=torch.float))
        if not np.all(np.isfinite(label.numpy())):
         raise ValueError(f"标签非法值: {label}")

        data.y = label
        data.pdb_id = pdb_file
        dataset.append(data)

    torch.save(dataset, SAVE_PATH)
    print(f"✅ Saved {len(dataset)} samples to {SAVE_PATH}")

