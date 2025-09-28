# =============================
# enzyme_kinetics_gnn_v1.2
# 使用 Gaussian RBF 增强边特征 & 加入标准化 temperature
# 2025-05
# =============================

import os
import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from tqdm import tqdm
import logging

# 尝试不同的导入方式
try:
    from torch_cluster import radius_graph
except ImportError:
    try:
        from torch_geometric.nn import radius_graph
    except ImportError:
        radius_graph = None

# ==== 配置参数 ====
CSV_PATH    = '/home/lizihao/Work/enzyme_prediction/src/simple2/Example_DLKcat_S.csv'
POCKET_DIR  = 'data/processed/pockets'
SAVE_PATH   = 'data/processed/dataset_DLKcat_S.pt'
DIST_CUTOFF = 4.0    # Å
RBF_CENTERS = 16     # Gaussian RBF 的中心数量
RBF_DMIN    = 0.0
RBF_DMAX    = 8.0
RBF_GAMMA   = 20.0   # Gaussian 带宽参数

# ==== Gaussian RBF 扩展函数 ====
def gaussian_rbf(edge_dist, num_centers=RBF_CENTERS, D_min=RBF_DMIN, D_max=RBF_DMAX, gamma=RBF_GAMMA):
    """
    将距离张量 [E,1] 映射为高斯 RBF 特征 [E, num_centers]
    """
    # 构造中心
    centers = torch.linspace(D_min, D_max, num_centers, device=edge_dist.device)  # [C]
    diff = edge_dist - centers.view(1, -1)  # [E, C]
    return torch.exp(-gamma * diff**2)      # [E, C]

# ==== 手动构建边的函数（不依赖torch-cluster） ====
def build_edges_manual(pos, cutoff):
    """
    手动构建半径图，不依赖torch-cluster
    pos: [N, 3] 原子坐标
    cutoff: 距离阈值
    返回: [2, E] 边索引
    """
    n_atoms = pos.shape[0]
    edges = []
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = torch.norm(pos[i] - pos[j])
            if dist <= cutoff:
                # 添加双向边（无向图）
                edges.append([i, j])
                edges.append([j, i])
    
    if len(edges) == 0:
        # 如果没有边，至少添加自环以保证图的连通性
        edges = [[i, i] for i in range(n_atoms)]
    
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

# ==== 原子特征编码器 ====
element_list = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']
residue_list = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','LIG']

element_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
element_encoder.fit(np.array([[e] for e in element_list]))
residue_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
residue_encoder.fit(np.array([[r] for r in residue_list]))

atomic_property_table = {
    'H':  {'Z': 1, 'mass': 1.008,  'electronegativity': 2.20, 'radius': 0.31},
    'C':  {'Z': 6, 'mass': 12.011, 'electronegativity': 2.55, 'radius': 0.76},
    'N':  {'Z': 7, 'mass': 14.007, 'electronegativity': 3.04, 'radius': 0.71},
    'O':  {'Z': 8, 'mass': 15.999, 'electronegativity': 3.44, 'radius': 0.66},
    'S':  {'Z':16, 'mass': 32.06,  'electronegativity': 2.58, 'radius': 1.05},
    'P':  {'Z':15, 'mass': 30.974, 'electronegativity': 2.19, 'radius': 1.07},
    'F':  {'Z': 9, 'mass': 18.998, 'electronegativity': 3.98, 'radius': 0.57},
    'Cl': {'Z':17, 'mass': 35.45,  'electronegativity': 3.16, 'radius': 1.02},
    'Br': {'Z':35, 'mass': 79.904, 'electronegativity': 2.96, 'radius': 1.20},
    'I':  {'Z':53, 'mass': 126.90, 'electronegativity': 2.66, 'radius': 1.39},
}

def get_elec_feature(max_atomic_number=20):
    return torch.tensor([
        [0]*16,
        [0,1]+[0]*14,
        [2,0]+[0]*14,
        [2,0,0,1]+[0]*12,
        [2,0,2,0]+[0]*12,
        [2,0,2,0,0,1]+[0]*10,
        [2,0,2,0,0,2]+[0]*10,
        [2,0,2,0,0,3]+[0]*10,
        [2,0,2,0,2,2]+[0]*10,
        [2,0,2,0,4,1]+[0]*10,
        [2,0,2,0,6,0]+[0]*10,
    ], dtype=torch.float32).numpy()
atomic_electronic_features = get_elec_feature()

# ==== 解析 PDB 口袋 ====
def parse_pocket(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pocket', pdb_path)
    atoms = []
    for atom in structure.get_atoms():
        if atom.element != 'H':
            res = atom.get_parent().get_resname()
            chain = atom.get_parent().get_full_id()[2]
            is_ligand = 1 if (res=='UNL' or chain==' ') else 0
            atoms.append({
                'coord': atom.coord,
                'element': atom.element,
                'residue': res if res in residue_list else 'LIG',
                'is_ligand': is_ligand
            })
    return atoms

# ==== 构图函数 ====
def build_graph(atoms, temperature):
    # 处理空原子列表的情况
    if len(atoms) == 0:
        raise ValueError("无法从空的原子列表构建图数据")
    
    coords   = np.array([a['coord'] for a in atoms])
    elements = np.array([[a['element']] for a in atoms])
    residues = np.array([[a['residue']] for a in atoms])
    is_lig   = np.array([[a['is_ligand']] for a in atoms])

    el_feat  = element_encoder.transform(elements)
    res_feat = residue_encoder.transform(residues)

    # 最近邻距离
    dmat = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    min_dists = np.min(dmat, axis=1, keepdims=True)

    nums = np.array([{'C':6,'N':7,'O':8,'S':16,'P':15,'F':9,'Cl':17,'Br':35,'I':53,'H':1}.get(e[0],0) for e in elements])
    elec = atomic_electronic_features[np.clip(nums,0,10)]

    props = np.array([
        [
            atomic_property_table.get(e[0],{'mass':0,'electronegativity':0,'radius':0})['mass'],
            atomic_property_table.get(e[0],{'mass':0,'electronegativity':0,'radius':0})['electronegativity'],
            atomic_property_table.get(e[0],{'mass':0,'electronegativity':0,'radius':0})['radius'],
        ]
        for e in elements
    ])

    x = np.hstack([el_feat, res_feat, is_lig, min_dists, elec, props])
    if np.isnan(x).any():
        raise ValueError("节点特征包含 NaN")

    pos = torch.tensor(coords, dtype=torch.float)
    
    # 使用radius_graph构建边，如果不可用则使用替代方法
    if radius_graph is not None:
        try:
            edge_index = radius_graph(pos, r=DIST_CUTOFF)
        except Exception as e:
            logging.warning(f"radius_graph调用失败: {e}，使用替代方法构建图边")
            edge_index = build_edges_manual(pos, DIST_CUTOFF)
    else:
        logging.warning("radius_graph不可用，使用替代方法构建图边")
        edge_index = build_edges_manual(pos, DIST_CUTOFF)
    
    row, col   = edge_index
    dists      = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)  # [E,1]

    # —— Gaussian RBF 扩展 —— 
    edge_attr = gaussian_rbf(dists)  # [E, RBF_CENTERS]

    if torch.isnan(edge_attr).any():
        raise ValueError("边特征包含 NaN")

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    # 标准化 temperature，并作为全局标量
    temp_scaled = (temperature - 303.15) / 10.0
    data.temperature = torch.tensor([temp_scaled], dtype=torch.float)

    return data

# ==== 主流程 ====
if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)
    dataset = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        uniprot     = row['uniprot']
        smiles      = row['substrate_smiles'].split(';')[0]
        temperature = row['temperature']
        import hashlib
        pid_hash    = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff
        pdb_file    = f"{uniprot}_{pid_hash}_10A.pdb"
        pdb_path    = os.path.join(POCKET_DIR, pdb_file)

        if not os.path.exists(pdb_path):
            continue
        atoms = parse_pocket(pdb_path)
        if len(atoms) < 3:
            continue

        kcat = row.get('kcat Wildtype', np.nan)
        km   = row.get('Km Wildtype', np.nan)
        if not np.isfinite(kcat) or not np.isfinite(km) or kcat <= 0 or km <= 0:
            continue

        data = build_graph(atoms, temperature)
        data.y = torch.log10(torch.tensor([kcat, km], dtype=torch.float))
        data.pdb_id = pdb_file
        dataset.append(data)

    torch.save(dataset, SAVE_PATH)
    print(f"✅ Saved {len(dataset)} samples to {SAVE_PATH}")
