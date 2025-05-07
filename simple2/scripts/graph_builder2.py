
# =============================
# enzyme_kinetics_gnn_v1.1
# 纯硬编码原子属性版：去除pqr依赖
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

# ==== 配置参数 ====
CSV_PATH = '/home/lizihao/Work/enzyme_prediction/data/cleaned_data.csv'
POCKET_DIR = 'data/processed/pockets'
SAVE_PATH = 'data/processed/dataset_NAN_nopqr_test.pt'
DIST_CUTOFF = 3.0  # 距离阈值（Å）

# ==== 编码器 ====
element_list = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']
residue_list = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','LIG']

element_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
element_encoder.fit(np.array([[e] for e in element_list]))

residue_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
residue_encoder.fit(np.array([[r] for r in residue_list]))

# ==== 原子属性表 ====
atomic_property_table = {
    'H':  {'Z': 1, 'mass': 1.008, 'electronegativity': 2.20, 'radius': 0.31},
    'C':  {'Z': 6, 'mass': 12.011, 'electronegativity': 2.55, 'radius': 0.76},
    'N':  {'Z': 7, 'mass': 14.007, 'electronegativity': 3.04, 'radius': 0.71},
    'O':  {'Z': 8, 'mass': 15.999, 'electronegativity': 3.44, 'radius': 0.66},
    'S':  {'Z': 16,'mass': 32.06, 'electronegativity': 2.58, 'radius': 1.05},
    'P':  {'Z': 15,'mass': 30.974, 'electronegativity': 2.19, 'radius': 1.07},
    'F':  {'Z': 9, 'mass': 18.998, 'electronegativity': 3.98, 'radius': 0.57},
    'Cl': {'Z':17, 'mass': 35.45, 'electronegativity': 3.16, 'radius': 1.02},
    'Br': {'Z':35, 'mass': 79.904, 'electronegativity': 2.96, 'radius': 1.20},
    'I':  {'Z':53, 'mass': 126.90, 'electronegativity': 2.66, 'radius': 1.39},
}

# ==== 价电子填充态 ====
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
def build_graph(atoms):
    coords = np.array([atom['coord'] for atom in atoms])
    elements = np.array([[atom['element']] for atom in atoms])
    residues = np.array([[atom['residue']] for atom in atoms])
    is_ligand = np.array([[atom['is_ligand']] for atom in atoms])

    element_features = element_encoder.transform(elements)
    residue_features = residue_encoder.transform(residues)
    dmat = np.linalg.norm(coords[:,None,:]-coords[None,:,:], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    min_dists = np.min(dmat, axis=1, keepdims=True)

    atomic_nums = np.array([{'C':6,'N':7,'O':8,'S':16,'P':15,'F':9,'Cl':17,'Br':35,'I':53,'H':1}.get(e[0],0) for e in elements])
    electronic_features = atomic_electronic_features[np.clip(atomic_nums,0,10)]

    props = np.array([
        [
            atomic_property_table.get(e[0], {'mass':0, 'electronegativity':0, 'radius':0})['mass'],
            atomic_property_table.get(e[0], {'mass':0, 'electronegativity':0, 'radius':0})['electronegativity'],
            atomic_property_table.get(e[0], {'mass':0, 'electronegativity':0, 'radius':0})['radius'],
        ]
        for e in elements
    ])

    node_features = np.hstack([element_features, residue_features, is_ligand, min_dists, electronic_features, props])
    
    # 检查节点特征是否包含 NaN
    if np.isnan(node_features).any():
        raise ValueError("节点特征包含 NaN 值")
    
    pos = torch.tensor(coords, dtype=torch.float)
    edge_index = radius_graph(pos, r=DIST_CUTOFF)

    # 计算边属性（例如距离）
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], dim=1).unsqueeze(1)  # [num_edges, 1]

    # 检查边属性是否包含 NaN
    if torch.isnan(edge_attr).any():
        raise ValueError("边属性包含 NaN 值")
    
    # 构建图数据
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,  # 添加边属性
        temp=200,
        pH=7
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

        # 过滤无效或非法值
        if not np.isfinite(kcat) or not np.isfinite(km):
            continue
        if kcat <= 0 or km <= 0:
            continue

        data = build_graph(atoms)
        label = torch.log10(torch.tensor([kcat, km], dtype=torch.float))
        if not np.all(np.isfinite(label.numpy())):
         raise ValueError(f"标签非法值: {label}")

        data.y = label
        data.pdb_id = pdb_file
        dataset.append(data)

    torch.save(dataset, SAVE_PATH)
    print(f"✅ Saved {len(dataset)} samples to {SAVE_PATH}")
