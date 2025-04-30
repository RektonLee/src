# =============================
# enzyme_kinetics_gnn_v1.0
# 基础版：物理属性增强口袋图构建 + GNN建模
# 2025-04
# =============================

import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from Bio.PDB import PDBParser
from biopandas.pdb import PandasPdb
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

# ==== 配置参数 ====
CSV_PATH = '/home/lizihao/Work/enzyme_prediction/data/cleaned_data.csv'
POCKET_DIR = 'data/processed/pockets'
SAVE_PATH = 'data/processed/dataset_v1.pt'
DIST_CUTOFF = 5.0  # 距离阈值（Å）

# ==== 节点特征编码器 ====
element_list = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']
residue_list = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','LIG']

element_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
element_encoder.fit(np.array([[e] for e in element_list]))

residue_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
residue_encoder.fit(np.array([[r] for r in residue_list]))

# ==== 电子结构特征（价电子填充态） ====
from torch import tensor, float32

def get_elec_feature(max_atomic_number=20):
    elec_feature = tensor([
        [0]*16,  # 0 None
        [0,1]+[0]*14,   # H
        [2,0]+[0]*14,   # He
        [2,0,0,1]+[0]*12,  # Li
        [2,0,2,0]+[0]*12,  # Be
        [2,0,2,0,0,1]+[0]*10,  # B
        [2,0,2,0,0,2]+[0]*10,  # C
        [2,0,2,0,0,3]+[0]*10,  # N
        [2,0,2,0,2,2]+[0]*10,  # O
        [2,0,2,0,4,1]+[0]*10,  # F
        [2,0,2,0,6,0]+[0]*10,  # Ne
    ], dtype=float32)
    return elec_feature

atomic_electronic_features = get_elec_feature().numpy()  # shape (11,16)

# ==== 主函数 ====
def parse_pocket(pdb_path, pqr_path):
    # 解析元素与残基信息
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
    # 解析电荷
    ppdb = PandasPdb().read_pdb(pqr_path)
    charges = ppdb.df['ATOM']['charge'].values
    return atoms, charges

# ==== Build graph ====
def build_graph(atoms, charges):
    coords = np.array([atom['coord'] for atom in atoms])
    elements = np.array([[atom['element']] for atom in atoms])
    residues = np.array([[atom['residue']] for atom in atoms])
    is_ligand = np.array([[atom['is_ligand']] for atom in atoms])

    element_features = element_encoder.transform(elements)
    residue_features = residue_encoder.transform(residues)
    min_dists = np.min(np.linalg.norm(coords[:,None,:]-coords[None,:,:], axis=-1) + np.eye(len(coords))*1e6, axis=1, keepdims=True)
    
    # 电子结构
    atomic_nums = np.array([{'C':6,'N':7,'O':8,'S':16,'P':15,'F':9,'Cl':17,'Br':35,'I':53,'H':1}.get(e[0],0) for e in elements])
    electronic_features = atomic_electronic_features[np.clip(atomic_nums,0,10)]  # 最多到Ne

    node_features = np.hstack([element_features, residue_features, is_ligand, min_dists, charges.reshape(-1,1), electronic_features])

    # 边
    pos = torch.tensor(coords, dtype=torch.float)
    edge_index = radius_graph(pos, r=DIST_CUTOFF)

    return torch.tensor(node_features, dtype=torch.float), pos, edge_index

# ==== 全流程 ====
if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv(CSV_PATH)

    dataset = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        uniprot = row['uniprot']
        smiles = row['substrate_smiles'].split(';')[0]
        import hashlib
        pid_hash = int(hashlib.sha256(smiles.encode()).hexdigest(),16) & 0xffff
        pdb_file = f"{uniprot}_{pid_hash}_10A.pdb"
        pqr_file = pdb_file.replace('.pdb','.pqr')
        pdb_path = os.path.join(POCKET_DIR, pdb_file)
        pqr_path = os.path.join(POCKET_DIR, pqr_file)

        if not os.path.exists(pdb_path) or not os.path.exists(pqr_path):
            print(f"缺失pdb或pqr: {pdb_path}, {pqr_path}")
            continue

        atoms, charges = parse_pocket(pdb_path, pqr_path)
        if len(atoms)<3:
            print(f"原子数过少: {pdb_path}")
            continue

        kcat = row.get('kcat Wildtype', 0.0)
        km = row.get('Km Wildtype', 0.0)
        if kcat <= 0 or km <= 0:
            print(f"标签缺失或为0: {uniprot}, kcat={kcat}, km={km}")
            continue

        x, pos, edge_index = build_graph(atoms, charges)
        label = torch.log10(torch.tensor([kcat, km], dtype=torch.float))

        data = Data(x=x, pos=pos, edge_index=edge_index, y=label, pdb_id=pdb_file)
        dataset.append(data)

    torch.save(dataset, SAVE_PATH)
    print(f"✅ Saved {len(dataset)} samples to {SAVE_PATH}")


