import os
import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import cdist
import pandas as pd
import hashlib
from tqdm import tqdm

# ==== 设置参数 ====
CSV_PATH = '/home/lizihao/Work/enzyme_prediction/data/cleaned_data.csv'
POCKET_DIR = 'data/processed/pockets'
SAVE_PATH = 'data/processed/dataset1.pt'  ##丰富信息
DIST_CUTOFF = 5.0

# ==== 初始化编码器 ====
element_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
element_encoder.fit(np.array([['C'], ['N'], ['O'], ['S'], ['P'], ['F'], ['Cl'], ['Br'], ['I'], ['H']]))

residue_types = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                 'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','LIG']
residue_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
residue_encoder.fit(np.array([[r] for r in residue_types]))

# ==== 工具函数 ====
def parse_pocket(pocket_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pocket', pocket_path)
    atoms = []
    for atom in structure.get_atoms():
        if atom.element != 'H':  # 过滤氢
            resname = atom.get_parent().get_resname()
            chain_id = atom.get_parent().get_full_id()[2]
            is_ligand = 1 if resname == 'UNL' or chain_id == ' ' else 0  # UNL或空链默认为配体
            atoms.append({
                'coord': atom.coord,
                'element': atom.element,
                'residue': resname if resname in residue_types else 'LIG',
                'is_ligand': is_ligand
            })
    return atoms

def build_graph(atoms):
    coords = np.array([atom['coord'] for atom in atoms])
    elements = np.array([[atom['element']] for atom in atoms])
    residues = np.array([[atom['residue']] for atom in atoms])
    is_ligand = np.array([[atom['is_ligand']] for atom in atoms])

    # 编码节点特征
    element_features = element_encoder.transform(elements)
    residue_features = residue_encoder.transform(residues)
    min_distances = np.min(cdist(coords, coords) + np.eye(len(coords)) * 1e6, axis=1, keepdims=True)
    node_features = np.hstack([element_features, residue_features, is_ligand, min_distances])

    # 构建边
    distances = cdist(coords, coords)
    edge_indices = np.array(np.where((distances < DIST_CUTOFF) & (distances > 0)))
    row, col = edge_indices
    edge_distance = distances[row, col].reshape(-1, 1)
    edge_type = np.abs(is_ligand[row] - is_ligand[col]) + is_ligand[row] * 2  # 0: pro-pro, 1: pro-lig, 2: lig-lig
    edge_attr = np.hstack([edge_distance, edge_type])

    return torch.tensor(node_features, dtype=torch.float), \
           torch.tensor(edge_indices, dtype=torch.long), \
           torch.tensor(edge_attr, dtype=torch.float)

# ==== 主流程 ====
if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)
    dataset = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        uniprot = row['uniprot']
        kcat = row.get('kcat Wildtype', 0.0)
        km = row.get('Km Wildtype', 0.0)
        label = [kcat if pd.notnull(kcat) else 0.0, km if pd.notnull(km) else 0.0]
        smiles = row['substrate_smiles'].split(';')[0]
        pocket_pattern = f"{uniprot}_{int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff}_10A.pdb"
        pocket_path = os.path.join(POCKET_DIR, pocket_pattern)

        if not os.path.exists(pocket_path):
            print(f"❌ Missing: {pocket_path}")
            continue

        try:
            atoms = parse_pocket(pocket_path)
            if len(atoms) < 5:
                print(f"⚠️ Too few atoms in {pocket_path}")
                continue

            x, edge_index, edge_attr = build_graph(atoms)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(label, dtype=torch.float),
                pdb_id=pocket_pattern
            )
            dataset.append(data)
        except Exception as e:
            print(f"⚠️ Error parsing {pocket_path}: {e}")

    torch.save(dataset, SAVE_PATH)
    print(f"✅ Finished. {len(dataset)} samples saved to {SAVE_PATH}")
