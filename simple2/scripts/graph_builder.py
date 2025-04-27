import os
import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import cdist
from tqdm import tqdm
import pandas as pd
import hashlib

# 设置参数
CSV_PATH = '/home/lizihao/Work/enzyme_prediction/data/cleaned_data.csv'  # 包含 uniprot、kcat、Km 等信息的 CSV 文件路径
POCKET_DIR = 'data/processed/pockets'      # 存储活性口袋 PDB 文件的目录
SAVE_PATH = 'data/processed/dataset_raw.pt'    # 保存生成图数据的路径
DIST_CUTOFF = 5.0                          # 定义边的距离阈值（单位：Å）

# 初始化 OneHotEncoder，用于元素类型的编码
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(np.array([['C'], ['N'], ['O'], ['S'], ['P'], ['F'], ['Cl'], ['Br'], ['I'], ['H']]))  # 可根据需要添加元素类型

# 读取 CSV 文件
df = pd.read_csv(CSV_PATH)

# 初始化 PDB 解析器
parser = PDBParser(QUIET=True)

dataset = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    uniprot = row["uniprot"]
    kcat = row.get("kcat Wildtype", None)
    km = row.get("Km Wildtype", None)
    label = [kcat if pd.notnull(kcat) else 0.0, km if pd.notnull(km) else 0.0]  # 将 kcat 和 Km 作为标签向量
    smiles = row.substrate_smiles.split(';')[0]
    pocket_pattern = f"{uniprot}_{int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff}_10A.pdb"
    pocket_path = os.path.join(POCKET_DIR, pocket_pattern)

    if not os.path.exists(pocket_path):
        print(f"❌ 文件不存在: {pocket_path}")
        continue

    try:
        structure = parser.get_structure("pocket", pocket_path)
        atoms = [atom for atom in structure.get_atoms() if atom.element != 'H']

        coords = np.array([atom.coord for atom in atoms])
        elements = np.array([[atom.element] for atom in atoms])
        node_features = encoder.transform(elements).toarray()

        distances = cdist(coords, coords)
        edge_index = np.array(np.where((distances < DIST_CUTOFF) & (distances > 0)))
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(label, dtype=torch.float),
            pdb_id=pocket_pattern
        )
        dataset.append(data)
    except Exception as e:
        print(f"⚠️ 解析失败 {pocket_path}: {e}")

# 保存生成的图数据
torch.save(dataset, SAVE_PATH)

# 显示样本数量和示例
print(f"样本数量: {len(dataset)}")
if dataset:
    print(f"示例 PDB 文件: {dataset[0].pdb_id}")
