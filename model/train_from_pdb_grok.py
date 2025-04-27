import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
import logging
import sys
import time
from datetime import datetime
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, Polypeptide
from rdkit import Chem
from rdkit.Chem import AllChem

# # 添加项目根目录到路径
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root_dir = os.path.dirname(current_dir)
# sys.path.append(project_root_dir)

from model_grok import ImprovedEnzymePredictionModel

# 设置日志
def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    return logger

# 模拟 identify_binding_site（提取活性位点原子）
def identify_binding_site(pdb_file, max_atoms=100, distance_threshold=5.0):
    """改进版的活性位点识别，结合多种方法确定活性中心"""
    try:
        # 1. 加载PDB结构
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        
        # 2. 确定活性中心位置
        active_site_center = None
        
        # 方法1: 使用fpocket检测口袋中心
        try:
            import subprocess
            # 运行fpocket检测
            subprocess.run(['fpocket', '-f', pdb_file], check=True)
            # 解析fpocket输出获取口袋中心
            pocket_file = pdb_file.replace('.pdb', '_out/pockets/pocket0_vert.pqr')
            if os.path.exists(pocket_file):
                with open(pocket_file) as f:
                    coords = []
                    for line in f:
                        if line.startswith('ATOM'):
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append([x,y,z])
                    if coords:
                        active_site_center = np.mean(coords, axis=0)
                        logging.info(f"使用fpocket检测到活性中心: {active_site_center}")
        except Exception as e:
            logging.warning(f"fpocket检测失败: {e}")
        
        # 方法2: 使用几何中心作为备选
        if active_site_center is None:
            atoms = [atom for atom in structure.get_atoms() if atom.element != 'H']
            coords = np.array([atom.get_coord() for atom in atoms])
            active_site_center = np.mean(coords, axis=0)
            logging.info(f"使用几何中心作为活性中心: {active_site_center}")

        # 3. 提取活性位点原子
        atom_types = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4}
        standard_aas = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 
                       'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 
                       'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
        
        binding_atoms = []
        for residue in structure.get_residues():
            resname = residue.get_resname().strip()
            if resname not in standard_aas:
                continue
                
            for atom in residue:
                if atom.element == 'H':
                    continue
                    
                atom_coord = atom.get_coord()
                dist = np.linalg.norm(atom_coord - active_site_center)
                
                if dist <= distance_threshold:
                    atom_element = atom.element.strip()
                    atom_type = atom_types.get(atom_element, 5)
                    try:
                        aa_idx = standard_aas.index(resname)
                    except ValueError:
                        aa_idx = len(standard_aas)
                    
                    binding_atoms.append({
                        'atom_type': atom_type,
                        'aa_type': aa_idx,
                        'coords': atom_coord,
                        'distance': dist
                    })

        # 4. 构建特征矩阵
        binding_atoms = sorted(binding_atoms, key=lambda x: x['distance'])[:max_atoms]
        atom_features = np.zeros((max_atoms, 27))
        atom_coords = np.zeros((max_atoms, 3))
        
        for j, atom_info in enumerate(binding_atoms):
            atom_features[j, atom_info['atom_type']] = 1
            atom_features[j, 5 + atom_info['aa_type']] = 1
            atom_features[j, -1] = atom_info['distance']
            atom_coords[j] = atom_info['coords']
            
        return atom_features, atom_coords

    except Exception as e:
        logging.error(f"Error in identify_binding_site for {pdb_file}: {str(e)}")
        return np.zeros((max_atoms, 27)), np.zeros((max_atoms, 3))

def _detect_pockets_with_fpocket(pdb_path):
        """使用fpocket检测蛋白质口袋"""
        # 运行fpocket
        fpocket_path = "/home/lizihao/Work/enzyme_prediction/fpocket" 
        output_dir = pdb_path.replace('.pdb', '_out')
        cmd = ["fpocket", '-f', pdb_path]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 解析fpocket输出
            pockets_file = os.path.join(output_dir, "pockets.txt")
            if not os.path.exists(pockets_file):
                return None
            
            pocket_centers = []
            with open(pockets_file, 'r') as f:
                lines = f.readlines()
                current_pocket = None
                score = None
                center = None
                
                for line in lines:
                    if line.startswith("Pocket"):
                        if current_pocket is not None and score is not None and center is not None:
                            pocket_centers.append((center, score, current_pocket))
                        
                        current_pocket = int(line.split()[1])
                    elif "Score" in line:
                        score = float(line.split(":")[1].strip())
                    elif "center" in line and "->" in line:
                        coords = line.split("->")[1].strip().split()
                        center = np.array([float(coords[0]), float(coords[1]), float(coords[2])])
            
            # 添加最后一个口袋
            if current_pocket is not None and score is not None and center is not None:
                pocket_centers.append((center, score, current_pocket))
            
            # 按评分排序
            pocket_centers.sort(key=lambda x: x[1], reverse=True)
            return [center for center, _, _ in pocket_centers]
        
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.error(f"运行fpocket失败: {e}")
            return None
      
# 构建图数据
def build_graph(atom_features, atom_coords, distance_threshold=4.0):
    """将原子特征和坐标构建为图"""
    num_atoms = len(atom_features)
    if num_atoms == 0:
        return None
    
    # 节点特征
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # 边：基于距离
    edge_index = []
    edge_attr = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(atom_coords[i] - atom_coords[j])
            if dist < distance_threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append([dist])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # 坐标（用于后续等变性扩展）
    pos = torch.tensor(atom_coords, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

# 从 SMILES 生成嵌入（备选方案）
def get_substrate_embedding(smiles, dim=384):
    """从 SMILES 生成底物嵌入（用 RDKit ECFP）"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=dim)
        return np.array(fp, dtype=np.float32)
    except Exception as e:
        logging.error(f"Error generating embedding for SMILES {smiles}: {str(e)}")
        return None

# 数据加载和预处理
def load_data(csv_file, npz_file, pdb_dir, max_rows=15, max_atoms=100):
    """加载数据，构建图和底物嵌入"""
    logger = logging.getLogger('training')
    df = pd.read_csv(csv_file).head(max_rows)  # 只读前 15 行
    npz_data = np.load(npz_file)
    
    graphs = []
    substrate_embeddings = []
    y_values = []
    indices = []
    atom_subsets = []
    logger.info(f"Processing {len(df)} rows from {csv_file}")
    for idx, row in df.iterrows():
        uniprot_id = row['uniprot']
        pdb_file = os.path.join(pdb_dir, f"{uniprot_id}.pdb")
        
        # 检查 PDB 文件
        if not os.path.exists(pdb_file):
            logger.warning(f"PDB file not found for {uniprot_id}, skipping")
            continue
        
               # 获取底物嵌入
        try:
            substrate_emb = npz_data['substrate_embeddings'][idx]
            y = npz_data['y'][idx]
        except Exception as e:
            logger.warning(f"Failed to load substrate embedding for index {idx}: {str(e)}")
            smiles = row.get('smiles')
            if smiles:
                substrate_emb = get_substrate_embedding(smiles)
                y = np.array([row['Km'], row['kcat']], dtype=np.float32)
                if substrate_emb is None:
                    logger.warning(f"Failed to generate embedding for {uniprot_id}, skipping")
                    continue
            else:
                logger.warning(f"No SMILES available for {uniprot_id}, skipping")
                continue
        
        # 提取活性位点
        atom_features, atom_coords = identify_binding_site(pdb_file, max_atoms=max_atoms)
        if atom_features is None:
            logger.warning(f"Failed to extract binding site for {uniprot_id}, skipping")
            continue
        
        # 构建图
        graph = build_graph(atom_features, atom_coords)
        if graph is None:
            logger.warning(f"Failed to build graph for {uniprot_id}, skipping")
            continue
        
 
        graphs.append(graph)
        substrate_embeddings.append(substrate_emb)
        y_values.append(y)
        indices.append(idx)
        
        # 保存活性位点原子子集
        atom_subsets.append({
            'uniprot_id': uniprot_id,
            'index': idx,
            'atom_features': atom_features,
            'atom_coords': atom_coords
        })
    
    return graphs, np.array(substrate_embeddings), np.array(y_values), np.array(indices), atom_subsets

# 评估指标
def evaluate_metrics(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in data_loader:
            binding_site, substrate, targets = batch
            binding_site = binding_site.to(device)
            substrate = substrate.to(device)
            targets = targets.to(device)
            
            outputs = model(binding_site, substrate)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    r2_km = r2_score(all_targets[:, 0], all_predictions[:, 0])
    r2_kcat = r2_score(all_targets[:, 1], all_predictions[:, 1])
    
    return {
        'loss': total_loss / len(data_loader),
        'r2_km': r2_km,
        'r2_kcat': r2_kcat
    }

# 训练函数
def train_model(csv_file, npz_file, pdb_dir, output_dir, epochs=100, batch_size=16, lr=0.001, weight_decay=1e-5):
    timestamp = datetime.now().strftime('%m%d_%H%M')
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger = setup_logger('training', os.path.join(output_dir, f'training_{timestamp}.log'))
    logger.info(f"Start training - Time: {timestamp}")
    logger.info(f"CSV: {csv_file}, NPZ: {npz_file}, PDB dir: {pdb_dir}")
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 加载数据
    graphs, substrate_embeddings, y, indices, atom_subsets = load_data(csv_file, npz_file, pdb_dir)
    if len(graphs) == 0:
        logger.error("No valid data loaded, exiting")
        sys.exit(1)
    
    # 保存活性位点原子子集
    atom_subset_file = os.path.join(output_dir, 'atom_subsets.npz')
    np.savez(atom_subset_file, 
             uniprot_ids=[item['uniprot_id'] for item in atom_subsets],
             indices=[item['index'] for item in atom_subsets],
             atom_features=[item['atom_features'] for item in atom_subsets],
             atom_coords=[item['atom_coords'] for item in atom_subsets])
    logger.info(f"Atom subsets saved to {atom_subset_file}")
    
    # 创建数据集
    dataset = list(zip(graphs, torch.tensor(substrate_embeddings, dtype=torch.float), torch.tensor(y, dtype=torch.float)))
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = GeometricDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = ImprovedEnzymePredictionModel(
        node_feature_dim=26,  # 5 (atom) + 21 (amino acid)
        substrate_dim=substrate_embeddings.shape[1]
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    
    train_losses = []
    val_losses = []
    r2_km_scores = []
    r2_kcat_scores = []
    
    best_val_metrics = {'loss': float('inf')}
    early_stopping_counter = 0
    early_stopping_patience = 20
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            binding_site, substrate, targets = batch
            binding_site = binding_site.to(device)
            substrate = substrate.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(binding_site, substrate)
            loss = model.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        val_metrics = evaluate_metrics(model, test_loader, device)
        scheduler.step(val_metrics['loss'])
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_metrics['loss'])
        r2_km_scores.append(val_metrics['r2_km'])
        r2_kcat_scores.append(val_metrics['r2_kcat'])
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.6f}, "
                   f"Val Loss: {val_metrics['loss']:.6f}, Km R²: {val_metrics['r2_km']:.4f}, "
                   f"kcat R²: {val_metrics['r2_kcat']:.4f}")
        
        if val_metrics['loss'] < best_val_metrics['loss']:
            best_val_metrics = val_metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(r2_km_scores, label='Km R²')
    plt.plot(r2_kcat_scores, label='kcat R²')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title('R² Scores')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_curves_{timestamp}.png'))
    plt.close()
    
    logger.info("\nFinal results:")
    logger.info(f"Best validation loss: {best_val_metrics['loss']:.6f}")
    logger.info(f"Best Km R²: {best_val_metrics['r2_km']:.4f}")
    logger.info(f"Best kcat R²: {best_val_metrics['r2_kcat']:.4f}")
    
    return model, best_val_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train enzyme prediction model from PDB files")
    parser.add_argument('--csv_file', type=str, default='/home/lizihao/Work/enzyme_prediction/data/cleaned_data.csv',
                       help='Path to cleaned_data.csv')
    parser.add_argument('--npz_file', type=str, default='/home/lizihao/Work/enzyme_prediction/output/processed/processed_data_20250310_205850.npz',
                       help='Path to pro_data.npz')
    parser.add_argument('--pdb_dir', type=str, default='/home/lizihao/Work/enzyme_prediction/src/output/pdb_files',
                       help='Directory containing PDB files')
    parser.add_argument('--output_dir', type=str, default='output/pdb_training',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    
    args = parser.parse_args()
    
    model, metrics = train_model(
        csv_file=args.csv_file,
        npz_file=args.npz_file,
        pdb_dir=args.pdb_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay
    )