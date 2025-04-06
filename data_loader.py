# src/data_loader.py
import os
import numpy as np
import pandas as pd
import torch
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from functools import lru_cache
import requests
import tempfile
import subprocess
import biotite.structure as struc
from biotite.structure.io import pdb
from Bio.PDB import PDBParser, Selection
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.HSExposure import HSExposureCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_output_dirs():
    """确保所有输出目录存在"""
    dirs = [
        'output/logs',
        'output/cache/structure_cache',
        'output/cache/molecule_embeddings_cache',
        'output/processed',
        'output/viz'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class ProteinStructureProcessor:
    def __init__(self, cache_dir='output/cache/structure_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.uniprot_info_cache = {}
        
    def predict_structure(self, sequence, uniprot_id=None):
        """使用ESMFold或从PDB获取蛋白质结构"""
        # 检查缓存
        sequence_hash = hash(sequence)
        pdb_cache_path = os.path.join(self.cache_dir, f"{sequence_hash}.pdb")
        
        if os.path.exists(pdb_cache_path):
            logging.info(f"使用缓存的PDB结构: {pdb_cache_path}")
            with open(pdb_cache_path, 'r') as f:
                return f.read()
        
        # 尝试从UniProt/PDB获取结构
        if uniprot_id and self._try_download_pdb(uniprot_id, pdb_cache_path):
            with open(pdb_cache_path, 'r') as f:
                return f.read()
        
        # 如果无法获取已有结构，则使用ESMFold预测
        try:
            logging.info(f"使用ESMFold预测蛋白质结构")
            # 这里需要安装esm库: pip install "fair-esm[esmfold]"
            import esm
            model = esm.pretrained.esmfold_v1()
            model.eval()
            
            with torch.no_grad():
                output = model.infer_pdb(sequence)
                
                # 保存到缓存
                with open(pdb_cache_path, 'w') as f:
                    f.write(output)
                
                return output
        except Exception as e:
            logging.error(f"ESMFold预测失败: {e}")
            # 如果ESMFold预测失败，创建一个简单的PDB文件
            return self._create_dummy_pdb(sequence)
    
    def _try_download_pdb(self, uniprot_id, output_path):
        """尝试从UniProt和PDB下载结构"""
        try:
            # 获取UniProt信息
            if uniprot_id not in self.uniprot_info_cache:
                uniprot_url = f"https://www.uniprot.org/uniprot/{uniprot_id}.xml"
                response = requests.get(uniprot_url)
                if response.status_code != 200:
                    return False
                self.uniprot_info_cache[uniprot_id] = response.text
            
            # 从UniProt信息中提取PDB ID
            import re
            pdb_ids = re.findall(r'<dbReference type="PDB" id="([^"]+)"', self.uniprot_info_cache[uniprot_id])
            
            if not pdb_ids:
                return False
            
            # 下载第一个PDB文件
            pdb_id = pdb_ids[0]
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(pdb_url)
            
            if response.status_code == 200:
                with open(output_path, 'w') as f:
                    f.write(response.text)
                logging.info(f"从PDB下载结构成功: {pdb_id}")
                return True
            
            return False
        except Exception as e:
            logging.error(f"从PDB下载结构失败: {e}")
            return False
    
    def _create_dummy_pdb(self, sequence):
        """为序列创建一个简单的线性PDB结构"""
        pdb_lines = []
        pdb_lines.append("HEADER    DUMMY STRUCTURE")
        aa_dict = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
        'X': 'XXX'
         }
        for i, aa in enumerate(sequence):
       
            if aa not in aa_dict:
                aa = 'X'  # 对未知氨基酸使用X
            
            # 确保坐标是有效的浮点数
            x = float(i * 3.8)  # 氨基酸间距约3.8埃
            y = 0.0
            z = 0.0
            
            atom_line = (f"ATOM  {i+1:5d}  CA  {aa_dict.get(aa, 'XXX')} A{i+1:4d}"
                        f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  ")
            pdb_lines.append(atom_line)
        
        pdb_lines.append("END")
        return "\n".join(pdb_lines)

    def identify_binding_site(self, structure_data, uniprot_id=None, sequence=None):
        """识别活性位点的中心位置，结合多种方法"""
        # 保存PDB结构到临时文件
        with tempfile.NamedTemporaryFile('w', suffix='.pdb', delete=False) as tmp:
            tmp.write(structure_data)
            tmp_pdb_path = tmp.name
        
        try:
            active_site = None
            
            # 方法1: 尝试从UniProt获取已知的活性位点信息
            if uniprot_id:
                active_site = self._get_active_site_from_uniprot(uniprot_id)
                if active_site:
                    logging.info(f"从UniProt获取到活性位点信息: {active_site}")
                    return self._get_coordinates_for_residues(tmp_pdb_path, active_site)
            
            # 方法2: 使用fpocket检测口袋
            try:
                pocket_centers = self._detect_pockets_with_fpocket(tmp_pdb_path)
                if pocket_centers:
                    logging.info(f"使用fpocket检测到{len(pocket_centers)}个口袋")
                    # 返回得分最高的口袋中心
                    return pocket_centers[0]
            except Exception as e:
                logging.warning(f"fpocket检测失败: {e}")
            
            # 方法3: 使用保守性和结构特征分析
            return self._analyze_structure_features(tmp_pdb_path, sequence)
            
        finally:
            # 清理临时文件
            if os.path.exists(tmp_pdb_path):
                os.remove(tmp_pdb_path)
                if os.path.exists(tmp_pdb_path.replace('.pdb', '_out')):
                    import shutil
                    shutil.rmtree(tmp_pdb_path.replace('.pdb', '_out'))
    
    def _get_active_site_from_uniprot(self, uniprot_id):
        """从UniProt获取活性位点信息"""
        try:
            if uniprot_id not in self.uniprot_info_cache:
                uniprot_url = f"https://www.uniprot.org/uniprot/{uniprot_id}.xml"
                response = requests.get(uniprot_url)
                if response.status_code != 200:
                    return None
                self.uniprot_info_cache[uniprot_id] = response.text
            
            # 解析XML找到活性位点注释
            import re
            # 查找活性位点标记
            active_site_positions = re.findall(r'<feature type="active site".+?position position="(\d+)"', 
                                            self.uniprot_info_cache[uniprot_id])
            
            # 查找活性位点范围
            active_site_ranges = re.findall(r'<feature type="active site".+?begin position="(\d+)".+?end position="(\d+)"', 
                                          self.uniprot_info_cache[uniprot_id])
            
            # 合并结果
            active_sites = [int(pos) for pos in active_site_positions]
            for begin, end in active_site_ranges:
                active_sites.extend(range(int(begin), int(end)+1))
            
            return sorted(set(active_sites)) if active_sites else None
        except Exception as e:
            logging.error(f"从UniProt获取活性位点信息失败: {e}")
            return None
    
    def _get_coordinates_for_residues(self, pdb_path, residue_indices):
        """获取特定残基的坐标"""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        
        # 获取第一个模型和链
        model = structure[0]
        chain = next(model.get_chains())
        
        # 收集残基坐标
        residue_coords = []
        for res_id in residue_indices:
            try:
                # PDB中的残基编号可能与序列位置不同
                residue = chain[(' ', res_id, ' ')]
                # 获取CA原子坐标
                ca_atom = residue['CA']
                residue_coords.append(ca_atom.get_coord())
            except KeyError:
                continue
        
        # 如果找到了残基，返回平均坐标
        if residue_coords:
            return np.mean(residue_coords, axis=0)
        return None
    
    def _detect_pockets_with_fpocket(self, pdb_path):
        """使用fpocket检测蛋白质口袋"""
        # 运行fpocket
        output_dir = pdb_path.replace('.pdb', '_out')
        cmd = ['fpocket', '-f', pdb_path]
        
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
    
    def _analyze_structure_features(self, pdb_path, sequence=None):
        """分析结构特征，识别可能的活性位点"""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        model = structure[0]
        
        # 获取所有残基
        residues = Selection.unfold_entities(model, 'R')
        
        # 计算每个残基的特征
        residue_features = []
        for residue in residues:
            if residue.get_id()[0] != ' ':  # 跳过非标准残基
                continue
            
            try:
                # 1. 计算残基深度
                rd = ResidueDepth(model, residue)
                depth = rd[residue][0]  # 平均残基深度
                
                # 2. 计算暴露面积
                hs = HSExposureCA(model)
                exposure = sum(1 for atom in residue if atom.get_name() in ['CA', 'CB', 'C', 'N', 'O'])
                
                # 3. 考虑残基类型 (某些氨基酸更可能出现在活性位点)
                amino_acid = residue.get_resname()
                activity_score = 0
                
                # 常见活性位点氨基酸: H, D, E, K, R, S, T, C
                active_site_propensity = {
                    'HIS': 1.0, 'ASP': 0.9, 'GLU': 0.8, 'LYS': 0.8, 'ARG': 0.8,
                    'SER': 0.7, 'THR': 0.7, 'CYS': 0.9, 'TYR': 0.7, 'TRP': 0.6
                }
                
                activity_score = active_site_propensity.get(amino_acid, 0.2)
                
                # 综合评分 (活性位点通常在蛋白质内部但不是太深，且由特定氨基酸组成)
                # 低深度(更靠近表面但不在最表面)的残基可能更接近活性位点
                depth_score = 1.0 if 1.5 < depth < 5.0 else 0.2
                
                # 计算最终评分
                final_score = activity_score * 0.6 + depth_score * 0.4
                
                residue_features.append((residue, final_score, depth))
            except Exception as e:
                continue
        
        # 按评分排序
        residue_features.sort(key=lambda x: x[1], reverse=True)
        
        # 选择评分最高的残基作为活性位点
        if residue_features:
            # 获取前10%评分最高的残基
            top_residues = residue_features[:max(1, int(len(residue_features) * 0.1))]
            
            # 获取这些残基的坐标
            coords = []
            for residue, _, _ in top_residues:
                for atom in residue:
                    if atom.get_name() == 'CA':
                        coords.append(atom.get_coord())
            
            # 返回这些残基坐标的平均值
            if coords:
                return np.mean(coords, axis=0)
        
        # 如果上述方法都失败，返回结构中心
        all_coords = [atom.get_coord() for residue in residues 
                    for atom in residue if atom.get_id() == 'CA']
        return np.mean(all_coords, axis=0) if all_coords else np.array([0, 0, 0])
                
    def extract_binding_site(self, pdb_str, uniprot_id=None, sequence=None, radius=10.0):
        """从PDB结构中提取活性位点周围的原子坐标"""
        # 解析PDB结构
        with tempfile.NamedTemporaryFile('w', suffix='.pdb', delete=False) as tmp:
            tmp.write(pdb_str)
            tmp_pdb_path = tmp.name
        
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', tmp_pdb_path)
            model = structure[0]
            
            # 识别活性位点
            binding_site_center = self.identify_binding_site(pdb_str, uniprot_id, sequence)
            
            # 提取周围原子
            atoms = []
            atom_types = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'H': 5}  # 原子类型映射
            
            for residue in Selection.unfold_entities(model, 'R'):
                if residue.get_id()[0] != ' ':  # 跳过非标准残基
                    continue
                
                for atom in residue:
                    atom_coord = atom.get_coord()
                    dist = np.linalg.norm(atom_coord - binding_site_center)
                    
                    if dist <= radius:
                        atom_element = atom.element.strip()
                        if not atom_element:
                            atom_element = atom.name[0]
                        
                        atom_type = atom_types.get(atom_element, 6)  # 默认为"其他"类型
                        
                        atoms.append({
                            'type': atom_type,
                            'coords': atom_coord,
                            'residue': residue.get_resname(),
                            'distance': dist
                        })
            
            return binding_site_center, atoms
        
        finally:
            if os.path.exists(tmp_pdb_path):
                os.remove(tmp_pdb_path)

def get_first_smiles(smiles_str):
    """从可能包含多个SMILES的字符串中提取第一个SMILES"""
    if not smiles_str or pd.isna(smiles_str):
        return None
    
    # 通常底物用分号分隔
    first_smiles = smiles_str.split(';')[0].strip()
    return first_smiles

class MoleculeEmbeddingGenerator:
    def __init__(self, cache_dir='output/cache/molecule_embeddings_cache'):
        # 使用UniMol或其他预训练模型
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            self.model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            self.model.eval()
        except Exception as e:
            logging.error(f"加载分子嵌入模型失败: {e}")
            self.tokenizer = None
            self.model = None
    
    def generate_embedding(self, smiles):
        """生成分子的嵌入表示"""
        # 检查缓存
        if not smiles or pd.isna(smiles):
            return np.zeros(768)  # 默认嵌入维度
        
        smiles_hash = hash(smiles)
        cache_file = os.path.join(self.cache_dir, f"{smiles_hash}.npy")
        
        if os.path.exists(cache_file):
            return np.load(cache_file)
        
        try:
            if self.tokenizer is None or self.model is None:
                # 如果模型加载失败，使用RDKit分子指纹作为后备
                return self._generate_fallback_fingerprint(smiles)
            
            import torch
            # 编码单个SMILES
            encoded = self.tokenizer(smiles, padding=True, truncation=True, return_tensors="pt")
            
            # 获取嵌入
            with torch.no_grad():
                outputs = self.model(**encoded)
                # 使用平均池化获得固定维度的表示
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                # 保存到缓存
                np.save(cache_file, embedding)
                
                return embedding
        except Exception as e:
            logging.error(f"生成分子嵌入失败: {e}，使用后备方法")
            return self._generate_fallback_fingerprint(smiles)
    
    def _generate_fallback_fingerprint(self, smiles):
        """使用RDKit生成分子指纹作为后备方法"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=768)
                return np.array(fp)
        except:
            pass
        
        return np.zeros(768)

def create_data_loaders(binding_site_features, substrate_features, y, train_indices, test_indices, batch_size=16):
    """创建PyTorch数据加载器"""
    # 分割数据
    x_train_binding = binding_site_features[train_indices]
    x_train_substrate = substrate_features[train_indices]
    y_train = y[train_indices]
    
    x_test_binding = binding_site_features[test_indices]
    x_test_substrate = substrate_features[test_indices]
    y_test = y[test_indices]
    
    # 转换为PyTorch张量
    x_train_binding = torch.tensor(x_train_binding, dtype=torch.float32)
    x_train_substrate = torch.tensor(x_train_substrate, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    
    x_test_binding = torch.tensor(x_test_binding, dtype=torch.float32)
    x_test_substrate = torch.tensor(x_test_substrate, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # 创建数据集
    train_dataset = TensorDataset(x_train_binding, x_train_substrate, y_train)
    test_dataset = TensorDataset(x_test_binding, x_test_substrate, y_test)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def load_and_preprocess_data(data_path, timestamp=None, save_processed=True, save_visualization=True):
    """更新的数据加载和预处理函数，增强错误处理"""
    # 设置专门的日志文件
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 确保日志目录存在
    os.makedirs("output/logs", exist_ok=True)
    log_file = f"output/logs/data_processing_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    logger = logging.getLogger("data_processor")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    logger.info(f"开始数据处理: {data_path}")
    
    try:
        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 尝试加载CSV文件
        try:
            df = pd.read_csv(data_path)
            logger.info(f"原始数据集大小: {len(df)} 样本")
        except Exception as e:
            raise ValueError(f"CSV文件加载失败: {e}")
        
        # 检查必要的列是否存在
        required_columns = ["protein_sequence", "substrate_smiles", "Km Wildtype", "kcat Wildtype"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据文件缺少必要的列: {missing_columns}")
        
        # 检查数据有效性
        if df["protein_sequence"].isnull().any():
            logger.warning(f"发现 {df['protein_sequence'].isnull().sum()} 条蛋白质序列为空")
            df = df.dropna(subset=["protein_sequence"])
        
        if df["substrate_smiles"].isnull().any():
            logger.warning(f"发现 {df['substrate_smiles'].isnull().sum()} 条底物SMILES为空")
            df = df.dropna(subset=["substrate_smiles"])
        
        # 检查Km和kcat值
        if (df["Km Wildtype"] <= 0).any() or (df["kcat Wildtype"] <= 0).any():
            logger.warning("发现Km或kcat值小于等于0，这些值在取log时会出现问题")
            # 过滤掉无效值
            df = df[(df["Km Wildtype"] > 0) & (df["kcat Wildtype"] > 0)]
        
        logger.info(f"数据清洗后大小: {len(df)} 样本")
        
        # 初始化处理器
        try:
            structure_processor = ProteinStructureProcessor()
        except Exception as e:
            logger.error(f"初始化结构处理器失败: {e}")
            raise RuntimeError(f"初始化结构处理器失败: {e}")
        
        try:
            mol_embedding_generator = MoleculeEmbeddingGenerator()
        except Exception as e:
            logger.error(f"初始化分子嵌入生成器失败: {e}")
            raise RuntimeError(f"初始化分子嵌入生成器失败: {e}")
        
        # 处理蛋白质结构和活性位点
        binding_site_features = []
        protein_sequences = []
        
        for i, row in tqdm(df.iterrows(), total=len(df), desc="处理蛋白质结构"):
            try:
                seq = row["protein_sequence"]
                uniprot_id = row.get("uniprot", None)
                
                logger.info(f"处理蛋白质 {i+1}/{len(df)}: {uniprot_id if uniprot_id else '未知'}")
                
                # 预测结构
                pdb_str = structure_processor.predict_structure(seq, uniprot_id)
                
                # 提取活性位点特征
                binding_center, binding_atoms = structure_processor.extract_binding_site(
                    pdb_str, uniprot_id, seq)
                
                # 处理原子特征
                if binding_atoms:
                    # 将原子特征转换为固定大小的特征向量
                    atom_features = np.zeros((100, 7))  # 最多100个原子，每个原子7个特征
                    
                    for j, atom in enumerate(binding_atoms[:100]):
                        atom_features[j, 0] = atom['type']  # 原子类型
                        atom_features[j, 1:4] = atom['coords']  # 坐标
                        atom_features[j, 4] = atom['distance']  # 到中心的距离
                    
                    binding_site_features.append(atom_features)
                    protein_sequences.append(seq)
                else:
                    logger.warning(f"蛋白质 {i+1} 未找到活性位点原子，使用零向量")
                    # 使用零向量
                    binding_site_features.append(np.zeros((100, 7)))
                    protein_sequences.append(seq)
            except Exception as e:
                logger.error(f"处理蛋白质 {i+1} 时出错: {e}")
                # 使用零向量作为后备
                binding_site_features.append(np.zeros((100, 7)))
                protein_sequences.append(row["protein_sequence"])
        
        # 处理底物分子，只使用第一个底物
        substrate_embeddings = []
        substrate_smiles = []
        
        for i, smiles in tqdm(enumerate(df["substrate_smiles"]), total=len(df), desc="处理底物分子"):
            try:
                first_smiles = get_first_smiles(smiles)
                substrate_smiles.append(first_smiles)
                
                emb = mol_embedding_generator.generate_embedding(first_smiles)
                substrate_embeddings.append(emb)
                
                if (i+1) % 10 == 0:
                    logger.info(f"已处理 {i+1}/{len(df)} 个底物")
            except Exception as e:
                logger.error(f"处理底物 {i+1} 时出错: {e}")
                # 使用零向量作为后备
                substrate_embeddings.append(np.zeros(768))
                substrate_smiles.append(smiles)
        
        # 转换为numpy数组
        substrate_embeddings = np.array(substrate_embeddings)
        binding_site_features = np.array(binding_site_features)
        
        # 处理标签 (取log)
        y = np.column_stack([
            np.log10(df["Km Wildtype"]),
            np.log10(df["kcat Wildtype"])
        ])
        
                # 划分训练测试集
        indices = np.arange(len(df))
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
        
        logger.info(f"特征形状 - 活性位点: {binding_site_features.shape}, 底物: {substrate_embeddings.shape}")
        
        # 创建数据加载器
        train_loader, test_loader = create_data_loaders(
            binding_site_features, substrate_embeddings, y, train_indices, test_indices)
        
        # 保存预处理数据
        if save_processed:
            # 确保processed目录存在
            os.makedirs("output/processed", exist_ok=True)
            output_path = f"output/processed/processed_data_{timestamp}.npz"
            np.savez(
                output_path, 
                binding_site_features=binding_site_features,
                substrate_embeddings=substrate_embeddings,
                y=y, 
                train_indices=train_indices,
                test_indices=test_indices,
                metadata={
                    'protein_sequences': protein_sequences,
                    'substrate_smiles': substrate_smiles
                }
            )
            logger.info(f"预处理数据保存至: {output_path}")
        
        # 保存可视化结果
        if save_visualization:
            viz_dir = save_processed_data_visualization(
                binding_site_features, substrate_embeddings, y, 
                protein_sequences, substrate_smiles
            )
            logger.info(f"数据可视化结果保存至: {viz_dir}")
        
        logger.info("数据处理成功完成")
        return train_loader, test_loader, binding_site_features, substrate_embeddings, y, train_indices, test_indices
    
    except Exception as e:
        logger.error(f"数据处理过程中发生错误: {e}", exc_info=True)
        raise
    finally:
        logger.handlers.remove(file_handler)

def save_processed_data_visualization(binding_site_features, substrate_embeddings, y, 
                                     protein_sequences, substrate_smiles, output_dir="output/viz/"):
    """保存处理后的数据可视化结果，方便检查"""
    logger = logging.getLogger(__name__)
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    viz_dir = f"{output_dir}_{timestamp}"
    os.makedirs(viz_dir, exist_ok=True)
    logger.info(f"开始生成可视化结果，输出目录: {viz_dir}")
    
    # 设置中文支持
    plt.rcParams['font.family'] = 'SimHei'  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 1. 保存基本统计信息
    stats = {
        "数据集大小": len(y),
        "活性位点特征形状": binding_site_features.shape,
        "底物嵌入形状": substrate_embeddings.shape,
        "标签形状": y.shape,
        "Km范围": [float(np.min(y[:, 0])), float(np.max(y[:, 0]))],
        "kcat范围": [float(np.min(y[:, 1])), float(np.max(y[:, 1]))]
    }
    
    with open(os.path.join(viz_dir, "stats.txt"), "w", encoding="utf-8") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    # 2. 保存样本数据
    sample_data = []
    for i in range(min(10, len(y))):
        sample_data.append({
            "索引": i,
            "蛋白质序列前30个字符": protein_sequences[i][:30] + "...",
            "底物SMILES": substrate_smiles[i],
            "log10(Km)": y[i, 0],
            "log10(kcat)": y[i, 1],
            "活性位点非零特征数": np.count_nonzero(binding_site_features[i]),
            "底物嵌入前5个值": substrate_embeddings[i][:5].tolist()
        })
    
    pd.DataFrame(sample_data).to_csv(os.path.join(viz_dir, "sample_data.csv"), index=False)
    


    plt.figure(figsize=(12, 8))
    
    # Binding site feature distribution
    plt.subplot(2, 2, 1)
    sns.histplot(binding_site_features.flatten(), bins=50)
    plt.title("Binding Site Feature Distribution")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    
    # Substrate embedding distribution
    plt.subplot(2, 2, 2)
    sns.histplot(substrate_embeddings.flatten(), bins=50)
    plt.title("Substrate Embedding Distribution")
    plt.xlabel("Embedding Value")
    plt.ylabel("Frequency")
    
    # Km distribution
    plt.subplot(2, 2, 3)
    sns.histplot(y[:, 0], bins=20)
    plt.title("log10(Km) Distribution")
    plt.xlabel("log10(Km)")
    plt.ylabel("Frequency")
    
    # kcat distribution
    plt.subplot(2, 2, 4)
    sns.histplot(y[:, 1], bins=20)
    plt.title("log10(kcat) Distribution")
    plt.xlabel("log10(kcat)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_distributions.png"))
    
    # 修改相关性热图的标题
    plt.figure(figsize=(10, 8))
    
    # 计算相关性矩阵
    n_features = min(20, binding_site_features.shape[2])
    selected_features = np.random.choice(binding_site_features.shape[2], n_features, replace=False)
    feature_sample = binding_site_features[:, 0, selected_features]
    corr_matrix = np.corrcoef(feature_sample.T)
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Binding Site Feature Correlations")
    plt.savefig(os.path.join(output_dir, "feature_correlations.png"))
   
   
    
    # 4. 保存特征相关性热图
    plt.figure(figsize=(10, 8))
    
    # 随机选择一些活性位点特征
    n_features = min(20, binding_site_features.shape[2])
    selected_features = np.random.choice(binding_site_features.shape[2], n_features, replace=False)
    
    # 计算相关性
    feature_sample = binding_site_features[:, 0, selected_features]  # 使用第一个原子的特征
    corr_matrix = np.corrcoef(feature_sample.T)
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Binding Site Feature Correlations")
    plt.savefig(os.path.join(viz_dir, "feature_correlations.png"))
    
    # 5. 保存原始数据样本
    with open(os.path.join(viz_dir, "raw_samples.txt"), "w", encoding="utf-8") as f:
        for i in range(min(5, len(protein_sequences))):
            f.write(f"样本 {i+1}:\n")
            f.write(f"蛋白质序列: {protein_sequences[i][:100]}...\n")
            f.write(f"底物SMILES: {substrate_smiles[i]}\n")
            f.write(f"log10(Km): {y[i, 0]}\n")
            f.write(f"log10(kcat): {y[i, 1]}\n")
            f.write(f"活性位点特征前10个: {binding_site_features[i, 0, :10].tolist()}\n")
            f.write(f"底物嵌入前10个: {substrate_embeddings[i][:10].tolist()}\n")
            f.write("\n" + "-"*50 + "\n\n")
    
    # 6. 保存3D可视化数据（用于后续可视化）
    binding_site_viz_data = []
    for i in range(min(5, len(binding_site_features))):
        # 提取非零原子
        atoms = []
        for j in range(binding_site_features.shape[1]):
            if np.any(binding_site_features[i, j, 1:4] != 0):  # 检查坐标是否非零
                atom_type = int(binding_site_features[i, j, 0])
                coords = binding_site_features[i, j, 1:4]
                atoms.append({
                    "type": atom_type,
                    "coords": coords.tolist()
                })
        
        binding_site_viz_data.append({
            "sample_id": i,
            "protein": protein_sequences[i][:50] + "...",
            "substrate": substrate_smiles[i],
            "atoms": atoms
        })
    
    # 保存为JSON
    import json
    with open(os.path.join(viz_dir, "binding_site_viz.json"), "w") as f:
        json.dump(binding_site_viz_data, f, indent=2)
    
    # 7. 生成简单的HTML报告
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>数据处理报告 - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
            img {{ max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>酶动力学参数预测 - 数据处理报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>数据集统计</h2>
        <div class="stats">
            <p>数据集大小: {len(y)} 样本</p>
            <p>活性位点特征形状: {binding_site_features.shape}</p>
            <p>底物嵌入形状: {substrate_embeddings.shape}</p>
            <p>Km范围 (log10): [{float(np.min(y[:, 0])):.4f}, {float(np.max(y[:, 0])):.4f}]</p>
            <p>kcat范围 (log10): [{float(np.min(y[:, 1])):.4f}, {float(np.max(y[:, 1])):.4f}]</p>
        </div>
        
        <h2>特征分布</h2>
        <img src="feature_distributions.png" alt="特征分布图">
        
        <h2>特征相关性</h2>
        <img src="feature_correlations.png" alt="特征相关性热图">
        
        <h2>样本数据</h2>
        <table>
            <tr>
                <th>索引</th>
                <th>蛋白质序列</th>
                <th>底物SMILES</th>
                <th>log10(Km)</th>
                <th>log10(kcat)</th>
            </tr>
    """
    
    for i in range(min(10, len(y))):
        html_report += f"""
            <tr>
                <td>{i}</td>
                <td>{protein_sequences[i][:30]}...</td>
                <td>{substrate_smiles[i]}</td>
                <td>{y[i, 0]:.4f}</td>
                <td>{y[i, 1]:.4f}</td>
            </tr>
        """
    
    html_report += """
        </table>
        
        <h2>处理说明</h2>
        <ul>
            <li>蛋白质结构通过ESMFold预测或从PDB获取</li>
            <li>活性位点通过UniProt注释、fpocket或结构分析识别</li>
            <li>底物使用ChemBERTa模型生成嵌入</li>
            <li>Km和kcat值取log10转换</li>
        </ul>
    </body>
    </html>
    """
    
    with open(os.path.join(viz_dir, "report.html"), "w") as f:
        f.write(html_report)
    
    logging.info(f"数据可视化结果已保存至: {viz_dir}")
    return viz_dir

# 如果直接运行此脚本，则处理数据
if __name__ == "__main__":
    import argparse
    
    # 确保输出目录存在
    ensure_output_dirs()
    
    parser = argparse.ArgumentParser(description="处理酶动力学数据")
    parser.add_argument("--data_path", type=str, default="/home/lizihao/Work/data_processed/cleaned_data.csv", help="原始数据路径")
    parser.add_argument("--save_processed", action="store_true", help="是否保存预处理数据")
    parser.add_argument("--save_visualization", action="store_true", help="是否保存可视化结果")
    
    args = parser.parse_args()
    
    # 处理数据
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_loader, test_loader, binding_site_features, substrate_embeddings, y, train_indices, test_indices = load_and_preprocess_data(
        args.data_path, timestamp=timestamp, save_processed=True, save_visualization=True  # 默认都保存
    )
    
    print(f"数据处理完成，共 {len(y)} 个样本")
    print(f"活性位点特征形状: {binding_site_features.shape}")
    print(f"底物嵌入形状: {substrate_embeddings.shape}")
    print(f"训练集大小: {len(train_indices)}, 测试集大小: {len(test_indices)}")
       