# src/data_loader.py
import os
from torch_geometric.loader import DataLoader as GeometricDataLoader
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
from transformers import AutoTokenizer, AutoModel,EsmForProteinFolding
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from Bio import SeqIO
from Bio.PDB import Structure
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
import esm
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union
import json
import hashlib
from pathlib import Path

# 创建日志目录
os.makedirs('output/logs', exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/logs/data_processing.log'),
        logging.StreamHandler()
    ]
)

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

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

class ProteinStructureProcessor:
    def __init__(self, cache_dir='output/cache/structure_cache', pdb_save_dir='output/pdb_files', sample_manager=None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        # 设置默认路径为与 src 并列的 output/cache 和 output/pdb_files
        self.cache_dir = cache_dir or os.path.join(base_dir, "output/cache/structure_cache")
        self.pdb_save_dir = pdb_save_dir or os.path.join(base_dir, "output/pdb_files")
        
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.pdb_save_dir, exist_ok=True)
        self.uniprot_info_cache = {}
        
        # 新增: SampleManager支持
        self.sample_manager = sample_manager
        
        # 初始化统计信息
        self.stats = {
            "local": 0,       # 本地加载的数量
            "pdb_download": 0, # 从PDB官网下载的数量
            "esm_predicted": 0, # 使用ESM预测的数量
            "failed": 0        # 失败的数量
        }

    def predict_structure_with_sample_id(self, sample_id, sequence, uniprot_id=None):
        """
        使用sample_id的结构预测方法 - 支持去重和失败记录
        """
        if not self.sample_manager:
            # 回退到原始方法
            return self.predict_structure(sequence, uniprot_id)
        
        try:
            # 获取蛋白质路径（支持去重）
            protein_path, is_shared = self.sample_manager.get_protein_path(sample_id)
            
            # 如果已存在且可读取
            if protein_path.exists() and protein_path.stat().st_size > 0:
                logging.info(f"从{'共享' if is_shared else '专用'}缓存加载: {protein_path}")
                with open(protein_path, 'r') as f:
                    content = f.read()
                self.stats["local"] += 1
                self.sample_manager.update_sample_status(sample_id, "completed", pdb_path=str(protein_path))
                return content
            
            # 更新状态为处理中
            self.sample_manager.update_sample_status(sample_id, "processing")
            
            # 尝试从 UniProt/PDB 获取结构
            if uniprot_id:
                pdb_cache_path = protein_path.with_suffix('.cache.pdb')
                if self._try_download_pdb(uniprot_id, str(pdb_cache_path)):
                    with open(pdb_cache_path, 'r') as f:
                        pdb_content = f.read()
                    # 复制到最终位置
                    with open(protein_path, 'w') as f:
                        f.write(pdb_content)
                    pdb_cache_path.unlink()  # 删除临时文件
                    self.stats["pdb_download"] += 1
                    self.sample_manager.update_sample_status(sample_id, "completed", pdb_path=str(protein_path))
                    return pdb_content
            
            # 使用 ESMFold 预测
            logging.info(f"使用 ESMFold 预测蛋白质结构 (sample_id: {sample_id})")
            
            # 临时取消长度限制进行测试
            # if len(sequence) > 400:  # 保守的长度限制，避免OOM
            #     error_msg = f"序列过长 ({len(sequence)} > 400)"
            #     self.sample_manager.log_failure(sample_id, "structure_prediction", error_msg, len(sequence))
            #     # 生成dummy PDB作为placeholder
            #     pdb_content = self._create_dummy_pdb(sequence)
            #     with open(protein_path, 'w') as f:
            #         f.write(pdb_content)
            #     logging.warning(f"⚠️ 序列过长，生成dummy结构: {sample_id} (长度: {len(sequence)})")
            #     return pdb_content
            
            tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
            model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
            model = model.cuda() # 如果有GPU, 使用cuda
            torch.backends.cuda.matmul.allow_tf32 = True
            tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids']
            tokenized_input = tokenized_input.cuda() # 如果有GPU, 使用cuda
            with torch.no_grad():
                output = model(tokenized_input)

            pdb_content = convert_outputs_to_pdb(output)[0] # 提取pdb内容
            
            # 保存到指定路径
            with open(protein_path, 'w') as f:
                f.write(pdb_content)
            
            self.stats["esm_predicted"] += 1
            self.sample_manager.update_sample_status(sample_id, "completed", pdb_path=str(protein_path))
            return pdb_content
            
        except Exception as e:
            logging.error(f"ESMFold 预测失败 (sample_id: {sample_id}): {e}")
            self.sample_manager.log_failure(sample_id, "structure_prediction", str(e), len(sequence))
            
            # 创建dummy PDB作为fallback
            pdb_content = self._create_dummy_pdb(sequence)
            try:
                with open(protein_path, 'w') as f:
                    f.write(pdb_content)
            except:
                pass
            
            self.stats["failed"] += 1
            return pdb_content

    def predict_structure(self, sequence, uniprot_id=None):
        """使用本地缓存、UniProt/PDB 或 ESMFold 获取蛋白质结构"""
        # 优先检查本地是否有以 UniProt ID 命名的 PDB 文件
        if uniprot_id:
            pdb_path = os.path.join(self.pdb_save_dir, f"{uniprot_id}.pdb")
            if os.path.exists(pdb_path):
                logging.info(f"从本地加载 PDB 文件: {pdb_path}")
                self.stats["local"] += 1  # 更新统计
                with open(pdb_path, 'r') as f:
                    return f.read()

        # 如果没有 UniProt ID，则使用序列哈希值
        sequence_hash = hash(sequence)
        pdb_cache_path = os.path.join(self.cache_dir, f"{sequence_hash}.pdb")
        if os.path.exists(pdb_cache_path):
            logging.info(f"使用缓存的 PDB 结构: {pdb_cache_path}")
            self.stats["local"] += 1  # 更新统计
            with open(pdb_cache_path, 'r') as f:
                pdb_content = f.read()
                self._save_pdb_file(pdb_content, uniprot_id, sequence_hash)
                return pdb_content

        # 尝试从 UniProt/PDB 获取结构
        if uniprot_id and self._try_download_pdb(uniprot_id, pdb_cache_path):
            with open(pdb_cache_path, 'r') as f:
                pdb_content = f.read()
                self.stats["pdb_download"] += 1  # 更新统计
                self._save_pdb_file(pdb_content, uniprot_id, sequence_hash)
                return pdb_content

        # 如果无法获取已有结构，则使用 ESMFold 预测
        try:
            logging.info(f"使用 ESMFold 预测蛋白质结构")
            
            # 使用文中加载 ESM 的方法
            tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
            model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
            model = model.cuda() # 如果有GPU, 使用cuda
            torch.backends.cuda.matmul.allow_tf32 = True
            tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids']
            tokenized_input = tokenized_input.cuda() # 如果有GPU, 使用cuda
            with torch.no_grad():
                output = model(tokenized_input)

            pdb_content = convert_outputs_to_pdb(output)[0] # 提取pdb内容
                # 保存到缓存
            
          
            # 保存到指定目录
            self.stats["esm_predicted"] += 1  # 更新统计
            self._save_pdb_file(pdb_content, uniprot_id, sequence_hash)
            return pdb_content
        except Exception as e:
            logging.error(f"ESMFold 预测失败: {e}")
            # 如果 ESMFold 预测失败，创建一个简单的 PDB 文件
            pdb_content = self._create_dummy_pdb(sequence)
            self._save_pdb_file(pdb_content, uniprot_id, sequence_hash)
            self.stats["failed"] += 1  # 更新统计
            return pdb_content
        

    def _save_pdb_file(self, pdb_content, uniprot_id, sequence_hash):
        """保存 PDB 文件到指定目录"""
        if uniprot_id:
            pdb_filename = f"{uniprot_id}.pdb"
        else:
            pdb_filename = f"sequence_{sequence_hash}.pdb"

        pdb_path = os.path.join(self.pdb_save_dir, pdb_filename)
        with open(pdb_path, 'w') as f:
            f.write(pdb_content)
        logging.info(f"PDB 文件已保存: {pdb_path}")

    def _try_download_pdb(self, uniprot_id, output_path):
        """尝试从 UniProt 和 PDB 下载结构"""
        try:
            # 获取 UniProt 信息
            if uniprot_id not in self.uniprot_info_cache:
                uniprot_url = f"https://www.uniprot.org/uniprot/{uniprot_id}.xml"
                response = requests.get(uniprot_url)
                if response.status_code != 200:
                    return False
                self.uniprot_info_cache[uniprot_id] = response.text

            # 从 UniProt 信息中提取 PDB ID
            import re
            pdb_ids = re.findall(r'<dbReference type="PDB" id="([^"]+)"', self.uniprot_info_cache[uniprot_id])

            if not pdb_ids:
                return False

            # 下载第一个 PDB 文件
            pdb_id = pdb_ids[0]
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(pdb_url)

            if response.status_code == 200:
                with open(output_path, 'w') as f:
                    f.write(response.text)
                logging.info(f"从 PDB 下载结构成功: {pdb_id}")
                return True

            return False
        except Exception as e:
            logging.error(f"从 PDB 下载结构失败: {e}")
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
            logging.info(f"保守口袋")
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
        """Extracts atomic coordinates around the binding site from a PDB structure.
        This function processes a PDB structure to identify and extract atomic coordinates
        within a specified radius of the binding site center.
        Args:
            pdb_str (str): PDB structure contents as a string.
            uniprot_id (str, optional): UniProt ID of the protein. Defaults to None.
            sequence (str, optional): Amino acid sequence of the protein. Defaults to None.
            radius (float, optional): Radius in Angstroms around binding site to extract atoms. 
                Defaults to 10.0.
        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: 3D coordinates of the binding site center
                - list: List of dictionaries containing atom information, where each dict has:
                    - type (int): Atomic element type (0-6 mapping)
                    - coords (numpy.ndarray): 3D coordinates of the atom
                    - residue (str): Residue name
                    - distance (float): Distance from binding site center
        Raises:
            Structure related exceptions from Bio.PDB
        """
        """从PDB结构中提取活性位点周围的原子坐标"""
        # DB结构中提取活性位点周围的原子坐标，并构建图数据"""
        with tempfile.NamedTemporaryFile('w', suffix='.pdb', delete=False) as tmp:
            tmp.write(pdb_str)
            tmp_pdb_path = tmp.name
        
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', tmp_pdb_path)
            model = structure[0]
            
            # 识别活性位点中心
            binding_site_center = self.identify_binding_site(pdb_str, uniprot_id, sequence)
            
            # 提取周围原子
            atoms = []
            atom_types = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'H': 5}
            residue_types = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

            for residue in Selection.unfold_entities(model, 'R'):
                if residue.get_id()[0] != ' ':  # 跳过非标准残基
                    continue
                for atom in residue:
                    atom_coord = atom.get_coord()
                    dist = np.linalg.norm(atom_coord - binding_site_center)
                    if dist <= radius:
                        atom_element = atom.element.strip() or atom.name[0]
                        atom_type = atom_types.get(atom_element, 6)
                        atoms.append({
                            'type': atom_type,
                            'coords': atom_coord,
                            'residue': residue.get_resname(),
                            'distance': dist
                        })

            # 构建图
            node_features = []
            edge_index = []
            edge_attr = []

            for i, atom in enumerate(atoms):
                atom_type_one_hot = np.zeros(6)
                atom_type_one_hot[atom['type']] = 1
                residue_one_hot = np.zeros(20)
                residue_one_hot[residue_types.get(atom['residue'][0], 19)] = 1  # 默认用 Y 表示未知
                feature = np.concatenate([
                    atom_type_one_hot,        # 6 维
                    atom['coords'],           # 3 维
                    [atom['distance']],       # 1 维
                    residue_one_hot           # 20 维
                ])
                node_features.append(feature)

            # 计算边（距离阈值 5Å）
            coords = np.array([atom['coords'] for atom in atoms])
            for i in range(len(atoms)):
                for j in range(i + 1, len(atoms)):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist < 5.0:  # 阈值
                        edge_index.append([i, j])
                        edge_index.append([j, i])  # 无向图
                        edge_attr.append([dist])

            # 转换为张量
            x = torch.tensor(np.array(node_features), dtype=torch.float)
            edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)
            edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)

            # 图数据对象
            graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                            center=torch.tensor(binding_site_center, dtype=torch.float))
            return binding_site_center, graph_data
        
        finally:
            if os.path.exists(tmp_pdb_path):
                os.remove(tmp_pdb_path)
    
    def extract_atoms_for_graph(self, pdb_str, uniprot_id=None, sequence=None, radius=10.0):
        """从PDB结构中提取原子列表，格式与build_graph函数兼容"""
        with tempfile.NamedTemporaryFile('w', suffix='.pdb', delete=False) as tmp:
            tmp.write(pdb_str)
            tmp_pdb_path = tmp.name
        
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', tmp_pdb_path)
            model = structure[0]
            
            # 识别活性位点中心
            binding_site_center = self.identify_binding_site(pdb_str, uniprot_id, sequence)
            
            # 提取周围原子，格式与parse_pocket兼容
            atoms = []
            residue_list = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                          'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','LIG']

            for residue in Selection.unfold_entities(model, 'R'):
                if residue.get_id()[0] != ' ':  # 跳过非标准残基
                    continue
                for atom in residue:
                    atom_coord = atom.get_coord()
                    dist = np.linalg.norm(atom_coord - binding_site_center)
                    if dist <= radius:
                        atom_element = atom.element.strip() or atom.name[0]
                        res = residue.get_resname()
                        chain = residue.get_full_id()[2]
                        is_ligand = 1 if (res == 'UNL' or chain == ' ') else 0
                        
                        atoms.append({
                            'coord': atom_coord,
                            'element': atom_element,
                            'residue': res if res in residue_list else 'LIG',
                            'is_ligand': is_ligand
                        })
            
            return atoms
        
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

def create_data_loaders(binding_site_graphs, substrate_features, y, train_indices, test_indices, batch_size=16):
    """创建支持图数据的PyTorch数据加载器"""
    train_graphs = [binding_site_graphs[i] for i in train_indices]
    test_graphs = [binding_site_graphs[i] for i in test_indices]
    x_train_substrate = torch.tensor(substrate_features[train_indices], dtype=torch.float32)
    x_test_substrate = torch.tensor(substrate_features[test_indices], dtype=torch.float32)
    y_train = torch.tensor(y[train_indices], dtype=torch.float32)
    y_test = torch.tensor(y[test_indices], dtype=torch.float32)

    # 创建数据集
    train_dataset = list(zip(train_graphs, x_train_substrate, y_train))
    test_dataset = list(zip(test_graphs, x_test_substrate, y_test))

    # 创建数据加载器
    train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = GeometricDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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
        binding_site_graphs = []  # 改为存储图数据
        protein_sequences = []
        
        for i, row in tqdm(df.iterrows(), total=len(df), desc="处理蛋白质结构"):
            try:
                seq = row["protein_sequence"]
                uniprot_id = row.get("uniprot", None)
                
                logger.info(f"处理蛋白质 {i+1}/{len(df)}: {uniprot_id if uniprot_id else '未知'}")
                
                # 预测结构
                pdb_str = structure_processor.predict_structure(seq, uniprot_id)
                
                # 提取活性位点特征（返回图数据）
                binding_center, binding_graph = structure_processor.extract_binding_site(pdb_str, uniprot_id, seq)
                binding_site_graphs.append(binding_graph)
                protein_sequences.append(seq)
            except Exception as e:
                logger.error(f"处理蛋白质 {i+1} 时出错: {e}")
                # 使用空图作为后备
                dummy_graph = Data(x=torch.zeros(1, 30), edge_index=torch.tensor([[0], [0]], dtype=torch.long))
                binding_site_graphs.append(dummy_graph)
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
        # 创建数据加载器
        train_loader, test_loader = create_data_loaders(
            binding_site_graphs, substrate_embeddings, y, train_indices, test_indices)

    # 保存预处理数据（调整保存格式）
        if save_processed:
            output_path = f"output/processed/processed_data_{timestamp}.npz"
            np.savez(
                output_path,
                substrate_embeddings=substrate_embeddings,
                y=y,
                train_indices=train_indices,
                test_indices=test_indices,
                metadata={
                    'protein_sequences': protein_sequences,
                    'substrate_smiles': substrate_smiles
                }
            )
            # 图数据单独保存（因为 npz 不支持复杂对象）
            torch.save(binding_site_graphs, f"output/processed/graphs_{timestamp}.pt")
            logger.info(f"预处理数据保存至: {output_path} 和 graphs_{timestamp}.pt")
        
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


def load_from_local_structures_and_npz(csv_path, npz_path, pdb_dir, save=True, timestamp=None):


    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    df = pd.read_csv(csv_path)
    npz_data = np.load(npz_path)

    logger = logging.getLogger("local_loader")
    logger.setLevel(logging.INFO)

    structure_processor = ProteinStructureProcessor(pdb_save_dir=pdb_dir)

    binding_site_graphs = []
    protein_sequences = []
    substrate_embeddings = []
    substrate_smiles = []
    valid_indices = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="处理本地PDB结构"):
        uniprot_id = row.get("uniprot", None)
        seq = row["protein_sequence"]
        pdb_path = os.path.join(pdb_dir, f"{uniprot_id}.pdb")

        if not os.path.exists(pdb_path):
            logger.warning(f"跳过蛋白质 {i}: {uniprot_id}，本地结构不存在")
            continue

        try:
            with open(pdb_path, "r") as f:
                pdb_str = f.read()
            center, graph = structure_processor.extract_binding_site(pdb_str, uniprot_id, seq)

            binding_site_graphs.append(graph)
            protein_sequences.append(seq)
            substrate_embeddings.append(npz_data["substrate_embeddings"][i])
            # substrate_smiles.append(get_first_smiles(row["substrate_smiles"]))
            valid_indices.append(i)
        except Exception as e:
            logger.warning(f"跳过蛋白质 {i}，处理失败: {e}")
            continue

    # 标签取log
    y = np.column_stack([
        np.log10(df.loc[valid_indices, "Km Wildtype"]),
        np.log10(df.loc[valid_indices, "kcat Wildtype"])
    ])
    substrate_embeddings = np.array(substrate_embeddings)

    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

    train_loader, test_loader = create_data_loaders(
        binding_site_graphs, substrate_embeddings, y, train_indices, test_indices)

    if save:
        os.makedirs("output/processed", exist_ok=True)
        np.savez(f"output/processed/local_processed_{timestamp}.npz",
                 substrate_embeddings=substrate_embeddings,
                 y=y,
                 train_indices=train_indices,
                 test_indices=test_indices,
                 metadata={
                     "protein_sequences": protein_sequences,
                     "substrate_smiles": substrate_smiles
                 })
        torch.save(binding_site_graphs, f"output/processed/graphs_{timestamp}.pt")
        logger.info("保存完毕")

    return train_loader, test_loader, binding_site_graphs, substrate_embeddings, y, train_indices, test_indices

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
    train_loader, test_loader, binding_site_features, substrate_embeddings, y, train_indices, test_indices = load_and_preprocess_pdbdata(
        args.data_path, timestamp=timestamp, save_processed=True, save_visualization=True  # 默认都保存
    )
    
    print(f"数据处理完成，共 {len(y)} 个样本")
    print(f"活性位点特征形状: {binding_site_features.shape}")
    print(f"底物嵌入形状: {substrate_embeddings.shape}")
    print(f"训练集大小: {len(train_indices)}, 测试集大小: {len(test_indices)}")
