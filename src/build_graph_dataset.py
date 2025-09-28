#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版脚本：将CSV数据转换为PyTorch Geometric Data对象
增加了角度特征和几何增强
用于构建训练数据集
"""

import pandas as pd
import os
import hashlib
import torch
import numpy as np
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import logging
import math

# 导入graph_builder_rbf中的函数
from graph_builder_rbf import build_graph, parse_pocket, gaussian_rbf

def compute_angle_features(edge_index, pos, max_neighbors=10):
    """
    计算键角特征 (Bond Angles)
    对每条边，计算其与相邻边的夹角
    
    Args:
        edge_index: [2, E] 边索引
        pos: [N, 3] 原子坐标
        max_neighbors: 最大邻居数，用于控制计算复杂度
    
    Returns:
        angle_features: [E, angle_dim] 角度特征
    """
    row, col = edge_index
    num_edges = edge_index.shape[1]
    
    # 计算边向量
    edge_vec = pos[col] - pos[row]  # [E, 3]
    edge_length = torch.norm(edge_vec, dim=1, keepdim=True)  # [E, 1]
    edge_vec_norm = edge_vec / (edge_length + 1e-8)  # [E, 3] 归一化
    
    angle_features = []
    
    for i in range(num_edges):
        center_atom = row[i]  # 中心原子
        neighbor_atom = col[i]  # 邻居原子
        
        # 找到中心原子的所有邻居（除了当前邻居）
        center_neighbors = edge_index[1][edge_index[0] == center_atom]
        other_neighbors = center_neighbors[center_neighbors != neighbor_atom]
        
        if len(other_neighbors) == 0:
            # 如果没有其他邻居，使用零向量
            angle_features.append(torch.zeros(4))  # [cos_min, cos_max, cos_mean, num_angles]
            continue
        
        # 限制邻居数量以控制计算复杂度
        if len(other_neighbors) > max_neighbors:
            other_neighbors = other_neighbors[:max_neighbors]
        
        # 计算当前边与其他边的夹角余弦值
        current_vec = edge_vec_norm[i]  # [3]
        cos_angles = []
        
        for other_neighbor in other_neighbors:
            # 找到对应的边索引
            other_edge_idx = ((edge_index[0] == center_atom) & (edge_index[1] == other_neighbor)).nonzero(as_tuple=True)[0]
            if len(other_edge_idx) > 0:
                other_vec = edge_vec_norm[other_edge_idx[0]]  # [3]
                cos_angle = torch.dot(current_vec, other_vec).clamp(-1, 1)
                cos_angles.append(cos_angle)
        
        if len(cos_angles) > 0:
            cos_angles = torch.stack(cos_angles)
            # 统计特征：最小值、最大值、均值、角度数量
            angle_stats = torch.tensor([
                cos_angles.min(),
                cos_angles.max(), 
                cos_angles.mean(),
                len(cos_angles) / max_neighbors  # 归一化的角度数量
            ])
        else:
            angle_stats = torch.zeros(4)
        
        angle_features.append(angle_stats)
    
    return torch.stack(angle_features)  # [E, 4]

def compute_dihedral_features(edge_index, pos, max_dihedrals=5):
    """
    计算二面角特征 (Dihedral Angles)
    对每条边，计算涉及该边的二面角
    
    Args:
        edge_index: [2, E] 边索引
        pos: [N, 3] 原子坐标
        max_dihedrals: 最大二面角数量
    
    Returns:
        dihedral_features: [E, dihedral_dim] 二面角特征
    """
    row, col = edge_index
    num_edges = edge_index.shape[1]
    
    dihedral_features = []
    
    for i in range(num_edges):
        atom_b = row[i]  # 中心边的起点
        atom_c = col[i]  # 中心边的终点
        
        # 找到atom_b和atom_c的邻居
        b_neighbors = edge_index[1][edge_index[0] == atom_b]
        c_neighbors = edge_index[1][edge_index[0] == atom_c]
        
        # 去除中心边上的原子
        b_others = b_neighbors[b_neighbors != atom_c]
        c_others = c_neighbors[c_neighbors != atom_b]
        
        dihedrals = []
        count = 0
        
        # 计算二面角 A-B-C-D
        for atom_a in b_others:
            for atom_d in c_others:
                if count >= max_dihedrals:
                    break
                
                # 计算二面角
                vec_ba = pos[atom_a] - pos[atom_b]
                vec_bc = pos[atom_c] - pos[atom_b]
                vec_cb = pos[atom_b] - pos[atom_c]
                vec_cd = pos[atom_d] - pos[atom_c]
                
                # 计算法向量
                n1 = torch.cross(vec_ba, vec_bc)
                n2 = torch.cross(vec_cb, vec_cd)
                
                # 归一化
                n1_norm = n1 / (torch.norm(n1) + 1e-8)
                n2_norm = n2 / (torch.norm(n2) + 1e-8)
                
                # 计算二面角余弦值
                cos_dihedral = torch.dot(n1_norm, n2_norm).clamp(-1, 1)
                dihedrals.append(cos_dihedral)
                count += 1
            
            if count >= max_dihedrals:
                break
        
        if len(dihedrals) > 0:
            dihedrals = torch.stack(dihedrals)
            # 统计特征
            dihedral_stats = torch.tensor([
                dihedrals.min(),
                dihedrals.max(),
                dihedrals.mean(),
                len(dihedrals) / max_dihedrals  # 归一化的二面角数量
            ])
        else:
            dihedral_stats = torch.zeros(4)
        
        dihedral_features.append(dihedral_stats)
    
    return torch.stack(dihedral_features)  # [E, 4]

def enhanced_build_graph(atoms, temperature):
    """
    增强版图构建函数，添加角度和二面角特征
    """
    # 使用原始的build_graph函数
    data = build_graph(atoms, temperature)
    
    # 提取边信息
    edge_index = data.edge_index
    pos = data.pos
    
    # 计算角度特征
    print("计算键角特征...")
    angle_features = compute_angle_features(edge_index, pos)
    
    # 计算二面角特征  
    print("计算二面角特征...")
    dihedral_features = compute_dihedral_features(edge_index, pos)
    
    # 将新特征添加到边特征中
    original_edge_attr = data.edge_attr  # [E, 16] (RBF features)
    enhanced_edge_attr = torch.cat([
        original_edge_attr,      # [E, 16] 原始RBF特征
        angle_features,          # [E, 4]  键角特征
        dihedral_features        # [E, 4]  二面角特征
    ], dim=1)  # [E, 24]
    
    # 更新数据对象
    data.edge_attr = enhanced_edge_attr
    
    print(f"边特征维度从 {original_edge_attr.shape[1]} 增加到 {enhanced_edge_attr.shape[1]}")
    
    return data

def main():
    # 读取你的训练数据
    df = pd.read_csv('kcat_data_for_training_corrected.csv')
    print(f'Loading {len(df)} samples from CSV')

    # 设置pocket目录 - 适配新的文件结构
    POCKET_BASE_DIR = '/home/lizihao/Work/enzyme_prediction/PGNN/sample_data/samples'
    SAVE_PATH = 'kcat_dataset_enhanced1.pt'  # 使用新文件名

    dataset = []
    successful_count = 0
    failed_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = row['sample_id']
        smiles = row['substrate_smiles']  # 使用正确的列名
        kcat_value = row['kcat_value']
        temperature = row['temperature']
        ec = row['ec']
        
        # 计算pocket文件名 - 适配新的文件结构
        pocket_hash = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff
        base_name = f'{sample_id}_{pocket_hash}'
        pocket_pdb = os.path.join(POCKET_BASE_DIR, sample_id, f'{base_name}_10A.pdb')
        
        if not os.path.exists(pocket_pdb):
            failed_count += 1
            continue
        
        try:
            atoms = parse_pocket(pocket_pdb)
            if len(atoms) < 3:
                failed_count += 1
                continue
            
            # 使用增强版图构建函数
            print(f"构建增强图: {sample_id}")
            data = enhanced_build_graph(atoms, temperature)
            data.y = torch.log10(torch.tensor([kcat_value, 1.0], dtype=torch.float))
            data.pdb_id = f'{sample_id}_{pocket_hash}_10A.pdb'
            data.sample_id = sample_id
            data.ec = ec
            
            dataset.append(data)
            successful_count += 1
            
        except Exception as e:
            print(f'Error processing {sample_id}: {e}')
            failed_count += 1
            continue

    torch.save(dataset, SAVE_PATH)
    print(f'✅ Saved {len(dataset)} samples to {SAVE_PATH}')
    print(f'Successful: {successful_count}, Failed: {failed_count}')
    print(f'Success rate: {successful_count/(successful_count+failed_count)*100:.1f}%')

if __name__ == '__main__':
    main()
