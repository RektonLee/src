import os
import numpy as np
import torch
from torch_geometric.data import DataLoader
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# ==== 设置参数（与你的代码保持一致） ====
SAVE_PATH = 'data/processed/dataset1.pt'
SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== 初始化编码器（与你的代码保持一致） ====
element_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
element_encoder.fit(np.array([['C'], ['N'], ['O'], ['S'], ['P'], ['F'], ['Cl'], ['Br'], ['I'], ['H']]))

residue_types = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                 'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','LIG']
residue_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
residue_encoder.fit(np.array([[r] for r in residue_types]))

# ==== 特征重要性分析函数 ====
def shap_feature_importance(dataset_path, save_dir):
    # 加载数据集
    dataset = torch.load(dataset_path)
    
    # 提取节点特征和标签
    node_features = []
    labels = []
    for data in dataset:
        node_features.append(data.x.cpu().numpy())  # 节点特征
        labels.append(data.y.cpu().numpy())  # 标签 [kcat, Km]
    
    # 合并所有节点的特征
    node_features = np.vstack(node_features)
    labels = np.vstack(labels)
    
    # 构造特征名称
    element_names = [f"Element_{e}" for e in element_encoder.categories_[0]]
    residue_names = [f"Residue_{r}" for r in residue_encoder.categories_[0]]
    feature_names = element_names + residue_names + ['Is_Ligand', 'Min_Distance']
    
    # 为了加快计算，随机采样部分节点（比如 10,000 个节点）
    np.random.seed(42)
    sample_size = min(10000, node_features.shape[0])
    indices = np.random.choice(node_features.shape[0], sample_size, replace=False)
    node_features_sampled = node_features[indices]
    labels_sampled = labels[indices]
    
    # 训练一个代理模型（XGBoost）来解释 GNN 的预测
    # 分别对 kcat 和 Km 训练模型
    models = {}
    for i, target in enumerate(['kcat', 'Km']):
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model.fit(node_features_sampled, labels_sampled[:, i])
        models[target] = model
    
    # 使用 SHAP 解释模型
    explainer = {}
    shap_values = {}
    for target in ['kcat', 'Km']:
        explainer[target] = shap.TreeExplainer(models[target])
        shap_values[target] = explainer[target](node_features_sampled)
    
    # 可视化：生成 SHAP summary plot
    for target in ['kcat', 'Km']:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[target], node_features_sampled, feature_names=feature_names, 
                          show=False, plot_type="bar")
        plt.title(f'SHAP Feature Importance for {target}', fontsize=14)
        plt.xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'shap_importance_{target}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
        # 更详细的 beeswarm plot，展示特征值对预测的正负影响
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[target], node_features_sampled, feature_names=feature_names, 
                          show=False)
        plt.title(f'SHAP Beeswarm Plot for {target}', fontsize=14)
        plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'shap_beeswarm_{target}.png'), dpi=300, bbox_inches='tight')
        plt.close()

# ==== 运行 ====
if __name__ == "__main__":
    shap_feature_importance(SAVE_PATH, SAVE_DIR)
    print(f"✅ SHAP feature importance plots saved to {SAVE_DIR}/shap_importance_*.png")