import sys
import os
import numpy as np
import torch
import shap

import matplotlib
matplotlib.use('tkagg') 
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# 设置项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_dir)
sys.path.insert(0, project_root_dir)

from models.model import ImprovedEnzymePredictionModel

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_with_shap(model, test_data, device='cuda:1', output_dir='output/evaluation'):
    """评估模型并计算SHAP值"""
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    model = model.to(device)
    
    # 准备测试数据
    binding_site_test = torch.FloatTensor(test_data['binding_site_features']).to(device)
    substrate_test = torch.FloatTensor(test_data['substrate_features']).to(device)
    y_test = test_data['y']
    
    # 转换为numpy数组，用于SHAP计算
    binding_site_np = binding_site_test.cpu().numpy()
    substrate_np = substrate_test.cpu().numpy()
    
    # 展平特征，便于SHAP处理
    binding_site_flat = binding_site_np.reshape(binding_site_np.shape[0], -1)  # [n_samples, 100*7]
    substrate_flat = substrate_np  # [n_samples, 384]
    X_test = np.hstack([binding_site_flat, substrate_flat])  # [n_samples, 100*7 + 384]
    
    # 创建特征名称
    feature_names = []
    for i in range(100):  # 100个残基
        for j in range(7):  # 7个特征
            feature_names.append(f"binding_site_{i}_{j}")
    for i in range(substrate_flat.shape[1]):
        feature_names.append(f"substrate_{i}")
    
    # 定义模型预测函数（SHAP需要numpy输入输出）
    def model_predict(X):
        binding_site = X[:, :100*7].reshape(-1, 100, 7)
        substrate = X[:, 100*7:]
        
        binding_site_tensor = torch.FloatTensor(binding_site).to(device)
        substrate_tensor = torch.FloatTensor(substrate).to(device)
        
        with torch.no_grad():
            predictions = model(binding_site_tensor, substrate_tensor).cpu().numpy()
        return predictions
    
    # 选择少量样本进行SHAP计算（SHAP计算成本较高）
    X_test_sample = X_test[:100]  # 选择前100个样本
    y_test_sample = y_test[:100]
    
    # 计算SHAP值
    logging.info("开始计算SHAP值...")
    explainer = shap.KernelExplainer(model_predict, X_test_sample)
    
    # 计算SHAP值（对Km和kcat分别计算）
    shap_values = explainer.shap_values(X_test_sample, nsamples=200)
    
    # 记录形状信息用于调试
    logging.info(f"SHAP values shape (Km): {shap_values[0].shape}")
    logging.info(f"SHAP values shape (kcat): {shap_values[1].shape}")
    logging.info(f"X_test_sample shape: {X_test_sample.shape}")
    logging.info(f"Number of features: {len(feature_names)}")
    
    # shap_values[0] 对应Km预测，shap_values[1] 对应kcat预测
    shap_values_km = shap_values[0]
    shap_values_kcat = shap_values[1]
    
    # 可视化：Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_km, 
        X_test_sample,
        feature_names=feature_names,
        plot_type="bar", 
        max_display=20,
        show=False
    )
    plt.title("Top 20 Feature Importance for Km Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_km.png"), bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_kcat, 
        X_test_sample,
        feature_names=feature_names,
        plot_type="bar", 
        max_display=20,
        show=False
    )
    plt.title("Top 20 Feature Importance for kcat Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_kcat.png"), bbox_inches='tight')
    plt.close()
    
    # 可视化：Force Plot（单个样本）
    sample_idx = 0
    plt.figure(figsize=(15, 3))
    shap.force_plot(
        explainer.expected_value[0], 
        shap_values_km[sample_idx], 
        X_test_sample[sample_idx],
        feature_names=feature_names,
        matplotlib=True, 
        show=False
    )
    plt.title(f"SHAP Force Plot for Km (Sample {sample_idx})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_force_km_sample_{sample_idx}.png"), 
                bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(15, 3))
    shap.force_plot(
        explainer.expected_value[1], 
        shap_values_kcat[sample_idx], 
        X_test_sample[sample_idx],
        feature_names=feature_names,
        matplotlib=True, 
        show=False
    )
    plt.title(f"SHAP Force Plot for kcat (Sample {sample_idx})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_force_kcat_sample_{sample_idx}.png"), 
                bbox_inches='tight')
    plt.close()
    
    # 常规评估指标
    predictions = model_predict(X_test)
    metrics = {
        'mse_km': mean_squared_error(y_test[:, 0], predictions[:, 0]),
        'mse_kcat': mean_squared_error(y_test[:, 1], predictions[:, 1]),
        'r2_km': r2_score(y_test[:, 0], predictions[:, 0]),
        'r2_kcat': r2_score(y_test[:, 1], predictions[:, 1]),
        'pearson_km': pearsonr(y_test[:, 0], predictions[:, 0])[0],
        'pearson_kcat': pearsonr(y_test[:, 1], predictions[:, 1])[0]
    }
    
    logging.info("\n评估结果:")
    logging.info(f"R² - Km: {metrics['r2_km']:.4f}, kcat: {metrics['r2_kcat']:.4f}")
    logging.info(f"Pearson r - Km: {metrics['pearson_km']:.4f}, kcat: {metrics['pearson_kcat']:.4f}")
    
    return metrics, predictions

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="评估模型并计算SHAP值")
    parser.add_argument('--model_path', 
                       default="/home/lizihao/Work/enzyme_prediction/output/training_20250310_205850/checkpoints/best_model.pth",
                       type=str, help='模型检查点路径')
    parser.add_argument('--data_path',
                       default="/home/lizihao/Work/enzyme_prediction/output/processed/processed_data_20250310_205850.npz", 
                       type=str, help='NPZ数据文件路径')
    parser.add_argument('--device', type=str, default='cuda:1', help='使用的设备')
    parser.add_argument('--output_dir', type=str, default='output/evaluation', help='输出目录')
    args = parser.parse_args()
    
    # 加载数据
    data = np.load(args.data_path)
    test_data = {
        'binding_site_features': data['binding_site_features'][data['test_indices']],
        'substrate_features': data['substrate_embeddings'][data['test_indices']],
        'y': data['y'][data['test_indices']]
    }
    
    # 加载模型
    model = ImprovedEnzymePredictionModel(binding_site_dim=7, substrate_dim=384)
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估并计算SHAP值
    metrics, predictions = evaluate_with_shap(model, test_data, device=args.device, output_dir=args.output_dir)