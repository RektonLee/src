import sys
import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr  
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import seaborn as sns
import argparse
from torch.utils.data import DataLoader, TensorDataset

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 获取当前脚本的目录 (src/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_dir)
sys.path.insert(0, project_root_dir)

from models.model import ImprovedEnzymePredictionModel

def show_prediction_examples(y_true, predictions, n_examples=5):
    """展示具体的预测示例"""
    # 将log10值转换回原始值
    km_true = 10**y_true[:, 0]
    kcat_true = 10**y_true[:, 1]
    km_pred = 10**predictions[:, 0]
    kcat_pred = 10**predictions[:, 1]
    
    logging.info("\n预测示例:")
    logging.info("=" * 80)
    logging.info(f"{'序号':>6} {'真实Km':>12} {'预测Km':>12} {'Km误差%':>10} "
                f"{'真实kcat':>12} {'预测kcat':>12} {'kcat误差%':>10}")
    logging.info("-" * 80)
    
    # 计算相对误差
    km_errors = np.abs(km_true - km_pred) / km_true * 100
    kcat_errors = np.abs(kcat_true - kcat_pred) / kcat_true * 100
    
    # 选择要展示的示例
    total_error = km_errors + kcat_errors
    best_indices = np.argsort(total_error)[:n_examples//2]
    worst_indices = np.argsort(total_error)[-n_examples//2:]
    example_indices = np.concatenate([best_indices, worst_indices])
    
    for idx in example_indices:
        logging.info(f"{idx:6d} {km_true[idx]:12.6f} {km_pred[idx]:12.6f} {km_errors[idx]:10.2f} "
                    f"{kcat_true[idx]:12.6f} {kcat_pred[idx]:12.6f} {kcat_errors[idx]:10.2f}")

def evaluate_model(model, test_data, device='cuda:1'):
    """评估模型性能"""
    try:
        model.eval()
        model = model.to(device)
        
        # 准备数据
        binding_site_test = torch.FloatTensor(test_data['binding_site_features']).to(device)
        substrate_test = torch.FloatTensor(test_data['substrate_features']).to(device)
        y_test = test_data['y']
        
        with torch.no_grad():
            predictions = model(binding_site_test, substrate_test).cpu().numpy()
            
            # 计算评估指标 (log10尺度)
            metrics = {
                'mse_km': mean_squared_error(y_test[:, 0], predictions[:, 0]),
                'mse_kcat': mean_squared_error(y_test[:, 1], predictions[:, 1]),
                'r2_km': r2_score(y_test[:, 0], predictions[:, 0]),
                'r2_kcat': r2_score(y_test[:, 1], predictions[:, 1]),
                'pearson_km': pearsonr(y_test[:, 0], predictions[:, 0])[0],
                'pearson_kcat': pearsonr(y_test[:, 1], predictions[:, 1])[0]
            }
            
            # 计算原始尺度的相对误差
            km_true = np.power(10, y_test[:, 0])
            kcat_true = np.power(10, y_test[:, 1])
            km_pred = np.power(10, predictions[:, 0])
            kcat_pred = np.power(10, predictions[:, 1])
            
            rel_error_km = np.abs(km_true - km_pred) / km_true * 100
            rel_error_kcat = np.abs(kcat_true - kcat_pred) / kcat_true * 100
            
            metrics.update({
                'median_rel_error_km': np.median(rel_error_km),
                'median_rel_error_kcat': np.median(rel_error_kcat)
            })
            
            return metrics, predictions
    except Exception as e:
        logging.error(f"评估过程中出错: {str(e)}")
        raise Exception(f"评估过程中出错: {str(e)}")

def plot_scatter_predictions(y_true, predictions, save_path='output/evaluation'):
    """绘制预测值与真实值的散点图"""
    try:
        os.makedirs(save_path, exist_ok=True)
        
        # # 设置中文字体
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Km散点图
        km_true = 10**y_true[:, 0]
        km_pred = 10**predictions[:, 0]
        ax1.scatter(km_true, km_pred, alpha=0.5)
        ax1.set_xlabel('True Km (μM)', fontsize=12)
        ax1.set_ylabel('Prediction Km (μM)', fontsize=12)
        ax1.set_title(f'Km Prediction Scatter Plot \nR² = {r2_score(y_true[:, 0], predictions[:, 0]):.4f}\n'
                     f'Pearson r = {pearsonr(y_true[:, 0], predictions[:, 0])[0]:.4f}', 
                     fontsize=14)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # 添加Km对角线
        km_min = min(km_true.min(), km_pred.min())
        km_max = max(km_true.max(), km_pred.max())
        ax1.plot([km_min, km_max], [km_min, km_max], 'r--', label='Ideal Prediction Line')
        ax1.legend()
        
        # kcat散点图
        kcat_true = 10**y_true[:, 1]
        kcat_pred = 10**predictions[:, 1]
        ax2.scatter(kcat_true, kcat_pred, alpha=0.5)
        ax2.set_xlabel('True kcat (s⁻¹)', fontsize=12)
        ax2.set_ylabel('Prediction kcat值 (s⁻¹)', fontsize=12)
        ax2.set_title(f'kcat Prediction scatter plot\nR² = {r2_score(y_true[:, 1], predictions[:, 1]):.4f}\n'
                     f'Pearson r = {pearsonr(y_true[:, 1], predictions[:, 1])[0]:.4f}', 
                     fontsize=14)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # 添加kcat对角线
        kcat_min = min(kcat_true.min(), kcat_pred.min())
        kcat_max = max(kcat_true.max(), kcat_pred.max())
        ax2.plot([kcat_min, kcat_max], [kcat_min, kcat_max], 'r--', label='Ideal Prediction Line')
        ax2.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_name = os.path.join(save_path, f'predictions_scatter_{timestamp}.png')
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        logging.info(f"散点图已保存至: {save_name}")
        
        plt.close()
    except Exception as e:
        logging.error(f"绘制散点图时出错: {str(e)}")
        raise


def plot_error_distribution(y_true, predictions, save_path='output/evaluation'):
    """绘制预测误差分布图"""
    try:
        os.makedirs(save_path, exist_ok=True)
        
        # 计算相对误差
        km_true = 10**y_true[:, 0]
        kcat_true = 10**y_true[:, 1]
        km_pred = 10**predictions[:, 0]
        kcat_pred = 10**predictions[:, 1]
        
        rel_error_km = np.abs(km_true - km_pred) / km_true * 100
        rel_error_kcat = np.abs(kcat_true - kcat_pred) / kcat_true * 100
        
        # 将误差限制在0-100%范围内
        rel_error_km = np.clip(rel_error_km, 0, 100)
        rel_error_kcat = np.clip(rel_error_kcat, 0, 100)
        
        # 创建图像
        plt.figure(figsize=(12, 5))
        
        # Km误差分布
        plt.subplot(1, 2, 1)
        sns.histplot(rel_error_km, bins=30)
        plt.xlabel('Km Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title(f'Km Prediction Error Distribution\nMedian Error: {np.median(rel_error_km):.2f}%')
        plt.xlim(0, 100)
        
        # kcat误差分布
        plt.subplot(1, 2, 2)
        sns.histplot(rel_error_kcat, bins=30)
        plt.xlabel('kcat Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title(f'kcat Prediction Error Distribution\nMedian Error: {np.median(rel_error_kcat):.2f}%')
        plt.xlim(0, 100)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_name = os.path.join(save_path, f'error_distribution_{timestamp}.png')
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        logging.info(f"误差分布图已保存至: {save_name}")
        
        plt.close()
    except Exception as e:
        logging.error(f"绘制误差分布图时出错: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="评估酶动力学预测模型")
        parser.add_argument('--model_path', type=str, required=True, 
                           help='模型检查点路径')
        parser.add_argument('--data_path', type=str, required=True,
                           help='NPZ数据文件路径')
        parser.add_argument('--device', type=str, default='cuda:1',
                           help='使用的设备')
        parser.add_argument('--output_dir', type=str, default='output/evaluation',
                           help='输出目录')
        args = parser.parse_args()

        # 检查文件是否存在
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"找不到模型文件: {args.model_path}")
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"找不到数据文件: {args.data_path}")
            
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)

        # 设置日志文件
        log_file = os.path.join(args.output_dir, f'evaluation_{time.strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)

        logging.info(f"开始评估 - 使用设备: {args.device}")
        logging.info(f"模型路径: {args.model_path}")
        logging.info(f"数据路径: {args.data_path}")
        
        # 加载数据
        data = np.load(args.data_path)
        if 'test_indices' not in data:
            raise ValueError("数据文件中缺少test_indices")
            
        # 准备测试数据
        test_data = {
            'binding_site_features': data['binding_site_features'][data['test_indices']],
            'substrate_features': data['substrate_embeddings'][data['test_indices']],
            'y': data['y'][data['test_indices']]
        }
        
        logging.info(f"测试集大小: {len(test_data['y'])}")
        
        # 加载模型
        model = ImprovedEnzymePredictionModel(
            binding_site_dim=7,
            substrate_dim=384
        )
        
        # 加载模型权重
        checkpoint = torch.load(args.model_path, map_location=args.device)
        if 'model_state_dict' not in checkpoint:
            raise ValueError("模型文件格式不正确，缺少model_state_dict")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 评估模型
        metrics, predictions = evaluate_model(model, test_data, device=args.device)
        
        # 绘制图像
        plot_scatter_predictions(test_data['y'], predictions, save_path=args.output_dir)
        plot_error_distribution(test_data['y'], predictions, save_path=args.output_dir)
        
        # 展示预测示例
        show_prediction_examples(test_data['y'], predictions, n_examples=6)
        
        # 输出评估结果
        logging.info("\n评估结果:")
        logging.info(f"MSE - Km: {metrics['mse_km']:.6f}, kcat: {metrics['mse_kcat']:.6f}")
        logging.info(f"R² - Km: {metrics['r2_km']:.4f}, kcat: {metrics['r2_kcat']:.4f}")
        logging.info(f"Pearson r - Km: {metrics['pearson_km']:.4f}, kcat: {metrics['pearson_kcat']:.4f}")
        logging.info(f"相对误差中位数 - Km: {metrics['median_rel_error_km']:.2f}%, "
                    f"kcat: {metrics['median_rel_error_kcat']:.2f}%")
                    
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        raise
