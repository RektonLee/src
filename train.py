import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import sys
import time
from datetime import datetime
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader as GeometricDataLoader
# 导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ImprovedEnzymePredictionModel
from data_loader import load_and_preprocess_data

# 设置日志
def setup_logger(name, log_file, level=logging.INFO):
    """设置logger"""
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # 添加控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def evaluate_metrics(model, data_loader, device):
    """评估模型性能，计算R²等指标"""
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for binding_site, substrate, targets in data_loader:
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
    
    # 计算R²
    r2_km = r2_score(all_targets[:, 0], all_predictions[:, 0])
    r2_kcat = r2_score(all_targets[:, 1], all_predictions[:, 1])
    
    # 计算MAE
    mae_km = np.mean(np.abs(all_predictions[:, 0] - all_targets[:, 0]))
    mae_kcat = np.mean(np.abs(all_predictions[:, 1] - all_targets[:, 1]))
    
    return {
        'loss': total_loss / len(data_loader),
        'r2_km': r2_km,
        'r2_kcat': r2_kcat,
        'mae_km': mae_km,
        'mae_kcat': mae_kcat
    }

import matplotlib.pyplot as plt

def train_model(data_path=None, processed_data_path=None, epochs=100, batch_size=16, 
                lr=0.001, weight_decay=1e-5):
    """训练模型主函数"""
    # 创建输出目录
    timestamp = datetime.now().strftime('%m%d_%H')
    output_dir = f"output/pdb_smile_data/training_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger('training', os.path.join(output_dir, 'training.log'))
    logger.info(f"开始训练 - 时间: {timestamp}")
    logger.info(f"数据路径: {data_path if data_path else processed_data_path}")
    
    # 检查CUDA可用性
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
   # 加载数据
    if processed_data_path and os.path.exists(processed_data_path):
        logger.info(f"从预处理数据加载: {processed_data_path}")
        data = np.load(processed_data_path)
        substrate_embeddings = data['substrate_embeddings']
        y = data['y']
        train_indices = data['train_indices']
        test_indices = data['test_indices']
        binding_site_graphs = torch.load(processed_data_path.replace('.npz', '.pt'))
        
        train_loader, test_loader = create_data_loaders(
            binding_site_graphs, substrate_embeddings, y, 
            train_indices, test_indices, batch_size=batch_size
        )
    else:
        logger.info(f"从原始数据加载并预处理: {data_path}")
        train_loader, test_loader, binding_site_graphs, substrate_embeddings, y, train_indices, test_indices = load_and_preprocess_data(
            data_path, timestamp=timestamp
        )
    
    # 初始化模型
    model = ImprovedEnzymePredictionModel(
        node_feature_dim=30,  # 与 extract_binding_site 的特征维度匹配
        substrate_dim=substrate_embeddings.shape[1],
    ).to(device)
    
    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    r2_km_scores = []
    r2_kcat_scores = []
    
    # 训练循环
    best_val_metrics = {'loss': float('inf')}
    early_stopping_counter = 0
    early_stopping_patience = 20
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch in train_loader:
            binding_site, substrate, targets = batch  # batch 是 (Data, tensor, tensor)
            binding_site = binding_site.to(device)  # 图数据自动处理
            substrate = substrate.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(binding_site, substrate)
            loss = model.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 评估阶段
        val_metrics = evaluate_metrics(model, test_loader, device)
        scheduler.step(val_metrics['loss'])
        
        # 记录训练信息
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_metrics['loss'])
        r2_km_scores.append(val_metrics['r2_km'])
        r2_kcat_scores.append(val_metrics['r2_kcat'])
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss/len(train_loader):.6f}, "
            f"Val Loss: {val_metrics['loss']:.6f}, "
            f"Km R²: {val_metrics['r2_km']:.4f}, "
            f"kcat R²: {val_metrics['r2_kcat']:.4f}"
        )
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_metrics['loss']:
            best_val_metrics = val_metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # 早停
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 6))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    
    # 绘制 R² 曲线
    plt.subplot(1, 2, 2)
    plt.plot(r2_km_scores, label="Km R²")
    plt.plot(r2_kcat_scores, label="kcat R²")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("R² Scores")
    plt.legend()
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()
    
    # 训练结束，记录最终结果
    logger.info("\n最终结果:")
    logger.info(f"最佳验证损失: {best_val_metrics['loss']:.6f}")
    logger.info(f"最佳Km R²: {best_val_metrics['r2_km']:.4f}")
    logger.info(f"最佳kcat R²: {best_val_metrics['r2_kcat']:.4f}")
    
    return model, best_val_metrics


def create_data_loaders(binding_site_graphs, substrate_features, y, train_indices, test_indices, batch_size=16):
    # 分割训练和测试数据
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
    # return train_loader, test_loader
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练酶动力学参数预测模型")
    parser.add_argument("--data_path", default="/home/lizihao/Work/enzyme_prediction/data/cleaned_data.csv",type=str, help="原始数据路径")
    parser.add_argument("--processed_data", type=str, help="预处理数据路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    
    args = parser.parse_args()
    
    if not args.data_path and not args.processed_data:
        print("错误：必须提供--data_path或--processed_data参数")
        sys.exit(1)
    
    # 训练模型
    model, best_metrics = train_model(
        data_path=args.data_path,
        processed_data_path=args.processed_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay
    )