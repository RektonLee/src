#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版训练脚本 - 针对4K+数据集的超参数优化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import os
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import GNN_model as MD
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def compute_metrics(y_true_log, y_pred_log):
    y_true_log = y_true_log.numpy()
    y_pred_log = y_pred_log.numpy()
    return {
        'MAE': mean_absolute_error(y_true_log, y_pred_log),
        'RMSE': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        'R2': r2_score(y_true_log, y_pred_log),
        'Pearson': pearsonr(y_true_log.flatten(), y_pred_log.flatten())[0]
    }

def get_optimized_hyperparams(dataset_size):
    """根据数据集大小返回优化的超参数"""
    if dataset_size < 3000:
        return {
            'hidden_dim': 256,
            'num_layers': 6,
            'heads': 8,
            'dropout': 0.1,
            'batch_size': 32,
            'lr': 1e-3,
            'weight_decay': 1e-5
        }
    elif dataset_size < 8000:  # 4K-8K
        return {
            'hidden_dim': 384,
            'num_layers': 8,
            'heads': 12,
            'dropout': 0.15,
            'batch_size': 64,
            'lr': 1e-3,
            'weight_decay': 1e-4
        }
    else:  # 8K+
        return {
            'hidden_dim': 512,
            'num_layers': 10,
            'heads': 16,
            'dropout': 0.2,
            'batch_size': 128,
            'lr': 1e-3,
            'weight_decay': 2e-4
        }

def train_optimized(dataset_path, save_dir="outputs", max_epochs=800, early_stopping_patience=50):
    dataset = torch.load(dataset_path)
    
    # 检查数据集是否包含 NaN
    for data in dataset:
        if torch.isnan(data.x).any() or torch.isnan(data.y).any():
            raise ValueError("数据集中包含 NaN 值")
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)

    # 根据数据集大小获取优化超参数
    dataset_size = len(dataset)
    hyperparams = get_optimized_hyperparams(dataset_size)
    
    print(f"数据集大小: {dataset_size}")
    print(f"优化超参数: {hyperparams}")

    # === Load dataset ===
    data_list = torch.load(dataset_path)
    print(data_list[0])
    
    np.random.shuffle(data_list)
    split = int(0.8 * len(data_list))
    train_loader = DataLoader(data_list[:split], 
                            batch_size=hyperparams['batch_size'], 
                            shuffle=True)
    val_loader = DataLoader(data_list[split:], 
                          batch_size=hyperparams['batch_size'])

    # === Initialize optimized model ===
    node_input_dim = data_list[0].x.shape[1]
    edge_input_dim = data_list[0].edge_attr.shape[1]
    print(f"Node input dim: {node_input_dim}, Edge input dim: {edge_input_dim}")
    
    model = MD.PocketGNNKcatOnly(
        node_input_dim=node_input_dim, 
        edge_input_dim=edge_input_dim,
        hidden_dim=hyperparams['hidden_dim'],
        num_layers=hyperparams['num_layers'],
        heads=hyperparams['heads'],
        dropout=hyperparams['dropout']
    ).to(device)
    
    # 优化器配置
    optimizer = optim.Adam(model.parameters(), 
                          lr=hyperparams['lr'], 
                          weight_decay=hyperparams['weight_decay'])
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                patience=20, verbose=True, min_lr=1e-6)
    
    # 损失函数 - 使用Huber Loss (对异常值更鲁棒)
    criterion = nn.HuberLoss(delta=1.0)
    
    best_val_loss = float('inf')
    patience_counter = 0

    # 记录历史
    train_loss_history = []
    val_loss_history = []
    r2_history = []
    pearson_history = []
    lr_history = []
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 只使用kcat标签
            actual_batch_size = batch.num_graphs
            y_reshaped = batch.y.reshape(actual_batch_size, 2)
            log_y = y_reshaped[:, 0:1]  # 只取kcat列
            
            out = model(batch)
            loss = criterion(out, log_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            
        train_loss = np.mean(train_losses)

        # === Validation ===
        model.eval()
        val_losses = []
        y_true_log, y_pred_log = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                actual_batch_size = batch.num_graphs
                y_reshaped = batch.y.reshape(actual_batch_size, 2)
                log_y = y_reshaped[:, 0:1]
                
                out = model(batch)
                loss = criterion(out, log_y)
                val_losses.append(loss.item())
                y_true_log.append(log_y.cpu())
                y_pred_log.append(out.cpu())
                
        val_loss = np.mean(val_losses)
        y_true_log = torch.cat(y_true_log, dim=0)
        y_pred_log = torch.cat(y_pred_log, dim=0)
        metrics = compute_metrics(y_true_log, y_pred_log)
        
        # 记录历史
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        r2_history.append(metrics['R2'])
        pearson_history.append(metrics['Pearson'])
        lr_history.append(optimizer.param_groups[0]['lr'])
        
        # 学习率调度
        scheduler.step(val_loss)

        # === Logging ===
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("R2/val", metrics['R2'], epoch)
        writer.add_scalar("Pearson/val", metrics['Pearson'], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | R2: {metrics['R2']:.3f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # === Early Stopping & Model Saving ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            print(f"✅ 新的最佳模型保存 (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"🛑 早停触发 (patience: {early_stopping_patience})")
            break

    writer.close()
    
    # 绘制训练曲线
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    epochs_range = range(1, len(train_loss_history) + 1)
    ax1.plot(epochs_range, train_loss_history, label='Train Loss')
    ax1.plot(epochs_range, val_loss_history, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # R2和Pearson曲线
    ax2.plot(epochs_range, r2_history, label='R² Score', color='green')
    ax2.plot(epochs_range, pearson_history, label='Pearson Correlation', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('R² and Pearson Correlation')
    ax2.legend()
    ax2.grid(True)
    
    # 学习率曲线
    ax3.plot(epochs_range, lr_history, color='red')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True)
    
    # 最终预测散点图
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            batch_size = batch.num_graphs
            log_y = batch.y.view(batch_size, -1)[:, 0:1]
            out = model(batch)
            all_y_true.append(log_y.cpu())
            all_y_pred.append(out.cpu())
    
    all_y_true = torch.cat(all_y_true, dim=0).numpy().flatten()
    all_y_pred = torch.cat(all_y_pred, dim=0).numpy().flatten()
    
    ax4.scatter(all_y_true, all_y_pred, alpha=0.6)
    ax4.plot([all_y_true.min(), all_y_true.max()], 
             [all_y_true.min(), all_y_true.max()], 'r--')
    ax4.set_xlabel('True kcat (log10)')
    ax4.set_ylabel('Predicted kcat (log10)')
    final_r2 = r2_score(all_y_true, all_y_pred)
    ax4.set_title(f'Final Prediction (R² = {final_r2:.3f})')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_summary.png'), dpi=300)
    plt.close()
    
    # 保存训练历史
    history_df = pd.DataFrame({
        'Epoch': epochs_range,
        'Train_Loss': train_loss_history,
        'Val_Loss': val_loss_history,
        'R2': r2_history,
        'Pearson': pearson_history,
        'Learning_Rate': lr_history
    })
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
    
    # 保存最终统计
    final_stats = {
        'Dataset_Size': dataset_size,
        'Final_R2': final_r2,
        'Best_Val_Loss': best_val_loss,
        'Total_Epochs': len(train_loss_history),
        'Hyperparameters': hyperparams
    }
    
    with open(os.path.join(save_dir, 'final_stats.json'), 'w') as f:
        import json
        json.dump(final_stats, f, indent=2)
    
    print(f"✅ 优化训练完成!")
    print(f"📊 最终R²: {final_r2:.3f}")
    print(f"📁 结果保存在: {save_dir}")
    
    return final_r2

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="kcat_dataset_enhanced1.pt")
    parser.add_argument('--save_dir', type=str, default='outputs/kcat_optimized_model')
    parser.add_argument('--max_epochs', type=int, default=800)
    parser.add_argument('--patience', type=int, default=50)
    
    args = parser.parse_args()
    
    train_optimized(args.dataset, args.save_dir, args.max_epochs, args.patience)
