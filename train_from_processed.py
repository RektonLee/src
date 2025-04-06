import sys
import os

# 获取当前脚本的目录 (src/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (enzyme_prediction/)，假设 src 目录在项目根目录下
project_root_dir = os.path.dirname(current_dir)
# 将项目根目录添加到 Python 的模块搜索路径中
sys.path.insert(0, project_root_dir)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import time
import logging
import argparse
from models.model import ImprovedEnzymePredictionModel
from sklearn.model_selection import train_test_split
import ast

def setup_logging(timestamp):
    """设置日志记录"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, f"train_processed_{timestamp}.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger()

def load_preprocessed_data(data_path):
    """加载预处理好的数据"""
    logger = logging.getLogger()
    logger.info(f"Loading preprocessed data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    def parse_numpy_array(x):
        """解析embedding字符串到numpy数组"""
        try:
            # 移除首尾的方括号
            x = x.strip('[]')
            # 处理多行格式，将所有数字提取出来
            numbers = []
            for line in x.split('\n'):
                # 清理每行的空白字符和方括号
                line = line.strip(' []')
                if line:
                    # 分割并转换为浮点数
                    nums = [float(n) for n in line.split() if n]
                    numbers.extend(nums)
            return np.array(numbers, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error parsing embedding: {str(e)}")
            logger.error(f"Problematic embedding string: {x[:100]}...")  # 打印前100个字符用于调试
            return np.zeros(768, dtype=np.float32)  # 默认embedding维度为768

    def parse_fingerprints(x):
        """解析指纹字符串到numpy数组"""
        try:
            if isinstance(x, str):
                fps = eval(x)
                # 确保每个指纹都是相同长度的数组
                return [np.array([int(bit) for bit in fp], dtype=np.float32) for fp in fps]
            return [np.zeros(2048, dtype=np.float32)]  # 默认指纹长度为2048
        except Exception as e:
            logger.error(f"Error parsing fingerprint: {str(e)}")
            return [np.zeros(2048, dtype=np.float32)]

    # 打印一些示例数据用于调试
    logger.info("Sample embedding string:")
    logger.info(df['embd'].iloc[0][:200])  # 打印第一行embedding的前200个字符
    
    # 应用转换
    df['embd'] = df['embd'].apply(parse_numpy_array)
    df['substrate_fingerprints'] = df['substrate_fingerprints'].apply(parse_fingerprints)
    
    # 验证数据形状
    sample_embedding = df['embd'].iloc[0]
    logger.info(f"Sample embedding shape: {sample_embedding.shape}")
    
    # 准备特征
    protein_features = np.stack(df['embd'].values)
    logger.info(f"Protein features shape: {protein_features.shape}")
    
    # 填充substrate_features为相同长度
    max_substrates = max(len(fps) for fps in df['substrate_fingerprints'])
    fingerprint_length = 2048  # 指纹长度
    
    def pad_fingerprints(fps):
        """填充指纹序列到固定长度"""
        padded = []
        for fp in fps[:max_substrates]:
            padded.append(fp)
        while len(padded) < max_substrates:
            padded.append(np.zeros(fingerprint_length, dtype=np.float32))
        return np.array(padded)
    
    substrate_features = np.array([pad_fingerprints(fps) for fps in df['substrate_fingerprints']])
    logger.info(f"Substrate features shape: {substrate_features.shape}")
    
    # 准备标签
    y = df[['Km_log', 'kcat_log']].values
    logger.info(f"Labels shape: {y.shape}")
    
    return protein_features, substrate_features, y

def create_processed_data_loaders(protein_features, substrate_features, y, batch_size=32):
    """创建数据加载器"""
    # 划分训练集和测试集
    indices = np.arange(len(protein_features))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    # 转换为PyTorch张量
    protein_train = torch.tensor(protein_features[train_indices], dtype=torch.float32)
    substrate_train = torch.tensor(substrate_features[train_indices], dtype=torch.float32)
    y_train = torch.tensor(y[train_indices], dtype=torch.float32)
    
    protein_test = torch.tensor(protein_features[test_indices], dtype=torch.float32)
    substrate_test = torch.tensor(substrate_features[test_indices], dtype=torch.float32)
    y_test = torch.tensor(y[test_indices], dtype=torch.float32)
    
    # 创建数据加载器
    train_dataset = TensorDataset(protein_train, substrate_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, (protein_test, substrate_test, y_test)

def train_model(model, train_loader, test_data, epochs, optimizer, criterion, scheduler, device):
    """训练模型"""
    logger = logging.getLogger()
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for protein_batch, substrate_batch, y_batch in train_loader:
            protein_batch = protein_batch.to(device)
            substrate_batch = substrate_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(protein_batch, substrate_batch)
            loss = criterion(outputs, y_batch)
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # 评估
        model.eval()
        with torch.no_grad():
            protein_test, substrate_test, y_test = test_data
            protein_test = protein_test.to(device)
            substrate_test = substrate_test.to(device)
            y_test = y_test.to(device)
            
            test_outputs = model(protein_test, substrate_test)
            test_loss = criterion(test_outputs, y_test).item()
        
        # 更新学习率
        scheduler.step(test_loss)
        
        # 保存最佳模型
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict()
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {np.mean(train_losses):.4f}, "
                   f"Test Loss: {test_loss:.4f}")
    
    return best_model_state, best_loss

def main():
    parser = argparse.ArgumentParser(description="Train model from preprocessed data")
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to preprocessed data CSV file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    args = parser.parse_args()
    
    # 设置
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(timestamp)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    protein_features, substrate_features, y = load_preprocessed_data(args.data_path)
    train_loader, test_data = create_processed_data_loaders(
        protein_features, substrate_features, y, args.batch_size
    )
    
    # 初始化模型和训练组件
    model = ImprovedEnzymePredictionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 训练模型
    best_model_state, best_loss = train_model(
        model, train_loader, test_data, args.epochs,
        optimizer, criterion, scheduler, device
    )
    
    # 保存最佳模型
    checkpoint_path = os.path.join(args.output_dir, f'best_model_{timestamp}.pth')
    torch.save(best_model_state, checkpoint_path)
    logger.info(f"Best model saved to {checkpoint_path} with loss {best_loss:.4f}")

if __name__ == '__main__':
    main()