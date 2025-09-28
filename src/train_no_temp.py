import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch_geometric.loader import DataLoader
# from torch.utils.tensorboard import SummaryWriter  # 暂时注释掉，避免依赖问题
import GNN_model as MD
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')  # 确保在没有GUI的环境中使用
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

def train(dataset_path, save_dir="outputs", batch_size=32, lr=1e-3, max_epochs=500):
    dataset = torch.load(dataset_path, weights_only=False)
    
    # 检查数据集是否包含 NaN
    for data in dataset:
        if torch.isnan(data.x).any() or torch.isnan(data.y).any():
            raise ValueError("数据集中包含 NaN 值")
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    # writer = SummaryWriter(save_dir)  # 暂时注释掉

    # === Load dataset ===
    data_list = torch.load(dataset_path, weights_only=False)  # List[Data]
    print(data_list[0])  # 打印第一个图数据
    actual_num_atom_types = data_list[0].x.shape[1]
    np.random.shuffle(data_list)
    split = int(0.8 * len(data_list))
    train_loader = DataLoader(data_list[:split], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_list[split:], batch_size=batch_size)

    # === Initialize model (使用不含温度项的模型) ===
    node_input_dim = data_list[0].x.shape[1] #default 10
    edge_input_dim = data_list[0].edge_attr.shape[1]
    model = MD.PocketGNNWithAttentionNoTemp(node_input_dim=node_input_dim, edge_input_dim=edge_input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    # 添加损失记录列表
    train_loss_history = []
    val_loss_history = []
    r2_history = []
    pearson_history = []
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 重新组织标签 - 将相邻的kcat和Km配对
            actual_batch_size = batch.num_graphs  # 使用实际的 batch size
            y_reshaped = batch.y.reshape(actual_batch_size, 2)  # 每两个值组成一对[kcat, Km]
            log_y = y_reshaped
            
            if torch.isnan(batch.x).any():
                print("❌ batch.x 中含有 NaN")
            if torch.isnan(batch.edge_attr).any():
                print("❌ batch.edge_attr 中含有 NaN")
            if torch.isnan(batch.y).any():
                print("❌ batch.y 中含有 NaN")
            
            out = model(batch)
           
            loss = criterion(out, log_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
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
                
                # 重新组织标签
                actual_batch_size = batch.num_graphs  # 使用实际的 batch size
                y_reshaped = batch.y.reshape(actual_batch_size, 2)  # 每两个值组成一对[kcat, Km]
                log_y = y_reshaped
                
                try:
                    out = model(batch)
                except ValueError as e:
                    print(f"❌ NaN 输出，batch中数据文件: {[d.pdb_id for d in batch]}")
                    raise e
                loss = criterion(out, log_y)
                val_losses.append(loss.item())
                y_true_log.append(log_y.cpu())
                y_pred_log.append(out.cpu())
        val_loss = np.mean(val_losses)
        y_true_log = torch.cat(y_true_log, dim=0)
        y_pred_log = torch.cat(y_pred_log, dim=0)
        metrics = compute_metrics(y_true_log, y_pred_log)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        r2_history.append(metrics['R2'])
        pearson_history.append(metrics['Pearson'])
        # === Logging ===
        # writer.add_scalar("Loss/train", train_loss, epoch)  # 暂时注释掉
        # writer.add_scalar("Loss/val", val_loss, epoch)
        # writer.add_scalar("R2/val", metrics['R2'], epoch)
        # writer.add_scalar("Pearson/val", metrics['Pearson'], epoch)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | R2: {metrics['R2']:.3f}")

        # === Save best model ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

    # writer.close()  # 暂时注释掉
    print("✅ Training finished. Best model saved.")

    # 训练结束后绘制图像
    # 1. 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_epochs + 1), train_loss_history, label='Train Loss')
    plt.plot(range(1, max_epochs + 1), val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (No Temperature)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()
    
    # 2. R2和Pearson相关系数曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_epochs + 1), r2_history, label='R² Score')
    plt.plot(range(1, max_epochs + 1), pearson_history, label='Pearson Correlation')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('R² and Pearson Correlation During Training (No Temperature)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'metrics_curve.png'))
    plt.close()
    
    # 3. 预测vs真实值散点图 (使用最好的模型)
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            batch_size = batch.num_graphs

            log_y = batch.y.view(batch_size, -1)
            try:
                out = model(batch)
            except ValueError as e:
                print(f"❌ NaN 输出，batch中数据文件: {[d.pdb_id for d in batch]}")
                raise e
            all_y_true.append(log_y.cpu())  # 转回原始值
            all_y_pred.append(out.cpu())    # 转回原始值
    
    all_y_true = torch.cat(all_y_true, dim=0).numpy()
    all_y_pred = torch.cat(all_y_pred, dim=0).numpy()
    
    # 分别绘制kcat和Km的散点图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # kcat散点图
    ax1.scatter(all_y_true[:, 0], all_y_pred[:, 0], alpha=0.6)
    ax1.plot([all_y_true[:, 0].min(), all_y_true[:, 0].max()], 
             [all_y_true[:, 0].min(), all_y_true[:, 0].max()], 'r--')
    ax1.set_xlabel('True kcat')
    ax1.set_ylabel('Predicted kcat')
    r2_kcat = r2_score(all_y_true[:, 0], all_y_pred[:, 0])
    ax1.set_title(f'kcat: True vs Predicted (R² = {r2_kcat:.3f})')
    
    # Km散点图
    ax2.scatter(all_y_true[:, 1], all_y_pred[:, 1], alpha=0.6)
    ax2.plot([all_y_true[:, 1].min(), all_y_true[:, 1].max()], 
             [all_y_true[:, 1].min(), all_y_true[:, 1].max()], 'r--')
    ax2.set_xlabel('True Km')
    ax2.set_ylabel('Predicted Km')
    r2_km = r2_score(all_y_true[:, 1], all_y_pred[:, 1])
    ax2.set_title(f'Km: True vs Predicted (R² = {r2_km:.3f})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_scatter.png'))
    plt.close()
    
    # 4. 热力图
    for i, param_name in enumerate(['kcat', 'Km']):
        plt.figure(figsize=(8, 7))
        true_vals = all_y_true[:, i]
        pred_vals = all_y_pred[:, i]
        sns.kdeplot(x=true_vals, y=pred_vals, cmap="viridis", fill=True, thresh=0.05)
        plt.plot([true_vals.min(), true_vals.max()], 
                 [true_vals.min(), true_vals.max()], 'r--')
        plt.xlabel(f'True {param_name}')
        plt.ylabel(f'Predicted {param_name}')
        plt.title(f'{param_name}: Density Plot (No Temperature)')
        plt.savefig(os.path.join(save_dir, f'{param_name}_density.png'))
        plt.close()
    
    # 保存最终指标到文件
    metrics_df = pd.DataFrame({
        'Epoch': range(1, max_epochs + 1),
        'Train_Loss': train_loss_history,
        'Val_Loss': val_loss_history,
        'R2': r2_history,
        'Pearson': pearson_history
    })
    metrics_df.to_csv(os.path.join(save_dir, 'training_metrics.csv'), index=False)
    
    print("✅ Training finished. Best model and plots saved to", save_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="/home/lizihao/Work/enzyme_prediction/src/simple2/data/processed/dataset_NAN_nopqr_rbf_no_temp.pt", help='Path to .pt dataset')
    parser.add_argument('--save_dir', type=str, default='outputs/nopqr_attention_rbf_no_temp')
    args = parser.parse_args()
    from utils.metadata_utils import save_metadata

    # 训练开始时，加上这行保存metadata
    save_metadata(
        save_dir=args.save_dir,
        dataset_path=args.dataset,
        graph_builder_version='builder_rbf_no_temp',
        gnn_model_version='PocketGNNwithAttentionNoTemp',
        comments='Attention + rbf, no temperature'
    )

    train(args.dataset, args.save_dir)
