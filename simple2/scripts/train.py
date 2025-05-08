import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import GNN_model as MD
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')  # 确保在没有GUI的环境中使用
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.metadata_utils import save_metadata

# 修改compute_metrics函数
def compute_metrics(y_true_log, y_pred_log):
    y_true_log = y_true_log.numpy()
    y_pred_log = y_pred_log.numpy()
    y_true_original = 10**y_true_log 
    y_pred_original = 10**y_pred_log 
    return {
        'MAE': mean_absolute_error(y_true_log, y_pred_log),
        'RMSE': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        'R2_log': r2_score(y_true_log, y_pred_log),
        'R2_original': r2_score(y_true_original, y_pred_original),
        'Pearson': pearsonr(y_true_log.flatten(), y_pred_log.flatten())[0]
    }

def train(dataset_path, save_dir="outputs", batch_size=32, lr=1e-3, max_epochs=500, hidden_dim=256, num_layers=6, heads=8, dropout=0.1, optimizer_type='Adam', weight_decay=0.0, loss_fn_type='Mixed', lr_scheduler=None):
    dataset = torch.load(dataset_path)
    all_y = torch.cat([10**data.y for data in data_list])
    mean=all_y.mean().item()
    std=all_y.std().item()
    metadata.update({
    'mean': mean,
    'std': std
})
    save_metadata(save_dir=save_dir, **metadata)
    # 检查数据集是否包含 NaN
    for data in dataset:
        if torch.isnan(data.x).any() or torch.isnan(data.y).any():
            raise ValueError("数据集中包含 NaN 值")

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)

    # === Load dataset ===
    data_list = torch.load(dataset_path)  # List[Data]
    print(data_list[0])  # 打印第一个图数据
    actual_num_atom_types = data_list[0].x.shape[1]
    np.random.shuffle(data_list)
    split = int(0.8 * len(data_list))
    train_loader = DataLoader(data_list[:split], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_list[split:], batch_size=batch_size)

    # === Initialize model ===
    node_input_dim = data_list[0].x.shape[1] #default 10
    edge_input_dim = data_list[0].edge_attr.shape[1]
    model = MD.PocketGNNWithAttention(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout
    ).to(device)

    # === Initialize optimizer ===
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    # === Initialize loss function ===
    # 修改损失函数计算方式
    if loss_fn_type == 'MSE':
        criterion = nn.MSELoss()
    elif loss_fn_type == 'L1':
        criterion = nn.L1Loss()
    elif loss_fn_type == 'SmoothL1':
        criterion = nn.SmoothL1Loss()
    # 可以尝试混合损失
    elif loss_fn_type == 'Mixed':
        def mixed_loss(pred, target):
            normalized_pred = (pred - mean) / std
            normalized_target = (target - mean) / std
            mse = nn.MSELoss()(normalized_pred, normalized_target)
            # original_loss = nn.MSELoss()(10**pred, 10**target)
            return mse
        criterion = mixed_loss
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn_type}")

    best_val_loss = float('inf')

    # 添加损失记录列表
    train_loss_history = []
    val_loss_history = []
    r2_history_org = []
    r2_history_log=[]
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

            # NaN checks (as before)
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
        y_true_log, y_pred_log,y_true_original,y_pred_original = [], [],[],[]
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                # 重新组织标签
                actual_batch_size = batch.num_graphs  # 使用实际的 batch size
                y_reshaped = batch.y.reshape(actual_batch_size, 2)  # 每两个值组成一对[kcat, Km]
                log_y = y_reshaped
                original_y = 10**log_y # 假设原始值是log值的指数
                original_out=10**out
                try:
                    out = model(batch)
                except ValueError as e:
                    print(f"❌ NaN 输出，batch中数据文件: {[d.pdb_id for d in batch]}")
                    raise e
                loss = criterion(out, log_y)
                val_losses.append(loss.item())
                y_true_log.append(log_y.cpu())
                y_pred_log.append(out.cpu())
                # y_true_original.append(original_y.cpu())
                # y_pred_original.append(original_out.cpu())
        val_loss = np.mean(val_losses)
        y_true_log = torch.cat(y_true_log, dim=0)
        y_pred_log = torch.cat(y_pred_log, dim=0)
        # y_true_original=torch.cat(y_true_original,dim=0)
        # y_pred_original=torch.cat(y_pred_original,dim=0)
        # 修改指标计算部分
        metrics = compute_metrics(y_true_log, y_pred_log)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        r2_history_org.append(metrics['R2_original'])
        r2_history_log.append(metrics['R2_log'])
        pearson_history.append(metrics['Pearson'])
        # === Logging ===
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("R2/val", metrics['R2_log'], epoch)
        writer.add_scalar("Pearson/val", metrics['Pearson'], epoch)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"R2(log): {metrics['R2_log']:.3f} | R2(original): {metrics['R2_original']:.3f}")

        # === Save best model ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model_mix.pt"))

    writer.close()
    print("✅ Training finished. Best model saved.")

    # ... (rest of your plotting code remains the same) ...

    # 训练结束后绘制图像
    # 1. 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_epochs + 1), train_loss_history, label='Train Loss')
    plt.plot(range(1, max_epochs + 1), val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()
    
    # 2. R2和Pearson相关系数曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_epochs + 1), r2_history_org, label='R² Original Score')
    plt.plot(range(1, max_epochs + 1), r2_history_log, label='R² Log Score')
    plt.plot(range(1, max_epochs + 1), pearson_history, label='Pearson Correlation')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('R² and Pearson Correlation During Training')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'metrics_curve.png'))
    plt.close()
    
    # 3. 预测vs真实值散点图 (使用最好的模型)
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model_mix.pt")))
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
        plt.title(f'{param_name}: Density Plot')
        plt.savefig(os.path.join(save_dir, f'{param_name}_density.png'))
        plt.close()
    
    # 保存最终指标到文件
    metrics_df = pd.DataFrame({
        'Epoch': range(1, max_epochs + 1),
        'Train_Loss': train_loss_history,
        'Val_Loss': val_loss_history,
        'R2_log': r2_history_log,
        'R2_org': r2_history_org,
        'Pearson': pearson_history
    })
    metrics_df.to_csv(os.path.join(save_dir, 'training_metrics.csv'), index=False)
    
    print("✅ Training finished. Best model and plots saved to", save_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train GNN for Enzyme Kinetics Prediction")
    parser.add_argument('--dataset', type=str, default="/home/lizihao/Work/enzyme_prediction/src/simple2/data/processed/dataset_NAN_nopqr.pt", help='Path to .pt dataset')
    parser.add_argument('--save_dir', type=str, default='outputs/nopqr_attention_mix')

    # Model Hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads for GATConv')
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for AdamW')
    parser.add_argument('--loss_fn', type=str, default='Mixed', choices=['MSE', 'L1', 'SmoothL1'])
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=['StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'])

    args = parser.parse_args()

    # 在训练开始时，加上这行保存 metadata，包含所有超参数
    metadata = {
        'dataset_path': args.dataset,
        'graph_builder_version': 'builder2_elec_att',
        'gnn_model_version': 'PocketGNNwithAttention',
        'comments': 'HyperParameters.Enriched Pocket features (element + residue + ligand mark + local density), introduced charge, introduced attention in the graph',
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'heads': args.heads,
        'dropout': args.dropout,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'optimizer': args.optimizer,
        'weight_decay': args.weight_decay,
        'loss_fn': args.loss_fn,
        'lr_scheduler': args.lr_scheduler
        # 可以根据需要添加更多超参数
    }
    save_metadata(save_dir=args.save_dir, **metadata)

    train(args.dataset, args.save_dir,
          hidden_dim=args.hidden_dim,
          num_layers=args.num_layers,
          heads=args.heads,
          dropout=args.dropout,
          lr=args.lr,
          batch_size=args.batch_size,
          max_epochs=args.max_epochs,
          optimizer_type=args.optimizer,
          weight_decay=args.weight_decay,
          loss_fn_type=args.loss_fn,
          lr_scheduler=args.lr_scheduler)
