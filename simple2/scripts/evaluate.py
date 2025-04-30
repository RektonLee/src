import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.loader import DataLoader
from GNN_model import PocketGNN1
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from scipy import stats
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

def compute_metrics(y_true, y_pred):
    y_true = y_true.numpy() if torch.is_tensor(y_true) else np.array(y_true)
    y_pred = y_pred.numpy() if torch.is_tensor(y_pred) else np.array(y_pred)
    
    # 分别计算kcat和Km的指标
    metrics = {}
    for i, name in enumerate(['kcat', 'Km']):
        metrics.update({
            f'{name}_MAE': mean_absolute_error(y_true[:,i], y_pred[:,i]),
            f'{name}_R2': r2_score(y_true[:,i], y_pred[:,i]),
            f'{name}_Pearson': pearsonr(y_true[:,i], y_pred[:,i])[0]
        })
    
    # 添加综合指标
    metrics.update({
        'Overall_MAE': mean_absolute_error(y_true, y_pred),
        'Overall_R2': r2_score(y_true, y_pred)
    })
    return metrics
    # 将y_true从[N*2] reshape为[N,2]并取对数
    # y_true = y_true.view(-1, 2).numpy()
    y_true_log =y_true # 避免log(0)
    
    y_pred = y_pred.numpy()
    
    return {
        'MAE': mean_absolute_error(y_true_log, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true_log, y_pred)),
        'R2': r2_score(y_true_log, y_pred),
        'Pearson': pearsonr(y_true_log.flatten(), y_pred.flatten())[0]
    }

def scatter_plot(y_true, y_pred, save_path):
    # 这里已经分别绘制了kcat和Km的散点图
    plt.scatter(y_true[:,0], y_pred[:,0], label='kcat')  # 第一维是kcat
    plt.scatter(y_true[:,1], y_pred[:,1], label='Km')    # 第二维是Km
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Predicted vs True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_embeddings(model, loader, save_path, device):
    model.eval()
    embeddings, labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # 获取最后一层GNN的节点嵌入
            emb = model.get_graph_embedding(batch) 
            embeddings.append(emb.cpu())
            labels.append(batch.y.cpu())
    
    # 合并所有batch的数据
    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(embeddings)
    
    # 可视化
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], 
                    hue=labels[:,0], palette="viridis", 
                    alpha=0.7, size=labels[:,1])
    plt.title('t-SNE Visualization of Graph Embeddings')
    plt.savefig(os.path.join(save_path, 'tsne_embedding.png'))
    plt.close()

def evaluate(dataset_path, model_path, save_dir="outputs"):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    dataset = torch.load(dataset_path)
    loader = DataLoader(dataset, batch_size=32)

    # Load model
        # Load model
    sample_data = dataset[0]
    node_feature_dim = sample_data.x.size(1)  # 自动获取节点特征维度
    edge_feature_dim = sample_data.edge_attr.size(1)  # 自动获取边特征维度
    model = PocketGNN1(
        node_input_dim=node_feature_dim,
        edge_input_dim=edge_feature_dim,# 可根据需要调整
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            
            # 处理y_true: 展平后取log10
            true_log = torch.log10(batch.y.view(-1, 2))  # [N,2]
            y_true.append(true_log.cpu())
            
            # 处理y_pred: 直接使用模型输出 [N,2]
            y_pred.append(pred.cpu())
    
            # 合并时保持[N,2]形状
           
        y_true = torch.cat(y_true, dim=0)  # [total_samples, 2]
        y_pred = torch.cat(y_pred, dim=0)   # [total_samples, 2]
    metrics = compute_metrics(y_true, y_pred)
    print("\n📊 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    scatter_plot(y_true.numpy(), y_pred.numpy(), os.path.join(save_dir, "scatter_plot.png"))
    print(f"✅ Scatter plot saved to {save_dir}/scatter_plot.png")

def residual_analysis(y_true, y_pred, save_dir):
    residuals = y_true - y_pred
    
    plt.figure(figsize=(15,5))
    
    # 残差分布
    plt.subplot(131)
    sns.histplot(residuals.flatten(), kde=True)
    plt.title('Residual Distribution')
    
    # 残差vs预测值
    plt.subplot(132)
    plt.scatter(y_pred.flatten(), residuals.flatten(), alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Predicted')
    
    # QQ图
    plt.subplot(133)
    stats.probplot(residuals.flatten(), plot=plt)
    plt.title('Q-Q Plot')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residual_analysis.png'))
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/home/lizihao/Work/enzyme_prediction/src/simple2/data/processed/dataset1.pt')
    parser.add_argument('--model', type=str, default='/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/1/best_model.pt')
    parser.add_argument('--save_dir', type=str, default='outputs/v1')
    args = parser.parse_args()
    evaluate(args.dataset, args.model, args.save_dir)
