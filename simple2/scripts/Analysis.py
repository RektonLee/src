# final_analysis.py
# 包含残差分析、t-SNE、Input x Grad原子贡献、残基聚合、配体vs蛋白对比、电性属性相关性分析、多目标散点 + Pearson、残基类别归因统计（支持批量样本分析）

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader
from GNN_model import PocketGNN1
from sklearn.utils import shuffle

residue_list = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','LIG']

# ======= Residual Analysis =======
def residual_analysis(y_true, y_pred, save_dir, param_names=['kcat', 'Km']):
    os.makedirs(save_dir, exist_ok=True)
    residuals = y_true - y_pred
    for i, name in enumerate(param_names):
        res = residuals[:, i]
        plt.figure()
        plt.hist(res, bins=30, edgecolor='black')
        plt.title(f'{name} Residual Histogram')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(save_dir, f'{name}_residual_hist.png'))
        plt.close()

        plt.figure()
        stats.probplot(res, dist="norm", plot=plt)
        plt.title(f'{name} Residual Q-Q Plot')
        plt.savefig(os.path.join(save_dir, f'{name}_residual_qq.png'))
        plt.close()

# ======= Input Gradient Attribution =======
def compute_input_gradient(model, data, target_index=0):
    model.eval()
    data.x.requires_grad_(True)
    output = model(data)
    score = output[:, target_index].sum()
    grad = torch.autograd.grad(score, data.x, retain_graph=True)[0]
    attribution = (data.x * grad).sum(dim=1)
    return attribution.detach().cpu().numpy()

# ======= Node Attribution Bar =======
def visualize_node_attribution(attribution, save_path='atom_importance_bar.png', topk=20):
    topk_idx = np.argsort(np.abs(attribution))[-topk:][::-1]
    topk_vals = attribution[topk_idx]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=np.arange(topk), y=topk_vals)
    plt.title("Top-k Atom Contribution (Input x Gradient)")
    plt.xlabel("Atom Index")
    plt.ylabel("Importance Score")
    plt.savefig(save_path)
    plt.close()

# ======= Residue Class Attribution =======
def residue_class_importance(attribution, data, save_path='residue_class_bar.png'):
    if not hasattr(data, 'x') or not hasattr(data, 'pdb_id'):
        return
    residue_onehot = data.x[:, 10:31].detach().cpu().numpy() 
    contrib = attribution.reshape(-1, 1) * residue_onehot
    contrib_sum = contrib.sum(axis=0)
    contrib_dict = {res: val for res, val in zip(residue_list, contrib_sum)}
    sorted_items = sorted(contrib_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    labels, values = zip(*sorted_items)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(labels), y=list(values))
    plt.xticks(rotation=45)
    plt.title("Residue-level Aggregated Attribution")
    plt.ylabel("Total Contribution")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ======= kcat vs Km Scatter with Pearson =======
def plot_kcat_km_scatter(y_true, y_pred, save_path='scatter_kcat_km.png'):
    kcat_pred = y_pred[:, 0]
    km_pred = y_pred[:, 1]
    kcat_true = y_true[:, 0]
    km_true = y_true[:, 1]
    plt.figure(figsize=(8, 6))
    plt.scatter(kcat_pred, km_pred, label='Predicted', alpha=0.6)
    plt.scatter(kcat_true, km_true, label='True', alpha=0.6)
    plt.xlabel('kcat')
    plt.ylabel('Km')
    plt.legend()
    plt.title('kcat vs Km Prediction Distribution')
    plt.savefig(save_path)
    plt.close()

    r = np.corrcoef(kcat_pred, km_pred)[0,1]
    with open(save_path.replace('.png', '_pearson.txt'), 'w') as f:
        f.write(f"Pearson correlation between kcat and Km predictions: {r:.4f}\n")

# ======= t-SNE Visualization (with corrected colorbar) =======
def tsne_visualization(model, data_loader, device, save_path='tsne_kcat.png', param_index=0, max_points=1000):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            emb = model.get_graph_embedding(batch)
            embeddings.append(emb.cpu())
            labels.append(batch.y.view(-1, 2)[:, param_index].cpu())
    X = torch.cat(embeddings).numpy()
    Y = torch.cat(labels).numpy()
    X, Y = shuffle(X, Y, random_state=42)
    X, Y = X[:max_points], Y[:max_points]

    tsne = TSNE(n_components=2, perplexity=30)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, cmap='coolwarm', s=60)
    plt.colorbar(sc)
    plt.title("Graph Embedding t-SNE (colored by target)")
    plt.savefig(save_path)
    plt.close()

# ======= 主入口 =======
if __name__ == '__main__':
    DATASET_PATH = 'data/processed/dataset_NAN_nopqr.pt'
    MODEL_PATH = 'outputs/nopqr/best_model.pt'
    SAVE_DIR = 'analysis_results/nopqr'
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 8

    dataset = torch.load(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PocketGNN1(node_input_dim=dataset[0].x.shape[1], edge_input_dim=dataset[0].edge_attr.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    # ==== 残差分析 & 多目标散点图 ====
    all_y_true, all_y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            y = batch.y.view(-1, 2).cpu()
            pred = model(batch).cpu()
            all_y_true.append(y)
            all_y_pred.append(pred)

    all_y_true = torch.cat(all_y_true).numpy()
    all_y_pred = torch.cat(all_y_pred).numpy()
    residual_analysis(all_y_true, all_y_pred, SAVE_DIR)
    plot_kcat_km_scatter(all_y_true, all_y_pred, os.path.join(SAVE_DIR, 'scatter_kcat_km.png'))

    # ==== t-SNE for kcat and Km ====
    tsne_visualization(model, loader, DEVICE, save_path=os.path.join(SAVE_DIR, 'tsne_kcat.png'), param_index=0)
    tsne_visualization(model, loader, DEVICE, save_path=os.path.join(SAVE_DIR, 'tsne_km.png'), param_index=1)

    # ==== 单个样本归因分析 ====
    data = dataset[0].to(DEVICE)
    attribution = compute_input_gradient(model, data, target_index=0)
    visualize_node_attribution(attribution, save_path=os.path.join(SAVE_DIR, 'node_importance_bar.png'))
    residue_class_importance(attribution, data, save_path=os.path.join(SAVE_DIR, 'residue_class_bar.png'))
