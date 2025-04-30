import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
from GNN_model import PocketGNN1


def compute_metrics(y_true, y_pred):
    y_true = y_true.numpy() if torch.is_tensor(y_true) else np.array(y_true)
    y_pred = y_pred.numpy() if torch.is_tensor(y_pred) else np.array(y_pred)
    metrics = {}
    for i, name in enumerate(['kcat', 'Km']):
        metrics[f'{name}_MAE'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
        metrics[f'{name}_R2'] = r2_score(y_true[:, i], y_pred[:, i])
        metrics[f'{name}_Pearson'] = pearsonr(y_true[:, i], y_pred[:, i])[0]
    metrics['Overall_MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['Overall_R2'] = r2_score(y_true, y_pred)
    return metrics


def residual_analysis_tsne(y_true, y_pred, embeddings, save_dir):
    residuals = y_true - y_pred
    os.makedirs(save_dir, exist_ok=True)

    # Residuals vs True values
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true.flatten(), residuals.flatten(), alpha=0.6, c='teal', edgecolor='k')
    plt.axhline(0, linestyle='--', color='gray')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs True Values')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residual_vs_true.png'), dpi=300)
    plt.close()

    # t-SNE
    pca = PCA(n_components=min(50, embeddings.shape[1]), random_state=42)
    reduced = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_coords = tsne.fit_transform(reduced)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], 
                     c=residuals.flatten(), cmap='coolwarm', alpha=0.8, edgecolor='k', linewidth=0.2)
    plt.colorbar(sc, label='Residuals')
    plt.title('t-SNE of Graph Embeddings (colored by residuals)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tsne_residual.png'), dpi=300)
    plt.close()

    print("✅ residual_vs_true.png and tsne_residual.png saved.")


def evaluate(dataset_path, model_path, save_dir="outputs"):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    dataset = torch.load(dataset_path)
    loader = DataLoader(dataset, batch_size=32)

    # Init model
    node_feature_dim = dataset[0].x.size(1)
    edge_feature_dim = dataset[0].edge_attr.size(1)
    model = PocketGNN1(
        node_input_dim=node_feature_dim,
        edge_input_dim=edge_feature_dim,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred, all_embeddings = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            emb = model.get_graph_embedding(batch)
            y_true.append(torch.log10(batch.y.view(-1, 2)).cpu())
            y_pred.append(pred.cpu())
            all_embeddings.append(emb.cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

    metrics = compute_metrics(y_true, y_pred)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    residual_analysis_tsne(y_true.numpy(), y_pred.numpy(), all_embeddings, save_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/processed/dataset1.pt')
    parser.add_argument('--model', type=str, default='outputs/1/best_model.pt')
    parser.add_argument('--save_dir', type=str, default='outputs/tsne')
    args = parser.parse_args()
    evaluate(args.dataset, args.model, args.save_dir)