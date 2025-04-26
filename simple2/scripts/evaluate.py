import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.loader import DataLoader
from GNN_model import PocketGNN
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import os

def compute_metrics(y_true, y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Pearson': pearsonr(y_true.flatten(), y_pred.flatten())[0]
    }

def scatter_plot(y_true, y_pred, save_path):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true[:,0], y_pred[:,0], label='kcat', c='tab:blue', alpha=0.7)
    plt.scatter(y_true[:,1], y_pred[:,1], label='Km', c='tab:orange', alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Predicted vs True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate(dataset_path, model_path, save_dir="outputs"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    dataset = torch.load(dataset_path)
    loader = DataLoader(dataset, batch_size=32)

    # Load model
    model = PocketGNN(num_atom_types=20).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            y_true.append(batch.y.cpu())
            y_pred.append(pred.cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    metrics = compute_metrics(y_true, y_pred)
    print("\n📊 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    scatter_plot(y_true.numpy(), y_pred.numpy(), os.path.join(save_dir, "scatter_plot.png"))
    print(f"✅ Scatter plot saved to {save_dir}/scatter_plot.png")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, default='outputs/best_model.pt')
    parser.add_argument('--save_dir', type=str, default='outputs')
    args = parser.parse_args()
    evaluate(args.dataset, args.model, args.save_dir)
