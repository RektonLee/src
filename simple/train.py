import os
import torch
import argparse
import logging
import matplotlib.pyplot as plt
from model import ImprovedEnzymePredictionModel
from data_loader import load_data_from_local
import torch.optim as optim
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import numpy as np



BASE_DIR = "/home/lizihao/Work/enzyme_prediction/output/GNN0/all_m"
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "training.log"),
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()
logging.info("测试日志配置")
def plot_scatter(true, pred, name, save_dir):
    plt.figure(figsize=(5, 5))
    plt.scatter(true, pred, alpha=0.6)
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"{name} Prediction")
    plt.plot([min(true), max(true)], [min(true), max(true)], 'r--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name.lower()}_scatter.png"))
    plt.close()

def evaluate_metrics(model, data_loader, device, save_dir=FIGURE_DIR, prefix="val",ifplt=False):
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0
    with torch.no_grad():
        for binding_site, substrate, target in data_loader:
            binding_site = binding_site.to(device)
            substrate = substrate.to(device)
            target = target.to(device)
            pred = model(binding_site, substrate)
            total_loss += model.criterion(pred, target).item()
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    os.makedirs(save_dir, exist_ok=True)
    metrics = {}
    names = ['Km', 'kcat']

    for i, name in enumerate(names):
        r2_log = r2_score(targets[:, i], preds[:, i])
        pcc_log, _ = pearsonr(targets[:, i], preds[:, i])
        metrics[f'{name}_r2_log'] = r2_log
        metrics[f'{name}_pcc_log'] = pcc_log

        real_pred = 10 ** preds[:, i]
        real_true = 10 ** targets[:, i]
        r2_real = r2_score(real_true, real_pred)
        pcc_real, _ = pearsonr(real_true, real_pred)
        metrics[f'{name}_r2_real'] = r2_real
        metrics[f'{name}_pcc_real'] = pcc_real

        # 画散点图
        if ifplt:
            plot_scatter(real_true, real_pred, f"{prefix}_{name}_real", save_dir)
            plot_scatter(targets[:, i], preds[:, i], f"{prefix}_{name}_log", save_dir)

    metrics['loss'] = total_loss / len(data_loader)
    return metrics

def train(data_path, npz_path, pdb_dir, epochs=100):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data_from_local(data_path, npz_path, pdb_dir)
    model = ImprovedEnzymePredictionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    train_losses, val_losses = [], []
    r2_km_scores, r2_kcat_scores = [], []
    best_val_metrics = {'loss': float('inf')}
    early_stopping_patience = 20
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            binding_site, substrate, targets = batch
            binding_site = binding_site.to(device)
            substrate = substrate.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(binding_site, substrate)
            loss = model.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # val_metrics = evaluate_metrics(model, test_loader, device, save_dir=FIGURE_DIR, prefix=f"epoch{epoch+1}")
        if epoch == epochs - 1 or early_stopping_counter + 1 == early_stopping_patience:
            val_metrics = evaluate_metrics(
                model, test_loader, device,
                save_dir=FIGURE_DIR, prefix=f"epoch{epoch+1}", ifplt=True
            )
            evaluate_metrics(
                model, train_loader, device,
                save_dir=FIGURE_DIR, prefix=f"epoch{epoch+1}_train", ifplt=True
            )
        else:
            val_metrics = evaluate_metrics(
                model, test_loader, device,
                save_dir=FIGURE_DIR, prefix=f"epoch{epoch+1}", ifplt=False
            )
        scheduler.step(val_metrics['loss'])

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_metrics['loss'])
        r2_km_scores.append(val_metrics['Km_r2_log'])
        r2_kcat_scores.append(val_metrics['kcat_r2_log'])

        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss/len(train_loader):.6f}, "
            f"Val Loss: {val_metrics['loss']:.6f}, "
            f"Km R² (log): {val_metrics['Km_r2_log']:.4f}, "
            f"kcat R² (log): {val_metrics['kcat_r2_log']:.4f}, "
            f"Km R² (real): {val_metrics['Km_r2_real']:.4f}, "
            f"kcat R² (real): {val_metrics['kcat_r2_real']:.4f}"
        )

        if val_metrics['loss'] < best_val_metrics['loss']:
            best_val_metrics = val_metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(r2_km_scores, label="Km R²")
    plt.plot(r2_kcat_scores, label="kcat R²")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("R² Scores")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "training_curves.png"))
    plt.close()

    logger.info("\n最终结果:")
    logger.info(f"最佳验证损失: {best_val_metrics['loss']:.6f}")
    logger.info(f"最佳Km R²: {best_val_metrics['Km_r2_log']:.4f} (log), {best_val_metrics['Km_r2_real']:.4f} (real)")
    logger.info(f"最佳kcat R²: {best_val_metrics['kcat_r2_log']:.4f} (log), {best_val_metrics['kcat_r2_real']:.4f} (real)")

    # 最后对最佳模型再次进行全面评估并画图
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'))['model_state_dict'])
    evaluate_metrics(model, test_loader, device, save_dir=os.path.join(FIGURE_DIR, "best"), prefix="best",ifplt=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/home/lizihao/Work/data_processed/cleaned_data.csv")
    parser.add_argument("--npz", default="/home/lizihao/Work/enzyme_prediction/output/processed/processed_data_20250310_205850.npz")
    parser.add_argument("--pdb_dir", default="/home/lizihao/Work/enzyme_prediction/src/output/pdb_files")
    args = parser.parse_args()

    train(args.csv, args.npz, args.pdb_dir)
