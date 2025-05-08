import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Assuming GNN_model.py and utils.metadata_utils are in the PYTHONPATH or same directory
import GNN_model as MD # Make sure this import works in your environment
from utils.metadata_utils import save_metadata # Make sure this import works

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')  # Ensure in a non-GUI environment
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# compute_metrics now expects predictions and true values in the *scale used for loss calculation*
# If loss is on log-scale or scaled-log-scale, metrics will be on that same scale.
def compute_metrics(y_true_scaled, y_pred_scaled):
    y_true_scaled = y_true_scaled.numpy()
    y_pred_scaled = y_pred_scaled.numpy()
    # Ensure no NaNs or Infs, which can happen with bad predictions or log transforms
    if not (np.isfinite(y_true_scaled).all() and np.isfinite(y_pred_scaled).all()):
        print("Warning: NaN or Inf found in metrics calculation. Returning default poor metrics.")
        return {
            'MAE': float('nan'),
            'RMSE': float('nan'),
            'R2': -float('inf'), # Ensure R2 is very bad if there's an issue
            'Pearson': float('nan')
        }
    if len(y_true_scaled.flatten()) < 2 or len(y_pred_scaled.flatten()) < 2: # Pearson requires at least 2 samples
         print("Warning: Not enough samples for Pearson correlation. Returning NaN for Pearson.")
         pearson_val = float('nan')
    else:
        pearson_val = pearsonr(y_true_scaled.flatten(), y_pred_scaled.flatten())[0]
        if np.isnan(pearson_val): # Handle cases where pearsonr might return nan (e.g. zero variance)
            print("Warning: Pearson correlation resulted in NaN. Setting to 0.")
            pearson_val = 0.0


    return {
        'MAE': mean_absolute_error(y_true_scaled, y_pred_scaled),
        'RMSE': np.sqrt(mean_squared_error(y_true_scaled, y_pred_scaled)),
        'R2': r2_score(y_true_scaled, y_pred_scaled),
        'Pearson': pearson_val
    }

def train(dataset_path, save_dir="outputs", batch_size=32, lr=1e-3, max_epochs=500,
          hidden_dim=256, num_layers=6, heads=8, dropout=0.1,
          optimizer_type='Adam', weight_decay=0.0, loss_fn_type='MSE',
          lr_scheduler_name=None, metadata=None): # Added metadata as argument

    # Load the dataset (list of Data objects)
    # data_list = torch.load(dataset_path) # This was loaded twice in original code
    
    # It's better to pass metadata dict to the function
    if metadata is None:
        metadata = {}

    # --- Data Preparation ---
    # Assuming data.y are log10 transformed values as per common practice.
    # We will standardize these log10 values.
    data_list = torch.load(dataset_path)

    # Check for NaNs in input features and original log-scale labels BEFORE any processing
    for i, data in enumerate(data_list):
        if torch.isnan(data.x).any():
            raise ValueError(f"NaN found in data.x for sample {i} (pdb_id: {getattr(data, 'pdb_id', 'N/A')})")
        if torch.isnan(data.y).any():
            # If NaNs are possible in y and should be removed:
            # print(f"Warning: NaN found in data.y for sample {i} (pdb_id: {getattr(data, 'pdb_id', 'N/A')}). Skipping this sample.")
            # continue # Or handle appropriately
            raise ValueError(f"NaN found in data.y for sample {i} (pdb_id: {getattr(data, 'pdb_id', 'N/A')}) BEFORE scaling")

    # Concatenate all log-scale labels (y) to calculate mean and std for standardization
    all_log_labels = torch.cat([data.y.unsqueeze(0) for data in data_list if data.y is not None], dim=0) # Reshape y if it's 1D per sample
    
    # Ensure y is treated as 2D (num_samples * num_targets_per_sample)
    if all_log_labels.ndim == 1: # If y was [val1, val2, val3, val4...] for two targets, reshape
        num_targets = 2 # Assuming kcat and Km
        if len(all_log_labels) % num_targets != 0 and len(data_list) * num_targets == len(all_log_labels) :
             all_log_labels = all_log_labels.reshape(-1, num_targets)
        elif len(data_list) != all_log_labels.shape[0] : # if each data.y is already [target1, target2]
            # This case suggests data.y was [target1, target2] and cat made it [[t1,t2], [t1,t2], ...]
            # which is fine. The unsqueeze(0) was for individual data.y if they were 1D of targets.
            # If data.y is already [kcat_val, km_val], then all_log_labels will be (N, 2)
            pass # Shape should be (num_samples, num_targets)
        else: # Fallback if logic is tricky, assume each data.y is a flat list of its targets
            all_log_labels = torch.stack([d.y for d in data_list if d.y is not None])


    if all_log_labels.ndim == 1: # Should not happen if y is (kcat, Km) per sample
        print(f"Warning: all_log_labels is 1D with shape {all_log_labels.shape}. Attempting to infer targets.")
        # This might indicate an issue in how data.y is structured or concatenated.
        # For now, we'll assume it might be a flat list and try to proceed, but this needs verification.
        # If each sample has 2 targets, and they are flattened:
        if len(all_log_labels) == len(data_list) * 2:
             all_log_labels = all_log_labels.view(len(data_list), 2)
        else: # Cannot safely reshape
            raise ValueError(f"Cannot determine target structure from all_log_labels shape: {all_log_labels.shape}")


    mean_log_y = all_log_labels.mean(dim=0)
    std_log_y = all_log_labels.std(dim=0)

    # Avoid division by zero if std is 0 for any target (e.g., all target values are the same)
    std_log_y[std_log_y == 0] = 1.0 

    metadata.update({
        'mean_log_y': mean_log_y.tolist(), # Store as list for JSON serialization
        'std_log_y': std_log_y.tolist(),
        'training_target_scale': 'standardized_log10'
    })
    # Save metadata early, can be updated later if needed, or save at the end
    os.makedirs(save_dir, exist_ok=True) # Ensure save_dir exists
    save_metadata(save_dir=save_dir, **metadata)


    # Standardize y for each data object
    for data in data_list:
        if data.y is not None:
            data.y_original_log = data.y.clone() # Keep original log values if needed
            data.y = (data.y - mean_log_y) / std_log_y # Standardized log_y
            if torch.isnan(data.y).any():
                 raise ValueError(f"NaN found in data.y for sample (pdb_id: {getattr(data, 'pdb_id', 'N/A')}) AFTER scaling. Check mean/std values: mean={mean_log_y}, std={std_log_y}. Original y={data.y_original_log}")


    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(save_dir)

    # === Load dataset ===
    print(f"First data object (targets are now standardized log10 values): \n{data_list[0]}")
    
    np.random.shuffle(data_list) # Shuffle before splitting
    split_idx = int(0.8 * len(data_list))
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # === Initialize model ===
    node_input_dim = data_list[0].x.shape[1]
    edge_input_dim = data_list[0].edge_attr.shape[1]
    # Assuming model predicts two values (for kcat and Km in standardized log-space)
    # The model's output layer should be linear with 2 output neurons.
    model = MD.PocketGNNWithAttention(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
        # output_dim=2 # Ensure your model's output_dim is 2 for kcat, Km
    ).to(device)

    # === Initialize optimizer ===
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay) # Added weight_decay for SGD too
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    # === Initialize loss function ===
    # Now loss is calculated on standardized log-transformed values
    if loss_fn_type == 'MSE' or loss_fn_type == 'ScaledOriginalMSE': # Treat ScaledOriginalMSE as MSE on new scaled log targets
        criterion = nn.MSELoss()
    elif loss_fn_type == 'L1':
        criterion = nn.L1Loss()
    elif loss_fn_type == 'SmoothL1':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn_type}")

    # === Initialize LR Scheduler ===
    scheduler = None
    if lr_scheduler_name == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
    elif lr_scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, verbose=True) # Example step_size
    elif lr_scheduler_name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr/100, verbose=True) # Example T_max

    best_val_loss = float('inf')
    train_loss_history, val_loss_history, r2_history, pearson_history = [], [], [], []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses_epoch = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # batch.y should already be standardized log values, shaped (batch_size, 2)
            # If batch.y is flattened, reshape it. DataLoader should handle this if Data.y is (1,2) or (2,).
            # Assuming Data.y is already [target1_scaled_log, target2_scaled_log]
            # And DataLoader forms a batch where batch.y is [num_graphs, 2]
            
            # Check if batch.y needs reshaping (it should be [actual_batch_size, num_targets])
            actual_batch_size = batch.num_graphs
            num_targets = 2 # kcat, Km
            
            # The target `true_scaled_log_y` is batch.y directly if correctly batched
            true_scaled_log_y = batch.y 
            if true_scaled_log_y.shape != (actual_batch_size, num_targets):
                try:
                    true_scaled_log_y = true_scaled_log_y.reshape(actual_batch_size, num_targets)
                except RuntimeError as e:
                    print(f"Error reshaping batch.y. Original shape: {batch.y.shape}, target: ({actual_batch_size}, {num_targets})")
                    raise e


            if torch.isnan(batch.x).any(): print("❌ batch.x 中含有 NaN")
            if torch.isnan(batch.edge_attr).any(): print("❌ batch.edge_attr 中含有 NaN")
            if torch.isnan(true_scaled_log_y).any(): print("❌ true_scaled_log_y (batch.y) 中含有 NaN")

            pred_scaled_log_y = model(batch) # Model predicts standardized log values
            
            if torch.isnan(pred_scaled_log_y).any():
                print(f"❌ Model output (pred_scaled_log_y) contains NaN at epoch {epoch}. Skipping batch.")
                # Optionally, investigate which samples in batch caused this
                # for i in range(actual_batch_size):
                #     if torch.isnan(pred_scaled_log_y[i]).any():
                #         print(f"NaN prediction for sample with pdb_id: {getattr(batch[i], 'pdb_id', 'N/A')}")
                continue # Skip this batch if output is NaN

            loss = criterion(pred_scaled_log_y, true_scaled_log_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses_epoch.append(loss.item())
        
        if not train_losses_epoch: # Handle case where all training batches were skipped
            print(f"Epoch {epoch:03d} | No training batches processed. Skipping validation.")
            train_loss_history.append(float('nan')) # Or some placeholder
            val_loss_history.append(float('nan'))
            r2_history.append(float('nan'))
            pearson_history.append(float('nan'))
            continue

        current_train_loss = np.mean(train_losses_epoch)
        train_loss_history.append(current_train_loss)

        # === Validation ===
        model.eval()
        val_losses_epoch = []
        y_true_scaled_log_all, y_pred_scaled_log_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                actual_batch_size = batch.num_graphs
                num_targets = 2

                true_scaled_log_y_val = batch.y
                if true_scaled_log_y_val.shape != (actual_batch_size, num_targets):
                     true_scaled_log_y_val = true_scaled_log_y_val.reshape(actual_batch_size, num_targets)
                
                if torch.isnan(true_scaled_log_y_val).any():
                    print(f"❌ NaN in validation true_scaled_log_y_val (batch.y) at epoch {epoch}.")
                    # Decide how to handle: skip batch, or error out
                    # For now, we'll let it pass to compute_metrics which should handle NaNs
                
                pred_scaled_log_y_val = model(batch)

                if torch.isnan(pred_scaled_log_y_val).any():
                    print(f"❌ NaN in validation pred_scaled_log_y_val (model output) at epoch {epoch}.")
                    # Create dummy predictions if model outputs NaN to avoid crashing metrics
                    # but this will yield bad metrics as expected.
                    pred_scaled_log_y_val = torch.full_like(true_scaled_log_y_val, float('nan'))


                loss = criterion(pred_scaled_log_y_val, true_scaled_log_y_val)
                if not torch.isnan(loss): # Only append if loss is not NaN
                    val_losses_epoch.append(loss.item())
                
                y_true_scaled_log_all.append(true_scaled_log_y_val.cpu())
                y_pred_scaled_log_all.append(pred_scaled_log_y_val.cpu())
        
        if not val_losses_epoch: # Handle case where all validation batches resulted in NaN loss
             current_val_loss = float('nan')
        else:
            current_val_loss = np.mean(val_losses_epoch)
        
        val_loss_history.append(current_val_loss)

        y_true_scaled_log_tensor = torch.cat(y_true_scaled_log_all, dim=0)
        y_pred_scaled_log_tensor = torch.cat(y_pred_scaled_log_all, dim=0)
        
        # Metrics are now computed on standardized log-scale values
        metrics = compute_metrics(y_true_scaled_log_tensor, y_pred_scaled_log_tensor)
        r2_history.append(metrics['R2'])
        pearson_history.append(metrics['Pearson'])

        writer.add_scalar("Loss/train_scaled_log", current_train_loss, epoch)
        writer.add_scalar("Loss/val_scaled_log", current_val_loss, epoch)
        writer.add_scalar("R2/val_scaled_log", metrics['R2'], epoch)
        writer.add_scalar("Pearson/val_scaled_log", metrics['Pearson'], epoch)
        if scheduler:
             current_lr = optimizer.param_groups[0]['lr']
             writer.add_scalar("LearningRate", current_lr, epoch)


        print(f"Epoch {epoch:03d} | Train Loss (scaled_log): {current_train_loss:.4f} | Val Loss (scaled_log): {current_val_loss:.4f} | R2 (scaled_log): {metrics['R2']:.3f} | Pearson (scaled_log): {metrics['Pearson']:.3f}")

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_val_loss)
            elif scheduler is not None : # For other schedulers like StepLR, CosineAnnealingLR
                scheduler.step()


        if not np.isnan(current_val_loss) and current_val_loss < best_val_loss :
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model_scaled_log_targets.pt"))
            print(f"🚀 New best model saved at epoch {epoch} with val_loss: {best_val_loss:.4f}")


    writer.close()
    print("✅ Training finished.")
    if os.path.exists(os.path.join(save_dir, "best_model_scaled_log_targets.pt")):
        print("Best model saved as best_model_scaled_log_targets.pt")
    else:
        print("Warning: Best model was not saved (possibly due to NaN validation losses throughout training).")


    # === Plotting ===
    # Ensure mean_log_y and std_log_y are torch tensors for broadcasting if not already
    mean_log_y_tensor = torch.tensor(mean_log_y, device='cpu').float() # Ensure float for calculations
    std_log_y_tensor = torch.tensor(std_log_y, device='cpu').float()

    # 1. Loss curve (on scaled log values)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Loss (scaled_log)')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss (scaled_log)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Standardized Log10 Scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve_scaled_log.png'))
    plt.close()

    # 2. R2 and Pearson correlation curve (on scaled log values)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(r2_history) + 1), r2_history, label='R² Score (scaled_log)')
    plt.plot(range(1, len(pearson_history) + 1), pearson_history, label='Pearson Correlation (scaled_log)')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('R² and Pearson Correlation During Training (Standardized Log10 Scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'metrics_curve_scaled_log.png'))
    plt.close()

    # 3. Prediction vs. True value scatter plot (on ORIGINAL scale)
    # Load the best model for this plot
    best_model_path = os.path.join(save_dir, "best_model_scaled_log_targets.pt")
    if not os.path.exists(best_model_path):
        print(f"Warning: Best model file {best_model_path} not found. Skipping scatter plot.")
        return # Exit if no best model to load

    model.load_state_dict(torch.load(best_model_path, map_location=device)) # map_location for safety
    model.eval()

    all_y_true_original, all_y_pred_original = [], []
    with torch.no_grad():
        for batch in val_loader: # Use val_loader for consistency with validation metrics
            batch = batch.to(device)
            
            # True values: unstandardize and then un-log-transform batch.y
            # batch.y is standardized log. First, un-standardize:
            true_log_y = batch.y * std_log_y.to(device) + mean_log_y.to(device)
            # Then, un-log-transform (10^x)
            true_original_y = 10**true_log_y
            all_y_true_original.append(true_original_y.cpu())

            # Predicted values: model(batch) gives standardized log. Unscale and un-log.
            pred_scaled_log_y_plot = model(batch)
            pred_log_y = pred_scaled_log_y_plot * std_log_y.to(device) + mean_log_y.to(device)
            pred_original_y = 10**pred_log_y
            all_y_pred_original.append(pred_original_y.cpu())

    y_true_original_np = torch.cat(all_y_true_original, dim=0).numpy()
    y_pred_original_np = torch.cat(all_y_pred_original, dim=0).numpy()
    
    # Check for NaNs/Infs in original scale plots due to 10**large_number
    if not np.isfinite(y_true_original_np).all() or not np.isfinite(y_pred_original_np).all():
        print("Warning: NaN or Inf found in original scale true/predicted values for plotting. Clamping might occur or plots might be affected.")
        # Optionally, replace Infs with a large number or NaNs with mean for plotting, or filter them out
        # For simplicity, we'll let matplotlib handle them, but be aware.

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6)) # Increased figure size
    param_names = ['kcat', 'Km']

    for i, param_name in enumerate(param_names):
        ax = ax1 if i == 0 else ax2
        true_vals = y_true_original_np[:, i]
        pred_vals = y_pred_original_np[:, i]

        # Filter out NaNs/Infs for plotting this specific parameter if they exist
        finite_mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
        true_vals_finite = true_vals[finite_mask]
        pred_vals_finite = pred_vals[finite_mask]

        if len(true_vals_finite) < 2 : # Not enough data to plot or calculate R2
            ax.text(0.5, 0.5, 'Not enough finite data to plot', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'{param_name}: True vs Predicted (Original Scale)')
            continue


        ax.scatter(true_vals_finite, pred_vals_finite, alpha=0.5, label='Predictions')
        
        # Add y=x line based on the range of finite values
        min_val = min(true_vals_finite.min(), pred_vals_finite.min())
        max_val = max(true_vals_finite.max(), pred_vals_finite.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
        
        ax.set_xlabel(f'True {param_name} (Original Scale)')
        ax.set_ylabel(f'Predicted {param_name} (Original Scale)')
        
        r2_original_param = r2_score(true_vals_finite, pred_vals_finite)
        pearson_original_param, _ = pearsonr(true_vals_finite, pred_vals_finite)

        ax.set_title(f'{param_name}: True vs Predicted (Original Scale)\nR²={r2_original_param:.3f}, Pearson={pearson_original_param:.3f}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_scatter_original_scale_from_log_model.png'))
    plt.close()

    # 4. Heatmap/Density plot (on ORIGINAL scale)
    for i, param_name in enumerate(param_names):
        plt.figure(figsize=(8, 7))
        true_vals = y_true_original_np[:, i]
        pred_vals = y_pred_original_np[:, i]

        finite_mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
        true_vals_finite = true_vals[finite_mask]
        pred_vals_finite = pred_vals[finite_mask]
        
        if len(true_vals_finite) < 20 : # KDE plot needs sufficient points
            plt.text(0.5, 0.5, 'Not enough finite data for density plot', horizontalalignment='center', verticalalignment='center')
            plt.title(f'{param_name}: Density Plot (Original Scale)')
            plt.savefig(os.path.join(save_dir, f'{param_name}_density_original_scale_from_log_model.png'))
            plt.close()
            continue

        try:
            sns.kdeplot(x=true_vals_finite, y=pred_vals_finite, cmap="viridis", fill=True, thresh=0.05)
            min_val = min(true_vals_finite.min(), pred_vals_finite.min())
            max_val = max(true_vals_finite.max(), pred_vals_finite.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2) # Make line more visible
        except Exception as e:
            print(f"Could not generate density plot for {param_name}: {e}")
            plt.text(0.5, 0.5, f'Error generating density plot:\n{e}', horizontalalignment='center', verticalalignment='center', wrap=True)


        plt.xlabel(f'True {param_name} (Original Scale)')
        plt.ylabel(f'Predicted {param_name} (Original Scale)')
        plt.title(f'{param_name}: Density Plot (Original Scale)')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{param_name}_density_original_scale_from_log_model.png'))
        plt.close()

    # Save final metrics (these are from scaled log space, which drove training)
    metrics_df = pd.DataFrame({
        'Epoch': range(1, len(train_loss_history) + 1),
        'Train_Loss_scaled_log': train_loss_history,
        'Val_Loss_scaled_log': val_loss_history,
        'R2_scaled_log': r2_history,
        'Pearson_scaled_log': pearson_history
    })
    metrics_df.to_csv(os.path.join(save_dir, 'training_metrics_scaled_log.csv'), index=False)

    print(f"✅ Training finished. Plots and metrics saved to {save_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train GNN for Enzyme Kinetics Prediction")
    parser.add_argument('--dataset', type=str, default="/home/lizihao/Work/enzyme_prediction/src/simple2/data/processed/dataset_NAN_nopqr.pt", help='Path to .pt dataset where data.y is log10 transformed')
    parser.add_argument('--save_dir', type=str, default='outputs/nopqr_attention_log_targets_v2')

    # Model Hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3) # Initial LR
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--weight_decay', type=float, default=0.0)
    # Loss function is implicitly MSE on (standardized) log targets now
    parser.add_argument('--loss_fn', type=str, default='MSE', choices=['MSE', 'L1', 'SmoothL1']) # Removed ScaledOriginalMSE as it's now handled by log scaling
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau', choices=[None, 'StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'], help="Specify None to disable scheduler")


    args = parser.parse_args()

    # Prepare metadata dictionary
    # Ensure this metadata reflects the new approach (training on log-scale)
    metadata = {
        'dataset_path': args.dataset,
        'graph_builder_version': 'builder2_elec_att', # Keep as is or update if changed
        'gnn_model_version': 'PocketGNNwithAttention', # Keep as is or update if changed
        'comments': 'Training on standardized log10 transformed targets (kcat, Km). Attention in GNN. Enriched pocket features.',
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'heads': args.heads,
        'dropout': args.dropout,
        'lr_initial': args.lr,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'optimizer': args.optimizer,
        'weight_decay': args.weight_decay,
        'loss_fn_on_scaled_log': args.loss_fn,
        'lr_scheduler': args.lr_scheduler
        # mean_log_y and std_log_y will be added inside train()
    }
    
    # The train function will handle saving metadata after calculating mean_log_y and std_log_y
    train(dataset_path=args.dataset,
          save_dir=args.save_dir,
          batch_size=args.batch_size,
          lr=args.lr,
          max_epochs=args.max_epochs,
          hidden_dim=args.hidden_dim,
          num_layers=args.num_layers,
          heads=args.heads,
          dropout=args.dropout,
          optimizer_type=args.optimizer,
          weight_decay=args.weight_decay,
          loss_fn_type=args.loss_fn,
          lr_scheduler_name=args.lr_scheduler,
          metadata=metadata) # Pass the metadata dict