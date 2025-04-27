import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch_scatter
import logging
from datetime import datetime
import time
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm
import biotite.structure as bst
import biotite.structure.io as bstio
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.nn import radius_graph

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import EnzymeKineticsModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_file_logger(log_file):
    """Add file handler to logger"""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def load_pdb_structure(pdb_path):
    """Load protein structure from PDB file using biotite"""
    try:
        structure = bstio.load_structure(pdb_path)
        return structure
    except Exception as e:
        logger.error(f"Error loading PDB file {pdb_path}: {str(e)}")
        return None


def identify_binding_site(structure, center_coords, radius=10.0):
    """
    Identify binding site atoms around a given center point
    
    Args:
        structure: Biotite structure object
        center_coords: [x, y, z] coordinates of binding site center
        radius: Radius to consider for binding site atoms (Angstroms)
        
    Returns:
        binding_site_mask: Boolean mask for atoms in binding site
    """
    # Calculate distances of all atoms to center
    coords = structure.coord
    center = np.array(center_coords)
    
    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((coords - center)**2, axis=1))
    
    # Create mask for atoms within radius
    binding_site_mask = distances <= radius
    
    return binding_site_mask


def get_atom_features(structure, binding_site_mask):
    """
    Extract atom features for atoms in binding site
    
    Features include:
    - Atom element (one-hot encoded)
    - Residue type (one-hot encoded)
    - Secondary structure (one-hot encoded)
    
    Args:
        structure: Biotite structure object
        binding_site_mask: Boolean mask for atoms in binding site
        
    Returns:
        atom_features: [num_atoms, num_features] array of atom features
        atom_coords: [num_atoms, 3] array of atom coordinates
    """
    # Get binding site atoms
    binding_site = structure[binding_site_mask]
    
    # Extract basic properties
    elements = binding_site.element
    residues = binding_site.res_name
    atom_coords = binding_site.coord
    
    # Define common elements and amino acids for one-hot encoding
    common_elements = ['C', 'N', 'O', 'S', 'H', 'P']
    amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                  'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                  'THR', 'TRP', 'TYR', 'VAL']
    
    # Initialize feature array
    num_element_features = len(common_elements) + 1  # +1 for "other"
    num_residue_features = len(amino_acids) + 1  # +1 for "other"
    
    total_features = num_element_features + num_residue_features
    atom_features = np.zeros((len(binding_site), total_features))
    
    # Element one-hot encoding
    for i, element in enumerate(elements):
        if element in common_elements:
            idx = common_elements.index(element)
        else:
            idx = len(common_elements)  # "other" category
        atom_features[i, idx] = 1
    
    # Residue one-hot encoding
    offset = num_element_features
    for i, residue in enumerate(residues):
        if residue in amino_acids:
            idx = amino_acids.index(residue)
        else:
            idx = len(amino_acids)  # "other" category
        atom_features[i, offset + idx] = 1
    
    return atom_features, atom_coords


def create_binding_site_graph(atom_features, atom_coords, cutoff=7.0):
    """
    Create a geometric graph from binding site atoms
    
    Args:
        atom_features: [num_atoms, num_features] array of atom features
        atom_coords: [num_atoms, 3] array of atom coordinates
        cutoff: Distance cutoff for edges (Angstroms)
        
    Returns:
        data: PyTorch Geometric Data object with graph representation
    """
    # Convert to PyTorch tensors
    x = torch.tensor(atom_features, dtype=torch.float)
    pos = torch.tensor(atom_coords, dtype=torch.float)
    
    # Create edges based on distance
    edge_index = radius_graph(pos, r=cutoff, loop=False)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, pos=pos)
    
    return data


def process_pdb_files(csv_path, pdb_dir, output_dir, binding_sites_info=None, npz_path=None):
   """
    Process PDB files to create binding site graphs
    
    Args:
        csv_path: Path to CSV file with enzyme data
        pdb_dir: Directory with PDB files
        output_dir: Directory to save processed data
        binding_sites_info: Optional dict mapping uniprot_id to binding site center
        npz_path: Path to .npz file containing substrate embeddings
        
    Returns:
        binding_site_graphs: List of PyG Data objects for binding sites
        processed_indices: List of indices in original CSV that were processed
        uniprot_ids: List of processed uniprot_ids
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)  
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV with {len(df)} entries")
    
    # 加载 .npz 文件并检查条目数
    if npz_path:
        try:
            npz_data = np.load(npz_path,allow_pickle=True)
            substrate_embeddings = npz_data['substrate_embeddings']
            if len(df) != len(substrate_embeddings):
                logger.warning(f"CSV file has {len(df)} entries, but .npz file has {len(substrate_embeddings)} entries. Indices may not align.")
        except Exception as e:
            logger.error(f"Error loading .npz file {npz_path}: {str(e)}")
            npz_data = None
    else:
        npz_data = None

    binding_site_graphs = []
    processed_indices = []
    uniprot_ids = []
    
    # Default binding site centers if not provided
    if binding_sites_info is None:
        binding_sites_info = {}
    
    # Process each entry
    skipped = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing PDB files"):
        uniprot_id = row.get('uniprot')
        
        if not uniprot_id:
            skipped += 1
            continue
            
        # Check if PDB file exists
        pdb_path = os.path.join(pdb_dir, f"{uniprot_id}.pdb")
        if not os.path.exists(pdb_path):
            logger.warning(f"PDB file not found for {uniprot_id}, skipping")
            skipped += 1
            continue
        
        # Load structure
        structure = load_pdb_structure(pdb_path)
        if structure is None:
            skipped += 1
            continue
        
        # Get binding site center
        binding_site_center = binding_sites_info.get(uniprot_id)
        if binding_site_center is None:
            # Use geometric center of structure as default
            binding_site_center = structure.coord.mean(axis=0)
        
        # Identify binding site
        binding_site_mask = identify_binding_site(structure, binding_site_center)
        
        # Check if binding site has enough atoms
        if np.sum(binding_site_mask) < 10:
            logger.warning(f"Too few atoms in binding site for {uniprot_id}, skipping")
            skipped += 1
            continue
        
        # Extract features and create graph
        atom_features, atom_coords = get_atom_features(structure, binding_site_mask)
        graph = create_binding_site_graph(atom_features, atom_coords)
        
        # 关联行号
        if npz_data is not None:
            graph.csv_index = idx  # 将 CSV 行号添加到图数据中
        
        # Save binding site graph
        binding_site_graphs.append(graph)
        processed_indices.append(idx)
        uniprot_ids.append(uniprot_id)
        
        # Save individual binding site data for future use
        binding_site_output = {
            'atom_features': atom_features,
            'atom_coords': atom_coords,
            'csv_index': idx,
            'uniprot_id': uniprot_id
        }
        np.savez(
            os.path.join(output_dir, f"binding_site_{uniprot_id}.npz"),
            **binding_site_output
        )
    
    logger.info(f"Processed {len(binding_site_graphs)} structures, skipped {skipped}")
    
    # Save all binding site graphs together
    torch.save(binding_site_graphs, os.path.join(output_dir, "binding_site_graphs.pt"))
    
    # Save mapping between processed indices and original CSV
    np.savez(
        os.path.join(output_dir, "processed_mapping.npz"),
        processed_indices=np.array(processed_indices),
        uniprot_ids=np.array(uniprot_ids)
    )
    
    return binding_site_graphs, processed_indices, uniprot_ids


def create_data_loaders(binding_site_graphs, substrate_embeddings, y, 
                       train_indices, test_indices, batch_size=16):
    """
    Create data loaders for training and testing
    
    Args:
        binding_site_graphs: List of PyG Data objects
        substrate_embeddings: Array of substrate embeddings
        y: Array of targets (Km and kcat)
        train_indices: Indices for training set
        test_indices: Indices for test set
        batch_size: Batch size
        
    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
    """
    # Convert embeddings and targets to tensors
    substrate_tensor = torch.tensor(substrate_embeddings, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Create dataset tuples (binding_site_graph, substrate_embedding, target)
    train_dataset = [(binding_site_graphs[i], substrate_tensor[i], y_tensor[i]) 
                     for i in train_indices]
    test_dataset = [(binding_site_graphs[i], substrate_tensor[i], y_tensor[i])
                    for i in test_indices]
    
    # Create data loaders
    train_loader = GeometricDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = GeometricDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader


def evaluate_metrics(model, data_loader, device):
    """
    Evaluate model performance metrics
    
    Args:
        model: PyTorch model
        data_loader: DataLoader with test data
        device: Device to run evaluation on
        
    Returns:
        metrics: Dict of evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for binding_site, substrate, targets in data_loader:
            binding_site = binding_site.to(device)
            substrate = substrate.to(device)
            targets = targets.to(device)
            
            outputs = model(binding_site, substrate)
            loss = model.criterion(outputs, targets)
            total_loss += loss.item() * len(targets)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics
    r2_km = r2_score(all_targets[:, 0], all_predictions[:, 0])
    r2_kcat = r2_score(all_targets[:, 1], all_predictions[:, 1])
    
    # Original scale metrics (Km and kcat are in log10 scale)
    km_true = 10 ** all_targets[:, 0]
    kcat_true = 10 ** all_targets[:, 1]
    km_pred = 10 ** all_predictions[:, 0]
    kcat_pred = 10 ** all_predictions[:, 1]
    
    r2_km_orig = r2_score(km_true, km_pred)
    r2_kcat_orig = r2_score(kcat_true, kcat_pred)
    
    # Calculate relative errors
    rel_error_km = np.abs(km_true - km_pred) / km_true * 100
    rel_error_kcat = np.abs(kcat_true - kcat_pred) / kcat_true * 100
    
    median_rel_error_km = np.median(rel_error_km)
    median_rel_error_kcat = np.median(rel_error_kcat)
    
    return {
        'loss': total_loss / len(data_loader.dataset),
        'r2_km': r2_km,
        'r2_kcat': r2_kcat,
        'r2_km_orig': r2_km_orig,
        'r2_kcat_orig': r2_kcat_orig,
        'median_rel_error_km': median_rel_error_km,
        'median_rel_error_kcat': median_rel_error_kcat
    }


def plot_training_curves(train_losses, val_losses, r2_km_scores, r2_kcat_scores, save_path):
    """Plot and save training curves"""
    plt.figure(figsize=(12, 6))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    
    # Plot R² scores
    plt.subplot(1, 2, 2)
    plt.plot(r2_km_scores, label="Km R²")
    plt.plot(r2_kcat_scores, label="kcat R²")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("R² Scores")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_model(binding_site_graphs, substrate_embeddings, y, train_indices, test_indices,
                epochs=100, batch_size=16, lr=0.001, weight_decay=1e-5, device='cuda'):
    """
    Train enzyme kinetics prediction model
    
    Args:
        binding_site_graphs: List of PyG Data objects with binding site graphs
        substrate_embeddings: Array of substrate embeddings
        y: Array of targets (Km and kcat)
        train_indices: Indices for training set
        test_indices: Indices for test set
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay coefficient
        device: Device to train on
        
    Returns:
        model: Trained model
        best_metrics: Best validation metrics
    """
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        binding_site_graphs, substrate_embeddings, y,
        train_indices, test_indices, batch_size
    )
    
    # Initialize model
    node_feature_dim = binding_site_graphs[0].x.shape[1]
    substrate_dim = substrate_embeddings.shape[1]
    
    model = EnzymeKineticsModel(
        node_feature_dim=node_feature_dim,
        substrate_dim=substrate_dim,
        hidden_dim=128,
        output_dim=256
    ).to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training tracking
    train_losses = []
    val_losses = []
    r2_km_scores = []
    r2_kcat_scores = []
    
    best_val_metrics = {'loss': float('inf')}
    early_stopping_counter = 0
    early_stopping_patience = 20
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for binding_site, substrate, targets in train_loader:
            binding_site = binding_site.to(device)
            substrate = substrate.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(binding_site, substrate)
            loss = model.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(targets)
        
        train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        val_metrics = evaluate_metrics(model, test_loader, device)
        val_losses.append(val_metrics['loss'])
        r2_km_scores.append(val_metrics['r2_km'])
        r2_kcat_scores.append(val_metrics['r2_kcat'])
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Logging
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.6f}, "
            f"Val Loss: {val_metrics['loss']:.6f}, "
            f"Km R²: {val_metrics['r2_km']:.4f}, "
            f"kcat R²: {val_metrics['r2_kcat']:.4f}"
        )
        
        # Check for improvement
        if val_metrics['loss'] < best_val_metrics['loss']:
            best_val_metrics = val_metrics
            early_stopping_counter = 0
            # Save best model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, os.path.join(output_dir, "checkpoints", "best_model.pth"))
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final training curves
    plot_training_curves(
        train_losses, val_losses, r2_km_scores, r2_kcat_scores,
        os.path.join(output_dir, "training_curves.png")
    )
    
    return model, best_val_metrics


def main(args):
    """Main function to run the training pipeline"""
    global output_dir
    # 在main函数参数解析部分添加
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension size in GNN')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    # Setup output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"training_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    binding_sites_dir = os.path.join(output_dir, "binding_sites")
    os.makedirs(binding_sites_dir, exist_ok=True)
    
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, "training.log")
    setup_file_logger(log_file)
    
    logger.info(f"Starting training pipeline with arguments: {args}")
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Step 1: Process PDB files to create binding site graphs
    logger.info("Processing PDB files...")
    binding_site_graphs, processed_indices, uniprot_ids = process_pdb_files(
        args.csv_path, args.pdb_dir, binding_sites_dir
    )
    
    # Step 2: Load substrate embeddings and targets
    if args.preprocessed_data:
        logger.info(f"Loading preprocessed data from {args.preprocessed_data}")
        data = np.load(args.preprocessed_data)
        substrate_embeddings = data['substrate_embeddings']
        y = data['y']
        
        # 验证索引对齐
        if len(processed_indices) == 0:
            raise ValueError("No valid PDB files processed, check input data")
            
        # 获取处理后的有效索引对应的嵌入和目标
        valid_substrate_emb = substrate_embeddings[processed_indices]
        valid_y = y[processed_indices]
        
        # 创建新的训练测试划分
        num_samples = len(valid_substrate_emb)
        indices = np.random.permutation(num_samples)
        split = int(num_samples * 0.8)
        train_indices = indices[:split]
        test_indices = indices[split:]
        
        # 转换为PyTorch张量
        substrate_tensor = torch.tensor(valid_substrate_emb, dtype=torch.float32)
        y_tensor = torch.tensor(valid_y, dtype=torch.float32)
        
        # 保存处理后的数据集索引
        np.savez(
            os.path.join(output_dir, "processed_indices.npz"),
            original_indices=processed_indices,
            train_indices=train_indices,
            test_indices=test_indices
        )
        logger.info(f"Created new train-test split with {len(train_indices)} training samples")

    # Step 3: Initialize model and data loaders
    logger.info("Initializing model...")
    node_feature_dim = binding_site_graphs[0].x.shape[1]
    substrate_dim = substrate_tensor.shape[1]
    
    model = EnzymeKineticsModel(
        node_feature_dim=node_feature_dim,
        substrate_dim=substrate_dim,
        hidden_dim=256,  # 增大隐藏层维度
        output_dim=2
    ).to(device)
    
    # 创建数据加载器
    train_dataset = [
        (binding_site_graphs[i], substrate_tensor[i], y_tensor[i]) 
        for i in train_indices
    ]
    test_dataset = [
        (binding_site_graphs[i], substrate_tensor[i], y_tensor[i])
        for i in test_indices
    ]
    
    train_loader = GeometricDataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=True
    )
    test_loader = GeometricDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )
