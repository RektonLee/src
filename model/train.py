import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
from datetime import datetime
from sklearn.metrics import r2_score
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from biopandas.pdb import PandasPdb
from scipy.spatial import distance
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
from tqdm import tqdm

# Add root directory to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import required modules
from model import EnzymeKineticsPredictionModel
from utils.binding_site_utils import identify_binding_site, process_binding_site
from utils.data_utils import load_substrate_embeddings, create_geometric_graph

# Setup logging
def setup_logger(log_dir, name="training"):
    """Set up logger for the training process"""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def extract_binding_site_features(pdb_path, distance_cutoff=10.0):
    """
    Extract binding site atoms and their features from a PDB file
    
    Args:
        pdb_path: Path to the PDB file
        distance_cutoff: Distance cutoff for binding site atoms (Å)
        
    Returns:
        binding_site_atoms: DataFrame containing binding site atoms
        binding_site_coords: Numpy array of atom coordinates
        binding_site_features: Numpy array of atom features
    """
    # Implementation will go here
    pass

def create_binding_site_graph(atoms, coords, features, edge_cutoff=5.0):
    """
    Create a geometric graph from binding site atoms
    
    Args:
        atoms: DataFrame of binding site atoms
        coords: Numpy array of atom coordinates
        features: Numpy array of atom features
        edge_cutoff: Distance cutoff for creating edges (Å)
        
    Returns:
        data: PyTorch Geometric Data object representing the binding site graph
    """
    # Implementation will go here
    pass

def process_dataset(data_path, pdb_dir, output_dir, substrate_data_path, 
                    binding_site_cutoff=10.0, edge_cutoff=5.0, max_entries=None):
    """
    Process PDB files and substrate data to create dataset for training
    
    Args:
        data_path: Path to cleaned_data.csv
        pdb_dir: Directory containing PDB files
        output_dir: Directory to save processed data
        substrate_data_path: Path to substrate embeddings
        binding_site_cutoff: Distance cutoff for binding site atoms (Å)
        edge_cutoff: Distance cutoff for graph edges (Å)
        max_entries: Maximum number of entries to process (for debugging)
        
    Returns:
        binding_site_graphs: List of PyTorch Geometric Data objects for binding sites
        substrate_embeddings: Numpy array of substrate embeddings
        targets: Numpy array of targets (Km, kcat)
        valid_indices: List of valid indices that were successfully processed
    """
    # Implementation will go here
    pass

def train_model(binding_site_graphs, substrate_embeddings, targets, 
               train_indices, test_indices, output_dir,
               epochs=100, batch_size=32, lr=0.001, weight_decay=1e-5):
    """
    Train the enzyme kinetics prediction model
    
    Args:
        binding_site_graphs: List of PyTorch Geometric Data objects
        substrate_embeddings: Numpy array of substrate embeddings
        targets: Numpy array of targets (Km, kcat)
        train_indices: Indices for training set
        test_indices: Indices for test set
        output_dir: Directory to save model and results
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        weight_decay: Weight decay for regularization
        
    Returns:
        model: Trained model
        best_metrics: Dictionary of best validation metrics
    """
    # Implementation will go here
    pass

def plot_training_curves(train_losses, val_losses, r2_km_scores, r2_kcat_scores, output_dir):
    """
    Plot training curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        r2_km_scores: List of R² scores for Km
        r2_kcat_scores: List of R² scores for kcat
        output_dir: Directory to save plots
    """
    # Implementation will go here
    pass

def main():
    """Main function to orchestrate the training process"""
    parser = argparse.ArgumentParser(description="Train enzyme kinetics model from PDB files")
    
    parser.add_argument("--data_path", 
                        default="/home/lizihao/Work/enzyme_prediction/data/cleaned_data.csv",
                        help="Path to cleaned data CSV file")
    parser.add_argument("--pdb_dir", 
                        default="/home/lizihao/Work/enzyme_prediction/src/output/pdb_files",
                        help="Directory containing PDB files")
    parser.add_argument("--substrate_data", 
                        default="pro_data.npz",
                        help="Path to processed substrate data")
    parser.add_argument("--output_dir", 
                        help="Directory to save outputs")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, 
                        help="Weight decay for regularization")
    parser.add_argument("--binding_site_cutoff", type=float, default=10.0, 
                        help="Distance cutoff for binding site atoms (Å)")
    parser.add_argument("--edge_cutoff", type=float, default=5.0, 
                        help="Distance cutoff for graph edges (Å)")
    parser.add_argument("--test_ratio", type=float, default=0.2, 
                        help="Ratio of data to use for testing")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--max_entries", type=int, default=None, 
                        help="Maximum number of entries to process (for debugging)")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f"output/pdb_gnn_training/{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "binding_sites"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "graphs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
    
    # Setup logger
    logger = setup_logger(args.output_dir)
    logger.info("Starting training process...")
    logger.info(f"Arguments: {args}")
    
    # Process dataset
    logger.info("Processing PDB files and creating binding site graphs...")
    binding_site_graphs, substrate_embeddings, targets, valid_indices = process_dataset(
        args.data_path, 
        args.pdb_dir,
        args.output_dir,
        args.substrate_data,
        binding_site_cutoff=args.binding_site_cutoff,
        edge_cutoff=args.edge_cutoff,
        max_entries=args.max_entries
    )
    
    # Split dataset
    num_samples = len(valid_indices)
    indices = np.random.permutation(num_samples)
    test_size = int(num_samples * args.test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    logger.info(f"Total valid samples: {num_samples}")
    logger.info(f"Training samples: {len(train_indices)}")
    logger.info(f"Testing samples: {len(test_indices)}")
    
    # Train model
    logger.info("Training model...")
    model, best_metrics = train_model(
        binding_site_graphs,
        substrate_embeddings,
        targets,
        train_indices,
        test_indices,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Save processed data
    logger.info("Saving processed data...")
    np.savez(
        os.path.join(args.output_dir, "processed_data.npz"),
        substrate_embeddings=substrate_embeddings,
        targets=targets,
        valid_indices=np.array(valid_indices),
        train_indices=train_indices,
        test_indices=test_indices
    )
    
    torch.save(binding_site_graphs, os.path.join(args.output_dir, "binding_site_graphs.pt"))
    
    logger.info("Training completed!")
    logger.info(f"Best metrics: {best_metrics}")
    

if __name__ == "__main__":
    main()