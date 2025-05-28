import os
import torch
from torch_geometric.data import Data

def load_data(file_path):
    # Load data from a specified file path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Assuming the data is stored in a specific format, implement loading logic here
    # For example, loading a PyTorch Geometric Data object
    data = torch.load(file_path)
    return data

def preprocess_data(data):
    # Preprocess the data, e.g., normalization or transformation
    data.x = normalize(data.x)
    data.pos = normalize(data.pos)
    data.edge_attr = normalize(data.edge_attr)
    data.temperature = normalize(data.temperature.unsqueeze(0)).squeeze(0)
    return data

def normalize(tensor):
    # Normalize a tensor
    return (tensor - tensor.mean()) / tensor.std()

def save_data(data, file_path):
    # Save the processed data to a specified file path
    torch.save(data, file_path)