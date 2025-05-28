from typing import List, Tuple
import torch
from torch_geometric.data import Data
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def compute_feature_importance(data: Data, target: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes feature importance for the attributes x, pos, edge_attr, and temperature.

    Parameters:
    - data: A PyTorch Geometric Data object containing the features.
    - target: The target values for which feature importance is computed.

    Returns:
    - importances: An array of feature importances.
    - feature_names: An array of feature names corresponding to the importances.
    """
    # Prepare the feature matrix
    features = torch.cat([data.x, data.pos, data.edge_attr, data.temperature.unsqueeze(0)], dim=1).numpy()
    
    # Train a Random Forest model
    model = RandomForestRegressor()
    model.fit(features, target.numpy())
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Define feature names
    feature_names = ['x_' + str(i) for i in range(data.x.shape[1])] + \
                    ['pos_' + str(i) for i in range(data.pos.shape[1])] + \
                    ['edge_attr_' + str(i) for i in range(data.edge_attr.shape[1])] + \
                    ['temperature']
    
    return importances, feature_names

def evaluate_feature_importance(data_list: List[Data], target_list: List[torch.Tensor]) -> None:
    """
    Evaluates feature importance across multiple data samples.

    Parameters:
    - data_list: A list of PyTorch Geometric Data objects.
    - target_list: A list of target tensors corresponding to each data object.
    """
    all_importances = []
    all_feature_names = []
    
    for data, target in zip(data_list, target_list):
        importances, feature_names = compute_feature_importance(data, target)
        all_importances.append(importances)
        all_feature_names = feature_names  # Assuming feature names are consistent across samples
    
    # Average importances across all samples
    avg_importances = np.mean(all_importances, axis=0)
    
    # Print or return the results
    for name, importance in zip(all_feature_names, avg_importances):
        print(f"Feature: {name}, Importance: {importance:.4f}")