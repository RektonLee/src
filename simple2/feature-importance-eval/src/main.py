import torch
from torch_geometric.data import Data
from feature_importance import evaluate_feature_importance
from data_utils import load_data

def main():
    # Load the data
    data = load_data('/home/lizihao/Work/enzyme_prediction/src/simple2/data/processed/dataset_NAN_nopqr_rbf.pt')  # Update with actual data path

    # Evaluate feature importance
    importance_scores = evaluate_feature_importance(data)

    # Print the results
    for feature, score in importance_scores.items():
        print(f"Feature: {feature}, Importance Score: {score}")

if __name__ == "__main__":
    main()