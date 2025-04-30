import os
import json
from datetime import datetime

def save_metadata(save_dir, dataset_path, graph_builder_version, gnn_model_version, comments=""):
    metadata = {
        "build_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset_path": dataset_path,
        "graph_builder_version": graph_builder_version,
        "gnn_model_version": gnn_model_version,
        "comments": comments
    }
    os.makedirs(save_dir, exist_ok=True)
    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Metadata saved to {metadata_path}")

def load_metadata(save_dir):
    metadata_path = os.path.join(save_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata.json found in {save_dir}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata
