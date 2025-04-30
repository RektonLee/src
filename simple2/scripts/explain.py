import torch
import numpy as np
import os
from torch_geometric.loader import DataLoader
from GNN_model import PocketGNN

def write_importance_to_pdb(pdb_path, importance_array, output_pdb):
    lines = open(pdb_path).readlines()
    atom_idx = 0
    with open(output_pdb, "w") as fout:
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                if atom_idx < len(importance_array):
                    b_factor = importance_array[atom_idx]
                    newline = line[:60] + f"{b_factor:6.2f}" + line[66:]
                    fout.write(newline)
                    atom_idx += 1
                else:
                    fout.write(line)
            else:
                fout.write(line)
    print(f"✅ Importance scores written to B-factor in {output_pdb}")

def explain(model_path, graph_path, pocket_pdb_path, task_idx=0, save_dir="outputs_explain"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    # Load graph
    data_list = torch.load(graph_path)  # Assume List[Data]
    loader = DataLoader(data_list, batch_size=1)
       # Load model
    sample_data = data_list[0]
    node_feature_dim = sample_data.x.size(1)
    edge_feature_dim = sample_data.edge_attr.size(1)
    model = PocketGNN1(
        node_input_dim=node_feature_dim,
        edge_input_dim=edge_feature_dim,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_importances = []

    for batch in loader:
        batch = batch.to(device)
        batch.x.requires_grad_(True)
        output = model(batch)[:, task_idx].sum()
        output.backward()

        grads = batch.x.grad.detach().cpu()
        inputs = batch.x.detach().cpu()

        importance = (grads * inputs).sum(dim=1)
        all_importances.append(importance)

    importances = torch.cat(all_importances, dim=0)
    importances = importances.numpy()

    # Save npy
    np.save(os.path.join(save_dir, "atom_importances.npy"), importances)
    print(f"✅ Atom importance scores saved at {save_dir}/atom_importances.npy")

    # Save colored pdb
    colored_pdb = os.path.join(save_dir, "pocket_colored.pdb")
    write_importance_to_pdb(pocket_pdb_path, importances, colored_pdb)

    print(f"\\n🎨 To visualize in PyMOL, run:")
    print(f"load {colored_pdb}")
    print(f"spectrum b, blue_white_red, pocket_colored")
    print(f"show sticks, pocket_colored")

    return importances

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--graph', type=str, required=True)
    parser.add_argument('--pocket', type=str, required=True)
    parser.add_argument('--task', type=int, default=0, help="0=kcat, 1=Km")
    parser.add_argument('--save_dir', type=str, default='outputs_explain')
    args = parser.parse_args()

    explain(args.model, args.graph, args.pocket, args.task, args.save_dir)
