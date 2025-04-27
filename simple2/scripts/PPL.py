import os
import subprocess
import argparse

def run_docking():
    print("\n🚀 Step 1: Running docking to extract pockets...")
    subprocess.run(["python", "docking.py"], check=True)

def run_graph_builder():
    print("\n🚀 Step 2: Building pocket graphs...")
    subprocess.run(["python", "graph_builder.py"], check=True)

def run_train():
    print("\n🚀 Step 3: Training GNN model...")
    subprocess.run([
        "python", "train.py",
        "--dataset", "data/processed/graph_dataset.pt",
        "--save_dir", "outputs"
    ], check=True)

def run_evaluate():
    print("\n🚀 Step 4: Evaluating model and plotting...")
    subprocess.run([
        "python", "evaluate.py",
        "--dataset", "data/processed/graph_dataset.pt",
        "--model", "outputs/best_model.pt",
        "--save_dir", "outputs"
    ], check=True)

def run_explain():
    print("\n🚀 Step 5: Explaining model with pocket heatmap...")
    subprocess.run([
        "python", "explain.py",
        "--model", "outputs/best_model.pt",
        "--graph", "data/processed/graph_dataset.pt",
        "--pocket", "data/processed/pocket_for_gnn.pdb",
        "--task", "0",
        "--save_dir", "outputs_explain"
    ], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_docking", action="store_true", help="Skip docking if pockets already prepared")
    args = parser.parse_args()

    if not args.skip_docking:
        run_docking()

    run_graph_builder()
    run_train()
    run_evaluate()
    run_explain()

    print("\n✅ Pipeline completed successfully!")
