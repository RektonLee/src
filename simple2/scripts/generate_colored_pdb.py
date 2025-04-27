import numpy as np

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, required=True, help="Pocket PDB file")
    parser.add_argument('--importance', type=str, required=True, help="Importance .npy file")
    parser.add_argument('--out', type=str, default="colored_pocket.pdb", help="Output colored PDB path")
    args = parser.parse_args()

    pocket_pdb = args.pdb
    importance_npy = args.importance
    output_pdb = args.out

    importance = np.load(importance_npy)
    write_importance_to_pdb(pocket_pdb, importance, output_pdb)

    print(f"""
    🧪 To visualize in PyMOL, run:
    load {output_pdb}
    spectrum b, blue_white_red, {output_pdb}
    show sticks, {output_pdb}
    """)