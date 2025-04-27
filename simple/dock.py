#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os, subprocess, sys, re
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina
from Bio.PDB import PDBParser, NeighborSearch, Select, PDBIO

PREP_BIN = "/home/lizihao/ADFRsuite_x86_64Linux_1.0/bin"

def clean_altlocs(infile, outfile):
    with open(infile) as fin, open(outfile, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                if line[12:16].strip() == "K":
                    continue
                altloc = line[16]
                if altloc in (" ", "A"):
                    fout.write(line[:16] + " " + line[17:])  # 清掉 altLoc 标记
            else:
                fout.write(line)

def smiles_to_3d(smiles, outfile):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    AllChem.UFFOptimizeMolecule(mol)
    Chem.MolToPDBFile(mol, outfile)
def move_ligand_to_box_center(lig_pdbqt, target_center, output_pdbqt):
    lines = []
    coords = []
    for line in open(lig_pdbqt):
        if line.startswith("HETATM"):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
        lines.append(line)
    
    coords = np.array(coords)
    current_center = coords.mean(axis=0)
    shift = np.array(target_center) - current_center

    # 重写 pdbqt 坐标
    with open(output_pdbqt, "w") as fout:
        i = 0
        for line in lines:
            if line.startswith("HETATM"):
                x = float(line[30:38]) + shift[0]
                y = float(line[38:46]) + shift[1]
                z = float(line[46:54]) + shift[2]
                new_line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
                fout.write(new_line)
                i += 1
            else:
                fout.write(line)

    print(f"✅ 已将 ligand 平移至 box 中心：{target_center}")

def extract_box_from_pdb(pocket_pdb, margin=2.0):
    coords = []
    with open(pocket_pdb) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except:
                    continue  # 跳过坐标格式有误的行
    if len(coords) == 0:
        raise ValueError(f"❌ 没有从 {pocket_pdb} 中提取出任何坐标点，请检查文件格式是否正确。")
    coords = np.array(coords)
    min_coord = coords.min(axis=0)
    max_coord = coords.max(axis=0)
    center = ((max_coord + min_coord) / 2).tolist()
    size = (max_coord - min_coord + margin).tolist()
    return center, size

def extract_pocket_atoms(pdb_path, ligand_chain='L', radius=10.0):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("cx", pdb_path)[0]
    lig_atoms = [a for a in struct.get_atoms() if a.get_parent().get_parent().id == ligand_chain]
    prot_atoms = [a for a in struct.get_atoms() if a not in lig_atoms]
    ns = NeighborSearch(prot_atoms)
    pocket_atoms = set(lig_atoms)
    for la in lig_atoms:
        pocket_atoms.update(ns.search(la.coord, radius))
    class Sel(Select):
        def accept_atom(self, atom): return atom in pocket_atoms
    io = PDBIO(); io.set_structure(struct); io.save("pocket10A.pdb", Sel())

# ### === 路径和输入 ===
# prot_pdb = "P30838.pdb"
# ligand_smi = "ligand1.smi"

# ### === 步骤 1: 清理 altLocs ===
# clean_altlocs(prot_pdb, "clean.pdb")

# ### === 步骤 2: SMILES → 3D PDB ===
# smiles = open(ligand_smi).read().strip().split()[0]
# smiles_to_3d(smiles, "ligand.pdb")

# ### === 步骤 3: prepare_ligand / receptor ===
# subprocess.run([f"{PREP_BIN}/prepare_ligand", "-l", "ligand.pdb", "-A", "hydrogens", "-o", "ligand.pdbqt"], check=True)
# subprocess.run([f"{PREP_BIN}/prepare_receptor", "-r", "clean.pdb", "-A", "hydrogens", "-U", "lps_waters_nonstdres", "-o", "receptor.pdbqt"], check=True)
# os.makedirs("autosite_out", exist_ok=True)
# ### === 步骤 4: AutoSite 找口袋 ===
# subprocess.run(["/home/lizihao/ADFRsuite_x86_64Linux_1.0/bin/autosite", "-r", "receptor.pdbqt", "-o", "autosite_out"], check=True)
# pockets = [f for f in os.listdir("autosite_out") if f.endswith(".pdb")]
# pocket_pdb = f"autosite_out/{sorted(pockets)[0]}"
# center, size = extract_box_from_pdb(pocket_pdb)
# move_ligand_to_box_center("ligand.pdbqt", center, "ligand_moved.pdbqt")

# ### === 步骤 5: 对接 ===
# v = Vina()
# v.set_receptor(rigid_pdbqt_filename="receptor.pdbqt")
# v.set_ligand_from_file("ligand_moved.pdbqt")
# v.compute_vina_maps(center=center, box_size=size)

# print("预对接打分：", v.score()[0])
# print("优化后打分：", v.optimize()[0])
# v.write_pose("ligand_min.pdbqt", overwrite=True)

# v.dock(exhaustiveness=16, n_poses=20)
# v.write_poses("vina_out.pdbqt", n_poses=5, overwrite=True)

# === 步骤 6: 拆出 pose0，合并复合物，提取口袋原子 ===
# subprocess.run(["lizihao@bio502-134:~/autodock_vina_1_1_2_linux_x86/bin/vina_split", "--input", "vina_out.pdbqt", "--ligand_out", "pose0.pdbqt", "--ligand_format", "pdbqt"], check=True)
with open("pose0.pdb", "w") as w:
    for ln in open("vina_out_ligand_1.pdbqt"):
        if ln.startswith(("ATOM", "HETATM")):
            w.write(ln[:66] + "\n")

with open("complex.pdb", "w") as w:
    w.writelines(open("clean.pdb"))
    for ln in open("pose0.pdb"):
        # 给配体打个链ID 'L'，方便识别
        if ln.startswith("ATOM") or ln.startswith("HETATM"):
            ln = ln[:21] + 'L' + ln[22:]
        w.write(ln)

extract_pocket_atoms("complex.pdb")
print("✅ 完成！10 Å 口袋结构已写入 pocket10A.pdb")
