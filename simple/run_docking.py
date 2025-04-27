#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess, sys, re, json, tempfile, os
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina
from Bio.PDB import PDBParser, NeighborSearch, Select, PDBIO
PREP_BIN = "/home/lizihao/ADFRsuite_x86_64Linux_1.0/bin"
# prot_pdb, smi_file = sys.argv[1], sys.argv[2]
prot_pdb="P30838.pdb"
smi_file = "ligand1.smi"

### 1. ligand: SMILES -> 3D PDB -> PDBQT
mol = Chem.MolFromSmiles(open(smi_file).read().strip())
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
AllChem.UFFOptimizeMolecule(mol)
lig_pdb = "ligand.pdb"
Chem.MolToPDBFile(mol, lig_pdb)
subprocess.run([f"{PREP_BIN}/prepare_ligand", "-l", lig_pdb, "-A", "hydrogens", "-o", "ligand.pdbqt"], check=True)

subprocess.run([f"{PREP_BIN}/prepare_receptor", "-r", prot_pdb, "-A", "hydrogens", "-U", "nphs_lps_waters_nonstdres", "-o", "receptor.pdbqt"], check=True)

### 3. AutoSite 找 pocket + vina_search_box 算盒子
subprocess.run(["autosite", "-r", "receptor.pdbqt", "-o", "autosite_out"], check=True)
pocket_pdb = sorted([f for f in os.listdir("autosite_out") if f.endswith(".pdb")])[0]  # 取排名最高
subprocess.run(["vina_search_box", "--receptor", "receptor.pdbqt",
                "--pocket_pdb", f"autosite_out/{pocket_pdb}","--spacing", "1.0"],
                capture_output=True, text=True, check=True, stdout=open("box.txt","w"))

box = {}
for ln in open("box.txt"):
    m = re.match(r"(CENTER_|SIZE_)([XYZ])\s*=\s*([-\d\.]+)", ln)
    if m:
        box.setdefault(m.group(1)[:-1], []).append(float(m.group(3)))
center, size = box["CENTER"], box["SIZE"]

### 4. Vina docking (Python API)
v = Vina()
v.set_receptor(rigid_pdbqt_filename="receptor.pdbqt")
v.set_ligand_from_file("ligand.pdbqt")
v.compute_vina_maps(center=center, box_size=size)

print("Score before minimization:", v.score()[0])
print("Score after  minimization:", v.optimize()[0])
v.write_pose("lig_minimized.pdbqt", overwrite=True)

v.dock(exhaustiveness=16, n_poses=20)
v.write_poses("vina_out.pdbqt", n_poses=5, overwrite=True)

### 5. split pose0, merge complex, cut 10 Å pocket
subprocess.run(["vina_split", "--input", "vina_out.pdbqt",
                "--ligand_out", "pose0.pdbqt", "--ligand_format", "pdbqt"], check=True)
# strip extra cols → PDB
with open("pose0.pdb", "w") as w:
    for ln in open("pose0.pdbqt"):
        if ln.startswith(("ATOM","HETATM")):
            w.write(ln[:66]+"\n")

with open("complex.pdb", "w") as w:
    w.writelines(open(prot_pdb))
    w.writelines(open("pose0.pdb"))

parser = PDBParser(QUIET=True)
s = parser.get_structure("cx", "complex.pdb")[0]
lig_atoms = [a for a in s.get_atoms() if a.get_parent().id=="L"]
prot_atoms = [a for a in s.get_atoms() if a.get_parent().id!="L"]
ns = NeighborSearch(prot_atoms)
pocket = set(lig_atoms)
for la in lig_atoms:
    pocket.update(ns.search(la.coord, 10.0))

class Sel(Select):
    def accept_atom(self, atom): return atom in pocket

io = PDBIO(); io.set_structure(s); io.save("pocket10A.pdb", Sel())
print("✓ 运行完毕，已得到 pocket10A.pdb")
