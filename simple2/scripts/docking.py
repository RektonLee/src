#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import __main__
__main__.pymol_argv = ['pymol', '-c']
from joblib import Parallel, delayed
import os
import subprocess
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina
from Bio.PDB import PDBParser
from pymol2 import PyMOL
import hashlib
import shutil
PREP_BIN = "/home/lizihao/ADFRsuite_x86_64Linux_1.0/bin"
RAWPOCKET_DIR = "/home/lizihao/Work/enzyme_prediction/src/simple2/data/processed/rawpocket"
os.makedirs(RAWPOCKET_DIR, exist_ok=True)


def clean_altlocs(infile, outfile):
    with open(infile) as fin, open(outfile, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                if (line[12:16].strip() == "K" or line[12:16].strip() == "SE" or line[12:16].strip() == "ZN"):
                    continue
                altloc = line[16]
                if altloc in (" ", "A"):
                    fout.write(line[:16] + " " + line[17:])
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

    with open(output_pdbqt, "w") as fout:
        for line in lines:
            if line.startswith("HETATM"):
                x = float(line[30:38]) + shift[0]
                y = float(line[38:46]) + shift[1]
                z = float(line[46:54]) + shift[2]
                new_line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
                fout.write(new_line)
            else:
                fout.write(line)


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
                    continue
    if len(coords) == 0:
        raise ValueError(f"❌ 未提取到任何坐标点: {pocket_pdb}")
    coords = np.array(coords)
    min_coord = coords.min(axis=0)
    max_coord = coords.max(axis=0)
    center = ((max_coord + min_coord) / 2).tolist()
    size = (max_coord - min_coord + margin).tolist()
    return center, size


def extract_pocket_pymol(protein_path, ligand_path, output_path, cutoff=5.0):
    with PyMOL() as pymol:
        pymol.cmd.load(protein_path, "protein")
        pymol.cmd.load(ligand_path, "ligand")
        pymol.cmd.select("pocket", f"byres (ligand expand {cutoff})")
        pymol.cmd.save(f"../../{output_path}", "pocket")
        print(f"✅ PyMOL extracted pocket saved to {output_path}")


def run_preprocess(uniprot_id, smiles, prot_pdb_path, output_dir,index):
    base_name = f"{uniprot_id}_{int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff}"
    tmp_dir = f"tmp/{base_name}"
    os.makedirs(tmp_dir, exist_ok=True)
    orig_dir = os.getcwd()
    os.chdir(tmp_dir)

    try:
        # 自动生成输出文件名（"prot_clean.pdb"）
        output_path = os.path.splitext(prot_pdb_path)[0] + "_clean.pdb"
        
        # 调用函数
        clean_altlocs(prot_pdb_path, output_path)
        smiles_to_3d(smiles, "ligand.pdb")
        subprocess.run([f"{PREP_BIN}/prepare_ligand", "-l", "ligand.pdb", "-A", "hydrogens", "-o", "ligand.pdbqt"], check=True)
        subprocess.run([f"{PREP_BIN}/prepare_receptor", "-r", output_path, "-A", "hydrogens", "-U", "lps_waters_nonstdres", "-o", "receptor.pdbqt"], check=True)
        rawpocket_path = os.path.join(RAWPOCKET_DIR, f"{uniprot_id}_rawpocket.pdb")
        os.makedirs("autosite_out", exist_ok=True)
        if not os.path.exists(rawpocket_path):
            subprocess.run([f"{PREP_BIN}/autosite", "-r", "receptor.pdbqt", "-o", "autosite_out"], check=True)
            pocket_file = "receptor_cl_001.pdb"
            pocket_src = os.path.join("autosite_out", pocket_file)
            if os.path.exists(pocket_src):
                shutil.copyfile(pocket_src, rawpocket_path)
                print(f"✅ AutoSite口袋缓存成功: {rawpocket_path}")
            else:
                raise FileNotFoundError("❌ 未生成 receptor_cl_001.pdb")
        else:
            print(f"📦 使用缓存的 AutoSite 结果: {rawpocket_path}")

        # pockets = sorted([f for f in os.listdir("autosite_out") if f.endswith(".pdb")])
        # if len(pockets) == 0:
        #     # raise RuntimeError("❌ 未发现任何 pocket 文件")
        center, size = extract_box_from_pdb(rawpocket_path)
        # if len(pockets) >3:
        #     for p in pockets[3:]:
        #         os.remove(os.path.join("autosite_out", p))
        move_ligand_to_box_center("ligand.pdbqt", center, "ligand_moved.pdbqt")

        v = Vina()
        v.set_receptor(rigid_pdbqt_filename="receptor.pdbqt")
        v.set_ligand_from_file("ligand_moved.pdbqt")
        v.compute_vina_maps(center=center, box_size=size)
        v.dock(exhaustiveness=16, n_poses=5)
        v.write_poses("vina_out.pdbqt", n_poses=5, overwrite=True)

        subprocess.run(["/home/lizihao/autodock_vina_1_1_2_linux_x86/bin/vina_split", "--input", "vina_out.pdbqt", "--ligand", "vina_out_ligand_"], check=True)
        with open("pose0.pdb", "w") as w:
            for ln in open("vina_out_ligand_1.pdbqt"):
                if ln.startswith(("ATOM", "HETATM")):
                    w.write(ln[:66] + "\n")

        # with open("complex.pdb", "w") as w:
        #     w.writelines(open(f"{output_path}"))
            # for ln in open("pose0.pdb"):
            #     if ln.startswith("ATOM") or ln.startswith("HETATM"):
            #         ln = ln[:21] + 'L' + ln[22:]
            #     w.write(ln)

        out_pocket_path = os.path.join(output_dir, f"{base_name}_10A.pdb")
        # os.chdir(orig_dir)
        extract_pocket_pymol("receptor.pdbqt", "pose0.pdb", out_pocket_path, cutoff=5)
        print(f"✅ {base_name} 处理完成 -> {out_pocket_path}")
    except Exception as e:
        print(f"❌ 处理失败 {uniprot_id}: {e}")
        with open("/home/lizihao/Work/enzyme_prediction/src/simple2/data/failed_samples.txt", "a") as f:
            f.write(f"{uniprot_id},{index}\n")
    finally:
        os.chdir(orig_dir)
        subprocess.run(["rm", "-rf", tmp_dir])



if __name__ == "__main__":
    df = pd.read_csv("/home/lizihao/Work/enzyme_prediction/data/cleaned_data.csv")
    os.makedirs("data/processed/pockets", exist_ok=True)
    print(f"Working directory: {os.getcwd()}")
    # ...existing code...
    for idx,row in enumerate(df.itertuples()):
        uniprot = row.uniprot
        smiles = row.substrate_smiles.split(';')[0]
        
        pdb_path = f"/home/lizihao/Work/enzyme_prediction/src/output/pdb_files/{uniprot}.pdb"
        if os.path.exists(pdb_path):
            try:
                run_preprocess(uniprot, smiles, pdb_path, output_dir="data/processed/pockets",index=idx)
            except Exception as e:
                print(f"❌ 处理失败 {uniprot}: {e}")
        else:
            print(f"⚠️ 缺失 PDB 文件: {pdb_path}")
        print(f"✅ 处理完成 {idx+1}/{len(df)}")
    print("✅ 所有文件处理完成！")


# def preprocess_wrapper(idx, row):
#     uniprot = row.uniprot
#     smiles = row.substrate_smiles.split(';')[0]
#     pdb_path = f"/home/lizihao/Work/enzyme_prediction/src/output/pdb_files/{uniprot}.pdb"

#     if os.path.exists(pdb_path):
#         try:
#             run_preprocess(uniprot, smiles, pdb_path, output_dir="data/processed/pockets", index=idx)
#         except Exception as e:
#             print(f"❌ 处理失败 {uniprot}: {e}")
#     else:
#         print(f"⚠️ 缺失 PDB 文件: {pdb_path}")
#     print(f"✅ 处理完成 {idx+1}/{len(df)}")
# # print("✅ 所有文件处理完成！")

# if __name__ == "__main__":
#     df = pd.read_csv("/home/lizihao/Work/enzyme_prediction/data/cleaned_data.csv")
#     os.makedirs("data/processed/pockets", exist_ok=True)

#     Parallel(n_jobs=4)(delayed(preprocess_wrapper)(idx, row) for idx, row in enumerate(df.itertuples()))