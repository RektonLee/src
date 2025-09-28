#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import __main__
__main__.pymol_argv = ['pymol', '-c']
# from joblib import Parallel, delayed
import os
import subprocess
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina
from Bio.PDB import PDBParser, Select, PDBIO
from pymol2 import PyMOL
import hashlib
import shutil
PREP_BIN = "/home/lizihao/ADFRsuite_x86_64Linux_1.0/bin"
RAWPOCKET_DIR = "/home/lizihao/Work/enzyme_prediction/src/simple2/data/processed/rawpocket"
os.makedirs(RAWPOCKET_DIR, exist_ok=True)


def clean_altlocs(infile, outfile):
    metals = {"MG", "MN", "FE", "ZN", "CA", "CU", "CO", "NI", "NA", "K", "CL"}  # 常见金属离子
    with open(infile) as fin, open(outfile, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                atomname = line[12:16].strip()
                if atomname in {"K", "SE", "ZN"} or resname in metals:
                    continue  # 删除金属离子、杂原子
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

def create_simple_pdbqt(pdb_file, pdbqt_file):
    """创建简单的PDBQT文件用于Vina"""
    try:
        with open(pdb_file, 'r') as f:
            pdb_lines = f.readlines()
        
        with open(pdbqt_file, 'w') as f:
            # 写入REMARK行
            f.write("REMARK  Name = {}\n".format(os.path.basename(pdb_file)))
            f.write("REMARK  0 active torsions:\n")
            f.write("REMARK  status: ('A' for Active; 'I' for Inactive)\n")
            f.write("REMARK                            x       y       z     vdW  Elec       q    Type\n")
            f.write("REMARK                         _______ _______ _______ _____ _____    ______ ____\n")
            
            # 写入原子行
            atom_count = 0
            for line in pdb_lines:
                if line.startswith(('ATOM', 'HETATM')):
                    atom_count += 1
                    # 解析PDB格式
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    chain_id = line[21]
                    res_num = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    occupancy = float(line[54:60]) if line[54:60].strip() else 1.0
                    temp_factor = float(line[60:66]) if line[60:66].strip() else 0.0
                    
                    # 确定原子类型
                    atom_type = 'C'
                    if atom_name.startswith('N'):
                        atom_type = 'N'
                    elif atom_name.startswith('O'):
                        atom_type = 'O'
                    elif atom_name.startswith('S'):
                        atom_type = 'S'
                    elif atom_name.startswith('H'):
                        atom_type = 'H'
                    
                    # 计算电荷（简化版本）
                    charge = 0.0
                    if atom_name.startswith('H'):
                        charge = 0.1
                    elif atom_name.startswith('O'):
                        charge = -0.1
                    elif atom_name.startswith('N'):
                        charge = 0.1
                    
                    # 格式化ATOM行
                    atom_line = "ATOM  {:5d} {:4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f} {:5.2f} {:5.2f}    {:7.3f} {:2s}\n".format(
                        atom_count,
                        atom_name,
                        res_name,
                        chain_id,
                        res_num,
                        x,
                        y,
                        z,
                        occupancy,
                        temp_factor,
                        charge,
                        atom_type
                    )
                    f.write(atom_line)
        
        print(f"✅ 简单PDBQT文件创建成功: {pdbqt_file}")
        return True
        
    except Exception as e:
        print(f"❌ 创建简单PDBQT失败: {e}")
        return False

def find_binding_site_center(protein_pdb, ligand_pdb=None):
    """识别结合位点中心"""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', protein_pdb)
        
        # 如果有配体文件，基于配体位置计算中心
        if ligand_pdb and os.path.exists(ligand_pdb):
            ligand_structure = parser.get_structure('ligand', ligand_pdb)
            ligand_atoms = []
            for model in ligand_structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            ligand_atoms.append(atom.get_coord())
            
            if ligand_atoms:
                center = np.mean(ligand_atoms, axis=0)
                print(f"✅ 基于配体计算结合位点中心: {center}")
                return center
        
        # 否则基于蛋白质结构计算中心
        protein_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        protein_atoms.append(atom.get_coord())
        
        if protein_atoms:
            center = np.mean(protein_atoms, axis=0)
            print(f"✅ 基于蛋白质计算结合位点中心: {center}")
            return center
        
        # 默认中心
        default_center = [0.0, 0.0, 0.0]
        print(f"⚠️ 使用默认结合位点中心: {default_center}")
        return default_center
        
    except Exception as e:
        print(f"❌ 结合位点识别失败: {e}")
        return [0.0, 0.0, 0.0]

def extract_pocket_region(protein_pdb, center, radius=10.0, output_pdb=None):
    """提取口袋区域"""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', protein_pdb)
        
        class PocketSelect(Select):
            def accept_atom(self, atom):
                coord = atom.get_coord()
                distance = np.linalg.norm(coord - np.array(center))
                return distance <= radius
        
        if output_pdb is None:
            output_pdb = protein_pdb.replace('.pdb', '_pocket.pdb')
        
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_pdb, PocketSelect())
        
        print(f"✅ 口袋区域已保存: {output_pdb}")
        return output_pdb
        
    except Exception as e:
        print(f"❌ 口袋提取失败: {e}")
        return None


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


def create_fallback_pocket(protein_pdb_path, output_pocket_path):
    """
    当AutoSite失败时，创建基于蛋白质几何中心的简单口袋
    """
    import numpy as np
    
    # 读取蛋白质坐标
    coords = []
    with open(protein_pdb_path, 'r') as f:
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
        raise ValueError("❌ 未找到蛋白质原子坐标")
    
    coords = np.array(coords)
    center = coords.mean(axis=0).tolist()
    size = [20.0, 20.0, 20.0]  # 固定大小的盒子
    
    print(f"🔧 Fallback口袋中心: {center}")
    print(f"🔧 Fallback口袋大小: {size}")
    
    # 创建一个简单的口袋PDB文件（包含蛋白质中心附近的原子）
    radius = 10.0  # 10Å半径
    with open(output_pocket_path, 'w') as out_f:
        with open(protein_pdb_path, 'r') as in_f:
            for line in in_f:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        
                        # 计算到中心的距离
                        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                        if dist <= radius:
                            out_f.write(line)
                    except:
                        continue
    
    print(f"✅ Fallback口袋文件创建: {output_pocket_path}")
    return center, size


def extract_pocket_pymol(protein_path, ligand_path, output_path, cutoff=5.0):
    """使用Bio.PDB提取口袋区域（替代PyMOL）"""
    try:
        from Bio.PDB import PDBParser, PDBIO, Select
        import numpy as np
        
        print(f"🔍 加载蛋白质: {protein_path}")
        print(f"🔍 加载配体: {ligand_path}")
        
        parser = PDBParser(QUIET=True)
        
        # 读取蛋白质和配体结构
        protein_structure = parser.get_structure('protein', protein_path)
        ligand_structure = parser.get_structure('ligand', ligand_path)
        
        # 获取配体原子坐标
        ligand_atoms = []
        for model in ligand_structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        ligand_atoms.append(atom.get_coord())
        
        if not ligand_atoms:
            raise ValueError("配体文件中没有找到原子")
        
        ligand_coords = np.array(ligand_atoms)
        print(f"🔍 配体包含 {len(ligand_atoms)} 个原子")
        
        class PocketSelect(Select):
            def accept_residue(self, residue):
                # 检查残基中是否有原子在cutoff范围内
                for atom in residue:
                    atom_coord = atom.get_coord()
                    distances = np.linalg.norm(ligand_coords - atom_coord, axis=1)
                    if np.min(distances) <= cutoff:
                        return True
                return False
        
        # 使用绝对路径，避免相对路径问题
        abs_output_path = os.path.abspath(output_path)
        print(f"🔍 保存口袋到: {abs_output_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
        
        # 保存口袋区域
        io = PDBIO()
        io.set_structure(protein_structure)
        io.save(abs_output_path, PocketSelect())
        
        # 强制刷新文件系统缓存
        import time
        time.sleep(0.1)  # 短暂等待确保文件写入完成
        
        # 检查文件是否真的保存了
        if os.path.exists(abs_output_path):
            file_size = os.path.getsize(abs_output_path)
            if file_size > 0:
                print(f"✅ Bio.PDB extracted pocket saved to {abs_output_path} (size: {file_size} bytes)")
            else:
                raise ValueError(f"口袋文件为空: {abs_output_path}")
        else:
            print(f"❌ 口袋文件未保存: {abs_output_path}")
            raise FileNotFoundError(f"口袋文件未保存: {abs_output_path}")
        
    except Exception as e:
        print(f"❌ 口袋提取失败: {e}")
        # 输出更详细的调试信息
        print(f"   蛋白质文件: {protein_path} (存在: {os.path.exists(protein_path)})")
        print(f"   配体文件: {ligand_path} (存在: {os.path.exists(ligand_path)})")
        print(f"   输出路径: {abs_output_path}")
        print(f"   输出目录: {os.path.dirname(abs_output_path)} (存在: {os.path.exists(os.path.dirname(abs_output_path))})")
        raise


def run_preprocess(uniprot_id, smiles, prot_pdb_path, output_pocket_path, index):
    base_name = f"{uniprot_id}_{int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff}"
    tmp_dir = f"tmp/{base_name}"
    os.makedirs(tmp_dir, exist_ok=True)
    orig_dir = os.getcwd()
    
    # 确保prot_pdb_path是绝对路径
    prot_pdb_path = os.path.abspath(prot_pdb_path)
    
    os.chdir(tmp_dir)

    try:
        # 自动生成输出文件名（"prot_clean.pdb"）
        output_path = os.path.splitext(prot_pdb_path)[0] + "_clean.pdb"
        # 确保使用绝对路径
        output_path = os.path.abspath(output_path)
        if smiles=="C(=O)=O":
            raise ValueError("❌ SMILES 解析失败: C(=O)=O")
        # 调用函数
        clean_altlocs(prot_pdb_path, output_path)
        smiles_to_3d(smiles, "ligand.pdb")
        subprocess.run([f"{PREP_BIN}/prepare_ligand", "-l", "ligand.pdb", "-A", "hydrogens", "-o", "ligand.pdbqt"], check=True)
        subprocess.run([f"{PREP_BIN}/prepare_receptor", "-r", output_path, "-A", "hydrogens", "-U", "lps_waters_nonstdres", "-o", "receptor.pdbqt"], check=True)
        rawpocket_path = os.path.join(RAWPOCKET_DIR, f"{uniprot_id}_rawpocket.pdb")
        os.makedirs("autosite_out", exist_ok=True)
        use_fallback = False
        
        if not os.path.exists(rawpocket_path):
            subprocess.run([f"{PREP_BIN}/autosite", "-r", "receptor.pdbqt", "-o", "autosite_out"], check=True)
            pocket_file = "receptor_cl_001.pdb"
            pocket_src = os.path.join("autosite_out", pocket_file)
            if os.path.exists(pocket_src):
                shutil.copyfile(pocket_src, rawpocket_path)
                print(f"✅ AutoSite口袋缓存成功: {rawpocket_path}")
            else:
                print(f"⚠️ AutoSite未找到结合位点，使用fallback策略")
                use_fallback = True
        else:
            print(f"📦 使用缓存的 AutoSite 结果: {rawpocket_path}")

        # 如果AutoSite失败，使用fallback策略
        if use_fallback:
            # 创建基于蛋白质几何中心的简单口袋
            center, size = create_fallback_pocket(output_path, rawpocket_path)
        else:
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

        # 使用传入的输出路径，但是先切换回原始目录
        os.chdir(orig_dir)
        out_pocket_path = os.path.abspath(output_pocket_path)
        print(f"🔍 准备提取口袋: {out_pocket_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(out_pocket_path), exist_ok=True)
        
        # 切换回临时目录获取文件路径
        os.chdir(tmp_dir)
        receptor_path = os.path.abspath("receptor.pdbqt")
        pose_path = os.path.abspath("pose0.pdb")
        print(f"🔍 受体路径: {receptor_path}")
        print(f"🔍 配体路径: {pose_path}")
        
        # 检查文件是否存在
        if not os.path.exists(receptor_path):
            raise FileNotFoundError(f"受体文件不存在: {receptor_path}")
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"配体文件不存在: {pose_path}")
        
        extract_pocket_pymol(receptor_path, pose_path, out_pocket_path, cutoff=5)
        
        # 验证口袋文件是否成功生成
        if os.path.exists(out_pocket_path):
            file_size = os.path.getsize(out_pocket_path)
            print(f"✅ {base_name} 处理完成 -> {out_pocket_path} (size: {file_size} bytes)")
            return True  # 返回成功标志
        else:
            raise FileNotFoundError(f"口袋文件未成功保存到目标位置: {out_pocket_path}")
            
    except Exception as e:
        print(f"❌ 处理失败 {uniprot_id}: {e}")
        with open("/home/lizihao/Work/enzyme_prediction/src/simple2/data/failed_samples.txt", "a") as f:
            f.write(f"{uniprot_id},{index}\n")
        return False
    finally:
        os.chdir(orig_dir)
        # 延迟清理，确保文件已经保存完成
        try:
            subprocess.run(["rm", "-rf", tmp_dir], timeout=10)
        except subprocess.TimeoutExpired:
            print(f"⚠️ 清理临时目录超时: {tmp_dir}")
        except Exception as cleanup_e:
            print(f"⚠️ 清理临时目录时出错: {cleanup_e}")



if __name__ == "__main__":
    df = pd.read_csv("/home/lizihao/Work/enzyme_prediction/src/simple2/Example_DLKcat_S.csv")
    os.makedirs("data/processed/pockets", exist_ok=True)
    print(f"Working directory: {os.getcwd()}")
    # ...existing code...
    for idx,row in enumerate(df.itertuples()):
        uniprot = row.uniprot
        smiles = row.substrate_smiles.split(';')[0]
        
        pdb_path = f"/home/lizihao/Work/enzyme_prediction/src/simple2/output/pdb_files/{uniprot}.pdb"
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