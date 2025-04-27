#!/usr/bin/env bash
set -e
receptor=enzyme.pdbqt
ligand=substrate.pdbqt

# 1 识别口袋
autosite -r $receptor -o autosite_out
top_cluster=$(ls autosite_out/*_CL.0*.pdb | head -n1)

# 2 让 vina_search_box 读 pocket pdb 自动给网格
vina_search_box --receptor $receptor \
                --pocket_pdb $top_cluster \
                --spacing 1.0 > box.txt
source box.txt   # 导出 CENTER_* SIZE_*

# 3 对接
vina --receptor $receptor --ligand $ligand \
     --center_x $CENTER_X --center_y $CENTER_Y --center_z $CENTER_Z \
     --size_x $SIZE_X --size_y $SIZE_Y --size_z $SIZE_Z \
     --exhaustiveness 20 --out docked.pdbqt

# 4 提取 10 Å 口袋
vina_split --input docked.pdbqt --ligand_out pose_0.pdbqt
python merge_complex.py enzyme.pdb pose_0.pdbqt complex.pdb
python pocket_extract.py complex.pdb 10 pocket10A.pdb
