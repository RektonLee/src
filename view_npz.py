import numpy as np
import argparse

def view_array_slice(array, start=5, end=10, name=""):
    """显示数组指定范围的内容"""
    print(f"\n=== {name} (索引 {start}-{end}) ===")
    if array.ndim == 1:
        for i in range(start, min(end, len(array))):
            print(f"索引 {i}: {array[i]}")
    else:
        for i in range(start, min(end, len(array))):
            print(f"索引 {i}: {array[i, :10]}...")  # 只显示前10个数字

def view_npz_content(npz_path):
    """查看NPZ文件内容的函数"""
    data = np.load(npz_path, allow_pickle=True)
    
    print("\n=== NPZ文件内容概览 ===")
    print(f"文件路径: {npz_path}")
    
    # 显示基本信息
    for key in data.files:
        array = data[key]
        print(f"\n数组名称: {key}")
        print(f"形状: {array.shape}")
        print(f"类型: {array.dtype}")
    
    # 详细显示protein和substrate的embeddings
    if 'protein_features' in data.files:
        view_array_slice(data['protein_features'], 5, 10, "蛋白质嵌入")
    
    if 'substrate_features' in data.files:
        view_array_slice(data['substrate_features'], 5, 10, "底物指纹")
    
    # 如果有原始SMILES和序列信息，显示对应关系
    if 'metadata' in data.files:
        metadata = data['metadata'].item()
        if isinstance(metadata, dict):
            print("\n=== 原始序列信息 (索引 5-10) ===")
            if 'protein_sequences' in metadata:
                proteins = metadata['protein_sequences']
                for i in range(5, min(10, len(proteins))):
                    print(f"\n蛋白质 {i}:")
                    print(f"序列前50个字符: {proteins[i][:50]}...")
            
            if 'substrate_smiles' in metadata:
                smiles = metadata['substrate_smiles']
                print("\n=== SMILES信息 (索引 5-10) ===")
                for i in range(5, min(10, len(smiles))):
                    print(f"底物 {i}: {smiles[i]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看NPZ文件内容")
    parser.add_argument('npz_path', default="output/processed/processed_data_0406_21.npz", type=str, help='NPZ文件路径')
    args = parser.parse_args()
    
    view_npz_content(args.npz_path)