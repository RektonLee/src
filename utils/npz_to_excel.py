import numpy as np
import pandas as pd
import argparse
import os

def array_to_dataframe(array, key):
    """Helper function to convert numpy array to pandas DataFrame"""
    if array.ndim == 0:  # scalar
        return pd.DataFrame([array.item()], columns=[key])
    elif array.ndim == 1:  # 1D array
        return pd.DataFrame(array, columns=[key])
    elif array.ndim == 2:  # 2D array
        return pd.DataFrame(array)
    else:  # 3D or higher
        # Flatten the array while preserving the first dimension
        rows = array.shape[0]
        cols = np.prod(array.shape[1:])
        return pd.DataFrame(array.reshape(rows, cols))

def npz_to_excel(npz_path, output_dir='output/data_analysis', max_rows=1000):
    """将NPZ文件内容转换为Excel文件"""
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载NPZ文件
        data = np.load(npz_path, allow_pickle=True)
        
        # 获取文件名（不含扩展名）作为Excel文件名前缀
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        
        # 创建一个ExcelWriter对象
        excel_path = os.path.join(output_dir, f'{base_name}_data.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 遍历NPZ文件中的所有数组
            for key in data.files:
                try:
                    array = data[key]
                    
                    # 将数组转换为DataFrame
                    df = array_to_dataframe(array, key)
                    
                    # 如果数据太大，只保存部分行
                    if len(df) > max_rows:
                        print(f"Warning: {key} has {len(df)} rows, only saving first {max_rows} rows")
                        df = df.head(max_rows)
                    
                    # 保存到Excel的不同sheet中
                    sheet_name = key[:31]  # Excel限制sheet名最长31字符
                    df.to_excel(writer, sheet_name=sheet_name, index=True)
                    
                except Exception as e:
                    print(f"Warning: Could not process array '{key}': {str(e)}")
                    continue
        
        print(f"数据已保存至: {excel_path}")
        
        # 打印每个数组的基本信息
        print("\n数据集信息:")
        print("-" * 50)
        for key in data.files:
            array = data[key]
            print(f"数组名称: {key}")
            print(f"形状: {array.shape}")
            print(f"数据类型: {array.dtype}")
            if array.size > 0 and array.ndim > 0:
                try:
                    print(f"数值范围: [{array.min()}, {array.max()}]")
                except:
                    print("数值范围: 无法计算")
            print("-" * 50)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将NPZ文件转换为Excel格式")
    parser.add_argument('--npz_path', type=str, required=True, 
                       help='NPZ文件路径')
    parser.add_argument('--output_dir', type=str, default='output/data_analysis',
                       help='输出目录')
    parser.add_argument('--max_rows', type=int, default=1000,
                       help='每个sheet最大保存的行数')
    args = parser.parse_args()
    
    npz_to_excel(args.npz_path, args.output_dir, args.max_rows)