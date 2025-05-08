import os
import json
from datetime import datetime

def save_metadata(save_dir, **kwargs):
    """
    保存训练元数据到JSON文件
    
    参数:
        save_dir: 保存目录路径
        **kwargs: 要保存的元数据键值对
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 添加时间戳
    metadata = kwargs.copy()
    metadata['save_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 保存到JSON文件
    save_path = os.path.join(save_dir, 'metadata.json')
    with open(save_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✅ Metadata saved to {save_path}")

def load_metadata(save_dir):
    """
    从JSON文件加载元数据
    
    参数:
        save_dir: 保存目录路径
    
    返回:
        dict: 加载的元数据
    """
    save_path = os.path.join(save_dir, 'metadata.json')
    with open(save_path, 'r') as f:
        return json.load(f)