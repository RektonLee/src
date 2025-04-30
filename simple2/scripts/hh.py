import torch
dataset = torch.load('/home/lizihao/Work/enzyme_prediction/src/simple2/data/processed/dataset_raw.pt')
sample_data = dataset[0]  # 取第一个图数据

print("包含的图数据属性:", dir(sample_data))
print("是否有edge_attr:", hasattr(sample_data, 'edge_attr'))
if hasattr(sample_data, 'edge_attr'):
    print("edge_attr形状:", sample_data.edge_attr.shape)