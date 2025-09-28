import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import GNN_model as MD

# 配置
dataset_path = "/home/lizihao/Work/enzyme_prediction/src/simple2/data/processed/dataset_DLKcat_S.pt"
base_dir = "/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention_rbf"
model_path=f"{base_dir}/best_model.pt"
# model_path = "/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention/best_model.pt"
batch_size = 32

# 加载数据
data_list = torch.load(dataset_path)
# np.random.shuffle(data_list)
# split = int(0.8 * len(data_list))
# val_data = data_list[split:]
val_loader = DataLoader(data_list, batch_size=batch_size)

# 加载模型
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
node_input_dim = data_list[0].x.shape[1]
edge_input_dim = data_list[0].edge_attr.shape[1]
model = MD.PocketGNNWithAttention(
    node_input_dim=node_input_dim,
    edge_input_dim=edge_input_dim,
).to(device) #没有temperature的

# model = MD.PocketGNN_Gated(input_dim=node_input_dim).to(device)
# model = MD.PocketGNN1(node_input_dim=node_input_dim, edge_input_dim=edge_input_dim).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 推理
all_y_true = []
all_y_pred = []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        batch_size = batch.num_graphs
        log_y = batch.y.view(batch_size, -1)
        out = model(batch)
        all_y_true.append(log_y.cpu())
        all_y_pred.append(out.cpu())

all_y_true = torch.cat(all_y_true, dim=0).numpy()
all_y_pred = torch.cat(all_y_pred, dim=0).numpy()
print(all_y_pred)

# # 反log10变换，获取原始尺度的值
all_y_true = np.power(10, all_y_true)
all_y_pred = np.power(10, all_y_pred)

# 计算kcat和km的误差
kcat_error = abs(all_y_pred[:, 0] - all_y_true[:, 0])/all_y_true[:, 0]
km_error = abs(all_y_pred[:, 1] - all_y_true[:, 1])/all_y_true[:, 1]

df = pd.DataFrame({
    "kcat_original": all_y_true[:, 0],  # 原始尺度的kcat真实值
    "kcat_pred_original": all_y_pred[:, 0],  # 原始尺度的kcat预测值
    "kcat_error": kcat_error,  # kcat的误差
    "km_original": all_y_true[:, 1],  # 原始尺度的km真实值
    "km_pred_original": all_y_pred[:, 1],  # 原始尺度的km预测值
    "km_error": km_error  # km的误差
})

df.to_csv(f"{base_dir}/val_pred_results_EG_S.csv", index=False)
print(f"✅ 导出完成: {base_dir}/val_pred_results_EG.csv")