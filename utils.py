# src/utils.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import logging
import matplotlib
matplotlib.use('Agg')


def visualize_true_vs_predicted(y_true, y_pred, timestamp, label_1="Km_log", label_2="kcat_log"):
    """
    可视化真实值与预测值的对比

    参数:
    y_true (np.array): 真实标签值。
    y_pred (np.array): 预测标签值。
    timestamp (str): 时间戳，用于文件名。
    label_1 (str, optional): Km 的标签名称，默认为 "Km_log"。
    label_2 (str, optional): kcat 的标签名称，默认为 "kcat_log"。
    """
    plt.figure(figsize=(10, 5))
    # 绘制Km的预测值与真实值对比
    plt.subplot(1, 2, 1)
    plt.scatter(y_true[:, 0], y_pred[:, 0], color='b', label=f'{label_1} Prediction')
    plt.plot([min(y_true[:, 0]), max(y_true[:, 0])], [min(y_true[:, 0]), max(y_true[:, 0])], 'r--')  # 对角线
    plt.xlabel(f"True {label_1}")
    plt.ylabel(f"Predicted {label_1}")
    plt.title(f"True vs Predicted {label_1}")
    plt.legend()

    # 绘制kcat的预测值与真实值对比
    plt.subplot(1, 2, 2)
    plt.scatter(y_true[:, 1], y_pred[:, 1], color='g', label=f'{label_2} Prediction')
    plt.plot([min(y_true[:, 1]), max(y_true[:, 1])], [min(y_true[:, 1]), max(y_true[:, 1])], 'r--')  # 对角线
    plt.xlabel(f"True {label_2}")
    plt.ylabel(f"Predicted {label_2}")
    plt.title(f"True vs Predicted {label_2}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"logs/true_vs_pred_{timestamp}.png") # 保存到 logs 目录下
    plt.show()

def visualize_error_distribution(y_true, y_pred, timestamp):
    """
    可视化预测误差分布。

    参数:
    y_true (np.array): 真实标签值。
    y_pred (np.array): 预测标签值。
    timestamp (str): 时间戳，用于文件名。
    """
    errors = y_true - y_pred
    plt.figure(figsize=(10, 5))
    sns.histplot(errors[:, 0], kde=True, color='blue', label='Km Error', alpha=0.6)
    sns.histplot(errors[:, 1], kde=True, color='green', label='kcat Error', alpha=0.6)
    plt.xlabel('Prediction Error')
    plt.title('Prediction Error Distribution')
    plt.legend()
    plt.savefig(f"logs/error_dist_{timestamp}.png") # 保存到 logs 目录下
    plt.show()

def calculate_r2(y_true, y_pred):
    """
    计算 R² 分数。

    参数:
    y_true (np.array): 真实标签值。
    y_pred (np.array): 预测标签值。

    返回:
    tuple: Km 和 kcat 的 R² 分数。
    """
    r2_km = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_kcat = r2_score(y_true[:, 1], y_pred[:, 1])
    return r2_km, r2_kcat


def visualize_attention_weights(attention_weights, substrate_smiles_list, timestamp, filename="logs/attention_weights"):
    """
    可视化底物注意力权重。

    参数:
    attention_weights (torch.Tensor): 注意力权重，形状为 [batch_size, max_substrates]。
    substrate_smiles_list (list): 底物 SMILES 列表，每个元素是一个包含多个SMILES的列表。
    timestamp (str): 时间戳，用于文件名。
    filename (str, optional):  文件名，默认为 "attention_weights"。
    """
    attention_weights_np = attention_weights.detach().cpu().numpy()
    num_substrates = attention_weights_np.shape[1]
    batch_size = attention_weights_np.shape[0]

    for i in range(batch_size):
        smiles_for_sample = substrate_smiles_list[i][:num_substrates] # 确保只取前 num_substrates 个
        if not smiles_for_sample: # 如果该样本没有底物，则跳过
            continue

        plt.figure(figsize=(8, 6))
        sns.heatmap(attention_weights_np[i].reshape(1, -1),
                    annot=True,
                    cmap="viridis",
                    cbar=True,
                    xticklabels=smiles_for_sample, # 使用 SMILES 作为 x 轴标签
                    yticklabels=[f'Sample {i+1} Attention Weight']) # y 轴标签表示样本
        plt.title(f'Attention Weights for Substrates - Sample {i+1}')
        plt.xlabel('Substrates (SMILES)')
        plt.ylabel('Attention Weight')
        plt.tight_layout()
        file_path = f"{filename}_{timestamp}_sample_{i+1}.png" # 文件名包含样本编号
        plt.savefig(file_path)
        logging.info(f"Attention weights heatmap saved to {file_path}")
        plt.close() # 关闭图形，避免显示多个图形