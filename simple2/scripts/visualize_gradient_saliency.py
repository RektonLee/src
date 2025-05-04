import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载 gradient_saliency.npy 文件
file_path = '/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/v1_pretty/gradient_saliency.npy'
gradients = np.load(file_path)

# 可视化示例：绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(gradients, cmap='coolwarm', annot=False)
plt.title('Gradient Saliency Heatmap')
plt.xlabel('Features')
plt.ylabel('Samples')
plt.tight_layout()

# 保存图像
output_path = '/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/v1_pretty/gradient_saliency_heatmap.png'
plt.savefig(output_path)
plt.close()

print(f'✅ Gradient saliency heatmap saved to {output_path}')