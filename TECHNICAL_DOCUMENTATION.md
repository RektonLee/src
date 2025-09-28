## 酶动力学预测项目技术文档（全流程）

本技术文档系统性地介绍本项目从结构获取（PDB/ESMFold）、口袋提取与对接、图数据构建、GNN模型训练与预测、评估与可解释分析的全流程，覆盖核心文件、关键实现与使用方式。

目录
- 概览与数据流
- 环境与依赖
- 结构获取与对接
  - generate_pdb_fixed.py（ESMFold并行/去重/缓存）
  - pdb_quick_parallel.sh（并行调度与注册）
  - docking.py（口袋提取与对接）
- 图数据构建
  - graph_builder_rbf.py（核心构图与RBF边特征）
  - build_graph_dataset.py（增强边特征：角/二面角）
- 模型
  - GNN_model.py（基线与kcat-only版本）
  - GNN_model_enhanced.py（Enhanced模型）
- 训练脚本
  - train.py（kcat-only训练）
  - train_no_temp.py（无温度旧版）
  - train_optimized.py（大数据集超参与调度）
- 预测与并行
  - pred_range_fixed.py（范围预测）
  - run_parallel_fixed.sh（并行批跑与汇总）
  - pred.py（快速离线验证）
- 评估与可解释
  - evaluate.py（指标与残差）
  - explain.py（嵌入/梯度显著性）
- 最小可运行流程与命令
- 常见问题（FAQ）

---

### 概览与数据流

项目目标：预测酶动力学参数，当前主任务是仅预测 kcat（log10尺度）。

整体数据流：
1) 结构阶段（可选）
   - 通过 `generate_pdb_fixed.py` 获取或预测蛋白质结构（PDB）；
   - 使用 `docking.py` 提取结合口袋并生成口袋 PDB；
   - 并行流程可用 `pdb_quick_parallel.sh` 调度。
2) 构图阶段
   - `graph_builder_rbf.py` 将口袋 PDB 解析为图，节点/边特征（RBF距离编码）；
   - `build_graph_dataset.py` 在原RBF特征基础上增加键角与二面角特征，形成 24 维边特征；
3) 训练阶段
   - `GNN_model_enhanced.py` / `GNN_model.py` 定义模型；
   - `train.py` 使用 kcat-only 目标进行训练（log10）；
   - `train_optimized.py` 针对 4K+ 数据集的超参方案与调度；
4) 预测与评估
   - `pred_range_fixed.py`/`run_parallel_fixed.sh` 批量预测；
   - `evaluate.py` 与 `explain.py` 完成评估与可解释分析。

关键输入/输出：
- 输入 CSV：`kcat_data_for_training_corrected.csv`（列：`sample_id, sequence, substrate_smiles, kcat_value, ec, uniprot, temperature`）
- 口袋 PDB：`sample_data/samples/{sample_id}/{sample_id}_{hash}_10A.pdb`
- 图数据集：`.pt`（例如 `kcat_dataset_enhanced1.pt`）
- 训练输出：`outputs/.../best_model.pt` 与若干可视化图表

---

### 环境与依赖（建议）

- Python 3.10+
- PyTorch / PyTorch Geometric（含可选 `torch_cluster`）
- RDKit, BioPython, scikit-learn, seaborn, matplotlib, tqdm
- transformers（ESMFold 模型）
- 如果进行对接：AutoDock Vina/ADFRsuite（可选项，视 `docking.py` 路径）

注：项目对 `torch_cluster.radius_graph` 有兼容回退，缺失时自动使用纯 Python 边构建（`build_edges_manual`）。

---

## 结构获取与对接

### 1) data_loader.py（结构预测与活性位点识别）

文件：`data_loader.py`

核心职责：
- `ProteinStructureProcessor`：优先本地/下载/ESMFold 预测 PDB，带缓存与失败回退；
- 活性位点识别：尝试 UniProt 注释、fpocket 或结构启发式；
- 提供将 PDB 提取为口袋区域图的工具函数。

关键点：
- 方法 `predict_structure_with_sample_id(sample_id, sequence, uniprot_id)`：对接 `SampleManager` 实现序列级去重与共享 PDB；
- 方法 `extract_binding_site(...)`：在指定半径内抽取口袋原子并返回 `torch_geometric.data.Data`；
- 可选 `fpocket` 集成（若系统可用）。

### 2) generate_pdb_fixed.py（ESMFold并行/去重/缓存）

文件：`generate_pdb_fixed.py`

核心职责：
- 使用 transformers 的 `EsmForProteinFolding` 预测 PDB；
- 多 GPU 并行单序列模式；
- 全局序列去重数据库（避免重复预测），与 `SampleManager` 集成；
- OOM/失败回退（FP16、拆分、小序列兜底）。

典型用法：
```bash
python generate_pdb_fixed.py \
  --input your.csv \
  --sequence-column sequence \
  --sample-id-column sample_id \
  --use-sample-manager \
  --sample-data-dir sample_data \
  --gpus 0,1 --max-length 400 --truncate-mode skip --overwrite
```

### 3) pdb_quick_parallel.sh（并行调度与注册）

文件：`pdb_quick_parallel.sh`

核心职责：
- 将 CSV 按 GPU 数切块，分别调用 `generate_pdb_fixed.py` 并行运行；
- 启动前对所有样本进行 SampleManager 预注册（支持共享/去重机制）；
- 汇总基本统计。

用法：
```bash
bash pdb_quick_parallel.sh input.csv "0,1,2,3"
```

### 4) docking.py（口袋提取与对接）

文件：`docking.py`

核心职责：
- 通过 `prepare_ligand`/`prepare_receptor` 与 AutoDock Vina 工作流进行对接；
- 支持 `Bio.PDB` 的几何口袋截取（作为 PyMOL 的替代/后备）；
- 失败回退：若 AutoSite 失败，采用几何中心+固定盒子的 fallback 方案提取口袋；
- 产出：`*_10A.pdb` 风格的口袋 PDB（项目统一的口袋命名）。

小结：对于已有 `sample_data/samples/{sample_id}/{...}_10A.pdb` 的场景，可跳过结构预测与对接，仅使用后续构图与训练。

---

## 图数据构建

### 1) graph_builder_rbf.py（核心构图与RBF边特征）

文件：`graph_builder_rbf.py`

职责：
- `parse_pocket(pdb_path)`：解析 PDB（去氢），识别残基与是否配体；
- `build_graph(atoms, temperature)`：
  - 节点特征（约 52 维）：元素/残基 one-hot、是否配体、最近邻距离、电子结构、物理属性（质量/电负性/半径）；
  - 边：半径图（默认 4 Å），距离输入高斯 RBF 展开（16 维）；
  - 温度标准化为全局字段 `data.temperature`（旧逻辑，kcat-only 模式可忽略）。

关键函数（RBF）：
```python
def gaussian_rbf(edge_dist, num_centers=16, D_min=0.0, D_max=8.0, gamma=20.0):
    centers = torch.linspace(D_min, D_max, num_centers, device=edge_dist.device)
    diff = edge_dist - centers.view(1, -1)
    return torch.exp(-gamma * diff**2)
```

### 2) build_graph_dataset.py（增强边特征：角度与二面角）

文件：`build_graph_dataset.py`

职责：
- 在 `build_graph` 基础上增加边的几何统计特征：
  - 键角特征（对每条边，统计与相邻边的夹角余弦的 min/max/mean 与数量比例，共 4 维）；
  - 二面角特征（A-B-C-D 法向量夹角余弦的 min/max/mean 与数量比例，共 4 维）；
- 拼接得到增强的边特征：`16 (RBF) + 4 (angle) + 4 (dihedral) = 24 维`；
- 从 `kcat_data_for_training_corrected.csv` 读取样本，定位 `POCKET_BASE_DIR` 下口袋 PDB，构造 `Data`；
- 附带 `data.sample_id`、`data.ec` 便于跟踪；
- 目标 `data.y = [log10(kcat_value), log10(1.0)]`（第二位为占位）。

核心逻辑：
```python
# 增强边特征拼接
enhanced_edge_attr = torch.cat([
    original_edge_attr,  # [E,16]
    angle_features,      # [E,4]
    dihedral_features    # [E,4]
], dim=1)                # -> [E,24]
```

运行：
```bash
python build_graph_dataset.py
# 输出: kcat_dataset_enhanced1.pt
```

---

## 模型

### 1) GNN_model.py（基线与kcat-only版本）

文件：`GNN_model.py`

包含：
- `PocketGNN`/`PocketGNN1`：GCN/自定义边增强层的基线（双目标[kcat, Km]）；
- `PocketGNNWithAttention`/`PocketGNNWithAttentionNoTemp`：基于 GATConv 的注意力模型（是否融合温度可选）；
- `PocketGNNKcatOnly`：仅预测 kcat 的 GAT 模型（与 24 维边特征兼容）。

`PocketGNNKcatOnly` 的输出头：
```python
self.mlp = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(), nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.ReLU(), nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, 1)  # 只输出 kcat
)
```

### 2) GNN_model_enhanced.py（Enhanced模型）

文件：`GNN_model_enhanced.py`

增强点：
- 残差连接 + 层归一化（更深更稳）；
- 边特征在所有层使用（通过 `edge_encoder`）；
- 多尺度池化（mean+max+sum）并融合；
- 跳跃连接的 MLP（保留输入信息流）。

读出与融合：
```python
graph_mean = global_mean_pool(x, batch)
graph_max  = global_max_pool(x, batch)
graph_sum  = global_add_pool(x, batch)
graph_x = torch.cat([graph_mean, graph_max, graph_sum], dim=1)
graph_x = self.pool_fusion(graph_x)
```

建议：当前默认训练脚本使用 `GNN_model.PocketGNNKcatOnly` 已可满足 kcat-only；如追求更高性能，可将 `train.py` 切到 `PocketGNNKcatEnhanced`。

---

## 训练脚本

### 1) train.py（kcat-only训练）

文件：`train.py`

要点：
- 读取 `.pt` 数据集（`Data` 列表）；
- 自动读取 `node_input_dim` 与 `edge_input_dim`（应为 24）；
- 使用 `PocketGNNKcatOnly`，只取 `y[:,0:1]` 为 kcat 标签；
- 记录曲线并保存最佳模型至 `outputs/.../best_model.pt`。

运行：
```bash
python train.py --dataset kcat_dataset_enhanced1.pt --save_dir outputs/kcat_enhanced_model
```

### 2) train_no_temp.py（无温度旧版）

文件：`train_no_temp.py`

- 历史脚本，双目标（kcat/Km）；建议在当前 kcat-only 方案下使用 `train.py`。

### 3) train_optimized.py（大数据集超参与调度）

文件：`train_optimized.py`

要点：
- 根据数据集大小（2K/4K/10K）自动设置 `hidden_dim/num_layers/heads/dropout/batch_size`；
- `HuberLoss` + `ReduceLROnPlateau` + 梯度裁剪 + 早停；
- 训练结束输出综合可视化与 `final_stats.json`。

运行：
```bash
python train_optimized.py --dataset kcat_dataset_enhanced1.pt --save_dir outputs/kcat_optimized_model
```

---

## 预测与并行

### 1) pred_range_fixed.py（范围预测）

文件：`pred_range_fixed.py`

功能：
- 按 `start/end` 范围读取 CSV；
- 通过 `SampleManager` 复用/定位口袋 PDB；
- 用 `graph_builder_rbf.build_graph` 构图并预测；
- 输出 `predictions.csv / successful_predictions.csv / failed_predictions.csv`。

运行（示例）：
```bash
python pred_range_fixed.py \
  --input kcat_data_successful_pdb2.csv \
  --model /path/to/best_model.pt \
  --output results/some_batch \
  --start 1 --end 1000 \
  --use-sample-manager --sample-data-dir sample_data --temperature 303.15
```

### 2) run_parallel_fixed.sh（并行批跑与汇总）

文件：`run_parallel_fixed.sh`

功能：
- 将 CSV 按块切分，启动多个 `pred_range_fixed.py` 后台任务；
- 汇总各 chunk 的 `predictions/successful/failed` 到 `final_results/`；
- 生成 `final_stats.csv`。

运行：
```bash
bash run_parallel_fixed.sh 8 sample_data   # 8 个分块，使用 sample_data 为 PDB 基目录
```

运行机制与与 docking.py 的关系：
- 该脚本按块并行调用 `pred_range_fixed.py`。
- 在 `pred_range_fixed.py` 中，会基于 `sample_id` + `substrate_smiles` 的哈希定位口袋文件路径：
  `sample_data/samples/{sample_id}/{sample_id}_{hash}_10A.pdb`。
- 若口袋文件已存在，直接进入构图与预测；否则 `pred_range_fixed.py` 将回退执行对接生成：
  - 首先调用 `SampleManager` 定位蛋白 PDB；
  - 调用 `docking.run_preprocess(uniprot_or_sample_id, smiles, pdb_path, pocket_pdb, index)`；
  - `run_preprocess` 内部流程（见下文“对接流程细节”）。

对接流程细节（`docking.py` 关键步骤）：
1. 预处理与准备：
   - `clean_altlocs()`：清理金属离子/杂原子/ALTLOC，生成干净的受体 PDB；
   - `smiles_to_3d()`：用 RDKit 将底物 SMILES 转为 3D PDB；
   - 使用 ADFRsuite 的 `prepare_ligand`/`prepare_receptor` 生成 `ligand.pdbqt` 与 `receptor.pdbqt`；
2. 结合位点与盒子：
   - 首选 AutoSite 检测口袋（输出 `receptor_cl_001.pdb`），失败则 fallback：计算蛋白几何中心+固定盒子大小；
   - `move_ligand_to_box_center()` 将配体移动到盒子中心；
3. 对接：
   - 用 `Vina` 设定盒子并进行对接，输出 `vina_out.pdbqt`；
   - 拆分姿态 `vina_split`，导出 `pose0.pdb`；
4. 口袋提取与保存：
   - `extract_pocket_pymol()` 使用 Bio.PDB（非 PyMOL 依赖）按配体 5Å 球邻域截取口袋，输出为目标 `pocket_10A.pdb` 路径；
   - `pred_range_fixed.py` 将使用该口袋文件继续构图与预测。

注意：若批量数据已提前生成口袋（`*_10A.pdb`），整个对接阶段将被自动跳过，从而大幅加速并行预测。

### 3) pred.py（快速离线验证）

文件：`pred.py`

功能：
- 直接对现有 `.pt` 图数据与已训练模型进行批量前向并导出预测结果（研究用）。

---

## 评估与可解释

### 1) evaluate.py（指标与残差）

文件：`evaluate.py`

职责：
- 计算 MAE/RMSE/R2/Pearson；
- 残差分析（分布、残差 vs 预测、Q-Q 图）；
- 可视化嵌入（t-SNE/PCA）可选。

### 2) explain.py（嵌入/梯度显著性）

文件：`explain.py`

职责：
- 生成节点特征的梯度显著性；
- 对嵌入或显著性做 PCA/t-SNE 可视化；
- 输出特征重要性图表。

---

## 最小可运行流程与命令

前置：确保所有口袋 PDB 已在 `sample_data/samples/{sample_id}/{sample_id}_{hash}_10A.pdb`。

1) 构建图数据集（含角度/二面角）
```bash
python build_graph_dataset.py
# 产出: kcat_dataset_enhanced1.pt
```

2) 训练（kcat-only）
```bash
python train.py --dataset kcat_dataset_enhanced1.pt --save_dir outputs/kcat_enhanced_model
# 产出: outputs/kcat_enhanced_model/best_model.pt
```

3) 并行预测（若需要对新CSV批量预测）
```bash
bash run_parallel_fixed.sh 8 sample_data
# 汇总: results/parallel_.../final_results/
```

4) 评估与可解释
```bash
python evaluate.py --dataset kcat_dataset_enhanced1.pt --model outputs/kcat_enhanced_model/best_model.pt --save_dir outputs/eval
python explain.py  --dataset kcat_dataset_enhanced1.pt --model outputs/kcat_enhanced_model/best_model.pt --save_dir outputs/explain
```

（可选）使用 Enhanced 模型：将 `train.py` 中的模型替换为 `GNN_model_enhanced.PocketGNNKcatEnhanced` 并适配超参。

---

## 常见问题（FAQ）

1) `torch_cluster.radius_graph` 不可用？
   - `graph_builder_rbf.py` 会自动回退到 `build_edges_manual`，性能稍降但功能不受影响。

2) 为什么边特征是 24 维？
   - 16 维 RBF 距离展开 + 4 维键角统计 + 4 维二面角统计。

3) kcat-only 训练，`data.y` 为什么仍是 2 维？
   - 兼容旧脚本的 batch 维 reshape（[N,2]），当前只取第一列作为 kcat 标签，第二列为占位。

4) 是否必须进行 `docking.py`？
   - 若已有口袋 PDB（`*_10A.pdb`），可直接构图训练；对接流程主要用于从结构自动生成口袋。

5) Enhanced 与基线模型如何选择？
   - 若追求稳定与速度，`PocketGNNKcatOnly` 足够；如需更高上限，可尝试 `PocketGNNKcatEnhanced`（更深、更强的读出与 MLP）。

---

致谢：该文档基于以下核心文件梳理而成：

- 构图：`graph_builder_rbf.py`, `build_graph_dataset.py`
- 模型：`GNN_model.py`, `GNN_model_enhanced.py`
- 训练：`train.py`, `train_no_temp.py`, `train_optimized.py`
- 预测：`pred_range_fixed.py`, `run_parallel_fixed.sh`, `pred.py`
- 结构/对接：`data_loader.py`, `generate_pdb_fixed.py`, `pdb_quick_parallel.sh`, `docking.py`



---

## 附录：模型与训练的深入讲解

### 模型设计哲学（针对分子口袋图）
- 信息保持：使用残差/跳跃连接，避免深层网络的信息衰减；
- 几何约束：边特征在所有层使用，RBF + 角/二面角编码空间结构；
- 多尺度表示：mean/max/sum 池化融合，兼顾整体与极值与总量；
- 训练稳定性：层归一化/Dropout/梯度裁剪/鲁棒损失。

### PocketGNNKcatOnly（训练默认）
- 主干：多层 `GATConv`（首层读入 24 维边特征），中间 `ELU` 激活；
- 读出：`global_mean_pool`；
- 输出头：三层 MLP（含 Dropout），最终 `Linear(..., 1)` 输出 kcat（log10）。

优点：结构简单、稳定，适合 2K-10K 规模数据。

### PocketGNNKcatEnhanced（可选强化）
- 边特征：`edge_encoder` 将 24 维边特征编码并在所有 GAT 层使用；
- 深度稳定：层层残差 + `LayerNorm`；
- 多尺度池化：mean + max + sum 三路并联后融合；
- 跳跃连接 MLP：引入输入级跳连 `skip_proj*` 保持信息高速公路。

适用：更大数据集、更高性能需求时替换 `train.py` 中模型为 Enhanced 版本。

### 训练细节
标签与尺度：
- 数据集中 `data.y` 为形如 `[log10(kcat), log10(1.0)]`；训练时只取第一列作为监督信号；
- 预测输出即为 `log10(kcat)`；可视化/评估时保持在 log10 尺度以稳定度量。

损失与优化：
- 基础版（`train.py`）：`MSELoss` + Adam(lr=1e-3) + 梯度裁剪；
- 优化版（`train_optimized.py`）：`HuberLoss(delta=1.0)`，`ReduceLROnPlateau` 自适应调 lr，早停机制，自动按数据规模设置隐藏维度/层数/heads/dropout/batch_size。

评估指标：
- `R2`、`MAE`、`RMSE`、`Pearson` 在 log 尺度上计算；
- 训练期间绘制 loss/R2/Pearson 曲线，结束后绘制预测-真实散点与密度图。

超参建议（经验）：
- 2K 样本：hidden_dim=256, layers=6, heads=8, dropout=0.1, batch=32；
- 4K-8K：hidden_dim=384, layers=8, heads=12, dropout=0.15, batch=64；
- 10K+：hidden_dim=512, layers=10, heads=16, dropout=0.2, batch=128。

数据与鲁棒性：
- 构图前确保口袋 PDB 有效且非空，原子数 <3 的样本建议跳过；
- 训练前检查 `data.x/edge_attr/y` 是否含 NaN；
- 若 `torch_cluster` 缺失，自动退化为纯 Python 边构建，训练可继续但速度稍慢。
