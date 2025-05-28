# 基于图神经网络的酶动力学参数预测系统（主代码在simple2/scripts）

## 项目概述

本项目是一个基于图神经网络（GNN）的酶动力学参数预测系统，主要用于预测酶的催化性能参数（kcat和Km）。系统通过分析蛋白质活性口袋的局部结构，结合图神经网络技术，实现了对酶催化性能的准确预测。

## 核心功能

1. **结构预处理**
   - 蛋白质结构标准化
   - 底物分子3D结构生成
   - 分子对接与口袋提取
   - 结构优化与清理

2. **图数据构建**
   - 口袋原子节点特征提取
   - 原子间边关系构建
   - 全局特征融合
   - 图数据标准化

3. **GNN模型预测**
   - 多种GNN架构支持
   - 注意力机制集成
   - 温度等全局特征融合
   - 双参数（kcat、Km）预测

4. **结果分析与可视化**
   - 预测性能评估
   - 注意力权重分析
   - 结构可解释性研究
   - 结果可视化展示

## 技术架构

### 1. 数据处理模块

- **docking.py**: 负责蛋白质-配体复合物的结构预处理
  - 分子对接
  - 口袋提取
  - 结构优化

- **graph_builder.py**: 图数据构建
  - 节点特征编码
  - 边特征生成
  - 图数据转换

### 2. 模型架构

- **GNN_model.py**: 包含多种GNN模型实现
  - `PocketGNN`: 基础GCN结构
  - `PocketGNN1`: 边增强GNN
  - `PocketGNN_Gated`: 门控图卷积网络
  - `PocketGNNWithAttention`: 注意力机制GNN（主力模型）

### 3. 训练与评估

- **train.py**: 模型训练主脚本
  - 数据加载与预处理
  - 模型训练循环
  - 验证与评估
  - 结果保存

- **evaluate.py**: 模型评估工具
  - 性能指标计算
  - 预测结果分析
  - 可视化生成

## 模型特点

1. **多特征融合**
   - 原子级特征（元素类型、残基类型等）
   - 边特征（距离、方向等）
   - 全局特征（温度等）

2. **注意力机制**
   - 图注意力层
   - 多头注意力
   - 可解释性分析

3. **温度感知**
   - 温度特征编码
   - 全局特征融合
   - 条件预测

## 使用指南

### 环境配置

```bash
# 创建虚拟环境
conda create -n enzyme_pred python=3.8
conda activate enzyme_pred

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

1. 准备蛋白质PDB文件
2. 准备底物SMILES
3. 准备实验数据（kcat、Km）

### 运行流程

1. **结构预处理**
```bash
python docking.py --pdb_file input.pdb --smiles "CC(=O)O"
```

2. **图数据构建**
```bash
python graph_builder.py --input_dir processed_pdbs --output dataset.pt
```

3. **模型训练**
```bash
python train.py --dataset dataset.pt --model PocketGNNWithAttention
```

4. **结果评估**
```bash
python evaluate.py --model_path best_model.pt --test_data test.pt
```

## 项目结构

```
.
├── docking.py          # 结构预处理
├── graph_builder.py    # 图数据构建
├── GNN_model.py        # GNN模型定义
├── train.py           # 训练脚本
├── evaluate.py        # 评估工具
├── utils/             # 工具函数
└── requirements.txt   # 依赖列表
```

## 性能指标

- 平均绝对误差（MAE）
- 均方根误差（RMSE）
- Pearson相关系数
- 决定系数（R²）

## 注意事项

1. 确保输入数据质量
2. 注意GPU内存使用
3. 定期保存模型检查点
4. 关注模型可解释性

## 未来改进

1. 增加更多特征工程
2. 优化模型架构
3. 提升计算效率
4. 增强可解释性

## 引用

如果您使用了本项目，请引用相关论文：
[待补充]

## 许可证

[待补充]

## 联系方式

[待补充]
