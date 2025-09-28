#!/bin/bash
# 快速并行PDB生成脚本

# 使用方法: ./quick_parallel.sh input.csv "0,1,2,3"
INPUT_CSV="$1"
GPUS="$2"

if [ -z "$INPUT_CSV" ] || [ -z "$GPUS" ]; then
    echo "使用方法: $0 <input.csv> <gpu_list>"
    echo "示例: $0 data.csv \"0,1,2,3\""
    exit 1
fi

# 解析GPU列表
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "🚀 启动 $NUM_GPUS 个并行进程"
echo "📁 输入: $INPUT_CSV"
echo "🖥️  GPU: $GPUS"

# 计算数据分块
TOTAL_LINES=$(wc -l < "$INPUT_CSV")
TOTAL_SAMPLES=$((TOTAL_LINES - 1))
CHUNK_SIZE=$((TOTAL_SAMPLES / NUM_GPUS))
REMAINDER=$((TOTAL_SAMPLES % NUM_GPUS))

echo "📊 总样本: $TOTAL_SAMPLES, 每块: ~$CHUNK_SIZE"

# 生成输出目录名（基于输入文件名）
BASE_NAME=$(basename "$INPUT_CSV" .csv)
OUTPUT_DIR="pdb_${BASE_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "📁 输出目录: $OUTPUT_DIR"

# 预先注册所有样本到共享SampleManager
echo "🔄 预注册所有样本到共享SampleManager..."
python3 -c "
from sample_manager import SampleManager
import pandas as pd

# 创建共享SampleManager
manager = SampleManager('$OUTPUT_DIR')

# 注册所有样本
df = pd.read_csv('$INPUT_CSV')
expected_samples = len(df)

# 检查是否需要重新注册（样本数量不匹配时重新注册）
if len(manager.sample_registry) != expected_samples:
    print(f'🔄 样本数量不匹配 (现有: {len(manager.sample_registry)}, 期望: {expected_samples})，重新注册...')
    # 清理现有注册
    manager.sample_registry.clear()
    manager.protein_index.clear()
    manager.ligand_index.clear()
    manager.save_metadata()
    
    # 重新注册
    manager.register_samples_from_csv('$INPUT_CSV')
    print(f'✅ 重新注册了 {len(df)} 个样本')
else:
    print(f'✅ 样本已存在且数量匹配 ({len(manager.sample_registry)} 个)')

# 显示去重统计
stats = manager.get_stats()
print(f'📊 去重统计:')
print(f'   总样本数: {stats[\"总样本数\"]}')
print(f'   唯一蛋白质数: {stats[\"唯一蛋白质数\"]}')
print(f'   去重率: {stats[\"去重率\"]}')
"

# 启动并行进程
PIDS=()
START_LINE=2

for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$i]}"
    
    # 计算当前块大小
    CURRENT_CHUNK_SIZE=$CHUNK_SIZE
    if [ $i -lt $REMAINDER ]; then
        CURRENT_CHUNK_SIZE=$((CHUNK_SIZE + 1))
    fi
    
    END_LINE=$((START_LINE + CURRENT_CHUNK_SIZE - 1))
    
    # 创建临时CSV
    TEMP_CSV="chunk_$((i+1)).csv"
    head -n 1 "$INPUT_CSV" > "$TEMP_CSV"
    sed -n "${START_LINE},${END_LINE}p" "$INPUT_CSV" >> "$TEMP_CSV"
    
    echo "📦 启动数据块 $((i+1)) (GPU $GPU_ID): 行 $START_LINE-$END_LINE"
    
    # 启动后台进程
    (
        echo "🔄 [GPU $GPU_ID] 开始处理数据块 $((i+1))"
        python3 generate_pdb_fixed.py \
            --input "$TEMP_CSV" \
            --sequence-column "sequence" \
            --sample-id-column "sample_id" \
            --use-sample-manager \
            --sample-data-dir "$OUTPUT_DIR" \
            --gpus "$GPU_ID" \
            --max-length 400 \
            --truncate-mode "skip" \
            --overwrite \
            --skip-registration
        echo "✅ [GPU $GPU_ID] 数据块 $((i+1)) 处理完成"
    ) &
    
    PIDS+=($!)
    START_LINE=$((END_LINE + 1))
done

echo "⏳ 等待所有进程完成..."
echo "🔄 进程ID: ${PIDS[*]}"

# 等待完成
SUCCESS=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "✅ 数据块 $((i+1)) 完成"
        ((SUCCESS++))
    else
        echo "❌ 数据块 $((i+1)) 失败"
    fi
done

echo "🎯 完成: $SUCCESS/$NUM_GPUS 个数据块成功"

# 生成最终去重统计报告
echo ""
echo "📊 最终去重统计报告:"
python3 -c "
from sample_manager import SampleManager
import os

manager = SampleManager('$OUTPUT_DIR')
stats = manager.get_stats()

print(f'   总样本数: {stats[\"总样本数\"]}')
print(f'   唯一蛋白质数: {stats[\"唯一蛋白质数\"]}')
print(f'   去重率: {stats[\"去重率\"]}')

# 统计实际生成的PDB文件
pdb_count = 0
for root, dirs, files in os.walk('$OUTPUT_DIR'):
    pdb_count += len([f for f in files if f.endswith('.pdb')])

print(f'   实际PDB文件数: {pdb_count}')
print(f'   节省的计算量: {stats[\"总样本数\"] - stats[\"唯一蛋白质数\"]} 个重复序列')
"

# 清理临时文件
rm -f chunk_*.csv

echo ""
echo "✅ 并行处理完成！重复序列已自动去重！"
echo "📁 PDB文件保存在: $OUTPUT_DIR"
echo "💡 使用此目录进行下一步预测: ./run_parallel_fixed.sh 4 $OUTPUT_DIR"
