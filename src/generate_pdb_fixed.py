# -*- coding: utf-8 -*-
"""Robust PDB generation script using ESMFold with multi-GPU, batching, length control & fallbacks.

Features added:
1. Multi-GPU support (DataParallel) with --gpus 0,1
2. Batching (--batch-size) to improve throughput when sequences are short
3. Max sequence length handling (--max-length) with truncate / skip / chunk modes
4. OOM resilience: retry with half precision (--fp16) and finally fallback to dummy linear CA trace
5. Caching: skip existing PDBs unless --overwrite
6. Detailed logging (progress, timings, fallbacks) saved to pdb_generation.log
7. Dummy PDB generator for failed predictions
8. Graceful degradation if no GPU: will run on CPU (slow)
9. **NEW: SampleManager integration for organized file management**
10. **NEW: Skip failed samples instead of using dummy PDB to ensure data quality**
"""

import os
import time
import math
import json
import logging
import argparse
from typing import List, Tuple
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from sample_manager import SampleManager  # New import

# ---------------------------------
# Logging setup
# ---------------------------------
def setup_logger(log_path='pdb_generation.log'):
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler()
        ]
    )


# ---------------------------------
# Utility: dummy PDB fallback (保留用于调试，但不用于生产)
# ---------------------------------
AA3 = {
    'A': 'ALA','C': 'CYS','D': 'ASP','E': 'GLU','F': 'PHE','G': 'GLY','H': 'HIS','I': 'ILE','K': 'LYS','L': 'LEU',
    'M': 'MET','N': 'ASN','P': 'PRO','Q': 'GLN','R': 'ARG','S': 'SER','T': 'THR','V': 'VAL','W': 'TRP','Y': 'TYR'
}

def make_dummy_pdb(sequence: str) -> str:
    """生成dummy PDB结构（仅用于调试，不用于生产预测）"""
    lines = ["HEADER    DUMMY STRUCTURE"]
    for i, aa in enumerate(sequence):
        res3 = AA3.get(aa, 'GLY')
        x = float(i * 3.8); y = 0.0; z = 0.0
        lines.append(f"ATOM  {i+1:5d}  CA  {res3} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  ")
    lines.append("END")
    return "\n".join(lines)


# ---------------------------------
# ESM output conversion
# ---------------------------------
def convert_outputs_to_pdb(outputs) -> List[str]:
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs_np = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs_np["atom37_atom_exists"]
    pdbs = []
    
    # 检查输出数据的有效性
    if outputs_np["aatype"].shape[0] == 0:
        logging.error("ESMFold输出为空，无法生成PDB")
        return []
    
    for i in range(outputs_np["aatype"].shape[0]):
        try:
            aa = outputs_np["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = outputs_np["residue_index"][i] + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=outputs_np["plddt"][i],
                chain_index=outputs_np.get("chain_index", [None])[i] if "chain_index" in outputs_np else None,
            )
            pdbs.append(to_pdb(pred))
        except Exception as e:
            logging.error(f"转换第{i}个结构时出错: {e}")
            # 如果转换失败，添加一个空的PDB字符串
            pdbs.append("")
    
    return pdbs


# ---------------------------------
# Predictor class
# ---------------------------------
class ESMFoldPredictor:
    def __init__(self, gpus: str, fp16: bool = False, max_batch_tokens: int = 4096):
        
        if gpus:
            self.device_ids = [int(i) for i in gpus.split(',') if i.strip() != '']
        use_cuda = torch.cuda.is_available() and len(self.device_ids) > 0
        self.device = torch.device(f'cuda:{self.device_ids[0]}' if use_cuda else 'cpu')
        self.max_batch_tokens = max_batch_tokens
        logging.info(f"Using device: {self.device}")
        torch.backends.cuda.matmul.allow_tf32 = True
        self.fp16 = fp16 and use_cuda
        self.original_fp16 = self.fp16  # 记录初始精度设置

        logging.info("Loading ESMFold model (facebook/esmfold_v1)...")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        
        # 为每个GPU创建独立的模型实例
        self.models = {}
        self.devices = {}
        
        for gpu_id in self.device_ids:
            device = torch.device(f'cuda:{gpu_id}')
            self.devices[gpu_id] = device
            
            # 为每个GPU加载独立的模型
            model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
            model.to(device)
            
            # 精度控制
            if self.fp16:
                model.half()
                logging.info(f"Model on GPU {gpu_id} converted to half precision (FP16)")
            
            model.eval()
            self.models[gpu_id] = model
            logging.info(f"Loaded ESMFold model on GPU {gpu_id}")
        
        logging.info(f"Initialized {len(self.device_ids)} GPU models for parallel single-sequence processing")

    def predict_single_sequence(self, sequence: str, gpu_id: int) -> str:
        """在指定GPU上预测单个序列"""
        try:
            device = self.devices[gpu_id]
            model = self.models[gpu_id]
            
            # 单个序列tokenize
            tokenized = self.tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
            input_ids = tokenized['input_ids'].to(device)
            
            with torch.no_grad():
                try:
                    output = model(input_ids)
                except IndexError as e:
                    if "index 0 is out of bounds for dimension 0 with size 0" in str(e):
                        logging.error(f"ESMFold内部错误 (GPU {gpu_id}): {e}")
                        raise ValueError(f"ESMFold模型内部错误: {e}")
                    else:
                        raise e
            
            # 转换为PDB
            pdb_contents = convert_outputs_to_pdb(output)
            
            if len(pdb_contents) == 0:
                raise ValueError("ESMFold输出为空")
            
            return pdb_contents[0]  # 返回第一个（也是唯一的）PDB
            
        except Exception as e:
            logging.error(f"序列预测失败 (GPU {gpu_id}): {e}")
            raise e

    def predict_parallel_single_sequences(self, sequences: List[str]) -> List[Tuple[int, str, str]]:
        """多GPU并行处理单序列：GPU0处理前一半，GPU1处理后一半"""
        import concurrent.futures
        import threading
        
        results = []
        total_sequences = len(sequences)
        
        # 按GPU数量分配序列
        sequences_per_gpu = total_sequences // len(self.device_ids)
        remainder = total_sequences % len(self.device_ids)
        
        # 创建任务列表
        tasks = []
        start_idx = 0
        
        for i, gpu_id in enumerate(self.device_ids):
            # 计算当前GPU处理的序列数量
            current_batch_size = sequences_per_gpu + (1 if i < remainder else 0)
            end_idx = start_idx + current_batch_size
            
            # 为当前GPU分配序列
            gpu_sequences = sequences[start_idx:end_idx]
            for j, seq in enumerate(gpu_sequences):
                seq_idx = start_idx + j
                tasks.append((seq_idx, seq, gpu_id))
            
            start_idx = end_idx
            logging.info(f"GPU {gpu_id} 将处理 {len(gpu_sequences)} 个序列 (索引 {start_idx-len(gpu_sequences)}-{start_idx-1})")
        
        # 使用线程池并行处理
        def process_sequence(task):
            seq_idx, sequence, gpu_id = task
            try:
                logging.info(f"Processing sequence {seq_idx} on GPU {gpu_id} (length: {len(sequence)})")
                pdb_content = self.predict_single_sequence(sequence, gpu_id)
                logging.info(f"Successfully processed sequence {seq_idx} on GPU {gpu_id}")
                return (seq_idx, sequence, pdb_content)
            except Exception as e:
                logging.error(f"序列 {seq_idx} 在GPU {gpu_id} 上处理失败: {e}")
                return (seq_idx, sequence, None)
        
        # 并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.device_ids)) as executor:
            future_to_task = {executor.submit(process_sequence, task): task for task in tasks}
            
            for future in concurrent.futures.as_completed(future_to_task):
                result = future.result()
                results.append(result)
        
        # 按原始顺序排序
        results.sort(key=lambda x: x[0])
        return results

    def predict_batch(self, sequences: List[str]) -> List[Tuple[str, str]]:
        """批处理预测，支持智能分组和多GPU并行"""
        results = []
        
        # 按长度排序，便于智能batching
        seq_with_idx = [(i, seq) for i, seq in enumerate(sequences)]
        seq_with_idx.sort(key=lambda x: len(x[1]))
        
        # 智能分组：相似长度的序列组成batch
        batches = self._group_sequences_by_length(seq_with_idx)
        
        for batch_indices, batch_seqs in batches:
            try:
                batch_results = self._predict_single_batch(batch_seqs)
                # 恢复原始顺序
                for idx, (seq, pdb) in zip(batch_indices, batch_results):
                    results.append((idx, seq, pdb))
            except Exception as e:
                logging.warning(f"Batch prediction failed: {e}, falling back to individual prediction")
                # 批处理失败时，逐个处理
                for idx, seq in zip(batch_indices, batch_seqs):
                    try:
                        individual_result = self._predict_single_batch([seq])
                        results.append((idx, seq, individual_result[0][1]))
                    except Exception as e2:
                        logging.error(f"Individual prediction failed for seq len={len(seq)}: {e2}")
                        # 方案1：跳过失败的样本，不生成dummy PDB
                        logging.error(f"跳过序列 {seq[:20]}... (长度: {len(seq)})")
                        continue
        
        # 恢复原始输入顺序
        results.sort(key=lambda x: x[0])
        return [(seq, pdb) for _, seq, pdb in results]
    
    def _group_sequences_by_length(self, seq_with_idx: List[Tuple[int, str]]) -> List[Tuple[List[int], List[str]]]:
        """按长度分组序列，优化批处理效率"""
        if not seq_with_idx:
            return []
        
        # 按长度排序
        seq_with_idx.sort(key=lambda x: len(x[1]))
        
        batches = []
        current_batch_indices = []
        current_batch_seqs = []
        current_tokens = 0
        
        for idx, seq in seq_with_idx:
            seq_tokens = len(seq)
            
            # 如果添加这个序列会超过token限制，开始新batch
            if current_tokens + seq_tokens > self.max_batch_tokens and current_batch_seqs:
                batches.append((current_batch_indices, current_batch_seqs))
                current_batch_indices = []
                current_batch_seqs = []
                current_tokens = 0
            
            current_batch_indices.append(idx)
            current_batch_seqs.append(seq)
            current_tokens += seq_tokens
        
        # 添加最后一个batch
        if current_batch_seqs:
            batches.append((current_batch_indices, current_batch_seqs))
        
        logging.info(f"Grouped {len(seq_with_idx)} sequences into {len(batches)} batches")
        return batches
    
    def _predict_single_batch(self, sequences: List[str]) -> List[Tuple[str, str]]:
        """预测单个batch，支持OOM回退"""
        try:
            # 批量tokenize（padding到最长序列）
            tokenized = self.tokenizer(sequences, return_tensors="pt", padding=True, add_special_tokens=False)
            input_ids = tokenized['input_ids'].to(self.device)
            
            # 精度控制
            if self.fp16:
                # 注意：input_ids是LongTensor，不能.half()，但attention_mask可以
                if 'attention_mask' in tokenized:
                    attention_mask = tokenized['attention_mask'].to(self.device).half()
                else:
                    attention_mask = None
            else:
                attention_mask = tokenized.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
            
            with torch.no_grad():
                # 传入attention_mask避免padding位置的计算
                if attention_mask is not None:
                    output = self.model(input_ids, attention_mask=attention_mask)
                else:
                    output = self.model(input_ids)
            
            # 转换为PDB
            pdb_contents = convert_outputs_to_pdb(output)
            
            # 方案1：检查PDB转换结果，如果失败则抛出异常
            if len(pdb_contents) == 0:
                logging.error("PDB转换结果为空，跳过这些序列")
                raise ValueError("ESMFold输出为空，无法生成有效的PDB结构")
            
            if len(pdb_contents) != len(sequences):
                logging.error(f"PDB数量({len(pdb_contents)})与序列数量({len(sequences)})不匹配")
                raise ValueError(f"PDB转换结果数量不匹配: 期望{len(sequences)}个，实际{len(pdb_contents)}个")
            
            return [(seq, pdb) for seq, pdb in zip(sequences, pdb_contents)]
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logging.warning(f"OOM encountered with batch size {len(sequences)}, attempting fallback")
                torch.cuda.empty_cache()
                
                # 首次OOM尝试FP16
                if not self.fp16 and self.device.type == 'cuda':
                    logging.info("Trying FP16 fallback...")
                    self.model.half()
                    self.fp16 = True
                    try:
                        return self._predict_single_batch(sequences)  # 递归重试
                    except Exception as e2:
                        logging.error(f"FP16 fallback failed: {e2}")
                
                # 如果batch size > 1，尝试分割
                if len(sequences) > 1:
                    logging.info("Trying to split batch...")
                    mid = len(sequences) // 2
                    left_results = self._predict_single_batch(sequences[:mid])
                    right_results = self._predict_single_batch(sequences[mid:])
                    return left_results + right_results
                
                # 单序列仍OOM，跳过该序列
                logging.error(f"Single sequence OOM (len={len(sequences[0])}), skipping sequence")
                raise ValueError(f"序列过长导致OOM，跳过序列: {len(sequences[0])}个氨基酸")
            else:
                raise e


# ---------------------------------
# Sequence preprocessing
# ---------------------------------
def handle_sequence(seq: str, max_length: int, mode: str) -> Tuple[str, dict]:
    """Apply max length策略.
    mode: truncate | skip | headtail
    """
    meta = {"original_length": len(seq), "processed_length": len(seq)}
    
    if len(seq) <= max_length:
        meta["mode"] = "unchanged"
        return seq, meta
    
    if mode == "skip":
        meta["mode"] = "skipped"
        meta["skipped"] = True
        meta["reason"] = f"Length {len(seq)} exceeds limit {max_length}"
        return seq, meta  # 返回原序列，但标记为跳过
    
    elif mode == "truncate":
        meta["mode"] = "truncated"
        meta["processed_length"] = max_length
        return seq[:max_length], meta
    
    elif mode == "headtail":
        # 保留头部和尾部，中间截断
        head_len = max_length // 2
        tail_len = max_length - head_len
        meta["mode"] = "headtail"
        meta["processed_length"] = max_length
        return seq[:head_len] + seq[-tail_len:], meta

    else:
        # default fallback to truncate
        meta["mode"] = "truncate"
        return seq[:max_length], meta


# ---------------------------------
# Main execution
# ---------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate PDBs using ESMFold with robustness features")
    parser.add_argument('--input', type=str, default='/home/lizihao/Work/enzyme_prediction/PGNN/km_test_data.csv', help='Input CSV with sample_id, sequence, smiles columns')
    parser.add_argument('--sequence-column', type=str, default='sequence', help='Column name for sequences')
    parser.add_argument('--sample-id-column', type=str, default='sample_id', help='Column name for sample IDs')
    parser.add_argument('--use-sample-manager', action='store_true', help='Use SampleManager for organized file management')
    parser.add_argument('--sample-data-dir', type=str, default='sample_data', help='SampleManager base directory')
    parser.add_argument('--output-dir', type=str, default='output/pdb_files', help='Directory to save PDBs (when not using SampleManager)')
    parser.add_argument('--gpus', type=str, default='0,1', help='Comma-separated GPU ids, e.g. 0,1 (empty for CPU)')
    parser.add_argument('--batch-size', type=int, default=1, help='Deprecated: use --max-batch-tokens instead')
    parser.add_argument('--max-batch-tokens', type=int, default=8192, help='Max tokens per batch (auto-adjusted for multi-GPU)')
    parser.add_argument('--max-length', type=int, default=500, help='Max sequence length for ESMFold (400 recommended for stability)')
    parser.add_argument('--truncate-mode', type=str, choices=['truncate','skip','headtail'], default='skip', help='Strategy when sequence exceeds max length')
    parser.add_argument('--fp16', action='store_true', help='Load model in FP16 (saves memory)')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing (saves memory)')
    parser.add_argument('--cpu-offload', action='store_true', help='Offload model to CPU between predictions')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing PDB files')
    parser.add_argument('--skip-registration', action='store_true', help='Skip sample registration (assume already registered)')
    parser.add_argument('--report-json', type=str, default='pdb_report.json', help='Save summary JSON')
    args = parser.parse_args()

    setup_logger()

    # 设置输出目录和样本管理
    if args.use_sample_manager:
        sample_manager = SampleManager(args.sample_data_dir)
        
        # 根据skip_registration参数决定是否注册样本
        if not args.skip_registration:
            # 从CSV注册样本
            sample_manager.register_samples_from_csv(args.input)
            logging.info(f"Registered samples from {args.input}")
        else:
            # 检查是否已经注册，如果没有则重新注册
            if len(sample_manager.sample_registry) == 0:
                logging.info(f"Sample registry is empty, registering samples from {args.input}")
                sample_manager.register_samples_from_csv(args.input)
            else:
                logging.info(f"Skipping sample registration (assume already registered)")
            
        output_dir = sample_manager.base_dir
        logging.info(f"Using SampleManager with base dir: {output_dir}")
        logging.info(f"Total registered samples: {len(sample_manager.sample_registry)}")
    else:
        sample_manager = None
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    required_columns = [args.sequence_column]
    if args.use_sample_manager:
        required_columns.append(args.sample_id_column)
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input CSV")

    # 优化的预测器初始化
    predictor = ESMFoldPredictor(
        gpus=args.gpus, 
        fp16=args.fp16, 
        max_batch_tokens=args.max_batch_tokens
    )
    
    # 应用额外的内存优化
    if args.gradient_checkpointing and hasattr(predictor.model, 'gradient_checkpointing_enable'):
        predictor.model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing")

    sequences = df[args.sequence_column].tolist()
    sample_ids = df[args.sample_id_column].tolist() if args.use_sample_manager else [f"seq_{i+1}" for i in range(len(sequences))]
    report = []
    start_time = time.time()

    # 单序列多GPU并行处理逻辑
    logging.info(f"Processing {len(sequences)} sequences using {len(predictor.device_ids)} GPUs in parallel")
    
    # 全局序列去重机制
    import hashlib
    import json
    from pathlib import Path
    
    # 序列去重数据库文件
    SEQUENCE_DB_FILE = os.path.join(output_dir, "sequence_database.json")
    sequence_database = {}
    
    # 加载现有的序列数据库
    if os.path.exists(SEQUENCE_DB_FILE):
        try:
            with open(SEQUENCE_DB_FILE, 'r') as f:
                sequence_database = json.load(f)
            logging.info(f"加载了 {len(sequence_database)} 个已处理的序列")
        except Exception as e:
            logging.warning(f"无法加载序列数据库: {e}")
            sequence_database = {}
    
    # 预处理所有序列 - 实现真正的序列级别去重
    processed_sequences = []
    
    for idx, (raw_seq, sample_id) in enumerate(zip(sequences, sample_ids)):
        proc_seq, meta = handle_sequence(raw_seq, args.max_length, args.truncate_mode)
        
        # 计算序列hash
        seq_hash = hashlib.md5(proc_seq.encode()).hexdigest()
        
        # 检查序列是否已经处理过
        if seq_hash in sequence_database:
            # 序列已存在，使用共享PDB
            shared_info = sequence_database[seq_hash]
            shared_pdb_path = Path(shared_info['pdb_path'])
            
            # 检查共享PDB文件是否仍然存在
            if shared_pdb_path.exists():
                # 使用索引机制，不复制文件，只记录共享信息
                if args.use_sample_manager:
                    # 为SampleManager创建符号链接或记录共享信息
                    current_pdb_path, _ = sample_manager.get_protein_path(sample_id)
                    if not current_pdb_path.exists() or args.overwrite:
                        # 创建符号链接指向共享PDB
                        current_pdb_path.parent.mkdir(parents=True, exist_ok=True)
                        if current_pdb_path.exists():
                            current_pdb_path.unlink()  # 删除现有文件
                        current_pdb_path.symlink_to(shared_pdb_path)
                        logging.info(f"[{idx+1}/{len(sequences)}] 创建共享PDB链接: {sample_id} -> {shared_info['original_sample_id']}")
                        
                        # 更新SampleManager记录共享信息
                        sample_manager.update_sample_status(sample_id, "shared", 
                                                          pdb_path=str(current_pdb_path),
                                                          shared_from=shared_info['original_sample_id'],
                                                          shared_pdb_path=str(shared_pdb_path))
                    else:
                        logging.info(f"[{idx+1}/{len(sequences)}] 跳过 {sample_id} (PDB已存在)")
                else:
                    # 非SampleManager模式，创建符号链接
                    pdb_filename = f"seq_{idx+1}.pdb"
                    pdb_path = os.path.join(output_dir, pdb_filename)
                    if not os.path.exists(pdb_path) or args.overwrite:
                        if os.path.exists(pdb_path):
                            os.remove(pdb_path)
                        os.symlink(shared_pdb_path, pdb_path)
                        logging.info(f"[{idx+1}/{len(sequences)}] 创建共享PDB链接: {sample_id} -> {shared_info['original_sample_id']}")
                
                meta['status'] = 'cached_shared'
                meta['is_shared'] = True
                meta['shared_from'] = shared_info['original_sample_id']
                meta['shared_pdb_path'] = str(shared_pdb_path)
                meta['pdb_path'] = str(current_pdb_path if args.use_sample_manager else pdb_path)
                report.append(meta)
                continue
            else:
                # 共享PDB文件不存在，从数据库中移除
                logging.warning(f"共享PDB文件不存在: {shared_pdb_path}，从数据库中移除")
                del sequence_database[seq_hash]
        
        # 检查是否需要跳过
        if args.use_sample_manager:
            pdb_path, is_shared = sample_manager.get_protein_path(sample_id)
            if pdb_path.exists() and not args.overwrite:
                if is_shared:
                    logging.info(f"[{idx+1}/{len(sequences)}] Skip {sample_id} (using shared PDB)")
                else:
                    logging.info(f"[{idx+1}/{len(sequences)}] Skip existing {sample_id}")
                meta['status'] = 'cached'
                meta['is_shared'] = is_shared
                report.append(meta)
                continue
        else:
            pdb_filename = f"seq_{idx+1}.pdb"
            pdb_path = os.path.join(output_dir, pdb_filename)
            if os.path.exists(pdb_path) and not args.overwrite:
                meta['status'] = 'cached'
                report.append(meta)
                continue
        
        if meta.get('skipped'):
            # 跳过超长序列，不生成文件
            logging.info(f"[{idx+1}/{len(sequences)}] Skipped {sample_id} (length: {len(raw_seq)} > {args.max_length})")
            if args.use_sample_manager:
                sample_manager.log_failure(sample_id, "length_check", f"Sequence too long: {len(raw_seq)} > {args.max_length}", len(raw_seq))
            meta['status'] = 'skipped_too_long'
            meta['reason'] = f'Length {len(raw_seq)} exceeds limit {args.max_length}'
            report.append(meta)
            continue
        
        processed_sequences.append((idx, proc_seq, sample_id, pdb_path, meta))
        logging.info(f"[{idx+1}/{len(sequences)}] Added {sample_id} for processing (length: {len(proc_seq)})")
    
    logging.info(f"Total sequences to process: {len(processed_sequences)}")
    
    if not processed_sequences:
        logging.info("No sequences to process")
    else:
        # 提取序列用于并行处理
        seqs_for_processing = [item[1] for item in processed_sequences]
        
        # 多GPU并行单序列预测
        try:
            logging.info(f"Starting parallel single-sequence prediction on {len(predictor.device_ids)} GPUs")
            logging.info(f"Processing {len(seqs_for_processing)} sequences")
            parallel_results = predictor.predict_parallel_single_sequences(seqs_for_processing)
            logging.info(f"Got {len(parallel_results)} results from parallel processing")
            
            # 保存结果
            for (idx, proc_seq, sample_id, pdb_path, meta), (result_idx, result_seq, pdb_content) in zip(processed_sequences, parallel_results):
                logging.info(f"Processing result for {sample_id}: pdb_content is {'None' if pdb_content is None else f'{len(pdb_content)} chars'}")
                if pdb_content is not None:
                    try:
                        with open(pdb_path, 'w') as f:
                            f.write(pdb_content)
                        logging.info(f"Successfully saved PDB to {pdb_path}")
                        
                        # 将新序列添加到序列数据库
                        seq_hash = hashlib.md5(proc_seq.encode()).hexdigest()
                        sequence_database[seq_hash] = {
                            'original_sample_id': sample_id,
                            'pdb_path': str(pdb_path),
                            'sequence_length': len(proc_seq),
                            'created_time': time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        logging.info(f"添加新序列到数据库: {sample_id} (hash: {seq_hash[:12]})")
                        
                        if args.use_sample_manager:
                            sample_manager.update_sample_status(sample_id, "completed", pdb_path=str(pdb_path))
                        
                        meta['status'] = 'ok'
                        if not args.use_sample_manager:
                            meta['output_file'] = os.path.basename(pdb_path)
                        report.append(meta)
                        
                        logging.info(f"[{idx+1}/{len(sequences)}] Completed {sample_id}")
                    except Exception as e:
                        logging.error(f"Failed to save PDB file {pdb_path}: {e}")
                        meta['status'] = 'failed'
                        meta['error'] = f'File save failed: {e}'
                        report.append(meta)
                else:
                    # 处理失败的情况
                    if args.use_sample_manager:
                        sample_manager.log_failure(sample_id, "structure_prediction", "ESMFold prediction failed", len(proc_seq))
                    
                    meta['status'] = 'failed'
                    meta['error'] = 'ESMFold prediction failed'
                    report.append(meta)
                    logging.error(f"[{idx+1}/{len(sequences)}] Failed {sample_id}: ESMFold prediction failed")
        
        except Exception as e:
            logging.error(f"Parallel prediction failed: {e}")
            # 记录所有序列的失败信息
            for idx, proc_seq, sample_id, pdb_path, meta in processed_sequences:
                if args.use_sample_manager:
                    sample_manager.log_failure(sample_id, "structure_prediction", str(e), len(proc_seq))
                
                meta['status'] = 'failed'
                meta['error'] = str(e)
                report.append(meta)
                logging.error(f"[{idx+1}/{len(sequences)}] Failed {sample_id}: {str(e)}")
        
        # CPU offload option
        if args.cpu_offload:
            for gpu_id, model in predictor.models.items():
                model.cpu()
            torch.cuda.empty_cache()

    # 生成报告
    end_time = time.time()
    total_time = end_time - start_time
    
    # 统计信息
    stats = {
        "total_sequences": len(sequences),
        "processed": len([r for r in report if r.get('status') == 'ok']),
        "cached": len([r for r in report if r.get('status') == 'cached']),
        "cached_shared": len([r for r in report if r.get('status') == 'cached_shared']),
        "skipped_too_long": len([r for r in report if r.get('status') == 'skipped_too_long']),
        "failed": len([r for r in report if r.get('status') == 'failed']),
        "total_time_seconds": total_time,
        "avg_time_per_sequence": total_time / len(sequences) if sequences else 0
    }
    
    logging.info(f"✅ Processing complete!")
    logging.info(f"Total sequences: {stats['total_sequences']}")
    logging.info(f"Successfully processed: {stats['processed']}")
    logging.info(f"Cached (skipped): {stats['cached']}")
    logging.info(f"Cached (shared): {stats['cached_shared']}")
    logging.info(f"Skipped (too long): {stats['skipped_too_long']}")
    logging.info(f"Failed: {stats['failed']}")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Average time per sequence: {stats['avg_time_per_sequence']:.2f}s")
    
    # 保存报告
    report_path = os.path.join(output_dir, args.report_json)
    with open(report_path, 'w') as f:
        json.dump({
            "stats": stats,
            "report": report
        }, f, indent=2)
    logging.info(f"Report saved to {report_path}")
    
    # 保存序列数据库
    try:
        with open(SEQUENCE_DB_FILE, 'w') as f:
            json.dump(sequence_database, f, indent=2)
        logging.info(f"序列数据库已保存: {SEQUENCE_DB_FILE} ({len(sequence_database)} 个序列)")
    except Exception as e:
        logging.error(f"保存序列数据库失败: {e}")
    
    # 如果使用SampleManager，生成额外报告
    if args.use_sample_manager:
        # 保存失败摘要
        failures = sample_manager.get_failed_samples_summary()
        if not failures.empty:
            failure_path = os.path.join(output_dir, 'failures_summary.csv')
            failures.to_csv(failure_path, index=False)
            logging.info(f"Failure summary saved to {failure_path}")


if __name__ == '__main__':
    main()
