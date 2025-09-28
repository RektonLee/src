#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Range prediction script - supports specifying sample range for prediction
Used for parallel task allocation
"""

import torch
import pandas as pd
import numpy as np
from data_loader import ProteinStructureProcessor, MoleculeEmbeddingGenerator
from GNN_model import PocketGNNWithAttention, PocketGNNWithAttentionNoTemp
from sample_manager import SampleManager
import logging
import os
from graph_builder_rbf import build_graph, parse_pocket, DIST_CUTOFF, RBF_CENTERS, RBF_DMIN, RBF_DMAX, RBF_GAMMA
from docking import run_preprocess
import hashlib
import argparse

# Get correct dimensions from graph_builder_rbf.py
NODE_INPUT_DIM = 52
EDGE_INPUT_DIM = RBF_CENTERS

def predict_kinetics_range(input_data, model_path, output_dir='predictions', 
                          start_idx=0, end_idx=None, temperature=303.15, 
                          use_sample_manager=True, sample_data_dir='sample_data'):
    """
    Predict enzyme kinetics parameters for specified range
    
    Parameters:
    input_data: CSV file path or DataFrame
    model_path: trained model path
    output_dir: output directory
    start_idx: start index
    end_idx: end index (None means to the end)
    temperature: temperature
    use_sample_manager: whether to use SampleManager
    sample_data_dir: SampleManager base directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, 'prediction.log')
    
    # Clear existing logging configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Reconfigure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logging.info(f"Starting range prediction [{start_idx}:{end_idx}], log file: {log_file}")
    
    # Parse input data
    if isinstance(input_data, str):
        input_df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data
    else:
        raise ValueError("input_data must be CSV path or DataFrame")
    
    # Apply range filter
    if end_idx is None:
        end_idx = len(input_df)
    
    input_df = input_df.iloc[start_idx:end_idx].copy()
    logging.info(f"Processing sample range: {start_idx}-{end_idx} (total {len(input_df)} samples)")
    
    # Check required columns
    required_columns = ['sequence', 'smiles']
    if use_sample_manager:
        required_columns.append('sample_id')
    
    missing_columns = [col for col in required_columns if col not in input_df.columns]
    if missing_columns:
        raise ValueError(f"Input data missing required columns: {missing_columns}")
    
    # Compatible with original data format
    if 'sample_id' not in input_df.columns and not use_sample_manager:
        input_df['sample_id'] = [f"sample_{start_idx + i + 1}" for i in range(len(input_df))]
    
    # Initialize SampleManager (if enabled)
    sample_manager = None
    if use_sample_manager:
        sample_manager = SampleManager(sample_data_dir)
        logging.info(f"SampleManager initialized, managing {len(sample_manager.sample_registry)} samples")
        
        # Register samples from input data if not already registered
        if isinstance(input_data, str):
            sample_manager.register_samples_from_csv(input_data, temperature)
            logging.info(f"Registered samples from {input_data}")
        else:
            # Register from DataFrame
            temp_csv = os.path.join(output_dir, 'temp_input.csv')
            input_df.to_csv(temp_csv, index=False)
            sample_manager.register_samples_from_csv(temp_csv, temperature)
            os.remove(temp_csv)  # Clean up temp file
            logging.info(f"Registered {len(input_df)} samples from DataFrame")
    
    # Initialize processor
    structure_processor = ProteinStructureProcessor(sample_manager=sample_manager)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 首先检查模型是否包含温度模块
    try:
        state_dict = torch.load(model_path, map_location=device)
        has_temp_module = any(key.startswith('temp_mlp') for key in state_dict.keys())
        logging.info(f"模型包含温度模块: {has_temp_module}")
    except Exception as e:
        logging.error(f"❌ 无法加载模型状态字典: {str(e)}")
        raise
    
    # 根据模型类型创建相应的模型
    if has_temp_module:
        model = PocketGNNWithAttention(
            node_input_dim=NODE_INPUT_DIM,
            edge_input_dim=EDGE_INPUT_DIM,
            hidden_dim=256,
            num_layers=6,
            heads=8,
            dropout=0.1
        ).to(device)
        logging.info("✅ 创建包含温度模块的模型")
    else:
        model = PocketGNNWithAttentionNoTemp(
            node_input_dim=NODE_INPUT_DIM,
            edge_input_dim=EDGE_INPUT_DIM,
            hidden_dim=256,
            num_layers=6,
            heads=8,
            dropout=0.1
        ).to(device)
        logging.info("✅ 创建不包含温度模块的模型")
    
    try:
        model.load_state_dict(state_dict)
        logging.info(f"✅ 成功加载模型: {model_path}")
    except Exception as e:
        logging.error(f"❌ 加载模型失败: {str(e)}")
        raise
    
    model.eval()
    
    results = []
    total_samples = len(input_df)
    
    for i, row in input_df.iterrows():
        sample_id = row.get('sample_id', f'sample_{start_idx + i + 1}')
        seq = row['sequence']
        smiles = row['smiles']
        uniprot_id = row.get('uniprot', None)
        experimental_km_log10 = row.get('experimental value[log10]', None)  # Note column name
        
        try:
            logging.info(f"Processing sample {i+1}/{total_samples}: {sample_id}")
            
            # 1. Get protein structure
            if use_sample_manager and sample_manager:
                protein_path, is_shared = sample_manager.get_protein_path(sample_id)
                
                if protein_path.exists() and protein_path.stat().st_size > 0:
                    pdb_path = str(protein_path)
                    if is_shared:
                        logging.info(f"Using shared PDB file: {pdb_path}")
                    else:
                        logging.info(f"Using existing PDB file: {pdb_path}")
                else:
                    # 检查是否是符号链接
                    if protein_path.is_symlink():
                        # 解析符号链接
                        real_path = protein_path.resolve()
                        if real_path.exists() and real_path.stat().st_size > 0:
                            pdb_path = str(real_path)
                            logging.info(f"Using shared PDB file via symlink: {pdb_path}")
                        else:
                            logging.warning(f"Skip sample {sample_id}: Shared PDB file not found ({real_path})")
                            if use_sample_manager and sample_manager:
                                sample_manager.log_failure(sample_id, "missing_shared_pdb", f"Shared PDB file not found: {real_path}", len(seq))
                            
                            results.append({
                                'sample_id': sample_id,
                                'sequence': seq,
                                'smiles': smiles,
                                'kcat_pred': None,
                                'km_pred': None,
                                'km_pred_log10': None,
                                'experimental_km_log10': experimental_km_log10,
                                'km_error_log10': None,
                                'km_error_relative': None,
                                'temperature': temperature,
                                'error': f'Shared PDB file not found: {real_path}'
                            })
                            continue
                    else:
                        logging.warning(f"Skip sample {sample_id}: PDB file not found ({protein_path})")
                        if use_sample_manager and sample_manager:
                            sample_manager.log_failure(sample_id, "missing_pdb", f"PDB file not found: {protein_path}", len(seq))
                        
                        results.append({
                            'sample_id': sample_id,
                            'sequence': seq,
                            'smiles': smiles,
                            'kcat_pred': None,
                            'km_pred': None,
                            'km_pred_log10': None,
                            'experimental_km_log10': experimental_km_log10,
                            'km_error_log10': None,
                            'km_error_relative': None,
                            'temperature': temperature,
                            'error': f'PDB file not found: {protein_path}'
                        })
                        continue
            else:
                pdb_content = structure_processor.predict_structure(seq, uniprot_id)
                temp_pdb_dir = os.path.join(output_dir, 'temp_pdbs')
                os.makedirs(temp_pdb_dir, exist_ok=True)
                pdb_path = os.path.join(temp_pdb_dir, f"{sample_id}.pdb")
                with open(pdb_path, 'w') as f:
                    f.write(pdb_content)
                logging.info(f"Created temporary PDB file: {pdb_path}")
            
            logging.info("Protein structure prediction completed")

            # 2. Docking step - 检查是否已有pocket文件
            if use_sample_manager:
                pocket_dir = os.path.join(sample_data_dir, "samples", sample_id)
            else:
                pocket_dir = output_dir
            os.makedirs(pocket_dir, exist_ok=True)
            
            pocket_hash = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) & 0xffff
            base_name = f"{uniprot_id or sample_id}_{pocket_hash}"
            pocket_pdb = os.path.join(pocket_dir, f"{base_name}_10A.pdb")
            
            # 检查pocket文件是否已存在
            if os.path.exists(pocket_pdb) and os.path.getsize(pocket_pdb) > 0:
                logging.info(f"✅ 使用已存在的pocket文件: {pocket_pdb}")
            else:
                try:
                    logging.info(f"🔄 开始docking处理: {sample_id}")
                    result = run_preprocess(uniprot_id or sample_id, smiles, pdb_path, pocket_pdb, i)
                    if not result:
                        raise RuntimeError("Docking preprocessing failed")
                    
                    if not os.path.exists(pocket_pdb):
                        raise FileNotFoundError(f"Pocket file not generated: {pocket_pdb}")
                    
                    logging.info(f"✅ Docking完成，pocket文件: {pocket_pdb}")
                except Exception as e:
                    logging.error(f"❌ Docking失败: {str(e)}")
                    raise

            # 3. Extract pocket atoms
            atoms = parse_pocket(pocket_pdb)
            logging.info(f"Pocket atom extraction completed, {len(atoms)} atoms")
            
            if len(atoms) == 0:
                raise ValueError("No valid atoms in pocket file, cannot predict")

            # 4. Generate graph data
            graph_data = build_graph(atoms, temperature)
            logging.info("Graph data construction completed")

            # 5. Prediction
            with torch.no_grad():
                graph_data = graph_data.to(device)
                
                # 如果模型不包含温度模块，移除温度信息
                if not has_temp_module and hasattr(graph_data, 'temperature'):
                    delattr(graph_data, 'temperature')
                    logging.info("✅ 移除温度信息以匹配模型")
                
                predictions = model(graph_data)

            if predictions is None or predictions.numel() == 0:
                raise ValueError("Model returned empty prediction results")
            
            if predictions.shape[0] == 0:
                raise ValueError("Model returned prediction results with 0 size in dimension 0")
            
            if predictions.shape[1] < 2:
                raise ValueError(f"Model returned prediction results with insufficient dimensions, expected at least 2, got {predictions.shape[1]}")

            # 6. Convert back to original scale
            kcat_pred = 10 ** predictions[0][0].item()
            km_pred = 10 ** predictions[0][1].item()

            result = {
                'sample_id': sample_id,
                'sequence': seq,
                'smiles': smiles,
                'kcat_pred': kcat_pred,
                'km_pred': km_pred,
                'km_pred_log10': np.log10(km_pred),
                'experimental_km_log10': experimental_km_log10,
                'temperature': temperature
            }
            
            # Calculate errors if experimental values exist
            if experimental_km_log10 is not None:
                result['km_error_log10'] = abs(np.log10(km_pred) - experimental_km_log10)
                result['km_error_relative'] = abs(km_pred - 10**experimental_km_log10) / (10**experimental_km_log10)

            # Add extra info if using SampleManager
            if use_sample_manager and sample_manager:
                sample_info = sample_manager.sample_registry.get(sample_id)
                if sample_info:
                    result['protein_hash'] = sample_info.protein_hash
                    result['pdb_path'] = sample_info.pdb_path

            results.append(result)
            
            # Fix Km display format
            if km_pred < 0.01:
                km_display = f"{km_pred:.2e}"
            else:
                km_display = f"{km_pred:.4f}"
            
            logging.info(f"Prediction completed - kcat: {kcat_pred:.2f}, Km: {km_display}")

        except Exception as e:
            logging.error(f"Error processing sample {sample_id}: {str(e)}")
            if use_sample_manager and sample_manager:
                sample_manager.log_failure(sample_id, "prediction", str(e), len(seq))

            results.append({
                'sample_id': sample_id,
                'sequence': seq,
                'smiles': smiles,
                'kcat_pred': None,
                'km_pred': None,
                'km_pred_log10': None,
                'experimental_km_log10': experimental_km_log10,
                'km_error_log10': None,
                'km_error_relative': None,
                'temperature': temperature,
                'error': str(e)
            })
        
        finally:
            # Clean up temporary PDB files
            try:
                if 'pdb_path' in locals() and os.path.exists(pdb_path):
                    if 'temp_pdbs' in pdb_path:
                        os.remove(pdb_path)
                        logging.debug(f"Cleaned up temporary PDB file: {pdb_path}")
            except Exception as e:
                logging.debug(f"Warning during temporary file cleanup: {e}")
    
    # Clean up temporary directory
    try:
        temp_pdb_dir = os.path.join(output_dir, 'temp_pdbs')
        if os.path.exists(temp_pdb_dir):
            import shutil
            shutil.rmtree(temp_pdb_dir)
            logging.info("Temporary PDB files cleaned up")
    except Exception as e:
        logging.warning(f"Warning during temporary file cleanup: {e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    logging.info(f"Prediction results saved to: {os.path.join(output_dir, 'predictions.csv')}")
    
    # Generate statistics
    valid_predictions = results_df.dropna(subset=['kcat_pred', 'km_pred'])
    failed_predictions = results_df[results_df['kcat_pred'].isna() | results_df['km_pred'].isna()]
    
    # Save successful predictions
    if len(valid_predictions) > 0:
        success_columns = ['sample_id', 'sequence', 'smiles', 'kcat_pred', 'km_pred', 'km_pred_log10', 'experimental_km_log10', 'km_error_log10', 'km_error_relative', 'temperature']
        available_columns = [col for col in success_columns if col in valid_predictions.columns]
        success_df = valid_predictions[available_columns].copy()
        success_df.to_csv(os.path.join(output_dir, 'successful_predictions.csv'), index=False)
        logging.info(f"Successful predictions saved: {len(success_df)} samples")
    
    # Save failed samples
    if len(failed_predictions) > 0:
        failed_df = failed_predictions[['sample_id', 'sequence', 'smiles', 'error']].copy()
        failed_df.to_csv(os.path.join(output_dir, 'failed_predictions.csv'), index=False)
        logging.info(f"Failed samples saved: {len(failed_df)} samples")
    
    # Generate statistics
    if len(valid_predictions) > 0:
        stats = {
            'Sample Range': f"{start_idx}-{end_idx}",
            'Total Samples': len(results_df),
            'Successful Predictions': len(valid_predictions),
            'Failed/Skipped': len(failed_predictions),
            'Success Rate': f"{len(valid_predictions)/len(results_df)*100:.1f}%",
            'kcat Prediction Range': f"{valid_predictions['kcat_pred'].min():.2f} - {valid_predictions['kcat_pred'].max():.2f}",
            'Km Prediction Range': f"{valid_predictions['km_pred'].min():.2e} - {valid_predictions['km_pred'].max():.2e}"
        }
        pd.DataFrame([stats]).to_csv(os.path.join(output_dir, 'stats.csv'), index=False)
        logging.info("Statistics report generated")
    
    logging.info("Range prediction task completed!")
    return results_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Range prediction script")
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--model', type=str, 
                       default="/home/lizihao/Work/enzyme_prediction/src/simple2/outputs/nopqr_attention_rbf/best_model.pt",
                       help='Model file path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, help='End index (exclusive)')
    parser.add_argument('--temperature', type=float, default=303.15, help='Temperature (K)')
    parser.add_argument('--use-sample-manager', action='store_true', help='Use SampleManager')
    parser.add_argument('--sample-data-dir', type=str, default='sample_data', help='SampleManager base directory')
    
    args = parser.parse_args()
    
    # Run range prediction
    results = predict_kinetics_range(
        input_data=args.input,
        model_path=args.model,
        output_dir=args.output,
        start_idx=args.start,
        end_idx=args.end,
        temperature=args.temperature,
        use_sample_manager=args.use_sample_manager,
        sample_data_dir=args.sample_data_dir
    )
    
    print(f"Range prediction completed [{args.start}:{args.end}], results saved to: {args.output}")
    
    # Print brief statistics
    valid_count = len(results.dropna(subset=['kcat_pred', 'km_pred']))
    total_count = len(results)
    failed_count = total_count - valid_count
    print(f"Prediction results: {valid_count}/{total_count} samples successful, {failed_count} skipped/failed")
