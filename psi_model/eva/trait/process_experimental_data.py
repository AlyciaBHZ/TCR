#!/usr/bin/env python3
"""
处理实验数据，将V/J区域ID映射到序列，匹配MHC和peptide
确保输出格式与训练数据一致，保留所有实验分数
现在支持所有四种TCR基因类型：TRAV, TRAJ, TRBV, TRBJ
"""

import os
import pandas as pd
import logging
from pathlib import Path
import re
from datetime import datetime
from typing import Dict, Optional

def setup_logging(output_dir):
    """设置日志"""
    log_file = output_dir / f'process_experimental_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file

def load_tcr_sequences():
    """从tcr_seq.csv加载TCR序列数据（使用正确的文件路径）"""
    tcr_file = "../../../data/collected data/final_data/tcr_seq.csv"
    
    if not os.path.exists(tcr_file):
        raise FileNotFoundError(f"找不到TCR序列文件: {tcr_file}")
    
    logging.info(f"加载TCR序列文件: {tcr_file}")
    tcr_df = pd.read_csv(tcr_file)
    
    # 创建基因名称到序列的映射
    gene_to_seq = {}
    for _, row in tcr_df.iterrows():
        gene_name = row['Gene']
        sequence = row['Sequence']
        if pd.notna(sequence) and sequence != 'Not Found':
            gene_to_seq[gene_name] = sequence
    
    logging.info(f"成功加载 {len(gene_to_seq)} 个TCR基因序列")
    
    # 统计每种基因类型的数量
    trav_count = len([g for g in gene_to_seq.keys() if g.startswith('TRAV')])
    traj_count = len([g for g in gene_to_seq.keys() if g.startswith('TRAJ')])
    trbv_count = len([g for g in gene_to_seq.keys() if g.startswith('TRBV')])
    trbj_count = len([g for g in gene_to_seq.keys() if g.startswith('TRBJ')])
    
    logging.info(f"TCR基因统计: TRAV={trav_count}, TRAJ={traj_count}, TRBV={trbv_count}, TRBJ={trbj_count}")
    
    return gene_to_seq

def load_mhc_sequences():
    """从mhc_seq.csv加载MHC序列数据"""
    mhc_file = "../../../data/collected data/final_data/mhc_seq.csv"
    
    if not os.path.exists(mhc_file):
        raise FileNotFoundError(f"找不到MHC序列文件: {mhc_file}")
    
    logging.info(f"加载MHC序列文件: {mhc_file}")
    mhc_df = pd.read_csv(mhc_file)
    
    # 创建MHC名称到序列的映射
    mhc_to_seq = {}
    for _, row in mhc_df.iterrows():
        mhc_name = row['Name']
        sequence = row['Sequence']
        if pd.notna(sequence) and sequence != 'Not Found':
            mhc_to_seq[mhc_name] = sequence
    
    logging.info(f"成功加载 {len(mhc_to_seq)} 个MHC序列")
    return mhc_to_seq

def standardize_gene_name(gene_name: str) -> Optional[str]:
    """
    标准化基因名称 - 基于trn_tst.ipynb的实现
    支持所有四种TCR基因类型：TRAV, TRAJ, TRBV, TRBJ
    """
    if pd.isna(gene_name) or gene_name == '':
        return None
    
    # 移除多余的空格并转换为大写
    gene_name = str(gene_name).strip().upper()
    
    # 处理Beta链V基因
    if gene_name.startswith('TRBV') or gene_name.startswith('TREV'):
        # 处理TREV -> TRBV转换
        if gene_name.startswith('TREV'):
            gene_name = gene_name.replace('TREV', 'TRBV', 1)
        
        # 确保格式为TRBV7-2这样的标准格式
        match = re.match(r'TRBV(\d+)(?:-(\d+))?', gene_name)
        if match:
            family, member = match.groups()
            if member:
                return f'TRBV{family}-{member}'
            return f'TRBV{family}'
    
    # 处理Beta链J基因
    elif gene_name.startswith('TRBJ') or gene_name.startswith('TREJ'):
        # 处理TREJ -> TRBJ转换
        if gene_name.startswith('TREJ'):
            gene_name = gene_name.replace('TREJ', 'TRBJ', 1)
            
        # 确保格式为TRBJ2-1这样的标准格式
        match = re.match(r'TRBJ(\d+)(?:-(\d+))?', gene_name)
        if match:
            family, member = match.groups()
            if member:
                return f'TRBJ{family}-{member}'
            return f'TRBJ{family}'
    
    # 处理Alpha链V基因
    elif gene_name.startswith('TRAV'):
        match = re.match(r'TRAV(\d+)(?:-(\d+))?', gene_name)
        if match:
            family, member = match.groups()
            if member:
                return f'TRAV{family}-{member}'
            return f'TRAV{family}'
            
    # 处理Alpha链J基因
    elif gene_name.startswith('TRAJ'):
        match = re.match(r'TRAJ(\d+)(?:-(\d+))?', gene_name)
        if match:
            family, member = match.groups()
            if member:
                return f'TRAJ{family}-{member}'
            return f'TRAJ{family}'
    
    return gene_name

def get_sequence_for_gene(gene_id: str, gene_to_seq: Dict[str, str], 
                         unmapped_ids: Dict[str, Optional[str]]) -> Optional[str]:
    """
    智能获取基因ID对应的序列 - 基于trn_tst.ipynb的实现
    """
    if pd.isna(gene_id) or gene_id == '':
        return None
        
    gene_id = str(gene_id).strip()
    original_id = gene_id
    
    # 1. 尝试直接匹配
    if gene_id in gene_to_seq:
        return gene_to_seq[gene_id]
    
    # 2. 尝试标准化匹配
    standardized_id = standardize_gene_name(gene_id)
    if standardized_id and standardized_id in gene_to_seq:
        unmapped_ids[original_id] = standardized_id
        return gene_to_seq[standardized_id]
    
    # 3. 尝试移除等位基因信息
    base_id = gene_id.split('*')[0]  # 移除等位基因信息
    if base_id in gene_to_seq:
        unmapped_ids[original_id] = base_id
        return gene_to_seq[base_id]
    
    # 4. 尝试基因家族优先匹配
    base_id_std = standardize_gene_name(base_id) if base_id != gene_id else base_id
    if base_id_std:
        for full_id in gene_to_seq:
            if full_id.startswith(base_id_std):
                unmapped_ids[original_id] = full_id
                return gene_to_seq[full_id]
    
    # 5. 尝试宽松的家族匹配
    for gene_type in ['TRBV', 'TRBJ', 'TRAV', 'TRAJ']:
        if gene_id.startswith(gene_type):
            # 提取数字部分
            match = re.search(r'(\d+)', gene_id)
            if match:
                family_num = match.group(1)
                pattern = f'{gene_type}{family_num}'
                for full_id in gene_to_seq:
                    if full_id.startswith(pattern):
                        unmapped_ids[original_id] = full_id
                        return gene_to_seq[full_id]
    
    # 记录完全无法映射的ID
    unmapped_ids[original_id] = None
    return None

def map_mhc_to_sequence(mhc_id, mhc_sequences):
    """将MHC ID映射到序列，使用改进的匹配策略"""
    # 直接匹配
    if mhc_id in mhc_sequences:
        return mhc_sequences[mhc_id]
    
    # 尝试添加HLA-前缀
    hla_format = f"HLA-{mhc_id}"
    if hla_format in mhc_sequences:
        return mhc_sequences[hla_format]
    
    # 尝试不同的格式变体
    variants_to_try = [
        mhc_id,
        f"HLA-{mhc_id}",
        mhc_id.replace("*", ""),  # 移除*
        f"HLA-{mhc_id.replace('*', '')}",  # HLA-前缀且移除*
    ]
    
    # 对于A*02:01这样的格式，也尝试A*02等简化版本
    if ":" in mhc_id:
        base_allele = mhc_id.split(":")[0]  # A*02:01 -> A*02
        variants_to_try.extend([
            base_allele,
            f"HLA-{base_allele}",
            base_allele.replace("*", ""),
            f"HLA-{base_allele.replace('*', '')}",
        ])
    
    # 尝试所有变体
    for variant in variants_to_try:
        if variant in mhc_sequences:
            return mhc_sequences[variant]
    
    # 尝试模糊匹配（查找包含关键部分的序列）
    key_part = mhc_id.replace("*", "").replace(":", "")  # A0201
    for seq_name in mhc_sequences.keys():
        if key_part in seq_name.replace("*", "").replace(":", "").replace("-", ""):
            return mhc_sequences[seq_name]
    
    return None

def process_experimental_data(exp_data_file, output_dir):
    """处理实验数据文件"""
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_file = setup_logging(output_dir)
    logging.info(f"开始处理实验数据: {exp_data_file}")
    
    try:
        # 加载序列映射
        gene_to_seq = load_tcr_sequences()
        mhc_to_seq = load_mhc_sequences()
        
        # 读取实验数据
        logging.info(f"读取实验数据文件: {exp_data_file}")
        exp_df = pd.read_csv(exp_data_file)
        logging.info(f"实验数据包含 {len(exp_df)} 条记录")
        logging.info(f"实验数据列名: {exp_df.columns.tolist()}")
        
        # 检查需要的列是否存在
        required_cols = ['cdr3_beta_aa', 'TRAV', 'TRAJ', 'TRBV', 'TRBJ', 'epitope_aa', 'hla_long']
        missing_cols = [col for col in required_cols if col not in exp_df.columns]
        if missing_cols:
            raise ValueError(f"实验数据缺少必要的列: {missing_cols}")
        
        # 记录无法映射的ID
        unmapped_ids = {}
        unmapped_mhc = {}
        
        # 映射统计
        mapping_stats = {
            'total_processed': 0,
            'alpha_mapped': {'trav': 0, 'traj': 0},
            'beta_mapped': {'trbv': 0, 'trbj': 0},
            'alpha_unmapped': {'trav': 0, 'traj': 0},
            'beta_unmapped': {'trbv': 0, 'trbj': 0},
            'mhc_mapped': 0,
            'mhc_unmapped': 0
        }
        
        # 处理数据
        processed_data = []
        for idx, row in exp_df.iterrows():
            mapping_stats['total_processed'] += 1
            
            # 获取CDR3β序列
            cdr3_b = row['cdr3_beta_aa']
            if pd.isna(cdr3_b):
                continue
            
            # 获取CDR3α序列（如果有的话）
            cdr3_a = row.get('cdr3_alpha_aa', '')
            
            # 获取表位序列
            peptide = row['epitope_aa']
            if pd.isna(peptide):
                continue
            
            # 获取所有四种基因ID
            trav_id = row['TRAV']
            traj_id = row['TRAJ']
            trbv_id = row['TRBV']
            trbj_id = row['TRBJ']
            
            # 获取MHC
            hla_long = row['hla_long']
            
            # 标准化基因名称并获取序列
            trav_std = standardize_gene_name(trav_id)
            traj_std = standardize_gene_name(traj_id)
            trbv_std = standardize_gene_name(trbv_id)
            trbj_std = standardize_gene_name(trbj_id)
            
            trav_seq = get_sequence_for_gene(trav_std, gene_to_seq, unmapped_ids)
            traj_seq = get_sequence_for_gene(traj_std, gene_to_seq, unmapped_ids)
            trbv_seq = get_sequence_for_gene(trbv_std, gene_to_seq, unmapped_ids)
            trbj_seq = get_sequence_for_gene(trbj_std, gene_to_seq, unmapped_ids)
            
            # 更新映射统计
            if trav_seq is not None:
                mapping_stats['alpha_mapped']['trav'] += 1
            else:
                mapping_stats['alpha_unmapped']['trav'] += 1
                
            if traj_seq is not None:
                mapping_stats['alpha_mapped']['traj'] += 1
            else:
                mapping_stats['alpha_unmapped']['traj'] += 1
                
            if trbv_seq is not None:
                mapping_stats['beta_mapped']['trbv'] += 1
            else:
                mapping_stats['beta_unmapped']['trbv'] += 1
                
            if trbj_seq is not None:
                mapping_stats['beta_mapped']['trbj'] += 1
            else:
                mapping_stats['beta_unmapped']['trbj'] += 1
            
            # 获取MHC序列
            mhc_seq = map_mhc_to_sequence(hla_long, mhc_to_seq)
            if mhc_seq is not None:
                mapping_stats['mhc_mapped'] += 1
            else:
                mapping_stats['mhc_unmapped'] += 1
            
            # 只保留成功映射beta链TCR的记录（alpha链可选，MHC可选）
            if trbv_seq is not None and trbj_seq is not None:
                record = {
                    # 训练数据格式的核心字段
                    'peptide': peptide.strip(),
                    'mhc': mhc_seq if mhc_seq is not None else '',
                    'l_v': trav_seq if trav_seq is not None else '',  # Alpha链V区域
                    'l_j': traj_seq if traj_seq is not None else '',  # Alpha链J区域
                    'h_v': trbv_seq,  # Beta链V区域
                    'h_j': trbj_seq,  # Beta链J区域
                    'cdr3_b': cdr3_b.strip(),
                    'cdr3_a': cdr3_a.strip() if pd.notna(cdr3_a) else '',  # 添加CDR3α信息
                    
                    # 保留所有实验分数和元信息
                    'hla_short': row.get('hla_short', ''),
                    'hla_long': hla_long,
                    'log2FoldChange': row.get('log2FoldChange', None),
                    'lfcSE': row.get('lfcSE', None),
                    'stat': row.get('stat', None),
                    'pvalue': row.get('pvalue', None),
                    'padj': row.get('padj', None),
                    'baseMean': row.get('baseMean', None),
                    'vdjdb_score': row.get('vdjdb_score', None),
                    'data_origin': row.get('data_origin', ''),
                    
                    # 原始基因ID信息
                    'original_trav': trav_id,
                    'original_traj': traj_id,
                    'original_trbv': trbv_id,
                    'original_trbj': trbj_id,
                    'name': row.get('name', '')
                }
                processed_data.append(record)
        
        # 创建处理后的DataFrame
        result_df = pd.DataFrame(processed_data)
        
        # 去除重复项（基于核心字段）
        initial_len = len(result_df)
        result_df = result_df.drop_duplicates(subset=['peptide', 'mhc', 'l_v', 'l_j', 'h_v', 'h_j', 'cdr3_b'])
        duplicates_removed = initial_len - len(result_df)
        
        logging.info(f"成功处理 {len(result_df)} 条记录")
        logging.info(f"移除重复记录 {duplicates_removed} 条")
        
        # 打印映射统计
        logging.info("=== TCR基因映射统计 ===")
        for gene_type in ['trav', 'traj']:
            mapped = mapping_stats['alpha_mapped'][gene_type]
            unmapped = mapping_stats['alpha_unmapped'][gene_type]
            total = mapped + unmapped
            success_rate = (mapped / total * 100) if total > 0 else 0
            logging.info(f"{gene_type.upper()}: {mapped}/{total} ({success_rate:.1f}%)")
        
        for gene_type in ['trbv', 'trbj']:
            mapped = mapping_stats['beta_mapped'][gene_type]
            unmapped = mapping_stats['beta_unmapped'][gene_type]
            total = mapped + unmapped
            success_rate = (mapped / total * 100) if total > 0 else 0
            logging.info(f"{gene_type.upper()}: {mapped}/{total} ({success_rate:.1f}%)")
        
        mhc_total = mapping_stats['mhc_mapped'] + mapping_stats['mhc_unmapped']
        mhc_success_rate = (mapping_stats['mhc_mapped'] / mhc_total * 100) if mhc_total > 0 else 0
        logging.info(f"MHC: {mapping_stats['mhc_mapped']}/{mhc_total} ({mhc_success_rate:.1f}%)")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存训练格式数据（只包含训练需要的字段）
        training_format_file = output_dir / f"experimental_data_training_format_{timestamp}.csv"
        training_cols = ['peptide', 'mhc', 'l_v', 'l_j', 'h_v', 'h_j', 'cdr3_b']
        training_df = result_df[training_cols].copy()
        training_df.to_csv(training_format_file, index=False)
        logging.info(f"训练格式数据保存到: {training_format_file}")
        
        # 保存完整信息的数据（包含所有实验分数）
        full_data_file = output_dir / f"experimental_data_with_scores_{timestamp}.csv"
        result_df.to_csv(full_data_file, index=False)
        logging.info(f"完整数据（含分数）保存到: {full_data_file}")
        
        # 保存统计信息
        stats_file = output_dir / f"processing_stats_{timestamp}.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("实验数据处理统计\n")
            f.write("================\n\n")
            f.write(f"原始记录数: {len(exp_df):,}\n")
            f.write(f"成功处理记录数: {len(result_df):,}\n")
            f.write(f"处理成功率: {len(result_df)/len(exp_df)*100:.2f}%\n")
            f.write(f"移除重复记录数: {duplicates_removed:,}\n")
            f.write(f"唯一CDR3β序列数: {result_df['cdr3_b'].nunique():,}\n")
            f.write(f"唯一CDR3α序列数: {result_df[result_df['cdr3_a'] != '']['cdr3_a'].nunique():,}\n")
            f.write(f"唯一表位数: {result_df['peptide'].nunique():,}\n")
            f.write(f"唯一MHC序列数: {result_df[result_df['mhc'] != '']['mhc'].nunique():,}\n")
            f.write(f"唯一TRAV序列数: {result_df[result_df['l_v'] != '']['l_v'].nunique():,}\n")
            f.write(f"唯一TRAJ序列数: {result_df[result_df['l_j'] != '']['l_j'].nunique():,}\n")
            f.write(f"唯一TRBV序列数: {result_df['h_v'].nunique():,}\n")
            f.write(f"唯一TRBJ序列数: {result_df['h_j'].nunique():,}\n")
            f.write(f"唯一HLA类型数: {result_df['hla_short'].nunique():,}\n")
            f.write(f"有log2FoldChange分数的记录数: {result_df['log2FoldChange'].notna().sum():,}\n")
            f.write(f"有p值的记录数: {result_df['pvalue'].notna().sum():,}\n")
            f.write(f"有VDJdb分数的记录数: {result_df['vdjdb_score'].notna().sum():,}\n")
            
            f.write(f"\n=== TCR基因映射详细统计 ===\n")
            for gene_type in ['trav', 'traj']:
                mapped = mapping_stats['alpha_mapped'][gene_type]
                unmapped = mapping_stats['alpha_unmapped'][gene_type]
                total = mapped + unmapped
                success_rate = (mapped / total * 100) if total > 0 else 0
                f.write(f"{gene_type.upper()}: {mapped}/{total} ({success_rate:.1f}%)\n")
            
            for gene_type in ['trbv', 'trbj']:
                mapped = mapping_stats['beta_mapped'][gene_type]
                unmapped = mapping_stats['beta_unmapped'][gene_type]
                total = mapped + unmapped
                success_rate = (mapped / total * 100) if total > 0 else 0
                f.write(f"{gene_type.upper()}: {mapped}/{total} ({success_rate:.1f}%)\n")
            
            mhc_total = mapping_stats['mhc_mapped'] + mapping_stats['mhc_unmapped']
            mhc_success_rate = (mapping_stats['mhc_mapped'] / mhc_total * 100) if mhc_total > 0 else 0
            f.write(f"MHC: {mapping_stats['mhc_mapped']}/{mhc_total} ({mhc_success_rate:.1f}%)\n")
        
        logging.info(f"统计信息保存到: {stats_file}")
        
        # 保存未映射ID信息
        unmapped_file = output_dir / f"unmapped_ids_{timestamp}.txt"
        with open(unmapped_file, 'w', encoding='utf-8') as f:
            f.write("未能直接映射的基因ID和MHC\n")
            f.write("============================\n\n")
            
            f.write("TCR基因 - 完全无法映射的ID:\n")
            unmapped_count = 0
            for gene_id, mapped_id in sorted(unmapped_ids.items()):
                if mapped_id is None:
                    f.write(f"  - {gene_id}\n")
                    unmapped_count += 1
            f.write(f"\n总计: {unmapped_count} 个\n\n")
            
            f.write("TCR基因 - 使用替代映射的ID:\n")
            substituted_count = 0
            for gene_id, mapped_id in sorted(unmapped_ids.items()):
                if mapped_id is not None:
                    f.write(f"  - {gene_id} -> {mapped_id}\n")
                    substituted_count += 1
            f.write(f"\n总计: {substituted_count} 个\n\n")
            
            f.write("MHC - 完全无法映射的ID:\n")
            mhc_unmapped_count = 0
            for mhc_id, mapped_id in sorted(unmapped_mhc.items()):
                if mapped_id is None:
                    f.write(f"  - {mhc_id}\n")
                    mhc_unmapped_count += 1
            f.write(f"\n总计: {mhc_unmapped_count} 个\n\n")
            
            f.write("MHC - 使用替代映射的ID:\n")
            mhc_substituted_count = 0
            for mhc_id, mapped_id in sorted(unmapped_mhc.items()):
                if mapped_id is not None:
                    f.write(f"  - {mhc_id} -> {mapped_id}\n")
                    mhc_substituted_count += 1
            f.write(f"\n总计: {mhc_substituted_count} 个\n")
        
        logging.info(f"未映射ID信息保存到: {unmapped_file}")
        
        return {
            'training_format_file': training_format_file,
            'full_data_file': full_data_file,
            'stats_file': stats_file,
            'unmapped_file': unmapped_file,
            'log_file': log_file,
            'processed_count': len(result_df),
            'success_rate': len(result_df)/len(exp_df)*100,
            'mhc_mapped_rate': mapping_stats['mhc_mapped']/mhc_total*100 if mhc_total > 0 else 0,
            'mapping_stats': mapping_stats
        }
        
    except Exception as e:
        logging.error(f"处理过程中出错: {str(e)}")
        raise

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='处理实验数据，将所有四种TCR基因ID和MHC映射为序列')
    parser.add_argument('--exp_data', type=str, required=True,
                       help='实验数据文件路径')
    parser.add_argument('--output_dir', type=str, default='processed_experimental_data',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 处理数据
    result = process_experimental_data(args.exp_data, args.output_dir)
    
    print(f"\n处理完成!")
    print(f"训练格式数据: {result['training_format_file']}")
    print(f"完整数据（含分数）: {result['full_data_file']}")
    print(f"TCR映射成功率: {result['success_rate']:.2f}%")
    print(f"MHC映射成功率: {result['mhc_mapped_rate']:.2f}%")
    
    # 打印详细映射统计
    stats = result['mapping_stats']
    print(f"\n=== 详细映射统计 ===")
    for gene_type in ['trav', 'traj']:
        mapped = stats['alpha_mapped'][gene_type]
        unmapped = stats['alpha_unmapped'][gene_type]
        total = mapped + unmapped
        success_rate = (mapped / total * 100) if total > 0 else 0
        print(f"{gene_type.upper()}: {mapped}/{total} ({success_rate:.1f}%)")
    
    for gene_type in ['trbv', 'trbj']:
        mapped = stats['beta_mapped'][gene_type]
        unmapped = stats['beta_unmapped'][gene_type]
        total = mapped + unmapped
        success_rate = (mapped / total * 100) if total > 0 else 0
        print(f"{gene_type.upper()}: {mapped}/{total} ({success_rate:.1f}%)")
    
    print(f"日志文件: {result['log_file']}")

if __name__ == '__main__':
    main() 