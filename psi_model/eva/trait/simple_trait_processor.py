#!/usr/bin/env python3
"""
简化版trait绑定数据处理器，只使用Python内置库
专门用于处理具有明确绑定标签的数据集
"""

import csv
import os
import re
from datetime import datetime

def load_tcr_sequences():
    """从tcr_seq.csv加载TCR序列数据"""
    gene_to_seq = {}
    
    if not os.path.exists('tcr_seq.csv'):
        print("Error: 找不到TCR序列文件: tcr_seq.csv")
        return gene_to_seq
    
    print("加载TCR序列文件: tcr_seq.csv")
    
    with open('tcr_seq.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gene_name = row['Gene']
            sequence = row['Sequence']
            if sequence and sequence != 'Not Found':
                gene_to_seq[gene_name] = sequence
    
    print(f"成功加载 {len(gene_to_seq)} 个TCR基因序列")
    return gene_to_seq

def load_mhc_sequences():
    """从mhc_seq.csv加载MHC序列数据"""
    mhc_to_seq = {}
    
    if not os.path.exists('mhc_seq.csv'):
        print("Error: 找不到MHC序列文件: mhc_seq.csv")
        return mhc_to_seq
    
    print("加载MHC序列文件: mhc_seq.csv")
    
    with open('mhc_seq.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mhc_name = row['Name']
            sequence = row['Sequence']
            if sequence and sequence != 'Not Found':
                mhc_to_seq[mhc_name] = sequence
    
    print(f"成功加载 {len(mhc_to_seq)} 个MHC序列")
    return mhc_to_seq

def standardize_gene_name(gene_name):
    """标准化基因名称"""
    if not gene_name or gene_name == '':
        return None
    
    gene_name = str(gene_name).strip().upper()
    
    # 处理Beta链V基因
    if gene_name.startswith('TRBV') or gene_name.startswith('TREV'):
        if gene_name.startswith('TREV'):
            gene_name = gene_name.replace('TREV', 'TRBV', 1)
        
        match = re.match(r'TRBV(\d+)(?:-(\d+))?', gene_name)
        if match:
            family, member = match.groups()
            if member:
                return f'TRBV{family}-{member}'
            return f'TRBV{family}'
    
    # 处理Beta链J基因
    elif gene_name.startswith('TRBJ') or gene_name.startswith('TREJ'):
        if gene_name.startswith('TREJ'):
            gene_name = gene_name.replace('TREJ', 'TRBJ', 1)
            
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

def get_sequence_for_gene(gene_id, gene_to_seq):
    """获取基因ID对应的序列"""
    if not gene_id or gene_id == '':
        return None
        
    gene_id = str(gene_id).strip()
    
    # 1. 尝试直接匹配
    if gene_id in gene_to_seq:
        return gene_to_seq[gene_id]
    
    # 2. 尝试标准化匹配
    standardized_id = standardize_gene_name(gene_id)
    if standardized_id and standardized_id in gene_to_seq:
        return gene_to_seq[standardized_id]
    
    # 3. 尝试移除等位基因信息
    base_id = gene_id.split('*')[0]
    if base_id in gene_to_seq:
        return gene_to_seq[base_id]
    
    # 4. 尝试基因家族优先匹配
    base_id_std = standardize_gene_name(base_id) if base_id != gene_id else base_id
    if base_id_std:
        for full_id in gene_to_seq:
            if full_id.startswith(base_id_std):
                return gene_to_seq[full_id]
    
    return None

def extract_peptide_from_pmhc(pmhc_string):
    """从pMHC字符串中提取peptide"""
    parts = pmhc_string.split('_')
    if len(parts) >= 2:
        return parts[1]  # ELAGIGILTV
    return None

def extract_mhc_from_pmhc(pmhc_string):
    """从pMHC字符串中提取MHC"""
    parts = pmhc_string.split('_')
    if len(parts) >= 1:
        mhc_part = parts[0]  # A0201
        if len(mhc_part) >= 5 and mhc_part.startswith(('A', 'B', 'C')):
            allele_family = mhc_part[0]
            allele_group = mhc_part[1:3]
            allele_protein = mhc_part[3:5]
            return f"HLA-{allele_family}*{allele_group}:{allele_protein}"
    return None

def map_mhc_to_sequence(mhc_id, mhc_sequences):
    """将MHC ID映射到序列"""
    if not mhc_id:
        return None
        
    # 直接匹配
    if mhc_id in mhc_sequences:
        return mhc_sequences[mhc_id]
    
    # 尝试不同的格式变体
    variants_to_try = [
        mhc_id,
        f"HLA-{mhc_id}",
        mhc_id.replace("HLA-", ""),
        mhc_id.replace("*", ""),
        f"HLA-{mhc_id.replace('*', '')}",
    ]
    
    # 对于A*02:01这样的格式，也尝试A*02等简化版本
    if ":" in mhc_id:
        base_allele = mhc_id.split(":")[0]
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
    
    return None

def process_file(filename, gene_to_seq, mhc_to_seq):
    """处理单个文件"""
    print(f"处理文件: {filename}")
    
    processed_data = []
    total_rows = 0
    successful_rows = 0
    
    with open(filename, 'r', encoding='utf-8') as f:
        # 尝试检测分隔符
        first_line = f.readline()
        f.seek(0)
        
        if '\t' in first_line:
            reader = csv.DictReader(f, delimiter='\t')
        else:
            reader = csv.DictReader(f)
        
        for row in reader:
            total_rows += 1
            
            # 获取基本信息
            pmhc = row.get('pMHC', '')
            binding_status = row.get('Binding', '')
            
            # 提取peptide和MHC
            peptide = extract_peptide_from_pmhc(pmhc)
            mhc_id = extract_mhc_from_pmhc(pmhc)
            
            if not peptide:
                continue
                
            # 获取CDR3序列
            cdr3_a = row.get('CDR3a', '')
            cdr3_b = row.get('CDR3b', '')
            
            if not cdr3_b:
                continue
                
            # 获取基因ID
            trav_id = row.get('TRAV', '')
            traj_id = row.get('TRAJ', '')
            trbv_id = row.get('TRBV', '')
            trbj_id = row.get('TRBJ', '')
            
            # 获取序列
            trav_seq = get_sequence_for_gene(trav_id, gene_to_seq)
            traj_seq = get_sequence_for_gene(traj_id, gene_to_seq)
            trbv_seq = get_sequence_for_gene(trbv_id, gene_to_seq)
            trbj_seq = get_sequence_for_gene(trbj_id, gene_to_seq)
            
            # 获取MHC序列
            mhc_seq = map_mhc_to_sequence(mhc_id, mhc_to_seq)
            
            # 只保留成功映射beta链TCR的记录
            if trbv_seq and trbj_seq:
                binary_label = 1 if binding_status == 'Binding' else 0
                
                record = {
                    'peptide': peptide.strip(),
                    'mhc': mhc_seq if mhc_seq else '',
                    'l_v': trav_seq if trav_seq else '',
                    'l_j': traj_seq if traj_seq else '',
                    'h_v': trbv_seq,
                    'h_j': trbj_seq,
                    'cdr3_b': cdr3_b.strip(),
                    'cdr3_a': cdr3_a.strip() if cdr3_a else '',
                    'binding_label': binary_label,
                    'binding_status': binding_status,
                    'donor': row.get('Donor', ''),
                    'pmhc': pmhc,
                }
                processed_data.append(record)
                successful_rows += 1
    
    print(f"  原始行数: {total_rows}")
    print(f"  成功处理: {successful_rows}")
    return processed_data

def save_data(data, output_dir):
    """保存数据"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存训练格式数据
    training_file = os.path.join(output_dir, f"trait_binding_training_{timestamp}.csv")
    with open(training_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['peptide', 'mhc', 'l_v', 'l_j', 'h_v', 'h_j', 'cdr3_b', 'binding_label'])
        for record in data:
            writer.writerow([
                record['peptide'], record['mhc'], record['l_v'], record['l_j'],
                record['h_v'], record['h_j'], record['cdr3_b'], record['binding_label']
            ])
    
    # 保存完整数据
    full_file = os.path.join(output_dir, f"trait_binding_full_{timestamp}.csv")
    with open(full_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['peptide', 'mhc', 'l_v', 'l_j', 'h_v', 'h_j', 'cdr3_b', 'cdr3_a', 
                     'binding_label', 'binding_status', 'donor', 'pmhc']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    return training_file, full_file

def main():
    """主函数"""
    print("开始处理trait绑定数据")
    
    # 加载序列映射
    gene_to_seq = load_tcr_sequences()
    mhc_to_seq = load_mhc_sequences()
    
    if not gene_to_seq:
        print("无法加载TCR序列，退出")
        return
    
    # 处理数据文件
    pos_file = 'A0201_ELAGIGILTV_MART-1_Cancer_binder_pos.txt'
    neg_file = 'A0201_ELAGIGILTV_MART-1_Cancer_binder_neg.txt'
    
    print("\n处理正向绑定数据...")
    pos_data = process_file(pos_file, gene_to_seq, mhc_to_seq)
    
    print("\n处理负向绑定数据...")
    neg_data = process_file(neg_file, gene_to_seq, mhc_to_seq)
    
    # 合并数据
    all_data = pos_data + neg_data
    print(f"\n合并后总数据量: {len(all_data)} 条")
    print(f"正向绑定样本: {len(pos_data)} 条")
    print(f"负向绑定样本: {len(neg_data)} 条")
    
    # 去除重复项（基于核心字段）
    seen = set()
    unique_data = []
    for record in all_data:
        key = (record['peptide'], record['mhc'], record['l_v'], record['l_j'], 
               record['h_v'], record['h_j'], record['cdr3_b'])
        if key not in seen:
            seen.add(key)
            unique_data.append(record)
    
    duplicates_removed = len(all_data) - len(unique_data)
    print(f"移除重复记录: {duplicates_removed} 条")
    print(f"最终数据量: {len(unique_data)} 条")
    
    # 统计标签分布
    binding_count = sum(1 for r in unique_data if r['binding_label'] == 1)
    non_binding_count = len(unique_data) - binding_count
    
    print(f"绑定样本: {binding_count} 条 ({binding_count/len(unique_data)*100:.2f}%)")
    print(f"非绑定样本: {non_binding_count} 条 ({non_binding_count/len(unique_data)*100:.2f}%)")
    
    # 保存数据
    output_dir = 'processed_trait_binding_data'
    training_file, full_file = save_data(unique_data, output_dir)
    
    print(f"\n处理完成!")
    print(f"训练格式数据: {training_file}")
    print(f"完整数据: {full_file}")

if __name__ == '__main__':
    main() 