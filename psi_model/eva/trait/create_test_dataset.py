#!/usr/bin/env python3
"""
从训练数据中筛选测试数据集
包含所有positive binding + positive*1.5的negative binding
"""

import csv
import random
import os
from datetime import datetime

def create_test_dataset(input_file, output_dir='processed_trait_binding_data'):
    """创建测试数据集"""
    
    print("开始创建测试数据集...")
    print(f"读取数据文件: {input_file}")
    
    # 读取数据并分离positive和negative样本
    positive_samples = []
    negative_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            if row['binding_label'] == '1':
                positive_samples.append(row)
            else:
                negative_samples.append(row)
    
    print(f"读取完成:")
    print(f"  Positive样本: {len(positive_samples)}")
    print(f"  Negative样本: {len(negative_samples)}")
    
    # 计算需要的negative样本数量
    negative_needed = int(len(positive_samples) * 1.5)
    print(f"  需要的negative样本数: {negative_needed}")
    
    # 检查是否有足够的negative样本
    if len(negative_samples) < negative_needed:
        print(f"警告: 可用negative样本({len(negative_samples)})少于需要的数量({negative_needed})")
        print(f"将使用所有可用的negative样本")
        selected_negative = negative_samples
    else:
        # 随机选择negative样本
        random.seed(42)  # 设置随机种子以确保结果可重现
        selected_negative = random.sample(negative_samples, negative_needed)
    
    # 合并数据
    test_data = positive_samples + selected_negative
    
    print(f"测试数据集创建完成:")
    print(f"  总样本数: {len(test_data)}")
    print(f"  Positive样本: {len(positive_samples)} ({len(positive_samples)/len(test_data)*100:.2f}%)")
    print(f"  Negative样本: {len(selected_negative)} ({len(selected_negative)/len(test_data)*100:.2f}%)")
    
    # 打乱数据顺序
    random.shuffle(test_data)
    
    # 保存测试数据集
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存训练格式的测试数据
    test_file = os.path.join(output_dir, f"trait_test_dataset_{timestamp}.csv")
    with open(test_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_data)
    
    # 也保存一个只包含训练所需列的版本
    training_format_file = os.path.join(output_dir, f"trait_test_training_format_{timestamp}.csv")
    training_columns = ['peptide', 'mhc', 'l_v', 'l_j', 'h_v', 'h_j', 'cdr3_b', 'binding_label']
    
    with open(training_format_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(training_columns)
        for row in test_data:
            writer.writerow([row[col] for col in training_columns])
    
    print(f"\n测试数据集已保存:")
    print(f"  完整格式: {test_file}")
    print(f"  训练格式: {training_format_file}")
    
    # 验证标签分布
    pos_count = sum(1 for row in test_data if row['binding_label'] == '1')
    neg_count = len(test_data) - pos_count
    
    print(f"\n最终标签分布验证:")
    print(f"  Binding (1): {pos_count}")
    print(f"  Non-binding (0): {neg_count}")
    print(f"  比例: {pos_count}:{neg_count} = 1:{neg_count/pos_count:.1f}")
    
    return test_file, training_format_file

def main():
    """主函数"""
    # 查找最新的训练数据文件
    data_dir = 'processed_trait_binding_data'
    
    if not os.path.exists(data_dir):
        print(f"错误: 找不到数据目录 {data_dir}")
        return
    
    # 查找训练格式文件
    training_files = [f for f in os.listdir(data_dir) if f.startswith('trait_binding_training_')]
    
    if not training_files:
        print(f"错误: 在 {data_dir} 中找不到训练数据文件")
        return
    
    # 使用最新的文件
    latest_file = sorted(training_files)[-1]
    input_file = os.path.join(data_dir, latest_file)
    
    print(f"使用训练数据文件: {input_file}")
    
    # 创建测试数据集
    test_file, training_format_file = create_test_dataset(input_file)
    
    print(f"\n创建测试数据集完成!")

if __name__ == '__main__':
    main() 