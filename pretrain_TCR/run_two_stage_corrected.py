#!/usr/bin/env python3
"""
修正的两阶段训练流程
阶段1: 大规模无标注TCR数据预训练 (50万-100万条)
阶段2: 高质量标注数据微调 (22万条)
"""

import os
import sys
import subprocess
import argparse
import torch

def stage1_large_scale_pretraining():
    """
    阶段1: 大规模TCR数据预训练
    目标: 学习通用的TCR序列表示和语法规律
    """
    print("🚀 STAGE 1: LARGE-SCALE TCR PRETRAINING")
    print("="*60)
    print("目标: 从大规模TCR数据学习通用表示")
    print("数据: TCRdb + VDJdb_full + IEDB_full (50万+ 序列)")
    print("任务: 纯粹的Masked Language Modeling")
    print("="*60)
    
    # 检查大规模数据是否准备好
    large_data_path = '../data/large_scale_tcr_pretrain.csv'
    
    if not os.path.exists(large_data_path):
        print("❌ 大规模数据集未找到")
        print("\n请先准备大规模TCR数据：")
        print("1. 运行: python pretrain_large_scale.py")
        print("2. 按照指南下载和处理数据")
        print("3. 然后重新运行此脚本")
        return False
    
    # 运行大规模预训练
    print("开始大规模预训练...")
    try:
        result = subprocess.run([
            'python', 'pretrain_large_scale.py'
        ], check=True, capture_output=True, text=True)
        
        print("✅ 阶段1完成: 大规模预训练")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 阶段1失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def stage2_high_quality_finetuning(pretrained_model_path):
    """
    阶段2: 高质量数据微调
    目标: 学习特定的TCR-peptide-MHC功能关联
    """
    print("\n🎯 STAGE 2: HIGH-QUALITY DATA FINE-TUNING")
    print("="*60)
    print("目标: 学习特定的生物功能关联")
    print("数据: 您的22万高质量标注数据")
    print("任务: 条件化的TCR-peptide-MHC预测")
    print("="*60)
    
    # 检查预训练模型
    if not os.path.exists(pretrained_model_path):
        print(f"❌ 预训练模型未找到: {pretrained_model_path}")
        return False
    
    # 检查高质量数据
    high_quality_data = '../data/trn.csv'
    if not os.path.exists(high_quality_data):
        print(f"❌ 高质量训练数据未找到: {high_quality_data}")
        return False
    
    print(f"加载预训练模型: {pretrained_model_path}")
    print(f"微调数据: {high_quality_data}")
    
    # 运行微调
    try:
        result = subprocess.run([
            'python', 'finetune.py',
            '--pretrained_model', pretrained_model_path,
            '--freeze_strategy', 'partial',  # 部分冻结策略
            '--learning_rate', '1e-5',       # 微调用较小学习率
            '--batch_size', '256'            # 适中的batch size
        ], check=True, capture_output=True, text=True)
        
        print("✅ 阶段2完成: 高质量微调")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 阶段2失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def analyze_data_strategy():
    """
    分析数据使用策略
    """
    print("📊 DATA STRATEGY ANALYSIS")
    print("="*50)
    
    print("传统策略 (之前的建议):")
    print("  ❌ 22万高质量数据预训练 → 扩展数据微调")
    print("  问题: 浪费了高质量标注信息")
    
    print("\n正确策略 (您的建议):")
    print("  ✅ 大规模无标注数据预训练 → 22万高质量数据微调")
    print("  优势:")
    print("    1. 预训练学习通用TCR语法")
    print("    2. 微调学习特定功能关联")
    print("    3. 充分利用高质量标注")
    print("    4. 符合现代预训练范式")
    
    print("\n数据规模预期:")
    print("  阶段1预训练: 50万-100万条TCR序列")
    print("  阶段2微调: 22万条高质量功能标注")
    print("  总计: 70万-120万条数据")
    
    print("\n理论基础:")
    print("  1. 大规模无监督预训练 → 通用表示")
    print("  2. 任务特定有监督微调 → 功能关联")
    print("  3. 这正是BERT/GPT的成功模式")

def main():
    parser = argparse.ArgumentParser(description='Two-Stage TCR Training Pipeline')
    parser.add_argument('--skip_stage1', action='store_true', 
                       help='Skip large-scale pretraining')
    parser.add_argument('--pretrained_model', type=str,
                       help='Path to pretrained model for stage 2')
    
    args = parser.parse_args()
    
    print("🧬 psiCLM Two-Stage Training Pipeline")
    print("基于psihē理论的TCR序列生成模型训练")
    print("="*60)
    
    # 分析策略
    analyze_data_strategy()
    
    pretrained_model_path = None
    
    # 阶段1: 大规模预训练
    if not args.skip_stage1:
        success = stage1_large_scale_pretraining()
        if not success:
            print("❌ 流程终止: 阶段1失败")
            return
        
        pretrained_model_path = './saved_model/large_scale_pretrain/best_large_scale_pretrain'
    else:
        pretrained_model_path = args.pretrained_model
        if not pretrained_model_path:
            print("❌ 跳过阶段1时必须提供预训练模型路径")
            return
    
    # 阶段2: 高质量微调
    success = stage2_high_quality_finetuning(pretrained_model_path)
    if not success:
        print("❌ 流程终止: 阶段2失败")
        return
    
    print("\n🎉 TWO-STAGE TRAINING COMPLETED!")
    print("="*50)
    print("模型训练完成，具备以下能力：")
    print("1. 通用TCR序列理解 (来自大规模预训练)")
    print("2. 特定功能预测 (来自高质量微调)")
    print("3. 条件化生成 (TCR-peptide-MHC关联)")
    
    print("\n下一步建议：")
    print("1. 运行评估脚本测试模型性能")
    print("2. 使用attention可视化验证学习效果")
    print("3. 进行wet lab实验验证")

if __name__ == '__main__':
    main() 