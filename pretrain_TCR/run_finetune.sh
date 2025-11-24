#!/bin/bash

# CDR3β-Peptide-MHC Paired Data Finetuning Script
# 基于预训练的CDR3β模型进行配对数据微调

echo "Starting CDR3β-Peptide-MHC Paired Data Finetuning..."
echo "=============================================="

# 设置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 预训练模型路径 (使用最新的检查点)
PRETRAINED_MODEL="$SCRIPT_DIR/saved_model/cdr3_pretrain/cdr3_model_epoch_500"

# 训练数据路径
TRAINING_DATA="$PROJECT_ROOT/data/trn.csv"

# 检查文件是否存在
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "Error: Pretrained model not found at $PRETRAINED_MODEL"
    echo "Available checkpoints:"
    ls -la "$SCRIPT_DIR/saved_model/cdr3_pretrain/"
    exit 1
fi

if [ ! -f "$TRAINING_DATA" ]; then
    echo "Error: Training data not found at $TRAINING_DATA"
    exit 1
fi

echo "Pretrained model: $PRETRAINED_MODEL"
echo "Training data: $TRAINING_DATA"
echo "Starting finetuning..."
echo ""

# 运行微调脚本
cd "$SCRIPT_DIR"
python finetune_pairs.py \
    --pretrained "$PRETRAINED_MODEL" \
    --data "$TRAINING_DATA"

echo ""
echo "Finetuning completed!"
echo "Check the results in: $SCRIPT_DIR/saved_model/paired_finetune/" 