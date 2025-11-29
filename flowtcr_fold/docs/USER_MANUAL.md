# FlowTCR-Fold v2.0 用户手册

> **两步走 TCR 设计**: Scaffold 检索 → CDR3β 生成

---

## 📋 开发状态总览

| 模块 | 状态 | 说明 |
|------|------|------|
| **Step 1: Scaffold Retrieval** | ✅ 完成 | `train_scaffold_retrieval.py` |
| **Step 2: FlowTCR-Gen** | 🚧 待开发 | CDR3β 生成模型 |
| **Step 3: TCRFold-Light** | 🚧 待开发 | 结构评估（可选） |
| **完整 Pipeline** | 🚧 待整合 | 端到端推理 |

---

## 目录

1. [快速开始](#1-快速开始)
2. [数据准备](#2-数据准备)
3. [Step 1: Scaffold 检索训练](#3-step-1-scaffold-检索训练)
4. [Step 2: CDR3β 生成训练](#4-step-2-cdr3β-生成训练-待开发)
5. [Step 3: 结构评估](#5-step-3-结构评估-可选)
6. [完整 Pipeline 推理](#6-完整-pipeline-推理)
7. [下一步开发路线图](#7-下一步开发路线图)
8. [常见问题](#8-常见问题)

---

## 1. 快速开始

### 1.1 环境配置

```bash
# 创建环境
conda create -n flowtcr python=3.9
conda activate flowtcr

# 安装核心依赖
pip install torch transformers biopython pandas numpy tqdm

# ESM-2 (用于蛋白质编码，必需)
pip install fair-esm

# 注意：LoRA 使用内置实现 (immuno_plm.py 中的 LoRALinear)，无需 peft
```

### 1.2 验证安装

```bash
cd /mnt/rna01/zwlexa/project/TCR

# 测试 import
python -c "
from flowtcr_fold.Immuno_PLM.immuno_plm import ImmunoPLM
from flowtcr_fold.data.tokenizer import BasicTokenizer
print('✅ 导入成功!')
"
```

---

## 2. 数据准备

### 2.1 数据格式要求

**JSONL 格式** (推荐):

```jsonl
{"pep": "GILGFVFTL", "mhc": "HLA-A*02:01", "cdr3_b": "CASSLGQAYEQYF", "h_v": "TRBV19*01", "h_j": "TRBJ2-7*01", "l_v": "TRAV12-1*01", "l_j": "TRAJ33*01", "h_v_seq": "MGVTQTP...", "h_j_seq": "EAFF...", "l_v_seq": "MTRV...", "l_j_seq": "QLIF..."}
```

**必需字段**:

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `pep` | string | 抗原肽序列 | `GILGFVFTL` |
| `mhc` | string | MHC 等位基因 | `HLA-A*02:01` |
| `h_v_seq` | string | Heavy V 基因**序列** | `MGVTQTP...` |
| `h_j_seq` | string | Heavy J 基因**序列** | `EAFF...` |
| `l_v_seq` | string | Light V 基因**序列** | `MTRV...` |
| `l_j_seq` | string | Light J 基因**序列** | `QLIF...` |

**可选字段** (用于辅助分类):

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `h_v` | string | Heavy V 基因**名称** | `TRBV19*01` |
| `h_j` | string | Heavy J 基因**名称** | `TRBJ2-7*01` |
| `l_v` | string | Light V 基因**名称** | `TRAV12-1*01` |
| `l_j` | string | Light J 基因**名称** | `TRAJ33*01` |
| `cdr3_b` | string | CDR3β 序列 (Step 2 用) | `CASSLGQAYEQYF` |

### 2.2 检查现有数据

```bash
# 查看数据格式
head -1 flowtcr_fold/data/tst.jsonl | python -m json.tool

# 统计样本数
wc -l flowtcr_fold/data/*.jsonl
# 输出: trn.jsonl (133MB), val.jsonl (8.2MB), tst.jsonl (15MB)
```

### 2.3 CSV 转 JSONL

如果你的数据是 CSV 格式：

```bash
# 注意：数据已经是 JSONL 格式 (trn.jsonl, val.jsonl, tst.jsonl)
# 如果需要从 CSV 转换：
python flowtcr_fold/data/convert_csv_to_jsonl.py \
    --input flowtcr_fold/data/trn.csv \
    --output flowtcr_fold/data/trn_new.jsonl
```

### 2.4 数据验证脚本

```python
# validate_data.py
import json

def validate_jsonl(path):
    required = ["pep", "mhc", "h_v_seq", "h_j_seq", "l_v_seq", "l_j_seq"]
    errors = []
    
    with open(path) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            for field in required:
                if field not in row or not row[field]:
                    errors.append(f"Line {i+1}: missing {field}")
    
    if errors:
        print(f"❌ Found {len(errors)} errors:")
        for e in errors[:10]:
            print(f"  {e}")
    else:
        print(f"✅ All {i+1} samples valid!")

validate_jsonl("flowtcr_fold/data/trn.jsonl")
```

---

## 3. Step 1: Scaffold 检索训练

### 3.1 架构说明

**Immuno-PLM (Scaffold Retrieval)**:
- **输入**: pMHC 序列 + V/J 基因序列
- **编码器**: ESM-2 + LoRA (共享权重)
- **训练目标**:
  - 4 路 InfoNCE: 拉近配对的 (pMHC, V/J) embedding
  - 4 分类头: 预测 Gene Name (辅助任务)
- **输出**: Scaffold Bank (V/J 序列 → embedding 映射)

```
pMHC ─────┐
          │  Shared     ┌─ InfoNCE(pMHC, HV)
HV_seq ───┤  ESM-2  ────├─ InfoNCE(pMHC, HJ)
HJ_seq ───┤  +LoRA      ├─ InfoNCE(pMHC, LV)
LV_seq ───┤             └─ InfoNCE(pMHC, LJ)
LJ_seq ───┘
                        ┌─ Classify(pMHC → HV_name)
          pMHC_emb ─────├─ Classify(pMHC → HJ_name)
                        ├─ Classify(pMHC → LV_name)
                        └─ Classify(pMHC → LJ_name)
```

### 3.2 训练命令

**基础训练** (快速测试):

```bash
cd /mnt/rna01/zwlexa/project/TCR

python -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval \
    --data flowtcr_fold/data/tst.jsonl \
    --epochs 5 \
    --batch_size 8 \
    --lr 1e-4 \
    --out_dir checkpoints/scaffold_test
```

**完整训练** (带验证集):

```bash
python -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval \
    --data flowtcr_fold/data/trn.jsonl \
    --val_data flowtcr_fold/data/val.jsonl \
    --use_esm \
    --use_lora \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --tau 0.07 \
    --cls_weight 0.2 \
    --patience 20 \
    --out_dir checkpoints/scaffold_v1
```

**参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | 必需 | 训练数据 (.jsonl) |
| `--val_data` | None | 验证数据 (用于早停) |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 (越大越好) |
| `--lr` | 1e-4 | 学习率 |
| `--tau` | 0.07 | InfoNCE 温度 (越小越严格) |
| `--cls_weight` | 0.2 | 分类损失权重 |
| `--patience` | 10 | 早停耐心 (epochs) |
| `--bank_mode` | benchmark | `benchmark` 或 `production` |
| `--out_dir` | checkpoints | 输出目录 |

### 3.3 训练输出

```
checkpoints/scaffold_v1/
├── scaffold_retriever.pt      # 模型权重
├── scaffold_bank.pt           # V/J embedding 库
├── gene_vocab.json            # Gene Name → ID 映射
└── training.log               # 训练日志
```

### 3.4 验证训练效果

```bash
# 查看训练曲线
grep "Epoch" checkpoints/scaffold_v1/training.log

# 期望看到:
# Epoch 1/100 | Train: loss=8.21 (NCE=7.89, CLS=1.60) | Val: loss=7.95, R@10_HV=0.12
# Epoch 10/100 | Train: loss=4.32 (NCE=3.98, CLS=1.70) | Val: loss=4.15, R@10_HV=0.45
# ...
# Epoch 50/100 | Train: loss=2.01 (NCE=1.72, CLS=1.45) | Val: loss=2.35, R@10_HV=0.78
```

### 3.5 Bank 模式说明

| 模式 | 数据来源 | 用途 |
|------|----------|------|
| `benchmark` | 仅 train | 评估泛化能力 (防止数据泄漏) |
| `production` | train + val + test | 实际应用 (最大覆盖) |

```bash
# Benchmark 模式 (评估用)
python -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval \
    --data data/trn.jsonl --bank_mode benchmark

# Production 模式 (部署用)
python -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval \
    --data data/trn.jsonl --bank_mode production \
    --bank_extra_data data/val.jsonl data/tst.jsonl
```

### 3.6 推理示例

```python
import torch
from flowtcr_fold.Immuno_PLM.train_scaffold_retrieval import ScaffoldRetriever, ScaffoldBank

# 加载模型和 Bank
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ScaffoldRetriever(...)
model.load_state_dict(torch.load("checkpoints/scaffold_v1/scaffold_retriever.pt"))
model.to(device).eval()

bank = ScaffoldBank()
bank.load("checkpoints/scaffold_v1/scaffold_bank.pt")
bank.build_tensors(device)

# 编码新 pMHC
pmhc_tokens = tokenize("GILGFVFTL HLA-A*02:01")
pmhc_emb = model.encode(pmhc_tokens.to(device), ...)

# 检索 Top-K V/J
results = bank.retrieve(pmhc_emb, top_k=5, gene_type="h_v")
for seq, name, score in results:
    print(f"HV: {name} (score={score:.3f})")
    print(f"    Sequence: {seq[:50]}...")
```

---

## 4. Step 2: CDR3β 生成训练 (🚧 待开发)

### 4.1 设计目标

**FlowTCR-Gen**:
- **输入**: pMHC embedding + Scaffold embeddings (来自 Step 1)
- **方法**: Flow Matching 或 Diffusion
- **输出**: CDR3β 序列

### 4.2 待实现文件

```
flowtcr_fold/FlowTCR_Gen/
├── flow_gen.py         # Flow Matching 模型
├── train_flow.py       # 训练脚本
└── sample.py           # 采样脚本
```

### 4.3 计划的训练命令

```bash
# (待实现)
python -m flowtcr_fold.FlowTCR_Gen.train_flow \
    --data flowtcr_fold/data/trn.jsonl \
    --plm_checkpoint checkpoints/scaffold_v1/scaffold_retriever.pt \
    --epochs 100 \
    --out_dir checkpoints/flow_v1
```

### 4.4 开发优先级

1. **数据准备**: 确保 `cdr3_b` 字段在 JSONL 中
2. **条件编码**: 整合 Step 1 的 embedding 作为条件
3. **Flow 模型**: 实现序列生成的 Flow Matching
4. **采样**: 实现温度控制和长度约束

---

## 5. Step 3: 结构评估 (🚧 可选)

### 5.1 设计目标

**TCRFold-Light**:
- **输入**: 完整 TCR 序列 (Scaffold + CDR3β)
- **方法**: Evoformer-lite + 能量监督
- **输出**: 结构置信度分数

### 5.2 待实现

- [ ] Evoformer-lite 前向传播
- [ ] EvoEF2 能量计算集成
- [ ] 训练脚本

---

## 6. 完整 Pipeline 推理

### 6.1 端到端流程

```
Input: (peptide, MHC)
    │
    ▼
┌─────────────────────────────┐
│  Step 1: Scaffold Retrieval │
│  pMHC → Top-K (HV,HJ,LV,LJ) │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Step 2: CDR3β Generation   │
│  (pMHC, Scaffold) → CDR3β   │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Step 3: Structure Scoring  │  (可选)
│  Full TCR → Confidence      │
└─────────────────────────────┘
    │
    ▼
Output: Ranked TCR candidates
```

### 6.2 计划的推理命令

```bash
# (待实现)
python -m flowtcr_fold.pipeline \
    --peptide "GILGFVFTL" \
    --mhc "HLA-A*02:01" \
    --scaffold_model checkpoints/scaffold_v1/scaffold_retriever.pt \
    --flow_model checkpoints/flow_v1/flow_gen.pt \
    --top_k_scaffolds 10 \
    --samples_per_scaffold 100 \
    --output results/designs.csv
```

---

## 7. 下一步开发路线图

### 7.1 当前可测试 (Step 1)

```bash
# 1. 准备测试数据 (使用已有的 tst.jsonl)
head -100 flowtcr_fold/data/tst.jsonl > flowtcr_fold/data/tiny.jsonl

# 2. 快速训练测试
python -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval \
    --data flowtcr_fold/data/tiny.jsonl \
    --use_esm \
    --use_lora \
    --epochs 3 \
    --batch_size 8 \
    --out_dir checkpoints/tiny_test

# 3. 验证输出
ls -la checkpoints/tiny_test/
python -c "
import torch
bank = torch.load('checkpoints/tiny_test/scaffold_bank.pt')
print(f'Bank contains {len(bank[\"h_v\"])} HV sequences')
"
```

### 7.2 开发优先级

| 优先级 | 任务 | 依赖 | 预计工时 |
|--------|------|------|----------|
| P0 | 完成 Step 1 完整训练 | 数据 | 1 天 |
| P0 | 验证 Bank 检索质量 | Step 1 | 0.5 天 |
| P1 | 实现 FlowTCR-Gen | Step 1 | 3 天 |
| P1 | 训练 Flow 模型 | 数据 | 2 天 |
| P2 | 整合端到端 Pipeline | Step 1+2 | 1 天 |
| P3 | 实现 TCRFold-Light | PDB 数据 | 5 天 |

### 7.3 数据需求清单

| 数据 | 用途 | 大小 | 状态 |
|------|------|------|------|
| `trn.jsonl` | Step 1 训练 | 133MB | ✅ 有 |
| `val.jsonl` | 验证/早停 | 8.2MB | ✅ 有 |
| `tst.jsonl` | 测试 | 15MB | ✅ 有 |
| `trn.csv` / `val.csv` / `tst.csv` | CSV 版本 | - | ✅ 有 |
| CDR3β 标签 | Step 2 训练 | - | ⚠️ 检查是否在 JSONL 中 |
| PDB 结构 | Step 3 训练 | - | 🚧 可选 |

### 7.4 检查 CDR3β 数据

```bash
# 检查 cdr3_b 字段是否存在
python -c "
import json
with open('flowtcr_fold/data/trn.jsonl') as f:
    sample = json.loads(f.readline())
    if 'cdr3_b' in sample:
        print(f'✅ cdr3_b exists: {sample[\"cdr3_b\"]}')
    else:
        print('❌ cdr3_b missing! Need to add for Step 2')
"
```

---

## 8. 常见问题

### Q1: 显存不足 (OOM)

```bash
# 减小 batch size
python -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval \
    --data data/trn.jsonl --batch_size 16

# 或使用 CPU (慢)
CUDA_VISIBLE_DEVICES="" python -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval ...
```

### Q2: InfoNCE loss 不下降

1. **增大 batch size**: 至少 32，最好 64+
2. **降低温度**: `--tau 0.05`
3. **检查数据**: 确保序列字段不为空

### Q3: 分类准确率很低

- 正常现象！分类是辅助任务
- Gene family 很多，准确率 30-50% 已经不错
- 重点关注 InfoNCE loss 和 Recall@K

### Q4: Bank 检索结果不好

1. 训练更多 epochs
2. 增大 batch size (InfoNCE 需要足够的负样本)
3. 检查 validation loss 是否收敛

### Q5: 如何扩展到新的 V/J 基因？

在 `production` 模式下：
```bash
python -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval \
    --data data/trn.jsonl \
    --bank_mode production \
    --bank_extra_data data/new_genes.jsonl
```

---

## 附录

### A. 关键文件

| 文件 | 路径 | 状态 |
|------|------|------|
| **Scaffold 训练** | `flowtcr_fold/Immuno_PLM/train_scaffold_retrieval.py` | ✅ |
| **模型定义** | `flowtcr_fold/Immuno_PLM/immuno_plm.py` | ✅ |
| **数据集** | `flowtcr_fold/data/scaffold_retrieval_dataset.py` | ✅ |
| **Tokenizer** | `flowtcr_fold/data/tokenizer.py` | ✅ |
| Flow 生成 | `flowtcr_fold/FlowTCR_Gen/flow_gen.py` | 🚧 |
| 结构预测 | `flowtcr_fold/TCRFold_Light/tcrfold_light.py` | 🚧 |

### B. 过时文件 (勿用)

| 文件 | 说明 |
|------|------|
| `train_plm.py` | ⚠️ DEPRECATED - 有 bug，已被 `train_scaffold_retrieval.py` 替代 |
| `train_siamese.py` | 已删除 |
| `train_parallel.py` | 已删除 |
| `train_gene_classifier.py` | 已删除 |

### C. 联系与支持

- 技术文档: `flowtcr_fold/docs/Plan_v2.0.md`
- README: `flowtcr_fold/README.md`

---

**文档版本**: 2.1  
**最后更新**: 2025-11-29  
**开发状态**: Step 1 ✅ | Step 2 🚧 | Step 3 🚧
