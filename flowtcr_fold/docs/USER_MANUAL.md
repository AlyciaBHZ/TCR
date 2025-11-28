# FlowTCR-Fold v2.0 用户手册

> **两步走 TCR 设计**: Scaffold 检索 → CDR3β 生成

---

## 目录

1. [快速开始](#1-快速开始)
2. [项目结构](#2-项目结构)
3. [数据准备](#3-数据准备)
4. [训练流程](#4-训练流程)
5. [推理使用](#5-推理使用)
6. [常见问题](#6-常见问题)

---

## 1. 快速开始

### 1.1 环境配置

```bash
# 创建环境
conda create -n flowtcr python=3.9
conda activate flowtcr

# 安装依赖
pip install torch transformers biopython pandas numpy

# (可选) ESM
pip install fair-esm

# (可选) LoRA 微调
pip install peft
```

### 1.2 三步运行

```bash
# Step 1: 训练 Immuno-PLM (骨架检索)
python flowtcr_fold/Immuno_PLM/train_plm.py --data data/trn.csv --epochs 100

# Step 2: 训练 FlowTCR-Gen (CDR3β 生成)
python flowtcr_fold/FlowTCR_Gen/train_flow.py --data data/trn.csv --epochs 100

# Step 3: 运行设计 Pipeline
python flowtcr_fold/FlowTCR_Gen/pipeline_impl.py --peptide "GILGFVFTL" --mhc "HLA-A*02:01"
```

---

## 2. 项目结构

```
flowtcr_fold/
├── README.md                    # 英文文档 (详细)
├── TODO.md                      # 任务跟踪
├── EVOEF2_INTEGRATION.md        # EvoEF2 集成文档
│
├── docs/
│   ├── USER_MANUAL.md           # 本手册
│   ├── Plan_v2.0.md             # 技术计划 v2.0 (中文详细)
│   ├── initial_plan.md          # 原始设计文档
│   └── initial_plan_update.md   # 更新的设计文档
│
├── data/
│   ├── dataset.py               # 数据集 (三元组采样)
│   ├── tokenizer.py             # 分词器
│   └── convert_csv_to_jsonl.py  # 数据转换
│
├── common/
│   └── utils.py                 # 工具函数 (checkpoint, 早停)
│
├── Immuno_PLM/                  # Stage 1: 骨架检索
│   ├── immuno_plm.py            # 模型定义
│   ├── train_plm.py             # 训练脚本
│   └── eval_plm.py              # 评估脚本
│
├── FlowTCR_Gen/                 # Stage 2: CDR3β 生成
│   ├── flow_gen.py              # Flow Matching 模型
│   ├── train_flow.py            # 训练脚本
│   └── pipeline_impl.py         # 完整 Pipeline
│
├── TCRFold_Light/               # Stage 3: 结构评估 (可选)
│   ├── tcrfold_light.py         # Evoformer-lite
│   ├── train_with_energy.py     # 能量监督训练
│   └── train_*.py               # 其他训练脚本
│
├── physics/                     # 物理工具
│   ├── evoef_runner.py          # EvoEF2 Python 封装
│   ├── energy_dataset.py        # 能量标签数据集
│   ├── test_evoef.py            # 测试脚本
│   └── README.md                # 物理模块文档
│
└── tools/
    └── EvoEF2/                  # EvoEF2 二进制 + 参数
```

---

## 3. 数据准备

### 3.1 数据格式

**必需字段**:
```csv
peptide,mhc,cdr3_b
GILGFVFTL,HLA-A*02:01,CASSLGQAYEQYF
...
```

**可选字段**:
```csv
peptide,mhc,cdr3_b,h_v,h_j,l_v,l_j
GILGFVFTL,HLA-A*02:01,CASSLGQAYEQYF,TRBV19*01,TRBJ2-7,TRAV12-1,TRAJ33
...
```

### 3.2 构建 Scaffold Bank

```bash
# 从训练数据提取唯一的 V/J 组合
python -c "
import pandas as pd
df = pd.read_csv('data/trn.csv')
scaffolds = df.groupby(['h_v','h_j','l_v','l_j']).size().reset_index(name='count')
scaffolds = scaffolds[scaffolds['count'] >= 5]  # 过滤低频
scaffolds.to_csv('data/scaffold_bank.csv', index=False)
print(f'唯一骨架数: {len(scaffolds)}')
"
```

### 3.3 数据清洗

```bash
# CSV 转 JSONL
python flowtcr_fold/data/convert_csv_to_jsonl.py \
    --input data/trn.csv \
    --output data/trn.jsonl
```

### 3.4 PDB 结构 (用于 TCRFold-Light)

```bash
# 创建结构目录
mkdir -p data/pdb_structures

# 下载来源:
# - STCRDab: http://opig.stats.ox.ac.uk/webapps/stcrdab/
# - TCR3d: https://tcr3d.ibbr.umd.edu/

# 放置 PDB 文件
cp *.pdb data/pdb_structures/
```

---

## 4. 训练流程

### 4.1 Stage 1: Immuno-PLM (骨架检索)

**目标**: 学习 TCR-pMHC 兼容性，支持 V/J 骨架检索。

```bash
python flowtcr_fold/Immuno_PLM/train_plm.py \
    --data data/trn.csv \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --tau 0.07 \
    --out_dir checkpoints/plm
```

**参数说明**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | - | 训练数据路径 |
| `--epochs` | 1 | 训练轮数 |
| `--batch_size` | 8 | 批次大小 (越大越好，建议 64+) |
| `--lr` | 1e-4 | 学习率 |
| `--tau` | 0.1 | InfoNCE 温度参数 |
| `--use_esm` | False | 是否使用 ESM-2 |
| `--mlm_weight` | 1.0 | MLM 损失权重 |

**验证**:
```bash
python flowtcr_fold/Immuno_PLM/eval_plm.py \
    --data data/val.csv \
    --checkpoint checkpoints/plm/immuno_plm.pt
```

### 4.2 Stage 2: FlowTCR-Gen (CDR3β 生成)

**目标**: 学习条件生成 CDR3β。

```bash
python flowtcr_fold/FlowTCR_Gen/train_flow.py \
    --data data/trn.csv \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --out_dir checkpoints/flow
```

### 4.3 Stage 3: TCRFold-Light (结构评估，可选)

**目标**: 预测结构特征，过滤候选。

```bash
# 需要 PDB 结构和 EvoEF2
python flowtcr_fold/TCRFold_Light/train_with_energy.py \
    --pdb_dir data/pdb_structures \
    --cache_dir data/energy_cache \
    --epochs 100 \
    --interface_weight 10.0
```

### 4.4 训练偏好

| 设置 | 值 |
|------|-----|
| Checkpoint 频率 | 每 50 epochs |
| 早停耐心 | 100 epochs 无提升 |
| 梯度裁剪 | max_norm=1.0 |

---

## 5. 推理使用

### 5.1 完整 Pipeline

```bash
python flowtcr_fold/FlowTCR_Gen/pipeline_impl.py \
    --peptide "GILGFVFTL" \
    --mhc "HLA-A*02:01" \
    --top_k_scaffolds 10 \
    --samples_per_scaffold 100 \
    --output results/designs.csv
```

### 5.2 Python API

```python
from flowtcr_fold.Immuno_PLM import ImmunoPLM
from flowtcr_fold.FlowTCR_Gen import FlowMatchingModel

# 加载模型
plm = ImmunoPLM.load("checkpoints/plm/immuno_plm.pt")
flow = FlowMatchingModel.load("checkpoints/flow/flow_gen.pt")

# Stage 1: 检索骨架
pmhc_emb = plm.encode_pmhc("GILGFVFTL", "HLA-A*02:01")
scaffold_bank = load_scaffold_bank("data/scaffold_bank.csv")
scaffold_embs = plm.encode_scaffolds(scaffold_bank)
top_scaffolds = retrieve_top_k(pmhc_emb, scaffold_embs, k=10)

# Stage 2: 生成 CDR3β
for scaffold in top_scaffolds:
    scaffold_emb = plm.encode_scaffold(scaffold)
    condition = concat(pmhc_emb, scaffold_emb)
    cdr3b = flow.sample(condition)
    print(f"Scaffold: {scaffold}, CDR3β: {cdr3b}")
```

### 5.3 输出格式

```csv
rank,scaffold_hv,scaffold_hj,scaffold_lv,scaffold_lj,cdr3b,tcrfold_score,evoef2_energy
1,TRBV19*01,TRBJ2-7,TRAV12-1,TRAJ33,CASSLGQAYEQYF,0.85,-12.3
2,...
```

---

## 6. 常见问题

### Q1: 显存不足

**问题**: 使用 ESM-2 时 CUDA OOM

**解决方案**:
```bash
# 方案1: 使用 BasicTokenizer (不使用 ESM)
python train_plm.py --data data/trn.csv  # 默认不用 ESM

# 方案2: 减小 batch size
python train_plm.py --data data/trn.csv --batch_size 16

# 方案3: 使用梯度累积 (需修改代码)
```

### Q2: EvoEF2 找不到

**问题**: `FileNotFoundError: EvoEF2 executable not found`

**解决方案**:
```bash
# 编译 EvoEF2
cd flowtcr_fold/tools/EvoEF2
git clone https://github.com/tommyhuangthu/EvoEF2 .
g++ -O3 --fast-math -o EvoEF2 src/*.cpp

# 验证
python flowtcr_fold/physics/test_evoef.py
```

### Q3: 数据格式错误

**问题**: `KeyError: 'peptide'`

**解决方案**:
检查 CSV 列名是否正确:
```bash
head -1 data/trn.csv
# 应该包含: peptide,mhc,cdr3_b
```

### Q4: 训练不收敛

**问题**: InfoNCE loss 不下降

**解决方案**:
1. 增大 Batch Size (至少 64)
2. 调整温度参数 `--tau 0.05`
3. 检查数据是否有重复

### Q5: 生成质量差

**问题**: 生成的 CDR3β 长度或组成不合理

**解决方案**:
1. 增加训练轮数
2. 检查条件编码是否正确
3. 添加长度约束到生成过程

---

## 附录

### A. 关键文件路径

| 文件 | 路径 | 用途 |
|------|------|------|
| 训练数据 | `data/trn.csv` | 主训练集 |
| 验证数据 | `data/val.csv` | 验证集 |
| 骨架库 | `data/scaffold_bank.csv` | V/J 组合库 |
| PLM 模型 | `checkpoints/plm/immuno_plm.pt` | Immuno-PLM 权重 |
| Flow 模型 | `checkpoints/flow/flow_gen.pt` | FlowTCR-Gen 权重 |
| 结构模型 | `checkpoints/tcrfold/tcrfold_light.pt` | TCRFold-Light 权重 |

### B. 详细文档

| 文档 | 位置 | 内容 |
|------|------|------|
| 项目总览 | `README.md` | 英文完整文档 |
| 技术计划 | `docs/Plan_v2.0.md` | 中文详细计划 |
| EvoEF2 集成 | `EVOEF2_INTEGRATION.md` | 物理模块文档 |
| 物理模块 | `physics/README.md` | EvoEF2 使用说明 |

---

**文档版本**: 2.0  
**最后更新**: 2025-11-28
