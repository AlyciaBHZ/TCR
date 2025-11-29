# FlowTCR-Fold v2.0 技术实施计划

> **核心更新**: 明确"两步走"策略（Scaffold Retrieval + CDR3β Generation），整合三大模块形成完整 Pipeline。

---

## 目录

1. [项目目标与核心挑战](#1-项目目标与核心挑战)
2. [两步走设计策略](#2-两步走设计策略)
3. [模块架构详解](#3-模块架构详解)
4. [数据基础设施](#4-数据基础设施)
5. [训练策略](#5-训练策略)
6. [推理流程](#6-推理流程)
7. [实施路线图](#7-实施路线图)
8. [关键技术决策](#8-关键技术决策)
9. [风险评估与应对](#9-风险评估与应对)

---

## 1. 项目目标与核心挑战

### 1.1 科学目标

**输入**: 目标 peptide-MHC (pMHC) 复合物  
**输出**: 能够特异性识别该 pMHC 的完整 TCR 序列

```
目标 pMHC → 完整 TCR 序列
           ├── V/J 骨架 (h_v, h_j, l_v, l_j)
           └── CDR3β 序列 (关键识别区域)
```

### 1.2 核心挑战

| 挑战 | 描述 | 难度 |
|------|------|------|
| **组合爆炸** | V/J 基因组合空间巨大（Vβ×Jβ×Vα×Jα ≈ 数十万种） | 🔴 高 |
| **离散空间** | V/J 基因是离散类别，不是连续序列 | 🔴 高 |
| **稀疏数据** | 很多 V/J 组合在训练数据中只出现几次 | 🟡 中 |
| **序列-结构耦合** | CDR3 必须与骨架结构兼容 | 🟡 中 |
| **特异性验证** | 难以实验验证生成序列的结合能力 | 🔴 高 |

### 1.3 解决方案概述

**Retrieve & Generate 范式**：将"生成完整 TCR"分解为两个可控子问题

1. **Scaffold Retrieval**: 从数据库检索与 MHC 兼容的 V/J 骨架
2. **CDR3β Generation**: 给定骨架和 pMHC，生成 CDR3β

**优势**：
- ✅ 避免离散空间的生成困难
- ✅ 利用已有数据的 V/J 组合
- ✅ 骨架-MHC 兼容性有物理意义
- ✅ CDR3 生成是成熟的条件生成问题

---

## 2. 两步走设计策略

### 2.1 完整流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         输入：目标 pMHC                              │
│                    (peptide + MHC 等位基因)                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              第一步：Scaffold Retrieval (骨架检索)              │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │                                                               │  │
│  │  模型：Immuno-PLM (InfoNCE 编码器)                             │  │
│  │                                                               │  │
│  │  输入：目标 pMHC 序列                                          │  │
│  │                                                               │  │
│  │  操作：                                                        │  │
│  │    1. 用 Immuno-PLM 编码 pMHC                                  │  │
│  │    2. 与 Scaffold Bank 中的嵌入计算相似度                       │  │
│  │    3. 取 Top-K 最相似的骨架                                    │  │
│  │                                                               │  │
│  │  输出：Top-K 个 (h_v, h_j, l_v, l_j) 序列                      │  │
│  │                                                               │  │
│  │  物理意义：找到与该 MHC 结构最兼容的 V 基因框架                  │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                    │                                │
│                                    ▼                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              第二步：CDR3β Generation (CDR3 生成)               │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │                                                               │  │
│  │  模型：FlowTCR-Gen (条件流匹配)                                 │  │
│  │                                                               │  │
│  │  输入：                                                        │  │
│  │    - 目标 pMHC 序列                                            │  │
│  │    - 第一步选出的 Scaffold 序列                                 │  │
│  │                                                               │  │
│  │  操作：                                                        │  │
│  │    - Dirichlet Flow Matching 在氨基酸单纯形上                   │  │
│  │    - 条件：pMHC + Scaffold 嵌入                                 │  │
│  │    - 采样多个候选 CDR3β                                        │  │
│  │                                                               │  │
│  │  输出：CDR3β 候选序列池                                         │  │
│  │                                                               │  │
│  │  物理意义：在固定骨架下，生成能结合目标肽段的 CDR3 环            │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                    │                                │
│                                    ▼                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              第三步：Structure Critique (结构评估，可选)         │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │                                                               │  │
│  │  模型：TCRFold-Light + EvoEF2                                  │  │
│  │                                                               │  │
│  │  操作：                                                        │  │
│  │    - TCRFold-Light 预测接触图和置信度                           │  │
│  │    - 过滤结构不合理的候选                                       │  │
│  │    - (可选) EvoEF2 计算精确结合能                               │  │
│  │    - 最终排序                                                  │  │
│  │                                                               │  │
│  │  输出：排序后的 Top 候选                                        │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       输出：完整 TCR 序列                            │
│                   (Scaffold + CDR3β = 完整链)                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 为什么不直接生成骨架？

我们最初考虑过 **"Model 2 直接生成 V/J 骨架"** 的方案，但存在根本困难：

| 问题 | 说明 |
|------|------|
| **离散类别** | V/J 基因是离散的类别（如 TRBV19*01），不是可插值的连续空间 |
| **组合爆炸** | Vβ ~50+ × Jβ ~13 × Vα ~50+ × Jα ~60+ = 几十万种组合 |
| **数据稀疏** | 很多组合在 20 万数据中可能只出现几次甚至零次 |
| **语义模糊** | 生成的"基因名"难以保证对应有效的蛋白质序列 |

**Retrieve & Generate 的优势**：
- 把"生成"问题转为"检索"问题，降低了难度
- 只从数据库中已有的 V/J 组合中选择，保证有效性
- 检索过程有明确的物理意义：找 MHC 兼容的骨架

---

## 3. 模块架构详解

### 3.1 Immuno-PLM (ESM-2 + LoRA + 拓扑偏置)

**角色**: 编码 TCR 和 pMHC 序列，支持骨架检索和条件生成。

**核心设计**: 我们不再冻结 ESM-2，而是通过 **LoRA (Low-Rank Adaptation)** 让它学习 TCR-pMHC 的特异性规律。

#### 3.1.1 架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Immuno-PLM 架构                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input Tokens                                                       │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │         ESM-2 (esm2_t33_650M_UR50D) + LoRA                  │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │  Self-Attention Layer                                   │ │   │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐ │ │   │
│  │  │  │ Q_proj  │  │ K_proj  │  │ V_proj  │  │  Out_proj   │ │ │   │
│  │  │  │ + LoRA  │  │ + LoRA  │  │ + LoRA  │  │   + LoRA    │ │ │   │
│  │  │  └─────────┘  └─────────┘  └─────────┘  └─────────────┘ │ │   │
│  │  └─────────────────────────────────────────────────────────┘ │   │
│  │  × 33 layers                                                 │   │
│  └──────────────────────────────┬──────────────────────────────┘   │
│                                 │                                   │
│                                 ▼                                   │
│                    Sequence Features [B, L, 1280]                   │
│                                 │                                   │
│                    ┌────────────┴────────────┐                      │
│                    │                         │                      │
│                    ▼                         ▼                      │
│          ┌─────────────────┐    ┌──────────────────────────┐       │
│          │   seq_proj      │    │   TopologyBias           │       │
│          │ [1280 → 256]    │    │   (from psi_model)       │       │
│          └────────┬────────┘    │   - 7-level hierarchy    │       │
│                   │             │   - pair_embed_lvl1/2    │       │
│                   │             └───────────┬──────────────┘       │
│                   │                         │                      │
│                   │    ┌────────────────────┘                      │
│                   │    │ pair_fusion [128 → 256]                   │
│                   │    ▼                                           │
│                   └──► + ◄──────────────────────────────────       │
│                        │                                           │
│                        ▼                                           │
│              Fused Features [B, L, 256]                            │
│                        │                                           │
│                        ▼                                           │
│              ┌─────────────────┐                                   │
│              │ Masked Pooling  │                                   │
│              │  + LayerNorm    │                                   │
│              └────────┬────────┘                                   │
│                       │                                            │
│                       ▼                                            │
│              Pooled [B, 256] ──► contrastive_head ──► [B, 256]    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 3.1.2 LoRA 配置详解

> **注意**: 使用内置 LoRA 实现（`immuno_plm.py` 中的 `LoRALinear`），无需 peft 依赖。

```python
# 内置 LoRA 实现 (immuno_plm.py)
class LoRALinear(nn.Module):
    """Lightweight LoRA for Linear layers: frozen base + trainable low-rank update."""
    
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        
        self.rank = rank
        self.scaling = alpha / rank
        in_dim = base.in_features
        out_dim = base.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora_out

# 使用方式 (ImmunoPLM.__init__)
if self.use_lora:
    inject_lora_linear(self.esm_model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
    # 输出: LoRA injected. Trainable params: X / Y (Z%)
        
        # 5. 输出层
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
```

#### 3.1.3 LoRA 优势

| 对比项 | 冻结 ESM-2 | 全量微调 | ESM-2 + LoRA |
|--------|-----------|----------|--------------|
| **可训练参数** | 0 | 650M | ~2.4M (0.36%) |
| **显存占用** | 低 | 极高 (>40GB) | 低 (~12GB) |
| **训练速度** | 快 | 慢 | 快 |
| **领域适应能力** | ❌ 无 | ✅ 最强 | ✅ 接近全量 |
| **灾难性遗忘** | ❌ 无 | ⚠️ 风险 | ✅ 安全 |

**结论**: LoRA 是最佳选择 — 以 0.36% 的参数成本，获得接近全量微调的性能。

#### 3.1.2 TopologyBias (层级成对嵌入)

来自 `psi_model/model.py` 的核心设计，7 层交互编码：

```
Level 0: Collapse Token 自指
Level 1: Collapse ↔ 所有区域 (观察者-被观察关系)
Level 2: HD (CDR3) 内部序列邻近关系
Level 3: HD 内部非邻近关系 (长程依赖)
Level 4: HD ↔ 条件区域 (CDR3 与 peptide/MHC 的交互) ← 关键！
Level 5+: 条件区域内部
Level N+: 条件区域之间
```

#### 3.1.3 训练目标

**确认的 Pipeline (2025-11-29)**：

```
A. 训练:
   - 5 次编码 (共享 Encoder): pMHC, HV, HJ, LV, LJ
   - 4 个 InfoNCE loss (主): pMHC ↔ HV/HJ/LV/LJ
   - 4 个 Classification loss (辅): 预测 Gene Name

B. 建库 (训练后执行一次):
   - 收集所有唯一的 V/J 序列
   - 编码后存入 Bank (HV ~65, HJ ~14, LV ~45, LJ ~61)

C. 推理:
   - query = Encoder(pMHC)
   - score = query @ Bank.T → argmax → 最佳 V/J 序列
```

**Loss Function**:

```python
# 4 并行 InfoNCE (主 loss)
loss_nce = InfoNCE(z_pmhc, z_hv) + InfoNCE(z_pmhc, z_hj) + \
           InfoNCE(z_pmhc, z_lv) + InfoNCE(z_pmhc, z_lj)

# 4 分类 loss (辅助)
loss_cls = CrossEntropy(logits_hv, hv_id) + ...

# 总 loss
loss = loss_nce + 0.2 * loss_cls
```

**为什么 Batch Random 是安全的？**
- Batch 内的其他样本是**真正不同的** pMHC-V/J 配对
- 不需要显式采样"负样本"，避免误标记
- 只要 Batch Size 够大（32+），效果就很好

### 3.2 FlowTCR-Gen (流匹配生成器)

**角色**: 给定 pMHC + Scaffold，生成 CDR3β 序列。

#### 3.2.1 架构设计

```python
class FlowMatchingModel(nn.Module):
    """
    离散流匹配 (Discrete Flow Matching) for CDR3β 生成
    
    流匹配设置：
    - 基分布 x_0: 氨基酸上的均匀分布
    - 目标分布 y: one-hot 真实序列
    - 插值: x_t = (1-t) * x_0 + t * y
    - 向量场目标: v* = y - x_0
    - 损失: ||v_θ(x_t, t, cond) - v*||²
    """
    
    def __init__(self, vocab_size=21, hidden_dim=256, n_layers=6):
        super().__init__()
        
        # 条件编码器
        self.condition_encoder = nn.Linear(condition_dim, hidden_dim)
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Transformer 主干
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8),
            num_layers=n_layers
        )
        
        # 向量场预测头
        self.vector_field_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x_t, t, condition):
        """
        x_t: [B, L, vocab_size] - 插值状态
        t: [B, 1] - 时间步
        condition: [B, cond_dim] - pMHC + Scaffold 嵌入
        
        返回: [B, L, vocab_size] - 预测的向量场
        """
        # 编码条件
        cond_emb = self.condition_encoder(condition)  # [B, D]
        
        # 时间嵌入
        time_emb = self.time_embed(t)  # [B, D]
        
        # 融合
        h = x_t  # 简化，实际需要投影
        h = h + cond_emb.unsqueeze(1) + time_emb.unsqueeze(1)
        
        # Transformer
        h = self.transformer(h)
        
        # 向量场预测
        v_pred = self.vector_field_head(h)
        
        return v_pred
```

#### 3.2.2 条件输入

| 条件 | 来源 | 维度 | 必需 |
|------|------|------|------|
| pMHC 嵌入 | Immuno-PLM.encode_pmhc() | [D] | ✅ 是 |
| Scaffold 嵌入 | Immuno-PLM.encode_scaffold() | [D] | ✅ 是 |
| TM-align PSSM | 外部工具 | [L, 20] | ⚠️ 可选 |
| 几何特征 | TCRFold-Light | [D_geo] | ⚠️ 可选 |

#### 3.2.3 损失函数

```python
def flow_matching_loss(model, x_0, y, condition):
    """
    x_0: [B, L, vocab] - 均匀噪声
    y: [B, L, vocab] - one-hot 目标
    condition: [B, cond_dim] - 条件嵌入
    """
    # 采样时间步
    t = torch.rand(x_0.size(0), 1, 1, device=x_0.device)
    
    # 插值
    x_t = (1 - t) * x_0 + t * y
    
    # 目标向量场
    v_target = y - x_0
    
    # 预测向量场
    v_pred = model(x_t, t.squeeze(-1), condition)
    
    # MSE 损失
    loss = F.mse_loss(v_pred, v_target)
    
    return loss
```

### 3.3 TCRFold-Light (结构评估器)

**角色**: 预测结构特征，过滤不合理候选。

#### 3.3.1 架构设计

```python
class TCRFoldLight(nn.Module):
    """
    MSA-free Evoformer-lite
    
    来源: conditioned/src/Evoformer.py (保留 Triangle 模块，移除 MSA)
    
    输出:
    - distance: [B, L, L, n_bins] - 距离分布
    - contact: [B, L, L, 1] - 接触概率
    - energy: [B, 1] - 能量代理 (EvoEF2 监督)
    """
    
    def __init__(self, s_dim=512, z_dim=128, n_layers=12):
        super().__init__()
        
        # Evoformer 块 (无 MSA)
        self.evoformer = nn.ModuleList([
            EvoformerBlock(s_dim, z_dim) for _ in range(n_layers)
        ])
        
        # 预测头
        self.distance_head = nn.Linear(z_dim, 64)  # 64 bins
        self.contact_head = nn.Linear(z_dim, 1)
        self.energy_head = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
```

#### 3.3.2 损失函数

```python
def compute_physics_loss(pred, target, interface_mask, interface_weight=10.0):
    """
    物理损失函数（参照 USER_MANUAL）
    
    关键：界面残基权重 ×10
    """
    # 距离损失
    loss_dist = F.mse_loss(pred['distance'], target['distance'])
    
    # 接触损失 (界面加权)
    loss_contact_interface = F.binary_cross_entropy(
        pred['contact'] * interface_mask,
        target['contact'] * interface_mask
    )
    loss_contact_other = F.binary_cross_entropy(
        pred['contact'] * (1 - interface_mask),
        target['contact'] * (1 - interface_mask)
    )
    loss_contact = interface_weight * loss_contact_interface + loss_contact_other
    
    # 能量损失 (EvoEF2 监督)
    loss_energy = F.mse_loss(pred['energy'], target['energy'])
    
    return loss_dist + loss_contact + loss_energy
```

---

## 4. 数据基础设施

### 4.1 数据源

| 数据集 | 规模 | 字段 | 用途 |
|--------|------|------|------|
| **trn.csv** (Paired) | 20万+ | peptide, mhc, cdr3_b, h_v, h_j, l_v, l_j | Scaffold Bank, Immuno-PLM, FlowTCR-Gen |
| **TCRdb** | 大规模 | cdr3_b only | (可选) Flow 预训练 |
| **STCRDab / TCR3d** | ~500 | PDB 结构 | TCRFold-Light 训练 |

### 4.2 字段定义

```
必需字段:
├── peptide    : 抗原肽段序列 (8-15 aa)
├── mhc        : MHC 等位基因名或序列
└── cdr3_b     : CDR3β 序列 (生成目标)

可选字段:
├── h_v        : β 链 V 基因
├── h_j        : β 链 J 基因
├── l_v        : α 链 V 基因
├── l_j        : α 链 J 基因
└── cdr3_a     : CDR3α 序列
```

### 4.3 Scaffold Bank 构建

```python
import pandas as pd

# 读取配对数据
df = pd.read_csv("data/trn.csv")

# 提取唯一的 V/J 组合
scaffold_bank = df.groupby(['h_v', 'h_j', 'l_v', 'l_j']).agg({
    'peptide': 'first',      # 代表性 peptide
    'mhc': 'first',          # 代表性 MHC
    'cdr3_b': ['count', 'first']  # 出现次数和示例
}).reset_index()

scaffold_bank.columns = ['h_v', 'h_j', 'l_v', 'l_j', 
                         'rep_peptide', 'rep_mhc', 
                         'count', 'rep_cdr3b']

# 过滤低频组合
scaffold_bank = scaffold_bank[scaffold_bank['count'] >= 5]

scaffold_bank.to_csv("data/scaffold_bank.csv", index=False)
print(f"唯一骨架数: {len(scaffold_bank)}")
```

### 4.4 负样本策略

| 策略 | 描述 | 安全性 | 推荐 |
|------|------|--------|------|
| **Batch Random** | Batch 内其他样本作为负样本 | ✅ 绝对安全 | ✅ 默认 |
| **Peptide Decoy** | 同 MHC，相似 peptide (60-90% identity) | ⚠️ 中等 | 可选 |
| **CDR3 Mutant** | 同 pMHC，CDR3 2-3 点突变 | ⚠️ 中等 | 可选 |
| **Synthetic** | 突变锚点位置为相反电荷 | ✅ 安全 | 可选 |

---

## 5. 训练策略

### 5.1 Immuno-PLM 训练

#### 5.1.1 目标

学习 TCR-pMHC 兼容性，支持骨架检索。

#### 5.1.2 命令

```bash
# 基础模式 (快速调试)
python -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval \
    --data flowtcr_fold/data/trn.jsonl \
    --epochs 100 --batch_size 32

# ESM-2 + LoRA 模式 (生产环境)
python -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval \
    --data flowtcr_fold/data/trn.jsonl \
    --use_esm --use_lora --lora_rank 8 \
    --epochs 100 --batch_size 16 \
    --cls_weight 0.2
```

#### 5.1.3 训练要点

- **双塔架构**: pMHC 和 V/J 序列分开编码，共享 Encoder
- **4 并行 InfoNCE**: pMHC ↔ HV, pMHC ↔ HJ, pMHC ↔ LV, pMHC ↔ LJ
- **辅助分类 Loss**: 预测 Gene Name，加速收敛
- **Batch Size**: 尽量大（32+），提供足够的 batch 内负样本
- **Temperature (tau)**: 0.07，控制对比学习的锐度
- **Classification Weight**: 0.2，辅助监督信号

#### 5.1.4 验证指标

```python
def evaluate_retrieval(model, test_data, scaffold_bank, k=10):
    """
    评估检索能力：给定 pMHC，能否检索到正确的 scaffold？
    """
    correct = 0
    for sample in test_data:
        pmhc_emb = model.encode_pmhc(sample['pmhc'])
        scaffold_embs = model.encode_scaffolds(scaffold_bank)
        
        similarities = pmhc_emb @ scaffold_embs.T
        top_k = similarities.topk(k).indices
        
        if sample['true_scaffold_id'] in top_k:
            correct += 1
    
    recall_at_k = correct / len(test_data)
    return recall_at_k
```

### 5.2 FlowTCR-Gen 训练

#### 5.2.1 目标

学习条件生成 CDR3β。

#### 5.2.2 命令

```bash
python flowtcr_fold/FlowTCR_Gen/train_flow.py \
    --data data/trn.csv \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --out_dir checkpoints/flow
```

#### 5.2.3 训练要点

- **条件编码**: 使用预训练的 Immuno-PLM 编码 pMHC 和 Scaffold
- **冻结编码器**: 初期可冻结 Immuno-PLM，只训练 Flow 网络
- **时间采样**: 均匀采样 t ∈ [0, 1]

#### 5.2.4 验证指标

```python
def evaluate_generation(model, test_data):
    """
    评估生成质量
    """
    metrics = {
        'perplexity': [],      # 序列困惑度
        'length_match': [],    # 长度匹配率
        'motif_recall': [],    # 已知 motif 召回率
    }
    
    for sample in test_data:
        generated = model.sample(sample['condition'])
        # 计算各项指标
        ...
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

### 5.3 TCRFold-Light 训练

#### 5.3.1 训练阶段

| 阶段 | 数据 | 目标 |
|------|------|------|
| **Phase 1** | 通用 PDB (PPI) | 学习蛋白质接触预测 |
| **Phase 2** | TCR-pMHC (STCRDab) | 适应 TCR 特异性 |

#### 5.3.2 命令

```bash
# Phase 1: PPI 预训练
python flowtcr_fold/TCRFold_Light/train_ppi_impl.py \
    --pdb_dir data/pdb_ppi \
    --epochs 50

# Phase 2: TCR 微调
python flowtcr_fold/TCRFold_Light/train_with_energy.py \
    --pdb_dir data/pdb_structures \
    --epochs 100 \
    --interface_weight 10.0
```

### 5.4 训练偏好

| 设置 | 值 | 位置 |
|------|-----|------|
| Checkpoint 频率 | 每 50 epochs | `common/utils.py` |
| 早停耐心 | 100 epochs 无提升 | `common/utils.py` |
| 梯度裁剪 | max_norm=1.0 | 训练脚本 |

---

## 6. 推理流程

### 6.1 完整代码示例

```python
import torch
from flowtcr_fold.Immuno_PLM import ImmunoPLM
from flowtcr_fold.FlowTCR_Gen import FlowMatchingModel
from flowtcr_fold.TCRFold_Light import TCRFoldLight
from flowtcr_fold.physics import TCRStructureOptimizer

class TCRDesignPipeline:
    def __init__(self):
        # 加载模型
        self.plm = ImmunoPLM.load("checkpoints/plm/immuno_plm.pt")
        self.flow = FlowMatchingModel.load("checkpoints/flow/flow_gen.pt")
        self.critic = TCRFoldLight.load("checkpoints/tcrfold/tcrfold_light.pt")
        self.optimizer = TCRStructureOptimizer()
        
        # 加载 Scaffold Bank
        self.scaffold_bank = self.load_scaffold_bank("data/scaffold_bank.csv")
        self.scaffold_embs = self.precompute_scaffold_embeddings()
    
    def design(self, peptide, mhc, top_k_scaffolds=10, samples_per_scaffold=100):
        """
        完整设计流程
        
        Args:
            peptide: 目标肽段序列
            mhc: MHC 等位基因
            top_k_scaffolds: 检索的骨架数量
            samples_per_scaffold: 每个骨架生成的 CDR3β 数量
        
        Returns:
            排序后的 TCR 设计列表
        """
        # ==========================
        # Stage 1: Scaffold Retrieval
        # ==========================
        pmhc_emb = self.plm.encode_pmhc(peptide, mhc)
        
        similarities = pmhc_emb @ self.scaffold_embs.T
        top_indices = similarities.topk(top_k_scaffolds).indices
        top_scaffolds = [self.scaffold_bank[i] for i in top_indices]
        
        print(f"Stage 1: 检索到 {len(top_scaffolds)} 个骨架")
        
        # ==========================
        # Stage 2: CDR3β Generation
        # ==========================
        candidates = []
        
        for scaffold in top_scaffolds:
            scaffold_emb = self.plm.encode_scaffold(scaffold)
            condition = torch.cat([pmhc_emb, scaffold_emb], dim=-1)
            
            for _ in range(samples_per_scaffold):
                cdr3b = self.flow.sample(condition)
                candidates.append({
                    'scaffold': scaffold,
                    'cdr3b': cdr3b,
                    'pmhc_emb': pmhc_emb,
                    'scaffold_emb': scaffold_emb
                })
        
        print(f"Stage 2: 生成 {len(candidates)} 个候选")
        
        # ==========================
        # Stage 3: Structure Critique
        # ==========================
        scored_candidates = []
        
        for cand in candidates:
            # TCRFold-Light 评分
            score = self.critic.score(cand['scaffold'], cand['cdr3b'])
            cand['tcrfold_score'] = score['confidence']
            cand['contact_density'] = score['contact_density']
            cand['energy_pred'] = score['energy']
            
            # 过滤低质量候选
            if score['confidence'] > 0.7:
                scored_candidates.append(cand)
        
        print(f"Stage 3: 筛选后剩余 {len(scored_candidates)} 个候选")
        
        # ==========================
        # (Optional) EvoEF2 精修
        # ==========================
        if len(scored_candidates) > 0:
            top_candidates = sorted(
                scored_candidates, 
                key=lambda x: x['tcrfold_score'], 
                reverse=True
            )[:100]
            
            for cand in top_candidates:
                try:
                    energy = self.optimizer.compute_binding_energy(cand)
                    cand['evoef2_energy'] = energy
                except:
                    cand['evoef2_energy'] = float('inf')
            
            # 最终排序 (能量越低越好)
            top_candidates.sort(key=lambda x: x['evoef2_energy'])
        else:
            top_candidates = scored_candidates
        
        return top_candidates

# 使用示例
pipeline = TCRDesignPipeline()
designs = pipeline.design(
    peptide="GILGFVFTL",
    mhc="HLA-A*02:01",
    top_k_scaffolds=10,
    samples_per_scaffold=100
)

print(f"Top 设计:")
for i, design in enumerate(designs[:5]):
    print(f"  {i+1}. Scaffold: {design['scaffold']['h_v']}-{design['scaffold']['h_j']}")
    print(f"      CDR3β: {design['cdr3b']}")
    print(f"      Energy: {design.get('evoef2_energy', 'N/A')}")
```

---

## 7. 实施路线图

### 7.1 Phase 1: Scaffold Retrieval 验证 (Week 1-2)

| 任务 | 状态 | 优先级 |
|------|------|--------|
| 实现 train_scaffold_retrieval.py | ✅ 完成 | 🔴 P0 |
| 4 并行 InfoNCE + Classification | ✅ 完成 | 🔴 P0 |
| 构建 Scaffold Bank (build_bank) | ✅ 完成 | 🔴 P0 |
| 实现检索逻辑 (retrieve) | ✅ 完成 | 🔴 P0 |
| 评估 Recall@10 | ⬜ 待做 | 🔴 P0 |

**里程碑**: Recall@10 > 50%

### 7.2 Phase 2: FlowTCR-Gen 开发 (Week 3-4)

| 任务 | 状态 | 优先级 |
|------|------|--------|
| 实现条件编码 | ⬜ 待做 | 🔴 P0 |
| 训练 Flow 模型 | ⬜ 待做 | 🔴 P0 |
| 评估生成质量 | ⬜ 待做 | 🟡 P1 |

**里程碑**: 生成序列长度分布合理，perplexity < 10

### 7.3 Phase 3: Pipeline 集成 (Week 5-6)

| 任务 | 状态 | 优先级 |
|------|------|--------|
| 端到端 Pipeline | ⬜ 待做 | 🟡 P1 |
| TCRFold-Light 评分 | ⬜ 待做 | 🟡 P1 |
| EvoEF2 精修 | ⬜ 待做 | 🟡 P1 |

**里程碑**: 完整 Pipeline 可运行

### 7.4 Phase 4: 评估与优化 (Week 7-8)

| 任务 | 状态 | 优先级 |
|------|------|--------|
| Baseline 对比 | ⬜ 待做 | 🟢 P2 |
| 消融实验 | ⬜ 待做 | 🟢 P2 |
| 超参数调优 | ⬜ 待做 | 🟢 P2 |

---

## 8. 关键技术决策

### 8.1 ESM-2 vs BasicTokenizer

| 方案 | 优点 | 缺点 | 推荐场景 |
|------|------|------|---------|
| **BasicTokenizer** | 快速、轻量、易调试 | 无预训练知识 | 快速验证 |
| **ESM-2 (冻结)** | 通用蛋白质知识 | 显存大、推理慢 | 最终版本 |
| **ESM-2 + LoRA** | 领域适应 | 实现复杂 | 高性能需求 |

**建议**: 先用 BasicTokenizer 验证，确认后再升级到 ESM-2。

### 8.2 负样本策略

**结论**: 使用 **Batch Random** 作为默认策略，避免 False Negative 风险。

### 8.3 生成方法

| 方法 | 优点 | 缺点 |
|------|------|------|
| **Autoregressive** | 成熟、稳定 | 顺序依赖、慢 |
| **Flow Matching** | 全局视野、并行 | 实现复杂 |
| **Diffusion** | 生成质量高 | 采样慢 |

**选择**: Flow Matching（全局视野适合 CDR3 的结构约束）

---

## 9. 风险评估与应对

### 9.1 技术风险

| 风险 | 影响 | 应对 |
|------|------|------|
| InfoNCE 不收敛 | 🔴 高 | 增大 Batch Size，调整 tau |
| Flow 生成质量差 | 🔴 高 | 添加更多条件信息 |
| 检索效果不佳 | 🟡 中 | 增加训练数据、调整嵌入维度 |
| EvoEF2 速度慢 | 🟢 低 | 批处理、缓存 |

### 9.2 数据风险

| 风险 | 影响 | 应对 |
|------|------|------|
| V/J 组合覆盖不全 | 🟡 中 | 过滤低频组合，使用平滑 |
| 负样本污染 | 🔴 高 | 坚持 Batch Random 策略 |
| 结构数据不足 | 🟡 中 | 使用 AlphaFold 预测作为补充 |

---

## 附录

### A. 命令速查

```bash
# 数据准备
python flowtcr_fold/data/convert_csv_to_jsonl.py --input data/trn.csv --output data/trn.jsonl

# Scaffold Retrieval 训练 (Step 1)
python -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval \
    --data flowtcr_fold/data/trn.jsonl \
    --use_esm --use_lora --lora_rank 8 \
    --epochs 100 --batch_size 16

# FlowTCR-Gen 训练 (Step 2)
python flowtcr_fold/FlowTCR_Gen/train_flow.py --data data/trn.csv --epochs 100

# TCRFold-Light 训练 (Step 3, Optional)
python flowtcr_fold/TCRFold_Light/train_with_energy.py --pdb_dir data/pdb_structures

# 推理
python flowtcr_fold/FlowTCR_Gen/pipeline_impl.py --peptide "GILGFVFTL" --mhc "HLA-A*02:01"
```

### B. 相关文献

- **InfoNCE**: Oord et al., "Representation Learning with Contrastive Predictive Coding" (2018)
- **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
- **ESM-2**: Lin et al., "Evolutionary-scale prediction of atomic-level protein structure" (Science 2023)
- **EvoEF2**: Huang et al., Bioinformatics (2020)

---

**文档版本**: 2.0  
**最后更新**: 2025-11-28  
**维护者**: FlowTCR-Fold Team

