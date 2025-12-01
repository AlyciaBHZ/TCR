# Stage 3: TCRFold-Prophet Implementation Plan

> **Master Reference**: [../README.md](../README.md) (Section 4.3, Master Plan v3.1 Stage 3)
> 
> **Status**: 🔄 In Progress (30%)
> 
> **Timeline**: Week 6-10 (Plan v3.1)

---

## 1. 模块定位

### 1.1 在整体 Pipeline 中的角色

```
                    Stage 1: Immuno-PLM
                              │
                              ▼
                    Stage 2: FlowTCR-Gen
                              │
                              ▼
                    CDR3β sequence candidates
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ★ Stage 3: TCRFOLD-PROPHET (You Are Here)                     │
│  ─────────────────────────────────────────                      │
│  S_ψ: Structure predictor + E_φ: Energy surrogate              │
│  Output: Physically validated + ranked TCR candidates           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Final TCR designs
```

### 1.2 核心目标

- **结构验证**：预测 TCR-pMHC 复合物的结构特征（距离、接触）
- **能量评估**：E_φ surrogate 近似 EvoEF2 物理能量
- **候选筛选**：过滤高能量/结构异常的候选
- **MC 优化**：（可选）基于 E_φ 的 Monte Carlo 序列优化

### 1.3 Scope Tiers

| Tier | 组件 | 论文状态 |
|------|------|----------|
| 🔴 **Must Have** | S_ψ (Structure Predictor) | Required |
| 🔴 **Must Have** | E_φ (Energy Surrogate) | Required |
| 🔴 **Must Have** | Post-hoc Screening | Required |
| 🟡 **Should Have** | Offline MC Refinement | Strongly Recommended |
| 🟢 **Exploratory** | Gradient Guidance in Flow ODE | Optional |
| 🟢 **Exploratory** | MC-to-Training Loop | Future Work |

---

## 2. 当前实现状态

### 2.1 已完成 ✅

| 文件 | 功能 | 状态 |
|------|------|------|
| `tcrfold_light.py` | TCRFoldLight 基础类 | ⚠️ 需升级为 Prophet |
| `train_with_energy.py` | 能量监督训练 | ⚠️ 需适配 3-Phase |
| `train_ppi_impl.py` | PPI 预训练脚本 | ✅ 骨架完成 |
| `../physics/evoef_runner.py` | EvoEF2 Python 包装 | ✅ 可用 |

### 2.2 待实现 🔄

| 任务 | 优先级 | 依赖 |
|------|--------|------|
| PDB 数据下载和处理 | 🔴 高 | - |
| EvoEF2 批处理脚本 | 🔴 高 | - |
| Phase 3A: PPI 结构预训练 | 🔴 高 | PDB 数据 |
| Phase 3B: 能量 surrogate 训练 | 🔴 高 | 3A checkpoint |
| Phase 3C: TCR 微调 | 🔴 高 | 3B checkpoint |
| MC Refinement 集成 | 🟡 中 | E_φ 完成 |
| 与 Stage 2 集成 | 🟡 中 | Stage 2 完成 |

### 2.3 资源需求

| Phase | 数据量 | 训练时间 | GPU 内存 |
|-------|--------|----------|----------|
| 3A | ~50k PPI | 3-7 天 @4×A100 | ~40 GB |
| 3B | 同上 + EvoEF2 | 1-2 天 | ~20 GB |
| 3C | ~1k TCR | 几小时 | ~16 GB |

---

## 3. Step-by-Step Implementation Plan

### Phase 0: 数据准备 (Day 1-5)

#### Step 0.1: PDB 数据下载

```bash
# 创建数据目录
mkdir -p data/pdb_structures/raw
mkdir -p data/pdb_structures/processed

# 下载 PPI 结构 (约 50k)
# 方法 1: 使用 PDB REST API
python scripts/download_pdb.py \
    --query "complex AND protein-protein" \
    --max_count 50000 \
    --output_dir data/pdb_structures/raw

# 方法 2: 使用预编译列表
wget https://files.rcsb.org/download/<pdb_id>.pdb
```

#### Step 0.2: 结构预处理

```python
# scripts/preprocess_pdb.py
def preprocess_pdb(pdb_path: str, output_dir: str):
    """
    1. 提取链信息
    2. 清理非标准残基
    3. 提取接口残基
    4. 计算接触图
    """
    structure = PDBParser().get_structure('complex', pdb_path)
    
    for model in structure:
        chains = list(model.get_chains())
        if len(chains) < 2:
            continue  # 跳过单链
        
        # 提取序列和坐标
        seq_a = extract_sequence(chains[0])
        seq_b = extract_sequence(chains[1])
        coords_a = extract_coords(chains[0])
        coords_b = extract_coords(chains[1])
        
        # 计算接触图
        contact_map = compute_contact_map(coords_a, coords_b, threshold=8.0)
        
        # 保存处理后的数据
        save_processed(output_dir, pdb_id, seq_a, seq_b, coords_a, coords_b, contact_map)
```

#### Step 0.3: EvoEF2 批处理

```python
# scripts/batch_evoef2.py
from flowtcr_fold.physics.evoef_runner import EvoEFRunner

def batch_compute_energy(pdb_dir: str, output_cache: str):
    """
    对所有 PDB 计算 EvoEF2 能量
    """
    runner = EvoEFRunner()
    
    for pdb_file in glob(f"{pdb_dir}/*.pdb"):
        try:
            # 修复结构
            repaired = runner.repair_structure(pdb_file)
            
            # 计算 binding energy
            result = runner.compute_binding_energy(repaired)
            
            # 缓存结果
            save_to_cache(output_cache, pdb_file, result.total_energy)
        except Exception as e:
            log_error(pdb_file, e)
```

#### Step 0.4: TCR 数据准备

```bash
# 下载 TCR3d / STCRDab 数据
wget https://tcr3d.ibbr.umd.edu/downloads/structures.tar.gz
tar -xzf structures.tar.gz -C data/tcr_structures/

# 处理 TCR-pMHC 结构
python scripts/preprocess_tcr.py \
    --input_dir data/tcr_structures/ \
    --output_dir data/tcr_processed/ \
    --compute_evoef2
```

---

### Phase 3A: PPI 结构预训练 (Day 6-12)

#### Step 3A.1: 升级 TCRFoldLight → TCRFoldProphet

```python
# flowtcr_fold/TCRFold_Light/tcrfold_prophet.py

import torch
from torch import nn
from conditioned.src.Evoformer import Evoformer

class TCRFoldProphet(nn.Module):
    """
    TCRFold-Prophet: Evoformer-Single + IPA + Energy Head
    """
    def __init__(
        self,
        s_dim: int = 512,
        z_dim: int = 128,
        n_layers: int = 12,
        n_heads: int = 8,
    ):
        super().__init__()
        
        # 序列编码器 (可选 ESM-2)
        self.seq_encoder = nn.Embedding(21, s_dim)
        self.chain_type_embed = nn.Embedding(4, s_dim)  # TCRα, TCRβ, peptide, MHC
        
        # Pair 初始化
        self.pair_init = nn.Sequential(
            nn.Linear(s_dim * 2 + 64, z_dim),  # outer product + relpos
            nn.ReLU(),
        )
        
        # Evoformer trunk
        self.trunk = Evoformer(s_dim, z_dim, N_elayers=n_layers)
        
        # Structure head (IPA-like)
        self.struct_head = StructureHead(s_dim, z_dim)
        
        # Energy head
        self.energy_head = EnergyHead(z_dim)
        
        # Distance/Contact heads
        self.dist_head = nn.Linear(z_dim, 64)  # 64 distance bins
        self.contact_head = nn.Linear(z_dim, 1)
    
    def forward(self, seq_tokens, chain_types, pair_init=None):
        """
        Args:
            seq_tokens: [B, L] 序列 tokens
            chain_types: [B, L] 链类型 (0=TCRα, 1=TCRβ, 2=pep, 3=MHC)
            pair_init: [B, L, L, z_dim] 可选的 pair 初始化
        """
        B, L = seq_tokens.shape
        
        # Sequence embedding
        s = self.seq_encoder(seq_tokens) + self.chain_type_embed(chain_types)
        
        # Pair initialization
        if pair_init is None:
            s_i = s.unsqueeze(2).expand(-1, -1, L, -1)
            s_j = s.unsqueeze(1).expand(-1, L, -1, -1)
            relpos = self.relpos_embed(L).to(s.device)
            z = self.pair_init(torch.cat([s_i, s_j, relpos], dim=-1))
        else:
            z = pair_init
        
        # Evoformer
        s, z = self.trunk(s, z)
        
        # Outputs
        outputs = {
            's': s,
            'z': z,
            'dist_logits': self.dist_head(z),
            'contact_logits': self.contact_head(z).squeeze(-1),
            'energy': self.energy_head(z),
        }
        
        return outputs


class EnergyHead(nn.Module):
    """E_φ: 从 pair representation 预测能量"""
    def __init__(self, z_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim // 2),
            nn.ReLU(),
            nn.Linear(z_dim // 2, 1),
        )
    
    def forward(self, z):
        # Global pooling over pair representation
        z_pool = z.mean(dim=(1, 2))  # [B, z_dim]
        return self.mlp(z_pool).squeeze(-1)  # [B]
```

#### Step 3A.2: 结构预训练 Loss

```python
def compute_structure_loss(pred, target, interface_mask=None):
    """
    Phase 3A Loss: FAPE + Distance + Contact
    """
    # Distance loss (cross-entropy over bins)
    loss_dist = F.cross_entropy(
        pred['dist_logits'].reshape(-1, 64),
        target['dist_bins'].reshape(-1),
    )
    
    # Contact loss (binary cross-entropy)
    loss_contact = F.binary_cross_entropy_with_logits(
        pred['contact_logits'],
        target['contact_map'],
    )
    
    # Interface 加权
    if interface_mask is not None:
        loss_contact = (loss_contact * (1 + 9 * interface_mask)).mean()
    
    # (可选) FAPE loss - 如果有坐标预测
    # loss_fape = compute_fape(pred['coords'], target['coords'])
    
    return loss_dist + 0.3 * loss_contact
```

#### Step 3A.3: 训练脚本

```python
# flowtcr_fold/TCRFold_Light/train_ppi_impl.py

def train_phase_3a(config):
    """Phase 3A: General PPI structure pretraining"""
    
    # 数据
    train_dataset = PPIDataset(config.pdb_dir, split='train')
    val_dataset = PPIDataset(config.pdb_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # 模型
    model = TCRFoldProphet(
        s_dim=512,
        z_dim=128,
        n_layers=12,
    ).cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            pred = model(batch['seq_tokens'], batch['chain_types'])
            loss = compute_structure_loss(pred, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['seq_tokens'], batch['chain_types'])
                loss = compute_structure_loss(pred, batch)
                val_loss += loss.item()
        
        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{config.out_dir}/best.pt")
        
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
```

---

### Phase 3B: 能量 Surrogate 训练 (Day 13-15)

#### Step 3B.1: 加载 Phase 3A checkpoint

```python
def train_phase_3b(config):
    """Phase 3B: Energy surrogate fitting"""
    
    # 加载 3A 预训练
    model = TCRFoldProphet(...).cuda()
    model.load_state_dict(torch.load(f"{config.phase_a_ckpt}/best.pt"))
    
    # 冻结大部分 trunk，只训练最后几层 + energy head
    for name, param in model.named_parameters():
        if 'trunk' in name and 'layers.10' not in name and 'layers.11' not in name:
            param.requires_grad = False
    
    # 确保 energy head 可训练
    for param in model.energy_head.parameters():
        param.requires_grad = True
```

#### Step 3B.2: 能量 Loss

```python
def compute_energy_loss(pred, target):
    """
    Phase 3B Loss: MSE between E_φ and EvoEF2
    """
    loss_energy = F.mse_loss(pred['energy'], target['evoef2_energy'])
    
    # (可选) 添加 ranking loss
    # 确保 E_φ 能正确排序高能量 vs 低能量结构
    
    return loss_energy
```

#### Step 3B.3: 数据集增强（可选）

```python
class EnergyDataset(PPIDataset):
    """
    增加 decoy 结构用于能量训练
    """
    def __init__(self, pdb_dir, energy_cache, use_decoys=True):
        super().__init__(pdb_dir)
        self.energy_cache = load_energy_cache(energy_cache)
        self.use_decoys = use_decoys
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample['evoef2_energy'] = self.energy_cache[sample['pdb_id']]
        
        if self.use_decoys and torch.rand(1).item() < 0.3:
            # 30% 概率生成 decoy
            sample = self.generate_decoy(sample)
        
        return sample
    
    def generate_decoy(self, sample):
        """
        生成 decoy 结构:
        1. 坐标加噪声
        2. 接口局部旋转/平移
        3. 随机突变
        """
        # 加坐标噪声
        noise = torch.randn_like(sample['coords']) * 0.5
        sample['coords'] = sample['coords'] + noise
        
        # 重新计算 EvoEF2 (或使用近似)
        sample['evoef2_energy'] = sample['evoef2_energy'] + 50.0  # 假设 decoy 高能量
        sample['is_decoy'] = True
        
        return sample
```

---

### Phase 3C: TCR-specific 微调 (Day 16-18)

#### Step 3C.1: TCR 数据加载

```python
class TCRpMHCDataset(Dataset):
    """TCR-pMHC 结构数据集"""
    def __init__(self, tcr_dir, energy_cache):
        self.samples = self._load_tcr_structures(tcr_dir)
        self.energy_cache = energy_cache
    
    def _load_tcr_structures(self, tcr_dir):
        samples = []
        for pdb_file in Path(tcr_dir).glob("*.pdb"):
            # 解析 TCR-pMHC 结构
            structure = parse_tcr_pmhc(pdb_file)
            
            # 标注链类型
            chain_types = assign_chain_types(structure)
            # 0=TCRα, 1=TCRβ, 2=peptide, 3=MHC
            
            samples.append({
                'pdb_id': pdb_file.stem,
                'seq_tokens': structure.seq_tokens,
                'chain_types': chain_types,
                'coords': structure.coords,
                'contact_map': structure.contact_map,
            })
        
        return samples
```

#### Step 3C.2: 微调 Loss

```python
def compute_tcr_finetune_loss(pred, target):
    """
    Phase 3C Loss: Structure + Energy (all heads)
    """
    # Structure loss
    loss_struct = compute_structure_loss(pred, target)
    
    # Energy loss
    loss_energy = F.mse_loss(pred['energy'], target['evoef2_energy'])
    
    # 可选：CDR 区域加权
    cdr_mask = target.get('cdr_mask')
    if cdr_mask is not None:
        # CDR 区域的 contact 更重要
        loss_struct = reweight_by_cdr(loss_struct, cdr_mask)
    
    return loss_struct + 0.5 * loss_energy
```

---

### Phase MC: Monte Carlo 集成 (Day 19-21)

#### Step MC.1: 复用 psiMonteCarloSampler

```python
# 来自 psi_model/train.py
from psi_model.train import psiMonteCarloSampler

class EnergyGuidedMC:
    """
    基于 E_φ 的 Monte Carlo 优化
    """
    def __init__(self, energy_model, model_score_fn=None, alpha=1.0, beta=0.5):
        self.energy_model = energy_model  # TCRFoldProphet
        self.model_score_fn = model_score_fn  # FlowTCRGen.get_model_score
        self.alpha = alpha  # E_φ 权重
        self.beta = beta    # model score 权重
    
    def compute_energy(self, cdr3_seq, scaffold, pmhc):
        """计算混合能量"""
        # 预测结构
        with torch.no_grad():
            pred = self.energy_model(
                self.tokenize(cdr3_seq, scaffold, pmhc)
            )
            e_phi = pred['energy'].item()
        
        # Model score (可选)
        if self.model_score_fn:
            model_score = self.model_score_fn(cdr3_seq, {'scaffold': scaffold, 'pmhc': pmhc})
        else:
            model_score = 0
        
        return self.alpha * e_phi + self.beta * model_score
    
    def run(self, initial_cdr3, scaffold, pmhc, n_steps=1000, temp_schedule='linear'):
        """
        运行 MC 优化
        """
        current = initial_cdr3
        current_energy = self.compute_energy(current, scaffold, pmhc)
        best = current
        best_energy = current_energy
        
        for step in range(n_steps):
            # Temperature annealing
            temp = self.get_temperature(step, n_steps, temp_schedule)
            
            # Propose mutation
            candidate = self.propose_mutation(current)
            candidate_energy = self.compute_energy(candidate, scaffold, pmhc)
            
            # Metropolis-Hastings
            delta = candidate_energy - current_energy
            if delta < 0 or torch.rand(1).item() < torch.exp(-delta / temp):
                current = candidate
                current_energy = candidate_energy
                
                if current_energy < best_energy:
                    best = current
                    best_energy = current_energy
        
        return best, best_energy
    
    def propose_mutation(self, seq):
        """单点或多点突变"""
        seq_list = list(seq)
        pos = torch.randint(0, len(seq), (1,)).item()
        new_aa = random.choice('ACDEFGHIKLMNPQRSTVWY')
        seq_list[pos] = new_aa
        return ''.join(seq_list)
```

#### Step MC.2: 梯度引导 Proposal（可选）

```python
def gradient_informed_proposal(self, current_seq, scaffold, pmhc):
    """
    使用 E_φ 梯度指导 mutation 位置选择
    """
    tokens = self.tokenize(current_seq)
    tokens.requires_grad = True
    
    pred = self.energy_model(tokens, scaffold, pmhc)
    pred['energy'].backward()
    
    # 找到梯度最大的位置
    grad = tokens.grad.abs().sum(dim=-1)
    top_positions = grad.topk(3).indices.tolist()
    
    # 在这些位置提议突变
    pos = random.choice(top_positions)
    return self.mutate_at_position(current_seq, pos)
```

---

## 4. Reminders ⚠️

### 4.1 数据处理
- **PDB 清洗**：检查非标准残基、缺失原子
- **接口定义**：通常用 8Å 距离阈值
- **EvoEF2 修复**：某些 PDB 需要先 repair

### 4.2 训练配置
- **Phase 3A**: LR=1e-4, batch=4, epochs=100+
- **Phase 3B**: LR=1e-5, batch=8, 冻结大部分 trunk
- **Phase 3C**: LR=5e-6, batch=4, 全参数微调
- **能量归一化**：考虑对 EvoEF2 能量做标准化

### 4.3 评估指标
- **结构指标**：contact precision/recall, distance MAE
- **能量指标**：Pearson/Spearman 与 EvoEF2 的相关性
- **目标**：TCR 上 corr ≥ 0.7

### 4.4 代码风格
- **Checkpoint 路径**: `checkpoints/stage3_phase_a/`, `stage3_phase_b/`, `stage3_phase_c/`
- **日志**：每 epoch 打印 loss 分解和相关性

---

## 5. Checklist

### Phase 0: 数据准备
- [ ] 下载 ~50k PPI 结构
- [ ] 预处理脚本 `preprocess_pdb.py`
- [ ] EvoEF2 批处理脚本 `batch_evoef2.py`
- [ ] 下载 TCR3d / STCRDab 数据
- [ ] 预处理 TCR 结构

### Phase 3A: PPI 结构预训练
- [ ] 实现 `TCRFoldProphet` 类
- [ ] 实现 `StructureHead`（可选 IPA）
- [ ] 实现 `EnergyHead`
- [ ] 实现 `compute_structure_loss()`
- [ ] 训练脚本 `train_ppi_impl.py`
- [ ] 训练 100 epochs，保存 checkpoint

### Phase 3B: 能量 Surrogate
- [ ] 加载 3A checkpoint
- [ ] 实现参数冻结逻辑
- [ ] 实现 `compute_energy_loss()`
- [ ] （可选）实现 decoy 生成
- [ ] 训练脚本 `train_energy_surrogate.py`
- [ ] 验证 corr > 0.6 on PPI

### Phase 3C: TCR 微调
- [ ] 实现 `TCRpMHCDataset`
- [ ] 实现 `compute_tcr_finetune_loss()`
- [ ] 训练脚本 `train_tcr_impl.py`
- [ ] 验证 corr > 0.7 on TCR

### Phase MC: Monte Carlo
- [ ] 实现 `EnergyGuidedMC` 类
- [ ] 与 Stage 2 的 model score 集成
- [ ] 实现温度退火
- [ ] （可选）梯度引导 proposal

### Phase Integration: 端到端
- [ ] 与 Stage 2 pipeline 集成
- [ ] 实现 post-hoc screening
- [ ] 实现 ranking by E_φ
- [ ] 最终 EvoEF2 验证

### Phase Ablation: Ablation Studies (必做)
- [ ] E_φ vs EvoEF2 ranking 对比
- [ ] ±Decoy 训练对比
- [ ] MC hybrid energy 权重对比
- [ ] 生成 Ablation 结果表格

---

## 6. Ablation Studies (必做)

### 6.1 E_φ vs EvoEF2 Ranking

**目标**：验证 E_φ surrogate 是否能替代 EvoEF2 做候选筛选

```python
def ablation_energy_ranking(candidates, e_phi_model, evoef_runner):
    """
    比较 E_φ ranking 和 EvoEF2 ranking 的一致性
    """
    # E_φ 排序
    e_phi_scores = [e_phi_model.predict(c)['energy'] for c in candidates]
    ranking_phi = np.argsort(e_phi_scores)
    
    # EvoEF2 排序 (慢但准确)
    evoef_scores = [evoef_runner.compute_energy(c) for c in candidates]
    ranking_evoef = np.argsort(evoef_scores)
    
    # 计算 Spearman 相关性
    from scipy.stats import spearmanr
    corr, _ = spearmanr(ranking_phi, ranking_evoef)
    
    # 计算 Top-10 overlap
    top10_overlap = len(set(ranking_phi[:10]) & set(ranking_evoef[:10])) / 10
    
    return {'spearman_corr': corr, 'top10_overlap': top10_overlap}
```

**指标**：
- Spearman corr > 0.7
- Top-10 overlap > 50%

### 6.2 ±Decoy Training

**目标**：验证 decoy 结构对 E_φ 泛化的贡献

```python
ablation_configs = [
    {'name': 'no_decoy', 'use_decoys': False},
    {'name': 'with_decoy', 'use_decoys': True},
]

# 在 near-native 和 decoy 测试集上分别评估
# 预期：with_decoy 在 decoy 测试集上表现更好
```

### 6.3 MC Hybrid Energy Weights

**目标**：找到 E_φ 和 model score 的最优组合权重

```python
ablation_configs = [
    {'name': 'e_phi_only', 'alpha': 1.0, 'beta': 0.0},
    {'name': 'model_only', 'alpha': 0.0, 'beta': 1.0},
    {'name': 'hybrid_1:1', 'alpha': 0.5, 'beta': 0.5},
    {'name': 'hybrid_2:1', 'alpha': 0.67, 'beta': 0.33},
]

# E_total = alpha * E_phi + beta * ModelScore
```

### 6.4 Screening vs Full Pipeline

**目标**：验证后验筛选 vs 端到端的效果

```python
# 比较两种策略：
# 1. Flow → 全部候选 → EvoEF2 排序（慢）
# 2. Flow → E_φ 筛选 Top-10 → EvoEF2 精排（快）

# 指标：最终 Top-1 候选的 EvoEF2 能量分布
```

---

## 7. Exploratory (待做事项)

> 以下为可选探索项，不阻塞主线，但保留接口以便后续开发。

### 🟢 E1: Gradient Guidance in Flow ODE
- **目标**：在 Stage 2 ODE 中注入 Stage 3 的 ∇E_φ
- **公式**：`x_{t+Δt} = x_t + (v_θ - w∇E_φ)Δt`
- **接口预留**：Stage 2 的 `sample_with_cfg(..., energy_model, energy_weight)`
- **依赖**：E_φ 完成 + 可微传播
- **状态**：[ ] 待实现

### 🟢 E2: MC-to-Training Loop (Self-Play)
- **目标**：用 MC 优化的序列反馈训练 Stage 2
- **方案**：MC 找到的低能量序列作为额外正样本
- **接口预留**：`FlowTCRGen.add_positive_examples(seqs, weights)`
- **状态**：[ ] 待设计

### 🟢 E3: Gradient-Informed MC Proposal
- **目标**：用 ∇E_φ 指导 MC mutation 位置
- **方案**：在梯度大的位置优先 propose
- **接口预留**：`EnergyGuidedMC.propose_gradient_informed()`
- **状态**：[ ] 待实现

### 🟢 E4: Structure Prediction Head (IPA)
- **目标**：添加 IPA 头预测 3D 坐标
- **方案**：复用 AlphaFold IPA 架构
- **接口预留**：`TCRFoldProphet(..., use_ipa=True)`
- **状态**：[ ] 待实现（当前只有 distance/contact）

### 🟢 E5: Binding Affinity Regression
- **目标**：预测 TCR-pMHC 结合亲和力
- **数据**：需要实验测量的 Kd/EC50 数据
- **接口预留**：`TCRFoldProphet.predict_affinity()`
- **状态**：[ ] 待数据

---

## 8. 成功标准

| 指标 | Phase | 目标 |
|------|-------|------|
| Contact Precision | 3A | > 50% |
| Distance MAE | 3A | < 2.0 Å |
| Corr(E_φ, EvoEF2) on PPI | 3B | **> 0.6** |
| Corr(E_φ, EvoEF2) on TCR | 3C | **> 0.7** |
| MC 优化后能量降低 | MC | > 20 kcal/mol |
| Ablation: E_φ vs EvoEF2 ranking | - | Top-10 overlap > 50% |

---

## 9. 与其他 Stage 的接口

### 输入来自 Stage 2 (FlowTCR-Gen)

```python
# 从 Stage 2 获取 CDR3β 候选
from flowtcr_fold.FlowTCR_Gen import FlowTCRGen

flow_gen = FlowTCRGen.load("checkpoints/stage2_v1/best.pt")
cdr3b_candidates = flow_gen.sample(cond, n_samples=100)
model_scores = [flow_gen.get_model_score(c, cond) for c in cdr3b_candidates]
```

### 输出接口

```python
# Stage 3 提供的 API
class TCRFoldProphet:
    def predict(self, full_seq: Dict) -> Dict:
        """预测结构和能量"""
        return {
            'contact_map': ...,
            'distance_map': ...,
            'energy': ...,  # E_φ
        }
    
    def screen(self, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """筛选 top-k 候选"""
        scored = [(c, self.predict(c)['energy']) for c in candidates]
        return sorted(scored, key=lambda x: x[1])[:top_k]


class EnergyGuidedMC:
    def refine(self, cdr3_seq: str, scaffold: Dict, pmhc: Dict) -> Tuple[str, float]:
        """MC 优化返回最优序列和能量"""
        pass
```

---

**Last Updated**: 2025-12-01  
**Owner**: Stage 3 Implementation Team

