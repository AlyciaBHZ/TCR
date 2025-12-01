# Stage 2: FlowTCR-Gen Implementation Plan

> **Master Reference**: [../README.md](../README.md) (Section 4.2, Master Plan v3.1 Stage 2)
> 
> **Status**: 🔄 In Progress (40%)
> 
> **Timeline**: Week 3-5 (Plan v3.1)

---

## 1. 模块定位

### 1.1 在整体 Pipeline 中的角色

```
                    Stage 1: Immuno-PLM
                              │
                              ▼
                    Top-K scaffolds + pMHC embedding
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ★ Stage 2: FLOWTCR-GEN (You Are Here)                         │
│  ─────────────────────────────────────                          │
│  Topology-aware Dirichlet Flow Matching                         │
│  Output: CDR3β sequence candidates                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Stage 3: TCRFold-Prophet
```

### 1.2 核心目标

- **生成 CDR3β**：给定 pMHC + scaffold，生成多样且物理合理的 CDR3β 序列
- **拓扑感知**：使用 Collapse Token + Hierarchical Pairs 编码 TCR-pMHC 拓扑
- **可控生成**：支持 CFG (Classifier-Free Guidance) 调节条件强度

### 1.3 创新点（论文主打）

| 组件 | 描述 | 创新性 |
|------|------|--------|
| **Collapse Token (ψ)** | 可学习全局观察者 | 跨区域注意力聚合 |
| **Hierarchical Pair Embeddings** | 7-level 拓扑编码 | 注入 TCR-pMHC 结构先验 |
| **Dirichlet Flow Matching** | 氨基酸 simplex 上的连续生成 | 支持平滑插值和 CFG |

---

## 2. 当前实现状态

### 2.1 已完成 ✅

| 文件 | 功能 | 状态 |
|------|------|------|
| `flow_gen.py` | FlowMatchingModel 基础架构 | ✅ 可运行 |
| `SinusoidalTimeEmbedding` | 时间嵌入 | ✅ 完成 |
| `train_flow.py` | 训练脚本 | ⚠️ 需升级 |
| `sample.py` | ODE 采样 | ⚠️ 需添加 CFG |
| `pipeline_impl.py` | 端到端推理 | ⚠️ 需整合 Stage 1 |

### 2.2 待实现 🔄

| 任务 | 优先级 | 依赖 |
|------|--------|------|
| 集成 `CollapseAwareEmbedding` | 🔴 高 | psi_model 代码 |
| 集成 `SequenceProfileEvoformer` | 🔴 高 | psi_model 代码 |
| Hierarchical Pair IDs 生成 | 🔴 高 | - |
| Dirichlet Flow (x_t 注入) | 🔴 高 | - |
| CFG 实现 | 🔴 高 | - |
| Model Score Hook | 🟡 中 | - |
| Entropy/Profile 正则 | 🟡 中 | - |

### 2.3 Legacy 代码位置

```
psi_model/
├── model.py              # ⭐ CollapseAwareEmbedding, SequenceProfileEvoformer
├── model_original.py     # 原始版本（参考）
└── train.py              # psiMonteCarloSampler（参考）
```

---

## 3. Step-by-Step Implementation Plan

### Phase 1: 复用 psi_model 组件 (Day 1-3)

#### Step 1.1: 理解 `CollapseAwareEmbedding`

```python
# 来自 psi_model/model.py
class CollapseAwareEmbedding(nn.Module):
    """
    关键功能：
    1. Collapse Token (ψ): 可学习的全局观察者
    2. Hierarchical Pair IDs: 7-level 拓扑关系编码
    3. Region-specific weights: 不同区域的自适应权重
    """
    
    def create_hierarchical_pairs(self, ...):
        """
        返回 pair_ids [L, L]:
        - Level 0: ψ ↔ ψ
        - Level 1: ψ ↔ HD (CDR3)
        - Level 2: ψ ↔ 条件区域
        - Level 3: HD 内部
        - Level 4: 条件区域内部
        - Level 5: HD ↔ 条件区域
        - Level 6: 不同条件区域之间
        """
```

#### Step 1.2: 创建 FlowTCR-Gen 适配器

```python
# flowtcr_fold/FlowTCR_Gen/flow_gen.py 新增

from psi_model.model import CollapseAwareEmbedding, SequenceProfileEvoformer

class FlowTCRGenEncoder(nn.Module):
    """
    将 psi_model 组件适配为 FlowTCR-Gen 的条件编码器
    """
    def __init__(
        self,
        s_dim: int = 256,
        z_dim: int = 64,
        n_layers: int = 6,
        vocab_size: int = 21,
        max_len: int = 512,
    ):
        super().__init__()
        
        # 复用 psi_model 的嵌入层
        self.embedding = CollapseAwareEmbedding(
            s_in_dim=vocab_size,
            s_dim=s_dim,
            z_dim=z_dim,
            max_len=max_len,
        )
        
        # 复用 psi_model 的 Evoformer
        self.backbone = SequenceProfileEvoformer(
            s_dim=s_dim,
            z_dim=z_dim,
            n_layers=n_layers,
        )
    
    def forward(self, cdr3_xt, peptide, mhc, scaffold_seqs, conditioning_info):
        """
        Args:
            cdr3_xt: [B, L_cdr3, vocab] flow 中间状态
            peptide: [B, L_pep] peptide 序列
            mhc: [B, L_mhc] MHC 序列
            scaffold_seqs: Dict[str, Tensor] HV/HJ/LV/LJ 序列
            conditioning_info: List[str] 使用哪些条件
        
        Returns:
            s: [B, L_total, s_dim] 序列表征
            z: [B, L_total, L_total, z_dim] pair 表征
        """
        # 构建输入字典
        in_dict = {
            'hd': cdr3_xt,  # x_t 作为 HD 区域
            'pep': peptide,
            'mhc': mhc,
            **scaffold_seqs,
        }
        
        # 嵌入 + pair_ids
        s, z = self.embedding(in_dict, conditioning_info)
        
        # Evoformer 处理
        s, z = self.backbone(s, z)
        
        return s, z
```

#### Step 1.3: x_t 注入方式

```python
def inject_xt_into_embedding(self, x_t: torch.Tensor) -> torch.Tensor:
    """
    将 flow 中间状态 x_t 注入到嵌入空间
    
    方法：x_t 是 [B, L, vocab] 的软分布
    → 通过 embedding 矩阵的期望得到连续嵌入
    """
    # x_t: [B, L, vocab], embedding: [vocab, s_dim]
    # → [B, L, s_dim]
    emb = torch.matmul(x_t, self.token_embedding.weight)
    return emb + self.position_embedding
```

---

### Phase 2: Dirichlet Flow Matching (Day 4-6)

#### Step 2.1: Flow 插值定义

```python
def dirichlet_interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
    """
    Dirichlet Flow 插值:
    - x0: 先验分布 (uniform Dirichlet 或高熵分布)
    - x1: 目标分布 (one-hot ground truth)
    - t: 时间 [0, 1]
    
    x_t = (1 - t) * x0 + t * x1
    """
    return (1 - t) * x0 + t * x1


def sample_x0_dirichlet(batch_size: int, seq_len: int, vocab_size: int, alpha: float = 1.0):
    """
    从 Dirichlet(α, α, ..., α) 采样先验分布
    α = 1 时为均匀分布
    """
    dist = torch.distributions.Dirichlet(torch.ones(vocab_size) * alpha)
    return dist.sample((batch_size, seq_len))
```

#### Step 2.2: Flow Matching Loss

```python
def flow_matching_loss(
    model: nn.Module,
    x1: torch.Tensor,      # [B, L, vocab] one-hot target
    cond: Dict,            # 条件信息
    alpha: float = 1.0,    # Dirichlet 参数
) -> torch.Tensor:
    B, L, V = x1.shape
    device = x1.device
    
    # 1. 采样 x0 (先验)
    x0 = sample_x0_dirichlet(B, L, V, alpha).to(device)
    
    # 2. 采样 t ~ Uniform(0, 1)
    t = torch.rand(B, 1, 1, device=device)
    
    # 3. 计算 x_t
    x_t = dirichlet_interpolate(x0, x1, t)
    
    # 4. 目标速度场 v* = x1 - x0
    v_target = x1 - x0
    
    # 5. 模型预测速度场
    v_pred = model(x_t, t.squeeze(-1), cond)
    
    # 6. MSE loss
    loss = F.mse_loss(v_pred, v_target)
    
    return loss
```

#### Step 2.3: 完整训练循环

```python
def train_epoch(model, encoder, loader, optimizer, cfg_drop_prob=0.1):
    model.train()
    encoder.train()
    total_loss = 0
    
    for batch in loader:
        # 1. 编码条件
        cond = encoder(
            cdr3_xt=None,  # 训练时不需要
            peptide=batch['peptide'],
            mhc=batch['mhc'],
            scaffold_seqs=batch['scaffold_seqs'],
            conditioning_info=['pep', 'mhc', 'hv', 'hj', 'lv', 'lj'],
        )
        
        # 2. CFG: 随机 drop 条件
        if torch.rand(1).item() < cfg_drop_prob:
            cond = None  # 或用 learned uncond embedding
        
        # 3. 准备 target (one-hot CDR3β)
        x1 = F.one_hot(batch['cdr3b_tokens'], num_classes=model.vocab_size).float()
        
        # 4. Flow matching loss
        loss_flow = flow_matching_loss(model, x1, cond)
        
        # 5. (可选) Collapse entropy 正则
        loss_entropy = compute_collapse_entropy(encoder, batch)
        
        # 6. 总 loss
        loss = loss_flow + λ_ent * loss_entropy
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)
```

---

### Phase 3: CFG 实现 (Day 7-8)

#### Step 3.1: 训练时 Condition Drop

```python
class CFGWrapper(nn.Module):
    """
    Classifier-Free Guidance 包装器
    """
    def __init__(self, model, drop_prob=0.1):
        super().__init__()
        self.model = model
        self.drop_prob = drop_prob
        # 可学习的 unconditional embedding
        self.uncond_emb = nn.Parameter(torch.zeros(1, model.hidden_dim))
    
    def forward(self, x_t, t, cond, training=True):
        if training and torch.rand(1).item() < self.drop_prob:
            # Drop condition → 使用 uncond embedding
            cond = self.uncond_emb.expand(x_t.size(0), -1)
        return self.model(x_t, t, cond)
```

#### Step 3.2: 推理时 CFG

```python
def sample_with_cfg(
    model: nn.Module,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    seq_len: int,
    n_steps: int = 100,
    cfg_weight: float = 1.5,
) -> torch.Tensor:
    """
    CFG 采样:
    v_final = v_uncond + w * (v_cond - v_uncond)
    """
    device = cond.device
    B = cond.size(0)
    
    # 初始化 x_0 (uniform)
    x = torch.ones(B, seq_len, model.vocab_size, device=device) / model.vocab_size
    
    dt = 1.0 / n_steps
    
    for step in range(n_steps):
        t = torch.full((B, 1), step / n_steps, device=device)
        
        # 有条件预测
        v_cond = model(x, t, cond)
        
        # 无条件预测
        v_uncond = model(x, t, uncond)
        
        # CFG 组合
        v = v_uncond + cfg_weight * (v_cond - v_uncond)
        
        # Euler step
        x = x + v * dt
        
        # 投影回 simplex (归一化)
        x = F.softmax(x, dim=-1)
    
    # 最终解码
    tokens = x.argmax(dim=-1)
    return tokens
```

---

### Phase 4: Model Score Hook (Day 9)

#### Step 4.1: 定义 Model Score

```python
def compute_model_score(model, encoder, cdr3_tokens, cond):
    """
    计算生成序列的 model score，用于 hybrid MC energy
    
    可选定义:
    1. Flow cost: 积分 ||v_θ(x_t, t)||² dt
    2. Collapse scalar: ψ token 的某个投影
    3. Approximate NLL
    """
    # 方法 1: 近似 NLL (通过 ODE likelihood)
    x1 = F.one_hot(cdr3_tokens, model.vocab_size).float()
    
    # 反向 ODE 计算 log_prob
    log_prob = compute_ode_log_prob(model, x1, cond)
    
    return -log_prob  # 负 log prob 作为 score (越低越好)
```

#### Step 4.2: 导出 Hook

```python
class FlowTCRGen(nn.Module):
    def __init__(self, ...):
        ...
    
    def get_model_score(self, cdr3_seq: str, cond: Dict) -> float:
        """
        供 Stage 3 MC 使用的接口
        """
        tokens = self.tokenize(cdr3_seq)
        with torch.no_grad():
            score = compute_model_score(self.model, self.encoder, tokens, cond)
        return score.item()
```

---

### Phase 5: 评估指标 (Day 10)

#### Step 5.1: Recovery Rate

```python
def evaluate_recovery(model, val_loader, n_samples=10):
    """
    计算生成的 CDR3β 与真实序列的匹配率
    """
    exact_match = 0
    total = 0
    
    for batch in val_loader:
        cond = encode_condition(batch)
        
        for _ in range(n_samples):
            generated = model.sample(cond)
            for i, (gen, gt) in enumerate(zip(generated, batch['cdr3b'])):
                if gen == gt:
                    exact_match += 1
                total += 1
    
    return exact_match / total
```

#### Step 5.2: Diversity

```python
def evaluate_diversity(model, val_loader, n_samples=100):
    """
    计算生成序列的多样性 (unique ratio)
    """
    all_generated = set()
    
    for batch in val_loader:
        cond = encode_condition(batch)
        for _ in range(n_samples):
            generated = model.sample(cond)
            all_generated.update(generated)
    
    return len(all_generated) / (len(val_loader) * n_samples)
```

#### Step 5.3: Perplexity

```python
def evaluate_perplexity(model, val_loader):
    """
    计算验证集上的困惑度
    """
    total_nll = 0
    total_tokens = 0
    
    for batch in val_loader:
        cond = encode_condition(batch)
        x1 = F.one_hot(batch['cdr3b_tokens'], model.vocab_size).float()
        
        # 计算 NLL
        nll = compute_ode_log_prob(model, x1, cond)
        total_nll += nll.sum().item()
        total_tokens += x1.size(0) * x1.size(1)
    
    return torch.exp(torch.tensor(total_nll / total_tokens))
```

---

## 4. Reminders ⚠️

### 4.1 训练配置
- **CFG drop prob**: 0.1（训练时 10% 概率 drop 条件）
- **CFG weight**: 1.0-2.0（推理时可调）
- **λ_ent**: 0.01（collapse entropy 正则权重）
- **λ_prof**: 0.01（profile 正则权重）
- **ODE steps**: 100（采样步数）

### 4.2 长序列处理
- **MHC 序列**：可能很长（>200aa），需要截断或 chunked attention
- **拼接顺序**：`[ψ, CDR3β, peptide, MHC, HV, HJ, LV, LJ]`
- **位置编码**：每个区域有独立的位置 offset

### 4.3 与 Stage 1 接口
- **输入**：需要 Stage 1 的 `encode_pmhc()` 和 `retrieve_scaffolds()`
- **条件格式**：确保 embedding 维度匹配

### 4.4 代码风格
- **复用 psi_model**：不要 copy-paste，直接 import
- **Checkpoint 路径**：保存到 `checkpoints/stage2_v1/`
- **日志**：每 epoch 打印 loss 分解、recovery、diversity

---

## 5. Checklist

### Phase 1: 复用 psi_model
- [ ] 确认 `psi_model/model.py` 可 import
- [ ] 创建 `FlowTCRGenEncoder` 适配器类
- [ ] 实现 x_t 注入方式
- [ ] 测试 `create_hierarchical_pairs()` 输出正确

### Phase 2: Dirichlet Flow
- [ ] 实现 `sample_x0_dirichlet()`
- [ ] 实现 `dirichlet_interpolate()`
- [ ] 实现 `flow_matching_loss()`
- [ ] 修改 `train_flow.py` 使用新 loss

### Phase 3: CFG
- [ ] 实现 `CFGWrapper` 类
- [ ] 训练时 condition drop
- [ ] 实现 `sample_with_cfg()`
- [ ] 添加 `--cfg_weight` 命令行参数

### Phase 4: Model Score Hook
- [ ] 定义 `compute_model_score()` 函数
- [ ] 在 `FlowTCRGen` 类中导出 `get_model_score()` 接口
- [ ] 测试与 Stage 3 MC 的集成

### Phase 5: 评估指标
- [ ] 实现 `evaluate_recovery()`
- [ ] 实现 `evaluate_diversity()`
- [ ] 实现 `evaluate_perplexity()`
- [ ] 在验证循环中调用

### Phase 6: Ablation Studies (必做)
- [ ] 添加 `--use_collapse` 参数和开关
- [ ] 添加 `--use_hier_pairs` 参数和开关
- [ ] 实现 CFG weight sweep 脚本
- [ ] 实现 conditioning components ablation
- [ ] 生成 Ablation 结果表格

### Phase 6: Ablation Studies (必做)

#### Step 6.1: Collapse Token Ablation

**目标**：验证 Collapse Token (ψ) 的贡献（论文核心 claim）

```python
# 配置接口
ablation_configs = [
    {'name': 'with_collapse', 'use_collapse': True},   # 默认
    {'name': 'no_collapse', 'use_collapse': False},    # 去掉 ψ token
]

# 在 FlowTCRGenEncoder 中添加开关
class FlowTCRGenEncoder(nn.Module):
    def __init__(self, ..., use_collapse: bool = True):
        self.use_collapse = use_collapse
        if use_collapse:
            self.collapse_token = nn.Parameter(torch.randn(1, 1, s_dim))
```

**预期结果**：with_collapse 的 recovery/diversity 应显著高于 no_collapse

#### Step 6.2: Hierarchical Pairs Ablation

**目标**：验证 7-level 拓扑编码的贡献（论文核心 claim）

```python
ablation_configs = [
    {'name': 'hier_pairs', 'use_hier_pairs': True},      # 默认
    {'name': 'flat_pairs', 'use_hier_pairs': False},     # 所有 pair 同 level
]

# 在 create_hierarchical_pairs 中添加开关
def create_hierarchical_pairs(..., use_hier: bool = True):
    if not use_hier:
        return torch.zeros(L, L, dtype=torch.long)  # 全部 level=0
    # 正常 7-level 逻辑
```

#### Step 6.3: CFG Ablation

**目标**：验证 CFG 对生成质量的影响

```python
ablation_configs = [
    {'name': 'cfg_1.0', 'cfg_weight': 1.0},
    {'name': 'cfg_1.5', 'cfg_weight': 1.5},
    {'name': 'cfg_2.0', 'cfg_weight': 2.0},
    {'name': 'no_cfg', 'cfg_weight': 0.0},  # 纯无条件
]
```

#### Step 6.4: Conditioning Components Ablation

**目标**：验证各条件组件的贡献

```python
# 通过 conditioning_info 控制
ablation_configs = [
    {'name': 'full', 'cond': ['pep', 'mhc', 'hv', 'hj', 'lv', 'lj']},
    {'name': 'no_scaffold', 'cond': ['pep', 'mhc']},
    {'name': 'no_peptide', 'cond': ['mhc', 'hv', 'hj', 'lv', 'lj']},
    {'name': 'scaffold_only', 'cond': ['hv', 'hj', 'lv', 'lj']},
]
```

---

### Phase 7: 集成测试
- [ ] 端到端训练 100 epochs
- [ ] 验证 recovery > 30%
- [ ] 验证 PPL < 10
- [ ] 保存最佳 checkpoint

---

## 6. Ablation Checklist (必做)

| Ablation | 配置 | 指标 | 状态 |
|----------|------|------|------|
| ±Collapse Token | `use_collapse = T/F` | Recovery, Diversity | [ ] |
| ±Hierarchical Pairs | `use_hier_pairs = T/F` | Recovery, Diversity | [ ] |
| CFG weight sweep | `cfg_weight = {0, 1.0, 1.5, 2.0}` | Recovery vs Diversity trade-off | [ ] |
| Conditioning components | 见 Step 6.4 | Recovery | [ ] |

---

## 7. Exploratory (待做事项)

> 以下为可选探索项，不阻塞主线，但保留接口以便后续开发。

### 🟢 E1: Physics Gradient Guidance in ODE
- **目标**：在 ODE 采样中注入 ∇E_φ 梯度
- **公式**：`x_{t+Δt} = x_t + (v_θ - w∇E_φ)Δt`
- **接口预留**：`sample_with_cfg(..., energy_model=None, energy_weight=0.0)`
- **依赖**：Stage 3 E_φ 完成
- **状态**：[ ] 待实现

### 🟢 E2: Entropy Scheduling
- **目标**：在 ODE 不同阶段使用不同的 entropy 正则
- **方案**：早期高 entropy（探索），后期低 entropy（收敛）
- **接口预留**：`EntropyScheduler` 类
- **状态**：[ ] 待实现

### 🟢 E3: Multi-CDR Generation
- **目标**：同时生成 CDR3α 和 CDR3β
- **方案**：扩展 HD 区域包含双链
- **接口预留**：`generate(..., targets=['cdr3a', 'cdr3b'])`
- **状态**：[ ] 待设计

### 🟢 E4: Self-Play with Stage 3 Feedback
- **目标**：用 Stage 3 E_φ 评分反馈训练 Stage 2
- **方案**：对高分生成结果增加训练权重
- **接口预留**：`update_with_energy_feedback(generated, scores)`
- **状态**：[ ] 待设计

---

## 8. 成功标准

| 指标 | 目标 |
|------|------|
| Recovery Rate | **> 30%** |
| Diversity | **> 50%** unique in 100 samples |
| Perplexity | **< 10** |
| 训练时间 | < 48h @1×A100 |
| Ablation: ±collapse delta | 记录显著差异 |
| Ablation: ±hier_pairs delta | 记录显著差异 |

---

## 9. 与其他 Stage 的接口

### 输入来自 Stage 1 (Immuno-PLM)

```python
# 从 Stage 1 获取 scaffold
from flowtcr_fold.Immuno_PLM import ImmunoPLM

plm = ImmunoPLM.load("checkpoints/stage1_v1/best.pt")
scaffolds = plm.retrieve_scaffolds(pmhc_emb, top_k=10)
```

### 输出给 Stage 3 (TCRFold-Prophet)

```python
# Stage 2 提供的 API
class FlowTCRGen:
    def sample(self, cond: Dict, n_samples: int = 100) -> List[str]:
        """生成 CDR3β 序列"""
        pass
    
    def get_model_score(self, cdr3_seq: str, cond: Dict) -> float:
        """返回 model score 用于 hybrid MC"""
        pass
```

---

**Last Updated**: 2025-12-01  
**Owner**: Stage 2 Implementation Team

