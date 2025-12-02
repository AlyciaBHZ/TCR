# Stage 1: Immuno-PLM Implementation Plan

> **Master Reference**: [../README.md](../README.md) (Section 4.1, Master Plan v3.1 Stage 1)
> 
> **Status**: 🔄 In Progress (70%)
> 
> **Timeline**: Week 1-2 (Plan v3.1)

---

## 1. 模块定位

### 1.1 在整体 Pipeline 中的角色

```
┌─────────────────────────────────────────────────────────────────┐
│  Input: Target pMHC (peptide + MHC allele)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ★ Stage 1: IMMUNO-PLM (You Are Here)                          │
│  ─────────────────────────────────────                          │
│  Model p(V, J | MHC, peptide)                                   │
│  Output: Top-K scaffold priors for Stage 2                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Stage 2: FlowTCR-Gen
                              │
                              ▼
                    Stage 3: TCRFold-Prophet
```

### 1.2 核心目标

- **建模 p(V, J | MHC, peptide)**：MHC 是强信号，peptide 是弱修正
- **CDR3β 不作为输入**：仅用于统计分析，Stage 2 才生成
- **输出**：为每个 pMHC 提供 Top-K 个兼容的 V/J scaffold

### 1.3 创新点

| 组件 | 描述 | 与 baseline 对比 |
|------|------|------------------|
| Dual-group InfoNCE | MHC 分组 + pMHC 分组 | 传统只用单一分组 |
| Multi-label BCE | 多标签 gene ID 预测 | 传统用单标签分类 |
| Allele Embedding | 离散 HLA allele 嵌入 | 传统只用序列 |

---

## 2. 当前实现状态

### 2.1 已完成 ✅

| 文件 | 功能 | 状态 |
|------|------|------|
| `train_scaffold_retrieval.py` | 主训练脚本 | ✅ 可运行 |
| `ScaffoldRetrievalDataset` | 数据加载 | ✅ 支持 JSONL |
| `ScaffoldRetriever` | 模型架构 | ⚠️ 需升级为 v3.1 |
| `compute_infonce()` | InfoNCE 损失 | ⚠️ 一对一，需改多正样本 |
| `compute_classification_loss()` | 分类损失 | ⚠️ 单标签，需改多标签 |
| `ScaffoldBank` | 检索库 | ✅ 基本可用 |
| ESM-2 + LoRA | Backbone | ✅ 已集成 |

### 2.2 待实现 🔄

| 任务 | 优先级 | 依赖 |
|------|--------|------|
| Multi-positive InfoNCE | 🔴 高 | - |
| Dual-group masking (MHC + pMHC) | 🔴 高 | Multi-pos InfoNCE |
| Multi-label BCE | 🔴 高 | - |
| Allele Embedding Table | 🟡 中 | 数据清洗 |
| Top-K / KL 评估指标 | 🔴 高 | - |
| MHC-only baseline | 🟡 中 | - |

### 2.3 已知问题

1. **R@10 仅 1.1%**：当前 one-to-one InfoNCE 将同 peptide 的其他 scaffold 当负样本
2. **Gene name 混淆**：`h_v` 字段包含 `TRAV` 基因（α 链），需数据清洗
3. **长尾分布**：V/J gene 分布极度不均，需 pos_weight 或 focal loss
4. **Peptide 消融缺位**：当前未在同一模型内快速切换「含 peptide」vs「仅 MHC」输入，ablation 需集成。
5. **代码结构偏乱**：主要逻辑在 `train_scaffold_retrieval.py`，需按 v3.1 方案整理（src/、train.py/model.py 拆分，统一 ckpt 目录）。

### 2.4 代码清理与结构要求
- 以 `Immuno_PLM/train_scaffold_retrieval.py` 为主参考，梳理到 `src/` 下的模块化代码（e.g., `src/model.py`, `src/train.py`, `src/data.py`）。
- 启用早停与 checkpoint：保存到 `saved_model/stage1_v*/checkpoints/`、`other_results/`、`best_model/` 目录。
- CLI 需提供 ckpt 路径、early stopping、peptide on/off ablation 开关，保持与 plan v3.1 一致。

---

## 3. Step-by-Step Implementation Plan

### Phase 1: 数据准备 (Day 1-2)

#### Step 1.1: 数据清洗
```bash
# 检查 gene name 混淆
python -c "
import json
from collections import Counter
hv_genes = Counter()
with open('flowtcr_fold/data/trn.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('h_v'):
            hv_genes[obj['h_v'][:4]] += 1  # TRBV vs TRAV
print(hv_genes)
"
```

**预期输出**：确认是否存在 TRAV 混入 h_v 字段

#### Step 1.2: 构建 Allele Vocabulary
```python
# 在 train_scaffold_retrieval.py 中添加
class AlleleVocab:
    def __init__(self, data_path):
        self.allele2id = {"<UNK>": 0}
        # 从数据中收集所有 unique allele
        ...
```

#### Step 1.3: 预计算 pos_mask
```python
# 在 DataLoader 层面预计算分组 mask
def collate_fn_with_pos_mask(batch):
    # 按 MHC 分组
    mhc_ids = [s['mhc'] for s in batch]
    pos_mask_mhc = build_pos_mask(mhc_ids)
    
    # 按 pMHC 分组
    pmhc_ids = [(s['peptide'], s['mhc']) for s in batch]
    pos_mask_pmhc = build_pos_mask(pmhc_ids)
    
    return {
        ...,
        'pos_mask_mhc': pos_mask_mhc,
        'pos_mask_pmhc': pos_mask_pmhc,
    }
```

---

### Phase 2: Multi-positive InfoNCE (Day 3-4)

#### Step 2.1: 实现 `compute_infonce_multi_positive()`

```python
def compute_infonce_multi_positive(
    anchor: torch.Tensor,      # [B, D]
    positive: torch.Tensor,    # [B, D]
    pos_mask: torch.Tensor,    # [B, B] 1 表示同组
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Multi-positive InfoNCE: 同组样本共享正样本集合
    
    L = -log( sum_{j in P(i)} exp(s_ij/τ) / sum_{k} exp(s_ik/τ) )
    """
    # 计算相似度矩阵
    sim = anchor @ positive.T / temperature  # [B, B]
    
    # 数值稳定性
    sim_max = sim.max(dim=1, keepdim=True).values
    exp_sim = torch.exp(sim - sim_max)
    
    # 分子：所有正样本的 exp sum
    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    
    # 分母：所有样本的 exp sum
    all_sum = exp_sim.sum(dim=1)
    
    # 防止 log(0)
    loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)
    
    return loss.mean()
```

#### Step 2.2: 双层分组逻辑

```python
def train_epoch_v31(model, loader, optimizer, ...):
    for batch in loader:
        # 前向
        z_pmhc, z_hv, z_hj, z_lv, z_lj = model(batch)
        
        # MHC-group InfoNCE (主)
        loss_nce_mhc = (
            compute_infonce_multi_positive(z_pmhc, z_hv, batch['pos_mask_mhc']) +
            compute_infonce_multi_positive(z_pmhc, z_hj, batch['pos_mask_mhc']) +
            ...
        )
        
        # pMHC-group InfoNCE (辅)
        loss_nce_pmhc = (
            compute_infonce_multi_positive(z_pmhc, z_hv, batch['pos_mask_pmhc']) +
            ...
        )
        
        loss_nce = loss_nce_mhc + λ_pmhc * loss_nce_pmhc
        
        # Ablation toggle (peptide-off): optional forward with peptide masked to log R@K/KL
        if config.log_ablation_peptide_off:
            batch_masked = mask_peptide(batch)  # blank peptide tokens
            z_pmhc_masked, z_hv_m, z_hj_m, z_lv_m, z_lj_m = model(batch_masked)
            loss_nce_mhc_masked = (
                compute_infonce_multi_positive(z_pmhc_masked, z_hv_m, batch['pos_mask_mhc']) +
                compute_infonce_multi_positive(z_pmhc_masked, z_hj_m, batch['pos_mask_mhc'])
            )
            # 只做日志，不反向，或在 ablation 模式下单独训练
```

---

### Phase 3: Multi-label BCE (Day 5)

#### Step 3.1: 构建 Multi-hot Target

```python
def build_multilabel_target(batch, gene_vocab):
    """
    为每个 MHC group 构建 multi-hot gene target
    
    Example:
        MHC="HLA-A*02:01" 对应的样本有 [TRBV19, TRBV12, TRBV19]
        → target_hv = [0, 0, ..., 1, ..., 1, ...]  # TRBV12 和 TRBV19 位置为 1
    """
    B = len(batch)
    num_hv = len(gene_vocab['h_v'])
    target_hv = torch.zeros(B, num_hv)
    
    # 按 MHC 分组聚合
    for i, sample in enumerate(batch):
        group_samples = get_same_mhc_samples(sample['mhc'], batch)
        for s in group_samples:
            if s['h_v'] in gene_vocab['h_v']:
                target_hv[i, gene_vocab['h_v'][s['h_v']]] = 1.0
    
    return target_hv
```

#### Step 3.2: 带 pos_weight 的 BCE

```python
def compute_classification_loss_multilabel(
    logits: torch.Tensor,      # [B, num_genes]
    target: torch.Tensor,      # [B, num_genes] multi-hot
    pos_weight: torch.Tensor,  # [num_genes] 类别权重
    valid_mask: torch.Tensor   # [B] 是否有效
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction='none'
    )
    loss = (loss.mean(dim=1) * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    return loss
```

---

### Phase 4: 评估指标 (Day 6-7)

#### Step 4.1: Top-K Recall

```python
def evaluate_topk_recall(model, val_loader, scaffold_bank, k_list=[1, 5, 10, 20]):
    """
    对每个验证样本的 pMHC:
    1. 编码得到 z_pmhc
    2. 检索 Top-K scaffold
    3. 检查真实使用的 gene 是否在 Top-K 中
    """
    results = {k: [] for k in k_list}
    
    for batch in val_loader:
        z_pmhc = model.encode_pmhc(batch)
        
        for k in k_list:
            topk_genes = scaffold_bank.retrieve(z_pmhc, 'h_v', k)
            hit = any(gene in batch['true_hv_set'] for gene in topk_genes)
            results[k].append(hit)
    
    return {k: np.mean(v) for k, v in results.items()}
```

#### Step 4.2: KL Divergence

```python
def evaluate_kl_divergence(model, val_loader, empirical_dist):
    """
    比较模型预测的 p(V|MHC) 与训练集经验分布的 KL 散度
    """
    kl_scores = []
    
    for mhc, p_emp in empirical_dist.items():
        z_pmhc = model.encode_mhc(mhc)
        logits = model.classify_hv(z_pmhc)
        p_model = F.softmax(logits, dim=-1)
        
        kl = F.kl_div(p_model.log(), p_emp, reduction='sum')
        kl_scores.append(kl.item())
    
    return np.mean(kl_scores)
```

---

### Phase 5: Baseline 对比 (Day 8)

#### Step 5.1: 频率 Baseline

```python
def frequency_baseline(train_data):
    """
    对每个 MHC，直接用训练集中的 V gene 频率作为预测分布
    """
    mhc_to_hv_counts = defaultdict(Counter)
    for sample in train_data:
        mhc_to_hv_counts[sample['mhc']][sample['h_v']] += 1
    
    # 转为概率分布
    mhc_to_hv_dist = {}
    for mhc, counts in mhc_to_hv_counts.items():
        total = sum(counts.values())
        mhc_to_hv_dist[mhc] = {g: c/total for g, c in counts.items()}
    
    return mhc_to_hv_dist
```

#### Step 5.2: MHC-only Model

```python
# 在输入中 mask 掉 peptide
def create_mhc_only_input(batch):
    batch_mhc_only = batch.copy()
    batch_mhc_only['peptide'] = [''] * len(batch['peptide'])  # 或用 [MASK] token
    return batch_mhc_only
```

#### Step 5.3: 内置 Peptide Ablation（同模型快速对比）
- 训练/评估参数：仅保留 `--ablation`（peptide-off）；默认训练会自动在评估阶段再跑一次 peptide-masked 前向并记录 R@K/KL（同一 checkpoint）。
- 作用：无需额外模型就能产出 pMHC vs MHC-only 指标；若需纯 MHC-only 训练，仍可将 peptide 全部置空并完整训练一版作为严格 baseline。

---

## 4. Reminders ⚠️

### 4.1 训练配置
- **λ_pmhc 初值**: 0.3（pMHC group 权重低于 MHC group）
- **λ_bce 初值**: 0.2（分类损失辅助）
- **pos_weight**: 需根据 gene 频率计算，稀有 gene 权重更高
- **Early stopping patience**: 20 epochs

### 4.2 数据问题
- **Gene name 清洗**: 当前数据已检查，无 TRAV 泄漏（保持监控即可）
- **缺失值处理**: LV/LJ 缺失时用 `<NONE>` token，不参与对应 loss
- **Batch 采样**: 确保每个 batch 内有足够多的同 MHC 样本

### 4.3 代码风格
- **命名规范**: 使用数字序列（M1, M2...）而非描述性名称
- **Checkpoint 路径**: 保存到 `checkpoints/stage1_v1/` 或 `stage1_v2/`
- **日志**: 每 epoch 打印 loss 分解和 R@10

---

## 5. Checklist

### Phase 1: 数据准备
- [x] Gene name 检查：当前数据无 TRAV 泄漏（无需额外清理）
 - [x] Allele 处理：保持简单字典映射（不引入类/序列 fallback，按需求待定）
 - [x] 实现 `collate_fn_with_pos_mask()` 预计算分组 mask
 - [x] 计算 gene 频率用于 pos_weight

### Phase 2: Multi-positive InfoNCE
- [x] 实现 `compute_infonce_multi_positive()` 函数
- [x] 修改 `train_epoch()` 使用双层 InfoNCE（仅 has_mhc 子集；缺 MHC 仅参与 peptide 分组）
- [x] 添加 `λ_pmhc` 超参数控制

### Phase 3: Multi-label BCE
- [x] 实现 `build_multilabel_target()` 函数（数据侧预建 multi-hot）
- [x] 实现 `compute_classification_loss_multilabel()` 函数（仅 has_mhc）
- [x] 添加 `λ_bce` 超参数控制

### Phase 4: 评估指标
- [x] 实现 `evaluate_topk_recall()` 函数（多 K 汇总）
- [x] 实现 `evaluate_kl_divergence()` 函数
- [x] 在 `evaluate()` 中调用并打印

### Phase 5: Baseline
- [x] 实现频率 baseline
- [x] 实现 MHC-only model 输入接口（peptide mask ablation）
- [x] CLI 精简：仅 `--ablation`（peptide-off），其余参数写死

### Phase 6: Ablation Studies (必做)
- [x] 实现 `evaluate_with_ablation()` 函数（自动 peptide-off 评估）
- [ ] pMHC vs MHC-only 对比记录
- [ ] λ_pmhc = {0.0, 0.3, 1.0} 对比记录
- [ ] ±BCE loss 对比记录
- [ ] 生成 Ablation 结果表格

### Phase 6: Ablation Studies (必做)

#### Step 6.1: Peptide Ablation（pMHC vs MHC-only）

**目标**：验证 peptide 对 V/J 预测的贡献

```python
# 配置接口
class AblationConfig:
    peptide_on: bool = True           # 是否使用 peptide
    log_ablation_peptide_off: bool = True  # 评估时自动跑 peptide-masked 版本

# 评估时同时输出两组指标
def evaluate_with_ablation(model, val_loader, scaffold_bank, config):
    results = {}
    
    # 1. 正常评估 (pMHC)
    results['pMHC'] = {
        'R@10': evaluate_topk_recall(model, val_loader, scaffold_bank, peptide_on=True),
        'KL': evaluate_kl_divergence(model, val_loader, peptide_on=True),
    }
    
    # 2. Ablation: MHC-only
    if config.log_ablation_peptide_off:
        results['MHC_only'] = {
            'R@10': evaluate_topk_recall(model, val_loader, scaffold_bank, peptide_on=False),
            'KL': evaluate_kl_divergence(model, val_loader, peptide_on=False),
        }
    
    # 3. 计算 delta
    results['delta_R@10'] = results['pMHC']['R@10'] - results['MHC_only']['R@10']
    
    return results
```

**预期结果**：
- 若 delta > 0：peptide 有正向贡献，支持 pMHC 设计
- 若 delta ≈ 0：peptide 对 V/J 预测无显著影响（符合生物学预期）

#### Step 6.2: Dual-group InfoNCE Ablation

**目标**：验证 MHC-group 和 pMHC-group 各自贡献

```python
# 训练时通过 λ_pmhc 控制
ablation_configs = [
    {'name': 'MHC_only_InfoNCE', 'λ_pmhc': 0.0},   # 只用 MHC 分组
    {'name': 'pMHC_only_InfoNCE', 'λ_pmhc': 1.0},  # 只用 pMHC 分组
    {'name': 'Dual_InfoNCE', 'λ_pmhc': 0.3},       # 双层（默认）
]

# 记录表格
# | Config | R@10_HV | R@10_HJ | KL |
```

#### Step 6.3: Multi-label BCE Ablation

**目标**：验证分类 loss 的辅助作用

```python
ablation_configs = [
    {'name': 'InfoNCE_only', 'λ_bce': 0.0},
    {'name': 'InfoNCE_BCE', 'λ_bce': 0.2},
]
```

---

### Phase 7: 集成测试
- [ ] 端到端训练 100 epochs
- [ ] 验证 R@10 > 20%（目标）
- [ ] 验证 KL(model) < KL(baseline)
- [ ] 保存最佳 checkpoint

---

## 6. Ablation Checklist (必做)

| Ablation | 配置 | 指标 | 状态 |
|----------|------|------|------|
| pMHC vs MHC-only | 默认评估 + `--ablation` (peptide-off) | R@10, KL | [ ] |
| MHC-group vs pMHC-group | `λ_pmhc = 0.0 / 0.3 / 1.0` | R@10, KL | [ ] |
| ±BCE loss | `λ_bce = 0.0 / 0.2` | R@10 | [ ] |
| Frequency baseline | N/A | R@10, KL | [ ] |

---

## 7. Exploratory (待做事项)

> 以下为可选探索项，不阻塞主线，但保留接口以便后续开发。

### 🟢 E1: Allele Sequence Fallback for Cold-Start
- **问题**：未见过的 HLA allele 无 embedding
- **方案**：用 ESM 编码 allele 序列作为 fallback
- **接口预留**：`AlleleVocab.get_or_compute(allele_name, allele_seq)`
- **状态**：[ ] 待实现

### 🟢 E2: Hard Negative Mining
- **问题**：当前只用 batch 内随机负样本
- **方案**：构造相似但不兼容的 pMHC-scaffold 对
- **接口预留**：`HardNegativeSampler` 类
- **状态**：[ ] 待实现

### 🟢 E3: Contrastive + Generative Joint Training
- **问题**：Stage 1 和 Stage 2 独立训练
- **方案**：用 Stage 2 生成的 CDR3β 反馈 Stage 1
- **接口预留**：`update_scaffold_bank_with_generated()`
- **状态**：[ ] 待设计

### 🟢 E4: Causal LM Head for Generative Scaffold
- **问题**：当前 Stage 1 只做检索，不能直接生成新的 V/J 序列
- **方案**：添加 Causal LM 头，将检索式变为生成式
- **输入**：masked scaffold + pMHC 作为 context
- **输出**：autoregressively generate V/J sequence
- **接口预留**：
  ```python
  class ImmunoPLM:
      def generate_scaffold(self, pmhc_emb: torch.Tensor, max_len: int = 128) -> str:
          """Causal generation of V/J sequence"""
          pass
  ```
- **训练**：在 retrieval loss 之外加 LM cross-entropy loss
- **优势**：可生成训练集未见的新 V/J 组合
- **状态**：[ ] 待设计

---

## 8. 成功标准

| 指标 | Baseline | 目标 |
|------|----------|------|
| R@10 (HV) | 1.1% | **> 20%** |
| R@10 (HJ) | ~1% | **> 20%** |
| KL vs 频率 | - | **< baseline** |
| 训练时间 | - | < 24h @1×A100 |
| Ablation delta (pMHC - MHC) | - | 记录（可正可负） |

---

## 9. 与其他 Stage 的接口

### 输出给 Stage 2 (FlowTCR-Gen)

```python
# Stage 1 提供的 API
class ImmunoPLM:
    def encode_pmhc(self, peptide: str, mhc: str) -> torch.Tensor:
        """返回 pMHC embedding [1, D]"""
        pass
    
    def retrieve_scaffolds(self, pmhc_emb: torch.Tensor, top_k: int = 10) -> List[Dict]:
        """返回 Top-K scaffold 信息"""
        return [
            {"h_v": "TRBV19*01", "h_j": "TRBJ2-7*01", "h_v_seq": "...", ...},
            ...
        ]
```

---

**Last Updated**: 2025-12-01  
**Owner**: Stage 1 Implementation Team

---

## 10. 工作日志 / Checklist
- 2025-12-02: 重构训练脚本到现有目录（data.py, losses.py, model.py, train_utils.py, train.py）；启用双组 InfoNCE + 多标签 BCE；缺 MHC 样本仅参与 peptide 分组弱权重 InfoNCE，不参与 MHC 分组/BCE；输出路径标准化 `saved_model/` 下的 checkpoints/best/other_results；allele 处理保持简单字典（未启用序列 fallback）；CLI 精简为固定路径/ESM+LoRA 默认，仅支持 `--ablation`（peptide-off）与 `--resume/--resume_best`。旧版本代码已归档至 `old_version/`。
  - 运行指引：
    - 默认（含 peptide，自动评估 peptide-off）：`python flowtcr_fold/Immuno_PLM/train.py`
    - Peptide-off 训练：`python flowtcr_fold/Immuno_PLM/train.py --ablation`
    - 恢复：`--resume` 或 `--resume_best`（路径写死）
