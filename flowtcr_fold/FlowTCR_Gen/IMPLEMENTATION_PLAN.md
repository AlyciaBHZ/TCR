# Stage 2: FlowTCR-Gen Implementation Plan

> **Master Reference**: [../README.md](../README.md) (Section 4.2, Master Plan v3.1 Stage 2)
> 
> **Status**: ✅ **Code Complete** (90%) — Ready for Training
> 
> **Timeline**: Week 3-5 (Plan v3.1)
>
> **Latest Update (2025-12-03)**:
> - ✅ All core modules implemented
> - ✅ Ablation switches integrated (±collapse, ±hier_pairs, ±cfg)
> - ✅ Model score hook for Stage 3 integration
> - 🔄 Awaiting training with real data

---

## 1. 模块定位

### 1.1 在整体 Pipeline 中的角色

```
                    Stage 1: Immuno-PLM (✅ R@10 88%)
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
| `encoder.py` | FlowTCRGenEncoder + CollapseAwareEmbedding + SequenceProfileEvoformer | ✅ 完成 |
| `dirichlet_flow.py` | DirichletFlowMatcher + CFGWrapper + 采样 | ✅ 完成 |
| `model_flow.py` | FlowTCRGen 主模型类 + Model Score Hook | ✅ 完成 |
| `data.py` | FlowTCRGenDataset + Tokenizer + collate_fn | ✅ 完成 |
| `metrics.py` | Recovery/Diversity/Perplexity 评估 | ✅ 完成 |
| `train.py` | 训练脚本 + Ablation 支持 | ✅ 完成 |
| `__init__.py` | 模块导出 | ✅ 完成 |

### 2.2 待运行 🔄

| 任务 | 状态 | 备注 |
|------|------|------|
| 端到端训练 | 🔄 待运行 | 依赖数据准备 |
| Ablation 实验 | 🔄 待运行 | `--ablation` 参数已实现 |
| Stage 1 集成 | 🔄 待测试 | 接口已设计 |

### 2.3 代码结构

```
flowtcr_fold/FlowTCR_Gen/
├── __init__.py           # 模块导出
├── encoder.py            # ⭐ CollapseAwareEmbedding + SequenceProfileEvoformer
├── dirichlet_flow.py     # ⭐ Dirichlet Flow Matching + CFG
├── model_flow.py         # ⭐ FlowTCRGen 主模型 + Model Score Hook
├── data.py               # Dataset + Tokenizer
├── metrics.py            # 评估指标
├── train.py              # 训练脚本
├── IMPLEMENTATION_PLAN.md
├── saved_model/
│   ├── stage2/
│   │   ├── checkpoints/
│   │   ├── best_model/
│   │   └── other_results/
│   └── ablation_*/       # Ablation 实验输出
└── old_version/          # 旧代码 (flow_gen.py, train_flow.py)
```

---

## 3. 核心 API

### 3.1 FlowTCRGen 主模型

```python
from flowtcr_fold.FlowTCR_Gen import FlowTCRGen

# 创建模型
model = FlowTCRGen(
    s_dim=256,
    z_dim=64,
    n_layers=6,
    vocab_size=25,
    use_collapse=True,       # Ablation 开关
    use_hier_pairs=True,     # Ablation 开关
    cfg_drop_prob=0.1,
)

# 训练
losses = model.training_step(batch)
# losses = {'loss': ..., 'mse_loss': ..., 'entropy_loss': ...}

# 生成
tokens = model.generate(
    cdr3_len=15,
    pep_one_hot=...,
    mhc_one_hot=...,
    scaffold_seqs={'hv': ..., 'hj': ...},
    n_steps=100,
    cfg_weight=1.5,
)

# Model Score (for Stage 3 MC integration)
score = model.get_model_score(cdr3_tokens, pep_one_hot, mhc_one_hot, scaffold_seqs)
```

### 3.2 训练命令

```bash
# 默认训练 (完整模型)
python flowtcr_fold/FlowTCR_Gen/train.py

# Ablation: 无 Collapse Token
python flowtcr_fold/FlowTCR_Gen/train.py --ablation no_collapse

# Ablation: 无 Hierarchical Pairs
python flowtcr_fold/FlowTCR_Gen/train.py --ablation no_hier

# Ablation: 无 CFG
python flowtcr_fold/FlowTCR_Gen/train.py --ablation no_cfg

# 恢复训练
python flowtcr_fold/FlowTCR_Gen/train.py --resume

# 评估模式
python flowtcr_fold/FlowTCR_Gen/train.py --eval_only --cfg_weight 1.5
```

---

## 4. Checklist

### Phase 1: 复用 psi_model 组件 ✅
- [x] 创建 `CollapseAwareEmbedding` (独立实现，不依赖 psi_model import)
- [x] 创建 `SequenceProfileEvoformer` 
- [x] 创建 `FlowTCRGenEncoder` 适配器类
- [x] 实现 x_t 注入方式 (soft embedding via matmul)
- [x] 实现 `create_hierarchical_pairs()` 7-level 拓扑编码
- [x] 添加 `use_collapse` 开关
- [x] 添加 `use_hier_pairs` 开关

### Phase 2: Dirichlet Flow Matching ✅
- [x] 实现 `sample_x0_dirichlet()` 和 `sample_x0_uniform()`
- [x] 实现 `dirichlet_interpolate()`
- [x] 实现 `FlowHead` 速度预测头
- [x] 实现 `DirichletFlowMatcher.flow_matching_loss()`
- [x] 添加 entropy 正则化

### Phase 3: CFG ✅
- [x] 实现训练时 condition drop (cfg_drop_prob=0.1)
- [x] 实现 `CFGWrapper` 类
- [x] 实现 `generate()` with CFG
- [x] 添加 `--cfg_weight` 命令行参数

### Phase 4: Model Score Hook ✅
- [x] 实现 `get_model_score()` - 基于 flow cost 积分
- [x] 实现 `get_collapse_scalar()` - 基于 collapse token 投影
- [x] 设计 Stage 3 集成接口

### Phase 5: 评估指标 ✅
- [x] 实现 `compute_recovery_rate()` - exact match, partial match
- [x] 实现 `compute_diversity()` - unique ratio, entropy
- [x] 实现 `FlowTCRGenEvaluator` 类
- [x] 在验证循环中调用

### Phase 6: Ablation Studies ✅ (已实现开关)
- [x] 添加 `--ablation no_collapse` 参数
- [x] 添加 `--ablation no_hier` 参数
- [x] 添加 `--ablation no_cfg` 参数
- [ ] 运行 Ablation 实验并记录结果

### Phase 7: 集成测试 🔄
- [ ] 端到端训练 100 epochs
- [ ] 验证 recovery > 30%
- [ ] 验证 diversity > 50%
- [ ] 验证 PPL < 10
- [ ] 保存最佳 checkpoint

---

## 5. Ablation Checklist

| Ablation | 配置 | 指标 | 状态 |
|----------|------|------|------|
| ±Collapse Token | `--ablation no_collapse` | Recovery, Diversity | 🔄 待运行 |
| ±Hierarchical Pairs | `--ablation no_hier` | Recovery, Diversity | 🔄 待运行 |
| CFG weight sweep | `--cfg_weight {0, 1.0, 1.5, 2.0}` | Recovery vs Diversity | 🔄 待运行 |
| Conditioning components | conditioning_info 参数 | Recovery | 🔄 待运行 |

---

## 6. 与其他 Stage 的接口

### 输入来自 Stage 1 (Immuno-PLM)

```python
# Stage 1 输出 scaffold 信息
scaffold = {
    'h_v': 'TRBV19*01',
    'h_v_seq': 'MGTSLLCWMALCLLGADHADTGVS...',
    'h_j': 'TRBJ2-7*01',
    'h_j_seq': 'YEQYFGPGTRLTVT',
    # ... l_v, l_j
}

# 转换为 Stage 2 输入
from flowtcr_fold.FlowTCR_Gen import FlowTCRGenTokenizer

tokenizer = FlowTCRGenTokenizer()
hv_tokens = tokenizer.encode(scaffold['h_v_seq'])
hv_one_hot = tokenizer.to_one_hot(torch.tensor(hv_tokens))
```

### 输出给 Stage 3 (TCRFold-Prophet)

```python
# Stage 2 提供的 API
class FlowTCRGen:
    def generate(self, ..., n_steps=100, cfg_weight=1.5) -> torch.Tensor:
        """生成 CDR3β token indices"""
        pass
    
    def get_model_score(self, cdr3_tokens, ...) -> torch.Tensor:
        """返回 model score 用于 hybrid MC energy"""
        pass
    
    def get_collapse_scalar(self, ...) -> torch.Tensor:
        """返回 collapse token 标量，可选用于快速评估"""
        pass
```

---

## 7. 成功标准

| 指标 | 目标 | 当前 |
|------|------|------|
| Recovery Rate | **> 30%** | 🔄 待训练 |
| Diversity | **> 50%** unique in 100 samples | 🔄 待训练 |
| Perplexity | **< 10** | 🔄 待训练 |
| 训练时间 | < 48h @1×A100 | 🔄 待验证 |
| Ablation: ±collapse delta | 记录显著差异 | 🔄 待实验 |
| Ablation: ±hier_pairs delta | 记录显著差异 | 🔄 待实验 |

---

## 8. Exploratory (待做事项)

> 以下为可选探索项，不阻塞主线，但接口已预留。

### 🟢 E1: Physics Gradient Guidance in ODE
- **目标**：在 ODE 采样中注入 ∇E_φ 梯度
- **公式**：`x_{t+Δt} = x_t + (v_θ - w∇E_φ)Δt`
- **接口预留**：`generate(..., energy_model=None, energy_weight=0.0)`
- **依赖**：Stage 3 E_φ 完成
- **状态**：[ ] 待实现

### 🟢 E2: Entropy Scheduling
- **目标**：在 ODE 不同阶段使用不同的 entropy 正则
- **方案**：早期高 entropy（探索），后期低 entropy（收敛）
- **状态**：[ ] 待实现

### 🟢 E3: Multi-CDR Generation
- **目标**：同时生成 CDR3α 和 CDR3β
- **方案**：扩展 CDR3 区域包含双链
- **状态**：[ ] 待设计

### 🟢 E4: Self-Play with Stage 3 Feedback
- **目标**：用 Stage 3 E_φ 评分反馈训练 Stage 2
- **方案**：对高分生成结果增加训练权重
- **状态**：[ ] 待设计

---

## 9. 工作日志

- **2025-12-03**: Stage 2 代码完成
  - 创建 `encoder.py`: CollapseAwareEmbedding + SequenceProfileEvoformer + FlowTCRGenEncoder
  - 创建 `dirichlet_flow.py`: DirichletFlowMatcher + CFGWrapper + 采样函数
  - 创建 `model_flow.py`: FlowTCRGen 主模型 + Model Score Hook + 生成接口
  - 创建 `data.py`: FlowTCRGenDataset + Tokenizer + collate_fn
  - 创建 `metrics.py`: Recovery/Diversity/Perplexity 评估
  - 更新 `train.py`: 完整训练流程 + Ablation 支持
  - Ablation 开关: `use_collapse`, `use_hier_pairs`, `cfg_drop_prob`
  - 下一步: 准备训练数据，开始 baseline 训练

---

**Last Updated**: 2025-12-03  
**Owner**: Stage 2 Implementation Team
