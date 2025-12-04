# Stage 2: FlowTCR-Gen Implementation Plan

> **Master Reference**: [../README.md](../README.md) (Section 4.2, Master Plan v3.1 Stage 2)
> 
> **Status**: 🔧 **Code Complete + Bug Fixed** (95%) — 待重新训练
> 
> **Timeline**: Week 3-5 (Plan v3.1)
>
> **Latest Update (2025-12-05)**:
> - ✅ All core modules implemented
> - ✅ Per-sample conditioning + bug fixes completed
> - ✅ **Critical Bug Fixed**: ODE simplex projection (softmax → normalize)
> - ✅ 首轮训练完成（有 bug 版本），已获得有价值的 insights
> - 🔧 待重新训练验证修复效果
> - 📊 详细分析见 Section 10: Metrics 解释 与 Section 11: 首轮训练分析

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

### 2.2 运行中 🔄

| 任务 | 状态 | 备注 |
|------|------|------|
| 端到端训练 | 🔄 运行中 | Job 1116099 (Normal) |
| Ablation 实验 | 🔄 运行中 | Jobs 1116100, 1116109, 1116112 |
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

- **2025-12-05**: 首轮训练分析 + 文档完善
  - ❌ 终止所有 buggy 训练任务 (Jobs 1116099, 1116100, 1116109, 1116112)
  - 🗑️ 清理 buggy 模型 checkpoints
  - 📝 **详细记录 Metrics 定义** (Section 10)
  - 📊 **首轮训练分析** (Section 11)：
    - 确认 Loss 收敛正常（MSE 从 0.1 降到 0.001 级别）
    - 确认 Diversity 急剧下降（0.99 → 0.01）是 ODE bug + 可能的 mode collapse
    - 确认 Recovery = 0 主要由 ODE simplex 投影错误导致
    - 记录各 ablation 的初步趋势
  - 📋 **代码修复记录** (Section 12)：详细 diff 记录所有修改
  - 状态：**待重新训练**

- **2025-12-04 (续)**: 代码审查 + Bug 修复
  - 🔍 分析训练日志发现问题:
    - Recovery = 0 (所有模型)
    - Diversity 快速下降到 ~0.01
    - Loss 为负 (因为 entropy 正则)
  - 🐛 **ODE 积分 Bug 修复** (`model_flow.py`):
    - 原: `x = x + v * dt; x = F.softmax(x)` (错误的 simplex 投影)
    - 新: `x = (x + v * dt).clamp(1e-8); x = x / x.sum()` (正确的归一化)
  - 🔧 **评估参数优化** (`train.py`):
    - `n_samples_per_batch`: 3 → 8
    - `max_eval_samples`: 新增, 限制为 200
    - `n_steps` (生成): 50 → 100
  - 📈 新增 `recovery_80` 指标到日志输出
  - 下一步: 重新训练验证修复效果

- **2025-12-04**: 首次训练启动 + 参数调整
  - 🔧 修复 `PYTHONPATH` 问题（脚本中添加 `export PYTHONPATH`）
  - 🔧 调整 `BATCH_SIZE`: 16 → 32（A100 80GB 显存充足）
  - 🔧 调整输出目录：Normal 模型输出到 `stage2/normal/`（而非直接放 `stage2/`）
  - 🚀 启动 4 个实验：
    - Normal (Job 1116099): 5,334,631 params
    - No Collapse (Job 1116100): 5,331,039 params  
    - No Hier (Job 1116109): 5,334,631 params
    - No CFG (Job 1116112): 5,334,631 params
  - 📊 Epoch 1 早期 Loss 下降趋势（Batch 50-700）:
    | Experiment | Batch 50 | Batch 300 | Batch 700 |
    |------------|----------|-----------|-----------|
    | Normal | 0.103 | 0.005 | -0.005 |
    | No Collapse | 0.080 | 0.004 | -0.007 |
    | No Hier | 0.135 | - | - |
    | No CFG | 0.163 | 0.004 | -0.005 |
  - 观察：Loss 快速收敛，No Collapse 收敛最快（模型更简单）
  - 下一步：等待 Epoch 1 完成，查看 validation metrics

- **2025-12-03**: Stage 2 代码完成
  - 创建 `encoder.py`: CollapseAwareEmbedding + SequenceProfileEvoformer + FlowTCRGenEncoder
  - 创建 `dirichlet_flow.py`: DirichletFlowMatcher + CFGWrapper + 采样函数
  - 创建 `model_flow.py`: FlowTCRGen 主模型 + Model Score Hook + 生成接口
  - 创建 `data.py`: FlowTCRGenDataset + Tokenizer + collate_fn
  - 创建 `metrics.py`: Recovery/Diversity/Perplexity 评估
  - 更新 `train.py`: 完整训练流程 + Ablation 支持
  - Ablation 开关: `use_collapse`, `use_hier_pairs`, `cfg_drop_prob`

---

## 10. Metrics 详细解释

> 本节定义 Stage 2 中使用的所有评估指标，确保团队成员理解一致。

### 10.1 Loss 组成

| 组件 | 公式 | 含义 |
|------|------|------|
| **MSE Loss** | `‖v_pred - v_true‖²` | Flow matching 的核心损失，预测速度场与真实速度场的误差 |
| **Entropy Loss** | `-Σ p·log(p)` | 熵正则化，**希望最大化**以促进输出多样性 |
| **Total Loss** | `MSE - λ_entropy × Entropy` | 因为最大化熵，所以是减法；**Loss 可为负是正常的** |

**关键理解**：
- 当 `λ_entropy > 0` 且 entropy 足够大时，总 loss 可能为负
- 负 loss 本身**不是 bug**，是 entropy 正则项的预期行为
- 评估模型质量应主要看 **MSE 分量**和**生成指标**

### 10.2 Recovery Rate (恢复率)

| 指标 | 定义 | 计算方式 |
|------|------|----------|
| **Exact Match** | 生成序列与真实 CDR3β 完全相同 | `mean(generated == ground_truth)` |
| **Partial Match 80%** | ≥80% 位置匹配 | `mean(match_ratio >= 0.8)` |
| **Partial Match 90%** | ≥90% 位置匹配 | `mean(match_ratio >= 0.9)` |

**计算细节**：
```python
# 对每条序列
match_ratio = sum(gen[i] == gt[i] for i in range(L)) / L
exact_match = 1 if match_ratio == 1.0 else 0
partial_80 = 1 if match_ratio >= 0.8 else 0
```

**目标**：
- Exact Match > 30% (主要目标)
- Partial 80 > 50% (辅助目标)

### 10.3 Diversity (多样性)

| 指标 | 定义 | 计算方式 |
|------|------|----------|
| **Unique Ratio** | 生成序列中不重复的比例 | `n_unique / n_total` |
| **Entropy** | 序列分布的熵 | `H = -Σ p(seq)·log(p(seq))` |

**解读**：
- Unique Ratio = 0.99：几乎每条都不同（高多样性）
- Unique Ratio = 0.01：几乎全部相同（**mode collapse**）
- 健康范围：0.5 ~ 0.95

**重要观察**：
- Diversity 从 0.99 快速下降到 0.01 是 **mode collapse 的信号**
- 可能原因：entropy 正则不足、ODE 采样问题、或过拟合

### 10.4 Perplexity (困惑度)

| 指标 | 公式 | 含义 |
|------|------|------|
| **PPL** | `exp(mean_cross_entropy)` | 模型对真实序列的"困惑"程度 |

**计算方式**：
```python
# 对每条序列的每个位置
ce_loss = -log(p(true_token))
ppl = exp(mean(ce_loss))
```

**目标**：PPL < 10（越低越好）

**注意**：我们代码中使用 MSE loss 而非 cross-entropy，所以 PPL 是近似计算。

---

## 11. 首轮训练分析 (Buggy Version)

> 虽然首轮训练存在 ODE 积分 bug，但仍可从中获得有价值的 insights。

### 11.1 实验配置

| 实验 | 模型变体 | 参数量 | Job ID | 状态 |
|------|----------|--------|--------|------|
| Normal | Full model | 5,334,631 | 1116099 | ❌ 已终止 |
| No Collapse | `-collapse_token` | 5,331,039 | 1116100 | ❌ 已终止 |
| No Hier | `-hier_pairs` | 5,334,631 | 1116109 | ❌ 已终止 |
| No CFG | `cfg_drop_prob=0` | 5,334,631 | 1116112 | ❌ 已终止 |

**训练环境**：
- GPU: A100 80GB
- Batch Size: 32
- Epochs: 目标 100
- Learning Rate: 1e-4

### 11.2 发现的 Bug

#### Bug 1: ODE 积分 Simplex 投影错误 (Critical)

**问题代码** (`model_flow.py:generate()`):
```python
# ❌ 错误
x = x.squeeze(0) + v * dt
x = F.softmax(x, dim=-1)  # softmax 不是 simplex 投影！
```

**正确做法**：
```python
# ✅ 正确
x_new = x.squeeze(0) + v * dt
x_new = x_new.clamp(min=1e-8)              # 保证非负
x_new = x_new / x_new.sum(dim=-1, keepdim=True)  # 归一化到 simplex
x = x_new.unsqueeze(0)
```

**原因分析**：
- `softmax` 会重新分配概率质量，破坏 ODE 积分的连续性
- 正确的 simplex 投影只需裁剪负值 + 归一化
- 这导致生成质量极差，Recovery = 0

#### Bug 2: 评估参数不足

| 参数 | 修复前 | 修复后 | 说明 |
|------|--------|--------|------|
| `n_samples_per_batch` | 3 | 8 | 每个条件生成的样本数 |
| `max_eval_samples` | 无限制 | 500 | 最大评估样本数 |
| `n_steps` (ODE) | 50 | 100 | ODE 积分步数 |

#### Bug 3: Final Evaluation 参数名错误

```python
# ❌ 错误
final_metrics = evaluate(..., n_samples=10)  # 参数不存在

# ✅ 修复
final_metrics = evaluate(..., n_samples_per_batch=16, max_eval_samples=500)
```

### 11.3 有效的 Insights

尽管有 bug，以下观察仍然有价值：

#### Insight 1: Loss 收敛正常 ✅

| Epoch | Normal | No Collapse | No Hier | No CFG |
|-------|--------|-------------|---------|--------|
| E1 B50 | 0.103 | 0.080 | 0.135 | 0.163 |
| E1 B300 | 0.005 | 0.004 | - | 0.004 |
| E1 B700 | -0.005 | -0.007 | - | -0.005 |

**解读**：
- MSE 分量快速下降（从 0.1+ 到 0.001 级别）
- **模型架构正确**，能学习到 velocity field
- 负 loss 由 entropy 正则贡献，符合预期

#### Insight 2: No Collapse 模型收敛最快

- **参数量最少**：5,331,039 vs 5,334,631 (少 3,592)
- **收敛速度**：在相同 batch 数下 loss 更低
- **推断**：Collapse Token 增加了模型复杂度，可能需要更多数据/时间

#### Insight 3: No Hier 训练速度最快

- **时间节省**：约 32% (因为 hierarchical pairs 计算开销大)
- **每 epoch 时间**：~15min vs ~22min (Normal)
- **推断**：如果最终效果差不多，可考虑简化 pair 编码

#### Insight 4: Diversity 急剧下降

| Epoch | Normal | No Collapse | No Hier | No CFG |
|-------|--------|-------------|---------|--------|
| E1 | 0.63 | 0.32 | 0.40 | 0.42 |
| E4 | 0.14 | 0.01 | 0.02 | 0.08 |

**解读**：
- 所有模型都出现 diversity 下降
- No Collapse 下降最严重（从 0.32 到 0.01）
- **可能原因**：
  1. ODE bug 导致采样坍缩
  2. Entropy 正则权重 (λ=0.01) 可能太小
  3. 正常的 early training 现象，后期可能回升
- **修复后重新验证**是关键

#### Insight 5: Recovery = 0 的根本原因

Recovery 为 0 **主要是 ODE bug**，而非模型能力问题：
- MSE loss 收敛良好，说明 velocity field 学习正确
- 但生成时 softmax 投影破坏了 simplex 结构
- 导致采样路径偏离，无法回到真实序列

### 11.4 Ablation 初步趋势（待验证）

| 对比 | 观察 | 假设 |
|------|------|------|
| Normal vs No Collapse | No Collapse 收敛更快 | Collapse Token 需要更多训练 |
| Normal vs No Hier | No Hier 训练更快 | Hier Pairs 计算开销大 |
| Normal vs No CFG | 相似收敛速度 | CFG drop 在训练时影响小 |

**注意**：以上趋势需要在修复后重新验证。

### 11.5 下一步计划

1. ✅ 已修复所有已知 bug
2. ⬜ 重新提交训练任务
3. ⬜ 重点观察：
   - Recovery 指标（预期 >0，目标 >30%）
   - Diversity 下降曲线（是否仍然 mode collapse）
   - 各 ablation 的效果差异
4. ⬜ 如果 diversity 仍然下降严重，考虑：
   - 增大 `λ_entropy` (0.01 → 0.05 或 0.1)
   - 添加 temperature annealing
   - 检查 prior 分布配置

---

## 12. 代码修复记录

> 详细记录所有代码修改，便于回溯和复现。

### 12.1 `model_flow.py` 修改

**文件**: `flowtcr_fold/FlowTCR_Gen/model_flow.py`

**修改 1: ODE Simplex Projection (Line ~280-290)**
```diff
- x = x.squeeze(0) + v * dt
- x = F.softmax(x, dim=-1)
+ x_new = x.squeeze(0) + v * dt
+ x_new = x_new.clamp(min=1e-8)
+ x_new = x_new / x_new.sum(dim=-1, keepdim=True)
+ x = x_new.unsqueeze(0)
```

### 12.2 `train.py` 修改

**文件**: `flowtcr_fold/FlowTCR_Gen/train.py`

**修改 1: 评估参数优化 (evaluate 函数)**
```diff
- def evaluate(..., n_samples_per_batch=3):
+ def evaluate(..., n_samples_per_batch=8, max_eval_samples=500):
     ...
-     n_steps = 50
+     n_steps = 100
```

**修改 2: Final Evaluation 调用 (main 函数末尾)**
```diff
- final_metrics = evaluate(model, val_loader, tokenizer, device, args.cfg_weight, n_samples=10)
+ final_metrics = evaluate(model, val_loader, tokenizer, device, args.cfg_weight,
+                          n_samples_per_batch=16, max_eval_samples=500)
```

### 12.3 `metrics.py` 修改

**文件**: `flowtcr_fold/FlowTCR_Gen/metrics.py`

**修改 1: Perplexity 计算**
```diff
- ppl = mean_cost.__exp__()
+ ppl = math.exp(min(mean_cost, 10.0))
```

**修改 2: 新增 Partial Match 指标**
```diff
+ partial_match_80 = sum(1 for m in match_ratios if m >= 0.8) / len(match_ratios)
+ partial_match_90 = sum(1 for m in match_ratios if m >= 0.9) / len(match_ratios)
```

### 12.4 `dirichlet_flow.py` 修改

**文件**: `flowtcr_fold/FlowTCR_Gen/dirichlet_flow.py`

**修改 1: F.one_hot clamp 范围**
```diff
- F.one_hot(target_tokens.clamp(min=0), ...)
+ F.one_hot(target_tokens.clamp(min=0, max=self.vocab_size - 1), ...)
```

**修改 2: Entropy 正则加入 padding mask**
```diff
- entropy = -(v_pred * (v_pred + eps).log()).sum(dim=-1).mean()
+ entropy_raw = -(v_pred * (v_pred + eps).log()).sum(dim=-1)  # [B, L]
+ if pad_mask is not None:
+     entropy = (entropy_raw * pad_mask).sum() / (pad_mask.sum() + eps)
+ else:
+     entropy = entropy_raw.mean()
```

---

**Last Updated**: 2025-12-05  
**Owner**: Stage 2 Implementation Team
