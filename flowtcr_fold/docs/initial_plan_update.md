# FlowTCR-Fold: 基于物理法则的生成式 TCR 设计框架 (Project Methodology)

## 1\. 总览与设计理念 (Overview & Design Rationale)

本项目旨在建立一套全新的 TCR-pMHC 特异性设计框架，解决传统深度学习模型存在的“幻觉”和“缺乏物理合理性”问题。我们将 **EvoDesign** 的经典进化/物理方法升级为 **Deep Generative 2.0** 版本。

**核心支柱 (Core Pillars):**

1.  **混合智能 (Hybrid Intelligence):** 我们不单一依赖数据拟合。系统结合了 **Discrete Flow Matching (离散流匹配)** 的全局搜索能力、**ESM-2** 的通用语义理解，以及 **EvoDesign (EvoEF2 & Structural Profiles)** 的物理第一性原理。
2.  **PSI-Model 拓扑感知 (PSI-Informed Topology):** 通用模型（如 ESM）缺乏对受体-配体复合物拓扑的显式理解。我们直接迁移了 **psi\_model** 中验证过的 **Hierarchical Pairwise Embedding (层级成对嵌入)** 和 **Collapse Token** 机制，赋予模型明确的“观察者-目标”几何感知能力。
3.  **对比特异性 (Specificity via Contrast):** 引入 **InfoNCE** 对比学习，迫使模型在特征空间中拉开真实结合者与高相似度非结合者（Hard Negatives）的距离，从而学习到精细的特异性决定法则。

-----

这是一个完全面向代码落地的实施计划。我省去了所有虚的时间规划，直接切入代码架构、数据流、模型逻辑和训练策略。

这份文档旨在指导你如何利用现有的代码资产（conditioned/src/Evoformer.py, model.py 等）构建 v3.0 系统。

FlowTCR-Fold v3.0 工程落地手册
0. 核心架构与代码复用策略
我们不重造轮子，而是将现有组件重组。

Encoder: ESM-2 (通用物理) + Legacy model.py (拓扑偏置) -> Immuno-PLM

Structure: Legacy Evoformer.py (去 MSA) -> TCRFold-Light

Generator: New Code -> FlowTCR-Gen (Discrete Flow)

第一步：数据基础设施 (Data Infra)
目标： 构建支持“困难负样本挖掘 (Hard Negative Mining)”的数据加载器。

1.1 数据源准备
我们需要三类数据：

通用 PDB (PPI): 用于结构模块预训练。

行动： 下载 PDB，筛选双链/多链复合物（去除单链），按 interface area > 400Å² 过滤。

TCR Repertoire: 使用你上传的 pretrain_TCR/pretrained_model/data/tcrdb/processed_tcrdb.csv。

Paired TCR-pMHC: 使用 data/trn.csv 和 STCRDab 数据。

1.2 FlowDataset 类实现
继承或重写 conditioned/data.py。

输入逻辑： 不再是简单的 Pair，而是 Triplet (Anchor, Positive, Negative)。

Hard Negative 逻辑 (关键)：

在线挖掘 (On-the-fly): 在 __getitem__ 中，有 30% 概率不随机采样负样本，而是：

取同一个 V-gene family 但 CDR3 序列差异大的 TCR（模拟同源但不结合）。

或者取序列相似度 > 80% 但 Label 为 0 的 pMHC（模拟 Decoy）。

Tokenization:

扩展 ESM Tokenizer，增加 [TRA], [TRB], [PEP], [MHC] 四个特殊 token。

序列拼接格式：[CLS][PEP]...[SEP][MHC]...[SEP][TRA]...[SEP][TRB]...


## 2\. 数据基础设施：困难负样本流 (The "Hard Negative" Pipeline)

为了训练模型对微小突变敏感，我们摒弃简单的正样本训练，构建基于 **三元组 (Triplets)** 的数据流 $(x_{anchor}, x_{pos}, x_{neg})$。

### 2.1 困难负样本挖掘策略 (Hard Negative Mining)

负样本 $x_{neg}$ 不是随机生成的，而是为了“欺骗”模型而精心构造的：

  * **Decoy Epitope (Type A):** 保持 TCR 序列不变，将 Peptide 替换为与真实抗原序列相似度极高 (\>80%) 且结合同一 MHC 等位基因，但实验验证**不结合**的序列。
  * **Decoy TCR (Type B):** 保持 pMHC 不变，选择一个具有相同 V/J 基因家族，但在 CDR3 区域存在 2-3 个关键点突变的**非结合** TCR 克隆。

### 2.2 基于物理的数据清洗

PDB 原始结构包含实验噪音（如原子碰撞）。

  * **工具:** **EvoEF2** (组内资源)。
  * **流程:** 所有训练用结构均经过 `RepairStructure` 预处理，优化侧链排布并最小化局部自由能，为结构模块提供“物理完美”的 Ground Truth。

-----

## 3\. 模块一：Immuno-PLM (混合感知编码器)

该模块是系统的“感知中枢”。它不仅融合了 ESM 的通用知识，更重要的是集成了我们在 **psi\_model** 中提出的 **Hierarchical Pairwise Embedding** 架构。

### 3.1 架构：ESM-2 + PSI-Model 层级成对嵌入

我们构建一个复合表征 $Z_{final}$：

$$Z_{final} = Z_{ESM} + Z_{PSI\_Hierarchical}$$

1.  **通用物理特征 ($Z_{ESM}$):**

      * 基于 **ESM-2 (650M)** 预训练权重。提取残基级特征并计算外积 (Outer Product)，捕捉通用的氨基酸共进化信号。

2.  **PSI-Model 层级拓扑偏置 ($Z_{PSI\_Hierarchical}$):**

      * **Source:** 迁移自 `psi_model/model.py` 中的 `create_hierarchical_pairs` 逻辑。
      * **Rationale:** 我们引入一个全局 **Collapse Token (聚合 Token)** 作为观察者中心，并定义了精细的分层交互逻辑，显式编码复合物的几何语义：
          * **Level 0:** Collapse Token 自指 (Self-reference)。
          * **Level 1:** Collapse Token $\leftrightarrow$ 任意区域 (Observer-Observed 关系)。
          * **Level 2:** 目标链 (Heavy Domain) 内部的序列邻近关系 (Sequential Neighbors)。
          * **Level 3:** 目标链内部的非邻近关系 (长程依赖)。
          * **Level 4 (关键):** 目标链 $\leftrightarrow$ 条件区域 (Target-Context，如 CDR3与Peptide的交互)。
          * **Level 5+:** 条件区域内部及条件区域之间的交互。
      * **Implementation:** 这些层级 ID 通过 `pair_embed_lvl1/2` 线性层投影为 Embeddings，直接注入到几何模块中，引导模型关注关键的受体-配体界面。

### 3.2 训练目标：InfoNCE 损失

为了学习结合能级差 (Energy Gap)，我们在全局序列序表征上应用 **InfoNCE (Noise Contrastive Estimation)**。

**定义:**
给定锚点 $z_a$，正样本 $z_p$，以及一组困难负样本 $\mathcal{N} = \{z_{n1}, z_{n2}, ...\}$：

$$L_{InfoNCE} = - \log \frac{\exp(\text{sim}(z_a, z_p) / \tau)}{\exp(\text{sim}(z_a, z_p) / \tau) + \sum_{z_n \in \mathcal{N}} \exp(\text{sim}(z_a, z_n) / \tau)}$$

  * **作用:** 该损失函数将真实结合者的 Embedding 拉近，同时剧烈推开 Decoy 的 Embedding。这是模型理解“差之毫厘，失之千里”的关键。

-----

## 4\. 模块二：TCRFold-Light (几何裁判与能量代理)

一个轻量级的、去 MSA 的结构预测模块，复用自我们现有的 Evoformer 代码，充当可微的“几何裁判”。

### 4.1 架构：MSA-Free Evoformer

  * **Source:** `psi_model` 或 `conditioned/src/Evoformer.py`。
  * **适配:** 移除计算昂贵的 `MSAColumnAttention` 和 `MSARowAttention`。
  * **核心:** 保留 **Triangle Multiplicative Update** 和 **Triangle Attention**。这些模块专门用于处理来自 Immuno-PLM 的高质量 **PSI-Model Pair Features ($Z_{final}$)**，实现毫秒级推理。

### 4.2 能量代理头 (Energy Surrogate Head)

为了打通深度学习与物理能量图景，我们在结构模块顶层增加一个回归头。

  * **预测:** $E_{pred} = \text{MLP}(\text{Structure Features})$
  * **监督目标:** 由 **EvoEF2** 计算的真实物理结合能 ($\Delta \Delta G$)。
  * **价值:** 这使得物理能量变得**可微 (Differentiable)**，允许我们在生成过程中反向传播“高能惩罚”，直接抑制不合理的构象。

-----

## 5\. 模块三：FlowTCR-Gen (物理引导生成器)

核心生成引擎，采用 **Discrete Flow Matching** 范式，并内嵌 **EvoDesign** 的进化先验。

### 5.1 生成范式：Discrete Flow Matching

取代传统的自回归 (Autoregressive) 解码（速度慢且缺乏全局视野），我们使用 **Dirichlet Conditional Flow Matching**。

  * **状态空间:** 氨基酸的概率单纯形 (Probability Simplex)。
  * **向量场:** 网络学习一个向量场 $v_t(x, t)$，将先验噪声分布流向生物序列分布。

### 5.2 条件输入：EvoDesign 结构 Profile

我们将 EvoDesign 的核心概念“结构决定序列偏好”整合为**强先验 (Strong Prior)**。

  * **构建:** 对于给定的 V/J 骨架，利用 `TM-align` 在 PDB 中搜索相似结构，构建 **PSSM (位置特异性打分矩阵)**。
  * **整合:** 该 PSSM 被拼接到输入噪声中。它将生成流限制在进化保守的流形内，防止生成结构上不可能的序列。

### 5.3 物理引导损失函数

$$L_{Total} = L_{Flow} + \lambda_{align} L_{Attention} + \lambda_{energy} L_{Surrogate}$$

  * **$L_{Flow}$:** 标准的流匹配损失。
  * **$L_{Attention}$:** **几何感知注意力 (Geometry-Aware Attention)**。强迫 Flow 网络的 Cross-Attention 图与 TCRFold-Light 预测的 Contact Map 对齐（基于 Legacy 逻辑：“注意力即接触”）。
  * **$L_{Surrogate}$:** 惩罚具有高预测能量的生成序列（通过能量头）。

-----

## 6\. 推理流水线：自校正循环 (The Self-Correcting Loop)

推理过程是一个从“全局搜索”到“局部分子力学优化”的漏斗模型：

1.  **粗粒度生成 (Global Search):**
      * 输入: pMHC + V/J 骨架 + EvoDesign Profile。
      * 输出: FlowTCR-Gen 快速采样生成候选序列池 (N=1000)。
2.  **几何筛选 (Geometric Critic):**
      * 运行 TCRFold-Light。
      * **过滤:** 剔除低置信度 (pLDDT) 或界面接触密度差的序列。
3.  **物理精修 (EvoDesign Refinement):**
      * 输入: Top 100 筛选后的序列。
      * **动作:** 固定骨架，仅对界面残基运行 **EvoEF2** 的侧链 Repacking (Monte Carlo)。
      * **目标:** 消除微小的原子碰撞，计算精确的结合能。
4.  **最终排序:**
      * 依据 EvoEF2 的物理结合能输出最终设计。

-----

## 7\. 代码实施架构 (Implementation Framework)
