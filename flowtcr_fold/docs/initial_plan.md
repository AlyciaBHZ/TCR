这是一份经过深度重构的 **METHODOLOGY.md**。

它完全剥离了我们之前的对话背景（如版本迭代、工具取舍的讨论），而是作为一份**独立、严谨的技术文档**，直接阐述现有的模型架构、设计理念（Rationale）以及具体的实施细节。它详细定义了 InfoNCE、EvoDesign Profile 等核心概念，适合任何技术背景的读者阅读。

-----

# Project Methodology: Physics-Grounded Generative TCR Design

## 1\. Overview & Design Rationale

本项目提出了一种全新的 TCR-pMHC 结合特异性设计框架，旨在解决传统深度学习方法普遍存在的“幻觉问题”和“物理合理性缺失”问题。

我们的核心理念是构建一个 **Hybrid System (混合系统)**，将数据驱动的生成能力与第一性原理（First-Principles）的物理约束深度融合：

1.  **Generative Backbone:** 利用 **Discrete Flow Matching** 取代传统的自回归生成，实现对全局序列空间的更高效探索。
2.  **Physical Grounding:** 引入 **EvoDesign** 体系（EvoEF2 能量函数 + 结构 Profile），不仅作为后处理的筛选器，更作为先验条件（Prior）和可微监督信号（Surrogate Loss）直接参与生成过程。
3.  **Explicit Topology:** 在通用蛋白质语言模型（ESM-2）的基础上，显式注入 TCR-pMHC 特异性的拓扑偏置，解决通用模型对复合物界面语义理解不足的问题。

-----

## 2\. Data Infrastructure: The "Hard Negative" Pipeline

高质量的表征学习依赖于高质量的负样本。为了防止模型仅学习到 V-gene 家族的简单统计规律，我们构建了基于 **Triplets (三元组)** 的数据流。

### 2.1 Hard Negative Mining Strategy

我们定义三元组 $(x_{anchor}, x_{pos}, x_{neg})$，其中负样本 $x_{neg}$ 通过以下策略动态生成，以逼迫模型学习精细的残基级特征：

  * **Decoy Epitope (Type A):** 保持 TCR 序列不变，将 Peptide 替换为与真实 Peptide 序列相似度极高（\>80%）但实验验证不结合的序列。
  * **Decoy TCR (Type B):** 保持 pMHC 不变，选择与真实 TCR 具有相同 V/J 基因家族，但在 CDR3 区域存在 2-3 个关键点突变的非结合克隆。

### 2.2 Physics-Based Data Cleaning

为了消除 PDB 晶体结构中的实验噪音（如原子碰撞、不合理的键长），所有训练用结构数据在预处理阶段均通过 **EvoEF2** 进行物理修复：

  * **Operation:** `RepairStructure`
  * **Purpose:** 优化侧链构象，最小化局部自由能，为结构预测模块提供“物理完美”的 Ground Truth。

-----

## 3\. Module I: Immuno-PLM (Hybrid Encoder)

该模块作为整个系统的“感知中枢”，负责提取富含物理和语义信息的特征表征。

### 3.1 Architecture: Hybrid Embedding

单纯的 ESM-2 虽然具备通用蛋白质物理知识，但缺乏对 TCR-pMHC 复合物多链拓扑的显式理解。我们提出 **Hybrid Embedding** 策略：

$$Z_{final} = Z_{ESM} + Z_{Topology}$$

1.  **Generic Physics ($Z_{ESM}$):**
      * 基于 **ESM-2 (650M)** 的预训练权重。提取其 Pair Representation（Outer Product），捕捉氨基酸层面的共进化信息。
2.  **Explicit Topology Bias ($Z_{Topology}$):**
      * **Rationale:** 显式编码“谁是受体、谁是配体”以及“链内 vs 链间”的相互作用。
      * **Implementation:** 复用我们在 `model.py` 中定义的拓扑编码器，将 Region ID（CDR1/2/3, MHC, Peptide）映射为可学习的 Pair Bias 矩阵。

### 3.2 Training Objective: InfoNCE Loss

为了将特异性结合信息注入 Embedding 空间，我们采用 **InfoNCE (Noise Contrastive Estimation)** 损失函数。

**Definition:**
InfoNCE 是一种对比学习损失，旨在最大化正样本对的互信息下界。对于一个 Anchor TCR $z_a$，正样本 $z_p$ 和一组负样本 $\{z_{n1}, z_{n2}, ...\}$，损失定义为：

$$L_{InfoNCE} = - \log \frac{\exp(sim(z_a, z_p) / \tau)}{\exp(sim(z_a, z_p) / \tau) + \sum_{k} \exp(sim(z_a, z_{nk}) / \tau)}$$

  * **Rationale:** 通过引入 Hard Negatives（$z_{nk}$），迫使 Embedding 空间不仅能聚类同源序列，还能区分微小突变导致的结合力丧失。
  * **$\tau$ (Temperature):** 控制分布的平滑程度，调节模型对困难样本的关注度。

-----

## 4\. Module II: TCRFold-Light (Geometric Critic)

该模块是系统的“几何裁判”，负责快速预测结构并提供可微的物理能量评估。

### 4.1 Architecture: MSA-free Evoformer

鉴于推理速度的需求，我们摒弃了昂贵的 MSA 构建过程，利用 Immuno-PLM 的强表征能力驱动轻量级几何模块。

  * **Backbone:** 基于 `Evoformer.py` 的精简版。
  * **Key Modification:** 移除了依赖 MSA 的 Row/Column Attention 层，保留处理 Pair 特征的 **Triangle Multiplicative Update** 和 **Triangle Attention**。
  * **Input:** Immuno-PLM 输出的 $Z_{final}$。

### 4.2 Energy Surrogate Head

为了将不可微的物理能量计算引入梯度下降过程，我们在 TCRFold-Light 顶层增加了一个 **Energy Head**。

  * **Function:** $E_{pred} = MLP(Structure\_Features)$
  * **Supervision:** 使用 **EvoEF2** 计算的真实物理能量（Binding Energy）作为标签进行回归训练。
  * **Rationale:** 使生成器能够通过反向传播感知“高能惩罚”，从而避免生成存在严重空间冲突（Clash）的序列。

-----

## 5\. Module III: FlowTCR-Gen (Physics-Guided Generator)

这是核心生成模块，采用 **Discrete Flow Matching** 范式，并集成了 **EvoDesign** 的先验信息。

### 5.1 Generative Paradigm: Discrete Flow Matching

我们采用 Dirichlet Conditional Flow Matching，在概率单纯形（Simplex）上对离散序列进行连续建模。

  * **Process:** 定义一个从先验分布（如均匀分布）到真实数据分布的向量场 $v_t$。模型学习预测该向量场，从而通过 ODE 求解生成序列。
  * **Why Flow:** 相比自回归（Autoregressive）模型，Flow 具有非因果的全局视野，能更好地协调 CDR3 首尾的结构约束。

### 5.2 Conditioning: EvoDesign Structural Profile

借鉴 EvoDesign 的思想，我们认为**结构决定序列偏好**。

  * **Structural Profile (PSSM):** 对于给定的 V/J Scaffold，我们利用 `TM-align` 在 PDB 库中检索相似结构，构建位置特异性打分矩阵（PSSM）。
  * **Integration:** 将此 PSSM 作为 **Strong Prior** 拼接到 Flow Network 的输入条件中。
  * **Rationale:** 利用进化信息限制搜索空间，防止模型生成生物学上完全不合理的氨基酸组合。

### 5.3 Loss Function

总损失函数显式包含了物理约束：

$$L_{Total} = L_{Flow} + \lambda_{align} L_{AttnAlign} + \lambda_{energy} L_{EnergySurrogate}$$

  * **$L_{Flow}$:** 向量场匹配损失。
  * **$L_{AttnAlign}$:** 几何感知注意力损失。强迫 Flow Network 的 Cross-Attention Map 拟合 TCRFold-Light 预测的 Contact Map。
  * **$L_{EnergySurrogate}$:** 物理能量惩罚。利用 4.2 节的 Energy Head 对生成序列的潜在能量进行约束。

-----

## 6\. Inference Pipeline: The Self-Correcting Loop

推理过程是一个从“全局搜索”到“局部分子力学优化”的漏斗模型：

1.  **Coarse Generation (Global Search):**
      * FlowTCR-Gen 基于 pMHC 和 EvoDesign Profile 快速采样生成初始候选序列池。
2.  **Geometric Filtering (Fast Screen):**
      * TCRFold-Light 快速预测候选序列的结构和 Confidence (pLDDT)。剔除结构崩塌的序列。
3.  **Physics Refinement (Local Optimization):**
      * 对筛选后的 Top 序列，调用 **EvoEF2** 进行 **Monte Carlo Sidechain Repacking**。
      * **Rationale:** 固定骨架，仅优化侧链构象以消除微小的原子碰撞，并计算精确的 Binding Affinity ($\Delta \Delta G$)。
4.  **Final Ranking:**
      * 依据 Refined Structure 的物理结合能进行最终排序。

-----

