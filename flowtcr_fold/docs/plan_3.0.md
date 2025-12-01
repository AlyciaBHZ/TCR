我直接帮你把现在所有东西整合成一个「总规划 v3.1」. 包含
三个 Stage 的最终定位


每个 Stage 的「实用版本」和「探索版本」


Stage 3 用你最新的 TCRFold‑Prophet 方案重写


一条可以照着走的执行时间线


你后面只要在这个大框架里迭代就行。

0. 总体目标和三大 Stage 分工
总体目标:
 给定目标 pMHC 序列和 MHC allele
Stage 1 给出一组生物学合理的 V/J scaffold 先验


Stage 2 在选定 scaffold 上生成多样的 CDR3β 序列


Stage 3 将候选 TCR 全部折叠成 TCR‑pMHC 结构, 用几何和物理双重标准筛选


三者分工:
Stage 1: 统计语义先验. 轻量, MHC 条件的 V/J 分布建模


Stage 2: 核心创新. 拓扑感知的离散 Flow 生成


Stage 3: 结构与物理验证. 类 tFold‑TCR 的单序列折叠 + EvoEF2 对齐


每个 Stage 都拆成
Practical 版本: 论文主线可用的最小闭环


Exploratory 版本: 有时间就加, 不会卡住主线



1. Stage 1: Immuno‑PLM 统筹规划
定位:
先明确一个原则
Stage 1 是“scaffold 先验模块”


核心任务是建模

 p(V gene, J gene | MHC, peptide)

 但我们知道从生物学上


条件里 MHC 是“强信号”, peptide 是“弱修正”


CDR3β 虽然和 peptide 强相关, 但那是 Stage 2 的主战场


所以在 Stage 1 里
CDR3β 不作为主要输入, 只在统计和分析中使用


MHC 与 V/J 的关系是主线, peptide 主要看有没有增益


可以按功能把字段分一下
字段
建议角色
peptide
作为条件的一部分, 参与 pMHC embedding, 也用于 grouping
mhc
作为条件的一部分, 同时抽 allele name 作为离散特征
mhc_sequence
主输入序列的一部分, 喂给 ESM
h_v, h_j, l_v, l_j
作为多标签 BCE 的监督, 多 hot target
h_v_sequence, h_j_sequence, l_v_sequence, l_j_sequence
作为对比学习 InfoNCE 的“另一侧”序列, 构 scaffold bank
cdr3_b
用于统计分析 (peptide→CDR3 多样性), Stage 1 训练本身可以不直接用

这样你在 Stage 1 内部就有两个互补的信息通路
“序列级别”的 pMHC ↔ scaffold 序列 InfoNCE


“类别级别”的 pMHC ↔ gene id 多标签分类


这其实就是最大化利用了 “sequence + id” 两种视角.

1.1 模型与输入
Backbone: esm2_t33_650M_UR50D


LoRA adapter: rank 16, alpha 32, 作用到 [query, key, value, dense]


输入格式:

 <MHC_ALLELE_EMB> MHC_Sequence Peptide_Sequence [SEP] (可选其他提示)

 其中 <MHC_ALLELE_EMB> 是一层独立的 allele embedding, 编码 HLA-A*02:01 这类标签。
<MHC_ALLELE_EMB>: 一个 lookup 表, 从 mhc id (如 "HLA-A*02:01") 映射到向量


mhc_sequence, peptide 正常用 ESM vocab tokenize


得到
z_pmhc_seq = ESM 输出的 CLS 或平均池化


再过两层投影得到最终的 z_pmhc 用于 InfoNCE 和 BCE


这样就把
MHC 作为强信号通过序列+allele embedding 双重进入


peptide 作为弱信号, 放在后半段序列, 由 ESM 自己学习它对整体 embedding 的修饰作用


2.2 scaffold 侧: V/J 序列和 id
对每条样本你有:
h_v_sequence, h_j_sequence, l_v_sequence, l_j_sequence


h_v, h_j, l_v, l_j (id)


处理方式:
对序列, 用同一个 ESM 编码器编码


可以 share 参数, 只是加不同的 chain type embedding 表示 HV/HJ/LV/LJ


z_hv_seq = enc_v(h_v_sequence_tokens)  # CLS pooling
z_hj_seq = enc_j(h_j_sequence_tokens)
...
 这些 z_*_seq 用于 InfoNCE.


对 id, 用独立的 embedding table 或 one‑hot


只出现在多标签 BCE 的输出空间里


输入仍然是 z_pmhc



3. InfoNCE + 多标签 BCE 的具体设计
3.1 多正样本 InfoNCE: 序列视角
目的: 学一个“pMHC ↔ 某条 V/J 序列相容性”的连续表示.
目标对:
(z_pmhc, z_hv_seq)


(z_pmhc, z_hj_seq)


(z_pmhc, z_lv_seq)


(z_pmhc, z_lj_seq)


LV/LJ 缺失就直接 mask 掉.
多正样本的构法
你现在的 group 其实可以有两种选法:
按 (peptide, MHC) 分组


按 (MHC) 分组


既然我们知道
MHC ↔ V/J 是强关系


peptide ↔ V/J 是弱关系


可以这样做:
主 InfoNCE 用 (MHC) 分组


辅 InfoNCE 用 (peptide, MHC) 分组


写成伪代码就是两条 loss:
# group by MHC
group_mhc = (mhc_id)               # 同 allele 归为一组
pos_mask_mhc = same_group(group_mhc)


loss_nce_mhc_hv = multi_pos_infonce(z_pmhc, z_hv_seq, pos_mask_mhc)
...


# group by peptide+MHC
group_pmhc = (peptide_hash, mhc_id)
pos_mask_pmhc = same_group(group_pmhc)


loss_nce_pmhc_hv = multi_pos_infonce(z_pmhc, z_hv_seq, pos_mask_pmhc)
...


loss_nce_total = loss_nce_mhc_* + λ_pmhc * loss_nce_pmhc_*

loss_nce_mhc_* 体现的是“在同一个 MHC 下常见的 V/J”


loss_nce_pmhc_* 体现的是“同一个肽在某个 MHC 上对 V/J 的微调偏好”


λ_pmhc 可以一开始设小一点, 比如 0.3, 让 MHC 信号占主导


3.2 多标签 BCE: 类别分布视角
InfoNCE 是样本级别 pair 对齐, 它没有显式看到“在某个 MHC 下, V gene 全局分布长什么样”.
这里就轮到多标签 BCE:
构标签
对每个 (peptide, MHC) group 或 (MHC) group, 聚合该组出现过的 gene id 形成 multi‑hot:
# group-level aggregation
for each group g:
    hv_ids_in_g = set(h_v_samples_in_group)
    target_hv[g] = multi_hot(hv_ids_in_g, num_hv)


    # 同理构 h_j, l_v, l_j 的 multi-hot

对应输入是该 group 的 z_pmhc[g]
如果 group 用 (MHC) 聚合, 可以为每个 allele 统计一次


如果用 (peptide, MHC), 样本更细但每组数据更少


一个折中做法:
把 (MHC) 作为主监督对象: target_hv_mhc, target_hj_mhc,...


(peptide, MHC) 作为扩展. 用更小权重监督, 视情况而定


BCE 形式
模型输出:
logits_hv = W_hv @ z_pmhc + b_hv    # [num_hv]
logits_hj = ...
...

Loss:
loss_bce = BCEWithLogits(logits_hv, target_hv, pos_weight=class_weights)
        +  BCEWithLogits(logits_hj, target_hj, ...)
        +  ...

注意:
V/J gene 分布极度长尾, 建议加 pos_weight 或 focal loss


缺失的 LV/LJ 可以只在有数据的 group 中计算 loss, 没有就跳过


组合总 loss
总体上, Stage 1 的 loss 可以这样写:
L_total = L_NCE_MHC
        + λ_pmhc * L_NCE_pMHC
        + λ_bce  * L_BCE

注意可以写成 - pep条件作为input时候的指示来同时进行两种模型个 (peptide, MHC) group  (MHC) group，可以非常迅速的对比有一个结果



4. 关键指标具体怎么算
4.1 Top‑K 覆盖率
定义:
给定验证集中一个 (peptide, MHC) 或 (MHC) 条件, 模型输出对所有 V/J gene 的打分或排序, 看真实使用过的 gene 中有没有落入前 K 名.
做法:
先离线统计验证集里, 对每个条件 g 的“真实 gene 集合”

 true_hv[g] = set(all hv_ids in val where group_id == g)


模型 inference:


对每个条件 g, 计算 z_pmhc[g]


输出 logits_hv[g], 排序得到 top‑K gene 列表 topk_hv[g]


统计:

 hit@g = (true_hv[g] ∩ topk_hv[g] != ∅) ? 1 : 0
R@K = mean_g hit@g


同理对 HJ/LV/LJ.
注意区分两种条件:
使用 (MHC) 作为 group 时的 R@K


使用 (peptide, MHC) 作为 group 时的 R@K


可以列成一张表.
4.2 分布 KL: 模型分布 vs 经验分布
对于每个条件 g:
经验分布:

 count_hv[g, v] = 出现次数
p_emp[g, v] = count_hv[g, v] / sum_v count_hv[g, v]


模型分布:

 p_model[g] = softmax(logits_hv[g])


KL:

 KL[g] = sum_v p_emp[g, v] * log(p_emp[g, v] / (p_model[g, v] + eps))


最后取平均:
KL_mean = mean_g KL[g]

你可以比较:
模型 KL vs 简单 MHC‑频率基线 KL.


也可以比较有无 peptide 时的 KL 差异.


4.3 与“仅用 MHC 的基线”的比较
基线实际上有两种:
简单频率基线:


对每个 MHC allele, 直接用训练集统计的 V/J 频率作为分布


Top‑K 和 KL 都能直接计算


“只用 MHC 作为输入”的模型:


相同架构, 但输入去掉 peptide 序列


可以直接 share 模型, 只是在构输入时把 peptide mask 掉, 得到:


z_mhc_only = encoder(<MHC_ALLELE_EMB> + mhc_seq + <MASK_PEP>)


比较:

 R@K(pMHC model)  vs  R@K(MHC-only model)
KL(pMHC model)   vs  KL(MHC-only model)


如果 pMHC 模型明显好于 MHC‑only 和频率基线, 那就说明 peptide 确实提供了额外信息, 否则你也有生物学上合理的解释: “V/J 主要由 MHC 决定, peptide 影响有限”.

5. 小结一下 Stage 1 的“最佳用法”决策
综合你给出的生物事实和数据格式, 我建议 Stage 1 的最终设计可以写成:
数据使用分工


MHC 序列 + allele id. 作为主条件输入


peptide 序列. 作为辅助条件输入, 让模型有机会学到细粒度偏好


V/J 序列. 用于 InfoNCE, 构建 pMHC ↔ scaffold 序列的连续空间


V/J id. 用于多标签 BCE, 建模在给定条件下的 gene 使用分布


CDR3β. 不直接喂 Stage 1 模型, 主要用于统计和后续 Stage 2.


loss 设计


InfoNCE: 多正样本, 至少按 MHC 分组, 视精力再加 pMHC 分组


多标签 BCE: 按 MHC 或 pMHC 的 group 构 multi‑hot gene 分布, 让模型显式拟合经验分布


最终 loss = InfoNCE + λ·BCE, λ 视验证集指标微调.


关键指标


对 (MHC) 和 (peptide, MHC) 两种条件


Top‑K 覆盖率


KL(p_emp || p_model)


基线


训练集频率分布


MHC‑only 模型


这套设计基本就是: 在 Stage 1 里充分榨干你现有的字段, 又尊重了“peptide↔CDR3 强/MHC↔VJ 强”的生物学事实, 不和 Stage 2 的职责重叠.


输出对象:


V gene 序列 embedding


J gene 序列 embedding
 用同一个 ESM backbone 处理 scaffold 序列, 取 CLS pooling 投到低维空间。





2. Stage 2: FlowTCR‑Gen 统筹规划
定位:
 整条课题的核心创新模块. 在给定 pMHC 和 scaffold 的条件下生成 CDR3β 序列, 通过拓扑感知 Evoformer 和 Dirichlet Flow Matching, 实现高多样性且物理可控的序列设计。
2.1 架构组件
条件编码器: 使用 legacy psi_model 里的


CollapseAwareEmbedding


SequenceProfileEvoformer (MSA‑free 版本)


输入布局:

 [ψ, CDR3β, peptide, MHC, (可选其他区域)]


Pair 表征: 由 7‑level Hierarchical Pair IDs 转成 embedding


保留你原来定义的 level 0..6 拓扑


明确标注 CDR3β 与 peptide, 与 MHC 的特殊交互 id


2.2 前向与 Flow Head
完整的前向示意:
def forward(self, x_t, t, cond):
    # cond 包含 scaffold, peptide, MHC 序列及其 pair_ids 布局信息
    
    full_seq = build_full_seq(x_t, cond)         # [B, L, *]
    pair_ids = build_pair_ids(cond)              # [B, L, L]
    z0 = self.pair_embedder(pair_ids)            # [B, L, L, d_z]
    
    s0 = embed(full_seq)                         # token embedding + collapse token
    s, z = self.backbone(s0, z0)                 # Evoformer
    
    cdr3_repr = slice_cdr3(s, cond)              # [B, L_cdr3, d]
    v_pred = self.flow_head(cdr3_repr)           # [B, L_cdr3, 20]
    return v_pred

关键点: Evoformer 始终处理“完整拼接序列”, x_t 只替换 CDR3 区域的输入表示。
2.3 Dirichlet Flow Matching
状态定义:


x1: 真实 CDR3β 的 one‑hot 概率


x0: Dirichlet 均匀分布或温度较高的先验分布


对每个样本采样 t, 构造插值:

 x_t = (1 − t) x0 + t x1


训练目标:

 v_true = x1 - x0
v_pred = model(x_t, t, cond)              # 只在 CDR3 区域
L_flow = ((v_pred - v_true) ** 2).mean()


同时保留 legacy 中的正则项作为辅助:


Collapse attention entropy 正则 (鼓励 ψ 集中注意低熵位点)


Sequence profile regularization


总 loss 可以写成:
L_total = L_flow
        + λ_ent * L_collapse_entropy
        + λ_prof * L_profile_reg

2.4 CFG 与 Physics Guidance 的分期
Classifier‑Free Guidance:


训练时以 p=0.1 随机将 condition 置空, 得到 v_uncond


推理时:

 v_final = v_uncond + w * (v_cond - v_uncond)


w 从 1.0 开始, 视条件依赖强度微调


物理梯度 Guidance:


依赖 Stage 3 的能量 surrogate 可微


建议放在 FlowTCR‑Gen v2 或之后


第一版只用物理评分做后验重排或 Monte Carlo 搜索, 不直接参与 Flow 的训练


2.5 Stage 2 Practical 与 Exploratory
Practical 版本:


只用 Flow Matching loss + attention/profile 正则


不接能量梯度, 后验筛选交给 Stage 3


评估: 重构率, 多样性, 与训练集距离, 与结构能量分布对照


Exploratory 版本:


加入 Physics guidance


尝试 small‑step gradient guidance, 或与 Monte Carlo 混合

现在的 Stage 2 主干是:
psi_model 的 Collapse token + Hierarchical pair + Evoformer 做条件编码


Dirichlet Flow Matching 做 CDR3β 的连续生成


支持 CFG, 未来接物理梯度 guidance


我建议补两点小东西, 方便将来和 Stage 3 接:
显式保留一个“模型内部 energy/loglikelihood”接口

 哪怕是 Flow Matching, 也可以定义一个简单的 model score:


对采样出的 CDR3 反推一个 approximate NLL 或 “Flow cost”


或者定义一个 proxy: Collapse token 的某个标量投影, 作为“模型偏好”的打分


这样 Stage 3 在做 MC 时可以组合:

 E_total = α · E_phi  +  β · ModelScore


在代码层面把 sampling ODE 封装好

 方便后面直接在 ODE step 里插入
 - w ∇_x E_phi(x) 这一项, 不用大改结构.


Stage2 的东西就先这样, 不打断你现有的思路.


3. Stage 3: TCRFold‑Prophet 统筹规划

定位:
 
Stage 3 拆成四个层级:
Trunk: 一个通用 PPI 结构 encoder (Evoformer‑Single + IPA)


Head1: 结构头, 输出坐标/距离/接触


Head2: 能量头 E_φ, 可微的 EvoEF2 surrogate


上游使用场景:


Flow 里的 gradient guidance


MC 在 CDR3 离散空间的搜索


3.1 数据总布局: 两层数据集
Data A: General PPI 结构集
来源: PDB 中约 50k 各类蛋白质复合物


每个样本记录:


Seq_A, Seq_B: 两条或多条链的序列


Coords: 每个残基 N/CA/C/CB 坐标


接口残基标注, 接触图


EvoEF2:


ΔG_bind 作为 binding 能量标签


或者再加 ComputeStability 得到 stability 标签


Data B: TCR‑pMHC 特定结构集
来源: TCR3d + STCRDab


Subset 清洗后, 约几百到一千个高质量复合体


同样带:


TCRα/β, peptide, MHC 的序列和坐标


EvoEF2 计算的 binding ΔG


这两个数据集的用法:
A 用来让 trunk 和能量头学到通用的“物理能量场”.


B 用来做 TCR 场景的 finetune, 把 energy/结构专门适配到 TCR‑pMHC.



3.2 模型分解: TCRFold‑Prophet + Energy Surrogate
我们把 Stage3 模型视作一个统一的网络:
Input: (Seq, Coords_initial/模板)

ESM encoder → Evoformer-Single trunk → IPA 结构头 + Energy head

Trunk: Evoformer‑Single
输入:


SingleRep: residue embedding


可以是 ESM 的 per residue 表征 + chain type embedding


PairRep: 基于序列位置和链对的初始 pair 特征, 例如:


relative positional encoding


是否同链


简单几何先验(例如模版距离, 如果有)


结构:


N 层 triangle attention + pair update + single attention


完整借用你 psi_model 的 Evoformer 实现, 去掉 MSA 相关部分


Head1: 结构预测 (StructHead)
使用 Invariant Point Attention 生成每个残基的局部坐标系


输出:


坐标 coords_pred


distance map, contact map


训练时用:


FAPE


distogram loss


interface contact loss


Head2: 能量 surrogate E_φ
这里是你说的“GVP 或 IPA”部分的落点.
选项 A: 直接在 Evoformer 输出之上加一个 GNN/GVP:


节点特征:


SingleRep


predicted coordinates 或 真实坐标


边特征:


PairRep


残基间距离, 是否在接口等


通过几层 GVP 或 message passing 得到一个 graph-level embedding h


Energy head: E_phi = MLP(h), 输出标量能量


选项 B: 直接用 PairRep 池化, 不额外加复杂 GNN


更轻, 但物理 inductive bias 稍弱


你可以先用 B 起步, 后面再升级到 A


训练目标:
L_energy = MSE( E_phi(Seq, Struct) , E_EvoEF2(Seq, Struct) )

Struct 可以是:
预训练阶段: 真实 PDB 坐标


使用阶段: TCRFold‑Prophet 的预测坐标 (可微)



3.3 训练 pipeline: 结构预训 + 能量拟合 + TCR 微调
我建议如下分阶段:
Phase 3A. General PPI 结构预训练
作用: 让 trunk 学会“怎么折叠”和“什么样的 interface 是合理的”.
数据: Data A (General PPI)


网络: trunk + StructHead


Loss:

 L_struct = L_FAPE  +  0.3 · L_dist


能量头暂时不训练, 或者只做很弱的辅助.


输出: 结构合理的通用 PPI encoder.
Phase 3B. General PPI 能量 surrogate 训练
作用: 在通用结构的基础上拟合 EvoEF2 的“物理能量场”.
冻结大部分 trunk, 解冻 trunk 最后几层和 EnergyHead


数据: 仍然是 Data A


Loss:

 L_surrogate = MSE( E_phi , E_EvoEF2 )


可以加一点 regularization, 例如约束 E_phi 在小扰动下平滑.


可选增强:
为每个 PPI 生成若干“诱饵结构”和小突变序列:


coordinate noise


接口局部 random rot/trans


random mutation around interface


用这些扩充训练集, 让 E_phi 能分辨“看起来合理 vs 明显高能”的结构.


Phase 3C. TCR‑pMHC 微调与对齐
作用: 把通用 PPI 能量和结构针对 TCR‑pMHC 再对齐一遍.
数据: Data B (TCR3d + STCRDab)


网络: trunk + StructHead + EnergyHead 全部一起微调


Loss 综合:

 L_total = L_FAPE  +  0.3·L_dist
        +  λ_E · MSE(E_phi, E_EvoEF2)
        +  λ_reg · (结构正则/接触图 loss 等)


目的是:
结构头在 TCR‑pMHC 上表现好 (pLDDT proxy 高, 接口合理)


能量头在 TCR‑pMHC 上的预测与 EvoEF2 有高相关性


目标仍然是 >0.7 的 Pearson/Spearman


到这一步为止, 你就有了:
一个可微的 E_phi(Seq, Struct)


一个可微的 Struct = F_theta(Seq, pMHC)


两者组成一个“端到端的物理评分模块”.

3.4 如何把 E_φ 接到 Stage 2 的 Flow ODE 上
你给的公式是:
x_{t+Δt} = x_t + [ v_θ(x_t,t) - w ∇_{x_t} E_phi(x_t) ] Δt

要让这个可实现, 需要两件事:
让 E_phi 对 CDR3 的连续表示 x_t 可微


在 ODE 的每个 step 里估算结构, 再 eval 能量和梯度


具体落地方式:
把 CDR3 的 one‑hot 概率 x_t 通过 embedding 做成连续向量 s, 用在 TCRFold‑Prophet 里:

 # 1. 连续 relax 的 CDR3 表达
cdr3_embed = AA_embedding(x_t)          # [L_cdr3, d]
full_seq_embed = concat(cdr3_embed, scaffold_embed, pMHC_embed, ...)

# 2. 通过 trunk 预测结构
coords_pred, pair_rep, single_rep = TCRFoldProphet(full_seq_embed)

# 3. 通过 E_phi 预测能量
E = E_phi(single_rep, pair_rep, coords_pred)

# 4. 反向传播求 ∇_{x_t} E
E.backward()
grad_x = x_t.grad


在 Flow 的 ODE step 里:

 v_flow = v_theta(x_t, t, cond)
grad_E = grad_x                         # 从上面计算来
x_next = x_t + (v_flow - w * grad_E) * Δt


计算量很大, 所以策略上可以:
只在若干离散时间点加能量梯度


例如每 5 个 ODE step 算一次 E_φ 梯度


或者只对 top‑N 候选序列加物理 guidance, 其余只用 Flow


这一部分可以作为 Stage2+Stage3 的 Exploratory 版本, Practical 版本先只用 E_φ 做后验筛选, 不必须上 GUIDED ODE.

3.5 Monte Carlo 在这个框架里的位置
现在有:
Stage 2 的生成器, 可以给你一个初始 CDR3 分布


Stage 3 的 E_φ, 可以快速给出能量


还有原来 psiMonteCarloSampler 里已经写好的 MC 框架


很自然的用法有三层:
3.5.1 经典 simulated annealing, 把 E_φ 当能量
对固定的 pMHC 和 scaffold:
初始:


从 FlowTCR‑Gen 采样一条或多条 CDR3 作为起点


每个 MC step:

 candidate = propose_mutation(current)          # 单点突变/多点突变
E_curr = E_phi(current_seq, struct_curr)
E_cand = E_phi(cand_seq, struct_cand)         # 通过 TCRFold-Prophet 预测结构后 eval 能量

ΔE = E_cand - E_curr
if ΔE < 0 or rand() < exp(-ΔE/T): accept


Temperature T 按 schedule 降低.


优点:
E_φ 是毫秒级, 可以做很多 step, 远比每步都跑 EvoEF2 快.


最后对 best 序列再调用一次真实 EvoEF2 做精确校准即可.


3.5.2 Hybrid energy: 模型分数 + 物理能量
把 Stage2 的模型偏好也纳入能量:
E_total = α · E_phi(Seq, Struct) + β · NLL_model(Seq | pMHC, Scaffold)

α 控制物理强度


β 控制“不要偏离训练分布太远”


MC 框架不变, 只是 compute_energy 换掉.
这个组合很适合做:
Flow 采样后的小范围局部搜索


也适合训练中每隔 N epoch 用 MC 找一些“物理更优”的样本, 再回灌到模型训练里 (类似 self‑play).


3.5.3 用 E_φ 的梯度来改进 proposal 分布
你原来的 MC 是纯随机 propose. 有了 E_φ 的梯度, 可以做一点 heuristic:
在 continuous relax 的 CDR3 上做一个小的 gradient step, 找到能量下降最大的几个位置和氨基酸候选


然后在离散空间中, 只从这些候选位置与氨基酸组合里采样 propose, 而不是全局 random


这样 MC 仍然是离散接受, 但 proposal 更“聪明”, 收敛更快.
实现上:
先对当前 CDR3 做一轮 forward/backward 得到 ∇_x E_φ


找到梯度绝对值最大的若干位点 i


对这些位置枚举若干替换氨基酸, 形成候选集合


从候选集合里随机选一个作为 MC 的 propose


这一步完全可以封装在 propose_mutation 里, 不改变外面的 MC 逻辑.

3.6 Stage 3 完整 pipeline 小结
整合一下就是:
离线准备


Data A: General PPI 结构 + EvoEF2 能量


Data B: TCR3d + STCRDab 结构 + EvoEF2 能量


写好结构修复、解析和标注脚本


模型训练


Phase 3A: General PPI 上预训练 TCRFold‑Prophet trunk + StructHead


Phase 3B: 同一数据上训练或微调 EnergyHead, 拟合 EvoEF2


Phase 3C: 在 TCR‑pMHC 上 fine‑tune trunk + heads, 对齐结构与能量


推理与优化


For each pMHC + scaffold:


Stage 2 Flow 采样一批 CDR3 作为初始候选


用 TCRFold‑Prophet + E_φ 预测结构和能量, 做第一轮筛选


对 top‑N 序列运行 MC:


能量函数 E_φ 或混合 α·E_φ + β·NLL


搜索得到一批局部能量最优的 CDR3


对最终 top‑K 序列, 调用真实 EvoEF2 计算能量, 做最后验证


Exploratory: 与 Flow ODE 的深度融合


在 Flow 的 ODE 采样步骤中周期性地加入 - w ∇ E_φ 项


把 Stage 3 真正变成“训练时即考虑物理能量”的一部分, 而不仅仅是后验筛选与 MC.



-----------
## 一、总体评价

| 维度 | 评分 | 评价 |
|------|------|------|
| **概念完整性** | ⭐⭐⭐⭐⭐ | 三个 Stage 分工明确，逻辑自洽 |
| **技术可行性** | ⭐⭐⭐⭐☆ | 大部分可行，少数需要调整 |
| **实现复杂度** | ⭐⭐⭐☆☆ | 中高复杂度，需要合理排期 |
| **创新性** | ⭐⭐⭐⭐⭐ | 多处创新点，论文价值高 |
| **Practical/Exploratory 划分** | ⭐⭐⭐⭐⭐ | 非常好，保证了主线不被阻塞 |

**总体结论**: ✅ **高度可行**，建议按此计划执行。

---

## 二、Stage 1 可行性分析

### ✅ 完全可行的部分

| 设计点 | 可行性 | 理由 |
|--------|--------|------|
| ESM-2 + LoRA (rank 16) | ✅ | 标准做法，已验证可行 |
| MHC 序列 + allele embedding 双输入 | ✅ | 很好的设计，充分利用信息 |
| 双层 InfoNCE (MHC 分组 + pMHC 分组) | ✅ | 创新且合理 |
| 多标签 BCE 替代单标签分类 | ✅ | 正确解决了之前的问题 |
| Top-K 和 KL 评估指标 | ✅ | 合理的 evaluation 设计 |
| MHC-only baseline 对比 | ✅ | 必要的 ablation |

### ⚠️ 需要注意的细节

| 问题 | 建议 |
|------|------|
| **MHC allele embedding 冷启动** | 对未见过的 allele，fallback 到 sequence-only 或 nearest neighbor |
| **Group 构造的计算复杂度** | 建议在 dataloader 外部预计算 pos_mask，不要每个 batch 临时算 |
| **λ 权重调参** | 先固定 λ_pmhc=0.3, λ_bce=0.2，后续用 val 指标调整 |

### 📊 预期效果

```
当前 v1 结果:  R@10 = 1.1%
预期 v2 结果:  R@10 = 20-40% (合理预期)
               KL(model) < KL(baseline)
```

---

## 三、Stage 2 可行性分析

### ✅ 完全可行的部分

| 设计点 | 可行性 | 理由 |
|--------|--------|------|
| 复用 psiCLM 的 CollapseAwareEmbedding | ✅ | 代码已存在，可直接复用 |
| 复用 SequenceProfileEvoformer | ✅ | 代码已存在 |
| 7-level Hierarchical Pair IDs | ✅ | 代码已存在 |
| Dirichlet Flow Matching | ✅ | 标准 flow matching 公式 |
| CFG (p=0.1 drop condition) | ✅ | 标准做法 |
| Collapse entropy + profile 正则 | ✅ | 已验证有效 |

### ⚠️ 需要调整的部分

| 问题 | 建议 |
|------|------|
| **长序列处理** | Evoformer 处理 [ψ + CDR3 + pep + MHC + scaffold] 可能很长 (~300-400 tokens)，建议限制 MHC 序列长度或用 chunked attention |
| **x_t 注入方式** | 建议用 `x_proj(x_t) + pos_emb` 替换原来的 one-hot embedding，保持维度一致 |
| **Flow head 设计** | 输出维度应该是 20 (不含 gap) 还是 21，需要明确 |

### 🔧 关键改动清单

```python
# 需要修改的文件
psi_model/model.py:
  - CollapseAwareEmbedding.forward(): 增加 x_t 输入分支
  - 新增 FlowHead 类
  - 修改 psiCLM 为 FlowTCRGen

# 新增的文件
flowtcr_fold/FlowTCR_Gen/flow_gen.py:
  - FlowMatchingModel 类
  - flow_matching_loss() 函数
  - sample() ODE 采样函数
```

---

## 四、Stage 3 可行性分析

### ✅ 可行但需要资源的部分

| 设计点 | 可行性 | 注意事项 |
|--------|--------|----------|
| Phase 3A: General PPI 预训练 | ✅ | 需要下载 PDB 数据 (~50K structures)，预计 50-100GB |
| Phase 3B: EvoEF2 能量标签 | ✅ | 需要预计算，每个结构 ~1-5 秒 |
| Phase 3C: TCR 微调 | ✅ | 数据量小 (~500)，训练快 |
| E_φ surrogate | ✅ | 标准 MLP 回归 |
| MC with E_φ | ✅ | 已有 psiMonteCarloSampler 代码 |

### ⚠️ 计算资源评估

| Phase | 数据量 | 预计训练时间 | GPU 显存需求 |
|-------|--------|--------------|--------------|
| 3A | 50K structures | 3-7 天 (4×A100) | ~40GB |
| 3B | 50K + decoys | 1-2 天 | ~20GB |
| 3C | 500 structures | 几小时 | ~16GB |

### 🔴 风险点

| 风险 | 严重程度 | 缓解方案 |
|------|----------|----------|
| E_φ 与 EvoEF2 相关性可能不到 0.7 | 🟡 中等 | 增加 decoy 数据；用 pairwise ranking loss |
| Guided ODE 计算量大 | 🟢 低 | 放在 Exploratory，Practical 用后验筛选 |

---

## 五、执行时间线建议

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Execution Timeline (12-16 weeks)                                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  WEEK 1-2: Stage 1 Practical                                              │
│  ──────────────────────────────                                            │
│  T1.1: 实现 multi-positive InfoNCE (双分组)                               │
│  T1.2: 实现 multi-label BCE + pos_weight                                  │
│  T1.3: 添加 MHC allele embedding                                          │
│  T1.4: 训练 + 评估 (Top-K, KL, vs baselines)                              │
│  ★ Milestone: R@10 > 20%, KL(model) < KL(baseline)                        │
│                                                                            │
│  WEEK 3-5: Stage 2 Practical                                              │
│  ──────────────────────────────                                            │
│  T2.1: 改造 psiCLM → FlowTCRGen (x_t 注入 + flow head)                    │
│  T2.2: 实现 flow_matching_loss() + 保留正则项                              │
│  T2.3: 实现 sample() ODE 采样                                              │
│  T2.4: 实现 CFG (p=0.1 drop condition)                                    │
│  T2.5: 训练 + 评估 (recovery rate, diversity, perplexity)                 │
│  ★ Milestone: Recovery > 30%, Perplexity < 10                             │
│                                                                            │
│  WEEK 6-8: Stage 3 Phase A+B (并行准备)                                    │
│  ──────────────────────────────────────                                    │
│  T3.1: 下载 PDB 数据 + 预处理脚本                                          │
│  T3.2: 预计算 EvoEF2 能量标签 (可并行)                                     │
│  T3.3: 训练 TCRFold-Prophet trunk + StructHead (Phase 3A)                 │
│  T3.4: 训练 EnergyHead (Phase 3B)                                          │
│  ★ Milestone: E_φ 与 EvoEF2 相关性 > 0.6                                  │
│                                                                            │
│  WEEK 9-10: Stage 3 Phase C + 整合                                        │
│  ──────────────────────────────────                                        │
│  T3.5: TCR-pMHC 微调 (Phase 3C)                                           │
│  T3.6: 整合 MC with E_φ                                                    │
│  T3.7: 端到端 pipeline 测试                                                │
│  ★ Milestone: E_φ 与 EvoEF2 相关性 > 0.7 on TCR                           │
│                                                                            │
│  WEEK 11-12: 端到端评估 + 论文                                             │
│  ──────────────────────────────────                                        │
│  T4.1: 完整 pipeline 评估 (vs baselines)                                  │
│  T4.2: Case studies (知名 epitope)                                        │
│  T4.3: 论文初稿                                                            │
│                                                                            │
│  WEEK 13+: Exploratory                                                    │
│  ─────────────────────────                                                 │
│  - Guided ODE (Stage 2+3 融合)                                            │
│  - Gradient-informed MC proposal                                          │
│  - Self-play (MC → 回灌训练)                                              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 六、建议补充到计划中的内容

### 6.1 数据版本控制

```
建议在 flowtcr_fold/data/ 下维护:
  - trn_v1.jsonl (原始)
  - trn_v2.jsonl (清洗后，修复 gene name 混淆)
  - scaffold_bank_v1.json
  - energy_labels/  (EvoEF2 预计算结果)
```

### 6.2 Checkpoint 命名规范

```
checkpoints/
├── stage1_v1/          # 当前训练的 scaffold_v1
├── stage1_v2/          # Plan 3.1 修正后
├── stage2_v1/
├── stage3_phase_a/
├── stage3_phase_b/
├── stage3_phase_c/
└── pipeline_v1/        # 端到端最佳组合
```

### 6.3 Ablation 设计

| Ablation | 目的 |
|----------|------|
| Stage1: MHC-only vs pMHC | 验证 peptide 是否提供额外信息 |
| Stage2: w/ vs w/o collapse token | 验证 ψ 的作用 |
| Stage2: w/ vs w/o hierarchical pairs | 验证拓扑先验的作用 |
| Stage3: E_φ vs EvoEF2 ranking | 验证 surrogate 的质量 |

---

## 七、最终建议

### ✅ 可以立即开始的

1. **Stage 1 修正**: 实现 multi-positive InfoNCE + multi-label BCE
2. **数据清洗**: 检查并修复 gene name 混淆问题
3. **Stage 2 骨架改造**: 把 psiCLM 改成 FlowTCRGen

### ⏳ 需要并行准备的

1. **PDB 数据下载**: 开始下载 general PPI 数据
2. **EvoEF2 预计算**: 写脚本批量计算能量标签

核心方法学 claim


主打: 拓扑感知的 FlowTCR‑Gen (hierarchical pair embedding + Collapse token + Dirichlet flow matching)


物理模块: 作为 supporting contribution, 用于证明生成出来的序列在结构与能量空间是合理且可控的


Stage 3 Practical scope


必须完成:


S_ψ: General PPI 上预训练的折叠网络


E_φ: 基于 PPI + TCR‑pMHC 的 EvoEF2 surrogate


Flow → S_ψ → E_φ 的后验筛选与排序


强烈建议纳入:


基于 E_φ 的 offline Monte Carlo refinement


Exploratory:


gradient guidance in Flow ODE


MC 生成样本用于二次训练等

