# FlowTCR 用户手册（你需要做的事）

## 模块与目录
- 数据：`flowtcr_fold/data/`（原始数据、清洗脚本）。
- 公共：`flowtcr_fold/common/utils.py`（每50 epoch保存、100无提升早停）。
- 模块：`Immuno_PLM/`，`TCRFold_Light/`，`FlowTCR_Gen/`（各自模型+训练/推理入口）。
- 文档：`docs/`（README、计划、手册）。
- 旧的 physics/inference 目录是占位，可在有权限时删除，不再被新脚本依赖。

## 你需要准备什么
- 数据字段：`peptide,mhc,cdr3_b`（可选 `h_v,h_j,l_v,l_j,cdr3_a`）。
- 清洗：`python flowtcr_fold/data/convert_csv_to_jsonl.py --input data/trn.csv --output data/trn.jsonl`。
- 硬负样本（当前启发式）：同 MHC、肽相似度≥0.8 的 decoy；CDR3 交换/2-3点突变；后续可用比对/规则替换。
- 结构/能量：需要收集 PDB/STCRDab/TCR3d 结构，安装 EvoEF2（RepairStructure/ComputeBinding）和 TM-align（生成 PSSM）后再接入。

## 如何运行（占位示例）
- PLM 训练：`python flowtcr_fold/Immuno_PLM/train_plm.py --data data/trn.csv --epochs 1 --batch_size 8`
- PLM 评估：`python flowtcr_fold/Immuno_PLM/eval_plm.py --data data/val.csv --checkpoint checkpoints/plm/immuno_plm.pt`
- Flow/结构：待接入真实条件/结构数据后再运行（当前为占位流场和随机张量）。

## 批次含义
- PLM：batch = 多条序列，每条 `[CLS]+pep+[SEP]+mhc+[SEP]+cdr3+[SEP]`；同一序列可做 MLM；InfoNCE 用正/负池化向量。
- 结构/Flow：目前占位；真实数据需包含序列/对偶/几何/能量监督与条件。

## 训练偏好（已写入脚本）
- 每 50 epoch 保存 checkpoint；100 epoch 无改进早停。

## 下一步优先级
1) 将 `psi_model/model.py` 的分层 pair/Collapse 融入 Immuno-PLM，完善批量拓扑 bias。
2) 用比对/规则改进硬负样本（肽 decoy、受控 CDR3 突变）。
3) 接入真实结构与 EvoEF2 能量，补齐 TCRFold-Light 的 distance/contact/energy 损失。
4) Flow 损失加入条件（pMHC/VJ/几何）与注意力-接触对齐，refine 调用 EvoEF2。
5) 有权限后清理旧的 physics/inference 目录以简化树。
