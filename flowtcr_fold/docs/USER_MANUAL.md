# FlowTCR 用户手册（面向使用者）

## 目录/路径
- 数据脚本: flowtcr_fold/data/（convert_csv_to_jsonl.py, tokenizer.py, dataset.py）
- 公共工具: flowtcr_fold/common/utils.py（50 epoch 存档；100 epoch 早停）
- 模型目录: Immuno_PLM/, TCRFold_Light/, FlowTCR_Gen/（各自包含 model/train/eval）
- 物理接口: physics/（EvoEF2 包装）、tools/EvoEF2/（编译或现成二进制），详见 EVOEF2_INTEGRATION.md
- 文档: docs/（本手册、initial_plan*.md、README）

## 数据准备
- 必填字段: peptide,mhc,cdr3_b；可选: h_v,h_j,l_v,l_j,cdr3_a
- 清洗: python flowtcr_fold/data/convert_csv_to_jsonl.py --input data/trn.csv --output data/trn.jsonl
- 负样本: dataset.py 已支持同 MHC peptide decoy（相似度 0.6-0.9）和 CDR3 互换/受控突变（避开末端），identity 阈值 0.75
- 结构: STCRDab/TCR3d PDB 放入 data/pdb_structures，供能量监督与评估

## 训练/评估
- PLM 训练: python flowtcr_fold/Immuno_PLM/train_plm.py --data data/trn.csv --epochs 1 --batch_size 8
- PLM 评估: python flowtcr_fold/Immuno_PLM/eval_plm.py --data data/val.csv --checkpoint checkpoints/plm/immuno_plm.pt
- Flow 训练: python flowtcr_fold/FlowTCR_Gen/train_flow.py --data data/trn.csv
- TCRFold-Light 能量监督: python flowtcr_fold/TCRFold_Light/train_with_energy.py --pdb_dir data/pdb_structures --cache_dir data/energy_cache
- 推理循环: python -c "from flowtcr_fold.FlowTCR_Gen.pipeline_impl import run_pipeline; print(run_pipeline(scaffold_pdb='data/scaffold.pdb'))"

## 批次与偏好
- PLM: batch 为多条序列 [CLS]+pep+[SEP]+mhc+[SEP]+cdr3+[SEP]，同批做 MLM；InfoNCE 以样本对为 anchor/positive/negatives
- Flow: batch 为 token 序列（长度取决于 tokenizer），损失为插值流匹配
- 结构: batch 为序列/对偶/距离/能量标签（energy_cache），接口残基权重更高
- 偏好: 每 50 epoch 存 checkpoint；连续 100 epoch 无提升早停

## EvoEF2 与 refine
- 编译: 在 tools/EvoEF2 运行 g++ -O3 --fast-math -o EvoEF2 src/*.cpp 或使用现成二进制；参数放在 tools/EvoEF2/params
- 自检: python flowtcr_fold/physics/test_evoef.py 验证可执行与参数路径
- 能量缓存: python flowtcr_fold/physics/energy_dataset.py --pdb_dir data/pdb_structures --cache_dir data/energy_cache
- refine: pipeline_impl 使用 TCRStructureOptimizer，需传 scaffold_pdb（V/J 框架），返回 (pdb, binding_energy) 排序

## 你需要做的事
1) 准备好 data/trn.csv 并跑清洗脚本
2) 整理 TCR-pMHC PDB 到 data/pdb_structures，先生成 energy_cache
3) 在 tools/EvoEF2 编译/检查 EvoEF2，运行 physics/test_evoef.py
4) 选择 scaffold_pdb 以便 pipeline_impl 做 EvoEF2 refine；确认链拆分规则（如 AB,C）
5) 后续安装 TM-align/FAPE 等外部工具时记录路径，更新 docs
