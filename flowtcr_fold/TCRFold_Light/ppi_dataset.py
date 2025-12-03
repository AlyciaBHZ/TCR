"""
PPIDataset: 统一的蛋白质-蛋白质相互作用数据集
==================================================

整合 Phase 0 数据流:
1. .npz 文件 (seq, coords, contact_map) from preprocess_ppi_pairs.py
2. JSONL 能量缓存 from compute_evoef2_batch.py

解决的接口对齐问题:
- 统一数据源: 使用预处理的 .npz 而非重新解析 PDB
- 统一缓存格式: 读取 JSONL 能量缓存，避免重复调用 EvoEF2
- 统一链分割: 沿用 .npz 中的链对划分
- 真实序列编码: 提供 AA index 编码而非零占位符

Usage:
    >>> from flowtcr_fold.TCRFold_Light.ppi_dataset import PPIDataset, collate_ppi_batch
    >>> dataset = PPIDataset(
    ...     npz_dir="flowtcr_fold/data/pdb_structures/processed",
    ...     energy_cache="flowtcr_fold/data/energy_cache.jsonl"
    ... )
    >>> loader = DataLoader(dataset, batch_size=4, collate_fn=collate_ppi_batch)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# ============================================================================
# 氨基酸编码
# ============================================================================

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"  # 20 标准氨基酸
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
AA_TO_IDX["X"] = 20  # Unknown
AA_TO_IDX["-"] = 21  # Gap
PAD_IDX = 22
VOCAB_SIZE = 23


def encode_sequence(seq: str) -> torch.Tensor:
    """将氨基酸序列编码为索引张量。"""
    indices = [AA_TO_IDX.get(aa.upper(), AA_TO_IDX["X"]) for aa in seq]
    return torch.tensor(indices, dtype=torch.long)


def one_hot_encode(seq: str, vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """将氨基酸序列编码为 one-hot 张量 [L, vocab_size]。"""
    indices = encode_sequence(seq)
    one_hot = torch.zeros(len(seq), vocab_size)
    one_hot.scatter_(1, indices.unsqueeze(1), 1.0)
    return one_hot


# ============================================================================
# PPIDataset
# ============================================================================

class PPIDataset(Dataset):
    """
    统一的 PPI 数据集。
    
    数据流:
        raw/*.pdb 
            ↓ preprocess_ppi_pairs.py
        processed/*.npz (seq_a, seq_b, coords_a, coords_b, contact_map, n_contacts)
            ↓ compute_evoef2_batch.py  
        energy_cache.jsonl (pdb_id → binding_energy)
            ↓
        PPIDataset (本类)
            ↓
        train_ppi.py
    """

    def __init__(
        self,
        npz_dir: str,
        energy_cache: Optional[str] = None,
        max_length: int = 512,
        contact_threshold: float = 8.0,
        min_contacts: int = 10,
        use_one_hot: bool = False,
        verbose: bool = False
    ):
        """
        Args:
            npz_dir: 预处理的 .npz 文件目录 (from preprocess_ppi_pairs.py)
            energy_cache: JSONL 能量缓存路径 (from compute_evoef2_batch.py)
                         如果为 None，binding_energy 返回 0.0
            max_length: 最大序列长度 (截断)
            contact_threshold: 接触距离阈值 (Å)，仅用于验证
            min_contacts: 最小接触数，过滤低质量样本
            use_one_hot: 是否使用 one-hot 编码 (否则用索引)
            verbose: 打印加载信息
        """
        self.npz_dir = Path(npz_dir)
        self.max_length = max_length
        self.contact_threshold = contact_threshold
        self.min_contacts = min_contacts
        self.use_one_hot = use_one_hot
        self.verbose = verbose

        # 1. 扫描 .npz 文件
        self.npz_files = self._scan_npz_files()
        if self.verbose:
            print(f"Found {len(self.npz_files)} .npz files in {npz_dir}")

        # 2. 加载能量缓存
        self.energy_cache = self._load_energy_cache(energy_cache)
        if self.verbose:
            print(f"Loaded {len(self.energy_cache)} energy entries from cache")

        # 3. 构建有效样本列表
        self.samples = self._build_sample_list()
        if self.verbose:
            print(f"PPIDataset: {len(self.samples)} valid samples")

    def _scan_npz_files(self) -> List[Path]:
        """扫描 .npz 文件。"""
        if not self.npz_dir.exists():
            raise ValueError(f"NPZ directory not found: {self.npz_dir}")
        return sorted(self.npz_dir.glob("*.npz"))

    def _load_energy_cache(self, cache_path: Optional[str]) -> Dict[str, float]:
        """
        加载 JSONL 能量缓存。
        
        格式 (每行):
            {"pdb_id": "1A0F", "chain_a": "A", "chain_b": "B", "binding_energy": -102.13, ...}
        
        返回:
            {sample_key: binding_energy} 其中 sample_key = "{pdb_id}_{chain_a}_{chain_b}"
        """
        if cache_path is None or not os.path.exists(cache_path):
            return {}

        cache = {}
        with open(cache_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    pdb_id = entry.get("pdb_id", "")
                    chain_a = entry.get("chain_a", "A")
                    chain_b = entry.get("chain_b", "B")
                    binding_energy = entry.get("binding_energy", 0.0)
                    
                    # 构建与 .npz 文件名匹配的 key
                    # .npz 文件名格式: {pdb_id}_{chain_a}_{chain_b}.npz
                    sample_key = f"{pdb_id}_{chain_a}_{chain_b}"
                    cache[sample_key] = binding_energy
                    
                    # 也存储简单 pdb_id (向后兼容)
                    if pdb_id and pdb_id not in cache:
                        cache[pdb_id] = binding_energy
                        
                except json.JSONDecodeError:
                    continue

        return cache

    def _build_sample_list(self) -> List[Dict]:
        """构建有效样本列表。"""
        samples = []

        for npz_path in self.npz_files:
            try:
                data = np.load(npz_path, allow_pickle=True)
                
                # 检查必要字段 (支持两种命名格式)
                # 格式1: coords_a, coords_b, n_contacts
                # 格式2: ca_a, ca_b, num_contacts (from preprocess_ppi_pairs.py)
                has_format1 = all(k in data for k in ["seq_a", "seq_b", "coords_a", "coords_b", "contact_map"])
                has_format2 = all(k in data for k in ["seq_a", "seq_b", "ca_a", "ca_b", "contact_map"])
                
                if not (has_format1 or has_format2):
                    continue

                # 获取接触数
                if "n_contacts" in data:
                    n_contacts = int(data["n_contacts"])
                elif "num_contacts" in data:
                    n_contacts = int(data["num_contacts"])
                else:
                    n_contacts = int(data["contact_map"].sum())
                
                # 过滤低质量样本
                if n_contacts < self.min_contacts:
                    continue

                # 提取 sample key 用于匹配能量
                stem = npz_path.stem  # e.g., "1A0F_AB"
                
                samples.append({
                    "npz_path": str(npz_path),
                    "sample_key": stem,
                    "n_contacts": n_contacts
                })

            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to load {npz_path}: {e}")
                continue

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回一个 PPI 样本。
        
        Returns:
            {
                'seq_a': [L_a] 氨基酸索引 (或 [L_a, vocab_size] one-hot)
                'seq_b': [L_b] 氨基酸索引 (或 [L_b, vocab_size] one-hot)
                'coords_a': [L_a, 3] Cα 坐标
                'coords_b': [L_b, 3] Cα 坐标
                'contact_map': [L_a, L_b] 接触图
                'distance_map': [L_a, L_b] 距离矩阵 (从坐标计算)
                'mask_a': [L_a] 有效残基掩码
                'mask_b': [L_b] 有效残基掩码
                'binding_energy': scalar
                'n_contacts': scalar
                'sample_key': str
            }
        """
        sample = self.samples[idx]
        data = np.load(sample["npz_path"], allow_pickle=True)

        # 1. 序列
        seq_a = str(data["seq_a"])
        seq_b = str(data["seq_b"])
        
        # 截断
        seq_a = seq_a[:self.max_length]
        seq_b = seq_b[:self.max_length]
        L_a, L_b = len(seq_a), len(seq_b)

        # 编码
        if self.use_one_hot:
            seq_a_enc = one_hot_encode(seq_a)
            seq_b_enc = one_hot_encode(seq_b)
        else:
            seq_a_enc = encode_sequence(seq_a)
            seq_b_enc = encode_sequence(seq_b)

        # 2. 坐标 (支持两种键名: coords_a/coords_b 或 ca_a/ca_b)
        coords_key_a = "coords_a" if "coords_a" in data else "ca_a"
        coords_key_b = "coords_b" if "coords_b" in data else "ca_b"
        coords_a = data[coords_key_a][:L_a].astype(np.float32)
        coords_b = data[coords_key_b][:L_b].astype(np.float32)

        # 3. 接触图
        contact_map = data["contact_map"][:L_a, :L_b].astype(np.float32)

        # 4. 距离矩阵 (从坐标计算)
        distance_map = self._compute_distance_map(coords_a, coords_b)

        # 5. 掩码
        mask_a = torch.ones(L_a, dtype=torch.long)
        mask_b = torch.ones(L_b, dtype=torch.long)

        # 6. 能量
        sample_key = sample["sample_key"]
        binding_energy = self.energy_cache.get(sample_key, 0.0)
        
        # 尝试只用 pdb_id
        if binding_energy == 0.0:
            pdb_id = sample_key.split("_")[0] if "_" in sample_key else sample_key
            binding_energy = self.energy_cache.get(pdb_id, 0.0)

        return {
            "seq_a": seq_a_enc,
            "seq_b": seq_b_enc,
            "coords_a": torch.from_numpy(coords_a),
            "coords_b": torch.from_numpy(coords_b),
            "contact_map": torch.from_numpy(contact_map),
            "distance_map": torch.from_numpy(distance_map),
            "mask_a": mask_a,
            "mask_b": mask_b,
            "binding_energy": torch.tensor(binding_energy, dtype=torch.float32),
            "n_contacts": torch.tensor(sample["n_contacts"], dtype=torch.long),
            "sample_key": sample_key
        }

    def _compute_distance_map(
        self, 
        coords_a: np.ndarray, 
        coords_b: np.ndarray
    ) -> np.ndarray:
        """计算链间距离矩阵 [L_a, L_b]。"""
        # coords_a: [L_a, 3], coords_b: [L_b, 3]
        diff = coords_a[:, None, :] - coords_b[None, :, :]  # [L_a, L_b, 3]
        dist = np.sqrt((diff ** 2).sum(axis=-1))  # [L_a, L_b]
        return dist.astype(np.float32)


# ============================================================================
# Collate Function
# ============================================================================

def collate_ppi_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    批处理变长 PPI 样本，支持不对称链长度。
    
    Padding 策略:
    - seq_a/mask_a/coords_a: pad 到 batch 内最大 L_a
    - seq_b/mask_b/coords_b: pad 到 batch 内最大 L_b
    - contact_map/distance_map: pad 到 [max_L_a, max_L_b]
    """
    B = len(batch)
    
    # 找最大长度
    max_L_a = max(item["mask_a"].size(0) for item in batch)
    max_L_b = max(item["mask_b"].size(0) for item in batch)

    # 判断是否 one-hot
    is_one_hot = batch[0]["seq_a"].dim() == 2
    
    if is_one_hot:
        vocab_size = batch[0]["seq_a"].size(1)
        seq_a_batch = torch.zeros(B, max_L_a, vocab_size)
        seq_b_batch = torch.zeros(B, max_L_b, vocab_size)
    else:
        seq_a_batch = torch.full((B, max_L_a), PAD_IDX, dtype=torch.long)
        seq_b_batch = torch.full((B, max_L_b), PAD_IDX, dtype=torch.long)

    coords_a_batch = torch.zeros(B, max_L_a, 3)
    coords_b_batch = torch.zeros(B, max_L_b, 3)
    contact_batch = torch.zeros(B, max_L_a, max_L_b)
    distance_batch = torch.zeros(B, max_L_a, max_L_b)
    mask_a_batch = torch.zeros(B, max_L_a, dtype=torch.long)
    mask_b_batch = torch.zeros(B, max_L_b, dtype=torch.long)
    energy_batch = torch.zeros(B)
    n_contacts_batch = torch.zeros(B, dtype=torch.long)
    sample_keys = []

    for i, item in enumerate(batch):
        L_a = item["mask_a"].size(0)
        L_b = item["mask_b"].size(0)

        seq_a_batch[i, :L_a] = item["seq_a"]
        seq_b_batch[i, :L_b] = item["seq_b"]
        coords_a_batch[i, :L_a] = item["coords_a"]
        coords_b_batch[i, :L_b] = item["coords_b"]
        contact_batch[i, :L_a, :L_b] = item["contact_map"]
        distance_batch[i, :L_a, :L_b] = item["distance_map"]
        mask_a_batch[i, :L_a] = item["mask_a"]
        mask_b_batch[i, :L_b] = item["mask_b"]
        energy_batch[i] = item["binding_energy"]
        n_contacts_batch[i] = item["n_contacts"]
        sample_keys.append(item["sample_key"])

    return {
        "seq_a": seq_a_batch,
        "seq_b": seq_b_batch,
        "coords_a": coords_a_batch,
        "coords_b": coords_b_batch,
        "contact_map": contact_batch,
        "distance_map": distance_batch,
        "mask_a": mask_a_batch,
        "mask_b": mask_b_batch,
        "binding_energy": energy_batch,
        "n_contacts": n_contacts_batch,
        "sample_keys": sample_keys
    }


# ============================================================================
# 便捷函数: 合并双链为单序列 (用于 Evoformer)
# ============================================================================

def merge_chains_for_evoformer(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    将双链批次合并为单序列格式，适配 Evoformer 输入。
    
    合并策略:
    - seq: [seq_a, <SEP>, seq_b]
    - coords: [coords_a, zeros, coords_b]
    - 生成 chain_mask 区分两条链
    - 生成 pair_type 标记链内/链间
    
    Returns:
        {
            'seq': [B, L_total] 合并序列
            'coords': [B, L_total, 3] 合并坐标
            'mask': [B, L_total] 有效掩码
            'chain_mask': [B, L_total] 链标识 (0=A, 1=sep, 2=B)
            'contact_map': [B, L_total, L_total] 全接触图
            'distance_map': [B, L_total, L_total] 全距离图
            'pair_type': [B, L_total, L_total] 0=intra-A, 1=inter, 2=intra-B
            'binding_energy': [B]
            'L_a': [B] 链 A 长度 (不含 SEP)
            'L_b': [B] 链 B 长度
        }
    """
    B = batch["seq_a"].size(0)
    L_a = batch["mask_a"].size(1)
    L_b = batch["mask_b"].size(1)
    L_total = L_a + 1 + L_b  # +1 for SEP token

    is_one_hot = batch["seq_a"].dim() == 3
    
    if is_one_hot:
        vocab_size = batch["seq_a"].size(2)
        seq_merged = torch.zeros(B, L_total, vocab_size)
        # SEP token 用 index 21 (gap)
        sep_onehot = torch.zeros(vocab_size)
        sep_onehot[21] = 1.0
    else:
        seq_merged = torch.full((B, L_total), PAD_IDX, dtype=torch.long)
        SEP_IDX = 21  # 使用 gap 作为 SEP

    coords_merged = torch.zeros(B, L_total, 3)
    mask_merged = torch.zeros(B, L_total, dtype=torch.long)
    chain_mask = torch.zeros(B, L_total, dtype=torch.long)
    
    # 全尺寸接触/距离图
    contact_full = torch.zeros(B, L_total, L_total)
    distance_full = torch.full((B, L_total, L_total), 999.0)  # 大距离表示无接触
    pair_type = torch.zeros(B, L_total, L_total, dtype=torch.long)

    for b in range(B):
        la = batch["mask_a"][b].sum().item()
        lb = batch["mask_b"][b].sum().item()
        
        # 序列
        if is_one_hot:
            seq_merged[b, :la] = batch["seq_a"][b, :la]
            seq_merged[b, la] = sep_onehot
            seq_merged[b, la+1:la+1+lb] = batch["seq_b"][b, :lb]
        else:
            seq_merged[b, :la] = batch["seq_a"][b, :la]
            seq_merged[b, la] = SEP_IDX
            seq_merged[b, la+1:la+1+lb] = batch["seq_b"][b, :lb]

        # 坐标
        coords_merged[b, :la] = batch["coords_a"][b, :la]
        # SEP 位置保持 0
        coords_merged[b, la+1:la+1+lb] = batch["coords_b"][b, :lb]

        # 掩码
        mask_merged[b, :la] = 1
        mask_merged[b, la] = 1  # SEP 也算有效
        mask_merged[b, la+1:la+1+lb] = 1

        # 链标识
        chain_mask[b, :la] = 0  # Chain A
        chain_mask[b, la] = 1    # SEP
        chain_mask[b, la+1:la+1+lb] = 2  # Chain B

        # 接触图 (链间部分)
        inter_contact = batch["contact_map"][b, :la, :lb]
        contact_full[b, :la, la+1:la+1+lb] = inter_contact
        contact_full[b, la+1:la+1+lb, :la] = inter_contact.T

        # 距离图
        inter_dist = batch["distance_map"][b, :la, :lb]
        distance_full[b, :la, la+1:la+1+lb] = inter_dist
        distance_full[b, la+1:la+1+lb, :la] = inter_dist.T
        
        # 链内距离 (自身为 0)
        for i in range(la):
            distance_full[b, i, i] = 0.0
        for i in range(lb):
            distance_full[b, la+1+i, la+1+i] = 0.0

        # Pair type: 0=intra-A, 1=inter, 2=intra-B
        pair_type[b, :la, :la] = 0  # intra-A
        pair_type[b, la+1:la+1+lb, la+1:la+1+lb] = 2  # intra-B
        pair_type[b, :la, la+1:la+1+lb] = 1  # inter A→B
        pair_type[b, la+1:la+1+lb, :la] = 1  # inter B→A

    return {
        "seq": seq_merged,
        "coords": coords_merged,
        "mask": mask_merged,
        "chain_mask": chain_mask,
        "contact_map": contact_full,
        "distance_map": distance_full,
        "pair_type": pair_type,
        "binding_energy": batch["binding_energy"],
        "L_a": batch["mask_a"].sum(dim=1),
        "L_b": batch["mask_b"].sum(dim=1)
    }


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", type=str, 
                        default="flowtcr_fold/data/pdb_structures/processed")
    parser.add_argument("--energy_cache", type=str, 
                        default="flowtcr_fold/data/energy_cache.jsonl")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    print("=" * 60)
    print("Testing PPIDataset")
    print("=" * 60)

    try:
        dataset = PPIDataset(
            npz_dir=args.npz_dir,
            energy_cache=args.energy_cache if os.path.exists(args.energy_cache) else None,
            verbose=True
        )

        if len(dataset) == 0:
            print("\nNo samples found. Make sure to run preprocess_ppi_pairs.py first.")
        else:
            print(f"\nDataset size: {len(dataset)}")
            
            # 测试单样本
            sample = dataset[0]
            print(f"\nSample keys: {list(sample.keys())}")
            print(f"  seq_a shape: {sample['seq_a'].shape}")
            print(f"  seq_b shape: {sample['seq_b'].shape}")
            print(f"  coords_a shape: {sample['coords_a'].shape}")
            print(f"  coords_b shape: {sample['coords_b'].shape}")
            print(f"  contact_map shape: {sample['contact_map'].shape}")
            print(f"  binding_energy: {sample['binding_energy'].item():.2f}")
            print(f"  n_contacts: {sample['n_contacts'].item()}")
            print(f"  sample_key: {sample['sample_key']}")

            # 测试 DataLoader
            loader = DataLoader(
                dataset, 
                batch_size=min(args.batch_size, len(dataset)),
                collate_fn=collate_ppi_batch
            )
            
            batch = next(iter(loader))
            print(f"\nBatch keys: {list(batch.keys())}")
            print(f"  seq_a batch shape: {batch['seq_a'].shape}")
            print(f"  contact_map batch shape: {batch['contact_map'].shape}")

            # 测试合并函数
            merged = merge_chains_for_evoformer(batch)
            print(f"\nMerged for Evoformer:")
            print(f"  seq shape: {merged['seq'].shape}")
            print(f"  contact_map shape: {merged['contact_map'].shape}")
            print(f"  pair_type shape: {merged['pair_type'].shape}")

            print("\n✅ PPIDataset test passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

