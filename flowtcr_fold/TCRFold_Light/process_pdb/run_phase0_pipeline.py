#!/usr/bin/env python
"""
Phase 0 自动化管线脚本
======================

流程:
1. 检查 batch*.txt 里的 PDB ID 是否全部下载
2. 未完成则 sleep 30 分钟后重试
3. 下载完成后运行 preprocess（跳过已存在的 .npz）
4. 最后运行 EvoEF2 批量能量计算

Usage:
    python flowtcr_fold/TCRFold_Light/process_pdb/run_phase0_pipeline.py

    # 或后台运行
    nohup python flowtcr_fold/TCRFold_Light/process_pdb/run_phase0_pipeline.py \
        > logs/phase0_pipeline.log 2>&1 &
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple

# ============================================================================
# 配置
# ============================================================================

DEFAULT_CONFIG = {
    "batch_files": [
        "flowtcr_fold/data/pdb/batch1.txt",
        "flowtcr_fold/data/pdb/batch2.txt",
        "flowtcr_fold/data/pdb/batch3.txt",
        "flowtcr_fold/data/pdb/batch4.txt",
        "flowtcr_fold/data/pdb/batch5.txt",
    ],
    "raw_dir": "flowtcr_fold/data/pdb_structures/raw",
    "processed_dir": "flowtcr_fold/data/pdb_structures/processed",
    "energy_cache": "flowtcr_fold/data/energy_cache.jsonl",
    "log_dir": "flowtcr_fold/logs",
    "sleep_minutes": 30,
    "check_interval_seconds": 10,  # 检查下载进度的间隔
}


# ============================================================================
# 日志设置
# ============================================================================

def setup_logging(log_dir: str) -> logging.Logger:
    """设置日志，同时输出到文件和控制台。"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"phase0_pipeline_{timestamp}.log")
    
    logger = logging.getLogger("Phase0Pipeline")
    logger.setLevel(logging.INFO)
    
    # 文件 handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"日志文件: {log_file}")
    
    return logger


# ============================================================================
# Step 1: 检查下载进度
# ============================================================================

def load_all_pdb_ids(batch_files: List[str]) -> Set[str]:
    """从 batch 文件加载所有 PDB ID。"""
    all_ids = set()
    
    for batch_file in batch_files:
        if not os.path.exists(batch_file):
            continue
        with open(batch_file, 'r') as f:
            content = f.read()
            # 支持逗号分隔和换行分隔
            for part in content.replace('\n', ',').split(','):
                pdb_id = part.strip().upper()
                if pdb_id and len(pdb_id) == 4:
                    all_ids.add(pdb_id)
    
    return all_ids


def get_downloaded_ids(raw_dir: str) -> Set[str]:
    """获取已下载的 PDB ID。"""
    downloaded = set()
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        return downloaded
    
    for f in raw_path.glob("*.pdb"):
        pdb_id = f.stem.upper()
        downloaded.add(pdb_id)
    
    # 也检查 .cif 文件
    for f in raw_path.glob("*.cif"):
        pdb_id = f.stem.upper()
        downloaded.add(pdb_id)
    
    return downloaded


def check_download_progress(
    batch_files: List[str], 
    raw_dir: str, 
    logger: logging.Logger
) -> Tuple[int, int, Set[str]]:
    """
    检查下载进度。
    
    Returns:
        (total, downloaded, missing_ids)
    """
    all_ids = load_all_pdb_ids(batch_files)
    downloaded_ids = get_downloaded_ids(raw_dir)
    
    missing_ids = all_ids - downloaded_ids
    
    total = len(all_ids)
    downloaded = len(downloaded_ids)
    
    logger.info(f"下载进度: {downloaded}/{total} ({100*downloaded/total:.1f}%)")
    
    if missing_ids:
        # 显示部分缺失的 ID
        sample = list(missing_ids)[:10]
        logger.info(f"缺失样本 (前10个): {sample}")
    
    return total, downloaded, missing_ids


def wait_for_download(
    batch_files: List[str],
    raw_dir: str,
    sleep_minutes: int,
    logger: logging.Logger,
    min_completion_ratio: float = 0.95,
    stable_check_count: int = 2
) -> bool:
    """
    等待下载完成或稳定。
    
    Args:
        min_completion_ratio: 最低完成比例 (默认 95%)
        stable_check_count: 连续稳定检查次数 (默认 2 次无变化则认为完成)
    
    Returns:
        True 如果下载完成/稳定，False 如果被中断
    """
    prev_downloaded = 0
    stable_count = 0
    
    while True:
        total, downloaded, missing = check_download_progress(batch_files, raw_dir, logger)
        
        # 完成条件 1: 100% 下载
        if downloaded >= total:
            logger.info("✅ 所有 PDB 文件下载完成！")
            return True
        
        # 完成条件 2: 达到最低比例且下载数量稳定
        completion_ratio = downloaded / total if total > 0 else 0
        
        if completion_ratio >= min_completion_ratio:
            if downloaded == prev_downloaded:
                stable_count += 1
                logger.info(f"下载数量稳定 ({stable_count}/{stable_check_count})")
                
                if stable_count >= stable_check_count:
                    logger.info(f"✅ 下载稳定在 {completion_ratio*100:.1f}%，继续处理")
                    logger.info(f"   (剩余 {len(missing)} 个可能不可用)")
                    return True
            else:
                stable_count = 0
        
        prev_downloaded = downloaded
        
        logger.info(f"下载进度 {completion_ratio*100:.1f}%，等待 {sleep_minutes} 分钟...")
        logger.info(f"还需下载: {len(missing)} 个文件")
        
        try:
            time.sleep(sleep_minutes * 60)
        except KeyboardInterrupt:
            logger.warning("用户中断等待")
            return False


# ============================================================================
# Step 2: 预处理 (跳过已存在的)
# ============================================================================

def get_processed_pairs(processed_dir: str) -> Set[str]:
    """获取已处理的 PPI 对 (从 .npz 文件名)。"""
    processed = set()
    processed_path = Path(processed_dir)
    
    if not processed_path.exists():
        return processed
    
    for f in processed_path.glob("*.npz"):
        processed.add(f.stem)
    
    return processed


def run_preprocess(
    raw_dir: str,
    processed_dir: str,
    logger: logging.Logger
) -> bool:
    """
    运行预处理脚本。
    
    Returns:
        True 如果成功
    """
    # 检查已处理的数量
    existing = get_processed_pairs(processed_dir)
    logger.info(f"已有 {len(existing)} 个 .npz 文件")
    
    # 获取待处理的 PDB 文件
    raw_path = Path(raw_dir)
    all_pdbs = list(raw_path.glob("*.pdb"))
    logger.info(f"原始 PDB 文件数: {len(all_pdbs)}")
    
    # 运行预处理脚本
    script_path = "flowtcr_fold/TCRFold_Light/process_pdb/preprocess_ppi_pairs.py"
    
    if not os.path.exists(script_path):
        logger.error(f"预处理脚本不存在: {script_path}")
        return False
    
    cmd = [
        sys.executable, script_path,
        "--pdb_dir", raw_dir,
        "--out_dir", processed_dir,
        "--cutoff", "8.0",
        "--min_len", "30",
        "--min_contacts", "10"
    ]
    
    logger.info(f"运行预处理: {' '.join(cmd)}")
    
    try:
        # 使用 subprocess.Popen 实时输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 实时读取输出
        line_count = 0
        for line in process.stdout:
            line = line.strip()
            if line:
                # 每 100 行记录一次，或者包含关键信息的行
                line_count += 1
                if line_count % 100 == 0 or "error" in line.lower() or "warning" in line.lower():
                    logger.info(f"[preprocess] {line}")
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"预处理失败，返回码: {process.returncode}")
            return False
        
        # 检查结果
        new_count = len(get_processed_pairs(processed_dir))
        logger.info(f"✅ 预处理完成！现有 {new_count} 个 .npz 文件 (新增 {new_count - len(existing)})")
        
        return True
        
    except Exception as e:
        logger.error(f"预处理异常: {e}")
        return False


# ============================================================================
# Step 3: EvoEF2 能量计算
# ============================================================================

def get_computed_energies(energy_cache: str) -> Set[str]:
    """获取已计算能量的 PDB ID。"""
    computed = set()
    
    if not os.path.exists(energy_cache):
        return computed
    
    import json
    with open(energy_cache, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                pdb_id = entry.get("pdb_id", "")
                if pdb_id:
                    computed.add(pdb_id)
            except:
                continue
    
    return computed


def run_evoef2_batch(
    raw_dir: str,
    energy_cache: str,
    logger: logging.Logger
) -> bool:
    """
    运行 EvoEF2 批量能量计算。
    
    Returns:
        True 如果成功
    """
    # 检查已计算的数量
    existing = get_computed_energies(energy_cache)
    logger.info(f"已有 {len(existing)} 个能量记录")
    
    # 检查 EvoEF2 是否可用
    evoef_path = "flowtcr_fold/tools/EvoEF2/EvoEF2"
    if not os.path.exists(evoef_path):
        logger.error(f"EvoEF2 可执行文件不存在: {evoef_path}")
        logger.error("请先运行: cd flowtcr_fold/tools/EvoEF2 && ./build.sh")
        return False
    
    script_path = "flowtcr_fold/TCRFold_Light/process_pdb/compute_evoef2_batch.py"
    
    if not os.path.exists(script_path):
        logger.error(f"能量计算脚本不存在: {script_path}")
        return False
    
    # 设置 PYTHONPATH 以便找到 flowtcr_fold 模块
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")
    
    cmd = [
        sys.executable, script_path,
        "--pdb_dir", raw_dir,
        "--output", energy_cache,
        "--repair",  # 修复结构
        "--append"   # 追加模式，跳过已计算的
    ]
    
    logger.info(f"运行 EvoEF2 能量计算: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        # 实时读取输出并记录
        ok_count = 0
        skip_count = 0
        warn_count = 0
        
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("[OK]"):
                ok_count += 1
                if ok_count % 100 == 0:
                    logger.info(f"[EvoEF2] 已处理 {ok_count} 个结构...")
            elif line.startswith("[SKIP]"):
                skip_count += 1
            elif line.startswith("[WARN]"):
                warn_count += 1
                logger.warning(f"[EvoEF2] {line}")
            else:
                logger.info(f"[EvoEF2] {line}")
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"EvoEF2 计算失败，返回码: {process.returncode}")
            return False
        
        # 检查结果
        new_count = len(get_computed_energies(energy_cache))
        logger.info(f"✅ EvoEF2 计算完成！")
        logger.info(f"   成功: {ok_count}, 跳过: {skip_count}, 警告: {warn_count}")
        logger.info(f"   总能量记录: {new_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"EvoEF2 计算异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 0 自动化管线")
    parser.add_argument("--skip_wait", action="store_true", 
                        help="跳过等待下载，直接开始处理")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="跳过预处理")
    parser.add_argument("--skip_evoef2", action="store_true",
                        help="跳过 EvoEF2 计算")
    parser.add_argument("--sleep_minutes", type=int, default=30,
                        help="等待间隔（分钟）")
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config["sleep_minutes"] = args.sleep_minutes
    
    # 设置日志
    logger = setup_logging(config["log_dir"])
    
    logger.info("=" * 60)
    logger.info("Phase 0 自动化管线启动")
    logger.info("=" * 60)
    logger.info(f"配置:")
    logger.info(f"  - Batch 文件: {len(config['batch_files'])} 个")
    logger.info(f"  - 原始目录: {config['raw_dir']}")
    logger.info(f"  - 处理目录: {config['processed_dir']}")
    logger.info(f"  - 能量缓存: {config['energy_cache']}")
    logger.info(f"  - 等待间隔: {config['sleep_minutes']} 分钟")
    
    # Step 1: 等待下载完成
    if not args.skip_wait:
        logger.info("\n" + "=" * 40)
        logger.info("Step 1: 检查 PDB 下载进度")
        logger.info("=" * 40)
        
        if not wait_for_download(
            config["batch_files"],
            config["raw_dir"],
            config["sleep_minutes"],
            logger
        ):
            logger.warning("下载等待被中断，退出")
            return
    else:
        logger.info("跳过下载等待检查")
    
    # Step 2: 预处理
    if not args.skip_preprocess:
        logger.info("\n" + "=" * 40)
        logger.info("Step 2: 运行 PPI 预处理")
        logger.info("=" * 40)
        
        if not run_preprocess(
            config["raw_dir"],
            config["processed_dir"],
            logger
        ):
            logger.error("预处理失败，退出")
            return
    else:
        logger.info("跳过预处理")
    
    # Step 3: EvoEF2 能量计算
    if not args.skip_evoef2:
        logger.info("\n" + "=" * 40)
        logger.info("Step 3: 运行 EvoEF2 能量计算")
        logger.info("=" * 40)
        
        if not run_evoef2_batch(
            config["raw_dir"],
            config["energy_cache"],
            logger
        ):
            logger.error("EvoEF2 计算失败")
            return
    else:
        logger.info("跳过 EvoEF2 计算")
    
    # 完成
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Phase 0 管线完成！")
    logger.info("=" * 60)
    
    # 最终统计
    processed_count = len(get_processed_pairs(config["processed_dir"]))
    energy_count = len(get_computed_energies(config["energy_cache"]))
    
    logger.info(f"最终统计:")
    logger.info(f"  - 处理的 PPI 对: {processed_count}")
    logger.info(f"  - 能量记录: {energy_count}")
    
    logger.info(f"\n下一步: 可以开始 Phase 3A 训练")
    logger.info(f"  python flowtcr_fold/TCRFold_Light/train_ppi.py ...")


if __name__ == "__main__":
    main()

