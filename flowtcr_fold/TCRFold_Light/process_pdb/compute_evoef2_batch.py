"""
Batch EvoEF2 energy computation for PDB complexes.
Outputs per-chain-pair energies (binding_energy + tiered components) keyed by
`sample_key = {pdb_id}_{chainA}{chainB}`, matching processed .npz naming.

Supports MULTIPROCESSING for parallel computation on multiple CPUs.

Usage (pairwise, recommended):
    # Single process
    python compute_evoef2_batch.py \
        --pdb_dir flowtcr_fold/data/pdb_structures/raw \
        --output flowtcr_fold/data/energy_cache.jsonl \
        --pairwise --append

    # Multi-process (16 CPUs)
    python compute_evoef2_batch.py \
        --pdb_dir flowtcr_fold/data/pdb_structures/raw \
        --output flowtcr_fold/data/energy_cache.jsonl \
        --pairwise --append --num_workers 16
"""

import argparse
import json
import os
import sys
import tempfile
import multiprocessing as mp
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import Optional, Set, Dict, Any, List, Tuple

from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from flowtcr_fold.physics.evoef_runner import EvoEF2Runner, EnergyTerms


class ChainSelect(Select):
    """Select specific chains for PDBIO."""
    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)
    
    def accept_chain(self, chain):
        return chain.get_id() in self.chain_ids


def extract_chains_to_pdb(
    structure,
    chain_ids: list,
    output_path: str
) -> bool:
    """Extract specific chains from structure and save to PDB."""
    io = PDBIO()
    io.set_structure(structure)
    try:
        io.save(output_path, ChainSelect(chain_ids))
        return True
    except Exception as e:
        return False


def cif_to_pdb(cif_path: Path, temp_dir: str) -> Optional[str]:
    """Convert CIF file to PDB format."""
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(cif_path.stem, str(cif_path))
    except Exception:
        return None
    
    out_pdb = os.path.join(temp_dir, f"{cif_path.stem}.pdb")
    io = PDBIO()
    io.set_structure(structure)
    try:
        io.save(out_pdb)
    except Exception:
        return None
    
    return out_pdb


def list_chain_ids(file_path: Path) -> Optional[List[str]]:
    """Return list of chain IDs for protein chains."""
    if file_path.suffix.lower() == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure(file_path.stem, str(file_path))
    except Exception:
        return None

    model = next(structure.get_models(), None)
    if model is None:
        return None

    chain_ids = [c.get_id() for c in model.get_chains() if c.get_id().strip()]
    if len(chain_ids) < 2:
        return None
    return chain_ids


def energy_terms_to_dict(terms: EnergyTerms) -> Dict[str, float]:
    """Convert EnergyTerms to a serializable dictionary."""
    return {
        'total': terms.total,
        'intra_vdw_att': terms.intra_vdw_att,
        'intra_vdw_rep': terms.intra_vdw_rep,
        'intra_elec': terms.intra_elec,
        'intra_desolv_polar': terms.intra_desolv_polar,
        'intra_desolv_hydro': terms.intra_desolv_hydro,
        'inter_S_vdw_att': terms.inter_S_vdw_att,
        'inter_S_vdw_rep': terms.inter_S_vdw_rep,
        'inter_S_elec': terms.inter_S_elec,
        'inter_S_desolv_polar': terms.inter_S_desolv_polar,
        'inter_S_desolv_hydro': terms.inter_S_desolv_hydro,
        'inter_S_ssbond': terms.inter_S_ssbond,
        'inter_S_hbond': terms.inter_S_hbond,
        'inter_D_vdw_att': terms.inter_D_vdw_att,
        'inter_D_vdw_rep': terms.inter_D_vdw_rep,
        'inter_D_elec': terms.inter_D_elec,
        'inter_D_desolv_polar': terms.inter_D_desolv_polar,
        'inter_D_desolv_hydro': terms.inter_D_desolv_hydro,
        'inter_D_ssbond': terms.inter_D_ssbond,
        'inter_D_hbond': terms.inter_D_hbond,
    }


def compute_full_energy(
    runner: EvoEF2Runner,
    pdb_path: str,
    chain_a: str,
    chain_b: str,
    temp_dir: str,
    do_repair: bool = True
) -> Optional[Dict[str, Any]]:
    """Compute full Tier 1 + Tier 3 energies for a PPI complex."""
    pdb_path = os.path.abspath(pdb_path)
    pdb_id = Path(pdb_path).stem
    
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
    except Exception:
        return None
    
    complex_pdb = pdb_path
    receptor_pdb = os.path.join(temp_dir, f"{pdb_id}_receptor_{chain_a}.pdb")
    ligand_pdb = os.path.join(temp_dir, f"{pdb_id}_ligand_{chain_b}.pdb")
    
    if do_repair:
        try:
            repaired_pdb = os.path.join(temp_dir, f"{pdb_id}_repaired.pdb")
            complex_pdb = runner.repair_structure(pdb_path, repaired_pdb)
            structure = parser.get_structure(pdb_id, complex_pdb)
        except Exception:
            complex_pdb = pdb_path
    
    if not extract_chains_to_pdb(structure, [chain_a], receptor_pdb):
        return None
    if not extract_chains_to_pdb(structure, [chain_b], ligand_pdb):
        return None
    
    result = {
        'pdb_id': pdb_id,
        'chain_a': chain_a,
        'chain_b': chain_b,
        'sample_key': f"{pdb_id}_{chain_a}{chain_b}",
        'E_complex': 0.0,
        'E_receptor': 0.0,
        'E_ligand': 0.0,
        'E_bind': 0.0,
        'binding_energy': 0.0,
        'energy_terms': {'complex': {}, 'receptor': {}, 'ligand': {}}
    }
    
    try:
        terms_complex = runner.compute_stability(complex_pdb)
        result['E_complex'] = terms_complex.total
        result['energy_terms']['complex'] = energy_terms_to_dict(terms_complex)
    except Exception:
        return None
    
    try:
        terms_receptor = runner.compute_stability(receptor_pdb)
        result['E_receptor'] = terms_receptor.total
        result['energy_terms']['receptor'] = energy_terms_to_dict(terms_receptor)
    except Exception:
        return None
    
    try:
        terms_ligand = runner.compute_stability(ligand_pdb)
        result['E_ligand'] = terms_ligand.total
        result['energy_terms']['ligand'] = energy_terms_to_dict(terms_ligand)
    except Exception:
        return None
    
    try:
        split_str = f"{chain_a},{chain_b}"
        binding_result = runner.compute_binding(complex_pdb, split=split_str)
        result['E_bind'] = binding_result.binding_energy
        result['binding_energy'] = binding_result.binding_energy
    except Exception:
        result['E_bind'] = result['E_complex'] - result['E_receptor'] - result['E_ligand']
        result['binding_energy'] = result['E_bind']
    
    return result


def process_single_file(args_tuple: Tuple) -> List[Dict]:
    """
    Process a single PDB/CIF file and return list of energy results.
    This function runs in a worker process.
    """
    file_path, existing_keys, do_pairwise, manual_split, do_repair = args_tuple
    
    file_path = Path(file_path)
    pdb_id = file_path.stem
    is_cif = file_path.suffix.lower() == ".cif"
    results = []
    
    # Determine chain pairs
    chain_pairs: List[Tuple[str, str]] = []
    if manual_split:
        parts = manual_split.split(",")
        if len(parts) >= 2:
            chain_pairs = [(parts[0], parts[1])]
        else:
            return results
    else:
        chain_ids = list_chain_ids(file_path)
        if not chain_ids:
            return results
        if do_pairwise:
            chain_pairs = list(combinations(chain_ids, 2))
        else:
            if len(chain_ids) >= 2:
                chain_pairs = [(chain_ids[0], chain_ids[1])]
    
    # Filter already computed
    chain_pairs = [(a, b) for a, b in chain_pairs 
                   if f"{pdb_id}_{a}{b}" not in existing_keys]
    
    if not chain_pairs:
        return results
    
    # Create per-process temp dir and runner
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            runner = EvoEF2Runner()
        except FileNotFoundError:
            return results
        
        # Handle CIF conversion
        if is_cif:
            pdb_path = cif_to_pdb(file_path, temp_dir)
            if pdb_path is None:
                return results
        else:
            pdb_path = str(file_path)
        
        for chain_a, chain_b in chain_pairs:
            result = compute_full_energy(
                runner=runner,
                pdb_path=pdb_path,
                chain_a=chain_a,
                chain_b=chain_b,
                temp_dir=temp_dir,
                do_repair=do_repair
            )
            if result is not None:
                results.append(result)
    
    return results


def load_existing_ids(jsonl_path: Path) -> Set[str]:
    """Load already computed sample keys from existing JSONL file."""
    existing = set()
    if not jsonl_path.exists():
        return existing
    
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "sample_key" in entry:
                    existing.add(entry["sample_key"])
                else:
                    key = f"{entry['pdb_id']}_{entry['chain_a']}{entry['chain_b']}"
                    existing.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    
    return existing


def main():
    ap = argparse.ArgumentParser(description="Compute EvoEF2 Tier 1+3 energies for PDB complexes.")
    ap.add_argument("--pdb_dir", required=True, help="Directory of .pdb/.cif files.")
    ap.add_argument("--output", required=True, help="Output JSONL path for energies.")
    ap.add_argument("--repair", action="store_true", help="Repair structures before energy computation.")
    ap.add_argument("--max_files", type=int, default=0, help="Limit number of files (0 = all).")
    ap.add_argument("--append", action="store_true", help="Append mode: skip already computed entries.")
    ap.add_argument("--pairwise", action="store_true", help="Compute every chain pair.")
    ap.add_argument("--split", help="Optional manual split 'A,B' to override pairwise mode.")
    ap.add_argument("--num_workers", type=int, default=1, 
                    help="Number of parallel workers (default: 1, use 0 for auto)")
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    
    # Collect files
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    cif_files = sorted(pdb_dir.glob("*.cif"))
    all_files = pdb_files + cif_files
    
    if args.max_files:
        all_files = all_files[:args.max_files]

    if not all_files:
        print(f"No .pdb or .cif files found in {pdb_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Determine number of workers
    if args.num_workers == 0:
        num_workers = mp.cpu_count()
    else:
        num_workers = args.num_workers
    
    print(f"[INFO] Found {len(pdb_files)} PDB + {len(cif_files)} CIF = {len(all_files)} total files", flush=True)
    print(f"[INFO] Using {num_workers} worker(s)", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing entries
    existing_keys = set()
    if args.append:
        existing_keys = load_existing_ids(out_path)
        print(f"[INFO] Append mode: found {len(existing_keys)} existing entries, will skip", flush=True)

    file_mode = "a" if args.append else "w"
    
    # Prepare arguments for each file
    task_args = [
        (str(f), existing_keys, args.pairwise or not args.split, args.split, args.repair)
        for f in all_files
    ]
    
    ok_count = 0
    warn_count = 0
    
    with out_path.open(file_mode) as fout:
        if num_workers == 1:
            # Single process mode
            for i, task in enumerate(task_args, 1):
                results = process_single_file(task)
                for result in results:
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
                    ok_count += 1
                    print(f"[OK] {Path(task[0]).name}: E_bind={result['E_bind']:.2f} "
                          f"(chains {result['chain_a']},{result['chain_b']})", flush=True)
                
                if i % 100 == 0:
                    print(f"[INFO] Progress: {i}/{len(all_files)} files, {ok_count} pairs computed", flush=True)
        else:
            # Multi-process mode
            with mp.Pool(processes=num_workers) as pool:
                for i, results in enumerate(pool.imap_unordered(process_single_file, task_args), 1):
                    for result in results:
                        fout.write(json.dumps(result) + "\n")
                        fout.flush()
                        ok_count += 1
                        print(f"[OK] {result['pdb_id']}: E_bind={result['E_bind']:.2f} "
                              f"(chains {result['chain_a']},{result['chain_b']})", flush=True)
                    
                    if i % 100 == 0:
                        print(f"[INFO] Progress: {i}/{len(all_files)} files, {ok_count} pairs computed", flush=True)
    
    print(f"\n[DONE] Computed: {ok_count} pairs from {len(all_files)} files", flush=True)


if __name__ == "__main__":
    main()
