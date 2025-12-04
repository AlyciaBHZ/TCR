"""
EvoEF2 Python Wrapper
=====================

Python interface to EvoEF2 (EvoDesign physical Energy Function) for:
- Structure repair and optimization
- Binding energy computation (ΔΔG)
- Stability analysis
- Mutant building

This wrapper calls the compiled EvoEF2.exe binary via subprocess.

References:
- Huang X, Pearce R, Zhang Y. Bioinformatics (2020), 36:1135-1142
- Pearce R, Huang X, et al. J Mol Biol (2019) 431: 2467-2476

Author: FlowTCR-Fold Team
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class EnergyTerms:
    """Energy components from EvoEF2 calculation."""

    total: float = 0.0

    # Reference energies (AA composition)
    reference: Dict[str, float] = None

    # Intra-residue energies
    intra_vdw_att: float = 0.0
    intra_vdw_rep: float = 0.0
    intra_elec: float = 0.0
    intra_desolv_polar: float = 0.0
    intra_desolv_hydro: float = 0.0

    # Inter-residue same-chain energies
    inter_S_vdw_att: float = 0.0
    inter_S_vdw_rep: float = 0.0
    inter_S_elec: float = 0.0
    inter_S_desolv_polar: float = 0.0
    inter_S_desolv_hydro: float = 0.0
    inter_S_ssbond: float = 0.0
    inter_S_hbond: float = 0.0

    # Inter-residue different-chain energies
    inter_D_vdw_att: float = 0.0
    inter_D_vdw_rep: float = 0.0
    inter_D_elec: float = 0.0
    inter_D_desolv_polar: float = 0.0
    inter_D_desolv_hydro: float = 0.0
    inter_D_ssbond: float = 0.0
    inter_D_hbond: float = 0.0

    def __post_init__(self):
        if self.reference is None:
            self.reference = {}

    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for easy analysis."""
        return {
            'total': self.total,
            'intra_vdw': self.intra_vdw_att + self.intra_vdw_rep,
            'intra_elec': self.intra_elec,
            'intra_desolv': self.intra_desolv_polar + self.intra_desolv_hydro,
            'inter_S_vdw': self.inter_S_vdw_att + self.inter_S_vdw_rep,
            'inter_S_elec': self.inter_S_elec,
            'inter_S_desolv': self.inter_S_desolv_polar + self.inter_S_desolv_hydro,
            'inter_S_hbond': self.inter_S_hbond,
            'inter_D_vdw': self.inter_D_vdw_att + self.inter_D_vdw_rep,
            'inter_D_elec': self.inter_D_elec,
            'inter_D_desolv': self.inter_D_desolv_polar + self.inter_D_desolv_hydro,
            'inter_D_hbond': self.inter_D_hbond,
        }


@dataclass
class BindingResult:
    """Result from ComputeBinding calculation."""

    binding_energy: float  # ΔΔG (kcal/mol)
    complex_energy: float
    receptor_energy: float
    ligand_energy: float
    energy_terms: EnergyTerms

    def __repr__(self):
        return f"BindingResult(ΔΔG={self.binding_energy:.2f} kcal/mol)"


# =============================================================================
# EvoEF2 Runner
# =============================================================================

class EvoEF2Runner:
    """
    Python wrapper for EvoEF2 executable.

    Usage:
        >>> runner = EvoEF2Runner(evoef_path="path/to/EvoEF2.exe")
        >>> runner.repair_structure("input.pdb", "output_repair.pdb")
        >>> result = runner.compute_binding("complex.pdb", split="A,BC")
        >>> print(f"Binding energy: {result.binding_energy:.2f}")
    """

    def __init__(
        self,
        evoef_path: Optional[str] = None,
        params_dir: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize EvoEF2 runner.

        Args:
            evoef_path: Path to EvoEF2 executable. If None, searches in default locations.
            params_dir: Path to parameter files directory. Auto-detected if None.
            verbose: Print detailed output from EvoEF2.
        """
        self.evoef_path = self._find_evoef(evoef_path)
        self.params_dir = self._find_params(params_dir)
        self.verbose = verbose

        if not os.path.exists(self.evoef_path):
            raise FileNotFoundError(
                f"EvoEF2 executable not found at {self.evoef_path}. "
                "Please compile EvoEF2 or specify path."
            )

    def _find_evoef(self, path: Optional[str]) -> str:
        """Find EvoEF2 executable."""
        if path and os.path.exists(path):
            return path

        # Search in common locations (Linux first, then Windows)
        # Priority: no extension (Linux) > .exe (Windows)
        search_paths = [
            "flowtcr_fold/tools/EvoEF2/EvoEF2",       # Linux first
            "flowtcr_fold/tools/EvoEF2/EvoEF2.exe",
            "tools/EvoEF2/EvoEF2",
            "tools/EvoEF2/EvoEF2.exe",
            "./EvoEF2",
            "./EvoEF2.exe",
        ]

        for sp in search_paths:
            if os.path.exists(sp):
                return os.path.abspath(sp)

        # Default to Linux version
        return "flowtcr_fold/tools/EvoEF2/EvoEF2"

    def _find_params(self, path: Optional[str]) -> str:
        """Find parameter files directory."""
        if path and os.path.exists(path):
            return path

        # Typically in same directory as executable
        evoef_dir = os.path.dirname(self.evoef_path) if hasattr(self, 'evoef_path') else ""
        param_dir = os.path.join(evoef_dir, "params")

        if os.path.exists(param_dir):
            return param_dir

        return evoef_dir  # Fallback

    def _run_command(
        self,
        command: str,
        pdb_path: str,
        output_path: Optional[str] = None,
        extra_args: Optional[Dict[str, str]] = None
    ) -> Tuple[str, str]:
        """
        Execute EvoEF2 command.

        Args:
            command: EvoEF2 command (e.g., "ComputeBinding")
            pdb_path: Input PDB file path
            output_path: Output file path (for some commands)
            extra_args: Additional command-line arguments

        Returns:
            (stdout, stderr) tuple
        """
        # EvoEF2 needs absolute paths when running from its own directory
        pdb_path_abs = os.path.abspath(pdb_path)
        
        cmd = [self.evoef_path, f"--command={command}", f"--pdb={pdb_path_abs}"]

        if extra_args:
            for key, val in extra_args.items():
                # Convert file paths in extra_args to absolute paths
                if key in ('mutant_file', 'output') and val and os.path.exists(val):
                    val = os.path.abspath(val)
                cmd.append(f"--{key}={val}")

        if self.verbose:
            print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(self.evoef_path) or ".",
                timeout=300  # 5 min timeout
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"EvoEF2 failed with return code {result.returncode}\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )

            return result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"EvoEF2 command timed out after 300s")
        except FileNotFoundError:
            raise FileNotFoundError(f"EvoEF2 executable not found: {self.evoef_path}")

    # =========================================================================
    # Core Functions
    # =========================================================================

    def repair_structure(
        self,
        pdb_path: str,
        output_path: Optional[str] = None,
        num_runs: int = 3
    ) -> str:
        """
        Repair incomplete side chains and optimize structure.

        This function:
        - Rebuilds missing side-chain atoms
        - Optimizes hydroxyl hydrogens (Ser, Thr, Tyr)
        - Flips His/Asn/Gln for H-bond optimization
        - Reduces steric clashes

        Args:
            pdb_path: Input PDB file
            output_path: Output file path. If None, creates <input>_Repair.pdb
            num_runs: Number of optimization cycles (default: 3)

        Returns:
            Path to repaired PDB file
        """
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        pdb_path_abs = os.path.abspath(pdb_path)
        
        # EvoEF2 creates output with "_Repair" suffix automatically
        stdout, _ = self._run_command(
            "RepairStructure",
            pdb_path,
            extra_args={"num_of_runs": str(num_runs)}
        )

        # EvoEF2 creates output in its working directory (EvoEF2's dir)
        evoef_dir = os.path.dirname(self.evoef_path)
        pdb_basename = os.path.basename(pdb_path)
        evoef_output = os.path.join(evoef_dir, pdb_basename.replace(".pdb", "_Repair.pdb"))
        
        # Also check if output was created next to input file
        default_output = pdb_path_abs.replace(".pdb", "_Repair.pdb")
        
        # Find where EvoEF2 actually created the file
        actual_output = None
        for candidate in [evoef_output, default_output]:
            if os.path.exists(candidate):
                actual_output = candidate
                break
        
        if not actual_output:
            raise RuntimeError(f"Repair failed: output file not created\n"
                             f"Searched: {evoef_output}, {default_output}\n{stdout}")
        
        # Move to user-specified output path if provided
        if output_path:
            output_path_abs = os.path.abspath(output_path)
            if actual_output != output_path_abs:
                os.makedirs(os.path.dirname(output_path_abs) or '.', exist_ok=True)
                os.rename(actual_output, output_path_abs)
                return output_path_abs
        
        return actual_output

    def compute_binding(
        self,
        pdb_path: str,
        split: Optional[str] = None
    ) -> BindingResult:
        """
        Compute binding energy (ΔΔG) for protein-protein complex.

        Binding energy = E(complex) - E(receptor) - E(ligand)

        Args:
            pdb_path: PDB file of complex
            split: How to split chains (e.g., "A,BC" means A vs BC).
                   If None, computes all pairwise interactions.

        Returns:
            BindingResult with ΔΔG and detailed energy terms

        Example:
            >>> result = runner.compute_binding("tcr_pmhc.pdb", split="AB,CD")
            >>> print(f"TCR-pMHC binding: {result.binding_energy:.2f} kcal/mol")
        """
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        extra_args = {}
        if split:
            extra_args["split"] = split

        stdout, _ = self._run_command("ComputeBinding", pdb_path, extra_args=extra_args)

        return self._parse_binding_output(stdout)

    def compute_stability(self, pdb_path: str) -> EnergyTerms:
        """
        Compute total stability (energy) of a protein/complex.

        Args:
            pdb_path: Input PDB file

        Returns:
            EnergyTerms object with detailed breakdown
        """
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        stdout, _ = self._run_command("ComputeStability", pdb_path)

        return self._parse_energy_output(stdout)

    def build_mutant(
        self,
        pdb_path: str,
        mutations: List[str],
        output_path: Optional[str] = None,
        num_runs: int = 10
    ) -> str:
        """
        Build mutant structure.

        Args:
            pdb_path: Input PDB file
            mutations: List of mutations (e.g., ["CA171A", "DB180E"])
            output_path: Output file path
            num_runs: Number of optimization cycles

        Returns:
            Path to mutant PDB file

        Example:
            >>> mutant = runner.build_mutant(
            ...     "wt.pdb",
            ...     ["AA95F", "AB100Y"],  # Chain A, position 95 -> F
            ...     output_path="mutant.pdb"
            ... )
        """
        # Create temporary mutation file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(','.join(mutations) + ';\n')
            mut_file = f.name

        try:
            stdout, _ = self._run_command(
                "BuildMutant",
                pdb_path,
                extra_args={
                    "mutant_file": mut_file,
                    "num_of_runs": str(num_runs)
                }
            )

            # EvoEF2 creates <input>_Model_1.pdb
            default_output = pdb_path.replace(".pdb", "_Model_1.pdb")
            if output_path and output_path != default_output:
                if os.path.exists(default_output):
                    os.rename(default_output, output_path)
                    return output_path

            return default_output

        finally:
            if os.path.exists(mut_file):
                os.unlink(mut_file)

    # =========================================================================
    # Output Parsing
    # =========================================================================

    def _parse_binding_output(self, stdout: str) -> BindingResult:
        """Parse ComputeBinding output."""
        lines = stdout.split('\n')

        binding_energy = 0.0
        complex_energy = 0.0
        receptor_energy = 0.0
        ligand_energy = 0.0

        # Parse energy terms first to get the total
        energy_terms = self._parse_energy_output(stdout)
        
        for line in lines:
            line_lower = line.lower()
            # EvoEF2 ComputeBinding outputs "Total = XXX" as binding energy
            if "total" in line_lower and "=" in line:
                match = re.search(r'Total\s*=\s*(-?\d+\.?\d*)', line, re.IGNORECASE)
                if match:
                    binding_energy = float(match.group(1))
            elif "binding energy" in line_lower or "binding ddg" in line_lower:
                match = re.search(r'[-+]?\d+\.?\d*', line)
                if match:
                    binding_energy = float(match.group())
            elif "complex energy" in line_lower:
                match = re.search(r'[-+]?\d+\.?\d*', line)
                if match:
                    complex_energy = float(match.group())
            elif "receptor energy" in line_lower:
                match = re.search(r'[-+]?\d+\.?\d*', line)
                if match:
                    receptor_energy = float(match.group())
            elif "ligand energy" in line_lower:
                match = re.search(r'[-+]?\d+\.?\d*', line)
                if match:
                    ligand_energy = float(match.group())

        # If binding_energy wasn't explicitly parsed, use total from energy_terms
        if binding_energy == 0.0 and energy_terms.total != 0.0:
            binding_energy = energy_terms.total

        return BindingResult(
            binding_energy=binding_energy,
            complex_energy=complex_energy,
            receptor_energy=receptor_energy,
            ligand_energy=ligand_energy,
            energy_terms=energy_terms
        )

    def _parse_energy_output(self, stdout: str) -> EnergyTerms:
        """Parse energy terms from output."""
        terms = EnergyTerms()

        # Regex patterns for energy terms (EvoEF2 output format: name = value)
        patterns = {
            'total': r'Total\s*[=:]\s*(-?\d+\.?\d*)',
            'intra_vdw_att': r'intraR_vdwatt\s*=\s*(-?\d+\.?\d*)',
            'intra_vdw_rep': r'intraR_vdwrep\s*=\s*(-?\d+\.?\d*)',
            'intra_elec': r'intraR_electr\s*=\s*(-?\d+\.?\d*)',
            'intra_desolv_polar': r'intraR_deslvP\s*=\s*(-?\d+\.?\d*)',
            'intra_desolv_hydro': r'intraR_deslvH\s*=\s*(-?\d+\.?\d*)',
            'inter_S_vdw_att': r'interS_vdwatt\s*=\s*(-?\d+\.?\d*)',
            'inter_S_vdw_rep': r'interS_vdwrep\s*=\s*(-?\d+\.?\d*)',
            'inter_S_elec': r'interS_electr\s*=\s*(-?\d+\.?\d*)',
            'inter_S_desolv_polar': r'interS_deslvP\s*=\s*(-?\d+\.?\d*)',
            'inter_S_desolv_hydro': r'interS_deslvH\s*=\s*(-?\d+\.?\d*)',
            'inter_S_ssbond': r'interS_ssbond\s*=\s*(-?\d+\.?\d*)',
            'inter_S_hbond': r'interS_hb(?:bbbb|scbb|scsc)_dis\s*=\s*(-?\d+\.?\d*)',
            'inter_D_vdw_att': r'interD_vdwatt\s*=\s*(-?\d+\.?\d*)',
            'inter_D_vdw_rep': r'interD_vdwrep\s*=\s*(-?\d+\.?\d*)',
            'inter_D_elec': r'interD_electr\s*=\s*(-?\d+\.?\d*)',
            'inter_D_desolv_polar': r'interD_deslvP\s*=\s*(-?\d+\.?\d*)',
            'inter_D_desolv_hydro': r'interD_deslvH\s*=\s*(-?\d+\.?\d*)',
            'inter_D_ssbond': r'interD_ssbond\s*=\s*(-?\d+\.?\d*)',
            'inter_D_hbond': r'interD_hb(?:bbbb|scbb|scsc)_dis\s*=\s*(-?\d+\.?\d*)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                setattr(terms, key, float(match.group(1)))

        # Calculate total if not explicitly provided
        if terms.total == 0.0:
            # Sum all parsed energy terms
            terms.total = (
                terms.intra_vdw_att + terms.intra_vdw_rep + terms.intra_elec +
                terms.intra_desolv_polar + terms.intra_desolv_hydro +
                terms.inter_S_vdw_att + terms.inter_S_vdw_rep + terms.inter_S_elec +
                terms.inter_S_desolv_polar + terms.inter_S_desolv_hydro +
                terms.inter_S_ssbond + terms.inter_S_hbond +
                terms.inter_D_vdw_att + terms.inter_D_vdw_rep + terms.inter_D_elec +
                terms.inter_D_desolv_polar + terms.inter_D_desolv_hydro +
                terms.inter_D_ssbond + terms.inter_D_hbond
            )

        return terms


# =============================================================================
# High-Level Interface for TCRFold-Light
# =============================================================================

class TCRStructureOptimizer:
    """
    High-level interface for TCR structure optimization in FlowTCR-Fold.

    Integrates with TCRFold-Light for:
    - Structure preprocessing (repair)
    - Energy supervision (ΔΔG labels)
    - Refinement (Monte Carlo repacking)
    """

    def __init__(self, evoef_path: Optional[str] = None):
        self.runner = EvoEF2Runner(evoef_path=evoef_path)

    def preprocess_pdb(self, pdb_path: str, output_dir: str) -> str:
        """
        Preprocess PDB for training: repair + energy compute.

        Args:
            pdb_path: Raw PDB file
            output_dir: Where to save processed files

        Returns:
            Path to repaired PDB
        """
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Repair structure
        basename = os.path.basename(pdb_path).replace('.pdb', '')
        repaired_path = os.path.join(output_dir, f"{basename}_repair.pdb")

        repaired = self.runner.repair_structure(pdb_path, repaired_path)

        return repaired

    def compute_binding_energy_batch(
        self,
        pdb_files: List[str],
        split_chains: List[str]
    ) -> np.ndarray:
        """
        Compute binding energies for a batch of structures.

        Args:
            pdb_files: List of PDB file paths
            split_chains: List of chain splits (e.g., ["AB,C", "AB,C", ...])

        Returns:
            Array of binding energies (kcal/mol)
        """
        energies = []

        for pdb, split in zip(pdb_files, split_chains):
            try:
                result = self.runner.compute_binding(pdb, split=split)
                energies.append(result.binding_energy)
            except Exception as e:
                print(f"Warning: Failed to compute binding for {pdb}: {e}")
                energies.append(0.0)  # Placeholder

        return np.array(energies)

    def refine_generated_sequences(
        self,
        scaffold_pdb: str,
        sequences: List[str],
        output_dir: str
    ) -> List[Tuple[str, float]]:
        """
        Refine generated CDR3 sequences using EvoEF2.

        This is the refinement step in the self-correction loop:
        1. Build mutant structures with generated sequences
        2. Optimize side-chains with EvoEF2
        3. Compute binding energies
        4. Return ranked (pdb_path, energy) pairs

        Args:
            scaffold_pdb: V/J scaffold PDB
            sequences: List of CDR3 sequences to graft
            output_dir: Where to save refined structures

        Returns:
            List of (refined_pdb_path, binding_energy) sorted by energy
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []

        for i, seq in enumerate(sequences):
            try:
                # Build mutant (this would need actual mutation mapping)
                # For now, placeholder logic
                mutant_path = os.path.join(output_dir, f"refined_{i}.pdb")

                # Repair structure
                repaired = self.runner.repair_structure(scaffold_pdb, mutant_path)

                # Compute energy
                energy_result = self.runner.compute_binding(repaired, split="AB,C")

                results.append((repaired, energy_result.binding_energy))

            except Exception as e:
                print(f"Warning: Failed to refine sequence {i}: {e}")

        # Sort by energy (lower is better)
        results.sort(key=lambda x: x[1])

        return results


# =============================================================================
# Utility Functions
# =============================================================================

def parse_pdb_chains(pdb_path: str) -> List[str]:
    """Extract chain IDs from PDB file."""
    chains = set()
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                chain = line[21:22].strip()
                if chain:
                    chains.add(chain)
    return sorted(chains)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example 1: Basic usage
    runner = EvoEF2Runner(verbose=True)

    # Repair structure
    repaired = runner.repair_structure("example.pdb")
    print(f"Repaired structure saved to: {repaired}")

    # Compute binding energy
    result = runner.compute_binding("complex.pdb", split="A,BC")
    print(f"Binding energy: {result.binding_energy:.2f} kcal/mol")
    print(f"Energy breakdown: {result.energy_terms.to_dict()}")

    # Example 2: High-level interface
    optimizer = TCRStructureOptimizer()

    # Preprocess for training
    processed = optimizer.preprocess_pdb("raw_tcr.pdb", "processed_data")

    # Batch energy computation
    energies = optimizer.compute_binding_energy_batch(
        ["tcr1.pdb", "tcr2.pdb"],
        ["AB,CD", "AB,CD"]
    )
    print(f"Batch energies: {energies}")
