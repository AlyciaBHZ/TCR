"""
Physics module for FlowTCR-Fold
================================

Integrates physics-based tools:
- EvoEF2: Energy function for structure optimization and binding energy
- TM-align: Structure alignment and PSSM generation (TODO)
- AlphaFold2: Structure prediction benchmarking (TODO)

Core classes:
- EvoEF2Runner: Python wrapper for EvoEF2.exe
- TCRStructureOptimizer: High-level interface for TCR workflows
- BindingResult, EnergyTerms: Data structures
"""

from .evoef_runner import (
    EvoEF2Runner,
    TCRStructureOptimizer,
    BindingResult,
    EnergyTerms,
    parse_pdb_chains,
)

__all__ = [
    "EvoEF2Runner",
    "TCRStructureOptimizer",
    "BindingResult",
    "EnergyTerms",
    "parse_pdb_chains",
]
