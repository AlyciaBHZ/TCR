"""
Test script for EvoEF2 wrapper
================================

Usage:
    python flowtcr_fold/physics/test_evoef.py

This script tests:
1. EvoEF2 executable detection
2. Structure repair functionality
3. Binding energy computation
4. Energy term parsing
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from flowtcr_fold.physics.evoef_runner import EvoEF2Runner, TCRStructureOptimizer


def test_evoef_installation():
    """Test 1: Check if EvoEF2 is installed."""
    print("=" * 60)
    print("Test 1: EvoEF2 Installation")
    print("=" * 60)

    try:
        runner = EvoEF2Runner(verbose=True)
        print(f"✓ EvoEF2 found at: {runner.evoef_path}")
        print(f"✓ Parameters dir: {runner.params_dir}")
        return runner
    except FileNotFoundError as e:
        print(f"✗ EvoEF2 not found: {e}")
        print("\nTo install EvoEF2:")
        print("  cd flowtcr_fold/tools/EvoEF2")
        print("  ./build.sh  # Linux/Mac")
        print("  # or compile with g++ on Windows")
        return None


def test_structure_repair(runner: EvoEF2Runner):
    """Test 2: Structure repair (requires example PDB)."""
    print("\n" + "=" * 60)
    print("Test 2: Structure Repair")
    print("=" * 60)

    # Check if we have a test PDB
    test_pdbs = [
        "flowtcr_fold/tools/EvoEF2/example.pdb",
        "data/example_tcr.pdb",
    ]

    test_pdb = None
    for pdb in test_pdbs:
        if os.path.exists(pdb):
            test_pdb = pdb
            break

    if not test_pdb:
        print("✗ No test PDB found. Skipping repair test.")
        print(f"  Searched: {test_pdbs}")
        return

    try:
        print(f"Input PDB: {test_pdb}")
        repaired = runner.repair_structure(test_pdb, num_runs=1)
        print(f"✓ Repaired PDB created: {repaired}")

        # Check file size
        size = os.path.getsize(repaired)
        print(f"  Output size: {size} bytes")

        # Cleanup
        if os.path.exists(repaired):
            os.remove(repaired)
            print("  Cleaned up test output")

    except Exception as e:
        print(f"✗ Repair failed: {e}")


def test_binding_energy(runner: EvoEF2Runner):
    """Test 3: Binding energy computation."""
    print("\n" + "=" * 60)
    print("Test 3: Binding Energy Computation")
    print("=" * 60)

    # Check for test complex
    test_complexes = [
        "flowtcr_fold/tools/EvoEF2/complex.pdb",
        "data/example_complex.pdb",
    ]

    test_complex = None
    for pdb in test_complexes:
        if os.path.exists(pdb):
            test_complex = pdb
            break

    if not test_complex:
        print("✗ No test complex found. Skipping binding test.")
        print(f"  Searched: {test_complexes}")
        return

    try:
        print(f"Input complex: {test_complex}")

        # Compute binding
        result = runner.compute_binding(test_complex, split=None)

        print(f"✓ Binding energy: {result.binding_energy:.2f} kcal/mol")
        print(f"  Complex energy: {result.complex_energy:.2f}")
        print(f"  Receptor energy: {result.receptor_energy:.2f}")
        print(f"  Ligand energy: {result.ligand_energy:.2f}")

        # Show energy breakdown
        breakdown = result.energy_terms.to_dict()
        print("\n  Energy breakdown:")
        for term, value in sorted(breakdown.items()):
            if abs(value) > 0.01:
                print(f"    {term:20s}: {value:8.2f}")

    except Exception as e:
        print(f"✗ Binding computation failed: {e}")


def test_high_level_interface():
    """Test 4: High-level TCRStructureOptimizer."""
    print("\n" + "=" * 60)
    print("Test 4: TCRStructureOptimizer Interface")
    print("=" * 60)

    try:
        optimizer = TCRStructureOptimizer()
        print("✓ TCRStructureOptimizer initialized")
        print(f"  Using EvoEF2 at: {optimizer.runner.evoef_path}")

        # Test batch energy computation (with dummy files)
        print("\n  Testing batch interface (will fail without real PDBs):")
        test_files = ["dummy1.pdb", "dummy2.pdb"]
        test_splits = ["AB,C", "AB,C"]

        # This will fail gracefully with warnings
        energies = optimizer.compute_binding_energy_batch(test_files, test_splits)
        print(f"  Batch result: {energies}")

    except Exception as e:
        print(f"✗ High-level interface test failed: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("EvoEF2 Python Wrapper Test Suite")
    print("=" * 60 + "\n")

    # Test 1: Installation
    runner = test_evoef_installation()
    if not runner:
        print("\n✗ EvoEF2 not installed. Cannot proceed with tests.")
        print("\nPlease compile EvoEF2 first:")
        print("  cd flowtcr_fold/tools/EvoEF2")
        print("  g++ -O3 --fast-math -o EvoEF2 src/*.cpp")
        return 1

    # Test 2-3: Core functionality (if test files exist)
    test_structure_repair(runner)
    test_binding_energy(runner)

    # Test 4: High-level interface
    test_high_level_interface()

    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)
    print("\nNote: Some tests were skipped due to missing test PDB files.")
    print("To run full tests, add example PDB files to:")
    print("  - flowtcr_fold/tools/EvoEF2/example.pdb")
    print("  - flowtcr_fold/tools/EvoEF2/complex.pdb")

    return 0


if __name__ == "__main__":
    sys.exit(main())
