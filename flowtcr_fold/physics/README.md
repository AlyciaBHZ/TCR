## EvoEF2 Python Wrapper for FlowTCR-Fold

This module provides a Python interface to **EvoEF2** (EvoDesign physical Energy Function) for physics-based protein design and analysis.

### **Core Features**

1. **Structure Repair**: Fix missing atoms, optimize side-chains
2. **Binding Energy**: Compute ΔΔG for protein-protein complexes
3. **Energy Supervision**: Provide labels for TCRFold-Light training
4. **Batch Processing**: Handle multiple structures efficiently

---

### **Installation**

#### **1. Compile EvoEF2**

```bash
cd flowtcr_fold/tools/EvoEF2

# Linux/Mac
g++ -O3 --fast-math -o EvoEF2 src/*.cpp

# Windows (with MinGW)
g++ -O3 -o EvoEF2.exe src/*.cpp
```

#### **2. Test Installation**

```bash
python flowtcr_fold/physics/test_evoef.py
```

Expected output:
```
✓ EvoEF2 found at: flowtcr_fold/tools/EvoEF2/EvoEF2.exe
✓ Parameters dir: flowtcr_fold/tools/EvoEF2/params
```

---

### **Quick Start**

#### **Basic Usage**

```python
from flowtcr_fold.physics import EvoEF2Runner

# Initialize runner
runner = EvoEF2Runner()

# Repair structure
repaired_pdb = runner.repair_structure("input.pdb", num_runs=3)

# Compute binding energy
result = runner.compute_binding("complex.pdb", split="AB,C")
print(f"ΔΔG = {result.binding_energy:.2f} kcal/mol")
```

#### **High-Level Interface**

```python
from flowtcr_fold.physics import TCRStructureOptimizer

# Initialize optimizer
optimizer = TCRStructureOptimizer()

# Preprocess PDB for training
processed = optimizer.preprocess_pdb("raw_tcr.pdb", "processed_data")

# Batch energy computation
energies = optimizer.compute_binding_energy_batch(
    pdb_files=["tcr1.pdb", "tcr2.pdb"],
    split_chains=["AB,CD", "AB,CD"]
)
```

---

### **Integration with TCRFold-Light**

#### **Energy-Supervised Training**

The `EnergyStructureDataset` pairs PDB structures with EvoEF2 energy labels:

```python
from flowtcr_fold.physics.energy_dataset import EnergyStructureDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = EnergyStructureDataset(
    pdb_dir="data/pdb_structures",
    cache_dir="data/energy_cache",
    verbose=True
)

# Use in training
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in loader:
    s = batch['s']  # Sequence embeddings
    z = batch['z']  # Pair embeddings
    energy_label = batch['energy']  # EvoEF2 ΔΔG

    # Train energy surrogate head
    loss = model.energy_loss(s, z, energy_label)
```

#### **Full Training Script**

```bash
python flowtcr_fold/TCRFold_Light/train_with_energy.py \
    --pdb_dir data/pdb_structures \
    --epochs 100 \
    --batch_size 4 \
    --interface_weight 10.0
```

This script implements:
- **L_dist**: Distance map MSE
- **L_contact**: Contact map BCE (interface-weighted ×10)
- **L_energy**: Energy surrogate MSE

---

### **API Reference**

#### **EvoEF2Runner**

**Constructor:**
```python
EvoEF2Runner(
    evoef_path: Optional[str] = None,
    params_dir: Optional[str] = None,
    verbose: bool = False
)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `repair_structure(pdb, output, num_runs=3)` | Fix missing atoms, optimize H-bonds | `str` (output path) |
| `compute_binding(pdb, split=None)` | Calculate ΔΔG | `BindingResult` |
| `compute_stability(pdb)` | Calculate total energy | `EnergyTerms` |
| `build_mutant(pdb, mutations, output)` | Build mutant structure | `str` (output path) |

#### **Data Structures**

**BindingResult:**
```python
@dataclass
class BindingResult:
    binding_energy: float      # ΔΔG (kcal/mol)
    complex_energy: float
    receptor_energy: float
    ligand_energy: float
    energy_terms: EnergyTerms  # Detailed breakdown
```

**EnergyTerms:**
- `total`: Total energy
- `intra_vdw`, `intra_elec`, `intra_desolv`: Intra-residue terms
- `inter_S_*`: Same-chain interactions
- `inter_D_*`: Different-chain interactions (binding interface)

---

### **File Structure**

```
flowtcr_fold/physics/
├── __init__.py              # Module exports
├── evoef_runner.py          # Core EvoEF2 wrapper (600+ lines)
├── energy_dataset.py        # Dataset for training (300+ lines)
├── test_evoef.py            # Test suite
├── README.md                # This file
└── (planned)
    ├── tmalign_runner.py    # TM-align for PSSM
    └── alphafold_benchmark.py
```

---

### **Energy Terms Explained**

EvoEF2 computes energy as:

**E_total = E_reference + E_intra + E_inter_S + E_inter_D**

Where:
- **E_reference**: AA composition bias (favors native-like sequences)
- **E_intra**: Within-residue energies (vdW, electrostatics, desolvation)
- **E_inter_S**: Same-chain interactions (protein folding)
- **E_inter_D**: Different-chain interactions (binding interface)

**For TCR-pMHC binding**, `E_inter_D` is most important.

---

### **Common Tasks**

#### **1. Prepare Training Data**

```bash
# Download TCR-pMHC structures from STCRDab
wget http://opig.stats.ox.ac.uk/webapps/stcrdab/download

# Organize into directory
mkdir -p data/pdb_structures
cp *.pdb data/pdb_structures/

# Compute energies (cached for future use)
python flowtcr_fold/physics/energy_dataset.py
```

#### **2. Interface Identification**

```python
# Load structure
from flowtcr_fold.physics.energy_dataset import EnergyStructureDataset

dataset = EnergyStructureDataset("data/pdb_structures", "data/cache")
sample = dataset[0]

# Interface residues (>5 contacts)
interface_mask = (sample['contact_map'].sum(dim=-1) > 5).float()
print(f"Interface residues: {interface_mask.sum().item()}")
```

#### **3. Refinement Pipeline**

```python
from flowtcr_fold.physics import TCRStructureOptimizer

optimizer = TCRStructureOptimizer()

# Refine generated sequences
refined = optimizer.refine_generated_sequences(
    scaffold_pdb="v_j_scaffold.pdb",
    sequences=["CASSYLQGAYEQYF", "CASSPLRGNTIYF"],
    output_dir="refined_structures"
)

# Ranked by energy
for pdb_path, energy in refined:
    print(f"{pdb_path}: {energy:.2f} kcal/mol")
```

---

### **Troubleshooting**

#### **Issue**: `EvoEF2 executable not found`

**Solution**: Compile EvoEF2 first:
```bash
cd flowtcr_fold/tools/EvoEF2
g++ -O3 --fast-math -o EvoEF2 src/*.cpp
```

#### **Issue**: `No parameter files found`

**Solution**: EvoEF2 needs parameter files in `params/` directory:
- `dun2010bb3per.lib` (rotamer library)
- `physics.txt` (energy weights)

These should be in `flowtcr_fold/tools/EvoEF2/params/`.

#### **Issue**: `Binding energy computation fails`

**Solution**: Ensure PDB has multiple chains:
```bash
grep "^ATOM" structure.pdb | awk '{print $5}' | sort -u
# Should show multiple chain IDs (A, B, C, etc.)
```

#### **Issue**: `Energy cache recomputation is slow`

**Solution**: Cache is saved in `energy_cache.json`. To skip recomputation:
```python
dataset = EnergyStructureDataset(..., recompute=False)
```

---

### **Performance Notes**

- **Structure Repair**: ~1-5 seconds per structure
- **Binding Energy**: ~2-10 seconds per complex
- **Batch Processing**: Can parallelize with `multiprocessing`

For large-scale training:
1. Precompute all energies once (cache them)
2. Use cached energies during training
3. Set `recompute=False` in dataset

---

### **References**

1. **Huang X, Pearce R, Zhang Y.** (2020)
   *EvoEF2: accurate and fast energy function for computational protein design.*
   Bioinformatics, 36:1135-1142

2. **Pearce R, Huang X, et al.** (2019)
   *EvoDesign: Designing Protein–Protein Binding Interactions Using Evolutionary Interface Profiles.*
   J Mol Biol, 431: 2467-2476

---

### **Next Steps**

- [ ] Add TM-align integration for PSSM generation
- [ ] Add FAPE loss from AlphaFold2
- [ ] Implement Monte Carlo side-chain repacking
- [ ] Add multi-GPU support for batch energy computation

---

### **Contact**

For bugs or feature requests related to this wrapper:
- Open an issue at: https://github.com/your_repo/issues

For EvoEF2 itself:
- Contact: xiaoqiah@umich.edu (original author)
