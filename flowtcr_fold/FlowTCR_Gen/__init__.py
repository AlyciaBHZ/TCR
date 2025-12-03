"""
FlowTCR-Gen: Topology-aware Dirichlet Flow Matching for CDR3β Generation

Stage 2 of the FlowTCR-Fold pipeline.

Key Components:
- FlowTCRGen: Main model class combining encoder and flow matcher
- FlowTCRGenEncoder: Collapse Token + Hierarchical Pairs + Evoformer
- DirichletFlowMatcher: Dirichlet flow matching with CFG support

Usage:
    from flowtcr_fold.FlowTCR_Gen import FlowTCRGen
    
    model = FlowTCRGen(s_dim=256, vocab_size=25)
    
    # Training
    losses = model.training_step(batch)
    
    # Generation
    tokens = model.generate(cdr3_len=15, pep_one_hot=..., mhc_one_hot=..., ...)
    
    # Model score for Stage 3
    score = model.get_model_score(cdr3_tokens=..., ...)
"""

from flowtcr_fold.FlowTCR_Gen.model_flow import FlowTCRGen
from flowtcr_fold.FlowTCR_Gen.encoder import (
    FlowTCRGenEncoder,
    CollapseAwareEmbedding,
    SequenceProfileEvoformer,
)
from flowtcr_fold.FlowTCR_Gen.dirichlet_flow import (
    DirichletFlowMatcher,
    CFGWrapper,
    sample_x0_dirichlet,
    sample_x0_uniform,
    dirichlet_interpolate,
)
from flowtcr_fold.FlowTCR_Gen.data import (
    FlowTCRGenDataset,
    FlowTCRGenTokenizer,
    create_dataloaders,
)
from flowtcr_fold.FlowTCR_Gen.metrics import (
    FlowTCRGenEvaluator,
    compute_recovery_rate,
    compute_diversity,
)

__all__ = [
    # Main model
    'FlowTCRGen',
    
    # Encoder components
    'FlowTCRGenEncoder',
    'CollapseAwareEmbedding', 
    'SequenceProfileEvoformer',
    
    # Flow matching
    'DirichletFlowMatcher',
    'CFGWrapper',
    'sample_x0_dirichlet',
    'sample_x0_uniform',
    'dirichlet_interpolate',
    
    # Data
    'FlowTCRGenDataset',
    'FlowTCRGenTokenizer',
    'create_dataloaders',
    
    # Evaluation
    'FlowTCRGenEvaluator',
    'compute_recovery_rate',
    'compute_diversity',
]
