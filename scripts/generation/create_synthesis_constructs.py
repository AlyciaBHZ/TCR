"""
Create Synthesis-Ready Full β-Chain Constructs
Combines V region + CDR3β + J region for complete TCR β-chain sequences
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import json

def log_message(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def extract_cdr3_region(full_v_gene, cdr3_sequence):
    """Extract the variable region before CDR3 and prepare for fusion"""

    # Remove the CDR3 start pattern from V gene if it exists
    # Common CDR3 start patterns in beta-chain: CASS, CASR, CATS, etc.
    v_region = full_v_gene

    # Find where the V gene ends (typically before the CDR3 start)
    cdr3_start_patterns = ['CASS', 'CASR', 'CATS', 'CAST', 'CALS', 'CARS']

    for pattern in cdr3_start_patterns:
        if pattern in v_region:
            # Find the last occurrence of the pattern
            last_idx = v_region.rfind(pattern)
            if last_idx > len(v_region) * 0.7:  # Only if it's toward the end
                v_region = v_region[:last_idx]
                break

    return v_region.rstrip()

def extract_j_region(full_j_gene, cdr3_sequence):
    """Extract the joining region after CDR3 and prepare for fusion"""

    # Remove the CDR3 end pattern from J gene if it exists
    j_region = full_j_gene

    # Common CDR3 end patterns: YEQYF, NEQFF, DTQYF, etc.
    cdr3_end_patterns = ['YEQYF', 'NEQFF', 'DTQYF', 'TEAFF', 'GELFF', 'ETQYF']

    for pattern in cdr3_end_patterns:
        if pattern in j_region:
            # Find the first occurrence and take everything after
            first_idx = j_region.find(pattern)
            if first_idx < len(j_region) * 0.3:  # Only if it's toward the beginning
                # Take everything after the pattern
                j_region = j_region[first_idx + len(pattern):]
                break

    return j_region.lstrip()

def create_full_beta_chain(v_gene, cdr3_sequence, j_gene, candidate_info):
    """Create a complete beta-chain sequence"""

    # Extract regions
    v_region = extract_cdr3_region(v_gene, cdr3_sequence)
    j_region = extract_j_region(j_gene, cdr3_sequence)

    # Combine: V + CDR3 + J
    full_sequence = v_region + cdr3_sequence + j_region

    # Create construct information
    construct_info = {
        'construct_id': f"TCR_B_{candidate_info['target_idx']}_{candidate_info['rank_in_target']:02d}",
        'peptide': candidate_info['peptide'],
        'mhc': candidate_info['mhc'],
        'cdr3_sequence': cdr3_sequence,
        'v_gene': v_gene[:50] + '...' if len(v_gene) > 50 else v_gene,  # Truncate for display
        'j_gene': j_gene[:50] + '...' if len(j_gene) > 50 else j_gene,
        'v_region': v_region,
        'j_region': j_region,
        'full_sequence': full_sequence,
        'sequence_length': len(full_sequence),
        'confidence': candidate_info['confidence'],
        'composite_score': candidate_info['composite_score'],
        'vj_confidence': candidate_info['vj_confidence'],
        'method': candidate_info['method']
    }

    return construct_info

def validate_construct(construct_info):
    """Validate the synthesized construct"""

    sequence = construct_info['full_sequence']
    cdr3 = construct_info['cdr3_sequence']

    validation_results = {
        'is_valid': True,
        'issues': []
    }

    # Check sequence length (variable domain length heuristic)
    # Typical TCR beta variable domain (V + CDR3 + J) ~90–130 AA
    if len(sequence) < 80 or len(sequence) > 160:
        validation_results['issues'].append(f"Unusual length: {len(sequence)} AA")

    # Check CDR3 is properly embedded
    if cdr3 not in sequence:
        validation_results['is_valid'] = False
        validation_results['issues'].append("CDR3 not found in full sequence")

    # Check for unusual characters
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    invalid_aas = set(sequence) - valid_aas
    if invalid_aas:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"Invalid amino acids: {invalid_aas}")

    # Check for very long homopolymers (>6 consecutive same AAs)
    for aa in valid_aas:
        if aa * 7 in sequence:
            validation_results['issues'].append(f"Long homopolymer: {aa}7+")

    construct_info['validation'] = validation_results
    return construct_info

def create_synthesis_constructs():
    """Main function to create synthesis-ready constructs"""

    log_message("=== Creating Synthesis-Ready Beta-Chain Constructs ===")

    # Load enhanced candidates
    log_message("Loading candidates with V/J information...")
    candidates_df = pd.read_csv('results/candidates_with_vj.csv')
    log_message(f"Loaded {len(candidates_df)} candidates")

    # Filter for top candidates (ranks 1-3 per target)
    top_candidates = candidates_df[candidates_df['rank_in_target'] <= 3].copy()
    log_message(f"Processing top {len(top_candidates)} candidates (ranks 1-3 per target)")

    synthesis_constructs = []

    for idx, row in top_candidates.iterrows():

        # Skip candidates without V/J gene information
        if pd.isna(row['recommended_v_gene']) or pd.isna(row['recommended_j_gene']):
            log_message(f"  Skipping candidate {row['sequence']}: Missing V/J genes")
            continue

        if row['recommended_v_gene'] == 'UNKNOWN' or row['recommended_j_gene'] == 'UNKNOWN':
            log_message(f"  Skipping candidate {row['sequence']}: Unknown V/J genes")
            continue

        # Create full beta-chain construct
        construct = create_full_beta_chain(
            v_gene=row['recommended_v_gene'],
            cdr3_sequence=row['sequence'],
            j_gene=row['recommended_j_gene'],
            candidate_info=row.to_dict()
        )

        # Validate construct
        construct = validate_construct(construct)

        synthesis_constructs.append(construct)

        if construct['validation']['is_valid']:
            status = "VALID"
        else:
            status = f"ISSUES: {'; '.join(construct['validation']['issues'])}"

        log_message(f"  {construct['construct_id']}: {len(construct['full_sequence'])} AA - {status}")

    log_message(f"\nCreated {len(synthesis_constructs)} synthesis constructs")

    # Convert to DataFrame for analysis
    constructs_df = pd.DataFrame(synthesis_constructs)

    # Save constructs
    output_file = 'results/synthesis_ready_constructs.csv'
    constructs_df.to_csv(output_file, index=False)
    log_message(f"Constructs saved to: {output_file}")

    # Summary statistics
    log_message("\n=== CONSTRUCT ANALYSIS ===")

    valid_constructs = constructs_df[constructs_df.apply(lambda x: x['validation']['is_valid'], axis=1)]
    log_message(f"Valid constructs: {len(valid_constructs)}/{len(constructs_df)}")

    if len(constructs_df) > 0:
        length_stats = constructs_df['sequence_length'].describe()
        log_message(f"Sequence length statistics:")
        log_message(f"  Mean: {length_stats['mean']:.1f} AA")
        log_message(f"  Range: {length_stats['min']:.0f} - {length_stats['max']:.0f} AA")

        # Confidence distribution
        conf_stats = constructs_df['confidence'].describe()
        log_message(f"Confidence distribution:")
        log_message(f"  Mean: {conf_stats['mean']:.3f}")
        log_message(f"  Range: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}")

        # Per target breakdown
        target_counts = constructs_df.groupby(['peptide', 'mhc']).size()
        log_message(f"\nConstructs per target:")
        for (pep, mhc), count in target_counts.items():
            log_message(f"  {pep}/{mhc}: {count} constructs")

    # Save summary
    summary = {
        'total_constructs': len(constructs_df),
        'valid_constructs': len(valid_constructs),
        'invalid_constructs': len(constructs_df) - len(valid_constructs),
        'length_stats': constructs_df['sequence_length'].describe().to_dict() if len(constructs_df) > 0 else {},
        'confidence_stats': constructs_df['confidence'].describe().to_dict() if len(constructs_df) > 0 else {},
        'per_target_counts': {f"{k[0]}_{k[1]}": v for k, v in target_counts.to_dict().items()} if len(constructs_df) > 0 else {},
        'analysis_date': datetime.now().isoformat()
    }

    with open('results/synthesis_constructs_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    log_message(f"Summary saved to: results/synthesis_constructs_summary.json")

    return constructs_df

if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    constructs_df = create_synthesis_constructs()
