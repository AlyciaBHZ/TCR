"""
Generate FASTA File for Wet-Lab Synthesis
Creates synthesis-ready FASTA sequences from constructed beta-chains
"""

import pandas as pd
from datetime import datetime
import json

def log_message(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def create_fasta_entry(construct):
    """Create a FASTA entry for a construct"""

    # Create descriptive header
    header = (f">{construct['construct_id']}_"
             f"{construct['peptide']}_{construct['mhc']}_"
             f"Conf{construct['confidence']:.3f}_"
             f"Score{construct['composite_score']:.3f}")

    # Format sequence with line breaks every 80 characters
    sequence = construct['full_sequence']
    formatted_sequence = '\n'.join([sequence[i:i+80] for i in range(0, len(sequence), 80)])

    return f"{header}\n{formatted_sequence}\n"

def add_synthesis_metadata(construct):
    """Add synthesis-specific metadata to construct entry"""

    # Extract information for synthesis notes
    metadata = {
        'target': f"{construct['peptide']}_{construct['mhc']}",
        'cdr3_sequence': construct['cdr3_sequence'],
        'v_gene_source': construct['v_gene'][:30] + '...' if len(construct['v_gene']) > 30 else construct['v_gene'],
        'j_gene_source': construct['j_gene'][:30] + '...' if len(construct['j_gene']) > 30 else construct['j_gene'],
        'sequence_length': construct['sequence_length'],
        'confidence': round(construct['confidence'], 3),
        'composite_score': round(construct['composite_score'], 3),
        'vj_confidence': round(construct['vj_confidence'], 3),
        'generation_method': construct['method'],
        'synthesis_priority': 'HIGH' if construct['confidence'] >= 0.8 else 'MEDIUM' if construct['confidence'] >= 0.7 else 'STANDARD'
    }

    return metadata

def generate_synthesis_fasta():
    """Main function to generate synthesis FASTA file"""

    log_message("=== Generating FASTA for Wet-Lab Synthesis ===")

    # Load synthesis constructs
    log_message("Loading synthesis-ready constructs...")
    constructs_df = pd.read_csv('results/synthesis_ready_constructs.csv')
    log_message(f"Loaded {len(constructs_df)} constructs")

    # Parse validation column (it's stored as string)
    constructs_df['is_valid'] = constructs_df['validation'].apply(
        lambda x: eval(x)['is_valid'] if pd.notna(x) and x != '' else False
    )

    # Filter for valid constructs only
    valid_constructs = constructs_df[constructs_df['is_valid']].copy()
    log_message(f"Using {len(valid_constructs)} valid constructs")

    # Extract target index from construct_id for sorting
    valid_constructs['target_idx'] = valid_constructs['construct_id'].apply(
        lambda x: int(x.split('_')[2])
    )

    # Sort by target index and composite score (highest first)
    valid_constructs = valid_constructs.sort_values(
        ['target_idx', 'composite_score'],
        ascending=[True, False]
    )

    # Generate FASTA content
    fasta_content = ""
    synthesis_metadata = []

    log_message("Generating FASTA entries...")

    for idx, row in valid_constructs.iterrows():
        construct = row.to_dict()

        # Add FASTA entry
        fasta_entry = create_fasta_entry(construct)
        fasta_content += fasta_entry + "\n"

        # Collect metadata
        metadata = add_synthesis_metadata(construct)
        synthesis_metadata.append(metadata)

        log_message(f"  {construct['construct_id']}: {len(construct['full_sequence'])} AA, "
                   f"Priority: {metadata['synthesis_priority']}")

    # Save FASTA file
    fasta_file = 'results/synthesis_ready_constructs.fasta'
    with open(fasta_file, 'w') as f:
        f.write(fasta_content)

    log_message(f"FASTA file saved to: {fasta_file}")

    # Save synthesis metadata
    metadata_file = 'results/synthesis_metadata.json'
    synthesis_summary = {
        'total_constructs': len(synthesis_metadata),
        'high_priority': sum(1 for m in synthesis_metadata if m['synthesis_priority'] == 'HIGH'),
        'medium_priority': sum(1 for m in synthesis_metadata if m['synthesis_priority'] == 'MEDIUM'),
        'standard_priority': sum(1 for m in synthesis_metadata if m['synthesis_priority'] == 'STANDARD'),
        'average_confidence': sum(m['confidence'] for m in synthesis_metadata) / len(synthesis_metadata),
        'average_length': sum(m['sequence_length'] for m in synthesis_metadata) / len(synthesis_metadata),
        'targets_covered': len(set(m['target'] for m in synthesis_metadata)),
        'generation_date': datetime.now().isoformat(),
        'constructs': synthesis_metadata
    }

    with open(metadata_file, 'w') as f:
        json.dump(synthesis_summary, f, indent=2)

    log_message(f"Synthesis metadata saved to: {metadata_file}")

    # Generate synthesis summary report
    log_message("\n=== SYNTHESIS SUMMARY ===")
    log_message(f"Total sequences ready for synthesis: {len(synthesis_metadata)}")
    log_message(f"Priority distribution:")
    log_message(f"  HIGH priority (conf >= 0.8): {synthesis_summary['high_priority']} constructs")
    log_message(f"  MEDIUM priority (conf 0.7-0.8): {synthesis_summary['medium_priority']} constructs")
    log_message(f"  STANDARD priority (conf < 0.7): {synthesis_summary['standard_priority']} constructs")

    log_message(f"\nSequence statistics:")
    log_message(f"  Average confidence: {synthesis_summary['average_confidence']:.3f}")
    log_message(f"  Average length: {synthesis_summary['average_length']:.1f} AA")
    log_message(f"  Targets covered: {synthesis_summary['targets_covered']}")

    # Per-target summary
    target_summary = {}
    for metadata in synthesis_metadata:
        target = metadata['target']
        if target not in target_summary:
            target_summary[target] = []
        target_summary[target].append(metadata)

    log_message(f"\nPer-target summary:")
    for target, constructs in target_summary.items():
        high_conf = sum(1 for c in constructs if c['confidence'] >= 0.8)
        avg_conf = sum(c['confidence'] for c in constructs) / len(constructs)
        log_message(f"  {target}: {len(constructs)} constructs, {high_conf} high-conf, avg {avg_conf:.3f}")

    # Generate synthesis recommendations
    log_message(f"\n=== SYNTHESIS RECOMMENDATIONS ===")
    log_message(f"Recommended synthesis order:")
    log_message(f"1. Start with HIGH priority constructs (n={synthesis_summary['high_priority']})")
    log_message(f"2. Continue with MEDIUM priority constructs (n={synthesis_summary['medium_priority']})")
    log_message(f"3. Optional: STANDARD priority constructs (n={synthesis_summary['standard_priority']})")

    log_message(f"\nFiles generated:")
    log_message(f"  {fasta_file} - Ready for synthesis")
    log_message(f"  {metadata_file} - Synthesis metadata and recommendations")

    return synthesis_summary

if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    synthesis_summary = generate_synthesis_fasta()