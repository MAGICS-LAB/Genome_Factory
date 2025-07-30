#!/usr/bin/env python3
"""
Enhanced GenomeBridge Generation Examples
Demonstrates various control mechanisms of enhanced GenomeBridge sequence generation

Based on real-world application scenarios:
1. Cloning vector design - Avoid restriction enzyme sites
2. Promoter design - Include regulatory elements, control GC content
3. Synthetic gene design - Balance expression and stability
4. Library construction - High diversity sequences
5. Species-specific sequences - Mimic specific organism composition
6. GenSLM model usage - Demonstrate advanced language model generation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from genomeBridge.Inference.generation.enhanced_controlled_generation import EnhancedGenomeBridgeGenerator
import pandas as pd
import numpy as np

def example_1_cloning_vector_design():
    """Example 1: Cloning vector design - Avoid common restriction enzyme sites"""
    print("=" * 70)
    print("Example 1: Cloning Vector Design (Avoiding Restriction Sites)")
    print("=" * 70)
    
    # Create generator using probabilistic model for precise control
    generator = EnhancedGenomeBridgeGenerator(base_model="probabilistic")
    
    # Avoid common restriction enzyme sites
    common_restriction_sites = [
        "GAATTC",  # EcoRI
        "GGATCC",  # BamHI
        "AAGCTT",  # HindIII
        "GTCGAC",  # SalI
        "CTGCAG",  # PstI
        "CCCGGG",  # SmaI
    ]
    
    # Set constraints
    generator.set_motif_constraints(avoid_motifs=common_restriction_sites)
    generator.set_gc_content_constraint(min_gc=0.45, max_gc=0.55)  # Balanced GC content
    generator.set_length_constraint(target_length=800)  # Vector fragment size
    
    # Generate sequences
    sequences = generator.generate_controlled_sequences(num_sequences=10)
    
    # Save and analyze results
    generator.save_sequences(sequences, "cloning_vector_fragments.csv")
    
    print(f"Generated {len(sequences)} vector fragment sequences")
    successful = sum(1 for seq in sequences if seq['satisfies_constraints'])
    print(f"Constraint satisfaction rate: {successful}/{len(sequences)} ({successful/len(sequences)*100:.1f}%)")
    
    # Verify restriction site avoidance
    for i, seq_data in enumerate(sequences[:3]):
        sequence = seq_data['sequence']
        found_sites = []
        for site in common_restriction_sites:
            if site in sequence:
                found_sites.append(site)
        print(f"Sequence {i+1}: Length={len(sequence)}, GC={seq_data['gc_content']:.1%}, Found restriction sites: {found_sites or 'None'}")

def example_2_promoter_design():
    """Example 2: Promoter design - Include regulatory elements, precise GC control"""
    print("\n" + "=" * 70)
    print("Example 2: Promoter Design (Including TATA box and CAAT box)")
    print("=" * 70)
    
    generator = EnhancedGenomeBridgeGenerator(base_model="probabilistic")
    
    # Required promoter elements
    required_elements = [
        "TATAAA",    # TATA box
        "CAAT",      # CAAT box
    ]
    
    # Still avoid restriction enzyme sites
    avoid_sites = ["GAATTC", "GGATCC"]
    
    # Set constraints
    generator.set_motif_constraints(
        require_motifs=required_elements,
        avoid_motifs=avoid_sites
    )
    generator.set_gc_content_constraint(
        min_gc=0.45, max_gc=0.55, 
        window_size=50, enforce_locally=True  # Local GC control
    )
    generator.set_length_constraint(target_length=500)  # Typical promoter length
    
    # Generate sequences
    sequences = generator.generate_controlled_sequences(num_sequences=8)
    
    # Save as FASTA format
    generator.save_sequences(sequences, "promoter_sequences.fasta", format_type="fasta")
    
    print(f"Generated {len(sequences)} promoter sequences")
    
    # Analyze regulatory element inclusion
    for i, seq_data in enumerate(sequences):
        sequence = seq_data['sequence']
        elements_found = []
        for element in required_elements:
            if element in sequence:
                elements_found.append(element)
        
        print(f"Promoter {i+1}: GC={seq_data['gc_content']:.1%}, Contains elements: {elements_found}")

def example_3_synthetic_gene_design():
    """Example 3: Synthetic gene design - Optimize expression and stability"""
    print("\n" + "=" * 70)
    print("Example 3: Synthetic Gene Design (Expression Optimization)")
    print("=" * 70)
    
    generator = EnhancedGenomeBridgeGenerator(base_model="probabilistic")
    
    # Gene design constraints
    # 1. Avoid problematic sequences
    problematic_motifs = [
        "AAAA", "TTTT", "GGGG", "CCCC",  # Homopolymers
        "GAATTC", "GGATCC",              # Restriction sites
        "ATGATG",                        # Multiple start codons
    ]
    
    # 2. Set appropriate composition (slightly GC-rich for expression)
    optimal_composition = {
        'A': 0.24,
        'T': 0.24, 
        'G': 0.26,
        'C': 0.26
    }
    
    # Set constraints
    generator.set_motif_constraints(avoid_motifs=problematic_motifs)
    generator.set_composition_constraint(optimal_composition)
    generator.set_gc_content_constraint(
        min_gc=0.48, max_gc=0.60,  # Optimal GC range for gene expression
        window_size=100, enforce_locally=False  # Global GC control
    )
    generator.set_length_constraint(target_length=1500)  # Gene size
    generator.set_diversity_parameters(temperature=1.2)  # Moderate diversity
    
    # Generate sequences
    sequences = generator.generate_controlled_sequences(num_sequences=5)
    
    # Save results
    generator.save_sequences(sequences, "synthetic_genes.csv")
    
    print(f"Generated {len(sequences)} synthetic gene sequences")
    
    # Analyze composition and characteristics
    for i, seq_data in enumerate(sequences):
        sequence = seq_data['sequence']
        
        # Calculate actual composition
        comp_a = sequence.count('A') / len(sequence)
        comp_t = sequence.count('T') / len(sequence)
        comp_g = sequence.count('G') / len(sequence)
        comp_c = sequence.count('C') / len(sequence)
        
        print(f"Gene {i+1}:")
        print(f"  Length: {len(sequence)} bp, GC: {seq_data['gc_content']:.1%}")
        print(f"  Composition: A={comp_a:.1%}, T={comp_t:.1%}, G={comp_g:.1%}, C={comp_c:.1%}")

def example_4_diversity_library():
    """Example 4: High diversity library construction"""
    print("\n" + "=" * 70)
    print("Example 4: High Diversity Sequence Library")
    print("=" * 70)
    
    generator = EnhancedGenomeBridgeGenerator(base_model="probabilistic")
    
    # Relaxed constraints for maximum diversity
    generator.set_gc_content_constraint(min_gc=0.2, max_gc=0.8)  # Very wide GC range
    generator.set_length_constraint(
        target_length=1200, 
        min_length=1000, 
        max_length=1400  # Allow length variation
    )
    generator.set_diversity_parameters(temperature=2.0)  # High temperature for diversity
    
    # Generate many sequences
    sequences = generator.generate_controlled_sequences(num_sequences=20)
    
    # Save results
    generator.save_sequences(sequences, "diversity_library.csv")
    
    print(f"Generated {len(sequences)} diverse sequences")
    
    # Analyze diversity
    gc_values = [seq['gc_content'] for seq in sequences]
    lengths = [seq['length'] for seq in sequences]
    
    print(f"GC content range: {min(gc_values):.1%} - {max(gc_values):.1%}")
    print(f"Length range: {min(lengths)} - {max(lengths)} bp")
    print(f"GC content standard deviation: {np.std(gc_values):.3f}")

def example_5_species_specific_sequences():
    """Example 5: Species-specific sequence simulation"""
    print("\n" + "=" * 70)
    print("Example 5: Species-Specific Sequences (E.coli vs Human)")
    print("=" * 70)
    
    # E.coli-like sequences
    print("Generating E.coli-like sequences...")
    generator_ecoli = EnhancedGenomeBridgeGenerator(base_model="probabilistic")
    
    # E.coli genome composition (~50% GC)
    ecoli_composition = {
        'A': 0.246,
        'T': 0.246,
        'G': 0.254, 
        'C': 0.254
    }
    
    generator_ecoli.set_composition_constraint(ecoli_composition)
    generator_ecoli.set_gc_content_constraint(min_gc=0.48, max_gc=0.52)
    generator_ecoli.set_length_constraint(target_length=1000)
    
    ecoli_sequences = generator_ecoli.generate_controlled_sequences(num_sequences=5)
    
    # Human-like sequences  
    print("\nGenerating Human-like sequences...")
    generator_human = EnhancedGenomeBridgeGenerator(base_model="probabilistic")
    
    # Human genome composition (~40% GC, AT-rich)
    human_composition = {
        'A': 0.295,
        'T': 0.295,
        'G': 0.205,
        'C': 0.205
    }
    
    generator_human.set_composition_constraint(human_composition)
    generator_human.set_gc_content_constraint(min_gc=0.38, max_gc=0.42)
    generator_human.set_length_constraint(target_length=1000)
    
    human_sequences = generator_human.generate_controlled_sequences(num_sequences=5)
    
    # Combine and save results
    all_sequences = []
    for seq in ecoli_sequences:
        seq['species_type'] = 'E_coli_like'
        all_sequences.append(seq)
    for seq in human_sequences:
        seq['species_type'] = 'Human_like'
        all_sequences.append(seq)
    
    df = pd.DataFrame(all_sequences)
    df.to_csv("species_specific_sequences.csv", index=False)
    
    # Comparative analysis
    print("\nSpecies-specific composition comparison:")
    print("E.coli-like sequences:")
    for i, seq in enumerate(ecoli_sequences):
        sequence = seq['sequence']
        print(f"  Sequence {i+1}: GC={seq['gc_content']:.1%}, "
              f"A={sequence.count('A')/len(sequence):.1%}, "
              f"T={sequence.count('T')/len(sequence):.1%}")
    
    print("Human-like sequences:")
    for i, seq in enumerate(human_sequences):
        sequence = seq['sequence']
        print(f"  Sequence {i+1}: GC={seq['gc_content']:.1%}, "
              f"A={sequence.count('A')/len(sequence):.1%}, "
              f"T={sequence.count('T')/len(sequence):.1%}")

def example_6_user_provided_prompts():
    """Example 6: Using user-provided DNA input as prompts"""
    print("\n" + "=" * 70)
    print("Example 6: User DNA Input-based Sequence Extension")
    print("=" * 70)
    
    # User-provided DNA sequences as starting points
    user_prompts = [
        "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGCCACAGGC",  # Contains His-tag
        "ATGGCTAGCAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATT",  # GFP start
        "ATGAGAGGATCGCATCACCATCACCATCACGGATCCGAATTCGAGCTCCGTCGACA",  # Expression vector
    ]
    
    generator = EnhancedGenomeBridgeGenerator(base_model="probabilistic")
    
    # Set reasonable constraints
    generator.set_gc_content_constraint(min_gc=0.4, max_gc=0.65)
    generator.set_motif_constraints(avoid_motifs=["GAATTC", "GGATCC"])  # Keep some but not all
    generator.set_length_constraint(target_length=900)  # Extend from prompt to 900bp
    
    # Generate using user prompts
    sequences = generator.generate_controlled_sequences(
        num_sequences=6,  # Generate 2 sequences per prompt
        user_prompts=user_prompts
    )
    
    # Save results
    generator.save_sequences(sequences, "user_prompt_extended.fasta", format_type="fasta")
    
    print(f"Generated {len(sequences)} extended sequences based on {len(user_prompts)} user prompts")
    
    # Analyze extension results
    for i, seq_data in enumerate(sequences):
        original_prompt = seq_data['prompt']
        extended_sequence = seq_data['sequence']
        
        print(f"Sequence {i+1}:")
        print(f"  Original prompt length: {len(original_prompt)} bp")
        print(f"  Extended length: {len(extended_sequence)} bp")
        print(f"  GC content: {seq_data['gc_content']:.1%}")
        print(f"  Prompt retained: {'✓' if extended_sequence.startswith(original_prompt) else '✗'}")

def example_7_genslm_advanced_generation():
    """Example 7: GenSLM model for advanced language model-based generation"""
    print("\n" + "=" * 70)
    print("Example 7: GenSLM Advanced Language Model Generation")
    print("=" * 70)
    
    try:
        # Create generator using GenSLM model
        generator = EnhancedGenomeBridgeGenerator(base_model="genslm")
        
        # Set sophisticated constraints for GenSLM
        generator.set_gc_content_constraint(min_gc=0.45, max_gc=0.60)
        generator.set_motif_constraints(
            avoid_motifs=["GAATTC", "GGATCC", "AAGCTT"],  # Common restriction sites
            require_motifs=["ATG"]  # Start codon
        )
        generator.set_composition_constraint({
            'A': 0.26, 'T': 0.26, 'G': 0.24, 'C': 0.24  # Slightly AT-rich
        })
        generator.set_diversity_parameters(temperature=1.1, top_k=40, top_p=0.9)
        generator.set_length_constraint(target_length=1200)
        
        # Generate sequences
        sequences = generator.generate_controlled_sequences(num_sequences=5)
        
        # Save results
        generator.save_sequences(sequences, "genslm_sequences.csv")
        
        print(f"Generated {len(sequences)} sequences using GenSLM model")
        
        # Analyze GenSLM results
        for i, seq_data in enumerate(sequences):
            sequence = seq_data['sequence']
            print(f"GenSLM Sequence {i+1}:")
            print(f"  Length: {len(sequence)} bp")
            print(f"  GC content: {seq_data['gc_content']:.1%}")
            print(f"  Contains ATG: {'✓' if 'ATG' in sequence else '✗'}")
            print(f"  Satisfies constraints: {'✓' if seq_data['satisfies_constraints'] else '✗'}")
            
    except Exception as e:
        print(f"GenSLM example failed (model may not be available): {e}")
        print("Skipping GenSLM example...")

def main():
    """Run all enhanced generation examples"""
    print("🧬 Enhanced GenomeBridge Sequence Generation Control Mechanisms Demo")
    print("Advanced control features extended from existing GenomeBridge models\n")
    
    try:
        example_1_cloning_vector_design()
        example_2_promoter_design()
        example_3_synthetic_gene_design()
        example_4_diversity_library()
        example_5_species_specific_sequences()
        example_6_user_provided_prompts()
        example_7_genslm_advanced_generation()
        
        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70)
        
        print("\nGenerated files:")
        print("- cloning_vector_fragments.csv")
        print("- promoter_sequences.fasta")
        print("- synthetic_genes.csv")
        print("- diversity_library.csv") 
        print("- species_specific_sequences.csv")
        print("- user_prompt_extended.fasta")
        print("- genslm_sequences.csv (if GenSLM available)")
        
        print("\n🎯 Demonstrated control mechanisms:")
        print("✓ GC content control (global and local windows)")
        print("✓ Motif avoidance (restriction enzyme sites)")
        print("✓ Motif inclusion (regulatory elements)")
        print("✓ Nucleotide composition control")
        print("✓ Diversity vs fidelity balance")
        print("✓ Precise length control")
        print("✓ Species-specific simulation")
        print("✓ User prompt sequence extension")
        print("✓ GenSLM advanced language model generation")
        print("✓ Multi-format output (CSV, FASTA)")
        
        print("\n📈 Application scenarios:")
        print("• Cloning vector design")
        print("• Promoter engineering")
        print("• Synthetic gene optimization")
        print("• Sequence library construction")
        print("• Species-specific research")
        print("• Advanced language model applications")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 