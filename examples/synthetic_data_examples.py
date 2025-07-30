#!/usr/bin/env python3
"""
GenomeBridge Synthetic Data and User FASTA Examples
Demonstrates comprehensive capabilities for handling user-provided data and generating synthetic datasets

Features demonstrated:
1. Loading user-provided FASTA files
2. Generating synthetic metagenomic datasets with species abundances
3. Applying controlled noise for data augmentation
4. Simulating complete metagenomic experiments
5. Data format conversion and validation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from genomeBridge.Data.Download.GenomeDataset import GenomeDataset
import pandas as pd
import numpy as np

def example_1_user_fasta_files():
    """Example 1: Loading and processing user-provided FASTA files"""
    print("=" * 70)
    print("Example 1: User-Provided FASTA Files")
    print("=" * 70)
    
    # Create some example FASTA files for demonstration
    os.makedirs("example_fasta_files", exist_ok=True)
    
    # Example FASTA file 1: E. coli-like sequences
    with open("example_fasta_files/ecoli_example.fasta", "w") as f:
        f.write(">sequence_1 E.coli chromosome\n")
        f.write("ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGCCACAGGCGGTTCGCTTATCGCTGTGGGCCGCACATC\n")
        f.write(">sequence_2 E.coli plasmid\n")
        f.write("GCTAGCGCTAGCGCTAGCATTGCCATTGCCTAGCTAGCTAGCTGAATTCGGATCCAAGCTTATCGATACCGTCGACC\n")
    
    # Example FASTA file 2: Human-like sequences  
    with open("example_fasta_files/human_example.fasta", "w") as f:
        f.write(">sequence_3 human_chr1\n")
        f.write("ATATATATATGCGCGCGCTATATATCGCGCGCGTATATATCGCGCGCGATATATTAGCTAGCTAGC\n")
        f.write(">sequence_4 human_chr2\n")
        f.write("AAAAAATTTTTTGGGGGGCCCCCCAAAAAATTTTTTGGGGGGCCCCCCAAAAATTTTTGGGGCCCC\n")
    
    # Load FASTA files into GenomeDataset
    fasta_files = [
        "example_fasta_files/ecoli_example.fasta",
        "example_fasta_files/human_example.fasta"
    ]
    
    dataset = GenomeDataset.from_fasta_files(
        fasta_files=fasta_files,
        species="Mixed_User_Data",
        download_folder="./User_FASTA_Dataset",
        validate_sequences=True
    )
    
    print(f"✅ User FASTA dataset created:")
    print(f"   - Species: {dataset.species}")
    print(f"   - Files loaded: {len(dataset.fna_files)}")
    print(f"   - Directory: {dataset.download_folder}")
    
    # Apply quality filtering to user data
    quality_results = dataset.quality_filter_sequences(
        min_length=50,
        max_length=1000,
        min_gc_content=0.2,
        max_gc_content=0.8
    )
    
    print(f"   - Quality filtered: {quality_results['passed_filter']}/{quality_results['total_sequences']} sequences")
    
    return dataset

def example_2_synthetic_metagenomic_dataset():
    """Example 2: Generate synthetic metagenomic dataset with species abundances"""
    print("\n" + "=" * 70)
    print("Example 2: Synthetic Metagenomic Dataset Generation")
    print("=" * 70)
    
    # Create a GenomeDataset instance for synthetic generation
    dataset = GenomeDataset("Synthetic", download_folder="./Synthetic_Metagenomics", download=False)
    
    # Define species abundances for a gut microbiome simulation
    gut_microbiome_abundances = {
        'Bacteroides fragilis': 0.35,      # Dominant species
        'Escherichia coli': 0.20,         # Common pathogen
        'Lactobacillus acidophilus': 0.15, # Beneficial bacteria
        'Clostridium difficile': 0.05,    # Pathogen (low abundance)
        'Bifidobacterium longum': 0.10,   # Beneficial bacteria
        'Enterococcus faecalis': 0.08,    # Opportunistic pathogen
        'Prevotella copri': 0.07          # Variable abundance
    }
    
    print(f"🦠 Simulating gut microbiome with {len(gut_microbiome_abundances)} species:")
    for species, abundance in gut_microbiome_abundances.items():
        print(f"   - {species}: {abundance:.1%}")
    
    # Generate synthetic metagenomic dataset
    synthetic_data = dataset.generate_synthetic_metagenomic_dataset(
        species_abundances=gut_microbiome_abundances,
        total_sequences=5000,
        sequence_length_range=(200, 2000),
        noise_level=0.02,  # 2% mutation rate
        output_format='both'
    )
    
    print(f"\n📊 Synthetic dataset statistics:")
    print(f"   - Total sequences generated: {len(synthetic_data['sequences'])}")
    print(f"   - Species representation:")
    for species, count in synthetic_data['species_counts'].items():
        print(f"     • {species}: {count} sequences")
    
    return synthetic_data

def example_3_data_augmentation():
    """Example 3: Apply controlled data augmentation for training enhancement"""
    print("\n" + "=" * 70)
    print("Example 3: Controlled Data Augmentation")
    print("=" * 70)
    
    # Create dataset and get some example sequences
    dataset = GenomeDataset("Synthetic", download_folder="./Augmentation_Test", download=False)
    
    # Example sequences for augmentation
    original_sequences = [
        "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGCCACAGGC",
        "GCTAGCGCTAGCGCTAGCATTGCCATTGCCTAGCTAGCTAGCTGAATTCGGATCC",
        "AAAAAATTTTTTGGGGGGCCCCCCAAAAAATTTTTTGGGGGGCCCCCCAAAAATTT",
        "CGCGCGCGTATATATCGCGCGCGATATATTAGCTAGCTAGCTATATATCGCGCGCG"
    ]
    
    print(f"🔬 Original sequences: {len(original_sequences)}")
    for i, seq in enumerate(original_sequences):
        print(f"   {i+1}. Length: {len(seq)}, GC: {dataset._calculate_gc_content(seq):.1%}")
    
    # Apply comprehensive data augmentation
    augmented_data = dataset.apply_data_augmentation(
        sequences=original_sequences,
        augmentation_strategies=['mutation', 'insertion', 'deletion', 'gc_shift'],
        noise_levels=[0.01, 0.005, 0.005, 0.02],
        augmentation_factor=3
    )
    
    print(f"\n🔄 Augmentation results:")
    print(f"   - Strategies used: {augmented_data['strategies_used']}")
    print(f"   - Noise levels: {augmented_data['noise_levels']}")
    print(f"   - Augmented sequences: {len(augmented_data['augmented_sequences'])}")
    
    # Analyze augmentation effects
    original_gc = [dataset._calculate_gc_content(seq) for seq in original_sequences]
    augmented_gc = [dataset._calculate_gc_content(seq) for seq in augmented_data['augmented_sequences']]
    
    print(f"\n📈 GC content analysis:")
    print(f"   - Original mean GC: {np.mean(original_gc):.1%} ± {np.std(original_gc):.1%}")
    print(f"   - Augmented mean GC: {np.mean(augmented_gc):.1%} ± {np.std(augmented_gc):.1%}")
    
    return augmented_data

def example_4_metagenomic_experiment_simulation():
    """Example 4: Complete metagenomic experiment simulation"""
    print("\n" + "=" * 70)
    print("Example 4: Complete Metagenomic Experiment Simulation")
    print("=" * 70)
    
    # Create dataset for experiment simulation
    dataset = GenomeDataset("Experiment", download_folder="./Metagenomic_Experiment", download=False)
    
    # Define baseline healthy gut microbiome
    healthy_gut_baseline = {
        'Bacteroides fragilis': 0.30,
        'Lactobacillus acidophilus': 0.25,
        'Bifidobacterium longum': 0.20,
        'Escherichia coli': 0.10,
        'Clostridium difficile': 0.05,
        'Enterococcus faecalis': 0.10
    }
    
    # Design experimental conditions
    experimental_design = {
        'sequences_per_condition': 3000,
        'sequence_length_range': (300, 1500),
        'base_noise_level': 0.015,
        'treatments': {
            'antibiotic_treatment': {
                'abundance_changes': {
                    'Lactobacillus acidophilus': 0.3,  # Reduced by antibiotics
                    'Bifidobacterium longum': 0.4,    # Reduced by antibiotics
                    'Clostridium difficile': 3.0,     # Opportunistic overgrowth
                    'Escherichia coli': 1.5           # Slight increase
                },
                'noise_level': 0.02  # Higher mutation rate due to stress
            },
            'probiotic_treatment': {
                'abundance_changes': {
                    'Lactobacillus acidophilus': 2.0,  # Increased by probiotics
                    'Bifidobacterium longum': 1.8,    # Increased by probiotics
                    'Clostridium difficile': 0.2,     # Reduced by competition
                    'Bacteroides fragilis': 0.9       # Slightly reduced
                },
                'noise_level': 0.01  # Lower mutation rate, healthier
            },
            'disease_state': {
                'abundance_changes': {
                    'Clostridium difficile': 5.0,     # Pathogenic overgrowth
                    'Enterococcus faecalis': 2.0,     # Opportunistic increase
                    'Lactobacillus acidophilus': 0.2, # Depleted beneficial
                    'Bifidobacterium longum': 0.3     # Depleted beneficial
                },
                'noise_level': 0.03  # High mutation rate in disease
            }
        }
    }
    
    print(f"🧪 Experimental design:")
    print(f"   - Conditions: Control + {len(experimental_design['treatments'])} treatments")
    print(f"   - Sequences per condition: {experimental_design['sequences_per_condition']}")
    print(f"   - Treatment conditions: {list(experimental_design['treatments'].keys())}")
    
    # Simulate complete experiment
    experiment_results = dataset.simulate_metagenomic_experiment(
        experimental_design=experimental_design,
        base_species_abundances=healthy_gut_baseline,
        output_dir="./Complete_Metagenomic_Experiment"
    )
    
    print(f"\n📊 Experiment simulation results:")
    for condition_name, condition_data in experiment_results['conditions'].items():
        sequences_count = len(condition_data['sequences'])
        avg_gc = np.mean([meta['gc_content'] for meta in condition_data['metadata']])
        print(f"   - {condition_name}: {sequences_count} sequences, avg GC: {avg_gc:.1%}")
    
    # Analyze species abundance changes
    print(f"\n🔬 Species abundance changes:")
    control_counts = experiment_results['conditions']['control']['species_counts']
    
    for treatment_name in experimental_design['treatments'].keys():
        treatment_counts = experiment_results['conditions'][treatment_name]['species_counts']
        print(f"\n   {treatment_name}:")
        for species in healthy_gut_baseline.keys():
            control_count = control_counts.get(species, 0)
            treatment_count = treatment_counts.get(species, 0)
            if control_count > 0:
                fold_change = treatment_count / control_count
                print(f"     • {species}: {fold_change:.2f}x change")
    
    return experiment_results

def example_5_format_conversion_and_validation():
    """Example 5: Data format conversion and sequence validation"""
    print("\n" + "=" * 70)
    print("Example 5: Data Format Conversion and Validation")
    print("=" * 70)
    
    # Load example dataset
    dataset = GenomeDataset("Validation", download_folder="./Format_Conversion", download=False)
    
    # Create sequences with various issues for validation demonstration
    problematic_sequences = [
        "ATGCCCGGGAAATTTCCCGGGAAA",  # Normal sequence
        "ATGC2CGGGAAAXTTCCCGGGAAA",  # Contains invalid characters
        "atgcccgggaaatttcccgggaaa",  # Lowercase (should be converted)
        "ATGC-CCG_GGAAA!!!TTTCCCGGGAAA",  # Contains special characters
        "",  # Empty sequence
        "NNNNNNNNNNNNNNNNNNN",  # All N's
        "ATGCCCGGGAAATTTCCCGGGAAAATGCCCGGGAAATTTCCCGGGAAA"  # Very long sequence
    ]
    
    print(f"🔍 Validating {len(problematic_sequences)} sequences with potential issues...")
    
    validated_sequences = []
    for i, seq in enumerate(problematic_sequences):
        print(f"\nSequence {i+1}: '{seq[:20]}{'...' if len(seq) > 20 else ''}'")
        print(f"   Original length: {len(seq)}")
        
        if seq:  # Skip empty sequences
            validated = dataset._validate_and_clean_sequence(seq)
            validated_sequences.append(validated)
            print(f"   Cleaned length: {len(validated)}")
            print(f"   GC content: {dataset._calculate_gc_content(validated):.1%}")
        else:
            print("   Skipped: Empty sequence")
    
    # Generate synthetic dataset for format conversion
    print(f"\n📁 Format conversion demonstration:")
    species_abundances = {
        'Test_Species_1': 0.6,
        'Test_Species_2': 0.4
    }
    
    synthetic_data = dataset.generate_synthetic_metagenomic_dataset(
        species_abundances=species_abundances,
        total_sequences=100,
        sequence_length_range=(100, 500),
        noise_level=0.01,
        output_format='both'  # Generate both FASTA and CSV
    )
    
    print(f"   ✅ Generated files:")
    for format_type, file_path in synthetic_data['output_files'].items():
        file_size = os.path.getsize(file_path) / 1024  # Size in KB
        print(f"     - {format_type.upper()}: {os.path.basename(file_path)} ({file_size:.1f} KB)")
    
    return synthetic_data

def example_6_integration_with_existing_workflows():
    """Example 6: Integration with existing GenomeBridge workflows"""
    print("\n" + "=" * 70)
    print("Example 6: Integration with Existing GenomeBridge Workflows")
    print("=" * 70)
    
    # Create synthetic dataset
    dataset = GenomeDataset("Integration", download_folder="./Workflow_Integration", download=False)
    
    # Generate synthetic data
    species_abundances = {
        'Synthetic_Pathogen': 0.2,
        'Synthetic_Commensal': 0.8
    }
    
    synthetic_data = dataset.generate_synthetic_metagenomic_dataset(
        species_abundances=species_abundances,
        total_sequences=1000,
        sequence_length_range=(500, 2000),
        noise_level=0.015,
        output_format='csv'
    )
    
    # Extract sequences for further processing
    sequences = synthetic_data['sequences']
    metadata = synthetic_data['metadata']
    
    # Apply class imbalance analysis (using existing GenomeBridge functionality)
    sample_data = []
    for seq, meta in zip(sequences, metadata):
        sample_data.append({
            'sequence': seq,
            'label': meta['species'],
            'length': meta['length'],
            'gc_content': meta['gc_content']
        })
    
    # Analyze class imbalance
    imbalance_analysis = dataset.analyze_class_imbalance(
        samples=sample_data,
        class_label_key='label'
    )
    
    print(f"🔬 Integration with existing workflows:")
    print(f"   - Class imbalance analysis: {imbalance_analysis['classification_type']}")
    print(f"   - Imbalance severity: {imbalance_analysis['imbalance_severity']}")
    print(f"   - Classes: {list(imbalance_analysis['class_counts'].keys())}")
    
    # Apply balancing if needed
    if imbalance_analysis['imbalance_severity'] in ['high', 'severe']:
        balanced_data = dataset.generate_balanced_dataset(
            samples=sample_data,
            class_label_key='label',
            balancing_strategy='auto'
        )
        print(f"   - Applied balancing: {len(balanced_data)} balanced samples")
    
    # Apply quality filtering
    sequence_list = [sample['sequence'] for sample in sample_data]
    quality_results = dataset.quality_filter_sequences(
        min_length=400,
        max_length=2200,
        min_gc_content=0.3,
        max_gc_content=0.7
    )
    
    print(f"   - Quality filtering: {quality_results['passed_filter']}/{quality_results['total_sequences']} passed")
    
    # Generate comprehensive report
    report = dataset.generate_comprehensive_balance_report(
        samples=sample_data,
        class_label_key='label',
        include_recommendations=True
    )
    
    print(f"   - Generated comprehensive report with {len(report['balancing_recommendations']['strategy_recommendations'])} recommendations")
    
    return {
        'synthetic_data': synthetic_data,
        'imbalance_analysis': imbalance_analysis,
        'quality_results': quality_results,
        'report': report
    }

def main():
    """Run all synthetic data and user FASTA examples"""
    print("🧬 GenomeBridge Synthetic Data & User FASTA Examples")
    print("Comprehensive demonstration of data handling and generation capabilities\n")
    
    try:
        # Run all examples
        example_1_user_fasta_files()
        example_2_synthetic_metagenomic_dataset()
        example_3_data_augmentation()
        example_4_metagenomic_experiment_simulation()
        example_5_format_conversion_and_validation()
        example_6_integration_with_existing_workflows()
        
        print("\n" + "=" * 70)
        print("✅ All synthetic data examples completed successfully!")
        print("=" * 70)
        
        print("\n🎯 Demonstrated capabilities:")
        print("✓ User-provided FASTA file loading and validation")
        print("✓ Synthetic metagenomic dataset generation with species abundances")
        print("✓ Controlled data augmentation with multiple strategies")
        print("✓ Complete metagenomic experiment simulation")
        print("✓ Data format conversion (FASTA ↔ CSV)")
        print("✓ Sequence validation and cleaning")
        print("✓ Integration with existing GenomeBridge workflows")
        print("✓ Species-specific nucleotide composition modeling")
        print("✓ Noise injection for robustness testing")
        print("✓ Multi-condition experimental design")
        
        print("\n📁 Generated outputs:")
        print("• User FASTA datasets in ./User_FASTA_Dataset/")
        print("• Synthetic metagenomics in ./Synthetic_Metagenomics/")
        print("• Augmentation tests in ./Augmentation_Test/")
        print("• Complete experiment in ./Complete_Metagenomic_Experiment/")
        print("• Format conversion examples in ./Format_Conversion/")
        print("• Workflow integration in ./Workflow_Integration/")
        
        print("\n🚀 Ready for:")
        print("• Model training with augmented datasets")
        print("• Benchmarking with controlled synthetic data")
        print("• Robustness testing with noisy sequences")
        print("• Metagenomic analysis pipeline development")
        print("• Custom dataset preparation and validation")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 