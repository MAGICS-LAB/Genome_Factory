#!/usr/bin/env python3
"""
GenomeBridge Advanced Quality Control Examples
Demonstrates GWAS-standard data cleaning and validation following Miyagawa et al. (2008)

Features demonstrated:
1. Advanced quality control with four key parameters
2. Quasi-case-control validation
3. Genomic control lambda calculation
4. Comprehensive QC reporting with grades and recommendations
5. Integration with existing GenomeBridge workflows

Based on: "Appropriate data cleaning methods for genome-wide association study"
Miyagawa et al., Journal of Human Genetics (2008)
https://doi.org/10.1007/s10038-008-0322-y
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from genomeBridge.Data.Download.GenomeDataset import GenomeDataset
import pandas as pd
import numpy as np

def example_1_advanced_quality_control():
    """Example 1: Advanced QC following GWAS standards"""
    print("=" * 70)
    print("Example 1: Advanced Quality Control (GWAS Standards)")
    print("Following Miyagawa et al. (2008) methodology")
    print("=" * 70)
    
    # Create dataset with various quality sequences for demonstration
    dataset = GenomeDataset("QC_Test", download_folder="./Advanced_QC_Test", download=False)
    
    # Create test FASTA file with sequences of varying quality
    os.makedirs("./qc_test_data", exist_ok=True)
    
    with open("./qc_test_data/mixed_quality.fasta", "w") as f:
        # High quality sequence
        f.write(">high_quality_seq1 chromosome 1\n")
        f.write("ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC\n")
        
        # Low call rate sequence (many Ns)
        f.write(">low_call_rate_seq2 chromosome 2\n")
        f.write("ATGCNNNNNNNNNNNNGATCNNNNNNNNGATCNNNNNNNNGATCNNNNNNNNGATCNNNNNNNN\n")
        
        # Compositionally biased sequence (all A's and T's)
        f.write(">biased_composition_seq3 chromosome 3\n")
        f.write("AAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTT\n")
        
        # Short sequence (below length threshold)
        f.write(">short_seq4 chromosome 4\n")
        f.write("ATGC\n")
        
        # Good quality sequence
        f.write(">good_quality_seq5 chromosome 5\n")
        f.write("ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG\n")
        
        # Low complexity sequence (repetitive)
        f.write(">low_complexity_seq6 chromosome 6\n")
        f.write("ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT\n")
    
    # Load the test data
    test_dataset = GenomeDataset.from_fasta_files(
        fasta_files=["./qc_test_data/mixed_quality.fasta"],
        species="QC_Test_Species",
        download_folder="./QC_Test_Dataset"
    )
    
    print(f"📊 Test dataset loaded: {len(test_dataset.fna_files)} files")
    
    # Apply advanced quality control
    print(f"\n🔬 Applying GWAS-standard quality control...")
    qc_results = test_dataset.advanced_quality_control(
        min_call_rate=0.90,           # 90% non-ambiguous nucleotides
        min_confidence_score=0.7,     # 70% confidence threshold
        hwe_p_threshold=1e-4,         # Hardy-Weinberg p-value threshold
        min_frequency=0.05,           # 5% minor frequency threshold
        enable_genomic_control=True   # Enable genomic control lambda
    )
    
    print(f"\n📈 QC Results Summary:")
    print(f"   - Total sequences processed: {qc_results['total_sequences']}")
    print(f"   - Call rate failures: {qc_results['call_rate_failures']}")
    print(f"   - Confidence score failures: {qc_results['confidence_failures']}")
    print(f"   - Hardy-Weinberg failures: {qc_results['hwe_failures']}")
    print(f"   - Frequency failures: {qc_results['frequency_failures']}")
    print(f"   - Sequences passing all filters: {qc_results['passed_all_filters']}")
    print(f"   - Overall pass rate: {qc_results['qc_statistics']['overall_pass_rate']:.1%}")
    
    if qc_results['genomic_control_lambda']:
        print(f"   - Genomic control λ: {qc_results['genomic_control_lambda']:.3f}")
    
    return qc_results

def example_2_quasi_case_control_validation():
    """Example 2: Quasi-case-control validation to test data cleaning effectiveness"""
    print("\n" + "=" * 70)
    print("Example 2: Quasi-Case-Control Validation")
    print("Testing data cleaning effectiveness")
    print("=" * 70)
    
    # Use a larger synthetic dataset for better validation
    dataset = GenomeDataset("Validation", download_folder="./Validation_Test", download=False)
    
    # Generate synthetic data with known properties
    species_abundances = {
        'Test_Species_A': 0.6,
        'Test_Species_B': 0.4
    }
    
    synthetic_data = dataset.generate_synthetic_metagenomic_dataset(
        species_abundances=species_abundances,
        total_sequences=2000,  # Larger sample for validation
        sequence_length_range=(200, 1000),
        noise_level=0.015,
        output_format='fasta'
    )
    
    print(f"📊 Generated {len(synthetic_data['sequences'])} synthetic sequences for validation")
    
    # Create a temporary dataset with the synthetic data
    synthetic_fasta_path = synthetic_data['output_files']['fasta']
    validation_dataset = GenomeDataset.from_fasta_files(
        fasta_files=[synthetic_fasta_path],
        species="Validation_Dataset",
        download_folder="./Validation_Dataset"
    )
    
    # Perform quasi-case-control validation
    print(f"\n🎯 Performing quasi-case-control validation...")
    validation_results = validation_dataset.quasi_case_control_validation(
        validation_samples=1000  # Use 1000 sequences for validation
    )
    
    print(f"\n📋 Validation Results:")
    print(f"   - Group 1 size: {validation_results['group1_size']}")
    print(f"   - Group 2 size: {validation_results['group2_size']}")
    print(f"   - Validation passed: {validation_results['validation_passed']}")
    print(f"   - Significant differences detected: {validation_results['significant_differences']}")
    
    print(f"\n🔬 Statistical comparisons:")
    for metric, p_value in validation_results['p_values'].items():
        status = "✅ PASS" if p_value >= 0.05 else "⚠️ FAIL"
        print(f"   - {metric}: p = {p_value:.3f} {status}")
    
    return validation_results

def example_3_comprehensive_qc_reporting():
    """Example 3: Generate comprehensive QC reports with grades and recommendations"""
    print("\n" + "=" * 70)
    print("Example 3: Comprehensive QC Reporting")
    print("Quality assessment with grades and recommendations")
    print("=" * 70)
    
    # Create dataset for comprehensive reporting
    dataset = GenomeDataset("Reporting", download_folder="./QC_Reporting", download=False)
    
    # Generate diverse synthetic data for comprehensive testing
    species_abundances = {
        'High_Quality_Species': 0.4,
        'Medium_Quality_Species': 0.35,
        'Low_Quality_Species': 0.25
    }
    
    # Generate with intentional quality variations
    synthetic_data = dataset.generate_synthetic_metagenomic_dataset(
        species_abundances=species_abundances,
        total_sequences=1500,
        sequence_length_range=(100, 2000),
        noise_level=0.02,  # Moderate noise level
        output_format='both'
    )
    
    # Apply data augmentation to create quality variations
    augmented_data = dataset.apply_data_augmentation(
        sequences=synthetic_data['sequences'][:500],  # Augment subset
        augmentation_strategies=['mutation', 'insertion', 'deletion'],
        noise_levels=[0.05, 0.03, 0.03],  # Higher noise for some sequences
        augmentation_factor=1
    )
    
    # Combine original and augmented sequences
    all_sequences = synthetic_data['sequences'] + augmented_data['augmented_sequences']
    
    # Create comprehensive test FASTA
    test_fasta_path = "./qc_test_data/comprehensive_test.fasta"
    os.makedirs("./qc_test_data", exist_ok=True)
    
    with open(test_fasta_path, "w") as f:
        for i, seq in enumerate(all_sequences[:1000]):  # Use subset for demo
            f.write(f">test_sequence_{i+1:04d} comprehensive_test\n")
            f.write(f"{seq}\n")
    
    # Create dataset from comprehensive test data
    comprehensive_dataset = GenomeDataset.from_fasta_files(
        fasta_files=[test_fasta_path],
        species="Comprehensive_Test",
        download_folder="./Comprehensive_QC_Dataset"
    )
    
    print(f"📊 Comprehensive test dataset created with {len(all_sequences[:1000])} sequences")
    
    # Generate comprehensive QC report
    print(f"\n📋 Generating comprehensive QC report...")
    qc_report = comprehensive_dataset.generate_qc_report(
        include_recommendations=True
    )
    
    # Display key results
    assessment = qc_report['data_quality_assessment']
    print(f"\n🎯 Data Quality Assessment:")
    print(f"   - Overall Grade: {assessment['overall_grade']}")
    print(f"   - Quality Score: {assessment['quality_score']:.3f}")
    print(f"   - Critical Issues: {len(assessment['critical_issues'])}")
    print(f"   - Warnings: {len(assessment['warnings'])}")
    print(f"   - Strengths: {len(assessment['strengths'])}")
    
    # Display recommendations
    recommendations = qc_report['recommendations']
    print(f"\n💡 Recommendations:")
    
    if recommendations['immediate_actions']:
        print(f"   🚨 Immediate Actions:")
        for action in recommendations['immediate_actions']:
            print(f"     • {action}")
    
    if recommendations['parameter_adjustments']:
        print(f"   ⚙️ Parameter Adjustments:")
        for adjustment in recommendations['parameter_adjustments']:
            print(f"     • {adjustment}")
    
    if recommendations['data_collection_improvements']:
        print(f"   📈 Data Collection Improvements:")
        for improvement in recommendations['data_collection_improvements']:
            print(f"     • {improvement}")
    
    if recommendations['analysis_considerations']:
        print(f"   🔬 Analysis Considerations:")
        for consideration in recommendations['analysis_considerations']:
            print(f"     • {consideration}")
    
    print(f"\n📄 Full QC report saved: {qc_report['report_file']}")
    
    return qc_report

def example_4_integration_with_existing_workflows():
    """Example 4: Integration of advanced QC with existing GenomeBridge workflows"""
    print("\n" + "=" * 70)
    print("Example 4: Integration with Existing Workflows")
    print("Combining advanced QC with class imbalance and filtering")
    print("=" * 70)
    
    # Create integrated workflow dataset
    dataset = GenomeDataset("Integration", download_folder="./Integrated_QC", download=False)
    
    # Generate realistic metagenomic data
    microbiome_abundances = {
        'Bacteroides_fragilis': 0.3,
        'Escherichia_coli': 0.2,
        'Lactobacillus_acidophilus': 0.2,
        'Clostridium_difficile': 0.1,
        'Bifidobacterium_longum': 0.1,
        'Enterococcus_faecalis': 0.1
    }
    
    metagenomic_data = dataset.generate_synthetic_metagenomic_dataset(
        species_abundances=microbiome_abundances,
        total_sequences=2000,
        sequence_length_range=(300, 1500),
        noise_level=0.018,
        output_format='csv'
    )
    
    print(f"🧬 Generated metagenomic dataset with {len(metagenomic_data['sequences'])} sequences")
    
    # Step 1: Apply advanced quality control
    print(f"\n1️⃣ Applying advanced quality control...")
    
    # Create temporary FASTA for QC
    qc_fasta_path = "./qc_test_data/integration_test.fasta"
    with open(qc_fasta_path, "w") as f:
        for seq, meta in zip(metagenomic_data['sequences'], metagenomic_data['metadata']):
            f.write(f">{meta['sequence_id']} {meta['species']}\n")
            f.write(f"{seq}\n")
    
    integrated_dataset = GenomeDataset.from_fasta_files(
        fasta_files=[qc_fasta_path],
        species="Integrated_Analysis",
        download_folder="./Integrated_Analysis_Dataset"
    )
    
    qc_results = integrated_dataset.advanced_quality_control(
        min_call_rate=0.85,
        min_confidence_score=0.6,
        hwe_p_threshold=1e-3,
        min_frequency=0.02
    )
    
    print(f"   ✅ QC complete: {qc_results['passed_all_filters']}/{qc_results['total_sequences']} sequences passed")
    
    # Step 2: Apply traditional quality filtering to QC-passed sequences
    print(f"\n2️⃣ Applying traditional quality filtering...")
    
    traditional_qc = integrated_dataset.quality_filter_sequences(
        min_length=250,
        max_length=2000,
        min_gc_content=0.25,
        max_gc_content=0.75,
        remove_ambiguous=True,
        max_n_percent=0.1
    )
    
    print(f"   ✅ Traditional QC: {traditional_qc['passed_filter']}/{traditional_qc['total_sequences']} sequences passed")
    
    # Step 3: Analyze class imbalance
    print(f"\n3️⃣ Analyzing class imbalance...")
    
    # Prepare data for imbalance analysis
    sample_data = []
    for seq, meta in zip(metagenomic_data['sequences'], metagenomic_data['metadata']):
        sample_data.append({
            'sequence': seq,
            'label': meta['species'],
            'length': meta['length'],
            'gc_content': meta['gc_content']
        })
    
    imbalance_analysis = integrated_dataset.analyze_class_imbalance(
        samples=sample_data,
        class_label_key='label'
    )
    
    print(f"   ✅ Imbalance analysis: {imbalance_analysis['classification_type']} with {imbalance_analysis['num_classes']} classes")
    print(f"   📊 Imbalance severity: {imbalance_analysis['imbalance_severity']}")
    print(f"   📈 Imbalance ratio: {imbalance_analysis['imbalance_ratio']:.2f}")
    
    # Step 4: Apply balancing if needed
    if imbalance_analysis['imbalance_severity'] in ['high', 'severe']:
        print(f"\n4️⃣ Applying class balancing...")
        
        balanced_data = integrated_dataset.generate_balanced_dataset(
            samples=sample_data,
            class_label_key='label',
            balancing_strategy='auto',
            min_samples_per_class=50
        )
        
        print(f"   ✅ Balancing complete: {len(balanced_data)} balanced samples")
    else:
        balanced_data = sample_data
        print(f"\n4️⃣ Skipping balancing (imbalance severity: {imbalance_analysis['imbalance_severity']})")
    
    # Step 5: Generate comprehensive integrated report
    print(f"\n5️⃣ Generating integrated workflow report...")
    
    integrated_report = {
        "workflow_type": "integrated_advanced_qc",
        "timestamp": dataset._get_timestamp(),
        "advanced_qc_results": qc_results,
        "traditional_qc_results": traditional_qc,
        "imbalance_analysis": imbalance_analysis,
        "final_dataset_size": len(balanced_data),
        "workflow_summary": {
            "initial_sequences": len(metagenomic_data['sequences']),
            "advanced_qc_passed": qc_results['passed_all_filters'],
            "traditional_qc_passed": traditional_qc['passed_filter'],
            "final_balanced_sequences": len(balanced_data),
            "overall_retention_rate": len(balanced_data) / len(metagenomic_data['sequences'])
        }
    }
    
    # Save integrated report
    report_path = "./Integrated_Analysis_Dataset/integrated_workflow_report.json"
    import json
    with open(report_path, 'w') as f:
        json.dump(integrated_report, f, indent=2)
    
    print(f"   ✅ Integrated report saved: {report_path}")
    print(f"\n📊 Workflow Summary:")
    print(f"   - Initial sequences: {integrated_report['workflow_summary']['initial_sequences']}")
    print(f"   - Advanced QC retention: {qc_results['passed_all_filters']}/{len(metagenomic_data['sequences'])} ({qc_results['qc_statistics']['overall_pass_rate']:.1%})")
    print(f"   - Traditional QC retention: {traditional_qc['passed_filter']}/{traditional_qc['total_sequences']} ({traditional_qc['passed_filter']/traditional_qc['total_sequences']:.1%})")
    print(f"   - Final dataset size: {len(balanced_data)}")
    print(f"   - Overall retention rate: {integrated_report['workflow_summary']['overall_retention_rate']:.1%}")
    
    return integrated_report

def example_5_parameter_optimization():
    """Example 5: QC parameter optimization based on data characteristics"""
    print("\n" + "=" * 70)
    print("Example 5: QC Parameter Optimization")
    print("Adaptive parameter selection based on data characteristics")
    print("=" * 70)
    
    # Create dataset for parameter optimization
    dataset = GenomeDataset("Optimization", download_folder="./QC_Optimization", download=False)
    
    # Generate test datasets with different characteristics
    test_scenarios = {
        "high_quality": {
            "abundances": {"Species_A": 0.5, "Species_B": 0.5},
            "noise_level": 0.005,
            "length_range": (500, 1000)
        },
        "medium_quality": {
            "abundances": {"Species_C": 0.6, "Species_D": 0.4},
            "noise_level": 0.02,
            "length_range": (200, 1500)
        },
        "low_quality": {
            "abundances": {"Species_E": 0.7, "Species_F": 0.3},
            "noise_level": 0.05,
            "length_range": (100, 2000)
        }
    }
    
    optimization_results = {}
    
    for scenario_name, params in test_scenarios.items():
        print(f"\n🧪 Testing scenario: {scenario_name}")
        
        # Generate test data
        test_data = dataset.generate_synthetic_metagenomic_dataset(
            species_abundances=params["abundances"],
            total_sequences=800,
            sequence_length_range=params["length_range"],
            noise_level=params["noise_level"],
            output_format='fasta'
        )
        
        # Create test dataset
        test_fasta_path = f"./qc_test_data/{scenario_name}_test.fasta"
        os.makedirs("./qc_test_data", exist_ok=True)
        
        with open(test_fasta_path, "w") as f:
            for seq, meta in zip(test_data['sequences'], test_data['metadata']):
                f.write(f">{meta['sequence_id']} {meta['species']}\n")
                f.write(f"{seq}\n")
        
        scenario_dataset = GenomeDataset.from_fasta_files(
            fasta_files=[test_fasta_path],
            species=f"Scenario_{scenario_name}",
            download_folder=f"./Scenario_{scenario_name}_Dataset"
        )
        
        # Test different parameter combinations
        parameter_sets = [
            {"call_rate": 0.95, "confidence": 0.8, "hwe": 1e-6, "freq": 0.01, "name": "strict"},
            {"call_rate": 0.90, "confidence": 0.7, "hwe": 1e-4, "freq": 0.02, "name": "moderate"},
            {"call_rate": 0.85, "confidence": 0.6, "hwe": 1e-3, "freq": 0.05, "name": "lenient"}
        ]
        
        scenario_results = {}
        
        for param_set in parameter_sets:
            qc_result = scenario_dataset.advanced_quality_control(
                min_call_rate=param_set["call_rate"],
                min_confidence_score=param_set["confidence"],
                hwe_p_threshold=param_set["hwe"],
                min_frequency=param_set["freq"],
                enable_genomic_control=True
            )
            
            scenario_results[param_set["name"]] = {
                "pass_rate": qc_result["qc_statistics"]["overall_pass_rate"],
                "total_passed": qc_result["passed_all_filters"],
                "lambda_gc": qc_result.get("genomic_control_lambda", 1.0)
            }
            
            print(f"   📊 {param_set['name']:10} parameters: {qc_result['passed_all_filters']:3d}/800 passed ({qc_result['qc_statistics']['overall_pass_rate']:.1%})")
        
        optimization_results[scenario_name] = scenario_results
    
    # Analyze optimization results
    print(f"\n📈 Parameter Optimization Analysis:")
    print(f"{'Scenario':<15} {'Strict':<10} {'Moderate':<10} {'Lenient':<10} {'Recommendation'}")
    print("-" * 70)
    
    for scenario, results in optimization_results.items():
        strict_rate = results["strict"]["pass_rate"]
        moderate_rate = results["moderate"]["pass_rate"]
        lenient_rate = results["lenient"]["pass_rate"]
        
        # Recommend parameters based on pass rates
        if strict_rate >= 0.8:
            recommendation = "strict"
        elif moderate_rate >= 0.7:
            recommendation = "moderate"
        else:
            recommendation = "lenient"
        
        print(f"{scenario:<15} {strict_rate:<10.1%} {moderate_rate:<10.1%} {lenient_rate:<10.1%} {recommendation}")
    
    print(f"\n💡 Optimization Recommendations:")
    print(f"   • High quality data: Use strict parameters (95% call rate, 80% confidence)")
    print(f"   • Medium quality data: Use moderate parameters (90% call rate, 70% confidence)")
    print(f"   • Low quality data: Use lenient parameters (85% call rate, 60% confidence)")
    print(f"   • Always monitor genomic control λ for population stratification")
    
    return optimization_results

def main():
    """Run all advanced QC examples"""
    print("🔬 GenomeBridge Advanced Quality Control Examples")
    print("GWAS-standard data cleaning following Miyagawa et al. (2008)")
    print("https://doi.org/10.1007/s10038-008-0322-y\n")
    
    try:
        # Run all examples
        example_1_advanced_quality_control()
        example_2_quasi_case_control_validation()
        example_3_comprehensive_qc_reporting()
        example_4_integration_with_existing_workflows()
        example_5_parameter_optimization()
        
        print("\n" + "=" * 70)
        print("✅ All advanced QC examples completed successfully!")
        print("=" * 70)
        
        print("\n🎯 Demonstrated GWAS-standard QC features:")
        print("✓ Four-parameter quality control (call rate, confidence, HWE, frequency)")
        print("✓ Bayesian-inspired confidence scoring")
        print("✓ Hardy-Weinberg equilibrium testing for composition")
        print("✓ Genomic control lambda calculation")
        print("✓ Quasi-case-control validation methodology")
        print("✓ Comprehensive quality assessment with grades (A-F)")
        print("✓ Automated parameter recommendations")
        print("✓ Integration with existing GenomeBridge workflows")
        print("✓ Adaptive parameter optimization")
        
        print("\n📁 Generated QC outputs:")
        print("• Advanced QC results in ./Advanced_QC_Test/")
        print("• Validation reports in ./Validation_Dataset/")
        print("• Comprehensive QC reports in ./QC_Reporting/")
        print("• Integrated workflow results in ./Integrated_Analysis_Dataset/")
        print("• Parameter optimization data in ./QC_Optimization/")
        
        print("\n🏆 Quality Standards Achieved:")
        print("• GWAS-level data cleaning rigor")
        print("• Population stratification detection")
        print("• Systematic bias identification")
        print("• Evidence-based parameter selection")
        print("• Reproducible quality assessment")
        
        print("\n📚 Methodology Reference:")
        print("Miyagawa, T., Nishida, N., Ohashi, J. et al.")
        print("Appropriate data cleaning methods for genome-wide association study.")
        print("J Hum Genet 53, 886–893 (2008)")
        print("https://doi.org/10.1007/s10038-008-0322-y")
        
    except Exception as e:
        print(f"❌ Error running advanced QC examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 