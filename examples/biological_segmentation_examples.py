#!/usr/bin/env python3
"""
GenomeBridge Biological Context-Aware Segmentation Examples
Demonstrates sophisticated, domain-specific genomic segmentation beyond basic cleaning

Addresses the critical limitation identified: "One size segmentation does not fit into all biological tasks"

Features demonstrated:
1. Promoter/TSS-centered segmentation for promoter prediction
2. Gene body segmentation for epigenetic mark prediction  
3. Enhancer region segmentation for regulatory element analysis
4. Chromatin domain segmentation for 3D organization
5. Expression-based segmentation following coexpression research
6. Position-based sequence extraction with biological context
7. Feature-based segmentation for specific sequence characteristics

References biological segmentation research:
- Segway semi-automated genomic annotation (https://segway.hoffmanlab.org/)
- Expression-based segmentation (Rubin & Green, BMC Genomics 2013)
- Chromatin organization and domain structure
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from genomeBridge.Data.Download.GenomeDataset import GenomeDataset
import pandas as pd
import numpy as np

def example_1_promoter_tss_segmentation():
    """
    Example 1: Promoter/TSS-centered segmentation for promoter prediction tasks
    
    Biological Context:
    - Promoter prediction requires sequences centered around transcription start sites (TSS)
    - Optimal window: 2kb upstream, 500bp downstream of TSS
    - Critical features: TATA boxes, CpG islands, initiator elements
    """
    print("=" * 70)
    print("Example 1: Promoter/TSS-Centered Segmentation")
    print("For promoter prediction and transcriptional regulation analysis")
    print("=" * 70)
    
    # Create dataset for promoter analysis
    dataset = GenomeDataset("PromoterAnalysis", download_folder="./Promoter_Segmentation", download=False)
    
    print("🎯 Biological Context: Promoter Prediction Task")
    print("   - Task: Identify promoter sequences and predict strength")
    print("   - Biological focus: Transcription start sites (TSS)")
    print("   - Sequence requirements: TSS-centered with regulatory context")
    print("   - Window size: 2000bp upstream + 500bp downstream")
    
    # Apply promoter-specific segmentation
    promoter_segments = dataset.biological_context_segmentation(
        segmentation_strategy='promoter_tss',
        annotation_file=None,  # Would use actual GTF/GFF3 in real implementation
        flanking_regions={'upstream': 2000, 'downstream': 500},
        feature_filters={
            'min_promoter_strength': 0.3,
            'require_tss_evidence': True,
            'exclude_pseudogenes': True
        }
    )
    
    print(f"\n📊 Promoter Segmentation Results:")
    print(f"   - Total promoter regions identified: {promoter_segments['total_regions_identified']}")
    print(f"   - Biologically relevant segments: {len(promoter_segments['biologically_relevant_segments'])}")
    
    # Analyze promoter-specific biological features
    context_summary = promoter_segments['biological_context_summary']
    if 'strategy_specific_metrics' in context_summary:
        metrics = context_summary['strategy_specific_metrics']
        print(f"   - Promoters with TATA boxes: {metrics.get('promoters_with_tata', 0)}")
        print(f"   - Promoters with CpG islands: {metrics.get('promoters_with_cpg', 0)}")
        print(f"   - Average promoter strength: {metrics.get('average_promoter_strength', 0):.2f}")
    
    print(f"\n🧬 Biological Significance:")
    print(f"   ✓ TSS-centered sequences optimal for promoter prediction models")
    print(f"   ✓ Includes core promoter elements (TATA, initiators)")
    print(f"   ✓ Captures regulatory context in flanking regions")
    print(f"   ✓ Filters out non-functional promoter-like sequences")
    
    return promoter_segments

def example_2_gene_body_epigenetic_segmentation():
    """
    Example 2: Gene body segmentation for epigenetic mark prediction
    
    Biological Context:
    - Epigenetic mark prediction requires entire gene body sequences
    - Different histone marks have distinct patterns: H3K4me3 (promoters), H3K36me3 (gene bodies)
    - Gene expression correlation with chromatin accessibility
    """
    print("\n" + "=" * 70)
    print("Example 2: Gene Body Segmentation for Epigenetic Analysis")
    print("For histone modification and chromatin state prediction")
    print("=" * 70)
    
    dataset = GenomeDataset("EpigeneticAnalysis", download_folder="./Epigenetic_Segmentation", download=False)
    
    print("🧬 Biological Context: Epigenetic Mark Prediction")
    print("   - Task: Predict histone modifications and chromatin states")
    print("   - Biological focus: Complete gene bodies with regulatory domains")
    print("   - Features: H3K4me3, H3K27ac, H3K36me3, chromatin accessibility")
    print("   - Sequence scope: Full gene length including introns")
    
    # Apply gene body segmentation
    gene_segments = dataset.biological_context_segmentation(
        segmentation_strategy='gene_body',
        annotation_file=None,  # Would use comprehensive gene annotation
        flanking_regions={'upstream': 1000, 'downstream': 1000},  # Include regulatory flanking
        feature_filters={
            'gene_biotype': ['protein_coding', 'lncRNA'],
            'min_gene_length': 1000,
            'exclude_pseudogenes': True,
            'include_alternative_transcripts': False
        }
    )
    
    print(f"\n📊 Gene Body Segmentation Results:")
    print(f"   - Total genes identified: {gene_segments['total_regions_identified']}")
    print(f"   - Gene body segments: {len(gene_segments['biologically_relevant_segments'])}")
    
    # Analyze gene-specific features
    context_summary = gene_segments['biological_context_summary']
    if 'strategy_specific_metrics' in context_summary:
        metrics = context_summary['strategy_specific_metrics']
        print(f"   - Protein-coding genes: {metrics.get('protein_coding_genes', 0)}")
        print(f"   - Average exon count: {metrics.get('average_exon_count', 0):.1f}")
        print(f"   - High expression genes: {metrics.get('high_expression_genes', 0)}")
    
    print(f"\n🎭 Epigenetic Predictions (Simulated):")
    for segment in gene_segments['biologically_relevant_segments'][:3]:  # Show first 3
        if 'histone_marks_likelihood' in segment:
            marks = segment['histone_marks_likelihood']
            print(f"   Gene {segment.get('gene_id', 'Unknown')}:")
            print(f"     • H3K4me3 (promoter): {marks.get('H3K4me3', 0):.2f}")
            print(f"     • H3K27ac (enhancer): {marks.get('H3K27ac', 0):.2f}")
            print(f"     • H3K36me3 (gene body): {marks.get('H3K36me3', 0):.2f}")
    
    print(f"\n🧬 Biological Significance:")
    print(f"   ✓ Complete gene context for epigenetic state modeling")
    print(f"   ✓ Includes exon-intron structure for splicing effects")
    print(f"   ✓ Captures gene expression regulatory potential")
    print(f"   ✓ Suitable for chromatin accessibility prediction")
    
    return gene_segments

def example_3_enhancer_regulatory_segmentation():
    """
    Example 3: Enhancer region segmentation for regulatory element prediction
    
    Biological Context:
    - Enhancers are distal regulatory elements that can act over large distances
    - Characterized by specific chromatin signatures (H3K27ac, p300 binding)
    - Form chromatin loops with target gene promoters
    """
    print("\n" + "=" * 70)
    print("Example 3: Enhancer Region Segmentation")
    print("For distal regulatory element and chromatin loop prediction")
    print("=" * 70)
    
    dataset = GenomeDataset("EnhancerAnalysis", download_folder="./Enhancer_Segmentation", download=False)
    
    print("🎭 Biological Context: Enhancer Identification & Analysis")
    print("   - Task: Identify and characterize enhancer elements")
    print("   - Biological focus: Distal regulatory regions")
    print("   - Features: TF binding density, chromatin loops, conservation")
    print("   - Regulatory scope: Long-range gene regulation")
    
    # Apply enhancer-specific segmentation
    enhancer_segments = dataset.biological_context_segmentation(
        segmentation_strategy='enhancer_regions',
        annotation_file=None,
        flanking_regions={'upstream': 1000, 'downstream': 1000},
        feature_filters={
            'min_tfbs_density': 0.2,
            'min_conservation_score': 0.6,
            'exclude_promoter_regions': True,
            'min_accessibility_score': 0.5
        }
    )
    
    print(f"\n📊 Enhancer Segmentation Results:")
    print(f"   - Potential enhancer regions: {enhancer_segments['total_regions_identified']}")
    print(f"   - High-confidence enhancers: {len(enhancer_segments['biologically_relevant_segments'])}")
    
    # Analyze enhancer characteristics
    print(f"\n🎯 Enhancer Characteristics (First 3 regions):")
    for i, segment in enumerate(enhancer_segments['biologically_relevant_segments'][:3]):
        print(f"   Enhancer {i+1}:")
        print(f"     • Type: {segment.get('enhancer_type', 'unknown')}")
        print(f"     • TFBS density: {segment.get('tfbs_density', 0):.2f}")
        print(f"     • Conservation score: {segment.get('conservation_score', 0):.2f}")
        print(f"     • Accessibility: {segment.get('accessibility_score', 0):.2f}")
        print(f"     • Target genes: {', '.join(segment.get('target_genes', []))}")
    
    print(f"\n🧬 Biological Significance:")
    print(f"   ✓ Identifies tissue-specific regulatory elements")
    print(f"   ✓ Predicts long-range chromatin interactions")
    print(f"   ✓ Captures enhancer-promoter communication")
    print(f"   ✓ Enables super-enhancer identification")
    
    return enhancer_segments

def example_4_chromatin_domain_segmentation():
    """
    Example 4: Chromatin domain segmentation for 3D organization analysis
    
    Biological Context:
    - Chromatin is organized into topological domains (TADs)
    - A/B compartmentalization reflects active/inactive regions
    - Insulator elements define domain boundaries
    """
    print("\n" + "=" * 70)
    print("Example 4: Chromatin Domain Segmentation")
    print("For 3D genome organization and topological domain analysis")
    print("=" * 70)
    
    dataset = GenomeDataset("ChromatinAnalysis", download_folder="./Chromatin_Segmentation", download=False)
    
    print("🏗️ Biological Context: 3D Chromatin Organization")
    print("   - Task: Identify topological domains and compartments")
    print("   - Biological focus: Chromatin 3D structure")
    print("   - Features: A/B compartments, insulator strength, gene density")
    print("   - Organization: Hierarchical domain structure")
    
    # Apply chromatin domain segmentation
    chromatin_segments = dataset.biological_context_segmentation(
        segmentation_strategy='chromatin_domains',
        annotation_file=None,
        flanking_regions={'upstream': 5000, 'downstream': 5000},
        feature_filters={
            'min_domain_size': 40000,
            'min_insulator_strength': 0.3,
            'require_boundary_elements': True
        }
    )
    
    print(f"\n📊 Chromatin Domain Results:")
    print(f"   - Chromatin domains identified: {chromatin_segments['total_regions_identified']}")
    print(f"   - High-confidence domains: {len(chromatin_segments['biologically_relevant_segments'])}")
    
    # Analyze chromatin organization
    print(f"\n🏗️ Chromatin Organization Analysis:")
    compartment_counts = {'A': 0, 'B': 0}
    total_domains = len(chromatin_segments['biologically_relevant_segments'])
    
    for segment in chromatin_segments['biologically_relevant_segments']:
        compartment = segment.get('compartment', 'unknown')
        if compartment in compartment_counts:
            compartment_counts[compartment] += 1
    
    print(f"   - A compartment domains (active): {compartment_counts['A']}")
    print(f"   - B compartment domains (inactive): {compartment_counts['B']}")
    print(f"   - A/B ratio: {compartment_counts['A']/max(1, compartment_counts['B']):.2f}")
    
    print(f"\n🧬 Biological Significance:")
    print(f"   ✓ Captures 3D chromatin organization principles")
    print(f"   ✓ Identifies topological domain boundaries")
    print(f"   ✓ Predicts chromatin compartmentalization")
    print(f"   ✓ Enables Hi-C interaction prediction")
    
    return chromatin_segments

def example_5_expression_based_segmentation():
    """
    Example 5: Expression-based segmentation following coexpression research
    
    Biological Context:
    - Genes with similar expression patterns often cluster together
    - Coexpression domains reflect shared regulatory mechanisms
    - Based on research by Rubin & Green (BMC Genomics 2013)
    """
    print("\n" + "=" * 70)
    print("Example 5: Expression-Based Coexpression Segmentation")
    print("Following expression-based segmentation research (Rubin & Green 2013)")
    print("=" * 70)
    
    dataset = GenomeDataset("ExpressionAnalysis", download_folder="./Expression_Segmentation", download=False)
    
    print("📊 Biological Context: Coexpression Domain Analysis")
    print("   - Task: Identify coordinately expressed gene clusters")
    print("   - Research basis: Expression-based segmentation (BMC Genomics 2013)")
    print("   - Features: Tissue specificity, functional coherence")
    print("   - Regulatory insight: Shared transcriptional control")
    
    # Apply expression-based segmentation
    expression_segments = dataset.biological_context_segmentation(
        segmentation_strategy='expression_based',
        annotation_file=None,
        flanking_regions={'upstream': 2000, 'downstream': 2000},
        feature_filters={
            'min_coexpression_strength': 0.7,
            'min_functional_coherence': 0.6,
            'tissue_specificity_threshold': 0.8
        }
    )
    
    print(f"\n📊 Expression Segmentation Results:")
    print(f"   - Coexpression domains: {expression_segments['total_regions_identified']}")
    print(f"   - Functionally coherent domains: {len(expression_segments['biologically_relevant_segments'])}")
    
    # Analyze coexpression patterns
    print(f"\n📈 Coexpression Domain Analysis:")
    expression_patterns = {}
    for segment in expression_segments['biologically_relevant_segments']:
        pattern = segment.get('expression_pattern', 'unknown')
        if pattern not in expression_patterns:
            expression_patterns[pattern] = 0
        expression_patterns[pattern] += 1
    
    for pattern, count in expression_patterns.items():
        print(f"   - {pattern} domains: {count}")
    
    print(f"\n🧬 Biological Significance:")
    print(f"   ✓ Identifies coregulated gene clusters")
    print(f"   ✓ Reveals tissue-specific regulatory domains")
    print(f"   ✓ Captures functional gene organization")
    print(f"   ✓ Supports evolutionary conservation analysis")
    
    return expression_segments

def example_6_position_based_extraction():
    """
    Example 6: Position-based sequence extraction with biological context
    
    Demonstrates precise control over sequence selection for specific biological questions
    """
    print("\n" + "=" * 70)
    print("Example 6: Position-Based Sequence Extraction")
    print("Precise genomic coordinate targeting with biological context")
    print("=" * 70)
    
    dataset = GenomeDataset("PositionAnalysis", download_folder="./Position_Extraction", download=False)
    
    print("📍 Biological Context: Targeted Sequence Analysis")
    print("   - Task: Extract sequences at specific biological features")
    print("   - Approach: Coordinate-based with biological annotation")
    print("   - Applications: ChIP-seq peaks, variant analysis, functional sites")
    
    # Define biologically meaningful coordinates
    target_coordinates = [
        {
            'chromosome': 'chr1',
            'start': 1000000,
            'end': 1001000,
            'region_type': 'promoter',
            'annotation': 'house_keeping_gene_promoter',
            'feature_name': 'GAPDH_promoter'
        },
        {
            'chromosome': 'chr2', 
            'start': 5000000,
            'end': 5002000,
            'region_type': 'enhancer',
            'annotation': 'tissue_specific_enhancer',
            'feature_name': 'liver_enhancer_cluster'
        },
        {
            'chromosome': 'chr3',
            'start': 10000000,
            'end': 10000200,
            'region_type': 'tfbs',
            'annotation': 'transcription_factor_binding_site',
            'feature_name': 'p53_binding_motif'
        }
    ]
    
    # Extract sequences using different strategies
    extraction_strategies = ['exact', 'flanking', 'feature_centered']
    
    for strategy in extraction_strategies:
        print(f"\n🎯 Extraction Strategy: {strategy}")
        
        position_results = dataset.position_based_sequence_extraction(
            genomic_coordinates=target_coordinates,
            extraction_strategy=strategy,
            biological_context=f"{strategy}_coordinate_extraction"
        )
        
        print(f"   - Extracted sequences: {len(position_results['extracted_sequences'])}")
        print(f"   - Biological context: {position_results['biological_context']}")
        
        # Show extracted features
        for seq_data in position_results['extracted_sequences']:
            region_type = seq_data.get('genomic_region', 'unknown')
            feature_name = seq_data.get('coordinates', {}).get('feature_name', 'unnamed')
            print(f"     • {region_type}: {feature_name}")
    
    print(f"\n🧬 Biological Applications:")
    print(f"   ✓ ChIP-seq peak sequence analysis")
    print(f"   ✓ Variant effect prediction at specific loci")
    print(f"   ✓ Transcription factor binding site characterization")
    print(f"   ✓ Regulatory element fine-mapping")
    
    return position_results

def example_7_feature_based_segmentation():
    """
    Example 7: Feature-based segmentation for specific sequence characteristics
    
    Segments genome based on biological sequence features rather than arbitrary windows
    """
    print("\n" + "=" * 70)
    print("Example 7: Feature-Based Genomic Segmentation") 
    print("Segmentation based on biological sequence characteristics")
    print("=" * 70)
    
    dataset = GenomeDataset("FeatureAnalysis", download_folder="./Feature_Segmentation", download=False)
    
    print("🔍 Biological Context: Sequence Feature Analysis")
    print("   - Task: Segment genome by biological sequence features")
    print("   - Features: CpG islands, repeats, conserved motifs")
    print("   - Applications: Methylation analysis, repeat evolution, motif discovery")
    
    # Define feature types to segment by
    feature_types = ['cpg_islands', 'repetitive_elements', 'conserved_motifs', 'low_complexity_regions']
    
    feature_results = dataset.feature_based_segmentation(
        feature_types=feature_types,
        sequence_features={
            'cpg_island_criteria': {'min_gc_content': 0.6, 'min_cpg_observed_expected': 0.6},
            'repeat_families': ['LINE', 'SINE', 'LTR'],
            'conservation_threshold': 0.8
        },
        biological_filters={
            'exclude_low_complexity': True,
            'min_feature_length': 200,
            'require_annotation_support': False
        }
    )
    
    print(f"\n📊 Feature Segmentation Results:")
    print(f"   - Feature types analyzed: {len(feature_results['feature_types'])}")
    print(f"   - Feature-based segments: {len(feature_results['feature_based_segments'])}")
    
    # Analyze feature distribution
    if 'biological_significance' in feature_results:
        significance = feature_results['biological_significance']
        print(f"   - Total features identified: {significance.get('total_features', 0)}")
        print(f"   - Average regulatory potential: {significance.get('regulatory_potential', 0):.2f}")
    
    print(f"\n🔬 Feature Type Distribution:")
    feature_counts = {}
    for segment in feature_results['feature_based_segments']:
        feature_type = segment.get('feature_type', 'unknown')
        feature_counts[feature_type] = feature_counts.get(feature_type, 0) + 1
    
    for feature_type, count in feature_counts.items():
        print(f"   - {feature_type}: {count} regions")
    
    print(f"\n🧬 Biological Applications:")
    print(f"   ✓ CpG island methylation analysis")
    print(f"   ✓ Repetitive element evolution studies") 
    print(f"   ✓ Conserved motif functional analysis")
    print(f"   ✓ Genomic feature annotation refinement")
    
    return feature_results

def example_8_comparative_segmentation_analysis():
    """
    Example 8: Comparative analysis of different segmentation strategies
    
    Demonstrates how different biological contexts require different segmentation approaches
    """
    print("\n" + "=" * 70)
    print("Example 8: Comparative Segmentation Strategy Analysis")
    print("Why one-size-fits-all segmentation fails for biological tasks")
    print("=" * 70)
    
    dataset = GenomeDataset("ComparativeAnalysis", download_folder="./Comparative_Segmentation", download=False)
    
    print("📊 Biological Context: Multi-Strategy Comparison")
    print("   - Objective: Compare segmentation strategies for different tasks")
    print("   - Hypothesis: Task-specific segmentation improves biological relevance")
    print("   - Methods: Side-by-side strategy comparison")
    
    # Define biological tasks and their optimal strategies
    biological_tasks = {
        'promoter_prediction': {
            'strategy': 'promoter_tss',
            'flanking': {'upstream': 2000, 'downstream': 500},
            'focus': 'transcriptional initiation'
        },
        'epigenetic_modeling': {
            'strategy': 'gene_body', 
            'flanking': {'upstream': 1000, 'downstream': 1000},
            'focus': 'chromatin modifications'
        },
        'regulatory_analysis': {
            'strategy': 'enhancer_regions',
            'flanking': {'upstream': 1000, 'downstream': 1000},
            'focus': 'distal regulation'
        },
        'domain_organization': {
            'strategy': 'chromatin_domains',
            'flanking': {'upstream': 5000, 'downstream': 5000},
            'focus': '3D chromatin structure'
        }
    }
    
    comparative_results = {}
    
    for task_name, task_config in biological_tasks.items():
        print(f"\n🎯 Analyzing task: {task_name}")
        print(f"   - Strategy: {task_config['strategy']}")
        print(f"   - Biological focus: {task_config['focus']}")
        
        # Apply task-specific segmentation
        task_results = dataset.biological_context_segmentation(
            segmentation_strategy=task_config['strategy'],
            flanking_regions=task_config['flanking']
        )
        
        comparative_results[task_name] = {
            'strategy': task_config['strategy'],
            'segments_identified': task_results['total_regions_identified'],
            'biological_segments': len(task_results['biologically_relevant_segments']),
            'average_length': task_results['biological_context_summary'].get('average_length', 0),
            'biological_focus': task_config['focus']
        }
        
        print(f"   - Segments identified: {task_results['total_regions_identified']}")
        print(f"   - Average segment length: {task_results['biological_context_summary'].get('average_length', 0):.0f} bp")
    
    # Summary comparison
    print(f"\n📊 Segmentation Strategy Comparison:")
    print(f"{'Task':<20} {'Strategy':<18} {'Segments':<10} {'Avg Length':<12} {'Biological Focus'}")
    print("-" * 85)
    
    for task_name, results in comparative_results.items():
        print(f"{task_name:<20} {results['strategy']:<18} {results['segments_identified']:<10} "
              f"{results['average_length']:<12.0f} {results['biological_focus']}")
    
    print(f"\n🧬 Key Insights:")
    print(f"   ✓ Different tasks require fundamentally different segmentation")
    print(f"   ✓ Segment length varies dramatically by biological context")
    print(f"   ✓ Feature focus changes with regulatory mechanism")
    print(f"   ✓ One-size-fits-all approaches lose biological specificity")
    
    print(f"\n💡 Biological Segmentation Principles:")
    print(f"   • Promoter tasks: TSS-centered, upstream regulatory context")
    print(f"   • Epigenetic tasks: Gene body-focused, chromatin domains")
    print(f"   • Regulatory tasks: Distal elements, long-range interactions")
    print(f"   • Structural tasks: Large domains, 3D organization")
    
    return comparative_results

def main():
    """Run all biological segmentation examples"""
    print("🧬 GenomeBridge Biological Context-Aware Segmentation Examples")
    print("Addressing the limitation: 'One size segmentation does not fit into all biological tasks'")
    print("Based on genomic segmentation research and biological principles\n")
    
    try:
        # Run all examples demonstrating biological context
        example_1_promoter_tss_segmentation()
        example_2_gene_body_epigenetic_segmentation()
        example_3_enhancer_regulatory_segmentation()
        example_4_chromatin_domain_segmentation()
        example_5_expression_based_segmentation()
        example_6_position_based_extraction()
        example_7_feature_based_segmentation()
        example_8_comparative_segmentation_analysis()
        
        print("\n" + "=" * 70)
        print("✅ All biological segmentation examples completed successfully!")
        print("=" * 70)
        
        print("\n🎯 Biological Context-Aware Capabilities Demonstrated:")
        print("✓ Promoter/TSS-centered segmentation for transcriptional analysis")
        print("✓ Gene body segmentation for epigenetic mark prediction")
        print("✓ Enhancer region segmentation for regulatory element analysis")
        print("✓ Chromatin domain segmentation for 3D organization studies")
        print("✓ Expression-based segmentation following coexpression research")
        print("✓ Position-based extraction with biological coordinate context")
        print("✓ Feature-based segmentation for sequence characteristic analysis")
        print("✓ Comparative analysis showing task-specific requirements")
        
        print("\n📁 Generated biological segmentation outputs:")
        print("• Promoter segments in ./Promoter_Segmentation/")
        print("• Epigenetic segments in ./Epigenetic_Segmentation/")
        print("• Enhancer segments in ./Enhancer_Segmentation/")
        print("• Chromatin domains in ./Chromatin_Segmentation/")
        print("• Expression domains in ./Expression_Segmentation/")
        print("• Position-based extracts in ./Position_Extraction/")
        print("• Feature-based segments in ./Feature_Segmentation/")
        print("• Comparative analysis in ./Comparative_Segmentation/")
        
        print("\n🧬 Biological Segmentation Principles Addressed:")
        print("• Task-specific biological context determines optimal segmentation")
        print("• Promoter prediction requires TSS-centered sequences")
        print("• Epigenetic analysis needs complete gene body context")
        print("• Regulatory element analysis focuses on distal enhancer regions")
        print("• Chromatin organization studies require large domain context")
        print("• Feature-based segmentation captures specific biological properties")
        
        print("\n📚 Research Foundations:")
        print("• Segway semi-automated genomic annotation (https://segway.hoffmanlab.org/)")
        print("• Expression-based segmentation (Rubin & Green, BMC Genomics 2013)")
        print("• ENCODE chromatin state segmentation principles")
        print("• Hi-C topological domain organization")
        print("• Epigenetic mark distribution patterns")
        
        print("\n🚀 Ready for:")
        print("• Domain-specific genomic machine learning model training")
        print("• Task-optimized sequence feature extraction")
        print("• Biologically-informed data preprocessing pipelines")
        print("• Context-aware genomic annotation refinement")
        print("• Multi-scale biological analysis workflows")
        
    except Exception as e:
        print(f"❌ Error running biological segmentation examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 