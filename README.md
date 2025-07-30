# GenomeBridge: An Integrated Platform for Tuning and Deploying Genomic Foundation Models

![overview](https://github.com/user-attachments/assets/5999aac8-3e01-4b54-ba3b-2239753c4dd8)


GenomeBridge is a Python-based integrated library for tuning and deploying genomic foundation models (GFMs). It simplifies data collection (including downloading DNA sequences from NCBI and preprocessing), offers advanced and efficient tuning methods (full tuning, LoRA, and adapter tuning), and supports inference (embedding extraction and DNA sequence generation) and benchmarking of various GFMs.

## Supported Models
The "Variant Type" column specifies how model variants differ: by parameter **Size** or by maximum input **Sequence Length**.

| Model Name             | Variant Type    | Variants                                   |
| ---------------------- | --------------- | ------------------------------------------------ |
| DNABERT-2              | Size            | 117M                                             |
| Hyenadna               | Sequence Length | 1K / 16K / 32K / 160K / 450K / 1M              |
| Nucleotide Transformer | Size            | 50M / 100M / 250M / 500M / 1B / 2.5B             |
| Caduceus               | Sequence Length | 1K / 131K                                        |
| GenomeOcean            | Size            | 100M / 500M / 4B                                 |
| EVO                    | Sequence Length | 8K / 131K                                        |

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/xxx/GenomeBridge.git
    cd GenomeBridge
2.  **Install dependencies:**
    ```bash
    # Install primary Python dependencies from requirements file
    pip install -r requirements.txt

    # Install CUDA Toolkit and Compiler 
    # Ensure you have a compatible NVIDIA driver installed and are in a Conda environment.
    conda install cudatoolkit==11.8 -c nvidia
    conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc

    # Install additional dependencies for specific features (e.g., Mamba support, Flash Attention)
    
    pip install mamba-ssm==2.2.2 flash-attn==2.7.2.post1

    # Install NCBI Datasets CLI (required for NCBI data download feature)
    conda install conda-forge::ncbi-datasets-cli

    # Install EVO from source
    git clone https://github.com/evo-design/evo.git
    cd evo
    pip install .
    cd .. 
    # IMPORTANT: Return to the GenomeBridge root directory before the next step

    # Install GenomeBridge in editable mode
    pip install -e .
    ```

3.  **Environment Notes:**
    *   For **Genome Ocean**, use `transformers==4.44.2`.
    *   For **other models**, use `transformers==4.29.2`.
    *   For **DNABERT-2**, ensure `triton` is uninstalled: `pip uninstall triton`.

## Usage via CLI (`genomebridge-cli`)

GenomeBridge uses YAML configuration files to define tasks. Example files are provided in `genomeBridge/Examples/`. You can customize the parameters within these files, ensuring you maintain the required YAML structure.

### Data Download

Download genomic data from NCBI:

1.  **Using a config file:** Specify download parameters in a YAML file.
    **Download by species:**
    ```bash
    genomebridge-cli download genomeBridge/Examples/download_by_species.yaml
    ```
    **Download by Link:**
    ```bash
    genomebridge-cli download genomeBridge/Examples/download_by_link.yaml
    ```
2.  **Interactively:** Run the command without a config file and follow the prompts in the terminal to specify your download criteria (supports both species-based and link-based downloads).
    ```bash
    genomebridge-cli download
    ```
*Note:* The list of species and their taxon IDs used for downloads is stored in `genomeBridge/Data/Download/Datasets_species_taxonid_dict.json`. This file is not exhaustive; you can extend it by adding new species-to-taxonID pairs to download data for other species as needed.

### Data Processing

GenomeBridge provides tools to prepare data for model fine-tuning. This includes processing data downloaded from NCBI or formatting your own custom datasets.

**1. Processing NCBI Data:**

*   Gather data downloaded using the `download` command into a single folder.
*   Run the processing command with a config file:
    ```bash
    genomebridge-cli process genomeBridge/Examples/process_data.yaml
    ```
*   The processed data will be ready for input into the model for fine-tuning.

**2. Preparing Custom Datasets:**

If you have your own dataset, format it as follows:

*   Separate your data into three CSV files: `train.csv`, `dev.csv`, and `test.csv`.
*   Each CSV file must have two columns:
    *   The first column should contain the DNA sequences (e.g., `sequence`).
    *   The second column should contain the corresponding labels (e.g., `label`).
        *   For **classification** tasks, labels should be integers (e.g., 0, 1, 2...).
        *   For **regression** tasks, labels should be continuous numbers.
*   Place these three CSV files (`train.csv`, `dev.csv`, `test.csv`) together in a single folder.
*   This folder can then be specified as the input data directory in your training configuration YAML file.

### Training

For fine-tuning GFMs, GenomeBridge supports two primary task types: **classification** and **regression**. You specify the desired `task_type` in the training YAML configuration file.

Fine-tune GFMs using different methods:

*   **Full Fine-tuning:**
    ```bash
    genomebridge-cli train genomeBridge/Examples/train_full.yaml
    ```
*   **LoRA (Low-Rank Adaptation):**
    ```bash
    genomebridge-cli train genomeBridge/Examples/train_lora.yaml
    ```
    *   Specify target modules in the YAML file:
        *   `all`: Targets all linear layers.
        *   `all_in_and_out_proj`: Targets input/output projection layers and the final classification layer.
        *   *Custom*: Specify module names directly.
*   **Adapter:**
    ```bash
    genomebridge-cli train genomeBridge/Examples/train_adapter.yaml
    ```
    *   Customize the adapter architecture in `genomeBridge/Train/workflow/adapter/adapter_model/Adapter.py` for potentially better performance on specific downstream tasks.

    *Note:* Training settings like batch size, learning rate, and epochs can be customized in the respective YAML files for all methods.

    **Note on Flash Attention:** To enable Flash Attention, set the `flash_attention` argument to `true` in your YAML configuration file. You must also enable mixed-precision training by setting either `bf16: true` or `fp16: true`. If `flash_attention` is set to `false`, or if a specific GFM does not support this argument, the model's default attention mechanism will be used.

    **Benchmarking:** After fine-tuning, performance metrics are saved to a JSON file. You can use these metrics for benchmarking (e.g., comparing the performance of different models or tuning methods on specific tasks).

### Inference

Use trained models for prediction, generation, or embedding extraction:

1.  **Prediction:** (Predict properties of DNA sequences). Ensure the `task_type` specified in your inference YAML file (`classification` or `regression`) matches the task the model was originally fine-tuned for.
    *   **Full:**
        ```bash
        genomebridge-cli inference genomeBridge/Examples/inference_full.yaml
        ```
    *   **LoRA:**
        ```bash
        genomebridge-cli inference genomeBridge/Examples/inference_lora.yaml
        ```
    *   **Adapter:**
        ```bash
        genomebridge-cli inference genomeBridge/Examples/inference_adapter.yaml
        ```
        *   *Note:* For Adapter-based classification, specify the number of labels (`num_label`) in the YAML. For regression, set `num_label: 1`. Full/LoRA methods infer this automatically.

2.  **Generation:** (Generate new DNA sequences based on existing ones). Applicable to compatible GFMs.
    *   For **GenomeOcean**:
        ```bash
        genomebridge-cli inference genomeBridge/Examples/inference_generation_genomeocean.yaml
        ```
    *   For **Evo**:
        ```bash
        genomebridge-cli inference genomeBridge/Examples/inference_generation_evo.yaml
        ```

3.  **Embedding Extraction:** (Extract the last hidden state embeddings from sequences).
    *   General Case:
        ```bash
        genomebridge-cli inference genomeBridge/Examples/inference_extract.yaml
        ```
    *   For **Evo** specifically:
        ```bash
        genomebridge-cli inference genomeBridge/Examples/inference_extract_evo.yaml
        ```


## Usage via Web UI

Access all GenomeBridge functionalities through a graphical interface:

```bash
genomebridge-cli webui
```

This command launches a web server. Open the provided URL in your browser to use the WebUI.

## Reference

```
LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models.
Zheng, Yaowei, Richong Zhang, Junhao Zhang, YeYanhan YeYanhan, and Zheyan Luo.
In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 400-410. 2024.
```

## Advanced Data Processing Features

GenomeBridge now includes sophisticated biological context-aware data processing capabilities that address domain-specific genomic analysis needs. These features go beyond basic data cleaning to provide biologically-informed segmentation, quality control, and synthetic data generation.

### 🧬 Biological Context-Aware Segmentation

The enhanced GenomeBridge addresses the critical limitation that "one size segmentation does not fit into all biological tasks" by providing task-specific genomic segmentation strategies.

#### Quick Start Example

```python
from genomeBridge.Data.Download.GenomeDataset import GenomeDataset

# Initialize dataset
dataset = GenomeDataset("YourSpecies", download_folder="./data", download=False)

# 1. Promoter prediction segmentation (TSS-centered)
promoter_segments = dataset.biological_context_segmentation(
    segmentation_strategy='promoter_tss',
    flanking_regions={'upstream': 2000, 'downstream': 500}
)

# 2. Gene body segmentation for epigenetic analysis
gene_segments = dataset.biological_context_segmentation(
    segmentation_strategy='gene_body',
    flanking_regions={'upstream': 1000, 'downstream': 1000}
)

# 3. Enhancer region segmentation for regulatory analysis
enhancer_segments = dataset.biological_context_segmentation(
    segmentation_strategy='enhancer_regions',
    flanking_regions={'upstream': 1000, 'downstream': 1000}
)

# 4. Position-based sequence extraction
position_results = dataset.position_based_sequence_extraction(
    genomic_coordinates=[{
        'chromosome': 'chr1',
        'start': 1000000,
        'end': 1001000,
        'region_type': 'promoter',
        'annotation': 'GAPDH_promoter'
    }],
    extraction_strategy='feature_centered'
)

# 5. Feature-based segmentation
feature_results = dataset.feature_based_segmentation(
    feature_types=['cpg_islands', 'conserved_motifs', 'repetitive_elements'],
    biological_filters={'min_conservation_score': 0.8}
)
```

### 📊 Advanced Quality Control

GenomeBridge implements GWAS-standard quality control following Miyagawa et al. (2008) methodology:

```python
# Apply advanced quality control
qc_results = dataset.advanced_quality_control(
    min_call_rate=0.95,
    min_confidence_score=0.8,
    hwe_p_threshold=1e-6,
    min_frequency=0.01
)

# Perform quasi-case-control validation
validation_results = dataset.quasi_case_control_validation(
    validation_samples=1000
)

# Generate comprehensive QC report
qc_report = dataset.generate_qc_report(
    qc_results=qc_results,
    validation_results=validation_results,
    include_recommendations=True
)

print(f"Data quality score: {qc_report['overall_quality_score']:.2f}")
print(f"Recommendations: {qc_report['recommendations']['priority_actions']}")
```

### 🧪 Synthetic Data Generation & Augmentation

Create biologically realistic synthetic genomic datasets:

```python
# Generate synthetic metagenomic dataset
synthetic_data = dataset.generate_synthetic_metagenomic_dataset(
    species_abundances={
        'Escherichia_coli': 0.4,
        'Bacillus_subtilis': 0.3,
        'Pseudomonas_aeruginosa': 0.3
    },
    total_sequences=5000,
    sequence_length_range=(500, 2000),
    noise_level=0.02
)

# Apply data augmentation with controlled noise
augmented_data = dataset.apply_data_augmentation(
    sequences=['ATGCGATCGATCG', 'CGATCGATCGATC'],
    augmentation_strategies=['mutation', 'insertion', 'gc_shift'],
    noise_levels=[0.01, 0.02, 0.03],
    augmentation_factor=3
)

# Load and process user-provided FASTA files
custom_dataset = GenomeDataset.from_fasta_files(
    fasta_files=['./custom_data/sequences.fasta'],
    species="CustomSpecies",
    validate_sequences=True
)
```

### 🎯 Unified Class Imbalance Handling

Handle both binary and multi-class imbalance scenarios:

```python
# Analyze class imbalance
imbalance_analysis = dataset.analyze_class_imbalance(
    samples=[
        {'sequence': 'ATGC...', 'label': 'rare_variant'},
        {'sequence': 'CGTA...', 'label': 'common_variant'},
        # ... more samples
    ],
    class_label_key='label'
)

# Generate balanced dataset
balanced_samples = dataset.generate_balanced_dataset(
    samples=samples,
    balancing_strategy='auto',  # Auto-selects best strategy
    class_label_key='label'
)

# Handle specialized imbalance (e.g., rare variant prediction)
specialized_balance = dataset.handle_specialized_imbalance(
    samples=samples,
    domain_type='rare_variant_prediction',
    variant_type_key='variant_type',
    frequency_threshold=0.01
)
```

### 🔬 Complete Example Workflow

Run the comprehensive biological segmentation examples:

```bash
# Execute all biological context-aware segmentation examples
python GenomeBridge/examples/biological_segmentation_examples.py
```

This will demonstrate:
- ✅ **Promoter/TSS-centered segmentation** for promoter prediction tasks
- ✅ **Gene body segmentation** for epigenetic mark prediction
- ✅ **Enhancer region segmentation** for regulatory element analysis
- ✅ **Chromatin domain segmentation** for 3D organization studies
- ✅ **Expression-based segmentation** following coexpression research
- ✅ **Position-based extraction** with biological coordinate context
- ✅ **Feature-based segmentation** for sequence characteristic analysis
- ✅ **Comparative analysis** showing task-specific requirements

### 📚 Research Foundations

The biological segmentation capabilities are based on established genomic research:

- **[Segway Framework](https://segway.hoffmanlab.org/)**: Semi-automated genomic annotation
- **[Expression-based Segmentation](https://bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-14-812)**: Coexpression domain identification (Rubin & Green, BMC Genomics 2013)
- **ENCODE Chromatin States**: Histone modification-based segmentation principles
- **Hi-C Domain Organization**: 3D chromatin structure and topological domains
- **GWAS Quality Control**: Following Miyagawa et al. (2008) methodology

### 🚀 Key Improvements

✨ **Task-Specific Biological Context**: Different genomic tasks now use appropriate segmentation strategies  
✨ **Position & Feature Control**: Precise sequence selection based on genomic coordinates and biological features  
✨ **Advanced Quality Control**: GWAS-standard QC with statistical validation  
✨ **Synthetic Data Generation**: Biologically realistic metagenomic dataset simulation  
✨ **Unified Imbalance Handling**: Comprehensive solution for both binary and multi-class scenarios  
✨ **HIPAA Compliance**: Enterprise-grade privacy protection for human genomic data  

These enhancements make GenomeBridge suitable for sophisticated genomic machine learning workflows that require biologically-informed data preprocessing and domain-specific feature extraction.

## 🔬 Model Interpretation and Analysis

GenomeBridge provides comprehensive tools for understanding and interpreting genomic foundation models through multiple complementary approaches. This unified framework combines attention visualization, feature importance analysis, sparse autoencoder (SAE) interpretation, and cooperative multi-model analysis to provide deep insights into model behavior and biological significance.

### 🎯 Key Interpretation Methods

#### 1. 🧠 Attention-Based Interpretation
- **🔍 Attention Map Visualization**: Interactive heatmaps showing attention patterns across biological sequences
- **📊 Feature Importance Analysis**: Gradient-based and integrated gradient methods for feature importance scoring
- **🤝 Cooperative Model Interpretation**: Ensemble predictions and interpretations from multiple genomic foundation models
- **🎨 Comprehensive Visualization**: Multi-panel reports with attention analysis, importance scores, and biological function predictions

#### 2. 🔬 Sparse Autoencoder (SAE) Analysis
- **🧬 Latent Feature Discovery**: Identify interpretable features learned by genomic foundation models
- **📈 Ridge Regression Evaluation**: Quantitative assessment of feature importance for downstream tasks
- **🎯 First-token vs Mean-pooled Analysis**: Compare different pooling strategies for sequence representation
- **📊 Feature Weight Analysis**: Understand which SAE features contribute most to biological predictions

### 🚀 Quick Start Guide

#### Method 1: Attention-Based Cooperative Interpretation

```python
from biological_insights_interpreter import (
    CooperativeInterpreter, 
    BiologicalInsightsVisualizer,
    AttentionVisualizer,
    FeatureImportanceAnalyzer
)

# Initialize with your trained models
models = {
    'dnabert': your_dnabert_model,
    'nucleotide_transformer': your_nt_model,
    'hyenadna': your_hyena_model
}

# Create cooperative interpreter
interpreter = CooperativeInterpreter(models)

# Set model weights for ensemble
interpreter.set_model_weights({
    'dnabert': 0.4,
    'nucleotide_transformer': 0.4,
    'hyenadna': 0.2
})

# Analyze a DNA sequence
sequence = "ATGCGATCGATCGATCGATCGATCGATC..."
input_tensor = preprocess_sequence(sequence)  # Your preprocessing function

# Generate comprehensive insights
insight = interpreter.cooperative_interpretation(sequence, input_tensor)

# Create visualization report
visualizer = BiologicalInsightsVisualizer()
report_dir = visualizer.create_comprehensive_report(insight)

print(f"Analysis complete! Check '{report_dir}' for detailed visualizations.")
```

#### Method 2: SAE-Based Feature Analysis

Complete workflow for SAE training and interpretation:

##### Step 1: Train SAE Model
```bash
# Navigate to the interpretation directory
cd GenomeBridge/genomeBridge/Interpretation/Attention_map

# Train the SAE model
python dna_training.py
```

##### Step 2: Downstream Evaluations with Ridge Regression

**A. First-token latent embedding analysis:**

Edit the following variables in `dna_sequence_analysis_firsttoken.py`:
```python
csv_path            = "<PATH_TO_YOUR_DATASET_CSV>"
sae_checkpoint_path = "<PATH_TO_YOUR_SAE_CHECKPOINT>"
output_path         = "<PATH_WHERE_YOU_WANT_FIRSTTOKEN_WEIGHTS_CSV>"
```

Then run:
```bash
python dna_sequence_analysis_firsttoken.py
```

**B. Mean-pooled latent embedding analysis:**

Edit the following variables in `dna_sequence_analysis.py`:
```python
csv_path            = "<PATH_TO_YOUR_DATASET_CSV>"
sae_checkpoint_path = "<PATH_TO_YOUR_SAE_CHECKPOINT>"
output_path         = "<PATH_WHERE_YOU_WANT_MEANPOOL_WEIGHTS_CSV>"
```

Then run:
```bash
python dna_sequence_analysis.py
```

### 🏃‍♂️ Running the Attention-Based Demo

To see the biological insights interpreter in action:

```bash
# Navigate to the interpretation directory
cd GenomeBridge/genomeBridge/Interpretation/Attention_map

# Run the demonstration script
python biological_insights_interpreter.py
```

This will:
1. Create mock genomic foundation models for demonstration
2. Analyze an example DNA sequence using cooperative interpretation
3. Generate comprehensive visualization reports including:
   - Attention analysis heatmaps
   - Feature importance visualizations
   - Summary reports with biological function predictions
   - Text-based interpretation reports

The demo generates output in the `biological_insights/` directory:
- `attention_analysis.png`: Multi-panel attention visualization
- `feature_importance.png`: Feature importance analysis
- `summary.png`: Comprehensive summary visualization
- `interpretation_report.txt`: Human-readable interpretation

### 📊 Interpretation Output Types

#### 1. Attention Analysis
- **Attention heatmaps**: Show which positions the model focuses on
- **Head-wise attention**: Distribution of attention across different attention heads
- **Position-wise attention**: Total attention received by each sequence position

#### 2. Feature Importance Analysis
- **Gradient-based scores**: Direct gradient importance for each position
- **Integrated gradients**: More stable importance attribution
- **Importance visualization**: Color-coded sequence representation

#### 3. SAE Feature Analysis
- **Latent activation patterns**: Visualization of which SAE features activate for different sequences
- **Feature importance rankings**: Quantitative ranking of features based on Ridge regression weights
- **Biological function correlation**: Mapping between SAE features and known biological functions

#### 4. Cooperative Insights
- **Ensemble predictions**: Weighted predictions from multiple models
- **Model agreement**: Consistency across different architectures
- **Comprehensive interpretation**: Human-readable biological insights

### 🔧 Advanced Usage and Customization

#### Individual Component Analysis

```python
# 1. Attention-only analysis
attention_viz = AttentionVisualizer(sequence_type="dna")
attention_fig = attention_viz.visualize_attention_heatmap(
    sequence=your_sequence,
    attention_weights=model_attention_weights,
    title="Promoter Region Attention Analysis"
)

# 2. Feature importance analysis
importance_analyzer = FeatureImportanceAnalyzer()

# Gradient-based importance
gradient_importance = importance_analyzer.compute_gradient_importance(
    model=your_model,
    input_tensor=input_tensor
)

# Integrated gradients for more robust importance
integrated_importance = importance_analyzer.compute_integrated_gradients(
    model=your_model,
    input_tensor=input_tensor,
    steps=50
)
```

#### Dynamic Model Weighting

```python
# Dynamic model weighting based on sequence characteristics
def dynamic_weighting(sequence):
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
    
    if gc_content > 0.6:  # High GC content
        return {'dnabert': 0.5, 'nucleotide_transformer': 0.3, 'hyenadna': 0.2}
    else:  # Normal/low GC content
        return {'dnabert': 0.3, 'nucleotide_transformer': 0.5, 'hyenadna': 0.2}

# Apply dynamic weighting
weights = dynamic_weighting(your_sequence)
interpreter.set_model_weights(weights)
```

### 📈 Performance Considerations

- **Memory Usage**: Both interpretation methods process sequences individually to manage memory efficiently
- **GPU Acceleration**: Automatically uses CUDA when available for faster inference
- **Batch Processing**: Supports batch analysis for large-scale genomic datasets
- **Caching**: Implements intelligent caching for repeated sequence analysis
- **SAE Training**: Requires substantial computational resources; consider using distributed training for large models

### 🔬 Research Applications

The unified interpretation framework is particularly useful for:

- **🧬 Promoter Analysis**: Understanding transcription factor binding sites through attention patterns and SAE features
- **🔬 Variant Effect Prediction**: Analyzing the impact of genetic variants using multiple interpretation methods
- **📊 Model Interpretability**: Comprehensive insights into what genomic foundation models learn at different levels
- **🎯 Feature Discovery**: Identifying important sequence motifs and patterns through complementary approaches
- **🤝 Model Comparison**: Comparing interpretation results across different model architectures and interpretation methods
- **🧪 Mechanistic Understanding**: Bridging between attention patterns and learned latent representations

### 🎯 Choosing the Right Interpretation Method

| Method | Best For | Advantages | Considerations |
|--------|----------|------------|----------------|
| **Attention-Based** | Real-time analysis, model comparison | Fast, intuitive visualization | Limited to attention-based models |
| **SAE-Based** | Deep feature analysis, mechanistic understanding | Interpretable features, quantitative evaluation | Requires additional training |
| **Cooperative** | Ensemble insights, robust interpretation | Multiple model perspectives, confidence estimation | Requires multiple trained models |

This comprehensive interpretation framework bridges the gap between powerful genomic foundation models and practical biological insights, making it easier for researchers to understand, validate, and trust AI-driven genomic analysis results.

## 🧬 Protein Structure-Guided Sequence Generation

GenomeBridge includes advanced protein sequence generation capabilities that combine genomic foundation models with structural constraints. This feature enables researchers to generate biologically realistic protein sequences while incorporating structural information through FoldMason integration.

### 🎯 Key Features

- **🔄 Structure-Aware Generation**: Generate protein sequences with structural constraints
- **🏗️ Multi-Model Support**: Compatible with Evo, Evo2, GenomeOcean, and GenSLM
- **📏 Length Control**: Flexible sequence length parameters for different applications
- **🎯 Genomic Context**: Generate sequences based on specific genomic coordinates
- **📊 Batch Processing**: Generate multiple variants for comprehensive analysis

### 🚀 Quick Start

Navigate to the protein generation directory:

```bash
cd GenomeBridge/genomeBridge/inference/protein_generation
```

### 🔬 Model-Specific Usage

#### 1. **Evo Model**

```bash
python autocomplete_structure_Evo.py \
    --gen_id NZ_JAYXHC010000003.1 \
    --start 157 \
    --end 1698 \
    --strand -1 \
    --prompt_start 0 \
    --prompt_end 600 \
    --structure_start 150 \
    --structure_end 500 \
    --num 100 \
    --min_seq_len 1000 \
    --max_seq_len 1200 \
    --foldmason_path /path/to/foldmason \
    --output_prefix outputs_Evo/gmp
```

#### 2. **Evo2 Model**

```bash
python autocomplete_structure_Evo2.py \
    --gen_id NZ_JAYXHC010000003.1 \
    --start 157 \
    --end 1698 \
    --strand -1 \
    --prompt_start 0 \
    --prompt_end 600 \
    --structure_start 150 \
    --structure_end 500 \
    --num 100 \
    --min_seq_len 1000 \
    --max_seq_len 1200 \
    --foldmason_path /path/to/foldmason \
    --output_prefix outputs_Evo2/gmp
```

#### 3. **GenomeOcean Model**

```bash
python autocomplete_structure_GO.py \
    --gen_id NZ_CP184171.1 \
    --start 1644 \
    --end 3236 \
    --strand -1 \
    --prompt_start 0 \
    --prompt_end 600 \
    --structure_start 150 \
    --structure_end 500 \
    --model_dir pGenomeOcean/GenomeOcean-4B \
    --num 100 \
    --min_seq_len 400 \
    --max_seq_len 500 \
    --foldmason_path /path/to/foldmason \
    --output_prefix outputs_GO2/gmp
```

#### 4. **GenSLM Model**

```bash
python autocomplete_structure_Genslm.py \
    --gen_id NZ_JAYXHC010000003.1 \
    --start 157 \
    --end 1698 \
    --strand -1 \
    --prompt_start 0 \
    --prompt_end 600 \
    --structure_start 150 \
    --structure_end 500 \
    --num 100 \
    --min_seq_len 350 \
    --max_seq_len 400 \
    --foldmason_path /path/to/foldmason \
    --output_prefix outputs_Genslm/gmp
