# Genome-Factory: An Integrated Platform for Tuning and Deploying Genomic Foundation Models

![overview](https://github.com/user-attachments/assets/5999aac8-3e01-4b54-ba3b-2239753c4dd8)


Genome-Factory is a Python-based integrated library for tuning and deploying genomic foundation models (GFMs). It simplifies data collection (including downloading DNA sequences from NCBI and preprocessing), offers advanced and efficient tuning methods (full tuning, LoRA, and adapter tuning), and supports inference (embedding extraction and DNA sequence generation) and benchmarking of various GFMs.

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
    git clone https://github.com/xxx/Genome-Factory.git
    cd Genome-Factory
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
    # IMPORTANT: Return to the Genome-Factory root directory before the next step

    # Install Genome-Factory in editable mode
    pip install -e .
    ```

3.  **Environment Notes:**
    *   For **Genome Ocean**, use `transformers==4.44.2`.
    *   For **other models**, use `transformers==4.29.2`.
    *   For **DNABERT-2**, ensure `triton` is uninstalled: `pip uninstall triton`.

## Usage via CLI (`genomefactory-cli`)

Genome-Factory uses YAML configuration files to define tasks. Example files are provided in `genomeFactory/Examples/`. You can customize the parameters within these files, ensuring you maintain the required YAML structure.

### Data Download

Download genomic data from NCBI:

1.  **Using a config file:** Specify download parameters in a YAML file.
    **Download by species:**
    ```bash
    genomefactory-cli download genomeFactory/Examples/download_by_species.yaml
    ```
    **Download by Link:**
    ```bash
    genomefactory-cli download genomeFactory/Examples/download_by_link.yaml
    ```
2.  **Interactively:** Run the command without a config file and follow the prompts in the terminal to specify your download criteria (supports both species-based and link-based downloads).
    ```bash
    genomefactory-cli download
    ```
*Note:* The list of species and their taxon IDs used for downloads is stored in `genomeFactory/Data/Download/Datasets_species_taxonid_dict.json`. This file is not exhaustive; you can extend it by adding new species-to-taxonID pairs to download data for other species as needed.

### Data Processing

Genome-Factory provides tools to prepare data for model fine-tuning. This includes processing data downloaded from NCBI or formatting your own custom datasets.

**1. Processing NCBI Data:**

*   Gather data downloaded using the `download` command into a single folder.
*   Run the processing command with a config file:
    ```bash
    genomefactory-cli process genomeFactory/Examples/process_normal.yaml
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

For fine-tuning GFMs, Genome-Factory supports two primary task types: **classification** and **regression**. You specify the desired `task_type` in the training YAML configuration file.

Fine-tune GFMs using different methods:

*   **Full Fine-tuning:**
    ```bash
    genomefactory-cli train genomeFactory/Examples/train_full.yaml
    ```
*   **LoRA (Low-Rank Adaptation):**
    ```bash
    genomefactory-cli train genomeFactory/Examples/train_lora.yaml
    ```
    *   Specify target modules in the YAML file:
        *   `all`: Targets all linear layers.
        *   `all_in_and_out_proj`: Targets input/output projection layers and the final classification layer.
        *   *Custom*: Specify module names directly.
    *   For **Evo**:
        ```bash
        genomefactory-cli train genomeFactory/Examples/train_evo_lora.yaml
        ```
*   **Adapter:**
    ```bash
    genomefactory-cli train genomeFactory/Examples/train_adapter.yaml
    ```
    *   Customize the adapter architecture in `genomeFactory/Train/workflow/adapter/adapter_model/Adapter.py` for potentially better performance on specific downstream tasks.

    *Note:* Training settings like batch size, learning rate, and epochs can be customized in the respective YAML files for all methods.

    **Note on Flash Attention:** To enable Flash Attention, set the `flash_attention` argument to `true` in your YAML configuration file. You must also enable mixed-precision training by setting either `bf16: true` or `fp16: true`. If `flash_attention` is set to `false`, or if a specific GFM does not support this argument, the model's default attention mechanism will be used.

    **Benchmarking:** After fine-tuning, performance metrics are saved to a JSON file. You can use these metrics for benchmarking (e.g., comparing the performance of different models or tuning methods on specific tasks).

### Inference

Use trained models for prediction, generation, or embedding extraction:

1.  **Prediction:** (Predict properties of DNA sequences). Ensure the `task_type` specified in your inference YAML file (`classification` or `regression`) matches the task the model was originally fine-tuned for.
    *   **Full:**
        ```bash
        genomefactory-cli inference genomeFactory/Examples/inference_full.yaml
        ```
    *   **LoRA:**
        ```bash
        genomefactory-cli inference genomeFactory/Examples/inference_lora.yaml
        ```
    *   **Adapter:**
        ```bash
        genomefactory-cli inference genomeFactory/Examples/inference_adapter.yaml
        ```
        *   *Note:* For Adapter-based classification, specify the number of labels (`num_label`) in the YAML. For regression, set `num_label: 1`. Full/LoRA methods infer this automatically.

2.  **Generation:** (Generate new DNA sequences based on existing ones). Applicable to compatible GFMs.
    *   For **GenomeOcean**:
        ```bash
        genomefactory-cli inference genomeFactory/Examples/inference_generation_genomeocean.yaml
        ```
    *   For **Evo**:
        ```bash
        genomefactory-cli inference genomeFactory/Examples/inference_generation_evo.yaml
        ```

3.  **Embedding Extraction:** (Extract the last hidden state embeddings from sequences).
    *   General Case:
        ```bash
        genomefactory-cli inference genomeFactory/Examples/inference_extract.yaml
        ```
    *   For **Evo** specifically:
        ```bash
        genomefactory-cli inference genomeFactory/Examples/inference_extract_evo.yaml
        ```


## Usage via Web UI

Access all Genome-Factory functionalities through a graphical interface:

```bash
genomefactory-cli webui
```

This command launches a web server. Open the provided URL in your browser to use the WebUI.

## Reference

```
LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models.
Zheng, Yaowei, Richong Zhang, Junhao Zhang, YeYanhan YeYanhan, and Zheyan Luo.
In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 400-410. 2024.
```

---

## 🧬 **Advanced Features in Process Data** ⚙️

---

Genome-Factory provides specialized dataset generation tools for common genomic machine learning tasks:

**Promoter Region Dataset**: Generate promoter vs. non-promoter classification data from EPDnew database (hg38, mm10, danRer11)
```bash
genomefactory-cli process genomeFactory/Examples/process_promoter.yaml
```

**Epigenetic Mark Dataset**: Create gene body sequences with H3K36me3 signal classification from ENCODE/Roadmap data (hg38, mm10)
```bash
genomefactory-cli process genomeFactory/Examples/process_emp.yaml
```

**Enhancer Region Dataset**: Build enhancer vs. non-enhancer classification data from FANTOM5 annotations (hg38, mm10)
```bash
genomefactory-cli process genomeFactory/Examples/process_enhancer.yaml
```

All datasets feature quality control, configurable train/val/test splits, and output CSV files with `sequence,label` format.

## 🔬 Model Interpretation and Analysis

Genome-Factory provides comprehensive tools for understanding and interpreting genomic foundation models through sparse autoencoder (SAE) interpretation to provide deep insights into model behavior and biological significance.

### 🎯 Key Interpretation Methods

#### 🔬 Sparse Autoencoder (SAE) Analysis
- **🧬 Latent Feature Discovery**: Identify interpretable features learned by genomic foundation models
- **📈 Ridge Regression Evaluation**: Quantitative assessment of feature importance for downstream tasks
- **🎯 First-token vs Mean-pooled Analysis**: Compare different pooling strategies for sequence representation
- **📊 Feature Weight Analysis**: Understand which SAE features contribute most to biological predictions

### 🚀 Quick Start Guide

#### SAE-Based Feature Analysis

Complete workflow for SAE training and interpretation:

##### Step 1: Train SAE Model

```bash
genomefactory-cli sae_train genomeFactory/Examples/sae_train.yaml
```

Configure the following parameters in the YAML file:

```yaml
data_file: "<YOUR_SEQUENCE_FILE>"
d_model: <MODEL_DIMENSION>
d_hidden: <HIDDEN_DIMENSION>
batch_size: <BATCH_SIZE>
lr: <LEARNING_RATE>
k: <K_VALUE>
auxk: <AUXK_VALUE>
dead_steps_threshold: <THRESHOLD_STEPS>
max_epochs: <MAX_EPOCHS>
num_devices: <NUM_DEVICES>
model_suffix: "<MODEL_SUFFIX>"
wandb_project: "<PROJECT_NAME>"
num_workers: <NUM_WORKERS>
model_name: "<MODEL_NAME>"
```

##### Step 2: Downstream Evaluations with Ridge Regression

**A. First-token latent embedding analysis:**

```bash
genomefactory-cli sae_train genomeFactory/Examples/sae_regression.yaml
```

Configure the following parameters in the YAML file:

```yaml
csv_path: "<FEATURE_CSV_PATH>"
sae_checkpoint_path: "<SAE_CHECKPOINT_PATH>"
output_path: "<OUTPUT_CSV_PATH>"
type: "first_token"
```

**B. Mean-pooled latent embedding analysis:**

```bash
genomefactory-cli sae_train genomeFactory/Examples/sae_regression.yaml
```

Configure the following parameters in the YAML file:

```yaml
csv_path: "<FEATURE_CSV_PATH>"
sae_checkpoint_path: "<SAE_CHECKPOINT_PATH>"
output_path: "<OUTPUT_CSV_PATH>"
type: "mean"
```

### 📊 Interpretation Output Types

#### SAE Feature Analysis
- **Latent activation patterns**: Visualization of which SAE features activate for different sequences
- **Feature importance rankings**: Quantitative ranking of features based on Ridge regression weights
- **Biological function correlation**: Mapping between SAE features and known biological functions

### 📈 Performance Considerations

- **Memory Usage**: SAE interpretation methods process sequences individually to manage memory efficiently
- **GPU Acceleration**: Automatically uses CUDA when available for faster inference
- **Batch Processing**: Supports batch analysis for large-scale genomic datasets
- **Caching**: Implements intelligent caching for repeated sequence analysis
- **SAE Training**: Requires substantial computational resources; consider using distributed training for large models

### 🔬 Research Applications

The SAE interpretation framework is particularly useful for:

- **🧬 Promoter Analysis**: Understanding transcription factor binding sites through SAE features
- **🔬 Variant Effect Prediction**: Analyzing the impact of genetic variants using SAE interpretation methods
- **📊 Model Interpretability**: Comprehensive insights into what genomic foundation models learn at different levels
- **🎯 Feature Discovery**: Identifying important sequence motifs and patterns through SAE approaches
- **🤝 Model Comparison**: Comparing interpretation results across different model architectures
- **🧪 Mechanistic Understanding**: Understanding learned latent representations

### 🎯 SAE-Based Interpretation Method

**SAE-Based interpretation** is ideal for:
- **Deep feature analysis and mechanistic understanding**
- **Advantages**: Interpretable features, quantitative evaluation
- **Considerations**: Requires additional training

This comprehensive interpretation framework bridges the gap between powerful genomic foundation models and practical biological insights, making it easier for researchers to understand, validate, and trust AI-driven genomic analysis results.
