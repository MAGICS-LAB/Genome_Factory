# GenomeBridge

GenomeBridge is a Python-based integrated library for tuning and deploying genomic foundation models (GFMs). It simplifies data collection (including downloading DNA sequences from NCBI and preprocessing), offers advanced and efficient tuning methods (full tuning, LoRA, and adapter tuning), and supports inference (embedding extraction and DNA sequence generation) and benchmarking of various GFMs.

## Supported Models

- **DNABERT-2**: 117M
- **Hyenadna**: 1K / 16K / 32K / 160K / 450K / 1M
- **Nucleotide Transformer**: 50M / 100M / 250M / 500M / 1B / 2.5B
- **Caduceus**: 1K / 131K
- **GenomeOcean**: 100M / 500M / 4B
- **EVO**: 8K/131K

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/WeiminWu2000/GenomeBridge.git
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

Access all GenomeAI functionalities through a graphical interface:

```bash
genomebridge-cli webui
```

This command launches a web server. Open the provided URL in your browser to use the WebUI.
