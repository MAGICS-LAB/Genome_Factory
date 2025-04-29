# GenomeAI

GenomeAI provides a unified interface to leverage state-of-the-art Genome Foundation Models (GFMs) for various bioinformatics tasks, including fine-tuning, inference,genomic data generation, data downloading, and processing.

## Supported Models

- **DNABERT-2**: 117M
- **Hyenadna**: 1K / 16K / 32K / 160K / 450K / 1M
- **Nucleotide Transformer**: 50M / 100M / 250M / 500M / 1B / 2.5B
- **Caduceus**: 1K / 131K
- **Genome Ocean**: 100M / 500M / 4B

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/WeiminWu2000/GenomeAI.git
    cd GenomeAI
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

3.  **Environment Notes:**
    *   For **Genome Ocean**, use `transformers==4.44.2`.
    *   For **other models**, use `transformers==4.29.2`.
    *   For **DNABERT-2**, ensure `triton` is uninstalled: `pip uninstall triton`.

## Usage via CLI (`genomeai-cli`)

GenomeAI uses YAML configuration files to define tasks. Example files are provided in `genomeAI/examples/`. You can customize the parameters within these files, ensuring you maintain the required YAML structure.

### Training

For fine-tuning GFMs, GenomeAI supports two primary task types: **classification** and **regression**. You specify the desired `task_type` in the training YAML configuration file.

Fine-tune GFMs using different methods:

*   **Full Fine-tuning:**
    ```bash
    genomeai-cli train genomeAI/examples/train_full.yaml
    ```
*   **LoRA (Low-Rank Adaptation):**
    ```bash
    genomeai-cli train genomeAI/examples/train_lora.yaml
    ```
    *   Specify target modules in the YAML file:
        *   `all`: Targets all linear layers.
        *   `all_in_and_out_proj`: Targets input/output projection layers and the final classification layer.
        *   *Custom*: Specify module names directly.
*   **Adapter:**
    ```bash
    genomeai-cli train genomeAI/examples/train_adapter.yaml
    ```
    *   Customize the adapter architecture in `genomeAI/Train/workflow/adapter/adapter_model/Adapter.py` for potentially better performance on specific downstream tasks.

    *Note:* Training settings like batch size, learning rate, and epochs can be customized in the respective YAML files for all methods.

    **Note on Flash Attention:** To enable Flash Attention, set the `flash_attention` argument to `true` in your YAML configuration file. You must also enable mixed-precision training by setting either `bf16: true` or `fp16: true`. If `flash_attention` is set to `false`, the model's default attention mechanism will be used.

### Inference

Use trained models for prediction or generation:

1.  **Prediction:** (Predict properties of DNA sequences). Ensure the `task_type` specified in your inference YAML file (`classification` or `regression`) matches the task the model was originally fine-tuned for.
    *   **Full:**
        ```bash
        genomeai-cli inference genomeAI/examples/inference_full.yaml
        ```
    *   **LoRA:**
        ```bash
        genomeai-cli inference genomeAI/examples/inference_lora.yaml
        ```
    *   **Adapter:**
        ```bash
        genomeai-cli inference genomeAI/examples/inference_adapter.yaml
        ```
        *   *Note:* For Adapter-based classification, specify the number of labels (`num_label`) in the YAML. For regression, set `num_label: 1`. Full/LoRA methods infer this automatically.

2.  **Generation:** (Generate new DNA sequences based on existing ones)
    *   Applicable to compatible GFMs.
    ```bash
    genomeai-cli inference genomeAI/examples/inference_generation.yaml
    ```

### Data Download

Download genomic data from NCBI:

1.  **Using a config file:** Specify download parameters in a YAML file.
   
    **Download by species:**
    ```bash
    genomeai-cli download genomeAI/examples/download_by_species.yaml
    ```
    **Download by Link:**
    ```bash
    genomeai-cli download genomeAI/examples/download_by_link.yaml
    ```
3.  **Interactively:** Run the command without a config file and follow the prompts in the terminal to specify your download criteria (supports both species-based and link-based downloads).
    ```bash
    genomeai-cli download
    ```
### Data Processing

GenomeAI provides tools to prepare data for model fine-tuning. This includes processing data downloaded from NCBI or formatting your own custom datasets.

**1. Processing NCBI Data:**

*   Gather data downloaded using the `download` command into a single folder.
*   Run the processing command with a config file:
    ```bash
    genomeai-cli process genomeAI/examples/process_data.yaml
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

## Usage via Web UI

Access all GenomeAI functionalities through a graphical interface:

```bash
genomeai-cli webui
```

This command launches a web server. Open the provided URL in your browser to use the WebUI.
