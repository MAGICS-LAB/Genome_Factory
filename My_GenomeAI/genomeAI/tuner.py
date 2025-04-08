import argparse
import yaml
import os
import subprocess
import sys


def run_train(config: dict):
    """
    Parse YAML (train_full.yaml / train_lora.yaml) and call Finetune.py for training.
    Handles both full and LoRA training, plus optional W&B usage, and custom saved model dir.
    """
    # Basic fields
    model_name_or_path = config.get("model", {}).get("model_name_or_path", "facebook/opt-125m")
    finetuning_type = config.get("method", {}).get("finetuning_type", "full").lower()
    use_lora = "False"
    if finetuning_type == "lora":
        use_lora = "True"

    # LoRA configs
    lora_r = config.get("method", {}).get("lora_r", 8)
    lora_alpha = config.get("method", {}).get("lora_alpha", 32)
    lora_dropout = config.get("method", {}).get("lora_dropout", 0.05)
    lora_target_modules = config.get("method", {}).get("lora_target", "Wqkv,dense,gated_layers,wo,classifier")

    # Dataset
    data_path = config.get("dataset", {}).get("data_path", "./dataset")
    kmer = config.get("dataset", {}).get("kmer", -1)

    # Training
    run_name = config.get("train", {}).get("run_name", "run")
    model_max_length = config.get("train", {}).get("model_max_length", 512)
    per_device_train_batch_size = config.get("train", {}).get("per_device_train_batch_size", 1)
    per_device_eval_batch_size = config.get("train", {}).get("per_device_eval_batch_size", 1)
    gradient_accumulation_steps = config.get("train", {}).get("gradient_accumulation_steps", 1)
    learning_rate = config.get("train", {}).get("learning_rate", 1e-4)
    num_train_epochs = config.get("train", {}).get("num_train_epochs", 1)
    lr_scheduler_type = config.get("train", {}).get("lr_scheduler_type", "cosine")
    warmup_ratio = config.get("train", {}).get("warmup_ratio", 0.1)
    classification = config.get("train", {}).get("classification", True)
    regression = config.get("train", {}).get("regression", False)
    bf16 = config.get("train", {}).get("bf16", False)
    fp16 = config.get("train", {}).get("fp16", False)
    ddp_timeout = config.get("train", {}).get("ddp_timeout", 180000000)
    logging_steps = config.get("train", {}).get("logging_steps", 100)
    save_steps = config.get("train", {}).get("save_steps", 100)
    evaluation_strategy = config.get("train", {}).get("evaluation_strategy", "steps")
    eval_steps = config.get("train", {}).get("eval_steps", 100)
    warmup_steps = config.get("train", {}).get("warmup_steps", 50)
    output_dir = config.get("output", {}).get("output_dir", "output")
    save_total_limit = config.get("train", {}).get("save_total_limit", 3)
    load_best_model_at_end = config.get("train", {}).get("load_best_model_at_end", True)
    overwrite_output_dir = config.get("output", {}).get("overwrite_output_dir", True)
    plot_loss = config.get("output", {}).get("plot_loss", False)

    # Optional custom saved model dir
    saved_model_dir = config.get("train", {}).get("saved_model_dir", None)

    # W&B usage
    use_wandb = config.get("train", {}).get("use_wandb", False)
    if not use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        if "WANDB_DISABLED" in os.environ:
            del os.environ["WANDB_DISABLED"]

    finetune_script = os.path.join(os.path.dirname(__file__), "Finetune.py")
    cmd = [
        "python", finetune_script,
        "--model_name", model_name_or_path,
        "--use_lora", use_lora,
        "--finetuning_type", finetuning_type,
        "--lora_r", str(lora_r),
        "--classification", str(classification),
        "--regression", str(regression),
        "--lora_alpha", str(lora_alpha),
        "--lora_dropout", str(lora_dropout),
        "--lora_target_modules", str(lora_target_modules),
        "--data_path", data_path,
        "--kmer", str(kmer),
        "--run_name", run_name,
        "--model_max_length", str(model_max_length),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--per_device_eval_batch_size", str(per_device_eval_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--learning_rate", str(learning_rate),
        "--num_train_epochs", str(num_train_epochs),
        "--lr_scheduler_type", lr_scheduler_type,
        "--warmup_ratio", str(warmup_ratio),
        "--bf16", str(bf16),
        "--fp16", str(fp16),
        "--ddp_timeout", str(ddp_timeout),
        "--logging_steps", str(logging_steps),
        "--save_steps", str(save_steps),
        "--evaluation_strategy", evaluation_strategy,
        "--eval_steps", str(eval_steps),
        "--warmup_steps", str(warmup_steps),
        "--output_dir", output_dir,
        "--save_total_limit", str(save_total_limit),
        "--load_best_model_at_end", str(load_best_model_at_end),
        "--overwrite_output_dir", str(overwrite_output_dir),
        "--plot_loss", str(plot_loss),
    ]

    if saved_model_dir:
        cmd += ["--saved_model_dir", saved_model_dir]

    print("[genomeai-cli] Running training command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_inference(config: dict):
    """
    Single script for inference: inference.py
    We read model_path from either config['inference']['model_path'] or fallback to config['model']['model_name_or_path'].
    We read dna from config['inference']['dna'], or fallback to a default.
    Then we call 'inference.py'.
    """
    num_labels = config.get("inference", {}).get("num_labels", 1)
    finetuning_type = config.get("method", {}).get("finetuning_type", "full").lower()
    classification = config.get("method", {}).get("classification", True)
    regression = config.get("method", {}).get("regression", False)
    inf_cfg = config.get("inference", {})
    model_path = inf_cfg.get("model_path")
    if not model_path:
        model_path = config.get("model", {}).get("model_name_or_path", "./Trained_model")

    dna = inf_cfg.get("dna", "ATTGGTGGAATGCACAGGATATTGTGAAGGAGTACAG...")
    if finetuning_type != "adapter":
        if classification:
            inference_script = os.path.join(os.path.dirname(__file__), "Inference.py")
        if regression:
            inference_script = os.path.join(os.path.dirname(__file__), "Inference_regression.py")
        cmd = [
        "python", inference_script,
        "--model_path", model_path,
        "--dna", dna
        ]
        print("[genomeai-cli] Running inference command:", " ".join(cmd))
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    else:
        if classification:
            inference_script = os.path.join(os.path.dirname(__file__), "Inference_adapter.py")
        if regression:
            inference_script = os.path.join(os.path.dirname(__file__), "Inference_adapter_regression.py")
        cmd = [
            "python", inference_script,
            "--model_path", model_path,
            "--dna", dna,
            "--num_labels", str(num_labels)
        ]
        print("[genomeai-cli] Running inference command:", " ".join(cmd))
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout


def run_webui():
    """
    A Gradio Web UI that includes a 'Download' tab with TWO separate mini-interfaces:
      - By species
      - By link
    Each is shown/hidden based on the radio selection.
    """
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is not installed. Please install it via pip install gradio.")
        sys.exit(1)

    ############################################################
    #                  Existing train / inference
    ############################################################
    def on_train_submit(
        classification,
        regression,
        model_name_or_path,
        finetuning_type,
        use_wandb,
        lora_r,
        lora_alpha,
        lora_dropout,
        lora_target,
        data_path,
        kmer,
        run_name,
        model_max_length,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        gradient_accumulation_steps,
        learning_rate,
        num_train_epochs,
        lr_scheduler_type,
        warmup_ratio,
        bf16,
        fp16,
        ddp_timeout,
        logging_steps,
        save_steps,
        evaluation_strategy,
        eval_steps,
        warmup_steps,
        output_dir,
        save_total_limit,
        load_best_model_at_end,
        overwrite_output_dir,
        plot_loss,
        saved_model_dir
    ):
        config = {
            "model": {
                "model_name_or_path": model_name_or_path
            },
            "method": {
                "finetuning_type": finetuning_type,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_target": lora_target,
            },
            "dataset": {
                "data_path": data_path,
                "kmer": kmer
            },
            "train": {
                "run_name": run_name,
                "classification": classification,
                "regression": regression,
                "model_max_length": model_max_length,
                "per_device_train_batch_size": per_device_train_batch_size,
                "per_device_eval_batch_size": per_device_eval_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "num_train_epochs": num_train_epochs,
                "lr_scheduler_type": lr_scheduler_type,
                "warmup_ratio": warmup_ratio,
                "bf16": bf16,
                "fp16": fp16,
                "ddp_timeout": ddp_timeout,
                "logging_steps": logging_steps,
                "save_steps": save_steps,
                "evaluation_strategy": evaluation_strategy,
                "eval_steps": eval_steps,
                "warmup_steps": warmup_steps,
                "save_total_limit": save_total_limit,
                "load_best_model_at_end": load_best_model_at_end,
                "use_wandb": use_wandb,
                "saved_model_dir": saved_model_dir
            },
            "output": {
                "output_dir": output_dir,
                "plot_loss": plot_loss,
                "overwrite_output_dir": overwrite_output_dir
            }
        }
        import subprocess

        try:
            output = run_train(config)
            return "Training finished successfully!"
        except subprocess.CalledProcessError as e:
            return f"Training failed: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    def run_train(inner_config):
        """
        Just calls the same code as run_train in the CLI scenario.
        But we can replicate or directly call the same function if we like.
        For brevity, we'll call run_train from this file or replicate the logic.
        We'll just replicate the command approach, or do the steps in-memory.
        For clarity, let's replicate the command approach:
        """
        import subprocess
        import os

        # We basically do the same as run_train(config) from above,
        # but in code. We'll do a quick approach: create a temp file, then pass it.
        import tempfile, json

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tf:
            yaml_path = tf.name
            import yaml
            yaml.dump(inner_config, tf)
        cmd = f"genomeai-cli train {yaml_path}"
        try:
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
            os.remove(yaml_path)
            return out
        except subprocess.CalledProcessError as e:
            os.remove(yaml_path)
            raise e


    def on_inference_submit(model_path, dna, finetuning_type,classification, regression,num_labels):
        config = {
            "method": {
                "finetuning_type": finetuning_type,
                "classification": classification,
                "regression": regression
            },
            "inference": {
                "num_labels": num_labels,
                "model_path": model_path,
                "dna": dna
            } 
        }
        import subprocess, tempfile, json
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tf:
            cfg_path = tf.name
            import yaml
            yaml.dump(config, tf)
        cmd = f"genomeai-cli inference {cfg_path}"
        try:
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
            os.remove(cfg_path)
            return f"Inference finished successfully!\n\nOutput:\n{out}"
        except subprocess.CalledProcessError as e:
            os.remove(cfg_path)
            return f"Inference failed: {e}\n\nOutput:\n{e.output}"
        except Exception as e:
            os.remove(cfg_path)
            return f"Error: {str(e)}"

    ############################################################
    #                  Download Logic
    ############################################################

    def on_download_species(species, folder):
        """
        Invokes the species-based approach:
          genomeai-cli download (with species in config).
        """
        import subprocess, tempfile
        import yaml

        cfg = {
            "download": {
                "species": species,
                "download_folder": folder if folder.strip() else None
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tf:
            path = tf.name
            yaml.dump(cfg, tf)

        cmd = f"genomeai-cli download {path}"
        try:
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
            os.remove(path)
            return f"Species-based download completed!\n\n{out}"
        except subprocess.CalledProcessError as e:
            os.remove(path)
            return f"Species-based download failed: {e}\n\nOutput:\n{e.output}"

    def on_download_link(link, folder):
        """
        Invokes direct link approach from CLI.
        """
        import subprocess, tempfile
        import yaml

        cfg = {
            "download": {
                "link": link,
                "download_folder": folder if folder.strip() else None
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tf:
            path = tf.name
            yaml.dump(cfg, tf)

        cmd = f"genomeai-cli download {path}"
        try:
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
            os.remove(path)
            return f"Link-based download completed!\n\n{out}"
        except subprocess.CalledProcessError as e:
            os.remove(path)
            return f"Link-based download failed: {e}\n\nOutput:\n{e.output}"

    # We'll do some logic to hide / show the species UI vs link UI
    import gradio as gr
    import json

    dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datasets_species_taxonid_dict.json")
    with open(dict_path, "r", encoding="utf-8") as f:
        species_dict = json.load(f)
    all_species = sorted(list(species_dict.keys()))

    def switch_mode(mode):
        """
        Return instructions to show/hide each panel depending on selection.
        """
        if mode == "By species":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    with gr.Blocks() as demo:
        gr.Markdown("# GenomeAI Web UI")

        ################################################
        # Train Tab
        ################################################
        with gr.Tab("Train"):
            gr.Markdown("## Training Parameters")
            model_name_or_path = gr.Textbox(value="zhihan1996/DNABERT-2-117M", label="Model Name or Path")
            finetuning_type = gr.Radio(choices=["full", "lora","adapter"], value="full", label="Finetuning Type")
            use_wandb = gr.Checkbox(value=False, label="Use Weights & Biases?")
            
            with gr.Group(visible=False) as lora_group:
                lora_r = gr.Number(value=8, label="LoRA r")
                lora_alpha = gr.Number(value=32, label="LoRA alpha")
                lora_dropout = gr.Number(value=0.05, label="LoRA dropout")
                lora_target = gr.Textbox(value="Wqkv,dense,gated_layers,wo,classifier", label="LoRA target modules")

            def update_lora_visibility(ft_type):
                return gr.update(visible=(ft_type == "lora"))
            
            finetuning_type.change(
                fn=update_lora_visibility,
                inputs=[finetuning_type],
                outputs=[lora_group]
            )

            classification = gr.Checkbox(value=True, label="Classification")
            regression = gr.Checkbox(value=False, label="Regression")
            data_path = gr.Textbox(value="./dataset", label="Data Path")
            kmer = gr.Number(value=-1, label="K-mer")
            run_name = gr.Textbox(value="run", label="Run Name")
            model_max_length = gr.Number(value=512, label="Model Max Length")
            per_device_train_batch_size = gr.Number(value=1, label="Per device Train Batch Size")
            per_device_eval_batch_size = gr.Number(value=1, label="Per device Eval Batch Size")
            gradient_accumulation_steps = gr.Number(value=1, label="Gradient Accum Steps")
            learning_rate = gr.Number(value=1e-4, label="Learning Rate")
            num_train_epochs = gr.Number(value=1, label="Num Train Epochs")
            lr_scheduler_type = gr.Textbox(value="cosine", label="LR Scheduler Type")
            warmup_ratio = gr.Number(value=0.1, label="Warmup Ratio")
            bf16 = gr.Checkbox(value=False, label="bf16")
            fp16 = gr.Checkbox(value=False, label="fp16")
            ddp_timeout = gr.Number(value=180000000, label="DDP Timeout")
            logging_steps = gr.Number(value=100, label="Logging Steps")
            save_steps = gr.Number(value=100, label="Save Steps")
            evaluation_strategy = gr.Textbox(value="steps", label="Evaluation Strategy")
            eval_steps = gr.Number(value=100, label="Eval Steps")
            warmup_steps = gr.Number(value=50, label="Warmup Steps")
            output_dir = gr.Textbox(value="output", label="Output Dir")
            save_total_limit = gr.Number(value=3, label="Save Total Limit")
            load_best_model_at_end = gr.Checkbox(value=True, label="Load Best Model at End")
            overwrite_output_dir = gr.Checkbox(value=True, label="Overwrite Output Dir")
            plot_loss = gr.Checkbox(value=False, label="Plot Loss")

            saved_model_dir = gr.Textbox(value="", label="Saved Model Dir (optional)")

            train_button = gr.Button("Start Training")
            train_output = gr.Textbox(label="Training Output")

            train_button.click(
                fn=on_train_submit,
                inputs=[
                    classification,
                    regression,
                    model_name_or_path,
                    finetuning_type,
                    use_wandb,
                    lora_r,
                    lora_alpha,
                    lora_dropout,
                    lora_target,
                    data_path,
                    kmer,
                    run_name,
                    model_max_length,
                    per_device_train_batch_size,
                    per_device_eval_batch_size,
                    gradient_accumulation_steps,
                    learning_rate,
                    num_train_epochs,
                    lr_scheduler_type,
                    warmup_ratio,
                    bf16,
                    fp16,
                    ddp_timeout,
                    logging_steps,
                    save_steps,
                    evaluation_strategy,
                    eval_steps,
                    warmup_steps,
                    output_dir,
                    save_total_limit,
                    load_best_model_at_end,
                    overwrite_output_dir,
                    plot_loss,
                    saved_model_dir
                ],
                outputs=train_output
            )

        ################################################
        # Inference Tab
        ################################################
        with gr.Tab("Inference"):
            gr.Markdown("## Inference Parameters")
            finetuning_type = gr.Radio(choices=["full", "lora","adapter"], value="full", label="Finetuning Type")
            classification = gr.Checkbox(value=True, label="Classification")
            regression = gr.Checkbox(value=False, label="Regression")
            num_labels = gr.Number(value=1, label="Number of Labels")
            model_path = gr.Textbox(value="./Trained_model", label="Model Path")
            dna = gr.Textbox(value="ATTGGTGGAATGCACAGGATATTGTGAAGGAGTACAG...", label="DNA Sequence")
            inf_button = gr.Button("Start Inference")
            inf_output = gr.Textbox(label="Inference Output")

            inf_button.click(
                fn=on_inference_submit,
                inputs=[model_path, dna, finetuning_type, classification, regression,num_labels],
                outputs=inf_output
            )

        ################################################
        # Download Tab
        ################################################
        with gr.Tab("Download"):
            gr.Markdown("## Download a Genome")

            # A radio to choose approach
            dl_mode = gr.Radio(choices=["By species", "By link"], value="By species", label="Select Download Mode")

            # "By species" UI
            with gr.Group(visible=True) as species_panel:
                species_dropdown = gr.Dropdown(choices=all_species, value="Homo sapiens", label="Species")
                folder_sp = gr.Textbox(value="", label="Download Folder (optional)")
            
            # "By link" UI
            with gr.Group(visible=False) as link_panel:
                link_text = gr.Textbox(value="", label="Direct Link (.fna.gz, etc)")
                folder_link = gr.Textbox(value="", label="Download Folder (optional)")

            dl_button = gr.Button("Download")
            dl_output = gr.Textbox(label="Download Output")

            def switch_mode(mode):
                """Hide or show species_panel vs. link_panel"""
                if mode == "By species":
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)

            dl_mode.change(
                fn=switch_mode,
                inputs=[dl_mode],
                outputs=[species_panel, link_panel]
            )

            # The main download function 
            def dl_unified_fn(mode, sp, folder_s, lk, folder_l):
                if mode == "By species":
                    return on_download_species(sp, folder_s)
                else:
                    return on_download_link(lk, folder_l)

            dl_button.click(
                fn=dl_unified_fn,
                inputs=[dl_mode, species_dropdown, folder_sp, link_text, folder_link],
                outputs=dl_output
            )

    demo.launch()


def run_download(config_path: str):
    """
    Minimal new function to handle 'download' subcommand from CLI.
    If config has:
      - download.species => do species approach
      - download.link => do link approach
    If both exist, prefer species
    If neither, interactive prompt
    """
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    import yaml
    import json
    import requests, shutil, gzip

    from genomeAI.GenomeDataset import GenomeDataset  # for species approach

    species = None
    link = None
    download_folder = None

    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        species = cfg.get("download", {}).get("species", None)
        link = cfg.get("download", {}).get("link", None)
        download_folder = cfg.get("download", {}).get("download_folder", None)

    if species and link:
        print("Detected both species and link in config. We'll prefer species approach.")
        link = None

    if not species and not link:
        # interactive approach
        print("Download modes:\n1) By species\n2) By link")
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            # species approach
            dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datasets_species_taxonid_dict.json")
            with open(dict_path, "r", encoding="utf-8") as fp:
                species_dict = json.load(fp)
            all_species = sorted(list(species_dict.keys()))
            print("Available species from Datasets_species_taxonid_dict.json:")
            for i, sp in enumerate(all_species):
                print(f"{i+1}. {sp}")
            sp_choice = input("Enter the number or the exact species name: ")
            if sp_choice.isdigit():
                idx = int(sp_choice) - 1
                if 0 <= idx < len(all_species):
                    species = all_species[idx]
                else:
                    print("Invalid choice.")
                    sys.exit(1)
            else:
                if sp_choice in all_species:
                    species = sp_choice
                else:
                    print("Invalid choice.")
                    sys.exit(1)
            folder_choice = input("Enter download folder path (leave empty for default): ")
            if folder_choice.strip():
                download_folder = folder_choice

        elif choice == "2":
            link = input("Paste direct link to .fna(.gz): ").strip()
            folder_c = input("Enter download folder (leave empty for default): ")
            if folder_c.strip():
                download_folder = folder_c
        else:
            print("Invalid choice.")
            sys.exit(1)

    # Actually download
    if species:
        print(f"Downloading by species: {species}")
        if download_folder:
            print(f"Download folder: {download_folder}")
        else:
            print(f"Default folder: ./{species.replace(' ', '_')}")
        try:
            GenomeDataset(species=species, download_folder=download_folder, download=True)
            print("Species-based download completed.")
        except Exception as e:
            print("Error during species-based download:", e)
    elif link:
        if not download_folder:
            download_folder = "./downloaded_genome"
        os.makedirs(download_folder, exist_ok=True)
        filename = link.split("/")[-1]
        local_path = os.path.join(download_folder, filename)

        import requests, shutil, gzip
        print(f"Downloading link: {link}\nStoring to: {local_path}")
        try:
            with requests.get(link, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            if filename.endswith(".gz"):
                # decompress
                decompressed = local_path[:-3]  # remove .gz
                print(f"Decompressing {local_path} -> {decompressed}")
                with gzip.open(local_path, "rb") as f_in, open(decompressed, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(local_path)  # remove original .gz
                print(f"Link-based download + decompress completed => {decompressed}")
            else:
                print(f"Link-based download completed => {local_path}")
        except Exception as e:
            print("Error during link-based download:", e)



