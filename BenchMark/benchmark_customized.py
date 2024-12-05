import os
import zipfile
import gdown
import json
import subprocess
import signal

def run_command(cmd):
    try:
        # Start the subprocess in a new session
        process = subprocess.Popen(
            cmd,
            shell=True,
            preexec_fn=os.setsid,
            executable='/bin/bash'
        )
        process.communicate()
    except KeyboardInterrupt:
        # Send SIGTERM to the entire process group
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.terminate()
        process.wait()
        raise

def benchmark_model(model, benchmark, per_device_train_batch_size_dict, gradient_accumulation_steps_dict):
    # To replicate the results in the paper, ensure that per_device_train_batch_size * gradient_accumulation_steps * num_gpu = 32
    
    if benchmark == "GUE":
        # Define download link and output filename
        url = 'https://drive.google.com/uc?id=1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2'
        output = 'GUE_data.zip'

        # Download the file
        gdown.download(url, output, quiet=False)

        # Extract the file to the current directory
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('.')

        # Get the data path (current directory)
        data_path = os.path.abspath('.')

        lr = 3e-5
        vocab=""
        seed = 42

        experiments = {
            "EMP": ['H3', 'H3K14ac', 'H3K36me3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K79me3', 'H3K9ac', 'H4', 'H4ac'],
            "prom_core": ['prom_core_all', 'prom_core_notata'],
            "prom_core_tata": ['prom_core_tata'],
            "prom_300": ['prom_300_all', 'prom_300_notata'],
            "prom_300_tata": ['prom_300_tata'],
            "splice": ['reconstructed'],
            "virus": ['covid'],
            "mouse": ['0', '1', '2', '3', '4'],
            "tf": ['0', '1', '2', '3', '4']
        }

        experiment_subdirs = {
            "EMP": "EMP",
            "prom_core": "prom",
            "prom_core_tata": "prom",
            "prom_300": "prom",
            "prom_300_tata": "prom",
            "splice": "splice",
            "virus": "virus",
            "mouse": "mouse",
            "tf": "tf"
        }

        if model == "DNABERT2":
            # Model-specific configurations for DNABERT2
            model_name_or_path = "zhihan1996/DNABERT-2-117M"
            train_script = "trainDNABERT2.py"
            output_dir = "output_GUE/dnabert2"

            for experiment_name, datasets in experiments.items():
                # Get the corresponding per_device_train_batch_size and gradient_accumulation_steps
                per_device_train_batch_size = per_device_train_batch_size_dict.get(experiment_name, 8)
                gradient_accumulation_steps = gradient_accumulation_steps_dict.get(experiment_name, 1)

                for data in datasets:
                    # Set specific parameters based on the experiment
                    if experiment_name == "EMP":
                        model_max_length = 128
                        per_device_eval_batch_size = 16
                        num_train_epochs = 3
                        save_steps = 200
                        eval_steps = 200
                        warmup_steps = 50
                    elif experiment_name == "prom_core":
                        model_max_length = 20
                        per_device_eval_batch_size = 16
                        num_train_epochs = 4
                        save_steps = 400
                        eval_steps = 400
                        warmup_steps = 50
                    elif experiment_name == "prom_core_tata":
                        model_max_length = 20
                        per_device_eval_batch_size = 16
                        num_train_epochs = 10
                        save_steps = 200
                        eval_steps = 200
                        warmup_steps = 50
                    elif experiment_name == "prom_300":
                        model_max_length = 70
                        per_device_eval_batch_size = 16
                        num_train_epochs = 4
                        save_steps = 400
                        eval_steps = 400
                        warmup_steps = 50
                    elif experiment_name == "prom_300_tata":
                        model_max_length = 70
                        per_device_eval_batch_size = 16
                        num_train_epochs = 10
                        save_steps = 200
                        eval_steps = 200
                        warmup_steps = 50
                    elif experiment_name == "splice":
                        model_max_length = 80
                        per_device_eval_batch_size = 16
                        num_train_epochs = 5
                        save_steps = 200
                        eval_steps = 200
                        warmup_steps = 50
                    elif experiment_name == "virus":
                        model_max_length = 256
                        per_device_eval_batch_size = 32
                        num_train_epochs = 8
                        save_steps = 200
                        eval_steps = 200
                        warmup_steps = 50
                    elif experiment_name == "mouse":
                        model_max_length = 30
                        per_device_eval_batch_size = 64
                        num_train_epochs = 5
                        save_steps = 200
                        eval_steps = 200
                        warmup_steps = 30
                        max_steps = 1000  # Specific to mouse data
                    elif experiment_name == "tf":
                        model_max_length = 30
                        per_device_eval_batch_size = 64
                        num_train_epochs = 3
                        save_steps = 200
                        eval_steps = 200
                        warmup_steps = 30
                    else:
                        # Default parameters
                        model_max_length = 128
                        per_device_eval_batch_size = 16
                        num_train_epochs = 3
                        save_steps = 200
                        eval_steps = 200
                        warmup_steps = 50

                    data_subdir = experiment_subdirs.get(experiment_name, "")
                    data_full_path = f"{data_path}/GUE/{data_subdir}/{data}"

                    # Build the command
                    cmd = f"""
                    python {train_script} \\
                        --model_name_or_path {model_name_or_path} \\
                        --data_path {data_full_path} \\
                        --kmer -1 \\
                        --run_name {model}_{vocab}_{lr}_{experiment_name}_{data}_seed{seed} \\
                        --model_max_length {model_max_length} \\
                        --per_device_train_batch_size {per_device_train_batch_size} \\
                        --per_device_eval_batch_size {per_device_eval_batch_size} \\
                        --gradient_accumulation_steps {gradient_accumulation_steps} \\
                        --learning_rate {lr} \\
                        --num_train_epochs {num_train_epochs} \\
                        --fp16 \\
                        --save_steps {save_steps} \\
                        --output_dir {output_dir} \\
                        --evaluation_strategy steps \\
                        --eval_steps {eval_steps} \\
                        --warmup_steps {warmup_steps} \\
                        --logging_steps 100000 \\
                        --overwrite_output_dir True \\
                        --log_level info \\
                        --find_unused_parameters False \\
                    """

                    # Include --max_steps for mouse data
                    if experiment_name == "mouse":
                        cmd = cmd.replace(
                            "--fp16 \\",
                            f"--fp16 \\\n    --max_steps {max_steps} \\"
                        )

                    # Execute the command
                    run_command(cmd)

        elif model == "HyenaDNA":
            # Model-specific configurations for HyenaDNA
            model_name_or_path = "LongSafari/hyenadna-medium-160k-seqlen-hf"
            train_script = "trainHyena.py"
            output_dir = "output_GUE/HyenaDNA"

            for experiment_name, datasets in experiments.items():
                # Get the corresponding per_device_train_batch_size and gradient_accumulation_steps
                per_device_train_batch_size = per_device_train_batch_size_dict.get(experiment_name, 8)
                gradient_accumulation_steps = gradient_accumulation_steps_dict.get(experiment_name, 1)

                for data in datasets:
                    # Set specific parameters based on the experiment
                    if experiment_name == "EMP":
                        model_max_length = 128
                        per_device_eval_batch_size = 16
                        num_train_epochs = 3
                        save_steps = 200
                        eval_steps = 200
                        warmup_steps = 50
                    elif experiment_name == "prom_core":
                        model_max_length = 20
                        per_device_eval_batch_size = 16
                        num_train_epochs = 4
                        save_steps = 400
                        eval_steps = 400
                        warmup_steps = 50
                    elif experiment_name == "prom_core_tata":
                        model_max_length = 20
                        per_device_eval_batch_size = 16
                        num_train_epochs = 10
                        save_steps = 200
                        eval_steps = 200
                        warmup_steps = 50
                    elif experiment_name == "prom_300":
                        model_max_length = 70
                        per_device_eval_batch_size = 16
                        num_train_epochs = 4
                        save_steps = 400
                        eval_steps = 400
                        warmup_steps = 50
                    elif experiment_name == "prom_300_tata":
                        model_max_length = 70
                        per_device_eval_batch_size = 16
                        num_train_epochs = 10
                        save_steps = 200
                        eval_steps = 200
                        warmup_steps = 50
                    elif experiment_name == "splice":
                        model_max_length = 80
                        per_device_eval_batch_size = 16
                        num_train_epochs = 5
                        save_steps = 200
                        eval_steps = 200
                        warmup_steps = 50
                    elif experiment_name == "virus":
                        model_max_length = 256
                        per_device_eval_batch_size = 32
                        num_train_epochs = 8
                        save_steps = 200
                        eval_steps = 200
                        warmup_steps = 50
                    elif experiment_name == "mouse":
                        model_max_length = 30
                        per_device_eval_batch_size = 64
                        num_train_epochs = 5
                        save_steps = 200
                        eval_steps = 200
                        warmup_steps = 30
                        max_steps = 1000  # Specific to mouse data
                    elif experiment_name == "tf":
                        model_max_length = 30
                        per_device_eval_batch_size = 64
                        num_train_epochs = 3
                        save_steps = 200
                        eval_steps = 200
                        warmup_steps = 30
                    else:
                        # Default parameters
                        model_max_length = 128
                        per_device_eval_batch_size = 16
                        num_train_epochs = 3
                        save_steps = 200
                        eval_steps = 200
                        warmup_steps = 50

                    data_subdir = experiment_subdirs.get(experiment_name, "")
                    data_full_path = f"{data_path}/GUE/{data_subdir}/{data}"

                    # Build the command
                    cmd = f"""
                    python {train_script} \\
                        --model_name_or_path {model_name_or_path} \\
                        --data_path {data_full_path} \\
                        --kmer -1 \\
                        --run_name {model}_{vocab}_{lr}_{experiment_name}_{data}_seed{seed} \\
                        --model_max_length {model_max_length} \\
                        --per_device_train_batch_size {per_device_train_batch_size} \\
                        --per_device_eval_batch_size {per_device_eval_batch_size} \\
                        --gradient_accumulation_steps {gradient_accumulation_steps} \\
                        --learning_rate {lr} \\
                        --num_train_epochs {num_train_epochs} \\
                        --fp16 \\
                        --save_steps {save_steps} \\
                        --output_dir {output_dir} \\
                        --evaluation_strategy steps \\
                        --eval_steps {eval_steps} \\
                        --warmup_steps {warmup_steps} \\
                        --logging_steps 100000 \\
                        --overwrite_output_dir True \\
                        --log_level info \\
                        --find_unused_parameters False \\
                    """

                    # Include --max_steps for mouse data
                    if experiment_name == "mouse":
                        cmd = cmd.replace(
                            "--fp16 \\",
                            f"--fp16 \\\n    --max_steps {max_steps} \\"
                        )

                    # Execute the command
                    run_command(cmd)


    if benchmark=="Promoter":
        data_path = os.path.abspath('.')

        lr = 3e-5
        vocab=""
        seed = 42
        if model == "DNABERT2":
            # Model-specific configurations for DNABERT2
            model_name_or_path = "zhihan1996/DNABERT-2-117M"
            train_script = "trainDNABERT2.py"
            output_dir = "output_Promoter/dnabert2"

            # Get the corresponding per_device_train_batch_size and gradient_accumulation_steps
            per_device_train_batch_size = per_device_train_batch_size_dict
            gradient_accumulation_steps = gradient_accumulation_steps_dict

            
            model_max_length = 128
            per_device_eval_batch_size = 16
            num_train_epochs = 3
            save_steps = 200
            eval_steps = 200
            warmup_steps = 50
            experiment_name="Promoter"
            data_full_path = f"{data_path}/Promoter_data"
            # Build the command
            cmd = f"""
            python {train_script} \\
                --model_name_or_path {model_name_or_path} \\
                --data_path {data_full_path} \\
                --kmer -1 \\
                --run_name {model}_{vocab}_{lr}_{experiment_name}_seed{seed} \\
                --model_max_length {model_max_length} \\
                --per_device_train_batch_size {per_device_train_batch_size} \\
                --per_device_eval_batch_size {per_device_eval_batch_size} \\
                --gradient_accumulation_steps {gradient_accumulation_steps} \\
                --learning_rate {lr} \\
                --num_train_epochs {num_train_epochs} \\
                --fp16 \\
                --save_steps {save_steps} \\
                --output_dir {output_dir} \\
                --evaluation_strategy steps \\
                --eval_steps {eval_steps} \\
                --warmup_steps {warmup_steps} \\
                --logging_steps 100000 \\
                --overwrite_output_dir True \\
                --log_level info \\
                --find_unused_parameters False \\
            """
            run_command(cmd)

        elif model == "HyenaDNA":
            # Model-specific configurations for HyenaDNA
            model_name_or_path = "LongSafari/hyenadna-medium-160k-seqlen-hf"
            train_script = "trainHyena.py"
            output_dir = "output_Promoter/HyenaDNA"

            # Get the corresponding per_device_train_batch_size and gradient_accumulation_steps
            per_device_train_batch_size = per_device_train_batch_size_dict
            gradient_accumulation_steps = gradient_accumulation_steps_dict

            
            model_max_length = 128
            per_device_eval_batch_size = 16
            num_train_epochs = 3
            save_steps = 200
            eval_steps = 200
            warmup_steps = 50
            experiment_name="Promoter"
            data_full_path = f"{data_path}/Promoter_data"
            # Build the command
            cmd = f"""
            python {train_script} \\
                --model_name_or_path {model_name_or_path} \\
                --data_path {data_full_path} \\
                --kmer -1 \\
                --run_name {model}_{vocab}_{lr}_{experiment_name}_seed{seed} \\
                --model_max_length {model_max_length} \\
                --per_device_train_batch_size {per_device_train_batch_size} \\
                --per_device_eval_batch_size {per_device_eval_batch_size} \\
                --gradient_accumulation_steps {gradient_accumulation_steps} \\
                --learning_rate {lr} \\
                --num_train_epochs {num_train_epochs} \\
                --fp16 \\
                --save_steps {save_steps} \\
                --output_dir {output_dir} \\
                --evaluation_strategy steps \\
                --eval_steps {eval_steps} \\
                --warmup_steps {warmup_steps} \\
                --logging_steps 100000 \\
                --overwrite_output_dir True \\
                --log_level info \\
                --find_unused_parameters False \\
            """
            run_command(cmd)

def getresult_GUE():
    # Define the output folder path
    output_folder = os.path.abspath('./output_GUE')
    results = {}

    # Walk through the output folder to find all "results" folders
    for root, dirs, files in os.walk(output_folder):
        if 'results' in root:
            # Inside "results", look for subfolders containing JSON files
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                json_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]
                for json_file in json_files:
                    json_path = os.path.join(subdir_path, json_file)
                    with open(json_path, 'r') as file:
                        data = json.load(file)
                        if 'eval_matthews_correlation' in data:
                            # Multiply eval_matthews_correlation by 100 and store it
                            results[subdir] = data['eval_matthews_correlation'] * 100

    # Save the results to All_model_GUE.json
    output_file = 'All_model_GUE.json'
    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)
    print(f"Results have been saved to {output_file}")

def getresult_Promoter():
    # Define the output folder path
    output_folder = os.path.abspath('./output_Promoter')
    results = {}

    # Walk through the output folder to find all "results" folders
    for root, dirs, files in os.walk(output_folder):
        if 'results' in root:
            # Inside "results", look for subfolders containing JSON files
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                json_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]
                for json_file in json_files:
                    json_path = os.path.join(subdir_path, json_file)
                    with open(json_path, 'r') as file:
                        data = json.load(file)
                        if 'eval_matthews_correlation' in data:
                            # Multiply eval_matthews_correlation by 100 and store it
                            results[subdir] = data['eval_matthews_correlation'] * 100

    # Save the results to All_model_GUE.json
    output_file = 'All_model_Promoter.json'
    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)
    print(f"Results have been saved to {output_file}")



per_device_train_batch_size_dict = {
    "EMP": 8,
    "prom_core": 8,
    "prom_core_tata": 8,
    "prom_300": 8,
    "prom_300_tata": 8,
    "splice": 8,
    "virus": 32,  # Different batch size for virus data
    "mouse": 8,
    "tf": 8
}

gradient_accumulation_steps_dict = {
    "EMP": 1,
    "prom_core": 1,
    "prom_core_tata": 1,
    "prom_300": 1,
    "prom_300_tata": 1,
    "splice": 1,
    "virus": 1,
    "mouse": 1,
    "tf": 1
}

# For DNABERT2

benchmark_model(
    model="DNABERT2",
    benchmark="GUE",
    per_device_train_batch_size_dict=per_device_train_batch_size_dict,
    gradient_accumulation_steps_dict=gradient_accumulation_steps_dict,
)
'''
# For HyenaDNA
benchmark_model(
    model="HyenaDNA",
    benchmark="GUE",
    per_device_train_batch_size_dict=per_device_train_batch_size_dict,
    gradient_accumulation_steps_dict=gradient_accumulation_steps_dict,
)
'''
#benchmark_model("HyenaDNA","Promoter",16,1)
#benchmark_model("DNABERT2","Promoter",8,1)