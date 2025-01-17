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
#需要后期考虑一下
def run_experiment(model,use_lora, vocab, lr, experiment_name, data, seed,
                   model_name_or_path, train_script, data_full_path,
                   model_max_length, per_device_train_batch_size,
                   per_device_eval_batch_size, gradient_accumulation_steps,
                   num_train_epochs, output_dir, eval_steps,
                   warmup_steps, save_steps, max_steps=None):
    cmd = f"""
    python {train_script} \\
        --model_name_or_path {model_name_or_path} \\
        --data_path {data_full_path} \\
        --kmer -1 \\
        --use_lora {use_lora}\\
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

    # Include --max_steps if needed
    if max_steps is not None:
        cmd = cmd.replace(
            "--fp16 \\",
            f"--fp16 \\\n    --max_steps {max_steps} \\"
        )

    run_command(cmd)

def finetune_model(model,per_device_train_batch_size, gradient_accumulation_steps,use_lora):
    
    

        data_path = os.path.abspath('.')

        lr = 3e-5
        vocab = ""
        seed = 42
        
        model_name_or_path = "zhihan1996/DNABERT-2-117M"
        train_script = "train.py"
        if use_lora==False:
            output_dir = "output_speciesclassification/dnabert2"
        else:
            output_dir = "output_speciesclassification_lora/dnabert2"

        model_max_length = 128
        per_device_eval_batch_size = 16
        num_train_epochs = 10
        save_steps = 50
        eval_steps = 50
        warmup_steps = 50
        experiment_name = "species_classification"
        data_full_path = f"{data_path}/species_classification"
        max_steps = None

        run_experiment(
            model,use_lora, vocab, lr, experiment_name, "", seed,
            model_name_or_path, train_script, data_full_path,
            model_max_length, per_device_train_batch_size,
            per_device_eval_batch_size, gradient_accumulation_steps,
            num_train_epochs, output_dir, eval_steps,
            warmup_steps, save_steps, max_steps
        )



finetune_model("DNABERT2", 8,1,use_lora=True)
