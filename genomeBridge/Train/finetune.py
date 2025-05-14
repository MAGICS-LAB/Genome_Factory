import os
import zipfile
import gdown
import json
import subprocess
import signal
import argparse

def run_command(cmd):
    """
    Minimal change: remove preexec_fn=os.setsid and use process.wait(),
    so Ctrl+C (KeyboardInterrupt) properly terminates the subprocess.
    """
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            executable='/bin/bash'  # removed preexec_fn=os.setsid
        )
        process.wait()  # wait until it finishes or is interrupted
    except KeyboardInterrupt:
        # If the user presses Ctrl+C, terminate the process gracefully
        process.terminate()
        process.wait()
        raise
def run_experiment(
    use_flash_attention,
    use_lora,
    lora_r,
    lora_alpha,
    lora_dropout,
    lora_target_modules,
    lr,
    model_name_or_path,
    train_script_path,
    data_full_path,
    model_max_length,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    gradient_accumulation_steps,
    num_train_epochs,
    output_dir,
    eval_steps,
    warmup_steps,
    save_steps,
    logging_steps,
    evaluation_strategy,
    lr_scheduler_type,
    warmup_ratio,
    fp16,
    bf16,
    ddp_timeout,
    run_name,
    overwrite_output_dir,
    save_total_limit,
    load_best_model_at_end,
    saved_model_dir,
):
    cmd_list = [
        "python",
        train_script_path,
        "--model_name_or_path", model_name_or_path,
        "--data_path", data_full_path,
        "--use_flash_attention", str(use_flash_attention),
        "--use_lora", str(use_lora),
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--lora_dropout", str(lora_dropout),
        "--lora_target_modules", str(lora_target_modules),
        "--run_name", str(run_name),
        "--model_max_length", str(model_max_length),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--per_device_eval_batch_size", str(per_device_eval_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--learning_rate", str(lr),
        "--num_train_epochs", str(num_train_epochs),
    ]

    if fp16 == "True" or fp16 is True:
        cmd_list.append("--fp16")
        cmd_list.append(str(fp16))
    if bf16 == "True" or bf16 is True:
        cmd_list.append("--bf16")
        cmd_list.append(str(bf16))


    # Append the rest
    cmd_list += [
        "--save_steps", str(save_steps),
        "--output_dir", str(output_dir),
        "--evaluation_strategy", str(evaluation_strategy),
        "--eval_steps", str(eval_steps),
        "--warmup_steps", str(warmup_steps),
        "--logging_steps", str(logging_steps),
        "--overwrite_output_dir", str(overwrite_output_dir),
        "--log_level", "info",
        "--find_unused_parameters", "False",
        "--lr_scheduler_type", str(lr_scheduler_type),
        "--warmup_ratio", str(warmup_ratio),
        "--ddp_timeout", str(ddp_timeout),
        "--save_total_limit", str(save_total_limit),
        "--load_best_model_at_end", str(load_best_model_at_end),
    ]


    if saved_model_dir:
        cmd_list += ["--saved_model_dir", saved_model_dir]

    final_cmd = " ".join(cmd_list)
    run_command(final_cmd)

def finetune_model(
    per_device_train_batch_size,
    gradient_accumulation_steps,
    use_flash_attention,
    use_lora,
    lora_r,
    lora_alpha,
    lora_dropout,
    lora_target_modules,
    data_path,
    run_name,
    model_max_length,
    per_device_eval_batch_size,
    lr,
    num_train_epochs,
    save_steps,
    output_dir,
    eval_steps,
    finetuning_type,
    warmup_steps,
    logging_steps,
    evaluation_strategy,
    lr_scheduler_type,
    warmup_ratio,
    fp16,
    bf16,
    classification,
    regression,
    ddp_timeout,
    overwrite_output_dir,
    save_total_limit,
    load_best_model_at_end,
    model_name_or_path="zhihan1996/DNABERT-2-117M",
    saved_model_dir=None
    
):
    if use_lora == "False":
        output_dir = "output_speciesclassification/dnabert2" if output_dir is None else output_dir
    else:
        output_dir = "output_speciesclassification_lora/dnabert2" if output_dir is None else output_dir

    data_full_path = os.path.abspath(data_path)
    
    if finetuning_type=="adapter":
        if classification=="True":
            if "evo" in model_name_or_path:
                train_script_path = os.path.join(os.path.dirname(__file__), "train_scripts/adapter/train_evo_adapter_classification.py")
            else:
                train_script_path = os.path.join(os.path.dirname(__file__), "train_scripts/adapter/train_adapter_classification.py")

        if regression=="True":
            if "evo" in model_name_or_path:
                train_script_path = os.path.join(os.path.dirname(__file__), "train_scripts/adapter/train_evo_adapter_regression.py")
            else:
                train_script_path = os.path.join(os.path.dirname(__file__), "train_scripts/adapter/train_adapter_regression.py")
    else:
        if classification=="True":
            train_script_path = os.path.join(os.path.dirname(__file__), "train_scripts/full_and_lora/train_classification.py")

        if regression=="True":
            train_script_path = os.path.join(os.path.dirname(__file__), "train_scripts/full_and_lora/train_regression.py")

    run_experiment(
        use_flash_attention=use_flash_attention,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        lr=lr,
        model_name_or_path=model_name_or_path,
        train_script_path=train_script_path,
        data_full_path=data_full_path,
        model_max_length=model_max_length,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        output_dir=output_dir,
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        evaluation_strategy=evaluation_strategy,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        fp16=str(fp16),
        bf16=str(bf16),
        ddp_timeout=ddp_timeout,
        run_name=run_name,
        overwrite_output_dir=str(overwrite_output_dir),
        save_total_limit=save_total_limit,
        load_best_model_at_end=str(load_best_model_at_end),
        saved_model_dir=(saved_model_dir if saved_model_dir else "")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--use_flash_attention", type=str, default="True")
    parser.add_argument("--use_lora", type=str, default="False")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="Wqkv,dense,gated_layers,wo,classifier")
    parser.add_argument("--data_path", type=str, default="./dataset")
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--finetuning_type", type=str, default="full")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--fp16", type=str, default="False")
    parser.add_argument("--bf16", type=str, default="False")
    parser.add_argument("--classification", type=str, default="True")
    parser.add_argument("--regression", type=str, default="False")

    parser.add_argument("--ddp_timeout", type=int, default=180000000)
    parser.add_argument("--overwrite_output_dir", type=str, default="True")
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--load_best_model_at_end", type=str, default="True")
    parser.add_argument("--saved_model_dir", type=str, default=None)

    args = parser.parse_args()

    finetune_model(
        use_flash_attention=args.use_flash_attention,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        data_path=args.data_path,
        run_name=args.run_name,
        model_max_length=args.model_max_length,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        lr=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        finetuning_type=args.finetuning_type,
        output_dir=args.output_dir,
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        classification=args.classification,
        regression=args.regression,
        evaluation_strategy=args.evaluation_strategy,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        fp16=(args.fp16.lower() == "true"),
        bf16=(args.bf16.lower() == "true"),
        ddp_timeout=args.ddp_timeout,
        overwrite_output_dir=(args.overwrite_output_dir.lower() == "true"),
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=(args.load_best_model_at_end.lower() == "true"),
        model_name_or_path=args.model_name,
        saved_model_dir=args.saved_model_dir
    )



