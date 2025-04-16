
# genomeAI/cli.py

import argparse
import yaml
import os
import subprocess
import sys

from .tuner import (
    run_train,
    run_inference,
    run_webui,
    run_download,
    run_process,
)


def main():
    parser = argparse.ArgumentParser(
        description="GenomeAI Command Line Interface"
    )
    parser.add_argument("command", choices=["train", "inference", "webui", "download","process"],
                        help="train or inference or webui or download or process")
    parser.add_argument("config_path", type=str, nargs="?",
                        help="Path to the YAML config file (not required for webui or download)")
    args = parser.parse_args()
    
    if args.command == "webui":
        run_webui()
        return

    if args.command == "download":
        run_download(args.config_path)
        return

    if not args.config_path:
        print("Error: config_path is required for train/inference commands.")
        sys.exit(1)

    # Load YAML config
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.command == "train":
        run_train(config)
    elif args.command == "inference":
        output = run_inference(config)
        print("Inference output:\n", output)
    elif args.command == "process":
        run_process(config)

if __name__ == "__main__":
    main()



