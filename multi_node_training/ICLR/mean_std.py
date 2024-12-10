import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
import numpy as np
import argparse

def mean_std_2(mean, std):
    n=3
    mean_v = (mean[0]+mean[1])/2
    std_v = np.sqrt(((n - 1) * std[0]**2 + (n - 1) * std[1]**2) / (n + n - 2))
    print(mean_v, std_v)
    return mean_v, std_v

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description='Generate sequences using the Evo model.')

    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    args = parser.parse_args()

    # Paths to the JSON files
    file_paths = [args.data_dir+'/results/18/eval_results.json', args.data_dir+'/results/30/eval_results.json', args.data_dir+'/results/42/eval_results.json']

    # List to hold the eval_f1 values
    f1_scores = []

    # Load each file and extract the eval_f1 value
    for path in file_paths:
        try:
            with open(path, 'r') as file:
                data = json.load(file)
                f1_scores.append(data['eval_f1'])
        except FileNotFoundError:
            print(f"Error: The file {path} does not exist.")
            break  # Stop the code if the file is not found

    # Calculate mean and standard deviation of eval_f1
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    # Save the results to a new JSON file
    results = {'mean_eval_f1': mean_f1, 'std_eval_f1': std_f1}
    with open(args.data_dir+'/results/f1_stats_results.json', 'w') as outfile:
        json.dump(results, outfile)

if __name__ == "__main__":
    main()

