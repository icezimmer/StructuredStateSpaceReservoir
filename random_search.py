import argparse
import itertools
import subprocess
import random
import numpy as np
import yaml


def read_yaml_to_dict(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run experiments based on YAML configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Read hyperparameters from the YAML file
    config = read_yaml_to_dict(args.config)

    max_configs = config.get('max_configs', 0)
    hyperparameters = config.get('hyperparameters', {})

    # Generate all combinations of hyperparameters
    keys, values = zip(*hyperparameters.items())

    # Calculate the total number of combinations
    total_experiment_count = np.prod([len(v) for v in values])

    # Randomly sample indices
    samples = set(random.sample(range(total_experiment_count), min(max_configs, total_experiment_count.item())))

    # Generate experiments only for sampled indices
    for idx, experiment in enumerate(itertools.product(*values)):
        if idx in samples:
            args = []
            experiment_dict = dict(zip(keys, experiment))
            for key, value in experiment_dict.items():
                if isinstance(value, bool):
                    if value:  # Only add the flag if it's True
                        args.append(f"--{key}")
                else:
                    args.append(f"--{key}={value}")

            # Prepare the command
            command = ["python", "run.py"] + args

            # Print the command for manual execution
            command_str = " ".join(command)
            print(command_str)

            # Execute the command
            subprocess.run(command)


if __name__ == "__main__":
    main()
