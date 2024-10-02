import argparse
import subprocess
import random
import numpy as np
from src.utils.experiments import read_yaml_to_dict


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run experiments based on YAML configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Read hyperparameters from the YAML file
    config = read_yaml_to_dict(args.config)

    max_configs = config.get('max_configs', 0)
    hyperparameters = config.get('hyperparameters', {})

    # Extract the keys and possible values for each hyperparameter
    keys, values = zip(*hyperparameters.items())

    # Perform random sampling of hyperparameter combinations
    sampled_experiments = []
    for _ in range(max_configs):
        experiment = {key: random.choice(value) for key, value in zip(keys, values)}
        sampled_experiments.append(experiment)

    # Generate and execute commands for the sampled experiments
    for experiment in sampled_experiments:
        args = []
        for key, value in experiment.items():
            if isinstance(value, bool):
                if value:  # Only add the flag if it's True
                    args.append(f"--{key}")
            else:
                args.append(f"--{key}={value}")

        # Prepare the command
        command = ["python", "train.py"] + args

        # Print the command for manual execution
        command_str = " ".join(command)
        print(command_str)

        # Execute the command and continue even if there's an error
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}. Continuing with next experiment.")


if __name__ == "__main__":
    main()
