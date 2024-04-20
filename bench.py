import itertools
import subprocess


def main():
    hyperparameters = {
        "task": ["smnist"],
        "block": ["S4R"],
        "conv": ["fft", "fft-freezeD"],
        #"kerneldrop": 0.0,
        "kernel": ["miniV", "miniV-freezeA", "miniV-freezeW", "miniV-freezeAW"],
        #"dt": null,
        "strong": [0.75],
        "weak": [0.9],
        #"kernellr": 0.001,
        #"kernelwd": 0.0,
        #"scaleB": 1.0,
        #"scaleC": 1.0,
        "neurons": [64],
        "encoder": ["reservoir"],
        "mix": ["conv1d+glu", "reservoir+glu"],
        "decoder": ["conv1d"],
        #"dropout": 0.0,
        "layers": [2],
        #"layerdrop": 0.0,
        "lr": [0.004],
        #"wd": 0.1,
        #"epochs": Infinity,
        #"patience": 10
    }

    # Generate all combinations of hyperparameters
    keys, values = zip(*hyperparameters.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Format the combinations into command line arguments
    for experiment in experiments:
        args = []
        for key, value in experiment.items():
            args.append(f"--{key}={value}")

        # Prepare the command
        command = ["python", "run.py"] + args
        command_str = " ".join(command)

        subprocess.run(command)  # Uncomment this line to run the experiments automatically

        # Print the command for manual execution
        print(command_str)


if __name__ == "__main__":
    main()
