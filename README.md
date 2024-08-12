# Machine Learning Framework

This repository houses a machine learning framework designed for ease of use in benchmarking classification tasks across a variety of datasets, including Sequential MNIST, CIFAR-10, and Pathfinder. The framework supports various models and configurations, facilitating comparative studies and experiments.

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.

## Building Tasks

The `build.py` script in the main directory is used to download and prepare the dataset relative to the task selected. Here's how you can build different tasks:

### Sequential MNIST

To set up a Sequential MNIST classification task, navigate to the main directory and execute:

```
python build.py --task smnist
```

### Pathfinder Task

For the Pathfinder task, which requires specifying the difficulty level and image resolution, use:

```
python build.py --task pathfinder --level easy --resolution 32
```

Here, `--level` and `--resolution` are additional arguments unique to the Pathfinder task.

## Running Tasks

After building your task, you can run it with the `train.py` script. This allows you to specify the device, the task, the batch size, the model configuration, and training hyperparameters.

### Example: Sequential MNIST with S4 block

python train.py --device cuda:0 --task smnist --batch 128 --block S4 --layers 2 --neurons 64 --lr 0.001 --epochs 100 --patience 10

This command runs the task in the selected device, setting the data in batch. You can select the block of the layer, the number of layers, the number of neurons for each layer, and specify the learning rate, the number of epochs, and the patience for early stopping.

### Configuration Files

To streamline model configuration and hyperparameter tuning, you can use a `*.yaml` configuration file tailored to a specific task and model. The `run.py` script will execute the configurations defined within the file. If the file includes multiple values for any hyperparameters, the script will automatically select a subset of possible configurations to run. The maximum number of configurations executed is determined by the max_configs parameter specified in the `*.yaml` file.

```
python run.py --config=configs/smnist/best/rssm_r.yaml
```

This command trains the RSSM model with the Ridge readout for the Sequential MNIST task with all the configuration and hyperparameters set in the config file.
