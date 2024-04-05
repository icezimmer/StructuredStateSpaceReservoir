# Machine Learning Framework

This repository houses a machine learning framework designed for ease of use in benchmarking classification tasks across a variety of datasets, including Sequential MNIST, CIFAR-10, and Pathfinder. The framework supports various models and configurations, facilitating comparative studies and experiments.

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.


## Building Tasks

The `build.py` script in the main directory is used to configure and prepare your classification tasks. Here's how you can build different tasks:

### Sequential MNIST

To set up a Sequential MNIST classification task, navigate to the main directory and execute:

```
python build.py --task smnist --device cuda:0 --batch 128
```

This command configures the task for Sequential MNIST, specifying the CUDA device and batch size.

### Pathfinder Task

For the Pathfinder task, which requires specifying the difficulty level and image resolution, use:

```
python build.py --task pathfinder --level easy --resolution 32 --device cuda:0 --batch 128
```

Here, `--level` and `--resolution` are additional arguments unique to the Pathfinder task.

## Running Tasks

After building your task, you can run it with the `run.py` script. This allows you to specify the model configuration and training parameters.

### Example: Running with an S4 Block

```
python run.py --task smnist --block S4 --layers 2 --neurons 64 --lr 0.001 --epochs 100
```

This command runs the task with the S4 block, setting up two layers, each with 64 neurons, and specifies the learning rate and number of epochs.

