import torch
import os
import logging
import argparse
import tensorflow as tf
from torchvision import datasets, transforms
from src.utils.experiments import set_seed
from src.torch_dataset.sequantial_image import SequentialImage2Classify
from src.utils.saving import save_data
from lra_benchmarks.data.pathfinder import Pathfinder32, Pathfinder64, Pathfinder128, Pathfinder256
from src.torch_dataset.torch_pathfinder import PathfinderDataset
from src.torch_dataset.torch_listops import ListOpsDataset
from torch.utils.data import DataLoader
from src.utils.split_data import random_split_dataset
from src.models.embedding.embedding import EmbeddingModel
from lra_benchmarks.data.listops import listops
from lra_benchmarks.listops import input_pipeline
import tensorflow_datasets as tfds
from src.ml.optimization import setup_optimizer
from src.ml.training import TrainModel
from src.utils.saving import load_data

tasks = ['smnist', 'pmnist', 'scifar10gs', 'scifar10', 'pathfinder', 'listops']

parser = argparse.ArgumentParser(description='Build Classification task.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--task', required=True, choices=tasks, help='Name of classification task.')

args, unknown = parser.parse_known_args()

if args.task == 'pathfinder':
    parser.add_argument('--level', default='easy', help='Difficulty level of the task')
    parser.add_argument('--resolution', default='32', help='Image resolution')

if args.task == 'listops':
    parser.add_argument('--num_dev_samples', type=int, default=98000, help='Number of train samples.')
    parser.add_argument('--num_test_samples', type=int, default=2000, help='Number of test samples.')
    parser.add_argument('--max_depth', type=int, default=10, help='Maximum tree depth of training sequences.')
    parser.add_argument('--max_args', type=int, default=10, help='Maximum number of arguments per operator in training sequences.')
    parser.add_argument('--max_length', type=int, default=2000, help='Maximum length per sequence in training sequences.')
    parser.add_argument('--min_length', type=int, default=500, help='Minimum length per sequence in training sequences.')

logging.basicConfig(level=logging.INFO)
args = parser.parse_args()

if args.task == 'pathfinder':
    task_name = f'{args.task}{args.resolution}{args.level}'
else:
    task_name = args.task

logging.info(f"Setting seed: {args.seed}")
set_seed(args.seed)

logging.info(f"Building task: {task_name}")

if args.task == 'smnist':
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to pytorch tensor with values in [0, 1] and shape (C, H, W)
        # transforms.Normalize((0.1307,), (0.3081,)),
    ])
    develop_dataset = SequentialImage2Classify(datasets.MNIST(root='./checkpoint/',
                                                              train=True,
                                                              transform=transform,
                                                              download=True))
    test_dataset = SequentialImage2Classify(datasets.MNIST(root='./checkpoint/',
                                                           train=False,
                                                           transform=transform,
                                                           download=True))
elif args.task == 'pmnist':
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to pytorch tensor with values in [0, 1] and shape (C, H, W)
        # transforms.Normalize((0.1307,), (0.3081,)),
    ])
    permutation = torch.randperm(28 * 28)
    develop_dataset = SequentialImage2Classify(dataset=datasets.MNIST(root='./checkpoint/',
                                                                      train=True,
                                                                      transform=transform,
                                                                      download=True),
                                               permutation=permutation)
    test_dataset = SequentialImage2Classify(dataset=datasets.MNIST(root='./checkpoint/',
                                                                   train=False,
                                                                   transform=transform,
                                                                   download=True),
                                            permutation=permutation)
elif args.task == 'scifar10':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    develop_dataset = SequentialImage2Classify(datasets.CIFAR10(root='./checkpoint/',
                                                                train=True,
                                                                transform=transform,
                                                                download=True))
    test_dataset = SequentialImage2Classify(datasets.CIFAR10(root='./checkpoint/',
                                                             train=False,
                                                             transform=transform,
                                                             download=True))
elif args.task == 'scifar10gs':
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
    ])
    develop_dataset = SequentialImage2Classify(datasets.CIFAR10(root='./checkpoint/',
                                                                train=True,
                                                                transform=transform,
                                                                download=True))
    test_dataset = SequentialImage2Classify(datasets.CIFAR10(root='./checkpoint/',
                                                             train=False,
                                                             transform=transform,
                                                             download=True))
elif args.task == 'pathfinder':
    pathfinders = {
        '32': Pathfinder32,
        '64': Pathfinder64,
        '128': Pathfinder128,
        '256': Pathfinder256,
    }
    builder_class = pathfinders[args.resolution]
    try:
        builder_dataset = builder_class()
        builder_dataset.download_and_prepare()

        # Load the dataset with shuffled files
        develop_dataset, test_dataset = builder_dataset.as_dataset(split=[args.level+'[:80%]', args.level+'[80%:]'],
                                                                   shuffle_files=True,
                                                                   decoders={'image': tfds.decode.SkipDecoding()})

        # Filter out examples with empty images
        develop_dataset = develop_dataset.filter(lambda x: tf.strings.length((x['image'])) > 0)
        test_dataset = test_dataset.filter(lambda x: tf.strings.length((x['image'])) > 0)

        def decode(x):
            decoded = {
                'input': tf.cast(tf.image.decode_png(x['image']), dtype=tf.int32),
                'label': x['label']
            }
            return decoded

        develop_dataset = develop_dataset.map(decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.map(decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Convert TensorFlow dataset to PyTorch dataset
        develop_dataset = PathfinderDataset(develop_dataset)
        test_dataset = PathfinderDataset(test_dataset)
    except FileNotFoundError:
        logging.error(f"Pathfinder dataset {args.level} level not found. To download the datasets,"
                      f"please download it from gs://long-range-arena/lra_release."
                      f"If permissions fail, you may download the entire gziped file at"
                      f"https://storage.googleapis.com/long-range-arena/lra_release.gz. Put the data behind the"
                      f"project directory")

elif args.task == 'listops':
    listops(task_name=task_name, num_develop_samples=args.num_dev_samples, num_test_samples=args.num_test_samples,
            max_depth=args.max_depth, max_args=args.max_args,
            max_length=args.max_length, min_length=args.min_length,
            output_dir=os.path.join('./checkpoint', 'datasets', task_name))

    develop_dataset, test_dataset, encoder = input_pipeline.get_datasets(
                                                                n_devices=4,
                                                                task_name=task_name,
                                                                data_dir=os.path.join('./checkpoint', 'datasets', task_name),
                                                                max_length=args.max_length)

    # Convert TensorFlow dataset to PyTorch dataset
    develop_dataset = ListOpsDataset(develop_dataset)
    test_dataset = ListOpsDataset(test_dataset)
else:
    raise ValueError('Task not found')

logging.info('Saving datasets')
save_data(develop_dataset, os.path.join('./checkpoint', 'datasets', task_name, 'develop_dataset'))
save_data(test_dataset, os.path.join('./checkpoint', 'datasets', task_name, 'test_dataset'))
