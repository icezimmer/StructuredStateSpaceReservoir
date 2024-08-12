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
from lra_benchmarks.data.listops import listops
from lra_benchmarks.listops import input_pipeline
import tensorflow_datasets as tfds
from torchtext.datasets import IMDB
from src.torch_dataset.torch_text import TextDataset
from src.utils.experiments import read_yaml_to_dict

tasks = ['smnist', 'pmnist', 'scifar10gs', 'scifar10', 'pathfinder', 'pathx', 'listops', 'imdb']

parser = argparse.ArgumentParser(description='Build Classification task.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--task', required=True, choices=tasks, help='Name of classification task.')

args, unknown = parser.parse_known_args()

logging.basicConfig(level=logging.INFO)
args = parser.parse_args()

logging.info(f"Setting seed: {args.seed}")
set_seed(args.seed)

logging.info(f"Building task: {args.task}")

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
elif args.task in ['pathfinder', 'pathx']:
    setting = read_yaml_to_dict(os.path.join('configs', args.task, 'setting.yaml'))
    
    data = setting.get('data', {})
    resolution = str(data.get('resolution'))
    level = data.get('level')

    learning = setting.get('learning')
    test_split = learning.get('test_split')
    train_split = str(int((1 - test_split) * 100)) + '%'

    pathfinders = {
        '32': Pathfinder32,
        '64': Pathfinder64,
        '128': Pathfinder128,
        '256': Pathfinder256,
    }
    builder_class = pathfinders[resolution]
    try:
        builder_dataset = builder_class()
        builder_dataset.download_and_prepare()

        # Load the dataset with shuffled files
        develop_dataset, test_dataset = builder_dataset.as_dataset(split=[level+'[:'+train_split+']', level+'['+train_split+':]'],
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
        logging.error(f"Pathfinder dataset {level} level not found. To download the datasets,"
                      f"please download it from gs://long-range-arena/lra_release."
                      f"If permissions fail, you may download the entire gziped file at"
                      f"https://storage.googleapis.com/long-range-arena/lra_release.gz. Put the data behind the"
                      f"project directory")

elif args.task == 'listops':
    setting = read_yaml_to_dict(os.path.join('configs', args.task, 'setting.yaml'))

    architecture = setting.get('architecture', {})
    vocab_size = architecture['d_input']
    max_length = architecture['kernel_size']

    data = setting.get('data', {})
    num_dev_samples = data['num_dev_samples']
    num_test_samples = data['num_test_samples']
    max_depth = data['max_depth']
    max_args = data['max_args']
    max_length = data['max_length']
    min_length = data['min_length']

    mode = setting.get('mode', "")
    listops(task_name=args.task, num_develop_samples=num_dev_samples, num_test_samples=num_test_samples,
            max_depth=max_depth, max_args=max_args,
            max_length=max_length, min_length=min_length,
            output_dir=os.path.join('./checkpoint', 'datasets', args.task))

    develop_dataset, test_dataset, encoder = input_pipeline.get_datasets(
                                                                n_devices=4,
                                                                task_name=args.task,
                                                                data_dir=os.path.join('./checkpoint', 'datasets', args.task),
                                                                max_length=args.max_length)

    # Convert TensorFlow dataset to PyTorch dataset
    develop_dataset = ListOpsDataset(develop_dataset)
    test_dataset = ListOpsDataset(test_dataset)

elif args.task == 'imdb':
    setting = read_yaml_to_dict(os.path.join('configs', args.task, 'setting.yaml'))
    
    architecture = setting.get('architecture', {})
    vocab_size = architecture['d_input']
    max_length = architecture['kernel_size']

    # Download and load IMDB dataset
    develop_dataset, test_dataset = IMDB(root='./checkpoint/')

    develop_dataset = TextDataset(dataset=develop_dataset, max_length=max_length)
    test_dataset = TextDataset(dataset=test_dataset, max_length=max_length)
else:
    raise ValueError('Task not found')

logging.info('Saving datasets')
save_data(develop_dataset, os.path.join('./checkpoint', 'datasets', args.task, 'develop_dataset'))
save_data(test_dataset, os.path.join('./checkpoint', 'datasets', args.task, 'test_dataset'))
