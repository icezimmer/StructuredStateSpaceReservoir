import torch
import os
import argparse
from torchvision import datasets, transforms
from src.utils.experiments import set_seed
from src.torch_dataset.sequantial_image import SequentialImage2Classify
from src.utils.saving import save_data
from lra_benchmarks.data.pathfinder import Pathfinder32, Pathfinder64, Pathfinder128, Pathfinder256
from src.torch_dataset.torch_pathfinder import PathfinderDataset

parser = argparse.ArgumentParser(description='Build Classification task.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--task', default='smnist', help='Name of classification task.')

args, unknown = parser.parse_known_args()

if args.task == 'pathfinder':
    parser.add_argument('--level', default='easy', help='Difficulty level of the task')
    parser.add_argument('--resolution', default='32', help='Image resolution')

args = parser.parse_args()

set_seed(args.seed)

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
        transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
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
    builder_dataset = builder_class()
    builder_dataset.download_and_prepare()
    develop_dataset, test_dataset = builder_dataset.as_dataset(split=[args.level + '[80%:]', args.level + '[:20%]'],
                                                               as_supervised=True)
    develop_dataset = PathfinderDataset(develop_dataset)
    test_dataset = PathfinderDataset(test_dataset)
else:
    raise ValueError('Task not found')

save_data(develop_dataset, os.path.join('./checkpoint', 'datasets', args.task, 'develop_dataset'))
save_data(test_dataset, os.path.join('./checkpoint', 'datasets', args.task, 'test_dataset'))
