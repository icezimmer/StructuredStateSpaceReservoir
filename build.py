from torchvision import datasets, transforms
from src.torch_dataset.sequantial_image import SequentialImage2Classify
from torch.utils.data import DataLoader
from src.utils.split_data import split_dataset
from src.utils.temp_data import save_temp_data
import os
import argparse
from lra_benchmarks.data.pathfinder import Pathfinder32, Pathfinder64, Pathfinder128, Pathfinder256
from src.torch_dataset.torch_pathfinder import PathfinderDataset

parser = argparse.ArgumentParser(description='Build Classification task.')
parser.add_argument('--task', default='smnist', help='Name of classification task.')

args, unknown = parser.parse_known_args()

if args.task == 'pathfinder':
    parser.add_argument('--level', default='easy', help='Difficulty level of the task')
    parser.add_argument('--resolution', default='32', help='Image resolution')

parser.add_argument('--batch', type=int, default=128, help='Batch size')
parser.add_argument('--device', default='cuda:3', help='Device for training')

args = parser.parse_args()

if args.task == 'smnist':
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to pytorch tensor with values in [0, 1] and shape (C, H, W)
        #transforms.Normalize((0.1307,), (0.3081,)),
    ])
    develop_dataset = SequentialImage2Classify(datasets.MNIST(root='./checkpoint/',
                                                              train=True,
                                                              transform=transform,
                                                              download=True), device_name=args.device)
    test_dataset = SequentialImage2Classify(datasets.MNIST(root='./checkpoint/',
                                                           train=False,
                                                           transform=transform,
                                                           download=True))
elif args.task == 'scifar10':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    develop_dataset = SequentialImage2Classify(datasets.CIFAR10(root='./checkpoint/',
                                                                train=True,
                                                                transform=transform,
                                                                download=True), device_name=args.device)
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
    develop_dataset = PathfinderDataset(develop_dataset, device_name=args.device)
    test_dataset = PathfinderDataset(test_dataset)
else:
    raise ValueError('Task not found')

train_dataset, val_dataset = split_dataset(develop_dataset)

develop_dataloader = DataLoader(develop_dataset, batch_size=args.batch, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, pin_memory=True, num_workers=128)

save_temp_data(develop_dataloader, os.path.join('checkpoint', args.task) + '_develop_dataloader')
save_temp_data(train_dataloader, os.path.join('checkpoint', args.task) + '_train_dataloader')
save_temp_data(val_dataloader, os.path.join('checkpoint', args.task) + '_val_dataloader')
save_temp_data(test_dataloader, os.path.join('checkpoint', args.task) + '_test_dataloader')