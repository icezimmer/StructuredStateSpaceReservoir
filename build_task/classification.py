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
parser.add_argument('--level', default='easy', help='Difficulty level of the task')
parser.add_argument('--resolution', type=int, default='32', help='Image resolution')
parser.add_argument('--batch', type=int, default=128, help='Batch size')
parser.add_argument('--device', default='cuda:3', help='Device for training')
args = parser.parse_args()

if args.task == 'smnist':
    develop_dataset = SequentialImage2Classify(datasets.MNIST(root='./saved_data/',
                                                              train=True,
                                                              transform=transforms.ToTensor(),
                                                              download=True), device_name=args.device)

    test_dataset = SequentialImage2Classify(datasets.MNIST(root='./saved_data/',
                                                           train=False,
                                                           transform=transforms.ToTensor(),
                                                           download=True))
elif args.task == 'pathfinder':
    pathfinders = {
        '32': Pathfinder32,
        '64': Pathfinder64,
        '128': Pathfinder128,
        '256': Pathfinder256,
    }

    builder_dataset = pathfinders[args.resolution]
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

# print('Develop dataset device: ', check_data_device(develop_dataloader))
# print('Test dataset device: ', check_data_device(test_dataloader))

save_temp_data(develop_dataloader, os.path.join('./saved_data', args.task) + '_develop_dataloader')
save_temp_data(train_dataloader, os.path.join('./saved_data', args.task) + '_train_dataloader')
save_temp_data(val_dataloader, os.path.join('./saved_data', args.task) + '_val_dataloader')
save_temp_data(test_dataloader, os.path.join('./saved_data', args.task) + '_test_dataloader')
