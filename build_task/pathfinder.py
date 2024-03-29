from lra_benchmarks.data.pathfinder import Pathfinder32, Pathfinder64, Pathfinder128, Pathfinder256
from src.torch_dataset.torch_pathfinder import PathfinderDataset
from src.utils.temp_data import save_temp_data
from torch.utils.data import DataLoader
from src.utils.split_data import split_dataset
import os
import argparse

parser = argparse.ArgumentParser(description='Build PathFinder task.')
parser.add_argument('--level', default='easy', help='Difficulty level of the task')
parser.add_argument('--resolution', type=int, default='32', help='Image resolution')
parser.add_argument('--batch', type=int, default=128, help='Batch size')
parser.add_argument('--device', default='cuda:3', help='Device for training')
args = parser.parse_args()

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
train_dataset, val_dataset = split_dataset(develop_dataset)
develop_dataloader = DataLoader(develop_dataset, batch_size=args.batch, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

test_dataset = PathfinderDataset(test_dataset)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, pin_memory=True, num_workers=128)

TASK = 'pathfinder'
save_temp_data(develop_dataloader, os.path.join('./saved_data', TASK) + '_develop_dataloader')
save_temp_data(train_dataloader, os.path.join('./saved_data', TASK) + '_train_dataloader')
save_temp_data(val_dataloader, os.path.join('./saved_data', TASK) + '_val_dataloader')
save_temp_data(test_dataloader, os.path.join('./saved_data', TASK) + '_test_dataloader')
