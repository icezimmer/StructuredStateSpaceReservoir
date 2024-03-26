from torchvision import datasets, transforms
from src.torch_dataset.sequantial_image import SequentialImage2Classify
from torch.utils.data import DataLoader
from src.utils.split_data import split_dataset
from src.utils.temp_data import save_temp_data
import os

develop_dataset = SequentialImage2Classify(datasets.MNIST(root='./saved_data/',
                                                          train=True,
                                                          transform=transforms.ToTensor(),
                                                          download=True), device_name='cuda:1')

train_dataset, val_dataset = split_dataset(develop_dataset)
develop_dataloader = DataLoader(develop_dataset, batch_size=128, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

test_dataset = SequentialImage2Classify(datasets.MNIST(root='./saved_data/',
                                                       train=False,
                                                       transform=transforms.ToTensor(),
                                                       download=True))
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=128)

# print('Develop dataset device: ', check_data_device(develop_dataloader))
# print('Test dataset device: ', check_data_device(test_dataloader))

TASK = 'smnist'
save_temp_data(develop_dataloader, os.path.join('./saved_data', TASK) + '_develop_dataloader')
save_temp_data(train_dataloader, os.path.join('./saved_data', TASK) + '_train_dataloader')
save_temp_data(val_dataloader, os.path.join('./saved_data', TASK) + '_val_dataloader')
save_temp_data(test_dataloader, os.path.join('./saved_data', TASK) + '_test_dataloader')
