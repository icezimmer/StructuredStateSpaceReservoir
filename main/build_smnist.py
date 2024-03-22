from torchvision import datasets, transforms
from src.torch_dataset.sequantial_image import SequentialImage2Classify
from torch.utils.data import DataLoader, random_split
from src.utils.temp_data import save_temp_data

develop_dataset = SequentialImage2Classify(datasets.MNIST(root='../saved_data/',
                                                          train=True,
                                                          transform=transforms.ToTensor(),
                                                          download=True), device_name='cuda:1')
train_size = int(0.8 * len(develop_dataset))
val_size = len(develop_dataset) - train_size
train_dataset, val_dataset = random_split(develop_dataset, [train_size, val_size])
develop_dataloader = DataLoader(develop_dataset, batch_size=128, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

test_dataset = SequentialImage2Classify(datasets.MNIST(root='../saved_data/',
                                                       train=False,
                                                       transform=transforms.ToTensor(),
                                                       download=True))
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=128)

save_temp_data(develop_dataloader, 'smnist_develop_dataloader')
save_temp_data(train_dataloader, 'smnist_train_dataloader')
save_temp_data(val_dataloader, 'smnist_val_dataloader')
save_temp_data(test_dataloader, 'smnist_test_dataloader')
