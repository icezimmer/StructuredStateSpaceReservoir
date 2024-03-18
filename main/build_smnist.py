import torch
from torchvision import datasets, transforms
from src.torch_dataset.sequantial_image import SequentialImage2Classify
from torch.utils.data import DataLoader
from src.utils.temp_data import save_temp_data

# Transformation to convert images to tensors and flatten them
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to (C, H, W) in the range [0.0, 1.0]
    transforms.Normalize((0.1307,), (0.3081,)),
])

trainset = SequentialImage2Classify(datasets.MNIST(root='../saved_data/',
                                                   train=True,
                                                   transform=transform,
                                                   download=True))
train_dataloader = DataLoader(trainset, batch_size=128, shuffle=True)

testset = SequentialImage2Classify(datasets.MNIST(root='../saved_data/',
                                                  train=False,
                                                  download=True,
                                                  transform=transform))
test_dataloader = DataLoader(testset, batch_size=len(testset), shuffle=False)

save_temp_data(train_dataloader, 'smnist_train_dataloader')
save_temp_data(test_dataloader, 'smnist_test_dataloader')
