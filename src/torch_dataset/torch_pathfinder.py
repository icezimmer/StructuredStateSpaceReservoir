from torch.utils.data import Dataset
import torch


import torch
from torch.utils.data import Dataset
import tensorflow as tf
import numpy as np


class PathfinderDataset(Dataset):
    def __init__(self, tf_dataset):
        self.data = [(image.numpy(), label.numpy()) for image, label in tf_dataset]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        # Convert NumPy arrays to PyTorch tensors
        image = torch.from_numpy(image)  # Convert to float
        image = image.permute(2, 0, 1)  # Change HWC to CHW
        # Normalize each channel of the image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        # Flatten the spatial dimensions and maintain the channel dimension
        image = image.view(image.shape[0], -1)  # Reshape to (3, 32*32)

        return image, label
