import torch
from torch.utils.data import Dataset


class SequentialImage2Classify(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, target = self.dataset[idx]
        target_one_hot = torch.zeros(10)
        target_one_hot[target] = 1.0

        # Return flattened image in shape (time steps = 784, dimensionality = 1)
        return image.view(1, -1), target_one_hot
