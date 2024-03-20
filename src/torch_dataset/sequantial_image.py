import torch
from torch.utils.data import Dataset


class SequentialImage2Classify(Dataset):

    def __init__(self, dataset, device_name):
        self.device = torch.device(device_name)
        self.data = [(image.to(device=self.device, dtype=torch.float32).view(image.shape[0], -1),
                      torch.tensor(label, device=self.device, dtype=torch.long))
                     for image, label in dataset]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
