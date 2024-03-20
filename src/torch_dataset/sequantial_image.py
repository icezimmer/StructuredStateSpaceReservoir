import torch
from torch.nn.functional import normalize
from torch.utils.data import Dataset


class SequentialImage2Classify(Dataset):

    def __init__(self, dataset, device_name):
        self.device = torch.device(device_name)
        self.data = []
        for image, label in dataset:
            image = image.to(device=self.device, dtype=torch.float32).view(image.shape[0], -1)
            image = normalize(image, p=2, dim=-1)
            label = torch.tensor(label, device=self.device, dtype=torch.long)
            self.data.append((image, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
