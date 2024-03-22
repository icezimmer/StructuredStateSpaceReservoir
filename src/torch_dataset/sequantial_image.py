import torch
from torch.nn.functional import normalize
from torch.utils.data import Dataset


class SequentialImage2Classify(Dataset):

    def __init__(self, dataset, device_name=None):
        self.data = []
        for image, label in dataset:
            image = image.to(dtype=torch.float32)
            image = image.view(image.shape[0], -1)
            image = normalize(image, p=2, dim=-1)
            label = torch.tensor(label, dtype=torch.long)
            if device_name is not None:
                image = image.to(torch.device(device_name))
                label = label.to(torch.device(device_name))
            self.data.append((image, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
