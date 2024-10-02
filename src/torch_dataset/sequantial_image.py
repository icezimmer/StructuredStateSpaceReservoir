import torch
from torch.utils.data import Dataset


class SequentialImage2Classify(Dataset):

    def __init__(self, dataset, permutation=None):
        self.data = []
        for image, label in dataset:
            image = image.to(dtype=torch.float32)
            image = image.view(image.shape[0], -1)  # (C, H, W) -> (C, H * W)
            if permutation is not None:
                image = image[:, permutation]  # Apply the permutation
            label = torch.tensor(label, dtype=torch.long)

            self.data.append((image, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
