import torch
from torch.utils.data import Dataset
from torch.nn.functional import normalize


class PathfinderDataset(Dataset):
    def __init__(self, tf_dataset):
        self.data = [(image.numpy(), label.numpy()) for image, label in tf_dataset]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        # Convert NumPy arrays to PyTorch tensors
        image = torch.from_numpy(image)
        # From (H, W, C) to (C, H, W)
        image = image.permute(2, 0, 1)  # (C, H, W)
        # Take only the first channel (grayscale)
        image = image[0:1, :, :]
        image = image.float()
        print(image)
        # Flatten the spatial dimensions and maintain the channel dimension
        image = image.view(image.shape[0], -1)  # Reshape to (H = num channels = 1, L = height * width = 32 * 32)
        # Normalize each channel of the image
        image = normalize(image, p=2, dim=-1)

        return image, label
