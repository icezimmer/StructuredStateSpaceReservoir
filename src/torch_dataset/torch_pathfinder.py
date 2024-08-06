import torch
from torch.utils.data import Dataset
import tensorflow_datasets as tfds


class PathfinderDataset(Dataset):
    def __init__(self, tf_dataset):
        self.data = []
        for example in tfds.as_numpy(tf_dataset):
            image, label = example
            # Take only the first channel (grayscale)
            image = image[:, :, 0:1]

            # Convert NumPy arrays to PyTorch tensors and ensure the type is float
            image = torch.from_numpy(image)

            image = image.to(dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.long)

            # Permute the dimensions from (H, W, C) to (C, H, W)
            image = image.permute(2, 0, 1)

            # Flatten the spatial dimensions while maintaining the channel dimension
            image = image.view(image.shape[0], -1)  # (C=1, H*W)

            # Transform to [0,1]
            image = image / 255.0

            # Store the preprocessed image and label
            self.data.append((image, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
