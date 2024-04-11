import torch
from torch.utils.data import Dataset


class PathfinderDataset(Dataset):
    def __init__(self, tf_dataset, device_name=None):
        self.data = []
        for image, label in tf_dataset:
            # Take only the first channel (grayscale)
            image, label = image[:, :, 0:1].numpy(), label.numpy()

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

            if device_name is not None:
                image = image.to(torch.device(device_name))
                label = label.to(torch.device(device_name))

            # Store the preprocessed image and label
            self.data.append((image, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
