import torch
from torch.utils.data import Dataset
import tensorflow_datasets as tfds


class PathfinderDataset(Dataset):
    def __init__(self, tf_dataset):
        self.data = []
        for example in tfds.as_numpy(tf_dataset):
            try:
                image, label = example['input'], example['label']
                # Check if the image is valid and not empty
                if image.size == 0:
                    raise ValueError("Empty image found")

                # Convert NumPy arrays to PyTorch tensors and ensure the type is float
                image = torch.from_numpy(image)

                image = image.to(dtype=torch.float32)
                label = torch.tensor(label, dtype=torch.long)

                # Permute the dimensions from (H, W, C=1) to (C=1, H, W)
                image = image.permute(2, 0, 1)

                # Flatten the spatial dimensions while maintaining the channel dimension
                image = image.view(image.shape[0], -1)  # (C=1, H*W)

                # Transform to [0,1]
                image = image / 255.0

                # Store the preprocessed image and label
                self.data.append((image, label))
            except Exception as e:
                print(f"Skipping example due to error: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
