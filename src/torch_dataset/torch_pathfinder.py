from torch.utils.data import Dataset
import torch


import torch
from torch.utils.data import Dataset
import tensorflow as tf
import numpy as np


class PathfinderDataset(Dataset):
    def __init__(self, tf_dataset):
        self.data = []
        self.labels = []

        for item in tf_dataset:
            image, label = item['image'], item['label']  # Use key indexing here

            # Ensure running in eager execution mode to use .numpy()
            if tf.executing_eagerly():
                image_np = image.numpy()  # Convert TensorFlow tensor to NumPy array
                label_np = label.numpy()  # Convert TensorFlow tensor to NumPy array

                # Convert NumPy arrays to PyTorch tensors
                image_torch = torch.from_numpy(image_np).float()  # Convert to float
                image_torch = image_torch.permute(2, 0, 1)  # Change HWC to CHW
                # Flatten the spatial dimensions and maintain the channel dimension
                image_torch = image_torch.view(3, -1)  # Reshape to (3, 32*32)

                label_torch = torch.tensor(label_np, dtype=torch.long)

                self.data.append(image_torch)
                self.labels.append(label_torch)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# class PathfinderDataset(Dataset):
#     def __init__(self, tf_dataset):
#         """
#         Initialize the dataset with TensorFlow data.
#         Args:
#         - tf_dataset: A TensorFlow dataset yielding tuples of (image, label).
#         """
#         # Convert TensorFlow dataset to a list of numpy arrays for compatibility.
#         # Note: This step might be memory-intensive for large datasets and could be optimized.
#         print(tf_dataset)
#         self.samples = list(tf_dataset)
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx: int):
#         # Retrieve the TensorFlow data sample (numpy array) by index.
#         image, label = self.samples[idx]
#         print(image)
#         print(label)
#         print(image.shape)
#         print(label.shape)
#
#         # Convert the numpy array and label to PyTorch tensors.
#         image = torch.from_numpy(np.array(image)).float()
#         label = torch.tensor(label, dtype=torch.long)
#
#         # Optionally, here you could transform the image (e.g., flattening, resizing) if necessary.
#         # Example for flattening as you mentioned: image = image.view(1, -1)
#
#         return image, label

# class PathfinderDataset(Dataset):
#     def __init__(self, dataset):
#         """
#         Initializes the dataset object.
#         :param dataset: A dataset object that yields tuples of (image, label),
#                         where `image` is a NumPy array or similar.
#         """
#         self.dataset = list(dataset)  # Convert the dataset to a list to easily access items
#
#     def __len__(self):
#         """
#         Returns the total number of items in the dataset.
#         """
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         """
#         Retrieves an item from the dataset at the specified index `idx`.
#         """
#         # Unpack the image and label from the dataset
#         input_batch, label_batch = self.dataset[idx]
#
#         # Check input_batch shape to adjust for batch or no batch mode
#         if len(input_batch.shape) == 4:  # If there's an existing batch dimension
#             input_batch = input_batch[0]  # Take the first item in the batch for simplicity
#
#         # Flatten the image tensor and handle multiple channels
#         num_input_features = input_batch.shape[2]  # Number of channels
#         length = input_batch.shape[0] * input_batch.shape[1]  # Total number of pixels
#
#         # Reshape the input to (C, L)
#         torch_input = torch.tensor(input_batch, dtype=torch.float32).permute(2, 0, 1).reshape(num_input_features,
#                                                                                               length)
#
#         # Convert label to torch tensor and ensure it has the correct shape
#         torch_label = torch.tensor(label_batch, dtype=torch.long)
#
#         return torch_input, torch_label