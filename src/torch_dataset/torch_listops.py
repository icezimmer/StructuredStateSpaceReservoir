from torch.utils.data import Dataset
import torch


class ListOpsDataset(Dataset):
    def __init__(self, tf_dataset, padding_idx):
        self.padding_idx = padding_idx
        self.data = []
        # Preprocess the entire dataset at initialization
        for sample in tf_dataset:
            # Convert the TF tensors to PyTorch tensors with long dtype for indices
            inputs = torch.tensor(sample['inputs'].numpy(), dtype=torch.long)
            targets = torch.tensor(sample['targets'].numpy(), dtype=torch.long)
            lengths = self._compute_lengths(inputs)
            self.data.append((inputs, targets, lengths))

    def _compute_lengths(self, inputs):
        """
        Compute the length of each padded time series in a batch.

        :param inputs: Tensor of shape (L,), where L is the sequence length.
        :return: Tensor of shape (), containing the lengths of the sequence.
        """
        # Create a mask where non-padding elements are True
        non_padding_mask = inputs != self.padding_idx  # Shape: (L)

        # Sum along the time dimension to get the length of each sequence
        lengths = non_padding_mask.sum(dim=-1)  # Shape: (,)

        return lengths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Simply return the preprocessed sample
        return self.data[idx]
