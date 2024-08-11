from torch.utils.data import Dataset
import torch


class ListOpsDataset(Dataset):
    def __init__(self, tf_dataset):
        self.data = []
        # Preprocess the entire dataset at initialization
        for sample in tf_dataset:
            # Convert the TF tensors to PyTorch tensors with long dtype for indices
            inputs = torch.tensor(sample['inputs'].numpy(), dtype=torch.long)
            targets = torch.tensor(sample['targets'].numpy(), dtype=torch.long)
            self.data.append((inputs, targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Simply return the preprocessed sample
        return self.data[idx]
