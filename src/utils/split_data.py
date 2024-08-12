from torch.utils.data import random_split
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def random_split_dataset(dataset, val_split=0.2):
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def stratified_split_dataset(dataset, val_split=0.2):
    # Automatically extract labels by assuming dataset[i] returns (data, label)
    labels = [label for _, label in dataset]
    
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=val_split,
        stratify=labels
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset
