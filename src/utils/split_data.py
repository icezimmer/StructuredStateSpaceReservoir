from torch.utils.data import random_split
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def random_split_dataset(dataset, val_split):
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def stratified_split_dataset(dataset, val_split):
    # Automatically extract labels by assuming dataset[i] returns (data, label) or (data, label, length)
    if len(dataset[0]) == 2:
        labels = [label for _, label in dataset]
    elif len(dataset[0]) == 3:
        labels = [label for _, label, _ in dataset]
    else:
        raise ValueError("Dataset must return either (inputs, labels) or (inputs, labels, lengths)")

    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=val_split,
        stratify=labels
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset
