from torch.utils.data import random_split


def random_split_dataset(dataset, trainable_portion=0.8):
    train_size = int(trainable_portion * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
