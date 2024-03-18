from torch.utils.data import Dataset


class SequentialImage2Classify(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, target = self.dataset[idx]
        print(image.view(image.shape[0], -1).shape)
        print(target.shape)

        # Return flattened image in shape (channels, time steps = rows * columns)
        return image.view(image.shape[0], -1), target
