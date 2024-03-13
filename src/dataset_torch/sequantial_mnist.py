from torch.utils.data import Dataset


class SequentialMNIST(Dataset):
    def __init__(self, mnist_dataset):
        """
        mnist_dataset: An instance of torchvision.datasets.MNIST
        """
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        # Retrieve the image and label from the original MNIST dataset
        img, label = self.mnist_dataset[idx]

        # Flatten the image. img is a torch.Tensor of shape (1, 28, 28) for MNIST
        # We want to reshape it to (1, 784) since 28*28=784
        img_flattened = img.view(1, -1)  # This reshapes it to (1, 784)
        return img_flattened, label
