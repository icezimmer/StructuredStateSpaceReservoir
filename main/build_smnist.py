from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.dataset_torch.sequantial_mnist import SequentialMNIST
from src.utils.temp_data import save_temp_data

transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

# Download and load the training data
trainset = datasets.MNIST('../saved_data/', download=True, train=True, transform=transform)
trainset = SequentialMNIST(trainset)
train_dataloader = DataLoader(trainset, batch_size=128, shuffle=True)

testset = datasets.MNIST(root='../saved_data/', train=False, download=True, transform=transform)
testset = SequentialMNIST(testset)
test_dataloader = DataLoader(testset, batch_size=128, shuffle=False)

save_temp_data(train_dataloader, 'train_dataloader')
save_temp_data(test_dataloader, 'test_dataloader')