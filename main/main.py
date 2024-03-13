import torch
import torch.optim as optim

#from src.models.s4.s4 import S4Block
from src.models.s4d.s4d import S4D
from src.models.ssrm.s5r import S5R
from src.models.nn.stacked import NaiveStacked
from src.task.image.seq2val import Seq2Val
from src.utils.test_torch import test_device
from src.ml.training import max_epochs
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.dataset_torch.sequantial_mnist import SequentialMNIST
from src.ml.evaluation import evaluate_classifier


class TestSequentialMNIST:
    def __init__(self, model_name, d_state, n_layers, device_id):
        self.model = TestSequentialMNIST.__construct_model(model_name, d_state, n_layers, device_id)

    @staticmethod
    def __construct_model(model_name, d_state, n_layers, device_id):
        # num_features_input = train_dataloader[0][0].size(1)
        num_features_input = 1

        # if model_name =='S4':
        #     ssm_block = S4Block(d_model=num_features_input, d_state=d_state)
        if model_name == 'S4D':
            ssm_block = S4D(d_input=num_features_input, d_state=d_state)
        elif model_name == 'S5R':
            ssm_block = S5R(d_input=num_features_input, d_state=d_state, high_stability=0.9, low_stability=1,
                            dynamics='discrete')
        else:
            ValueError('model undefined. Possible choice: [S4, S4D, S5R]')

        ssm_stacked = NaiveStacked(block=ssm_block, n_layers=n_layers)
        ssm_classifier = Seq2Val(ssm_stacked)

        ssm_classifier.to(torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"))
        test_device(ssm_classifier)

        return ssm_classifier


    def __fit_model(self, num_epochs, lr, train_dataloader):
        criterion = torch.nn.CrossEntropyLoss()  # Classification task: sigmoid layer + BCE loss (more stable)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        max_epochs(self.model, optimizer, criterion, train_dataloader, num_epochs=num_epochs)


    def __evaluate_model(self, test_dataloader):
        evaluate_classifier(self.model, test_dataloader)


    @staticmethod
    def main(model_name, d_state, n_layers, num_epochs, lr, device_id):
        # Input and output shape (B, H, L)
        #train_dataloader = load_temp_data('train_dataloader')

        test_smnist = TestSequentialMNIST(model_name, d_state, n_layers, device_id)

        # Define a transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        # Download and load the training data
        trainset = datasets.MNIST('../saved_data/', download=True, train=True, transform=transform)
        trainset = SequentialMNIST(trainset)
        train_dataloader = DataLoader(trainset, batch_size=128, shuffle=True)

        test_smnist.__fit_model(num_epochs, lr, train_dataloader)

        testset = datasets.MNIST(root='../saved_data/', train=False, download=True, transform=transform)
        testset = SequentialMNIST(testset)
        test_dataloader = DataLoader(testset, batch_size=128, shuffle=False)

        test_smnist.__evaluate_model(test_dataloader)



if __name__ == "__main__":
    TestSequentialMNIST.main(model_name='S4D', d_state=1, n_layers=1, num_epochs=1, lr=0.01, device_id=0)

