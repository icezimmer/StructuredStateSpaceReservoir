import torch
import torch.optim as optim

#from src.models.s4.s4 import S4Block
from src.models.s4d.s4d import S4D
from src.models.ssrm.s5r import S5R
from src.models.nn.stacked import NaiveStacked
from src.task.image.seq2val import Seq2Val
from src.utils.test_torch import test_device
from src.ml.training import TrainModel
from src.utils.temp_data import load_temp_data

class TestSequentialMNIST:
    def __init__(self, model_name, d_state, n_layers, device_train, device_eval):
        self.model = TestSequentialMNIST.__construct_model(model_name, d_state, n_layers)
        self.device_train = device_train
        self.device_eval = device_eval

    @staticmethod
    def __construct_model(model_name, d_state, n_layers):
        # num_features_input = train_dataloader[0][0].size(1)
        NUM_FEATURES_INPUT = 1

        # if model_name =='S4':
        #     ssm_block = S4Block(d_model=num_features_input, d_state=d_state)
        if model_name == 'S4D':
            ssm_block = S4D(d_input=NUM_FEATURES_INPUT, d_state=d_state)
        elif model_name == 'S5R':
            ssm_block = S5R(d_input=NUM_FEATURES_INPUT, d_state=d_state, high_stability=0.9, low_stability=1,
                            dynamics='discrete')
        else:
            ValueError('model undefined. Possible choice: [S4, S4D, S5R]')

        ssm_stacked = NaiveStacked(block=ssm_block, n_layers=n_layers)
        ssm_classifier = Seq2Val(ssm_stacked)

        return ssm_classifier


    def __fit_model(self, num_epochs, lr, train_dataloader):
        criterion = torch.nn.MSELoss()  # Classification task: sigmoid layer + BCE loss (more stable)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        trainer = TrainModel(self.model, optimizer, criterion, train_dataloader, self.device_train)
        test_device(trainer.model)
        trainer.max_epochs(num_epochs=num_epochs)


    # def __evaluate_model(self, test_dataloader):
    #     evaluate_classifier(self.model, test_dataloader, self.device_eval)


    @staticmethod
    def main(model_name, d_state, n_layers, num_epochs, lr, device_train, device_eval):
        # Input and output shape (B, H, L)
        #train_dataloader = load_temp_data('train_dataloader')

        smnist = TestSequentialMNIST(model_name, d_state, n_layers, device_train, device_eval)

        train_dataloader = load_temp_data('train_dataloader')

        smnist.__fit_model(num_epochs, lr, train_dataloader)

        test_dataloader = load_temp_data('test_dataloader')

        # smnist.__evaluate_model(test_dataloader)



if __name__ == "__main__":
    TestSequentialMNIST.main(model_name='S4D', d_state=10, n_layers=10, num_epochs=10, lr=0.001,
                             device_train='cuda:0', device_eval='cpu')

