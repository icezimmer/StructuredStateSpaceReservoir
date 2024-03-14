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
from src.ml.evaluation import EvaluateBinaryClassifier


class TestPathFinder:
    def __init__(self, model_name, d_state, n_layers):
        self.model = TestPathFinder.__construct_model(model_name, d_state, n_layers)

    @staticmethod
    def __construct_model(model_name, d_state, n_layers):
        num_features_input = 3

        # if model_name =='S4':
        #     ssm_block = S4Block(d_model=num_features_input, d_state=d_state)
        if model_name == 'S4D':
            ssm_block = S4D(d_input=num_features_input, d_state=d_state)
        elif model_name == 'S5R':
            ssm_block = S5R(d_input=num_features_input, d_state=d_state, high_stability=0.9, low_stability=1,
                            dynamics='discrete')
        else:
            raise ValueError('model undefined. Possible choice: [S4, S4D, S5R]')

        ssm_stacked = NaiveStacked(block=ssm_block, n_layers=n_layers)
        ssm_classifier = Seq2Val(ssm_stacked)

        return ssm_classifier

    def fit_model(self, num_epochs, lr, train_dataloader, device_name):
        criterion = torch.nn.BCEWithLogitsLoss()  # Classification task: sigmoid layer + BCE loss (more stable)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        trainer = TrainModel(self.model, optimizer, criterion, train_dataloader, device_name)
        test_device(trainer.model)
        trainer.max_epochs(num_epochs=num_epochs)

    def evaluate_model(self, test_dataloader, device_name):
        eval_bc = EvaluateBinaryClassifier(self.model, test_dataloader, device_name)
        test_device(eval_bc.model)
        eval_bc.evaluate()


if __name__ == "__main__":
    train_dataloader = load_temp_data('train_dataloader')
    test_dataloader = load_temp_data('test_dataloader')
    pathfinder = TestPathFinder(model_name='S4D', d_state=100, n_layers=10)
    pathfinder.fit_model(num_epochs=10, lr=0.001, train_dataloader=train_dataloader, device_name='cuda:1')
    pathfinder.evaluate_model(train_dataloader, device_name='cuda:1')
