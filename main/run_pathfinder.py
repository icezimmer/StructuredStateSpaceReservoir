import torch
import torch.optim as optim

#from src.models.s4.s4 import S4Block
from src.models.rnn.vanilla_rnn import RNNBlock
from src.models.s4d.s4d import S4D
from src.models.ssrm.s5r import S5R
from src.models.rnn.vanilla_rnn import RNNClassifier
from src.models.nn.stacked import NaiveStacked
from src.task.seq2val import Seq2Val
from src.utils.test_torch import test_device
from src.ml.training import TrainModel
from src.utils.temp_data import load_temp_data
from src.ml.evaluation import EvaluateBinaryClassifier


class TestPathFinder:
    def __init__(self, block, n_layers, *args, **kwargs):
        self.__CRITERION = torch.nn.BCEWithLogitsLoss()  # Classification task: sigmoid layer + BCE loss (more stable)
        self.model = self.__construct_model(block, n_layers, *args, **kwargs)

    def __construct_model(self, block_factory, n_layers, *args, **kwargs):
        stacked = NaiveStacked(block_factory=block_factory, n_layers=n_layers, *args, **kwargs)
        classifier = Seq2Val(stacked)
        #classifier = RNNClassifier(input_size=num_features_input, hidden_size=d_state, num_layers=n_layers, output_size=1)
        return classifier

    def fit_model(self, num_epochs, lr, train_dataloader, device_name):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        trainer = TrainModel(self.model, optimizer, self.__CRITERION, train_dataloader, device_name)
        test_device(trainer.model)
        trainer.max_epochs(num_epochs=num_epochs)

    def evaluate_model(self, test_dataloader, device_name):
        eval_bc = EvaluateBinaryClassifier(self.model, test_dataloader, device_name)
        test_device(eval_bc.model)
        eval_bc.evaluate()


if __name__ == "__main__":
    NUM_FEATURES_INPUT = 3
    train_dataloader = load_temp_data('pathfinder_train_dataloader')
    test_dataloader = load_temp_data('pathfinder_test_dataloader')
    pathfinder = TestPathFinder(block_factory=S4D, n_layers=4, d_input=NUM_FEATURES_INPUT, d_state=16)
    pathfinder.fit_model(num_epochs=10, lr=0.001, train_dataloader=train_dataloader, device_name='cuda:1')
    pathfinder.evaluate_model(train_dataloader, device_name='cuda:1')
