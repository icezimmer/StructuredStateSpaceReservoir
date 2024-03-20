import torch
import torch.optim as optim
from src.models.nn.stacked import NaiveStacked
from src.task.seq2vec import Seq2Vec
from src.utils.test_torch import test_device
from src.ml.training import TrainModel
from src.ml.evaluation import EvaluateClassifier


class TestClassifier:
    def __init__(self, block_factory, device_name, num_classes, n_layers, *args, **kwargs):
        self.__NUM_CLASSES = num_classes
        self.__CRITERION = torch.nn.CrossEntropyLoss()  # Classification task: sigmoid layer + BCE loss (more stable)
        self.device = torch.device(device_name)
        model = self.__construct_model(block_factory, n_layers, *args, **kwargs)
        self.model = model.to(self.device)

    def __construct_model(self, block_factory, n_layers, *args, **kwargs):
        stacked = NaiveStacked(block_factory=block_factory, n_layers=n_layers, *args, **kwargs)
        classifier = Seq2Vec(stacked, d_vec=self.__NUM_CLASSES)
        return classifier

    def fit_model(self, num_epochs, lr, train_dataloader):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        trainer = TrainModel(self.model, optimizer, self.__CRITERION, train_dataloader)
        test_device(trainer.model)
        trainer.max_epochs(num_epochs=num_epochs)

    def evaluate_model(self, test_dataloader):
        eval_bc = EvaluateClassifier(self.model, self.__NUM_CLASSES, test_dataloader)
        test_device(eval_bc.model)
        eval_bc.evaluate()
