import torch
import torch.optim as optim
from src.models.nn.stacked import NaiveStacked
from src.task.seq2vec import Seq2Vec
from src.utils.check_device import check_model_device
from src.ml.training import TrainModel
from src.ml.evaluation import EvaluateClassifier


class Classifier:
    def __init__(self, block_factory, device, num_classes, n_layers, *args, **kwargs):
        self.__NUM_CLASSES = num_classes
        self.__CRITERION = torch.nn.CrossEntropyLoss()  # Classification task: softmax layer + CE loss (more stable)
        model = self.__construct_model(block_factory, n_layers, *args, **kwargs)
        self.model = model.to(device)

    def __construct_model(self, block_factory, n_layers, *args, **kwargs):
        stacked = NaiveStacked(block_factory=block_factory, n_layers=n_layers, *args, **kwargs)
        classifier = Seq2Vec(stacked, d_vec=self.__NUM_CLASSES)
        return classifier

    def fit_model(self, lr, develop_dataloader, checkpoint_path, num_epochs=float('inf'), patience=None, *args, **kwargs):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        trainer = TrainModel(self.model, optimizer, self.__CRITERION, develop_dataloader)
        print('Model device in Training: ', check_model_device(trainer.model))
        if patience:
            trainer.early_stopping(num_epochs=num_epochs, patience=patience, checkpoint_path=checkpoint_path,
                                   *args, **kwargs)
        else:
            trainer.max_epochs(num_epochs=num_epochs, checkpoint_path=checkpoint_path)

    def evaluate_model(self, dataloader):
        eval_bc = EvaluateClassifier(self.model, self.__NUM_CLASSES, dataloader)
        print('Model device in Evaluation: ', check_model_device(eval_bc.model))
        eval_bc.evaluate()
