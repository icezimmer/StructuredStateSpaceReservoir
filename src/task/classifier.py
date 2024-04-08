import torch
import torch.optim as optim
from src.deep.residual import ResidualNetwork
from src.ml.training import TrainModel
from src.ml.evaluation import EvaluateClassifier


class Classifier:
    def __init__(self, block_cls, n_layers, d_input, d_model, num_classes,
                 layer_dropout, pre_norm,
                 **block_args):
        self.__TO_VEC = True
        self.__NUM_CLASSES = num_classes
        self.__CRITERION = torch.nn.CrossEntropyLoss()  # Classification task: softmax layer + CE loss (more stable)
        self.model = self.__construct_model(block_cls, n_layers, d_input, d_model, layer_dropout, pre_norm,
                                            **block_args)

    def __construct_model(self,  block_cls, n_layers, d_input, d_model, layer_dropout, pre_norm, **block_args):
        classifier = ResidualNetwork(block_cls=block_cls,
                                     n_layers=n_layers,
                                     d_input=d_input, d_model=d_model, d_output=self.__NUM_CLASSES,
                                     layer_dropout=layer_dropout, pre_norm=pre_norm,
                                     to_vec=self.__TO_VEC,
                                     **block_args)
        return classifier

    def fit_model(self, lr, develop_dataloader, checkpoint_path, num_epochs, patience, **es_args):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        trainer = TrainModel(self.model, optimizer, self.__CRITERION, develop_dataloader)
        if patience:
            trainer.early_stopping(num_epochs=num_epochs, patience=patience, checkpoint_path=checkpoint_path,
                                   **es_args)
        else:
            trainer.max_epochs(num_epochs=num_epochs, checkpoint_path=checkpoint_path)

    def evaluate_model(self, dataloader):
        eval_bc = EvaluateClassifier(self.model, self.__NUM_CLASSES, dataloader)
        eval_bc.evaluate()
