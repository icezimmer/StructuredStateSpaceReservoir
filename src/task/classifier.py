import torch
import torch.optim as optim
from src.models.deep.stacked import NaiveStacked
from src.ml.training import TrainModel
from src.ml.evaluation import EvaluateClassifier


class Classifier:
    def __init__(self, block_factory, n_layers, d_input, d_model, num_classes, *args, **kwargs):
        self.__NUM_CLASSES = num_classes
        self.__CRITERION = torch.nn.CrossEntropyLoss()  # Classification task: softmax layer + CE loss (more stable)
        self.model = self.__construct_model(block_factory, n_layers, d_input, d_model, *args, **kwargs)

    def __construct_model(self,  block_factory, n_layers, d_input, d_model, *args, **kwargs):
        classifier = NaiveStacked(block_factory=block_factory, n_layers=n_layers, d_input=d_input, d_model=d_model,
                                  d_output=self.__NUM_CLASSES, to_vec=True, *args, **kwargs)
        return classifier

    def fit_model(self, lr, develop_dataloader, checkpoint_path, num_epochs, patience, *args, **kwargs):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        trainer = TrainModel(self.model, optimizer, self.__CRITERION, develop_dataloader)
        # print('Model device in Training: ', check_model_device(trainer.model))
        if patience:
            trainer.early_stopping(num_epochs=num_epochs, patience=patience, checkpoint_path=checkpoint_path,
                                   *args, **kwargs)
        else:
            trainer.max_epochs(num_epochs=num_epochs, checkpoint_path=checkpoint_path)

    def evaluate_model(self, dataloader):
        eval_bc = EvaluateClassifier(self.model, self.__NUM_CLASSES, dataloader)
        # print('Model device in Evaluation: ', check_model_device(eval_bc.model))
        eval_bc.evaluate()
