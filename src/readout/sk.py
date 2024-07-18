import json
import torch
from sklearn import preprocessing
from sklearn.linear_model import RidgeClassifier
from matplotlib import pyplot as plt
from tqdm import tqdm

from sklearn.metrics import accuracy_score, confusion_matrix
from src.utils.check_device import check_model_device
import os


class Ridge:
    def __init__(self, reservoir_model, develop_dataloader, lambda_):
        self.reservoir_model = reservoir_model
        self.device = check_model_device(model=self.reservoir_model)
        self.develop_dataloader = develop_dataloader

        self.hidden_state = None
        self.label = None

        self.accuracy_value = None
        self.precision_value = None
        self.recall_value = None
        self.f1_value = None
        self.roc_auc_value = None
        self.confusion_matrix_value = None

        self.scaler = preprocessing.StandardScaler()
        self.readout_cls = RidgeClassifier(alpha=lambda_, solver='svd')

    # TODO: try to move to cpu the hidden_state of each layer before to stacked them in deep module
    def _gather(self, dataloader):
        self.reservoir_model.eval()
        with torch.no_grad():
            hidden_state_list = []
            label_list = []
            for input_batch, label_batch in tqdm(dataloader):
                input_batch = input_batch.to(self.device)  # (B, H, L)

                hidden_state_batch = self.reservoir_model(input_batch)  # (B, P, L-w)

                hidden_state_batch = hidden_state_batch.to('cpu')  # (B, P, L-w)
                label_batch = label_batch.to('cpu')  # (B, *)

                hidden_state_list.append(hidden_state_batch)
                label_list.append(label_batch)

            self.hidden_state = torch.cat(tensors=hidden_state_list, dim=0)  # (N, P, L-w) timeseries
            self.label = torch.cat(tensors=label_list, dim=0)  # (B, *) -> (N, *)

    def fit_(self):
        with torch.no_grad():
            self._gather(self.develop_dataloader)  # (N, P, L-w), (N,)

            label = torch.repeat_interleave(input=self.label, repeats=self.hidden_state.shape[-1], dim=0).numpy()  # (N*(L-w),)
            hidden_state = self.hidden_state.permute(0, 2, 1)  # (N, L-w, P)
            hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1]).numpy()  # (N*(L-w), P) = (num_samples, features)

            self.scaler = self.scaler.fit(hidden_state)
            hidden_state = self.scaler.transform(hidden_state)

            self.readout_cls = self.readout_cls.fit(hidden_state, label)

    # TODO: compute the rest of metrics
    def evaluate_(self, dataloader=None, saving_path=None):
        with torch.no_grad():
            prediction = self.predict_(dataloader)  # (N, K)

            prediction = prediction
            label = self.label.numpy()

            self.accuracy_value = accuracy_score(y_true=label, y_pred=prediction)
            self.confusion_matrix_value = confusion_matrix(y_true=label, y_pred=prediction)

            print(f"Accuracy: {self.accuracy_value}")
            print("Confusion Matrix:\n", self.confusion_matrix_value)

            if saving_path is not None:
                self._plot(saving_path)

    def predict_(self, dataloader=None):
        with torch.no_grad():
            if dataloader is not None:
                self._gather(dataloader)  # (N, P, L-w), (N,)
            else:
                if self.hidden_state is None or self.label is None:
                    raise ValueError("Hidden state and label are not set, please provide a dataloader.")

            hidden_state = self.hidden_state[:, :, -1].numpy()  # (N, P)
            hidden_state = self.scaler.transform(hidden_state)

            prediction = self.readout_cls.predict(X=hidden_state)  # (P+1, K)

        return prediction

    def _plot(self, saving_path):
        metrics_path = os.path.join(saving_path, 'metrics.json')
        confusion_matrix_path = os.path.join(saving_path, 'confusion_matrix.png')

        metrics = {
            "Accuracy": self.accuracy_value,
            "Precision": self.precision_value,
            "Recall": self.recall_value,
            "F1-Score": self.f1_value,
            "ROC-AUC": self.roc_auc_value
        }

        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            # Convert args namespace to dictionary and save as JSON
            json.dump(metrics, f, indent=4)

        # Plotting the confusion matrix
        fig, ax = plt.subplots()
        cax = ax.matshow(self.confusion_matrix_value, cmap=plt.cm.Purples)
        plt.title('Confusion Matrix')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(confusion_matrix_path)
        plt.close()
