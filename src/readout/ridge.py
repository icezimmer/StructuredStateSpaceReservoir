import json
import torch
from src.reservoir.matrices import ReservoirMatrix
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from sklearn.metrics import accuracy_score, confusion_matrix
from src.utils.check_device import check_model_device
import os


class LinearRegression(nn.Module):
    def __init__(self, d_input, d_output, to_vec):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output

        self.to_vec = to_vec

        structured_reservoir = ReservoirMatrix(d_in=d_output, d_out=d_input)  # transpose of matrix (left multipl.)
        self.register_buffer('W_out_t',
                             structured_reservoir.uniform_ring(max_radius=1.0, min_radius=0.0, field='real'))  # (P+1,K)

    def _one_hot_encoding(self, labels):
        y = torch.nn.functional.one_hot(input=labels, num_classes=self.d_output)
        y = y.to(dtype=torch.float32)
        return y

    # TODO: resolve bug for StackedEchoState
    def forward(self, X, y=None):
        if y is not None:
            if self.to_vec:
                y = self._one_hot_encoding(y)

            self.W_out_t = torch.linalg.pinv(X).mm(y)  # (P, K)
            o = None
        else:
            o = torch.einsum('np,pk -> nk', X, self.W_out_t)  # prediction
            o = torch.argmax(o, dim=1) if self.to_vec else o

        return o


class RidgeRegression(LinearRegression):
    def __init__(self, d_input, d_output, to_vec, lambda_):
        if lambda_ <= 0:
            raise ValueError("Regularization lambda must be positive for Ridge Regression.")

        super().__init__(d_input, d_output, to_vec)

        self.lambda_ = lambda_

        self.register_buffer('eye_matrix', torch.eye(n=self.d_input, dtype=torch.float32))

    def forward(self, X, y=None):
        if y is not None:
            if self.to_vec:
                y = self._one_hot_encoding(y)

            self.W_out_t = torch.linalg.inv(X.t().mm(X) + self.lambda_ * self.eye_matrix).mm(X.t()).mm(y)  # (P, K)
            o = None
        else:
            o = torch.einsum('np,pk -> nk', X, self.W_out_t)  # prediction
            o = torch.argmax(o, dim=1) if self.to_vec else o

        return o


class Ridge:
    def __init__(self, reservoir_model, develop_dataloader, d_output, to_vec, lambda_, bias=True):
        self.reservoir_model = reservoir_model
        self.device = check_model_device(model=self.reservoir_model)
        self.develop_dataloader = develop_dataloader
        self.bias = bias
        self.to_vec = to_vec

        self.hidden_state = None
        self.label = None

        self.accuracy_value = None
        self.precision_value = None
        self.recall_value = None
        self.f1_value = None
        self.roc_auc_value = None
        self.confusion_matrix_value = None

        d_input = self.reservoir_model.d_output
        if self.bias:
            d_input = d_input + 1

        if lambda_ == 0.0:
            self.readout_cls = LinearRegression(d_input=d_input, d_output=d_output, to_vec=self.to_vec)
        else:
            self.readout_cls = RidgeRegression(d_input=d_input, d_output=d_output, to_vec=self.to_vec, lambda_=lambda_)

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

            label = torch.repeat_interleave(input=self.label, repeats=self.hidden_state.shape[-1], dim=0)  # (N*(L-w),)
            hidden_state = self.hidden_state.permute(0, 2, 1)  # (N, L-w, P)
            hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1])  # (N*(L-w), P) = (num_samples, features)
            if self.bias:
                hidden_state = torch.cat(tensors=(hidden_state, torch.ones(size=(hidden_state.size(0), 1),
                                                                           dtype=torch.float32)),
                                         dim=1)  # (N*(L-w), P+1)

            _ = self.readout_cls(X=hidden_state, y=label)  # (P+1, K)

    # TODO: compute the rest of metrics
    def evaluate_(self, dataloader=None, saving_path=None):
        with torch.no_grad():
            prediction = self.predict_(dataloader)  # (N, K)

            prediction = prediction.numpy()
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

            hidden_state = self.hidden_state[:, :, -1]  # (N, P)
            if self.bias:
                hidden_state = torch.cat(tensors=(hidden_state, torch.ones(size=(hidden_state.size(0), 1),
                                                                           dtype=torch.float32)), dim=1)  # (N, P+1)

            prediction = self.readout_cls(X=hidden_state)  # (P+1, K)

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
