import torch
from sklearn.linear_model import RidgeClassifier
from src.reservoir.layers import RidgeRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from src.utils.check_device import check_model_device


class ReadOutClassifier:
    def __init__(self, reservoir_model, develop_dataloader, d_state, d_output, lambda_, **ridge_args):
        self.reservoir_model = reservoir_model
        self.device = check_model_device(model=self.reservoir_model)
        self.develop_dataloader = develop_dataloader
        #self.readout = RidgeClassifier(**ridge_args)
        self.readout = RidgeRegression(d_input=d_state, d_output=d_output, lambda_=lambda_, to_vec=True)

    def _gather(self, dataloader):
        self.reservoir_model.eval()
        with torch.no_grad():
            output_list = []
            label_list = []
            for input_batch, label_batch in dataloader:
                input_batch = input_batch.to(self.device)  # (B, H, L)
                label_batch = label_batch.to(self.device)  # (B,)

                output_batch = self.reservoir_model(input_batch)  # (B, P, L)

                output_batch = output_batch.to('cpu')
                label_batch = label_batch.to('cpu')

                output_list.append(output_batch)
                label_list.append(label_batch)

            output = torch.cat(tensors=output_list, dim=0)  # (N, P, L) timeseries
            label = torch.cat(tensors=label_list, dim=0)  # (N,)

            return output, label

    # TODO: check why the accuracy is ~0.1 like a random guess
    def fit_(self):
        with torch.no_grad():
            output, label = self._gather(self.develop_dataloader)  # (N, P, L), (N,)

            label = torch.repeat_interleave(input=label, repeats=output.shape[-1], dim=0)  # (N * L)
            output = output.permute(0, 2, 1)  # (N, L, P)
            output = output.reshape(-1, output.shape[-1])  # (N * L, P), N * L is the number of samples

            # output = output.reshape(output.shape[0], -1)  # (N, L * P) this works why?????

            self.readout.fit(output, label)

            # output = output.numpy()
            # label = label.numpy()

            # self.readout.fit(X=output, y=label)

            # prediction = self.readout.predict(X=output)

            # accuracy = accuracy_score(y_true=label, y_pred=prediction)
            # conf_matrix = confusion_matrix(y_true=label, y_pred=prediction)

            # print("Accuracy:", accuracy)
            # print("Confusion Matrix:\n", conf_matrix)

    # TODO: check why the accuracy is ~0.1 like a random guess
    def evaluate_(self, dataloader):
        with torch.no_grad():
            output, label = self._gather(dataloader)  # (N, P, L), (N,)

            output = output[:, :, -1]  # (N, P)

            prediction = self.readout(X=output)

            prediction = prediction.numpy()
            label = label.numpy()

            accuracy = accuracy_score(y_true=label, y_pred=prediction)
            conf_matrix = confusion_matrix(y_true=label, y_pred=prediction)

            print("Accuracy:", accuracy)
            print("Confusion Matrix:\n", conf_matrix)

        return output, label

    # TODO: check why the accuracy is ~0.1 like a random guess
    def predict_(self, dataloader):
        with torch.no_grad():
            output, _ = self._gather(dataloader)  # (N, P, L), (N,)

            output = output[:, :, -1]  # (N, P)

        return self.readout(output)
