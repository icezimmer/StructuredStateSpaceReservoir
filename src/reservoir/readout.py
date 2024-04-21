import torch
from src.utils.check_device import check_data_device


class TrainReservoir:
    def __init__(self, model, develop_dataloader, to_numpy):
        self.device = check_data_device(develop_dataloader)
        self.model = model.to(self.device)
        self.develop_dataloader = develop_dataloader
        self.to_vec = next(iter(develop_dataloader))[1].dim() == 1
        self.to_numpy = to_numpy

    # TODO: repair the bug in the code
    def __call__(self):
        with torch.no_grad():
            output_list = []
            label_list = []
            for input_batch, label_batch in self.develop_dataloader:
                output_batch = self.model(input_batch)  # (B, P, L)
                output_batch = output_batch.to('cpu')
                label_batch = label_batch.to('cpu')
                output_list.append(output_batch)
                label_list.append(label_batch)

            output = torch.cat(output_list, dim=0)  # (N, P, L)
            label = torch.cat(label_list, dim=0)  # (N, ?)

            if self.to_vec:
                label = label.unsqueeze(1).expand(-1, output.shape[-1]).reshape(-1)  # (N * L)
            output = output.reshape(output.shape[0] * output.shape[-1], -1)  # (N * L, P)

            # Convert to numpy arrays for scikit-learn
            if self.to_numpy:
                output = output.numpy()
                label = label.numpy()

            print('Output shapes for scikit-learn:')
            print('Label:', label.shape)
            print('Output:', output.shape)

        return output, label


class EvaluateReservoirClassifier:
    def __init__(self, model, dataloader, to_numpy):
        self.device = check_data_device(dataloader)
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.to_numpy = to_numpy

    # TODO: repair the bug in the code
    def __call__(self):
        with torch.no_grad():
            output_list = []
            label_list = []
            for input_batch, label_batch in self.dataloader:
                for t in range(input_batch.shape[-1]):
                    u = input_batch[:, :, t]  # (B, P)
                    if t == 0:
                        y, x = self.model.step(u)
                    y, x = self.model.step(u, x)  # (B, P), (B, P)
                output_list.append(y.to('cpu'))
                label_batch = label_batch.to('cpu')
                label_list.append(label_batch)

            output = torch.cat(output_list, dim=0)  # (N, P)
            label = torch.cat(label_list, dim=0)  # (N)

            # Convert to numpy arrays for scikit-learn
            if self.to_numpy:
                label = label.numpy()
                output = output.numpy()

            print('Output shapes for scikit-learn:')
            print('Output:', output.shape)
            print('Label:', label.shape)

        return output, label
