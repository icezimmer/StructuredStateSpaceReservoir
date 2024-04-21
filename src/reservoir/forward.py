import torch
from src.utils.check_device import check_data_device


class Reservoir2NN:
    def __init__(self, model, dataloader, to_numpy):
        self.device = check_data_device(dataloader)
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.to_vec = next(iter(dataloader))[1].dim() == 1
        self.to_numpy = to_numpy

    def _gather(self):
        with torch.no_grad():
            output_list = []
            label_list = []
            for input_batch, label_batch in self.dataloader:
                output_batch = self.model(input_batch)  # (B, P, L)
                output_batch = output_batch.to('cpu')
                label_batch = label_batch.to('cpu')
                output_list.append(output_batch)
                label_list.append(label_batch)

            output = torch.cat(output_list, dim=0)  # (N, P, L)
            label = torch.cat(label_list, dim=0)  # (N, ?)

        return output, label

    # TODO: repair the bug in the code
    def to_fit(self):
        with torch.no_grad():
            output, label = self._gather()  # (N, P, L), (N, ?)

            if self.to_vec:
                label = torch.repeat_interleave(label, output.shape[-1])  # (N * L)

            output = output.permute(0, 2, 1)  # (N, L, P)
            output = output.reshape(-1, output.shape[-1])  # (N * L, P), N * L is the number of samples

            # Convert to numpy arrays for scikit-learn
            if self.to_numpy:
                output = output.numpy()
                label = label.numpy()

            print('Output shapes for scikit-learn:')
            print('Output:', output.shape)
            print('Label:', label.shape)

        return output, label

    # TODO: repair the bug in the code
    def to_evaluate_classifier(self):
        with torch.no_grad():
            output, label = self._gather()  # (N, P, L), (N, ?)

            output = output[:, :, -1]  # (N, P)

            # Convert to numpy arrays for scikit-learn
            if self.to_numpy:
                label = label.numpy()
                output = output.numpy()

            print('Output shapes for scikit-learn:')
            print('Output:', output.shape)
            print('Label:', label.shape)

        return output, label
