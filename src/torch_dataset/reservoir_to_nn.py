import torch
from src.utils.check_device import check_model_device
from torch.utils.data import Dataset


class Reservoir2NN(Dataset):
    def __init__(self, reservoir_model, dataloader):
        self.reservoir_model = reservoir_model
        self.device = check_model_device(model=self.reservoir_model)
        self.dataloader = dataloader
        self.to_vec = next(iter(dataloader))[1].dim() == 1

        with torch.no_grad():
            output_list = []
            label_list = []

            for input_batch, label_batch in self.dataloader:
                input_batch = input_batch.to(device=self.device)
                label_batch = label_batch.to(device=self.device)  # (B, ?)
                output_batch = self.reservoir_model(input_batch)  # (B, P, L)
                output_list.append(output_batch.to(device='cpu'))
                label_list.append(label_batch.to(device='cpu'))

            output = torch.cat(output_list, dim=0)  # (N, P, L)
            label = torch.cat(label_list, dim=0)  # (N, ?)

            if self.to_vec:
                label = torch.repeat_interleave(label, output.shape[-1])  # (N * L)

            output = output.permute(0, 2, 1)  # (N, L, P)
            output = output.reshape(-1, output.shape[-1])  # (N * L, P), N * L is the number of samples

            self.output = output
            self.label = label

    def __len__(self):
        return self.output.shape[0]

    def __getitem__(self, idx):
        return self.output[idx], self.label[idx]
