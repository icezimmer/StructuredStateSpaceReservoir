import torch
from src.utils.check_device import check_model_device
from torch.utils.data import Dataset


class Reservoir2NN(Dataset):
    def __init__(self, reservoir_model, dataloader):
        self.reservoir_model = reservoir_model
        self.device = check_model_device(model=self.reservoir_model)
        self.dataloader = dataloader
        output_list = []
        label_list = []
        self.outputs = None
        self.labels = None

        with torch.no_grad():

            for input_batch, label_batch in self.dataloader:
                input_batch = input_batch.to(device=self.device)
                label_batch = label_batch.to(device=self.device)  # (B, ?)
                output_batch = self.reservoir_model(input_batch)  # (B, P, L)
                output_list.append(output_batch.to(device='cpu'))
                label_list.append(label_batch.to(device='cpu'))

            self.outputs = torch.cat(output_list, dim=0)
            self.labels = torch.cat(label_list, dim=0)

    def __len__(self):
        return self.outputs.shape[0]  # Total number of sequences

    def __getitem__(self, idx):
        # Return individual sequences and labels
        return self.outputs[idx], self.labels[idx]
