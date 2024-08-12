import torch
from tqdm import tqdm

from src.utils.check_device import check_model_device
from torch.utils.data import Dataset


class Reservoir2ReadOut(Dataset):
    def __init__(self, reservoir_model, dataloader):
        self.reservoir_model = reservoir_model
        self.device = check_model_device(model=self.reservoir_model)
        self.dataloader = dataloader
        output_list = []
        label_list = []
        self.outputs = None
        self.labels = None

        with torch.no_grad():

            for input_batch, label_batch in tqdm(self.dataloader):
                input_batch = input_batch.to(device=self.device)
                label_batch = label_batch.to(device=self.device)  # (B, *)
                output_batch = self.reservoir_model(input_batch)  # (B, P, L-w)
                output_list.append(output_batch.to(device='cpu'))
                label_list.append(label_batch.to(device='cpu'))

            self.outputs = torch.cat(output_list, dim=0)
            self.labels = torch.cat(label_list, dim=0)

    def __len__(self):
        return self.outputs.shape[0]  # Total number of sequences

    def __getitem__(self, idx):
        # Return individual sequences and labels
        return self.outputs[idx], self.labels[idx]
    
    def to_fit_offline_readout(self):
        X, y = self[:]  # (N, P, L-w), (N,)

        _, P, L = X.shape
        if L > 1:
            y = torch.repeat_interleave(input=y, repeats=L, dim=0)  # (N*(L-w),)
            X = X.permute(0, 2, 1)  # (N, L-w, P)
            X = X.reshape(-1, P)  # (N*(L-w), P) = (num_samples, features)
        else:
            X = X.squeeze(-1)  # (N, P) = (num_samples, features)

        return X, y
    
    def to_evaluate_offline_classifier(self):
        X, y = self[:]  # (N, P, L-w), (N,)
        X = X[..., -1]  # (N, P) = (num_samples, features)

        return X, y
