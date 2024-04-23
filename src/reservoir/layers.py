import torch
from src.reservoir.matrices import Reservoir, StructuredReservoir
import torch.nn as nn


class LinearReservoir(nn.Module):
    def __init__(self, d_input, d_output,
                 radius=1.0,
                 field='real'):
        super().__init__()

        self.d_input = d_input
        self.d_output = d_output
        self.field = field

        reservoir = Reservoir(d_in=self.d_input, d_out=self.d_output)
        W_in = reservoir.uniform_disk_matrix(radius=radius, field=field)
        self.register_buffer('W_in', W_in)

    def step(self, u):
        with torch.no_grad():
            u = torch.einsum('ph, bh -> bp', self.W_in, u)

        return u

    def forward(self, u):
        with torch.no_grad():
            u = torch.einsum('ph, bhl -> bpl', self.W_in, u)

        return u


class LinearStructuredReservoir(nn.Module):
    def __init__(self, d_input, d_output,
                 radius=1.0,
                 field='real'):
        super().__init__()

        self.d_input = d_input
        self.d_output = d_output
        self.field = field

        structured_reservoir = StructuredReservoir(d_in=self.d_input, d_out=self.d_output)
        W_in = structured_reservoir.uniform_disk_matrix(radius=radius, field=field)
        self.register_buffer('W_in', W_in)

    def step(self, u):
        with torch.no_grad():
            u = torch.einsum('ph, bh -> bp', self.W_in, u)

        return u

    def forward(self, u):
        with torch.no_grad():
            u = torch.einsum('ph, bhl -> bpl', self.W_in, u)

        return u


class RidgeRegression(nn.Module):
    def __init__(self, d_input, d_output, lambda_, to_vec):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output

        self.lambda_ = lambda_

        self.to_vec = to_vec

        reservoir = Reservoir(d_in=self.d_input, d_out=self.d_output)
        W_out = reservoir.uniform_disk_matrix(radius=1.0, field='real')
        # Initialize weights; using a simple random initialization here
        self.W_out = nn.Parameter(W_out, requires_grad=False)

    def _one_hot_encoding(self, labels):
        return torch.nn.functional.one_hot(labels, num_classes=self.d_output).float()

    def fit(self, X, y):
        with torch.no_grad():
            Y = self._one_hot_encoding(y) if self.to_vec else y.float().unsqueeze(1)
            X, Y = X.float(), Y.float()
            self.W_out.data = torch.inverse(X.t().mm(X) + self.lambda_ * torch.eye(X.size(1))).mm(X.t()).mm(Y)

    def forward(self, X):
        predictions = torch.einsum('np,pk -> nk', X, self.W_out)  # Transpose W_out to match dimensions
        return torch.argmax(predictions, dim=1) if self.to_vec else predictions
