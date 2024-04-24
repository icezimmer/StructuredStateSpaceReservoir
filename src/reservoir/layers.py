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
        u = torch.einsum('ph, bh -> bp', self.W_in, u)

        return u

    def forward(self, u):
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
        u = torch.einsum('ph, bh -> bp', self.W_in, u)

        return u

    def forward(self, u):
        u = torch.einsum('ph, bhl -> bpl', self.W_in, u)

        return u


class RidgeRegression(nn.Module):
    def __init__(self, d_state, d_output, lambda_, to_vec):
        super().__init__()
        self.d_state = d_state  # assuming that states ha bias
        self.d_output = d_output

        self.lambda_ = lambda_

        self.to_vec = to_vec

        reservoir = Reservoir(d_in=self.d_output, d_out=self.d_state)
        W_out_t = reservoir.uniform_disk_matrix(radius=1.0, field='real')  # (P, K)
        # Initialize weights; using a simple random initialization here
        self.W_out_t = nn.Parameter(W_out_t, requires_grad=False)

    def _one_hot_encoding(self, labels):
        return torch.nn.functional.one_hot(labels, num_classes=self.d_output).float()

    def fit(self, X, y):
        with torch.no_grad():
            if self.to_vec:
                y = self._one_hot_encoding(y)
                y = y.to(dtype=torch.float32)

            self.W_out_t.data = torch.linalg.pinv(
                X.t().mm(X) + self.lambda_ * torch.eye(self.d_state)
            ).mm(X.t()).mm(y)  # (P, K)

    def forward(self, X):
        predictions = torch.einsum('np,pk -> nk', X, self.W_out_t)
        return torch.argmax(predictions, dim=1) if self.to_vec else predictions
