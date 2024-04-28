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


class LinearRegression(nn.Module):
    def __init__(self, d_state, d_output, to_vec):
        super().__init__()
        self.d_state = d_state
        self.d_output = d_output

        self.to_vec = to_vec

    def _one_hot_encoding(self, labels):
        y = torch.nn.functional.one_hot(input=labels, num_classes=self.d_output)
        y = y.to(dtype=torch.float32)
        return y

    # TODO: resolve bug for StackedEchoState
    def forward(self, X, y):
        if self.to_vec:
            y = self._one_hot_encoding(y)

        W_out_t = torch.linalg.pinv(X).mm(y)  # (P, K)

        return W_out_t


class RidgeRegression(LinearRegression):
    def __init__(self, d_state, d_output, to_vec, lambda_):
        if lambda_ <= 0:
            raise ValueError("Regularization lambda must be positive for Ridge Regression.")

        super().__init__(d_state, d_output, to_vec)

        self.lambda_ = lambda_

        self.register_buffer('eye_matrix', torch.eye(n=self.d_state, dtype=torch.float32))

    # TODO: resolve bug for StackedEchoState
    def forward(self, X, y):
        if self.to_vec:
            y = self._one_hot_encoding(y)

        W_out_t = torch.linalg.inv(X.t().mm(X) + self.lambda_ * self.eye_matrix).mm(X.t()).mm(y)  # (P, K)

        return W_out_t